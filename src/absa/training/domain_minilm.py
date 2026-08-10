"""Domain-adapt and fine-tune a compact BERT encoder for restaurant ABSA."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)

from ..config import ABSA_OUTPUTS_DIR
from ..data.splits import split_official_data
from ..evaluation import compute_metrics
from ..labels import ID_TO_LABEL, LABEL_TO_ID
from ..tokenization.bert_dataset import AspectPairDataset
from .common import (
    BestCheckpoint,
    build_run_result,
    checkpoint_metadata,
    seed_everything,
    validate_training_parameters,
)
from .distilbert import preferred_device


MODEL_NAME = "google/bert_uncased_L-4_H-256_A-4"
MODEL_REVISION = "43ca9bbcdffc2f4220d09f6e4fc48c28bfffeede"


class _ReviewDataset(Dataset):
    def __init__(self, tokenizer, reviews: list[str], max_length: int) -> None:
        self.encodings = tokenizer(
            reviews,
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
        )["input_ids"]

    def __len__(self) -> int:
        return len(self.encodings)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return {"input_ids": self.encodings[index]}


def train_domain_minilm(
    train_rows,
    test_rows,
    *,
    seed: int = 42,
    adaptation_epochs: int = 3,
    classifier_epochs: int = 4,
    adaptation_learning_rate: float = 5e-5,
    classifier_learning_rate: float = 3e-5,
    batch_size: int = 16,
    patience: int = 2,
    max_length: int = 128,
    mlm_probability: float = 0.15,
    model_name: str = MODEL_NAME,
    model_revision: str = MODEL_REVISION,
    artifact_key: str = "domain_minilm",
    device: torch.device | None = None,
    evaluate_official_test: bool = True,
):
    """Adapt on training reviews, then select the classifier on development F1."""
    validate_training_parameters(
        epochs=classifier_epochs,
        batch_size=batch_size,
        learning_rate=classifier_learning_rate,
        weight_decay=0.01,
        max_length=max_length,
        patience=patience,
    )
    if adaptation_epochs < 1:
        raise ValueError("adaptation_epochs must be positive")
    if adaptation_learning_rate <= 0:
        raise ValueError("adaptation_learning_rate must be positive")
    if not 0 < mlm_probability < 1:
        raise ValueError("mlm_probability must be between 0 and 1")

    loader_generator = seed_everything(seed)
    splits = split_official_data(train_rows, test_rows, seed=seed)
    active_device = device or preferred_device()
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=model_revision,
    )

    unique_reviews = list(dict.fromkeys(row.review_raw for row in splits.train))
    review_dataset = _ReviewDataset(tokenizer, unique_reviews, max_length)
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=mlm_probability,
    )
    adaptation_loader = DataLoader(
        review_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=loader_generator,
        collate_fn=collator,
    )
    language_model = AutoModelForMaskedLM.from_pretrained(
        model_name,
        revision=model_revision,
    ).to(active_device)
    adaptation_optimizer = torch.optim.AdamW(
        language_model.parameters(),
        lr=adaptation_learning_rate,
        weight_decay=0.01,
    )
    adaptation_history = []
    adaptation_started = perf_counter()
    for epoch in range(1, adaptation_epochs + 1):
        language_model.train()
        epoch_loss = 0.0
        for batch in adaptation_loader:
            features = {name: value.to(active_device) for name, value in batch.items()}
            adaptation_optimizer.zero_grad()
            loss = language_model(**features).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(language_model.parameters(), 1.0)
            adaptation_optimizer.step()
            epoch_loss += loss.detach().cpu().item()
        adaptation_history.append(
            {
                "epoch": epoch,
                "train_loss": epoch_loss / len(adaptation_loader),
            }
        )
    adaptation_seconds = perf_counter() - adaptation_started

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        revision=model_revision,
        num_labels=len(LABEL_TO_ID),
    )
    incompatible = model.bert.load_state_dict(
        language_model.bert.state_dict(),
        strict=False,
    )
    expected_missing = {"pooler.dense.weight", "pooler.dense.bias"}
    if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "Unexpected MLM encoder transfer mismatch: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    del language_model
    model = model.to(active_device)
    model.config.id2label = dict(ID_TO_LABEL)
    model.config.label2id = dict(LABEL_TO_ID)

    classifier_loader = DataLoader(
        AspectPairDataset(tokenizer, splits.train, max_length=max_length),
        batch_size=batch_size,
        shuffle=True,
        generator=loader_generator,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=classifier_learning_rate,
        weight_decay=0.01,
    )
    loss_function = torch.nn.CrossEntropyLoss()
    inverse = {value: key for key, value in LABEL_TO_ID.items()}

    def score(rows):
        model.eval()
        loader = DataLoader(
            AspectPairDataset(tokenizer, rows, max_length=max_length),
            batch_size=batch_size,
        )
        predictions = []
        with torch.no_grad():
            for batch in loader:
                batch.pop("labels")
                features = {
                    name: value.to(active_device) for name, value in batch.items()
                }
                predictions.extend(model(**features).logits.argmax(1).cpu().tolist())
        return compute_metrics(
            [row.label for row in rows],
            [inverse[index] for index in predictions],
        )

    checkpoint = BestCheckpoint(patience=patience)
    history = []
    stopped_early = False
    classifier_started = perf_counter()
    for epoch in range(1, classifier_epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch in classifier_loader:
            labels = batch.pop("labels").to(active_device)
            features = {
                name: value.to(active_device) for name, value in batch.items()
            }
            optimizer.zero_grad()
            loss = loss_function(model(**features).logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.detach().cpu().item()
        development = score(splits.development)
        history.append(
            {
                "epoch": epoch,
                "train_loss": epoch_loss / len(classifier_loader),
                "development_macro_f1": float(development["macro_f1"]),
            }
        )
        if checkpoint.update(model, float(development["macro_f1"]), epoch):
            stopped_early = epoch < classifier_epochs
            break
    classifier_seconds = perf_counter() - classifier_started

    checkpoint.restore(model)
    development = score(splits.development)
    test = (
        score(splits.test)
        if evaluate_official_test
        else {"status": "not_evaluated_during_configuration_selection"}
    )
    config = {
        "model": artifact_key,
        "pretrained_model": model_name,
        "pretrained_revision": model_revision,
        "seed": seed,
        "device": str(active_device),
        "adaptation_objective": "masked_language_modeling",
        "adaptation_corpus": "unique_official_train_reviews_only",
        "adaptation_epochs": adaptation_epochs,
        "adaptation_learning_rate": adaptation_learning_rate,
        "mlm_probability": mlm_probability,
        "epochs_requested": classifier_epochs,
        "epochs_completed": len(history),
        "batch_size": batch_size,
        "learning_rate": classifier_learning_rate,
        "optimizer": "AdamW",
        "weight_decay": 0.01,
        "patience": patience,
        "max_length": max_length,
        "configuration_selection": "development_macro_f1_only",
        "official_test_evaluated": evaluate_official_test,
    }
    result = build_run_result(
        development=development,
        test=test,
        history=history,
        checkpoint=checkpoint,
        config=config,
        stopped_early=stopped_early,
        training_seconds=adaptation_seconds + classifier_seconds,
    )
    result["adaptation"] = {
        "history": adaptation_history,
        "training_seconds": adaptation_seconds,
        "review_count": len(unique_reviews),
    }
    result["parameter_count"] = sum(
        parameter.numel() for parameter in model.parameters()
    )
    return model, tokenizer, result


def save_artifact(
    model,
    tokenizer,
    metrics,
    output_dir: Path = ABSA_OUTPUTS_DIR,
    *,
    artifact_key: str = "domain_minilm",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / artifact_key
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    (checkpoint_dir / "training_run.json").write_text(
        json.dumps(checkpoint_metadata(metrics), indent=2) + "\n"
    )
    artifact_bytes = sum(
        path.stat().st_size for path in checkpoint_dir.rglob("*") if path.is_file()
    )
    metrics["artifact_bytes"] = artifact_bytes
    metrics["artifact_megabytes"] = artifact_bytes / (1024 * 1024)
    (output_dir / f"{artifact_key}_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )
