import json
from pathlib import Path
from time import perf_counter

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from ..config import ABSA_OUTPUTS_DIR
from ..data.splits import split_official_data
from ..evaluation import compute_metrics
from ..labels import ID_TO_LABEL, LABEL_TO_ID
from ..models.distilbert import ABSADistilBERT
from ..tokenization.bert_dataset import AspectPairDataset
from .common import (
    BestCheckpoint,
    build_run_result,
    checkpoint_metadata,
    seed_everything,
    validate_training_parameters,
)


def preferred_device() -> torch.device:
    """Prefer CUDA, then Apple Metal, while retaining CPU compatibility."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_distilbert(
    train_rows,
    test_rows,
    *,
    epochs: int = 2,
    batch_size: int = 8,
    seed: int = 42,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    patience: int = 2,
    max_length: int = 128,
    model_name: str = "distilbert-base-uncased",
    device: torch.device | None = None,
    evaluate_official_test: bool = True,
    class_weighting: str = "none",
    warmup_ratio: float = 0.0,
):
    """Train DistilBERT, selecting checkpoints only on development macro-F1."""
    validate_training_parameters(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_length=max_length,
        patience=patience,
    )
    loader_generator = seed_everything(seed)
    splits = split_official_data(train_rows, test_rows, seed=seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    active_device = device or preferred_device()
    model = ABSADistilBERT.from_pretrained_absa(model_name).to(active_device)
    model.config.id2label = dict(ID_TO_LABEL)
    model.config.label2id = dict(LABEL_TO_ID)
    if class_weighting not in {"none", "sqrt_balanced", "balanced"}:
        raise ValueError(
            "class_weighting must be one of: none, sqrt_balanced, balanced"
        )
    if not 0 <= warmup_ratio < 1:
        raise ValueError("warmup_ratio must be in [0, 1)")
    loader = DataLoader(
        AspectPairDataset(tokenizer, splits.train, max_length=max_length),
        batch_size=batch_size,
        shuffle=True,
        generator=loader_generator,
    )
    optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    total_steps = epochs * len(loader)
    scheduler = (
        get_linear_schedule_with_warmup(
            optimiser,
            num_warmup_steps=int(total_steps * warmup_ratio),
            num_training_steps=total_steps,
        )
        if warmup_ratio > 0
        else None
    )
    class_weights = None
    if class_weighting != "none":
        counts = torch.bincount(
            torch.tensor([LABEL_TO_ID[row.label] for row in splits.train]),
            minlength=len(LABEL_TO_ID),
        ).float()
        class_weights = counts.sum() / (len(LABEL_TO_ID) * counts)
        if class_weighting == "sqrt_balanced":
            class_weights = class_weights.sqrt()
        class_weights = class_weights.to(active_device)
    loss_function = torch.nn.CrossEntropyLoss(weight=class_weights)

    inverse = {value: key for key, value in LABEL_TO_ID.items()}

    def score(rows):
        model.eval()
        evaluation_loader = DataLoader(
            AspectPairDataset(tokenizer, rows, max_length=max_length),
            batch_size=batch_size,
        )
        predicted = []
        with torch.no_grad():
            for batch in evaluation_loader:
                batch.pop("labels")
                features = {name: tensor.to(active_device) for name, tensor in batch.items()}
                predicted.extend(model(**features).logits.argmax(1).cpu().tolist())
        return compute_metrics([row.label for row in rows], [inverse[item] for item in predicted])

    checkpoint = BestCheckpoint(patience=patience)
    history: list[dict[str, float | int]] = []
    stopped_early = False
    training_started = perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch in loader:
            labels = batch.pop("labels").to(active_device)
            features = {name: tensor.to(active_device) for name, tensor in batch.items()}
            optimiser.zero_grad()
            loss = loss_function(model(**features).logits, labels)
            loss.backward()
            optimiser.step()
            if scheduler is not None:
                scheduler.step()
            epoch_loss += loss.detach().cpu().item()
        development = score(splits.development)
        history.append(
            {
                "epoch": epoch,
                "train_loss": epoch_loss / len(loader),
                "development_macro_f1": float(development["macro_f1"]),
            }
        )
        if checkpoint.update(model, float(development["macro_f1"]), epoch):
            stopped_early = epoch < epochs
            break
    training_seconds = perf_counter() - training_started

    checkpoint.restore(model)
    development = score(splits.development)
    test = (
        score(splits.test)
        if evaluate_official_test
        else {"status": "not_evaluated_during_configuration_selection"}
    )
    config = {
        "model": "distilbert",
        "pretrained_model": model_name,
        "seed": seed,
        "device": str(active_device),
        "epochs_requested": epochs,
        "epochs_completed": len(history),
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "optimizer": "AdamW",
        "weight_decay": weight_decay,
        "patience": patience,
        "max_length": max_length,
        "configuration_selection": "development_macro_f1_only",
        "official_test_evaluated": evaluate_official_test,
        "class_weighting": class_weighting,
        "learning_rate_schedule": (
            "linear_warmup_decay" if scheduler is not None else "constant"
        ),
        "warmup_ratio": warmup_ratio,
    }
    return model, tokenizer, build_run_result(
        development=development,
        test=test,
        history=history,
        checkpoint=checkpoint,
        config=config,
        stopped_early=stopped_early,
        training_seconds=training_seconds,
    )


def save_artifact(model, tokenizer, metrics, output_dir: Path = ABSA_OUTPUTS_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "distilbert"
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    (checkpoint_dir / "training_run.json").write_text(
        json.dumps(checkpoint_metadata(metrics), indent=2) + "\n"
    )
    (output_dir / "distilbert_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
