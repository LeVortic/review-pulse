"""Export and verify a sub-100 MB FP16 BERT-Small artifact."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.absa.config import ABSA_DATA_DIR, ABSA_OUTPUTS_DIR
from src.absa.data.parser import parse_aspect_examples
from src.absa.data.splits import split_official_data
from src.absa.evaluation import compute_metrics
from src.absa.labels import LABEL_TO_ID
from src.absa.tokenization.bert_dataset import AspectPairDataset
from src.absa.training.distilbert import preferred_device
from src.absa.training.provenance import file_sha256, git_commit


def _score(model, tokenizer, rows, device: torch.device, batch_size: int, max_length: int):
    inverse = {value: key for key, value in LABEL_TO_ID.items()}
    loader = DataLoader(
        AspectPairDataset(tokenizer, rows, max_length=max_length),
        batch_size=batch_size,
    )
    predictions = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch.pop("labels")
            features = {name: value.to(device) for name, value in batch.items()}
            predictions.extend(model(**features).logits.argmax(1).cpu().tolist())
    return compute_metrics(
        [row.label for row in rows],
        [inverse[index] for index in predictions],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ABSA_DATA_DIR / "restaurants")
    parser.add_argument("--output-dir", type=Path, default=ABSA_OUTPUTS_DIR)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()

    source = args.output_dir / "domain_bert_small"
    destination = args.output_dir / "domain_bert_small_fp16"
    if not (source / "model.safetensors").exists():
        raise FileNotFoundError(f"Missing source checkpoint: {source}")

    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        source,
        local_files_only=True,
    ).half()
    model.save_pretrained(destination)
    tokenizer.save_pretrained(destination)

    active_device = preferred_device()
    reloaded = AutoModelForSequenceClassification.from_pretrained(
        destination,
        local_files_only=True,
        dtype="auto",
    ).to(active_device)
    source_metrics = json.loads(
        (args.output_dir / "domain_bert_small_metrics.json").read_text()
    )
    train_path = args.data_dir / "restaurants_train.xml"
    test_path = args.data_dir / "restaurants_test.xml"
    splits = split_official_data(
        parse_aspect_examples(train_path, "train"),
        parse_aspect_examples(test_path, "test"),
        seed=source_metrics["config"]["seed"],
    )
    test = _score(
        reloaded,
        tokenizer,
        splits.test,
        active_device,
        args.batch_size,
        args.max_length,
    )
    artifact_bytes = sum(
        path.stat().st_size for path in destination.rglob("*") if path.is_file()
    )
    payload = {
        "model": "domain_bert_small_fp16",
        "source_artifact": str(source),
        "precision": "float16",
        "device_verified": str(active_device),
        "artifact_bytes": artifact_bytes,
        "artifact_megabytes": artifact_bytes / (1024 * 1024),
        "under_100_mb": artifact_bytes < 100 * 1024 * 1024,
        "source_test": source_metrics["test"],
        "compressed_test": test,
        "macro_f1_delta": test["macro_f1"] - source_metrics["test"]["macro_f1"],
        "provenance": {
            "git_commit": git_commit(),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "train_file": str(train_path),
            "train_sha256": file_sha256(train_path),
            "test_file": str(test_path),
            "test_sha256": file_sha256(test_path),
        },
    }
    if not payload["under_100_mb"]:
        raise RuntimeError(
            f"Compressed artifact is {payload['artifact_megabytes']:.2f} MB, not under 100 MB"
        )
    (destination / "compression_run.json").write_text(json.dumps(payload, indent=2) + "\n")
    (args.output_dir / "domain_bert_small_fp16_metrics.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
