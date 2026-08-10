"""Run a bounded development-only configuration gate for DistilBERT."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.absa.config import ABSA_DATA_DIR, ABSA_OUTPUTS_DIR
from src.absa.data.parser import parse_aspect_examples
from src.absa.training.distilbert import preferred_device, train_distilbert
from src.absa.training.provenance import file_sha256, git_commit


CANDIDATES = (
    {
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "class_weighting": "none",
        "warmup_ratio": 0.0,
    },
    {
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "class_weighting": "none",
        "warmup_ratio": 0.05,
    },
    {
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "class_weighting": "none",
        "warmup_ratio": 0.1,
    },
    {
        "learning_rate": 3e-5,
        "weight_decay": 0.01,
        "class_weighting": "none",
        "warmup_ratio": 0.1,
    },
)


def _device(value: str) -> torch.device:
    if value == "auto":
        return preferred_device()
    device = torch.device(value)
    if value == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if value == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")
    return device


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select DistilBERT hyperparameters on development macro-F1."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ABSA_DATA_DIR / "restaurants",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ABSA_OUTPUTS_DIR / "distilbert_config_search.json",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
    )
    args = parser.parse_args()

    train_path = args.data_dir / "restaurants_train.xml"
    test_path = args.data_dir / "restaurants_test.xml"
    train_rows = parse_aspect_examples(train_path, "train")
    test_rows = parse_aspect_examples(test_path, "test")
    active_device = _device(args.device)
    records = []
    for candidate in CANDIDATES:
        _model, _tokenizer, result = train_distilbert(
            train_rows,
            test_rows,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            learning_rate=candidate["learning_rate"],
            weight_decay=candidate["weight_decay"],
            class_weighting=candidate["class_weighting"],
            warmup_ratio=candidate["warmup_ratio"],
            patience=args.patience,
            max_length=args.max_length,
            device=active_device,
            evaluate_official_test=False,
        )
        records.append(
            {
                **candidate,
                "best_epoch": result["best_epoch"],
                "development_macro_f1": result["development"]["macro_f1"],
                "training_seconds": result["training_seconds"],
                "official_test_evaluated": result["config"][
                    "official_test_evaluated"
                ],
            }
        )

    selected = max(
        enumerate(records),
        key=lambda item: (item[1]["development_macro_f1"], -item[0]),
    )[1]
    payload = {
        "selection_metric": "development_macro_f1",
        "official_test_evaluated": False,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "patience": args.patience,
        "max_length": args.max_length,
        "candidate_limit": len(CANDIDATES),
        "device": str(active_device),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "git_commit": git_commit(),
            "train_file": str(train_path),
            "train_sha256": file_sha256(train_path),
            "test_file": str(test_path),
            "test_sha256": file_sha256(test_path),
        },
        "candidates": records,
        "selected": {
            "learning_rate": selected["learning_rate"],
            "weight_decay": selected["weight_decay"],
            "class_weighting": selected["class_weighting"],
            "warmup_ratio": selected["warmup_ratio"],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
