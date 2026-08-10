"""Train the isolated restaurant-domain-adapted BERT-Small experiment."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.absa.config import ABSA_DATA_DIR, ABSA_OUTPUTS_DIR
from src.absa.data.parser import parse_aspect_examples
from src.absa.training.domain_minilm import save_artifact, train_domain_minilm
from src.absa.training.provenance import file_sha256, git_commit


MODEL_NAME = "google/bert_uncased_L-4_H-512_A-8"
MODEL_REVISION = "606e4d55252882ac25ba1f1d1a182075830f5a90"
ARTIFACT_KEY = "domain_bert_small"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Domain-adapt and fine-tune BERT-Small for restaurant ABSA."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ABSA_DATA_DIR / "restaurants",
    )
    parser.add_argument("--output-dir", type=Path, default=ABSA_OUTPUTS_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--adaptation-epochs", type=int, default=3)
    parser.add_argument("--classifier-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    train_path = args.data_dir / "restaurants_train.xml"
    test_path = args.data_dir / "restaurants_test.xml"
    model, tokenizer, metrics = train_domain_minilm(
        parse_aspect_examples(train_path, "train"),
        parse_aspect_examples(test_path, "test"),
        seed=args.seed,
        adaptation_epochs=args.adaptation_epochs,
        classifier_epochs=args.classifier_epochs,
        batch_size=args.batch_size,
        model_name=MODEL_NAME,
        model_revision=MODEL_REVISION,
        artifact_key=ARTIFACT_KEY,
    )
    metrics["provenance"] = {
        "git_commit": git_commit(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "train_file": str(train_path),
        "train_sha256": file_sha256(train_path),
        "test_file": str(test_path),
        "test_sha256": file_sha256(test_path),
    }
    save_artifact(
        model,
        tokenizer,
        metrics,
        args.output_dir,
        artifact_key=ARTIFACT_KEY,
    )
    print(
        json.dumps(
            {
                "development_macro_f1": metrics["development"]["macro_f1"],
                "test_accuracy": metrics["test"]["accuracy"],
                "test_macro_f1": metrics["test"]["macro_f1"],
                "best_epoch": metrics["best_epoch"],
                "parameter_count": metrics["parameter_count"],
                "artifact_megabytes": metrics["artifact_megabytes"],
                "training_seconds": metrics["training_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
