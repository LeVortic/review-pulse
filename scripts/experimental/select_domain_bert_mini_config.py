"""Tune BERT-Mini on development macro-F1, then evaluate one winner."""

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


CANDIDATES = (
    {"adaptation_epochs": 1, "adaptation_learning_rate": 5e-5, "classifier_learning_rate": 2e-5},
    {"adaptation_epochs": 1, "adaptation_learning_rate": 5e-5, "classifier_learning_rate": 3e-5},
    {"adaptation_epochs": 3, "adaptation_learning_rate": 5e-5, "classifier_learning_rate": 2e-5},
    {"adaptation_epochs": 3, "adaptation_learning_rate": 5e-5, "classifier_learning_rate": 3e-5},
    {"adaptation_epochs": 3, "adaptation_learning_rate": 5e-5, "classifier_learning_rate": 5e-5},
    {"adaptation_epochs": 5, "adaptation_learning_rate": 3e-5, "classifier_learning_rate": 3e-5},
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ABSA_DATA_DIR / "restaurants")
    parser.add_argument("--output-dir", type=Path, default=ABSA_OUTPUTS_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--classifier-epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=3)
    args = parser.parse_args()

    train_path = args.data_dir / "restaurants_train.xml"
    test_path = args.data_dir / "restaurants_test.xml"
    train_rows = parse_aspect_examples(train_path, "train")
    test_rows = parse_aspect_examples(test_path, "test")
    records = []
    for candidate in CANDIDATES:
        _model, _tokenizer, result = train_domain_minilm(
            train_rows,
            test_rows,
            seed=args.seed,
            classifier_epochs=args.classifier_epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            evaluate_official_test=False,
            **candidate,
        )
        records.append(
            {
                **candidate,
                "best_epoch": result["best_epoch"],
                "development_macro_f1": result["development"]["macro_f1"],
                "training_seconds": result["training_seconds"],
                "official_test_evaluated": result["config"]["official_test_evaluated"],
            }
        )

    selected_index, selected = max(
        enumerate(records),
        key=lambda item: (item[1]["development_macro_f1"], -item[0]),
    )
    selected_config = {key: selected[key] for key in CANDIDATES[0]}
    model, tokenizer, final = train_domain_minilm(
        train_rows,
        test_rows,
        seed=args.seed,
        classifier_epochs=args.classifier_epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        evaluate_official_test=True,
        artifact_key="domain_bert_mini_tuned",
        **selected_config,
    )
    provenance = {
        "git_commit": git_commit(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "train_file": str(train_path),
        "train_sha256": file_sha256(train_path),
        "test_file": str(test_path),
        "test_sha256": file_sha256(test_path),
    }
    final["provenance"] = provenance
    final["configuration_search"] = {
        "selection_metric": "development_macro_f1",
        "candidate_limit": len(CANDIDATES),
        "official_test_evaluated_during_search": False,
        "selected_candidate_index": selected_index,
        "candidates": records,
    }
    save_artifact(
        model,
        tokenizer,
        final,
        args.output_dir,
        artifact_key="domain_bert_mini_tuned",
    )
    search = {
        "selection_metric": "development_macro_f1",
        "official_test_evaluated_during_search": False,
        "seed": args.seed,
        "classifier_epochs": args.classifier_epochs,
        "batch_size": args.batch_size,
        "patience": args.patience,
        "provenance": provenance,
        "candidates": records,
        "selected": selected_config,
        "final": {
            "development_macro_f1": final["development"]["macro_f1"],
            "test_accuracy": final["test"]["accuracy"],
            "test_macro_f1": final["test"]["macro_f1"],
            "best_epoch": final["best_epoch"],
            "artifact_megabytes": final["artifact_megabytes"],
        },
    }
    output_path = args.output_dir / "domain_bert_mini_config_search.json"
    output_path.write_text(json.dumps(search, indent=2) + "\n")
    print(json.dumps(search, indent=2))


if __name__ == "__main__":
    main()
