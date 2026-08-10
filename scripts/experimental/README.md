# Experimental model scripts
the changes in `src/absa/training/distilbert.py` extend the existing DistilBERT trainer 
so it can support leakage-safe hyperparameter experiments while preserving its original 
default behavior.

These entry points reproduce bounded model-selection and artifact-size
experiments. They are intentionally separate from the canonical
`src.absa.training.runner` and `src.absa.evaluation.runner` pipeline.

## Compact BERT experiments

- `train_domain_minilm.py` trains the original domain-adapted BERT-Mini baseline.
- `select_domain_bert_mini_config.py` selects BERT-Mini hyperparameters using
  development macro-F1 and evaluates only the selected configuration on the
  official test set.
- `train_domain_bert_small.py` trains the FP32 BERT-Small comparison artifact.
- `export_domain_bert_small_fp16.py` exports BERT-Small in FP16 and verifies the
  reloaded artifact on the canonical three-class official test split.

## DistilBERT experiment

- `select_distilbert_config.py` runs the bounded development-only DistilBERT
  configuration gate.

Run scripts from the repository root, for example:

```bash
.venv/bin/python scripts/experimental/select_domain_bert_mini_config.py
.venv/bin/python scripts/experimental/export_domain_bert_small_fp16.py
```

Generated artifacts are written under `outputs/absa/`. These scripts do not add
their candidates to the canonical model registry or evaluation report.
