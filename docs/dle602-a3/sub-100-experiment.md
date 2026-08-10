### Sub-100 MB compact-encoder experiment

Two candidates were tested under a strict 100 MB artifact limit. BERT-Mini was
domain-adapted on unique official training reviews and tuned with six bounded
configurations. Candidate selection used development macro-F1 only; the official
test was evaluated once after the winner was fixed. BERT-Small retained its
validated FP32 weights and was exported to FP16, then reloaded and evaluated from
the compressed artifact.

| Candidate | Precision | Artifact | Test accuracy | Test macro-F1 |
|---|---|---:|---:|---:|
| Tuned BERT-Mini | FP32 | **43.31 MB** | 0.7598 | 0.6531 |
| BERT-Small | FP16 | 55.55 MB | **0.7938** | **0.6951** |

Tuning improved BERT-Mini from macro-F1 `0.6231` to `0.6531`. The winning
configuration used one masked-language-model adaptation epoch, adaptation learning
rate `5e-5`, classifier learning rate `3e-5`, and selected classifier epoch 8.
BERT-Small FP16 reduced the artifact from `110.42 MB` to `55.55 MB` (49.7%) and
reproduced the FP32 confusion matrix exactly on CUDA, with macro-F1 delta `0.0000`.

Under the 100 MB constraint, **BERT-Small FP16 is the better predictive
candidate**: it exceeds tuned BERT-Mini by `0.0339` accuracy and `0.0420` macro-F1
for an additional `12.25 MB`. BERT-Mini remains preferable only when minimum
storage is more important than the observed quality difference. Both trail the
256.11 MB DistilBERT result (`0.8366` accuracy, `0.7490` macro-F1).

The compression comparison is an artifact-size and predictive-quality result,
not a CPU portability or latency claim. FP16 verification was performed on CUDA;
CPU and MPS compatibility require separate deployment tests.

```bash
# Compact encoder selection and compression experiments.
.venv/bin/python scripts/experimental/select_domain_bert_mini_config.py
.venv/bin/python scripts/experimental/export_domain_bert_small_fp16.py
```
