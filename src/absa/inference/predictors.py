"""Artifact-backed predictor adapters for the ReviewPulse v3 contract."""

from pathlib import Path

import joblib
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ..config import ABSA_OUTPUTS_DIR
from ..interpretability.attention import align_attention
from ..interpretability.attribution import gradient_x_input_attribution
from ..interpretability.evidence import (
    ATTENTION_LIMITATIONS,
    ATTENTION_METHOD,
    ATTRIBUTION_LIMITATIONS,
    ATTRIBUTION_METHOD,
    supported_evidence,
    unsupported_evidence,
)
from ..labels import ID_TO_LABEL
from ..models.atae_lstm import ATAELSTM
from ..models.distilbert import ABSADistilBERT
from ..models.target_gru import TargetAgnosticGRU
from ..models.target_lstm import TargetAgnosticLSTM
from ..models.text_cnn import TextCNN
from ..tokenization.sequence import encode


MODEL_OPTIONS = {
    "absa_tfidf": "TF-IDF review-only",
    "absa_target_lstm": "LSTM review-only",
    "absa_atae_lstm": "ATAE-LSTM aspect-conditioned",
    "absa_distilbert": "DistilBERT sentence-pair",
}
OPTIONAL_MODEL_OPTIONS = {
    "absa_target_gru": "GRU review-only (exploratory)",
    "absa_text_cnn": "Text CNN review-only (exploratory)",
    "absa_bert_small_fp16": "BERT-Small sentence-pair (FP16 experiment)",
}
ALL_MODEL_OPTIONS = {
    "absa_tfidf": MODEL_OPTIONS["absa_tfidf"],
    "absa_target_lstm": MODEL_OPTIONS["absa_target_lstm"],
    **OPTIONAL_MODEL_OPTIONS,
    "absa_atae_lstm": MODEL_OPTIONS["absa_atae_lstm"],
    "absa_distilbert": MODEL_OPTIONS["absa_distilbert"],
}
COMPARISON_MODEL_OPTIONS = {
    "absa_tfidf": MODEL_OPTIONS["absa_tfidf"],
    "absa_target_lstm": MODEL_OPTIONS["absa_target_lstm"],
    "absa_atae_lstm": MODEL_OPTIONS["absa_atae_lstm"],
    "absa_bert_small_fp16": OPTIONAL_MODEL_OPTIONS["absa_bert_small_fp16"],
    "absa_distilbert": MODEL_OPTIONS["absa_distilbert"],
}

# Models that receive only the review and therefore repeat one prediction across
# aspects. They report token evidence as unsupported; the adapters below are the
# behavioural source of truth and `test_model_evidence_partition_is_exhaustive`
# guards this constant against drift.
REVIEW_ONLY_MODELS = frozenset(
    {"absa_tfidf", "absa_target_lstm", "absa_target_gru", "absa_text_cnn"}
)
ASPECT_CONDITIONED_MODELS = frozenset(
    {"absa_atae_lstm", "absa_bert_small_fp16", "absa_distilbert"}
)

DISTILBERT_ARTIFACT_HINT = (
    "Obtain the complete v3 artifact from the GitHub repository with Git LFS: "
    "run `git lfs pull` in a full checkout. See "
    "docs/dle602-a3/quickstart.md#path-b---run-all-six-v3-models-from-github."
)


def exposes_token_evidence(model_name: str) -> bool:
    """Return True when *model_name* can produce aspect-specific token evidence."""
    return model_name in ASPECT_CONDITIONED_MODELS


def _payload(aspect: str, logits: torch.Tensor, model_name: str, token_evidence: dict) -> dict:
    probabilities = torch.softmax(logits, dim=-1)[0]
    index = int(probabilities.argmax())
    return {
        "aspect": aspect,
        "label": ID_TO_LABEL[index],
        "confidence": float(probabilities[index]),
        "model": model_name,
        "token_evidence": token_evidence,
    }


class TfidfAspectPredictor:
    def __init__(self, path: Path = ABSA_OUTPUTS_DIR / "tfidf_baseline.joblib") -> None:
        self.model = joblib.load(path)

    def predict(self, review: str, aspect: str, model_name: str) -> dict:
        probabilities = self.model.predict_proba([review])[0]
        labels = list(self.model.classes_)
        index = probabilities.argmax()
        return {
            "aspect": aspect,
            "label": str(labels[index]),
            "confidence": float(probabilities[index]),
            "model": model_name,
            "token_evidence": unsupported_evidence(MODEL_OPTIONS[model_name]),
        }


class TargetLstmAspectPredictor:
    def __init__(self, path: Path = ABSA_OUTPUTS_DIR / "target_lstm.pt") -> None:
        artifact = torch.load(path, map_location="cpu", weights_only=True)
        self.vocab = artifact["vocab"]
        self.model = TargetAgnosticLSTM(len(self.vocab))
        self.model.load_state_dict(artifact["state_dict"])
        self.model.eval()

    def predict(self, review: str, aspect: str, model_name: str) -> dict:
        with torch.no_grad():
            logits = self.model(encode([review], self.vocab))
        return _payload(
            aspect,
            logits,
            model_name,
            unsupported_evidence(MODEL_OPTIONS[model_name]),
        )


class TargetGruAspectPredictor:
    """Clean-load adapter for the optional review-only GRU artifact."""

    def __init__(
        self,
        path: Path = ABSA_OUTPUTS_DIR / "target_gru.pt",
    ) -> None:
        artifact = torch.load(path, map_location="cpu", weights_only=True)
        self.vocab = artifact["vocab"]
        config = artifact["config"]
        self.max_length = int(config["max_length"])
        self.model = TargetAgnosticGRU(
            len(self.vocab),
            embedding_dim=int(config["embedding_dim"]),
            hidden_dim=int(config["hidden_dim"]),
            dropout=float(config["dropout"]),
        )
        self.model.load_state_dict(artifact["state_dict"])
        self.model.eval()

    def predict(self, review: str, aspect: str, model_name: str) -> dict:
        """Return the repeated review-only prediction for one supplied aspect."""
        with torch.no_grad():
            logits = self.model(
                encode(
                    [review],
                    self.vocab,
                    self.max_length,
                )
            )
        return _payload(
            aspect,
            logits,
            model_name,
            unsupported_evidence(ALL_MODEL_OPTIONS[model_name]),
        )


class TextCnnAspectPredictor:
    """Clean-load adapter for the optional review-only TextCNN artifact."""

    def __init__(
        self,
        path: Path = ABSA_OUTPUTS_DIR / "text_cnn.pt",
    ) -> None:
        artifact = torch.load(path, map_location="cpu", weights_only=True)
        self.vocab = artifact["vocab"]
        config = artifact["config"]
        self.max_length = int(config["max_length"])
        self.model = TextCNN(
            len(self.vocab),
            embedding_dim=int(config["embedding_dim"]),
            num_filters=int(config["num_filters"]),
            filter_widths=tuple(config["filter_widths"]),
            dropout=float(config["dropout"]),
        )
        self.model.load_state_dict(artifact["state_dict"])
        self.model.eval()

    def predict(self, review: str, aspect: str, model_name: str) -> dict:
        """Return the repeated review-only prediction for one supplied aspect."""
        with torch.no_grad():
            logits = self.model(
                encode(
                    [review],
                    self.vocab,
                    self.max_length,
                )
            )
        return _payload(
            aspect,
            logits,
            model_name,
            unsupported_evidence(ALL_MODEL_OPTIONS[model_name]),
        )


class AtaeLstmAspectPredictor:
    def __init__(self, path: Path = ABSA_OUTPUTS_DIR / "atae_lstm.pt") -> None:
        artifact = torch.load(path, map_location="cpu", weights_only=True)
        self.vocab = artifact["vocab"]
        self.model = ATAELSTM(len(self.vocab))
        self.model.load_state_dict(artifact["state_dict"])
        self.model.eval()

    def predict(self, review: str, aspect: str, model_name: str) -> dict:
        with torch.no_grad():
            logits, weights = self.model(
                encode([review], self.vocab),
                encode([aspect], self.vocab, 12),
                return_attention=True,
            )
        evidence = supported_evidence(
            aspect=aspect,
            method=ATTENTION_METHOD,
            tokens=align_attention(review, weights[0]),
            limitations=ATTENTION_LIMITATIONS,
        )
        return _payload(aspect, logits, model_name, evidence)


class DistilBertAspectPredictor:
    def __init__(self, path: Path = ABSA_OUTPUTS_DIR / "distilbert") -> None:
        path = Path(path)
        if not path.is_dir():
            raise FileNotFoundError(
                f"Missing v3 DistilBERT artifact directory: {path}\n\n"
                f"{DISTILBERT_ARTIFACT_HINT}"
            )
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = ABSADistilBERT.from_pretrained(path, local_files_only=True)
        self.model.eval()

    def predict(self, review: str, aspect: str, model_name: str) -> dict:
        logits, tokens = gradient_x_input_attribution(
            self.model,
            self.tokenizer,
            review,
            aspect,
        )
        evidence = supported_evidence(
            aspect=aspect,
            method=ATTRIBUTION_METHOD,
            tokens=tokens,
            limitations=ATTRIBUTION_LIMITATIONS,
        )
        return _payload(aspect, logits, model_name, evidence)


class BertSmallFp16AspectPredictor:
    """Adapter for the compressed domain-adapted BERT-Small experiment."""

    def __init__(
        self,
        path: Path = ABSA_OUTPUTS_DIR / "domain_bert_small_fp16",
    ) -> None:
        path = Path(path)
        if not path.is_dir():
            raise FileNotFoundError(
                f"Missing v3 BERT-Small FP16 artifact directory: {path}"
            )
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        # The on-disk weights remain FP16 (~55 MB). Upcast in memory because CPU
        # oneDNN does not universally support the FP16 backward pass required by
        # gradient × input attribution.
        self.model = AutoModelForSequenceClassification.from_pretrained(
            path,
            local_files_only=True,
        ).float()
        self.model.eval()

    def predict(self, review: str, aspect: str, model_name: str) -> dict:
        logits, tokens = gradient_x_input_attribution(
            self.model,
            self.tokenizer,
            review,
            aspect,
        )
        evidence = supported_evidence(
            aspect=aspect,
            method=ATTRIBUTION_METHOD,
            tokens=tokens,
            limitations=ATTRIBUTION_LIMITATIONS,
        )
        return _payload(aspect, logits, model_name, evidence)


def get_predictor(model_name: str):
    """Load an explicitly supported predictor, or fail with a controlled message."""
    predictors = {
        "absa_tfidf": TfidfAspectPredictor,
        "absa_target_lstm": TargetLstmAspectPredictor,
        "absa_target_gru": TargetGruAspectPredictor,
        "absa_text_cnn": TextCnnAspectPredictor,
        "absa_atae_lstm": AtaeLstmAspectPredictor,
        "absa_bert_small_fp16": BertSmallFp16AspectPredictor,
        "absa_distilbert": DistilBertAspectPredictor,
    }
    try:
        return predictors[model_name]()
    except KeyError as error:
        raise ValueError(f"Unknown v3 model: {model_name}") from error
