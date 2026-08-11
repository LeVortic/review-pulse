from pathlib import Path

import pytest

from src.absa.inference import predictors
from src.absa.inference.api import predict_aspects
from src.absa.inference.predictors import (
    ALL_MODEL_OPTIONS,
    BertSmallFp16AspectPredictor,
    COMPARISON_MODEL_OPTIONS,
    DistilBertAspectPredictor,
    MODEL_OPTIONS,
    OPTIONAL_MODEL_OPTIONS,
    get_predictor,
)


class _Predictor:
    def __init__(self):
        self.reviews = []

    def predict(self, review, aspect, model):
        self.reviews.append(review)
        return {"aspect": aspect, "label": "positive", "model": model}


def test_predict_aspects_deduplicates_and_preserves_order():
    result = predict_aspects("great food", ["food", " Service ", "FOOD"], "absa_tfidf", _Predictor())
    assert [item["aspect"] for item in result] == ["food", "Service"]


def test_predict_aspects_preserves_visible_review_whitespace_for_offsets():
    predictor = _Predictor()
    predict_aspects("  Great food!  ", ["food"], "absa_tfidf", predictor)
    assert predictor.reviews == ["  Great food!  "]


def test_predictor_registry_rejects_unknown_model_without_loading_an_artifact():
    assert "absa_atae_lstm" in MODEL_OPTIONS
    assert "absa_target_gru" in OPTIONAL_MODEL_OPTIONS
    assert "absa_text_cnn" in OPTIONAL_MODEL_OPTIONS
    assert "absa_bert_small_fp16" in OPTIONAL_MODEL_OPTIONS
    assert list(ALL_MODEL_OPTIONS) == [
        "absa_tfidf",
        "absa_target_lstm",
        "absa_target_gru",
        "absa_text_cnn",
        "absa_bert_small_fp16",
        "absa_atae_lstm",
        "absa_distilbert",
    ]
    assert list(COMPARISON_MODEL_OPTIONS) == [
        "absa_tfidf",
        "absa_target_lstm",
        "absa_atae_lstm",
        "absa_bert_small_fp16",
        "absa_distilbert",
    ]
    try:
        get_predictor("not-a-model")
    except ValueError as error:
        assert "Unknown v3 model" in str(error)
    else:
        raise AssertionError("Unknown models must not silently select a predictor")


def test_distilbert_predictor_reports_missing_local_artifact(tmp_path: Path) -> None:
    missing = tmp_path / "distilbert"

    with pytest.raises(FileNotFoundError) as caught:
        DistilBertAspectPredictor(missing)

    message = str(caught.value)
    assert str(missing) in message
    assert "git lfs pull" in message
    assert "quickstart.md#path-b---run-all-six-v3-models-from-github" in message


def test_bert_small_predictor_reports_missing_local_artifact(tmp_path: Path) -> None:
    missing = tmp_path / "domain_bert_small_fp16"

    with pytest.raises(FileNotFoundError, match="BERT-Small FP16"):
        BertSmallFp16AspectPredictor(missing)


def test_bert_small_predictor_upcasts_fp16_artifact_for_cpu_attribution(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class _Model:
        float_called = False
        eval_called = False

        def float(self):
            self.float_called = True
            return self

        def eval(self):
            self.eval_called = True
            return self

    model = _Model()
    monkeypatch.setattr(
        predictors.AutoTokenizer,
        "from_pretrained",
        lambda path, local_files_only: object(),
    )
    monkeypatch.setattr(
        predictors.AutoModelForSequenceClassification,
        "from_pretrained",
        lambda path, local_files_only: model,
    )

    loaded = BertSmallFp16AspectPredictor(tmp_path)

    assert loaded.model is model
    assert model.float_called is True
    assert model.eval_called is True


def test_v3_page_uses_explicit_selection_and_comparison_registries():
    page = Path("pages/2_ReviewPulse_v3_0_0.py").read_text()
    assert "ALL_MODEL_OPTIONS" in page
    assert "COMPARISON_MODEL_OPTIONS" in page
    assert "OPTIONAL_MODEL_OPTIONS" not in page
