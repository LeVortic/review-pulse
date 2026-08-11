import re

import numpy as np
import pytest
import torch

from src.absa.inference import predictors
from src.absa.interpretability.evidence import (
    ATTENTION_METHOD,
    ATTRIBUTION_METHOD,
    supported_evidence,
    unsupported_evidence,
)
from src.absa.interpretability.heatmap import (
    ALPHA_FLOOR,
    ALPHA_RANGE,
    render_token_heatmap_html,
)


def test_supported_and_unsupported_payloads_are_explicit():
    supported = supported_evidence(
        aspect="food",
        method=ATTENTION_METHOD,
        tokens=[{"token": "Food", "start": 0, "end": 4, "score": 1.0}],
        limitations="Limited.",
    )
    unsupported = unsupported_evidence("TF-IDF review-only")
    assert supported["status"] == "supported"
    assert supported["aspect"] == "food"
    assert "causal" in supported["caveat"]
    assert unsupported["status"] == "unsupported"
    assert unsupported["tokens"] == []
    assert "review-only" in unsupported["limitations"]


def test_heatmap_preserves_visible_text_and_escapes_html():
    review = "<Great> food!"
    tokens = [
        {"token": "<", "start": 0, "end": 1, "score": 0.1},
        {"token": "Great", "start": 1, "end": 6, "score": 0.8},
        {"token": ">", "start": 6, "end": 7, "score": 0.1},
        {"token": "food", "start": 8, "end": 12, "score": 0.5},
        {"token": "!", "start": 12, "end": 13, "score": 0.2},
    ]
    rendered = render_token_heatmap_html(review, tokens)
    assert "&lt;" in rendered and "&gt;" in rendered
    assert "<Great>" not in rendered
    assert "white-space: pre-wrap" in rendered
    assert "token score: 0.8000" in rendered


def test_heatmap_leaves_the_weakest_token_effectively_unshaded():
    """Softmax attention clusters in a narrow band; the floor must still read as off."""
    review = "abcd"
    tokens = [
        {"token": "a", "start": 0, "end": 1, "score": 0.24},
        {"token": "b", "start": 1, "end": 2, "score": 0.25},
        {"token": "c", "start": 2, "end": 3, "score": 0.26},
        {"token": "d", "start": 3, "end": 4, "score": 0.25},
    ]
    rendered = render_token_heatmap_html(review, tokens)
    alphas = [float(value) for value in re.findall(r"rgba\(255, 126, 95, ([0-9.]+)\)", rendered)]
    assert alphas[0] == pytest.approx(ALPHA_FLOOR)
    assert alphas[2] == pytest.approx(ALPHA_FLOOR + ALPHA_RANGE)
    assert max(alphas) - min(alphas) == pytest.approx(ALPHA_RANGE)


def test_heatmap_uses_a_flat_floor_when_every_score_is_equal():
    review = "ab"
    tokens = [
        {"token": "a", "start": 0, "end": 1, "score": 0.5},
        {"token": "b", "start": 1, "end": 2, "score": 0.5},
    ]
    rendered = render_token_heatmap_html(review, tokens)
    alphas = [float(value) for value in re.findall(r"rgba\(255, 126, 95, ([0-9.]+)\)", rendered)]
    assert alphas == [pytest.approx(ALPHA_FLOOR), pytest.approx(ALPHA_FLOOR)]


def test_heatmap_tooltip_reports_the_raw_score_not_the_scaled_value():
    review = "ab"
    tokens = [
        {"token": "a", "start": 0, "end": 1, "score": 0.0412},
        {"token": "b", "start": 1, "end": 2, "score": 0.1180},
    ]
    rendered = render_token_heatmap_html(review, tokens)
    assert "token score: 0.0412" in rendered
    assert "token score: 0.1180" in rendered


def test_heatmap_rejects_misaligned_payload():
    with pytest.raises(ValueError, match="not aligned"):
        render_token_heatmap_html(
            "Great food", [{"token": "great", "start": 0, "end": 5, "score": 1.0}]
        )


class _FakeBaseline:
    classes_ = ["negative", "neutral", "positive"]

    def predict_proba(self, _reviews):
        return np.array([[0.1, 0.2, 0.7]])


def test_tfidf_predictor_reports_unsupported_evidence(monkeypatch):
    monkeypatch.setattr(predictors.joblib, "load", lambda _path: _FakeBaseline())
    result = predictors.TfidfAspectPredictor().predict(
        "Great food", "food", "absa_tfidf"
    )
    assert result["token_evidence"]["status"] == "unsupported"
    assert "review-only" in result["token_evidence"]["limitations"]


def test_distilbert_predictor_returns_attribution_payload(monkeypatch):
    predictor = object.__new__(predictors.DistilBertAspectPredictor)
    predictor.model = object()
    predictor.tokenizer = object()
    monkeypatch.setattr(
        predictors,
        "gradient_x_input_attribution",
        lambda *_args: (
            torch.tensor([[0.1, 0.2, 0.7]]),
            [{"token": "Great", "start": 0, "end": 5, "score": 1.0}],
        ),
    )
    result = predictor.predict("Great food", "food", "absa_distilbert")
    assert result["label"] == "positive"
    assert result["token_evidence"]["status"] == "supported"
    assert result["token_evidence"]["method"] == ATTRIBUTION_METHOD


def test_bert_small_predictor_returns_attribution_payload(monkeypatch):
    predictor = object.__new__(predictors.BertSmallFp16AspectPredictor)
    predictor.model = object()
    predictor.tokenizer = object()
    monkeypatch.setattr(
        predictors,
        "gradient_x_input_attribution",
        lambda *_args: (
            torch.tensor([[0.1, 0.2, 0.7]]),
            [{"token": "Great", "start": 0, "end": 5, "score": 1.0}],
        ),
    )
    result = predictor.predict("Great food", "food", "absa_bert_small_fp16")
    assert result["label"] == "positive"
    assert result["token_evidence"]["status"] == "supported"
    assert result["token_evidence"]["method"] == ATTRIBUTION_METHOD
