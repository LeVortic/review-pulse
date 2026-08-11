"""Tests for the aspect-by-model comparison builder."""

import pytest

from src.absa.inference.comparison import (
    ARTIFACT_MISSING,
    GOLD_COLUMN,
    MATRIX_COLUMNS,
    MISSING_LABEL,
    build_comparison,
    format_cell,
)
from src.absa.inference.predictors import ALL_MODEL_OPTIONS, MODEL_OPTIONS


CORE_MODELS = list(MODEL_OPTIONS)
REVIEW = "Great food but the service was dreadful!"
ASPECTS = ["food", "service"]


class _ReviewOnlyFake:
    """Ignores the aspect, as a review-only baseline does."""

    def predict(self, review: str, aspect: str, model_name: str) -> dict:
        return {"aspect": aspect, "label": "positive", "confidence": 0.714, "model": model_name}


class _AspectAwareFake:
    """Returns a different label per aspect."""

    def predict(self, review: str, aspect: str, model_name: str) -> dict:
        label = "positive" if aspect == "food" else "negative"
        return {"aspect": aspect, "label": label, "confidence": 0.9, "model": model_name}


def _factory(overrides: dict | None = None):
    overrides = overrides or {}

    def _get(model_name: str):
        if model_name in overrides:
            raise overrides[model_name]
        if model_name in {
            "absa_atae_lstm",
            "absa_bert_small_fp16",
            "absa_distilbert",
        }:
            return _AspectAwareFake()
        return _ReviewOnlyFake()

    return _get


def test_format_cell_uses_title_case_and_one_decimal():
    assert format_cell("positive", 0.8251) == "Positive · 82.5%"
    assert format_cell("negative", 1.0) == "Negative · 100.0%"


def test_matrix_is_aspects_by_models():
    comparison = build_comparison(REVIEW, ASPECTS, CORE_MODELS, _factory())
    assert list(comparison.display.index) == ASPECTS
    assert list(comparison.display.columns) == [MATRIX_COLUMNS[name] for name in CORE_MODELS]
    assert comparison.display.shape == (2, len(CORE_MODELS))
    assert comparison.labels.shape == comparison.display.shape


def test_review_only_columns_repeat_one_value_across_aspects():
    """This repetition is the RQ1 finding the matrix exists to surface."""
    comparison = build_comparison(REVIEW, ASPECTS, CORE_MODELS, _factory())
    review_only = comparison.display[MATRIX_COLUMNS["absa_tfidf"]]
    assert review_only.nunique() == 1
    aspect_aware = comparison.display[MATRIX_COLUMNS["absa_atae_lstm"]]
    assert aspect_aware.nunique() == 2


def test_missing_artifact_marks_only_its_own_column():
    comparison = build_comparison(
        REVIEW,
        ASPECTS,
        CORE_MODELS,
        _factory({"absa_distilbert": FileNotFoundError("no checkpoint")}),
    )
    missing = comparison.display[MATRIX_COLUMNS["absa_distilbert"]]
    assert list(missing) == [ARTIFACT_MISSING, ARTIFACT_MISSING]
    assert list(comparison.labels[MATRIX_COLUMNS["absa_distilbert"]]) == [MISSING_LABEL] * 2
    intact = comparison.display[MATRIX_COLUMNS["absa_tfidf"]]
    assert ARTIFACT_MISSING not in list(intact)


@pytest.mark.parametrize("error", [FileNotFoundError("gone"), OSError("unreadable"), RuntimeError("bad")])
def test_every_artifact_error_type_becomes_a_sentinel(error):
    comparison = build_comparison(REVIEW, ASPECTS, CORE_MODELS, _factory({"absa_tfidf": error}))
    assert list(comparison.display[MATRIX_COLUMNS["absa_tfidf"]]) == [ARTIFACT_MISSING] * 2


def test_all_models_missing_still_returns_a_full_shaped_matrix():
    factory = _factory({name: FileNotFoundError("gone") for name in CORE_MODELS})
    comparison = build_comparison(REVIEW, ASPECTS, CORE_MODELS, factory)
    assert comparison.display.shape == (2, len(CORE_MODELS))
    assert (comparison.display == ARTIFACT_MISSING).all().all()


def test_empty_aspects_propagate_the_existing_value_error():
    with pytest.raises(ValueError, match="at least one non-empty aspect"):
        build_comparison(REVIEW, [" ", ""], CORE_MODELS, _factory())


def test_empty_review_raises_before_loading_any_model():
    loaded: list[str] = []

    def _tracking(model_name: str):
        loaded.append(model_name)
        return _ReviewOnlyFake()

    with pytest.raises(ValueError, match="must not be empty"):
        build_comparison("   ", ASPECTS, CORE_MODELS, _tracking)
    assert loaded == []


def test_empty_model_list_raises():
    with pytest.raises(ValueError, match="at least one model"):
        build_comparison(REVIEW, ASPECTS, [], _factory())


def test_unknown_model_value_error_is_not_swallowed_as_missing_artifact():
    def _get(model_name: str):
        raise ValueError(f"Unknown v3 model: {model_name}")

    with pytest.raises(ValueError, match="Unknown v3 model"):
        build_comparison(REVIEW, ASPECTS, CORE_MODELS, _get)


def test_aspects_are_deduplicated_once_for_every_model():
    comparison = build_comparison(REVIEW, ["food", "Food", " food "], CORE_MODELS, _factory())
    assert list(comparison.display.index) == ["food"]


def test_models_are_loaded_sequentially_in_the_requested_order():
    """Phase 3 deliberately does not parallelise model execution."""
    order: list[str] = []

    def _tracking(model_name: str):
        order.append(model_name)
        return _ReviewOnlyFake()

    build_comparison(REVIEW, ASPECTS, CORE_MODELS, _tracking)
    assert order == CORE_MODELS


def test_gold_column_leads_the_matrix_when_supplied():
    comparison = build_comparison(
        REVIEW,
        ASPECTS,
        CORE_MODELS,
        _factory(),
        gold={"food": "positive", "service": "negative"},
    )
    assert list(comparison.display.columns)[0] == GOLD_COLUMN
    assert list(comparison.display[GOLD_COLUMN]) == ["Positive", "Negative"]
    assert list(comparison.labels[GOLD_COLUMN]) == ["positive", "negative"]


def test_gold_column_is_absent_when_no_gold_is_supplied():
    comparison = build_comparison(REVIEW, ASPECTS, CORE_MODELS, _factory())
    assert GOLD_COLUMN not in comparison.display.columns
    comparison = build_comparison(REVIEW, ASPECTS, CORE_MODELS, _factory(), gold={})
    assert GOLD_COLUMN not in comparison.display.columns


def test_gold_column_leaves_unannotated_aspects_blank():
    comparison = build_comparison(
        REVIEW,
        ASPECTS,
        CORE_MODELS,
        _factory(),
        gold={"food": "positive"},
    )
    assert list(comparison.display[GOLD_COLUMN]) == ["Positive", ""]
    assert list(comparison.labels[GOLD_COLUMN]) == ["positive", MISSING_LABEL]


def test_compare_mode_excludes_the_exploratory_models():
    assert "absa_target_gru" not in MODEL_OPTIONS
    assert "absa_text_cnn" not in MODEL_OPTIONS


def test_every_model_has_a_matrix_column_name():
    """A missing entry would silently fall back to the long MODEL_OPTIONS label."""
    assert set(MATRIX_COLUMNS) == set(ALL_MODEL_OPTIONS)


def test_matrix_headers_keep_the_review_only_distinction():
    for model_name in ("absa_tfidf", "absa_target_lstm"):
        assert "review-only" in MATRIX_COLUMNS[model_name]
    for model_name in (
        "absa_atae_lstm",
        "absa_bert_small_fp16",
        "absa_distilbert",
    ):
        assert "aspect" in MATRIX_COLUMNS[model_name]
