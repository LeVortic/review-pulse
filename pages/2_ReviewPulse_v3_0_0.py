"""DLE602 aspect-based sentiment application."""

import streamlit as st

from src.absa.inference.api import predict_aspects
from src.absa.inference.comparison import build_comparison
from src.absa.inference.predictors import (
    ALL_MODEL_OPTIONS,
    COMPARISON_MODEL_OPTIONS,
    exposes_token_evidence,
)
from src.absa.interpretability.evidence import unsupported_evidence
from src.absa.samples import find_sample, get_random_sample
from src.app.absa_results import first_evidence, render_result_grid, style_comparison
from src.app.absa_service import load_aspect_predictor

SINGLE_MODEL = "Single model"
COMPARE_MODELS = "Compare models"
RQ1_HOOK = (
    "Review-only models repeat one prediction across aspects - this is the "
    "sentence-level limitation the aspect-conditioned models address. "
    "Switch to single-model mode for token evidence."
)

st.title("ReviewPulse v3.0.0")
st.caption("DLE602 · three-class aspect-based sentiment analysis")
st.info(
    "Enter one or more aspects manually. Review-only models intentionally receive "
    "only the review; ATAE-LSTM and DistilBERT receive the review-and-aspect pair. "
    "GRU and Text CNN are exploratory extensions and do not support token evidence."
)

mode = st.radio("Mode", [SINGLE_MODEL, COMPARE_MODELS], horizontal=True)
comparing = mode == COMPARE_MODELS

if comparing:
    model_name = None
    model_exposes_evidence = False
    st.caption(
        "Comparing the four core models plus the 55 MB BERT-Small FP16 experiment. "
        "GRU and Text CNN are not included."
    )
else:
    model_name = st.selectbox(
        "Model",
        list(ALL_MODEL_OPTIONS),
        format_func=ALL_MODEL_OPTIONS.__getitem__,
    )
    # The review-only limitation describes the selected model, not any single
    # aspect, so it renders once here instead of under every result card.
    model_exposes_evidence = exposes_token_evidence(model_name)
    if not model_exposes_evidence:
        st.caption(unsupported_evidence(ALL_MODEL_OPTIONS[model_name])["limitations"])


def load_sample() -> None:
    """Fill both inputs with a curated multi-aspect demonstration review."""
    sample = get_random_sample(st.session_state.get("absa_review", ""))
    st.session_state["absa_review"] = sample.review
    st.session_state["absa_aspects"] = sample.aspects


st.button("💡 Generate sample", on_click=load_sample)
review = st.text_area(
    "Review",
    placeholder="The food was great but the service was slow.",
    height=150,
    key="absa_review",
)
aspects = st.text_input(
    "Aspects, separated by commas",
    placeholder="food, service",
    key="absa_aspects",
)

# Gold polarity belongs to the exact annotated sentence, so it is looked up from
# the live text: editing the review drops the label instead of leaving a stale one.
sample = find_sample(review)
if sample is not None:
    st.caption(
        f"SemEval test sentence `{sample.sentence_id}` · {sample.scenario}. "
        "Gold polarity is shown with each prediction."
    )

if st.button(
    "Compare models" if comparing else "Classify aspects",
    type="primary",
    disabled=not review.strip() or not aspects.strip(),
):
    gold = sample.gold if sample is not None else None
    try:
        if comparing:
            comparison = build_comparison(
                review,
                aspects.split(","),
                list(COMPARISON_MODEL_OPTIONS),
                load_aspect_predictor,
                gold=gold,
            )
        else:
            predictor = load_aspect_predictor(model_name)
            results = predict_aspects(review, aspects.split(","), model_name, predictor)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        # build_comparison reports per-column artifact failures as sentinels, so
        # compare mode only reaches here through input validation and there is no
        # single selected model to blame.
        if comparing:
            st.error(f"Comparison request is invalid: {error}")
        else:
            st.error(f"The selected model is unavailable: {error}")
    else:
        if comparing:
            st.dataframe(style_comparison(comparison), width="stretch")
            st.caption(RQ1_HOOK)
        else:
            render_result_grid(results, review, model_exposes_evidence, gold)
            evidence = first_evidence(results) if model_exposes_evidence else None
            if evidence is not None:
                st.caption(
                    f"**{evidence['method']}** · darker shading indicates a larger "
                    "score within this aspect view."
                )
                st.caption(evidence["caveat"])
                st.caption(f"Limitation: {evidence['limitations']}")
