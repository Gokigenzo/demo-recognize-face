"""Tab 4 – Evaluation.

Builds a controlled test set from the stored embeddings (simulating Normal,
Glasses, Mask, Side Face, Dark Lighting cases by adding structured noise),
then reports a confusion matrix + accuracy/precision/recall/F1. A threshold
slider drives a live precision-vs-recall tradeoff curve.
"""
from __future__ import annotations

from typing import List

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from app import ui_helpers as ui
from ml import config, storage
from ml.evaluator import EvalSample, evaluate, threshold_sweep, evaluate_classifier

CASES = ["Normal", "Glasses", "Mask", "Side Face", "Dark Lighting"]
# Heavier perturbation == harder case (mask/side face degrade embeddings most).
CASE_NOISE = {
    "Normal": config.NOISE_NORMAL,
    "Glasses": config.NOISE_GLASSES,
    "Mask": config.NOISE_MASK,
    "Side Face": config.NOISE_SIDE_FACE,
    "Dark Lighting": config.NOISE_LOW_LIGHT,
}



def _build_test_set(seed: int = 7) -> List[EvalSample]:
    """Create labelled samples by perturbing stored embeddings + unknowns."""
    rng = np.random.default_rng(seed)
    db = storage.load_embeddings_db()
    samples: List[EvalSample] = []
    for uid, embs in db.items():
        base = np.mean(np.vstack(embs), axis=0)
        for case in CASES:
            noisy = base + rng.normal(0, CASE_NOISE[case], size=base.shape)
            noisy = noisy / (np.linalg.norm(noisy) or 1.0)
            samples.append(EvalSample(noisy.astype(np.float32), uid, case))
    # A few genuine unknowns (random vectors).
    for _ in range(max(3, len(db))):
        v = rng.normal(0, 1, size=config.EMBEDDING_DIM).astype(np.float32)
        v = v / (np.linalg.norm(v) or 1.0)
        samples.append(EvalSample(v, None, "Unknown Person"))
    return samples


def _confusion_heatmap(result) -> None:
    fig = go.Figure(data=go.Heatmap(
        z=result.confusion,
        x=["Correct ID", "Wrong ID", "Unknown"],
        y=result.labels,
        colorscale="Blues",
        text=result.confusion, texttemplate="%{text}",
        showscale=False,
    ))
    fig.update_layout(title="Confusion Matrix", height=320, margin=dict(t=40))
    st.plotly_chart(fig, width="stretch")


def _tradeoff_curve(samples) -> None:
    sweep = threshold_sweep(samples)
    ts = [r.threshold for r in sweep]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts, y=[r.precision for r in sweep], name="Precision", mode="lines+markers"))
    fig.add_trace(go.Scatter(x=ts, y=[r.recall for r in sweep], name="Recall", mode="lines+markers"))
    fig.add_trace(go.Scatter(x=ts, y=[r.f1 for r in sweep], name="F1", mode="lines+markers"))
    fig.update_layout(
        title="Precision / Recall tradeoff vs threshold",
        xaxis_title="Similarity threshold", yaxis_title="Score",
        height=380, yaxis_range=[0, 1.05],
    )
    st.plotly_chart(fig, width="stretch")


def render() -> None:
    ui.hero("5 · Evaluation", "Measure honestly before you trust a model.")

    db = storage.load_embeddings_db()
    if not db:
        st.info("Collect data in Tab 1 first — evaluation needs known identities.")
        ui.lesson("Sometimes predicting Unknown is better than guessing.")
        return

    threshold = st.slider(
        "Decision threshold (cosine similarity)",
        0.10, 0.90, config.DEFAULT_SIMILARITY_THRESHOLD, 0.05,
    )

    samples = _build_test_set()
    result = evaluate(samples, threshold)

    clf_bundle = storage.load_classifier()
    clf_test_acc = evaluate_classifier(samples, clf_bundle, threshold)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy (Similarity)", f"{result.accuracy:.0%}")
    m2.metric("Precision", f"{result.precision:.0%}")
    m3.metric("Recall", f"{result.recall:.0%}")
    m4.metric("F1", f"{result.f1:.0%}")
    if clf_test_acc is not None:
        m5.metric("Accuracy (Classifier)", f"{clf_test_acc:.0%}")
    else:
        m5.metric("Accuracy (Classifier)", "n/a", help="Train a model in Tab 2 first")

    c1, c2 = st.columns(2)
    with c1:
        _confusion_heatmap(result)
    with c2:
        # Removed "Accuracy by case" table per request.
        pass

    _tradeoff_curve(samples)

    ui.lesson("Sometimes predicting Unknown is better than guessing.")

