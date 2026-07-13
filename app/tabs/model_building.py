"""Tab 4 – Model Building.

Trains a *real* scikit-learn classifier (SVM or KNN) on the collected face
embeddings and visualizes the decision boundary it learns over a 2-D PCA
projection of the data. This is the moment the demo goes from "storing
vectors" to "a model that learns to separate identities".
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from app import ui_helpers as ui
from ml import storage
from ml.model_builder import (
    can_train,
    decision_boundary_mesh,
    train,
)

PALETTE = ["#4285F4", "#00BFA5", "#F4B400", "#EA4C89", "#9334E6", "#1E8E3E", "#E8710A"]


def _boundary_plot(model) -> None:
    """Filled decision regions + scatter of the training points."""
    xx, yy, zz = decision_boundary_mesh(model)
    if xx.size == 0:
        st.info("Not enough data to draw a decision boundary yet.")
        return

    n_classes = max(1, len(model.class_names))
    # Discrete colorscale so each class region gets a distinct pastel fill.
    colorscale = []
    for i in range(n_classes):
        c = PALETTE[i % len(PALETTE)]
        lo = i / n_classes
        hi = (i + 1) / n_classes
        colorscale.append([lo, c])
        colorscale.append([hi, c])

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            x=xx[0],
            y=yy[:, 0],
            z=zz,
            colorscale=colorscale,
            showscale=False,
            opacity=0.30,
            zmin=0,
            zmax=max(1, n_classes - 1),
            hoverinfo="skip",
        )
    )

    coords = model.coords_2d
    for idx, name in enumerate(model.class_names):
        mask = model.labels_2d == idx
        fig.add_trace(
            go.Scatter(
                x=coords[mask, 0],
                y=coords[mask, 1],
                mode="markers",
                name=name,
                marker=dict(
                    size=13,
                    color=PALETTE[idx % len(PALETTE)],
                    line=dict(width=1.5, color="white"),
                ),
            )
        )

    fig.update_layout(
        title=f"{model.kind} decision boundary (2-D PCA projection)",
        xaxis_title="PC1",
        yaxis_title="PC2",
        height=500,
        legend_title="Identity",
        margin=dict(t=50),
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Shaded regions are what the classifier predicts for any point in space; "
        "dots are the real training samples. The borders between colors are the "
        "**decision boundaries** the model learned."
    )


def render() -> None:
    ui.hero("4 · Model Building", "The model learns to draw boundaries between identities.")
    ui.pipeline(["Embeddings", "Train Classifier", "Decision Boundary", "Predict Identity"])

    ok, message = can_train()
    if not ok:
        st.info(message)
        ui.lesson("A model is only as good as the data you give it to learn from.")
        return

    c1, c2 = st.columns([1, 1])
    with c1:
        kind = st.radio(
            "Classifier",
            ["SVM", "KNN", "MLP"],
            horizontal=True,
            help="SVM finds smooth margins; KNN votes among neighbors; MLP is an iterative Neural Network.",
        )
    with c2:
        st.write("")
        train_clicked = st.button("🚀 Train model", width="stretch", type="primary")

    if train_clicked:
        # Placeholders for dynamic status & charts
        status_placeholder = st.empty()
        chart_placeholder = st.empty()

        live_epochs = []
        live_loss = []
        live_train_acc = []
        live_val_acc = []

        def training_callback(epoch, total_epochs, loss, train_acc, val_acc):
            live_epochs.append(epoch)
            live_loss.append(loss)
            live_train_acc.append(train_acc)
            live_val_acc.append(val_acc)

            status_placeholder.markdown(
                f"**Training progress:** Epoch {epoch}/{total_epochs} — "
                f"Loss: `{loss:.4f}` | Train Acc: `{train_acc:.0%}` | Val Acc: `{val_acc:.0%}`"
            )

            # Only MLP should display learning curves.
            if kind != "MLP":
                return

            from plotly.subplots import make_subplots

            fig = make_subplots(
                rows=1,
                cols=2,
                subplot_titles=("Training Loss Curve", "Accuracy Progression"),
                horizontal_spacing=0.15,
            )
            fig.add_trace(
                go.Scatter(
                    x=live_epochs,
                    y=live_loss,
                    mode="lines+markers",
                    name="Loss",
                    line=dict(color="#EA4C89", width=2),
                    marker=dict(size=4),
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=live_epochs,
                    y=live_train_acc,
                    mode="lines+markers",
                    name="Train Acc",
                    line=dict(color="#4285F4", width=2),
                    marker=dict(size=4),
                ),
                row=1,
                col=2,
            )
            fig.add_trace(
                go.Scatter(
                    x=live_epochs,
                    y=live_val_acc,
                    mode="lines+markers",
                    name="Val Acc",
                    line=dict(color="#00BFA5", width=2, dash="dash"),
                    marker=dict(size=4),
                ),
                row=1,
                col=2,
            )
            fig.update_layout(
                height=320,
                margin=dict(t=50, b=30, l=10, r=10),
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1
                ),
            )
            fig.update_xaxes(title_text="Epoch", row=1, col=1)
            fig.update_xaxes(title_text="Epoch", row=1, col=2)
            fig.update_yaxes(title_text="Loss", row=1, col=1)
            fig.update_yaxes(
                title_text="Accuracy", row=1, col=2, range=[0, 1.05]
            )

            chart_placeholder.plotly_chart(fig, width="stretch")
            import time

            time.sleep(0.04)


        with st.spinner(f"Training {kind} on collected embeddings…"):
            try:
                model = train(kind, persist=True, callback=training_callback)
                st.session_state["mb_model"] = model
                status_placeholder.empty()
                chart_placeholder.empty()
                st.success(f"Trained {kind} on {model.n_samples} samples.")
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
                return

    model = st.session_state.get("mb_model")
    if model is None:
        bundle = storage.load_classifier()
        if bundle is not None:
            # Reconstruct the learning curves if they were saved in the bundle
            st.caption(
                f"A previously trained **{bundle['kind']}** model is saved. "
                "Click *Train model* to retrain and visualize its boundary."
            )
        else:
            st.caption("Choose a classifier and click *Train model* to begin.")
        return

    m1, m2, m3 = st.columns(3)
    m1.metric("Train accuracy", f"{model.train_accuracy:.0%}")
    cv = model.cv_accuracy
    m2.metric("Cross-val accuracy", "n/a" if np.isnan(cv) else f"{cv:.0%}")
    m3.metric("Classes", len(model.class_names))

    if not np.isnan(cv) and (model.train_accuracy - cv) > 0.25:
        st.warning(
            "Train accuracy is much higher than cross-validated accuracy — "
            "a classic sign of **overfitting** to too little data."
        )

    # Plot training curves ONLY for MLP (SVM/KNN are non-iterative in this demo).
    if model.kind == "MLP" and model.epochs_loss:

        st.markdown("### 📈 Iterative Training Dynamics (Learning Curves)")
        from plotly.subplots import make_subplots
        epochs = list(range(1, len(model.epochs_loss) + 1))
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Training Loss Curve", "Accuracy Progression"),
            horizontal_spacing=0.15
        )
        fig.add_trace(
            go.Scatter(
                x=epochs, y=model.epochs_loss,
                mode="lines+markers", name="Loss",
                line=dict(color="#EA4C89", width=2),
                marker=dict(size=4)
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=epochs, y=model.epochs_train_acc,
                mode="lines+markers", name="Train Acc",
                line=dict(color="#4285F4", width=2),
                marker=dict(size=4)
            ),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(
                x=epochs, y=model.epochs_val_acc,
                mode="lines+markers", name="Val Acc",
                line=dict(color="#00BFA5", width=2, dash="dash"),
                marker=dict(size=4)
            ),
            row=1, col=2
        )
        fig.update_layout(
            height=320,
            margin=dict(t=50, b=30, l=10, r=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1),
        )
        fig.update_xaxes(title_text="Epoch", row=1, col=1)
        fig.update_xaxes(title_text="Epoch", row=1, col=2)
        fig.update_yaxes(title_text="Loss", row=1, col=1)
        fig.update_yaxes(title_text="Accuracy", row=1, col=2, range=[0, 1.05])
        st.plotly_chart(fig, width="stretch")
        st.info(
            "💡 **Learning Curve Insights:**\n"
            "- **Loss Curve:** Shows how the network minimizes errors via backpropagation. Ideally, it decreases smoothly.\n"
            "- **Accuracy Gap:** If Train Accuracy reaches 100% while Validation Accuracy stalls or drops, the model is **overfitting** (memorizing instead of generalizing)."
        )

    st.markdown("### 🗺️ Classifier Decision Space")
    _boundary_plot(model)

    ui.lesson("The model learns to draw boundaries between identities.")
