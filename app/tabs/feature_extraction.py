"""Tab 3 – Feature Extraction.

Explain that the model stores *features* (a 512-D embedding) rather than raw
pixels. Visualizes stored embeddings projected to 2-D via PCA so audiences
can literally see identity clusters form.
"""
from __future__ import annotations

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app import ui_helpers as ui
from ml import config, storage
from ml.embedder import embed_face, pca_project
from ml.face_detector import detect_largest_face


def _live_embedding_demo() -> None:
    st.markdown("#### Turn a face into a vector")
    src = st.camera_input("Capture a face", key="feat_cam") or st.file_uploader(
        "…or upload", type=["jpg", "jpeg", "png"], key="feat_up"
    )
    if src is None:
        return
    image = ui.file_to_bgr(src)
    face = detect_largest_face(image)
    if face is None:
        st.warning("No face detected.")
        return
    emb = embed_face(face)
    c1, c2 = st.columns([1, 2])
    with c1:
        st.image(ui.bgr_to_rgb(face.crop), caption="Face", width="stretch")
    with c2:
        st.markdown(f"**{config.EMBEDDING_DIM}-D Identity Vector** (first 16 of {config.EMBEDDING_DIM})")
        preview = np.asarray(emb[:16]).reshape(1, -1)
        fig = px.imshow(preview, color_continuous_scale="Blues", aspect="auto")
        fig.update_layout(height=120, margin=dict(l=0, r=0, t=10, b=0), coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch")


def _cluster_plot() -> None:
    db = storage.load_embeddings_db()
    users = storage.load_users()
    all_emb, labels = [], []
    for uid, embs in db.items():
        for e in embs:
            all_emb.append(e)
            labels.append(users.get(uid, {}).get("name", uid))

    if len(all_emb) < 2:
        st.info("Collect at least 2 samples (Tab 1) to visualize embedding clusters.")
        return

    coords = pca_project(all_emb, n_components=2)
    fig = go.Figure()
    for name in sorted(set(labels)):
        idx = [i for i, l in enumerate(labels) if l == name]
        fig.add_trace(go.Scatter(
            x=coords[idx, 0], y=coords[idx, 1], mode="markers",
            name=name, marker=dict(size=12, line=dict(width=1, color="white")),
        ))
    fig.update_layout(
        title="Embeddings projected to 2-D (PCA)",
        xaxis_title="PC1", yaxis_title="PC2", height=460,
        legend_title="Identity",
    )
    st.plotly_chart(fig, width="stretch")
    st.caption("Tight, well-separated clusters → easier, more reliable recognition.")


def render(show_hero: bool = True) -> None:
    if show_hero:
        ui.hero("3 · Feature Extraction", "The model remembers patterns, not photographs.")
        ui.pipeline(["Face", f"{config.EMBEDDING_DIM}-D Embedding", "Identity Vector", "Cluster"])
    else:
        st.markdown("### 🧬 Feature Extraction & Clusters")
        st.caption("The model remembers patterns, not photographs.")
        ui.pipeline(["Face", f"{config.EMBEDDING_DIM}-D Embedding", "Identity Vector", "Cluster"])

    _live_embedding_demo()
    st.divider()
    _cluster_plot()

    ui.lesson("The model remembers features, not images.")
