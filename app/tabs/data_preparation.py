"""Tab 2 – Data Preparation.

Augment *collected data* directly without requiring camera/upload again.

The app's persistent dataset currently stores **embeddings** (not face
crops/images). Therefore, augmentation is implemented as **embedding-level
jitter**: for each stored embedding we generate multiple synthetic variants
by adding structured noise and L2-normalizing.

This satisfies the “no re-capture” requirement while still increasing the
number and diversity of training samples.
"""
from __future__ import annotations

import numpy as np
import streamlit as st

from app import ui_helpers as ui
from ml import config, storage


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        return vec
    return (vec / norm).astype(np.float32)


def _augment_embedding(emb: np.ndarray, *, sigma: float, rng: np.random.Generator) -> np.ndarray:
    noise = rng.normal(0.0, sigma, size=emb.shape).astype(np.float32)
    return _l2_normalize(emb + noise)


def render(show_hero: bool = True) -> None:
    if show_hero:
        ui.hero("2 · Data Preparation", "Clean, balanced, augmented data beats raw data.")
    else:
        st.markdown("### ✨ Data Preparation & Augmentation")
        st.caption("Clean, balanced, augmented data beats raw data.")

    st.info(
        "Augmentation runs on your already-collected embeddings (no camera/upload needed)."
    )

    db = storage.load_embeddings_db()
    users = storage.load_users()

    if not db:
        st.warning("No embeddings found yet. Collect data in Tab 1 first.")
        ui.lesson("Better data creates better models.")
        return

    available_users = [uid for uid in db.keys()]

    target_all = st.checkbox("Augment all users", value=True)
    target_user = None

    if not target_all:
        if users:
            target_user = st.selectbox(
                "Augment samples for user",
                options=available_users,
                format_func=lambda uid: users.get(uid, {}).get("name", uid),
            )
        else:
            target_user = st.selectbox("Augment samples for user", options=available_users)

    sigma = st.slider(
        "Embedding noise strength (higher = harder augmentation)",
        0.005,
        0.25,
        0.05,
        0.005,
    )

    per_embedding = st.slider(
        "Synthetic variants per stored embedding",
        1,
        12,
        int(config.AUGMENTATIONS_PER_IMAGE),
        1,
    )

    seed = st.number_input("Random seed", min_value=0, step=1, value=7)

    if st.button("➕ Add augmented embeddings to dataset", type="primary"):
        rng = np.random.default_rng(int(seed))

        total_added = 0
        total_source = 0

        def process_user(uid: str) -> None:
            nonlocal total_added, total_source
            src = db.get(uid, [])
            total_source += len(src)
            if not src:
                return
            new_embs: list[np.ndarray] = []
            for emb in src:
                for _ in range(int(per_embedding)):
                    new_embs.append(_augment_embedding(emb, sigma=sigma, rng=rng))
            if new_embs:
                storage.add_embeddings(uid, new_embs)
                total_added += len(new_embs)

        if target_all:
            for uid in available_users:
                process_user(uid)
            st.success(f"Added {total_added} augmented samples across all users ✓")
        else:
            assert target_user is not None
            process_user(target_user)
            name = users.get(target_user, {}).get("name", target_user)
            st.success(f"Added {total_added} augmented samples to {name} ✓")

        st.caption(f"Source embeddings: {total_source} → Added: {total_added}")

    ui.lesson("Better data creates better models.")

