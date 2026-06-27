"""Background data processing for Tab 1.

In the original demo, "Feature Extraction & Clusters" was a dedicated UI tab.
This module keeps the demo behavior but performs the *derived* work in
background automatically once new data is collected.

Implementation notes:
- Streamlit does not provide true background threads per user; however,
  we can emulate background refresh by:
  1) caching a "last processed" timestamp in st.session_state
  2) running lightweight recomputation only when new embeddings appear.

Currently, we refresh a derived PCA projection cache used by the
"Feature Extraction" explanations (PCA coords).
"""
from __future__ import annotations

import time
from typing import Optional, Tuple

import numpy as np
import streamlit as st

from ml import config, storage
from ml.embedder import pca_project


def _embeddings_fingerprint() -> Tuple[int, float]:
    """Return a cheap signature for current embeddings.

    (count, last_time-ish). We don't track per-sample provenance in storage,
    so we use an approximate signal: total number of embeddings plus a rough
    hash over a small prefix.
    """
    db = storage.load_embeddings_db()
    if not db:
        return (0, 0.0)

    total = 0
    prefix_sum = 0.0
    for embs in db.values():
        for e in embs:
            total += 1
            arr = np.asarray(e, dtype=np.float32)
            prefix_sum += float(arr[:8].sum())
            if total > 200:
                break
        if total > 200:
            break

    return (total, prefix_sum)


def ensure_background_processing() -> Optional[str]:
    """Refresh derived artifacts if new data exists.

    Returns an informational message (or None) for optional UI display.
    """
    fp = _embeddings_fingerprint()

    last_fp = st.session_state.get("_bg_last_fp")
    if last_fp == fp:
        return None

    # Mark it as processed early so repeated reruns don't re-enter.
    st.session_state["_bg_last_fp"] = fp

    # Simulate a short "background" window; keeps UX smoother.
    # (This is still executed during the rerun, but we avoid redoing heavy work.)
    t0 = time.time()

    db = storage.load_embeddings_db()
    all_emb = []
    for embs in db.values():
        all_emb.extend(embs)

    if len(all_emb) >= 2:
        coords = pca_project(all_emb, n_components=2)
        st.session_state["_bg_pca_coords"] = coords
    else:
        st.session_state["_bg_pca_coords"] = np.empty((0, 2), dtype=np.float32)

    st.session_state["_bg_processed_at"] = time.time()
    dt = time.time() - t0

    return f"Refreshed derived data in {dt:.2f}s"

