"""Collected data overview helpers (Tab 1).

This module builds a Streamlit table that shows the current dataset snapshot
(users and how many embedding samples exist per user).
"""
from __future__ import annotations

import streamlit as st

from ml import storage


def _pose_count_hint(db) -> int:
    total = 0
    for embs in db.values():
        total += len(embs)
    return total


def render_dataset_table() -> None:
    """Render a table summarizing all collected data in the embedding DB."""
    db = storage.load_embeddings_db()
    users = storage.load_users()

    st.markdown("#### 📚 Collected dataset")


    if not db:
        st.info("No data collected yet. Register a pose in the Data Collection sub-tab.")
        return

    rows = []
    for uid, embs in db.items():
        rows.append(
            {
                "User": users.get(uid, {}).get("name", uid),
                "User ID": uid,
                "Samples": len(embs),
            }
        )

    st.dataframe(rows, use_container_width=True, hide_index=True)

