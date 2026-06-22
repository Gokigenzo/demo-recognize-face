"""Tab 1 · Data Processing.

Combines Data Collection, Data Preparation, and Feature Extraction into one unified tab.
"""
from __future__ import annotations

import streamlit as st

from app import ui_helpers as ui
from app.tabs import data_collection, data_preparation, feature_extraction


def render() -> None:
    ui.hero("1 · Data Processing", "Gather, prepare, and embed your data to train models.")

    tab_collect, tab_prep, tab_features = st.tabs([
        "📸 Data Collection",
        "✨ Data Preparation & Augmentation",
        "🧬 Feature Extraction & Clusters",
    ])

    with tab_collect:
        data_collection.render(show_hero=False)

    with tab_prep:
        data_preparation.render(show_hero=False)

    with tab_features:
        feature_extraction.render(show_hero=False)
