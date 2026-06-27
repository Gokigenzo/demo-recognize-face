"""Tab 1 · Data Processing.

Combines Data Collection, Data Preparation, and Feature Extraction into one unified tab.
"""
from __future__ import annotations

import streamlit as st

from app import ui_helpers as ui
from app.tabs import data_collection, data_preparation
from app.tabs.collected_data import render_dataset_table
from app.tabs.data_processing_bg import ensure_background_processing


def render() -> None:
    ui.hero("1 · Data Processing", "Gather, prepare, and embed your data to train models.")


    tab_collect, tab_prep = st.tabs([
        "📸 Data Collection",
        "✨ Data Preparation & Augmentation",
    ])

    with tab_collect:
        data_collection.render(show_hero=False)
        _msg = ensure_background_processing()
        if _msg:
            st.toast(_msg, icon="⚙️")
        render_dataset_table()


    with tab_prep:
        data_preparation.render(show_hero=False)


