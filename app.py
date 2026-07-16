"""ML Lifecycle Demo – Streamlit entrypoint.

A Google-Teachable-Machine-inspired, tab-driven walkthrough of the full
machine-learning lifecycle using a face-recognition attendance demo.

Run:
    streamlit run main.py
"""
from __future__ import annotations

import os

# Restrict multi-threading to 1 thread for stable execution in resource-constrained environments
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["ONNXRUNTIME_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import streamlit as st

from app import ui_helpers as ui
from app.tabs import (
    data_processing,
    model_building,
    evaluation,
    deployment,
    monitoring_feedback,
)

from ml import config, storage

st.set_page_config(
    page_title="ML Lifecycle Demo",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

config.ensure_dirs()
ui.inject_theme()

TABS = {
    "1 · Data Processing": data_processing.render,
    "2 · Model Building": model_building.render,
    "3 · Evaluation": evaluation.render,
    "4 · Deployment": deployment.render,
    "5 · Monitoring & Feedback": monitoring_feedback.render,
}



def sidebar() -> str:
    with st.sidebar:
        st.markdown("## 🧠 ML Lifecycle Demo")
        st.caption("Teach the full ML lifecycle in ~10 minutes.")
        choice = st.radio("Lifecycle stage", list(TABS.keys()), label_visibility="collapsed")

        st.divider()
        db = storage.load_embeddings_db()
        st.metric("Enrolled people", len(db))
        st.metric("Total samples", sum(len(v) for v in db.values()))

        st.divider()
        st.markdown("**💾 Demo dataset**")
        st.caption("Save a pre-enrolled dataset and reload it offline.")
        st.download_button(
            "⬇️ Export dataset",
            data=storage.export_bundle(),
            file_name="ml_demo_bundle.pkl",
            mime="application/octet-stream",
            width="stretch",
            help="Download users, embeddings, and the trained model as one file.",
        )
        uploaded = st.file_uploader(
            "⬆️ Import dataset",
            type=["pkl"],
            help="Load a previously exported demo bundle.",
            key="import_bundle",
        )
        if uploaded is not None and not st.session_state.get("_imported_bundle"):
            try:
                summary = storage.import_bundle(uploaded.getvalue(), replace=True)
                st.session_state["_imported_bundle"] = True
                clf = "with model" if summary["has_classifier"] else "no model"
                st.success(
                    f"Imported {summary['users']} people, "
                    f"{summary['embeddings']} samples ({clf}). Reload to refresh."
                )
            except ValueError as exc:
                st.error(str(exc))
        if uploaded is None and st.session_state.get("_imported_bundle"):
            # Allow a fresh import once the previous file is cleared.
            st.session_state["_imported_bundle"] = False

        st.divider()
        if st.button("🚀 Load Sample Dataset", width="stretch", help="Load a pre-populated dataset of historical scientists (Ada Lovelace, Alan Turing, Grace Hopper) with a trained SVM classifier"):
            try:
                import os
                sample_path = os.path.join(config.DATASETS_DIR, "sample_bundle.pkl")
                if not os.path.exists(sample_path):
                    storage.generate_historical_scientists_bundle(sample_path)
                with open(sample_path, "rb") as f:
                    summary = storage.import_bundle(f.read(), replace=True)
                st.session_state["_imported_bundle"] = True
                st.success(f"Loaded {summary['users']} historical scientists! Page will reload.")
                st.rerun()
            except Exception as exc:
                st.error(f"Failed to load sample dataset: {exc}")

        if st.button("🔄 Reset demo data", width="stretch"):
            storage.reset_all()
            st.session_state["_imported_bundle"] = False
            st.success("Demo reset. Reload to start fresh.")
        st.caption("Built with Streamlit · OpenCV · InsightFace · scikit-learn")
    return choice


def main() -> None:
    choice = sidebar()
    TABS[choice]()


if __name__ == "__main__":
    main()
