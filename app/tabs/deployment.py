"""Tab 5 – Deployment.

The "money" tab: run the full inference pipeline on a live capture and mark
attendance. Uses OpenCV VideoCapture for server-side webcam grabs, with a
``st.camera_input`` path as a browser-friendly alternative.

    Camera → Detection → Embedding → Similarity Search → Attendance
"""
from __future__ import annotations

import cv2
import streamlit as st

from app import ui_helpers as ui
from ml import config, storage
from ml.attendance_engine import identify, mark_attendance
from ml.embedder import embed_face
from ml.face_detector import detect_largest_face, draw_detection




def _capture_opencv(index: int = 0):
    """Grab a single frame from the server-side webcam (OpenCV)."""
    cap = cv2.VideoCapture(index)
    try:
        if not cap.isOpened():
            return None
        # Warm up a few frames so exposure settles.
        frame = None
        for _ in range(5):
            ok, frame = cap.read()
            if not ok:
                frame = None
        return frame
    finally:
        cap.release()


def _run_pipeline(image, threshold: float) -> None:
    face = detect_largest_face(image)
    if face is None:
        st.warning("No face detected in the frame.")
        return

    emb = embed_face(face)
    
    classifier_bundle = storage.load_classifier()
    if classifier_bundle is not None:
        from ml.attendance_engine import identify_with_classifier
        result = identify_with_classifier(emb, classifier_bundle, threshold=threshold)
    else:
        result = identify(emb, threshold=threshold)

    already_attended = False
    if result.is_known and result.user_id is not None:
        attendance_records = storage.load_attendance()
        already_attended = any(r["user_id"] == result.user_id for r in attendance_records)

    label = f"{result.name} ({result.confidence:.2f})"
    annotated = draw_detection(image, face, label)

    c1, c2 = st.columns([2, 1])
    with c1:
        st.image(ui.bgr_to_rgb(annotated), width="stretch")
    with c2:
        kind = "ok" if result.is_known else "bad"
        st.markdown(ui.pill("PRESENT" if result.is_known else "UNKNOWN", kind), unsafe_allow_html=True)
        st.metric("Identity", result.name)
        st.metric("Confidence", f"{result.confidence:.2f}")
        if result.candidates:
            st.markdown("**Top candidates**")
            st.dataframe(
                [{"Name": n, "Score": f"{s:.2f}"} for _id, n, s in result.candidates],
                width="stretch", hide_index=True,
            )
    if result.is_known:
        if already_attended:
            st.info(f"ℹ️ {result.name}, you have taken participation")
        else:
            mark_attendance(result)
            st.success(f"✅ Attendance logged for {result.name}")
    else:
        st.info("🛡️ Marked Unknown — refused to guess below the threshold.")



def render() -> None:
    ui.hero("6 · Deployment", "This is where Machine Learning creates business value.")
    ui.pipeline(["Camera", "Detection", "Embedding", "Similarity Search", "Attendance"])

    if not storage.load_embeddings_db():
        st.info("No enrolled identities yet — register people in Tab 1 first.")
        ui.lesson("This is where ML creates business value.")
        return

    threshold = st.slider(
        "Recognition threshold", 0.10, 0.90, config.DEFAULT_SIMILARITY_THRESHOLD, 0.05,
        key="deploy_thr",
    )

    mode = st.radio("Capture mode", ["Browser camera", "Server webcam (OpenCV)"], horizontal=True)
    image = None
    if mode == "Browser camera":
        shot = st.camera_input("Live capture", key="deploy_cam")
        if shot is not None:
            image = ui.file_to_bgr(shot)
    else:
        if st.button("📷 Grab frame from webcam", type="primary"):
            image = _capture_opencv()
            if image is None:
                st.error("Could not open webcam (index 0). Try 'Browser camera' mode.")

    if image is not None:
        _run_pipeline(image, threshold)

    # Attendance log.
    rows = storage.load_attendance()
    if rows:
        st.markdown("#### 🗒️ Attendance log")
        st.dataframe(rows[::-1], width="stretch", hide_index=True)

    ui.lesson("This is where ML creates business value.")
