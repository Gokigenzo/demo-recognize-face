"""Tab 1 – Data Collection.

Register a user and capture face poses. For each capture we show the full
pipeline: Original Image -> Detected Face -> Cropped Face, then store the
embedding so later tabs have data to work with.
"""
from __future__ import annotations

import streamlit as st

from app import ui_helpers as ui
from ml import config, storage
from ml.embedder import embed_face
from ml.face_detector import backend_name, detect_largest_face, draw_detection


def _slugify(name: str) -> str:
    return "".join(c for c in name.lower().strip().replace(" ", "_") if c.isalnum() or c == "_")


def render(show_hero: bool = True) -> None:
    if show_hero:
        ui.hero("1 · Data Collection", "Machine Learning starts by gathering examples.")
        ui.pipeline(["Original Image", "Detected Face", "Cropped Face", "Stored Embedding"])
    else:
        st.markdown("### 📸 Data Collection")
        st.caption("Machine Learning starts by gathering examples.")
        ui.pipeline(["Original Image", "Detected Face", "Cropped Face", "Stored Embedding"])

    st.caption(f"Detection backend: **{backend_name()}**")

    col_a, col_b = st.columns([1, 1])
    with col_a:
        name = st.text_input("👤 Person name", placeholder="e.g. Ada Lovelace")
    with col_b:
        pose = st.selectbox("📐 Pose", config.CAPTURE_POSES)

    st.write("Use your webcam or upload a photo for this pose:")
    tab_cam, tab_up = st.tabs(["📸 Camera", "⬆️ Upload"])
    image = None
    with tab_cam:
        shot = st.camera_input(f"Capture '{pose}' pose", key=f"cam_{pose}")
        if shot is not None:
            image = ui.file_to_bgr(shot)
    with tab_up:
        up = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"], key=f"up_{pose}")
        if up is not None:
            image = ui.file_to_bgr(up)

    if image is None:
        ui.lesson("Machine Learning learns from data.")
        return

    face = detect_largest_face(image)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Original**")
        st.image(ui.bgr_to_rgb(image), use_container_width=True)
    if face is None:
        st.warning("😕 No face detected. Try better lighting or face the camera.")
        ui.lesson("Machine Learning learns from data.")
        return
    with c2:
        st.markdown("**Detected**")
        st.image(ui.bgr_to_rgb(draw_detection(image, face, "face")), use_container_width=True)
    with c3:
        st.markdown("**Cropped**")
        st.image(ui.bgr_to_rgb(face.crop), use_container_width=True)

    st.markdown(
        ui.pill(f"Detection confidence: {face.confidence:.2f}", "ok"),
        unsafe_allow_html=True,
    )

    if st.button("💾 Register this pose", type="primary", disabled=not name.strip()):
        user_id = _slugify(name)
        embedding = embed_face(face)
        storage.upsert_user(user_id, name.strip())
        storage.add_embeddings(user_id, [embedding])
        st.success(f"Stored **{pose}** pose for **{name}** ✓")
        st.balloons()

    # Show current dataset snapshot.
    db = storage.load_embeddings_db()
    users = storage.load_users()
    if db:
        st.markdown("#### 📚 Current dataset")
        rows = [
            {"User": users.get(uid, {}).get("name", uid), "Samples": len(embs)}
            for uid, embs in db.items()
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)

    ui.lesson("Machine Learning learns from data.")
