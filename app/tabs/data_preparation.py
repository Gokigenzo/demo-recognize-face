"""Tab 2 – Data Preparation.

Take the collected faces and *augment* them (brightness, contrast, blur,
flip) to enrich the dataset. Shows a before/after gallery and the headline
"10 images -> 50 images" effect. Optionally folds augmented embeddings back
into the database to strengthen recognition.
"""
from __future__ import annotations

import numpy as np
import streamlit as st

from app import ui_helpers as ui
from ml import storage
from ml.augmenter import albumentations_available, augment_image
from ml.embedder import embed_crop
from ml.face_detector import detect_largest_face


def render(show_hero: bool = True) -> None:
    if show_hero:
        ui.hero("2 · Data Preparation", "Clean, balanced, augmented data beats raw data.")
    else:
        st.markdown("### ✨ Data Preparation & Augmentation")
        st.caption("Clean, balanced, augmented data beats raw data.")

    backend = "Albumentations" if albumentations_available() else "OpenCV (fallback)"
    st.caption(f"Augmentation backend: **{backend}**")

    st.write("Upload a face image to see how augmentation multiplies your dataset:")
    up = st.file_uploader("Base image", type=["jpg", "jpeg", "png"], key="prep_up")
    cam = st.camera_input("…or use the camera", key="prep_cam")

    src = up or cam
    if src is None:
        col1, col2, col3 = st.columns(3)
        col1.markdown("<div class='tm-card'><div class='tm-metric'>10</div>raw images</div>", unsafe_allow_html=True)
        col2.markdown("<div class='tm-card' style='text-align:center'><span class='tm-arrow'>→</span></div>", unsafe_allow_html=True)
        col3.markdown("<div class='tm-card'><div class='tm-metric'>50</div>after augmentation</div>", unsafe_allow_html=True)
        ui.lesson("Better data creates better models.")
        return

    image = ui.file_to_bgr(src)
    face = detect_largest_face(image)
    base_crop = face.crop if face is not None else image

    st.markdown("#### Before → After gallery")
    augmentations = augment_image(base_crop)

    cols = st.columns(len(augmentations) + 1)
    with cols[0]:
        st.markdown("**Original**")
        st.image(ui.bgr_to_rgb(base_crop), use_container_width=True)
    for col, (name, img) in zip(cols[1:], augmentations):
        with col:
            st.markdown(f"**{name}**")
            st.image(ui.bgr_to_rgb(img), use_container_width=True)

    multiplier = len(augmentations) + 1
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"<div class='tm-card'><div class='tm-metric'>1</div>original</div>", unsafe_allow_html=True)
    c2.markdown("<div class='tm-card' style='text-align:center;font-size:30px'>→</div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='tm-card'><div class='tm-metric'>{multiplier}</div>samples</div>", unsafe_allow_html=True)

    users = storage.load_users()
    if users:
        target = st.selectbox(
            "Add augmented samples to user",
            options=list(users.keys()),
            format_func=lambda uid: users[uid]["name"],
        )
        if st.button("➕ Add augmented embeddings to dataset", type="primary"):
            embeddings = []
            for _name, img in augmentations:
                emb = embed_crop(img)
                if emb is not None:
                    embeddings.append(np.asarray(emb))
            if embeddings:
                storage.add_embeddings(target, embeddings)
                st.success(f"Added {len(embeddings)} augmented samples to {users[target]['name']} ✓")
    else:
        st.info("Register a user in Tab 1 to attach augmented samples.")

    ui.lesson("Better data creates better models.")
