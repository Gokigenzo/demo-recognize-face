"""Tab 5 – Monitoring & Feedback Loop.

Shows how the system improves after deployment.

The backend already supports:
- logging feedback (ml/storage.py: append_feedback)
- adding corrected embeddings back to the embedding database
  (ml/feedback_engine.py: record_correction)

This tab provides a human-in-the-loop UI:
- inspect the latest live capture outcome (optionally upload an image)
- choose the correct identity (or Unknown)
- store the correction
- show monitoring/feedback stats
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import streamlit as st

from app import ui_helpers as ui
from ml import config, storage
from ml.attendance_engine import identify
from ml.embedder import embed_face
from ml.face_detector import detect_largest_face, draw_detection
from ml.feedback_engine import improvement_summary, record_correction


def _slugify(name: str) -> str:
    return "".join(c for c in name.lower().strip().replace(" ", "_") if c.isalnum() or c == "_")


def _capture_image_for_feedback() -> Optional[np.ndarray]:
    """Return a BGR image captured/uploaded by the user, or None."""
    mode = st.radio(
        "Feedback capture source",
        ["Browser camera", "Upload image"],
        horizontal=True,
    )
    if mode == "Browser camera":
        shot = st.camera_input("Capture a frame to correct", key="fb_cam")
        if shot is None:
            return None
        return ui.file_to_bgr(shot)

    up = st.file_uploader(
        "Upload an image (jpg/png)",
        type=["jpg", "jpeg", "png"],
        key="fb_up",
    )
    if up is None:
        return None
    return ui.file_to_bgr(up)


def render() -> None:
    ui.hero("5 · Monitoring & Feedback", "Live systems learn from mistakes." )

    pass

    ui.pipeline([

        "Deployment",
        "Unknown / Wrong prediction",
        "Human correction",
        "New embedding",
        "Continuous improvement",
    ])

    db = storage.load_embeddings_db()
    if not db:
        st.info("Collect data in Tab 1 first — feedback needs an embedding database.")
        ui.lesson("Monitoring & feedback closes the lifecycle after deployment.")
        return

    thr = st.slider(
        "Recognition threshold used for the prediction preview",
        0.10,
        0.90,
        config.DEFAULT_SIMILARITY_THRESHOLD,
        0.05,
        key="fb_thr",
        help="This threshold determines whether the system outputs Known vs Unknown.",
    )

    st.markdown("---")

    image = _capture_image_for_feedback()
    if image is None:
        ui.lesson("When the system says **Unknown** or guesses the wrong identity, a human corrects it here.")
        return

    face = detect_largest_face(image)
    if face is None:
        st.warning("No face detected. Try a clearer image.")
        return

    emb = embed_face(face)
    result = identify(emb, threshold=thr)

    annotated = draw_detection(image, face, f"{result.name} ({result.confidence:.2f})")

    c1, c2 = st.columns([2, 1])
    with c1:
        st.image(ui.bgr_to_rgb(annotated), width="stretch", caption="Preview (detection + decision)")
    with c2:
        kind = "ok" if result.is_known else "bad"
        st.markdown(ui.pill("PREDICTED: " + result.name, kind), unsafe_allow_html=True)
        st.metric("Confidence", f"{result.confidence:.2f}")
        st.caption("If this is wrong, correct it below and we will store the new embedding.")

    users = storage.load_users()
    user_options = [
        (uid, users.get(uid, {}).get("name", uid))
        for uid in db.keys()
    ]

    st.markdown("### ✅ Human correction (the feedback loop)")

    # If predictions are known, preselect that name; otherwise default to Unknown.
    predicted_user_id = result.user_id
    default_choice = "__unknown__" if predicted_user_id is None else predicted_user_id

    choice_labels = ["__unknown__"] + [uid for uid, _ in user_options]
    choice_names = {
        "__unknown__": "Unknown / Not enrolled",
        **{uid: name for uid, name in user_options},
    }

    correct_user_id = st.selectbox(
        "Correct identity",
        options=choice_labels,
        index=choice_labels.index(default_choice) if default_choice in choice_labels else 0,
        format_func=lambda uid: choice_names.get(uid, uid),
    )

    enroll_new = False
    new_name = ""
    if correct_user_id == "__unknown__":
        enroll_new = st.checkbox("➕ Enroll as a new person", value=True, help="Add this face as a new person in the database")
        if enroll_new:
            new_name = st.text_input("👤 Person Name", placeholder="e.g. Marie Curie")

    note = st.text_area(
        "Optional note (what went wrong / context)",
        placeholder="e.g., new hairstyle, different lighting, wrong threshold…",
    )

    if st.button("💾 Save correction & add embedding", type="primary", width="stretch"):
        if correct_user_id == "__unknown__":
            if enroll_new:
                if not new_name.strip():
                    st.error("Please enter a name for the new person.")
                    return
                new_user_id = _slugify(new_name)
                entry = record_correction(
                    embedding=emb,
                    correct_user_id=new_user_id,
                    correct_name=new_name.strip(),
                    predicted_name=result.name,
                    note=note,
                )
                st.success(f"Stored feedback & enrolled **{new_name.strip()}** with a new embedding! ✓")
                st.balloons()
                st.rerun()
            else:
                st.info("Marked as Unknown — no embedding added to a specific identity.")
                storage.append_feedback(
                    {
                        "timestamp": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
                        "predicted": result.name,
                        "corrected_to": "Unknown",
                        "correct_user_id": None,
                        "note": note,
                    }
                )
                st.success("Saved feedback entry.")
                st.rerun()

        correct_name = users.get(correct_user_id, {}).get("name", correct_user_id)
        entry = record_correction(
            embedding=emb,
            correct_user_id=correct_user_id,
            correct_name=correct_name,
            predicted_name=result.name,
            note=note,
        )
        st.success(f"Saved correction for {correct_name}. Embedding added to the database. ✓")
        st.balloons()
        st.rerun()

    st.markdown("---")
    st.markdown("### 📊 Monitoring snapshot")
    summary = improvement_summary()

    m1, m2, m3 = st.columns(3)
    m1.metric("Total corrections", summary["total_corrections"])
    m2.metric("Users with data", summary["users_with_data"])
    m3.metric("Total embeddings", summary["total_embeddings"])

    recent = summary.get("recent") or []
    if recent:
        st.markdown("#### Recent feedback")
        st.dataframe(recent, width="stretch", hide_index=True)
    else:
        st.caption("No feedback recorded yet. Save one correction to see entries here.")

    ui.lesson("Monitoring & feedback closes the lifecycle: errors become new training data, improving the next prediction.")

