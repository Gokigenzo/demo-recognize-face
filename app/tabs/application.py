"""Application page for the realtime attendance system.

This page demonstrates how a trained face recognition model is used in
production: live webcam inference, checklist-driven attendance tracking,
confidence-based identity confirmation, and export of final attendance data.
"""
from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional

import streamlit as st
import os
import uuid

from app import ui_helpers as ui
from ml import config, storage
from ml.attendance_session import AttendanceSession, SessionStatistics, PredictionResult
from ml.realtime_engine import RealtimeAttendanceEngine

def _load_attendance_session() -> AttendanceSession:
    users = storage.load_users()
    if not users:
        raise RuntimeError("No registered students available. Enroll people in the Data Collection tab first.")
    return AttendanceSession(users=users)


def _session_needs_refresh(session: AttendanceSession) -> bool:
    current_users = storage.load_users()
    if not current_users:
        return True
    return set(session.users.keys()) != set(current_users.keys())

LOGGER = logging.getLogger(__name__)


def _render_checklist(session: AttendanceSession) -> None:
    rows = []
    for student in session.student_checklist():
        is_present = student["present"]
        checkbox = "☑" if is_present else "☐"
        name = student["name"]
        conf = student["confidence"]
        if is_present:
            rows.append(
                f'<div style="padding: 6px 0; font-size: 14px; color: #111; line-height: 1.4;">'
                f'<strong>{checkbox} {name}</strong> <span style="color: #4b5563;">(Present — {conf * 100:.1f}%)</span></div>'
            )
        else:
            rows.append(
                f'<div style="padding: 6px 0; font-size: 14px; color: #111; line-height: 1.4;">{checkbox} {name}</div>'
            )

    html = (
        '<div style="max-height: 440px; overflow-y: auto; padding: 12px; '
        'border: 1px solid #e2e8f0; border-radius: 12px; background-color: #ffffff;">'
        + "".join(rows) + "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def _render_dashboard(stats: Dict[str, object]) -> None:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Students", stats["total_students"])
    c2.metric("Present", stats["present"])
    c3.metric("Absent", stats["absent"])
    c4.metric("Attendance Rate", stats["attendance_rate"])
    c5.metric("Current FPS", stats["current_fps"])

    c6, c7, c8, c9, c10 = st.columns(5)
    c6.metric("Unknown Faces", stats["unknown_faces"])
    c7.metric("Duplicate Recognitions", stats["duplicate_recognitions"])
    c8.metric("Average Confidence", stats["average_confidence"])
    c9.metric("Avg Recognition Time", stats["average_recognition_time"])
    c10.metric("Elapsed Session", stats["elapsed_session_time"])


def render() -> None:
    ui.hero("6 · Application", "Real-time attendance as a production-ready application page.")
    ui.pipeline(["Webcam", "Detection", "Embedding", "Inference", "Attendance"])

    if not storage.load_embeddings_db():
        st.info("No enrolled identities yet — register people in Tab 1 first.")
        ui.lesson("A trained model and enrolled students are required for the Application page.")
        return

    if storage.load_classifier() is None:
        st.warning("No trained classifier found. Train a model in the Model Building tab before using the Application page.")
        ui.lesson("Application only performs inference; it does not retrain models.")
        return

    # Sidebar parameters
    capture_mode = st.sidebar.radio(
        "Capture mode",
        ["Photo capture", "Upload photo"],
        index=0,
        key="app_capture_mode",
        help="Photo capture takes a static snapshot using your webcam. Upload photo allows uploading a classroom image.",
    )
    threshold = st.sidebar.slider(
        "Similarity Threshold",
        0.30,
        0.90,
        config.DEFAULT_SIMILARITY_THRESHOLD,
        0.05,
        key="app_threshold",
    )
    confirmation_frames = st.sidebar.slider(
        "Confirmation frames",
        1,
        10,
        5,
        1,
        key="app_confirmation_frames",
    )

    # Initialize Session
    if "attendance_session" not in st.session_state:
        try:
            st.session_state.attendance_session = _load_attendance_session()
        except RuntimeError as exc:
            st.error(str(exc))
            return

    if "session_uuid" not in st.session_state:
        st.session_state.session_uuid = str(uuid.uuid4())

    session: AttendanceSession = st.session_state.attendance_session
    if _session_needs_refresh(session):
        session = _load_attendance_session()
        st.session_state.attendance_session = session
        st.session_state.engine = None

    session.confirmation_frames = confirmation_frames

    # Initialize Engine
    if "engine" not in st.session_state or st.session_state.engine is None:
        try:
            engine = RealtimeAttendanceEngine(session=session, threshold=threshold)
            engine.load_classifier()
            st.session_state.engine = engine
        except Exception as exc:
            LOGGER.exception("Failed to initialize realtime attendance engine.")
            st.error(f"Unable to load the model: {exc}")
            return
    else:
        # Check and reload classifier dynamically if it was retrained
        try:
            st.session_state.engine.load_classifier()
        except Exception as exc:
            st.warning(f"Unable to reload updated model: {exc}")

    engine: RealtimeAttendanceEngine = st.session_state.engine
    engine.set_threshold(threshold)

    # UI Grid Setup
    dashboard_placeholder = st.empty()
    st.markdown("---")

    cols = st.columns([2, 1])
    with cols[0]:
        st.markdown("### Camera")
        camera_placeholder = st.empty()
        camera_control_placeholder = st.empty()
        camera_notice_placeholder = st.empty()

    with cols[1]:
        st.markdown("### Student Attendance Checklist")
        checklist_placeholder = st.empty()

        st.markdown("### Controls")
        if capture_mode == "Photo capture":
            st.info("Photo capture mode is active.")
        else:
            st.info("Upload photo mode is active.")

        if st.button("Reset Attendance", type="secondary", width="stretch"):
            session.reset()
            st.rerun()

        csv_data = session.export_csv()
        st.download_button(
            label="Download attendance.csv",
            data=csv_data,
            file_name="attendance.csv",
            mime="text/csv",
            key="download_attendance",
            width="stretch",
        )

    # Render Dashboard & Checklist Placeholders
    stats_helper = SessionStatistics(session)

    sound_placeholder = st.empty()
    last_records_count = len(session.records)

    # Render initial/current dashboard and checklist
    stats = stats_helper.get_stats(0.0)
    with dashboard_placeholder.container():
        _render_dashboard(stats)

    with checklist_placeholder.container():
        _render_checklist(session)

    # Input source based on mode
    image = None
    if capture_mode == "Photo capture":
        photo = camera_control_placeholder.camera_input("Capture a class photo", key="app_photo_cam")
        if photo is not None:
            image = ui.file_to_bgr(photo)
        camera_notice_placeholder.info("Take a photo to detect and mark attendance from that image.")
    elif capture_mode == "Upload photo":
        uploaded = camera_control_placeholder.file_uploader("Upload classroom photo", type=["jpg", "jpeg", "png"], key="app_upload_photo")
        if uploaded is not None:
            image = ui.file_to_bgr(uploaded)
        camera_notice_placeholder.info("Upload a photo to detect and mark attendance from that image.")

    # Process and display result if an image is provided
    if image is not None:
        annotated, predictions, annotations = engine.process_photo(image)
        camera_placeholder.image(ui.bgr_to_rgb(annotated), width="stretch")

        # Refresh dashboard and checklist after processing
        stats = stats_helper.get_stats(0.0)
        with dashboard_placeholder.container():
            _render_dashboard(stats)
        with checklist_placeholder.container():
            _render_checklist(session)

        # Play sound and show toast if a student was marked present
        if len(session.records) > last_records_count:
            new_record = session.records[-1]
            st.toast(f"✓ {new_record.name} marked as Present", icon="✅")
            sound_placeholder.markdown(
                f'<audio autoplay src="https://assets.mixkit.co/active_storage/sfx/2869/2869-200.wav" style="display:none;"></audio>',
                unsafe_allow_html=True
            )
            last_records_count = len(session.records)

    st.markdown("---")
    with st.expander("👤 Manage Student List (CRUD)", expanded=False):
        users = storage.load_users()
        crud_mode = st.radio(
            "Select Action",
            ["Read List", "Create Student", "Update Student", "Delete Student"],
            horizontal=True,
            key="student_crud_mode"
        )
        
        if crud_mode == "Read List":
            st.markdown("##### Enrolled Student Directory")
            if not users:
                st.info("No students enrolled in the database.")
            else:
                student_data = []
                for uid, u in users.items():
                    created = u.get("created_at", "N/A")
                    if isinstance(created, str) and "T" in created:
                        created = created.split("T")[0]
                    student_data.append({
                        "Name": u.get("name", uid),
                        "User ID": uid,
                        "Created At": created
                    })
                st.table(student_data)
                
        elif crud_mode == "Create Student":
            st.markdown("##### Add New Student")
            with st.form("create_student_form", clear_on_submit=True):
                new_name = st.text_input("Student Name")
                submitted = st.form_submit_button("Create & Retrain Model")
                if submitted:
                    if not new_name.strip():
                        st.error("Student Name cannot be empty.")
                    else:
                        import re
                        base_id = re.sub(r'[^a-z0-9_]', '', new_name.strip().lower().replace(" ", "_"))
                        user_id = base_id
                        counter = 1
                        while user_id in users:
                            user_id = f"{base_id}_{counter}"
                            counter += 1
                        
                        storage.upsert_user(user_id, new_name.strip())
                        st.success(f"Student '{new_name.strip()}' (ID: {user_id}) created successfully!")
                        
                        with st.spinner("Retraining model classifier..."):
                            try:
                                from ml import model_builder
                                existing_clf = storage.load_classifier()
                                kind = existing_clf.get("kind", "SVM") if existing_clf else "SVM"
                                
                                model_builder.train(kind=kind, persist=True)
                                st.toast("✅ Model retrained successfully with updated students!", icon="⚙️")
                                
                                if "engine" in st.session_state and st.session_state.engine is not None:
                                    st.session_state.engine._classifier_loaded = False
                                    st.session_state.engine.load_classifier()
                            except Exception as exc:
                                st.warning(f"Metadata updated. Classifier was not retrained: {exc}")
                        
                        st.rerun()
                        
        elif crud_mode == "Update Student":
            st.markdown("##### Update Student Details")
            if not users:
                st.info("No students available to update.")
            else:
                student_options = {f"{u.get('name', uid)} ({uid})": uid for uid, u in users.items()}
                selected_opt = st.selectbox("Select Student", list(student_options.keys()))
                selected_uid = student_options[selected_opt]
                
                with st.form("update_student_form"):
                    current_name = users[selected_uid].get("name", "")
                    updated_name = st.text_input("New Name", value=current_name)
                    submitted = st.form_submit_button("Update Name")
                    if submitted:
                        if not updated_name.strip():
                            st.error("Name cannot be empty.")
                        else:
                            users[selected_uid]["name"] = updated_name.strip()
                            storage.save_users(users)
                            st.success(f"Student ID '{selected_uid}' name updated to '{updated_name.strip()}'!")
                            st.rerun()
                            
        elif crud_mode == "Delete Student":
            st.markdown("##### Remove Student")
            if not users:
                st.info("No students available to delete.")
            else:
                student_options = {f"{u.get('name', uid)} ({uid})": uid for uid, u in users.items()}
                selected_opt = st.selectbox("Select Student to Delete", list(student_options.keys()))
                selected_uid = student_options[selected_opt]
                
                st.warning(f"Are you sure you want to delete {selected_opt}? This will also delete their face embeddings.")
                confirm = st.button("Confirm Delete", type="primary")
                if confirm:
                    users.pop(selected_uid, None)
                    storage.save_users(users)
                    
                    embeddings_db = storage.load_embeddings_db()
                    if selected_uid in embeddings_db:
                        embeddings_db.pop(selected_uid, None)
                        storage.save_embeddings_db(embeddings_db)
                        
                    st.success(f"Student '{selected_opt}' deleted successfully.")
                    st.rerun()

    ui.lesson("This page demonstrates production-style realtime inference without retraining.")
