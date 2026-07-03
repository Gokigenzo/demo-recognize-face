# File Index — Demo Face Recognition

This project is organized into UI tabs, ML backend utilities, persisted artifacts, and test coverage.

## Root files

- `app.py` — Streamlit application entrypoint and tab router.
- `README.md` — project overview, run instructions, and high-level usage.
- `DEMO_STRUCTURE_SUMMARY.md` — architecture summary and lifecycle explanation.
- `DEMO_SCRIPT.md` — guided narration for presenting the demo.
- `Dockerfile` — container build definition.
- `requirements.txt` — Python dependency list.
- `TODO.md` — project TODOs and planned enhancements.
- `test.py` — convenience test launcher / quick script.

## App UI

- `app/__init__.py` — package marker.
- `app/ui_helpers.py` — shared UI helpers, theme injection, image conversions, and styling.
- `app/tabs/__init__.py` — exposes tab renderers for the app.
- `app/tabs/data_processing.py` — Data Processing stage combining collection and preparation tabs.
- `app/tabs/data_collection.py` — capture and enrollment UI for registered identities.
- `app/tabs/data_preparation.py` — embedding augmentation and synthetic sample generation UI.
- `app/tabs/collected_data.py` — dataset overview table and sample metrics.
- `app/tabs/model_building.py` — classifier training and visualization UI.
- `app/tabs/evaluation.py` — evaluation metrics, threshold analysis, and confusion matrix UI.
- `app/tabs/deployment.py` — live deployment inference and attendance logging UI.
- `app/tabs/monitoring_feedback.py` — human-in-the-loop feedback and correction UI.
- `app/tabs/application.py` — production-style realtime attendance application page.

## ML backend

- `ml/__init__.py` — package marker.
- `ml/config.py` — application paths, constants, thresholds, and runtime directories.
- `ml/storage.py` — persistence layer for users, embeddings, classifier bundles, attendance logs, and export/import.
- `ml/embedder.py` — face embedding extraction and cosine similarity utilities.
- `ml/face_detector.py` — face detection abstraction with InsightFace and Haar fallback.
- `ml/insightface_app.py` — InsightFace loader and model availability handling.
- `ml/model_builder.py` — classifier training pipeline and PCA visualization support.
- `ml/evaluator.py` — evaluation metrics, threshold sweep, and test sample generation.
- `ml/attendance_engine.py` — identity recognition and attendance marking logic.
- `ml/feedback_engine.py` — feedback processing support.
- `ml/augmenter.py` — embedding-level augmentation utilities.
- `ml/realtime_engine.py` — continuous frame inference engine for the Application page; includes `FaceTracker` (IoU-based bounding box tracking) and state-machine-driven annotation rendering.
- `ml/attendance_session.py` — in-memory attendance session state, `RecognitionStateMachine` (state transitions: Detecting → Verifying → Confirmed / Unknown), `EventLogger` (timeline event tracking), and `SessionStatistics` (dashboard metrics).

## Persisted artifacts

- `datasets/` — stored datasets and user metadata (`users.json`).
- `logs/` — runtime logs and attendance CSV files.
- `models/` — saved embeddings database and trained classifier bundle.

## Tests

- `tests/__init__.py` — test package marker.
- `tests/test_integration.py` — integration-level validation.
- `tests/test_model_builder.py` — model training and data preparation tests.
- `tests/test_system.py` — end-to-end system checks.
- `tests/test_unit.py` — unit-level component tests.
- `tests/test_realtime.py` — unit tests for `RecognitionStateMachine`, `FaceTracker`, `EventLogger`, `SessionStatistics`, and `AttendanceSession` extensions.

## How to use this file

Use `FILE_INDEX.md` as a quick lookup for the project structure and the purpose of each major file. It is especially useful for onboarding, documentation, or when updating the demo with new stages.
