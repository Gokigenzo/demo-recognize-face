# Demo Structure & How It Works — Face Recognition ML Lifecycle

## High-level architecture

- **`app.py`**: Streamlit entrypoint and sidebar router.
  - Sets page config and theme.
  - Builds the lifecycle stage navigation.
  - Shows enrolled people / total samples and supports export/import bundles.
  - Includes a demo reset button for a fresh state.

- **`app/tabs/*`**: tab-driven stages that separate the ML lifecycle into presentation-friendly UI.
  - **Tab 1: Data Processing** (`app/tabs/data_processing.py`)
    - Contains:
      - `app/tabs/data_collection.py` — register a person with camera upload or file upload.
      - `app/tabs/data_preparation.py` — synthesize augmented embeddings and prepare the dataset.
      - `app/tabs/collected_data.py` — dataset summary table and sample counts.

  - **Tab 2: Model Building** (`app/tabs/model_building.py`)
    - Trains a scikit-learn classifier on stored embeddings.
    - Builds a 2D PCA visualization so users can see decision regions.

  - **Tab 3: Evaluation** (`app/tabs/evaluation.py`)
    - Generates test variations from enrolled embeddings.
    - Computes confusion matrix, accuracy, precision, recall, and F1.
    - Shows a threshold sweep that highlights the precision/recall tradeoff.

  - **Tab 4: Deployment** (`app/tabs/deployment.py`)
    - Demonstrates a single-frame live inference path.
    - Supports both a browser capture path (`st.camera_input`) and server-side OpenCV webcam.
    - Logs attendance rows when a known identity is accepted.

  - **Tab 5: Monitoring & Feedback** (`app/tabs/monitoring_feedback.py`)
    - Enables human-in-the-loop correction.
    - Stores feedback and optionally adds corrected embeddings back into the dataset.

  - **Tab 6: Application** (`app/tabs/application.py`)
    - Production-style realtime attendance page.
    - Uses continuous inference from either the server webcam or browser fallback.
    - Tracks faces across frames using IoU-based `FaceTracker`.
    - Applies `RecognitionStateMachine` per face (Detecting → Verifying → Confirmed / Unknown) to stabilize predictions.
    - Maintains a live student checklist, real-time event timeline (`EventLogger`), and session dashboard (`SessionStatistics`).
    - Supports export to `attendance.csv` and reset of runtime attendance state.
    - Audio and toast notifications on attendance confirmation.

- **`ml/*`**: backend utilities and inference pipelines.
  - `ml/config.py` — paths, constants, threshold defaults, and directory creation.
  - `ml/storage.py` — all persistence for users, embeddings, classifiers, attendance, and feedback.
  - `ml/embedder.py` — embedding extraction plus fallback embedding logic.
  - `ml/face_detector.py` — face detection with InsightFace SCRFD or OpenCV Haar fallback.
  - `ml/insightface_app.py` — InsightFace model loader and availability gating.
  - `ml/model_builder.py` — classifier training, cross-validation, and PCA visualization.
  - `ml/evaluator.py` — evaluation metrics and threshold sweep support.
  - `ml/attendance_engine.py` — similarity-based identity recognition and attendance utilities.
  - `ml/attendance_session.py` — in-memory attendance state, `RecognitionStateMachine`, `EventLogger` (timeline), `SessionStatistics` (dashboard), and CSV export.
  - `ml/realtime_engine.py` — continuous frame inference engine with `FaceTracker` (IoU tracking) and state-machine-driven annotation.
  - `ml/feedback_engine.py` — feedback storage and correction helpers.
  - `ml/augmenter.py` — embedding augmentation utility functions.

## Core concept: embeddings are the features

The demo does not classify raw pixels. Instead:
- Faces are detected and cropped.
- Face crops are converted into fixed-length **embeddings**.
- Those embeddings are the feature vectors used for training, evaluation, and identification.

`ml/embedder.py` guarantees:
- `config.EMBEDDING_DIM` is respected (512 dimensions),
- outputs are **L2-normalized**.

If InsightFace is available, it produces real ArcFace-style features. If not, a deterministic fallback descriptor keeps the demo functional.

## End-to-end lifecycle

### 1) Data Collection

File: `app/tabs/data_collection.py`
- Collect a person’s name, pose, and image.
- Detect the largest face in the input.
- Compute the embedding for that face.
- Persist the user and embeddings via `ml/storage.py`.

### 2) Data Preparation

File: `app/tabs/data_preparation.py`
- Augments existing embeddings rather than recapturing images.
- Adds controlled noise variants and renormalizes embeddings.
- Writes augmented samples back to the embeddings database.

### 3) Model Building

Files: `app/tabs/model_building.py`, `ml/model_builder.py`
- Builds a classifier from the stored embedding dataset.
- Supports SVM, KNN, and MLP training.
- Produces both a full-dimension prediction model and a 2D PCA visualization model.

### 4) Evaluation

Files: `app/tabs/evaluation.py`, `ml/evaluator.py`
- Requires enrolled identities and trained embeddings.
- Synthesizes test cases and unknown samples.
- Computes classification metrics and threshold tradeoffs.
- Demonstrates why a high-confidence reject can be better than a wrong label.

### 5) Deployment

Files: `app/tabs/deployment.py`, `ml/attendance_engine.py`
- Performs live single-frame inference.
- Supports browser capture and server webcam capture.
- Uses similarity thresholding to decide known vs unknown.
- Writes attendance entries for accepted identities.

### 6) Application

Files: `app/tabs/application.py`, `ml/realtime_engine.py`, `ml/attendance_session.py`
- Production-style realtime attendance application page.
- Streams continuous frames from webcam or browser fallback.
- Tracks faces across frames using IoU-based `FaceTracker` to maintain stable face IDs.
- Each tracked face is managed by a `RecognitionStateMachine` with states: Detecting → Verifying → Confirmed / Unknown.
- Temporal confirmation requires N consecutive identical predictions before marking attendance.
- Marks students present only once; duplicates are counted and logged.
- Real-time dashboard (`SessionStatistics`) shows present/absent/rate/FPS/unknowns/duplicates/avg confidence/avg inference time/elapsed time.
- Live event timeline (`EventLogger`) shows the last 100 recognition events with color coding.
- Exports runtime attendance as CSV without retraining.

## Persistence and portability

`ml/storage.py` handles:
- Users: `users.json`
- Embeddings: `embeddings_db.pkl`
- Classifier bundle: `classifier.pkl`
- Attendance log: `logs/attendance.csv`
- Feedback and monitoring logs

Bundles:
- `export_bundle()` / `import_bundle()` packages users, embeddings, and the trained classifier for demo portability.
- `reset_all()` clears persisted state so the demo can start fresh.

## Documentation mapping

- `README.md` explains usage, project structure, and quick startup.
- `DEMO_SCRIPT.md` provides a scripted demo narrative tied to the lifecycle tabs.
- `FILE_INDEX.md` is the new file lookup index for quick navigation.

