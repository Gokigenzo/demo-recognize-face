# Demo Structure & How It Works — Face Recognition ML Lifecycle

## High-level architecture
- **`app.py`**: Streamlit entrypoint + sidebar routing.
  - Sets page config, initializes directories, applies theme.
  - Shows a **Lifecycle stage** radio and routes to one of the tab renderers:
    - Data Processing
    - Model Building
    - Evaluation
    - Deployment
    - Monitoring & Feedback
  - Sidebar also shows enrolled people / total samples from the embeddings DB, and provides **export/import** bundle controls and demo reset.

- **`app/tabs/*`**: UI for each lifecycle stage.
  - **Tab 1: Data Processing** (`app/tabs/data_processing.py`)
    - Sub-tabs:
      - Data Collection (`app/tabs/data_collection.py`)
      - Data Preparation & Augmentation (`app/tabs/data_preparation.py`)
    - Also renders a dataset overview table (`app/tabs/collected_data.py`).

  - **Tab 2/4: Model Building** (`app/tabs/model_building.py`)
    - Trains a scikit-learn classifier on embeddings.
    - Visualizes learned class regions using a **2D PCA projection**.

  - **Tab 3/5: Evaluation** (`app/tabs/evaluation.py`)
    - Generates a controlled test set by perturbing stored embeddings.
    - Computes confusion matrix + Accuracy/Precision/Recall/F1.
    - Shows a threshold sweep precision-vs-recall tradeoff curve.

  - **Tab 4/6: Deployment** (`app/tabs/deployment.py`)
    - Live inference pipeline:
      - Camera/Frame → face detection → embedding → similarity search/threshold → attendance log.

  - **Tab 5: Monitoring & Feedback** (`app/tabs/monitoring_feedback.py`)
    - Human-in-the-loop correction UI.
    - Stores feedback and adds corrected embeddings back into the embedding DB.

- **`ml/*`**: backend logic.
  - **`ml/storage.py`**: persistence layer isolating all disk I/O.
  - **`ml/embedder.py`**: converts a face crop to a normalized embedding (512-D).
  - **`ml/model_builder.py`**: classifier training + PCA visualization model.
  - **`ml/evaluator.py`**: metrics computation + threshold sweeps.
  - **`ml/attendance_engine.py`**: runtime identification + attendance marking.
  - **`ml/face_detector.py` + `ml/insightface_app.py`**: face detection and InsightFace availability.

## Core concept: embeddings as the “features”
The demo does not attempt to classify directly from images. Instead:
- A detected face crop is converted into a fixed-length **embedding vector**.
- The stored embeddings are treated as the features used for:
  - training
  - evaluation
  - live identification

`ml/embedder.py` enforces two invariants:
- embedding dimensionality is `config.EMBEDDING_DIM` (512)
- embeddings are **L2-normalized**

If InsightFace is available, it uses the real **ArcFace/InsightFace embedding**. Otherwise it uses a deterministic 512-D fallback descriptor so the rest of the demo still behaves sensibly.

## End-to-end lifecycle: what happens in each stage

### 1) Data Collection (Tab 1 · Data Collection)
File: `app/tabs/data_collection.py`
- Input from UI:
  - Person name
  - Pose label
  - Image source: browser camera or upload
- Backend pipeline:
  1. detect largest face in the image (`detect_largest_face`)
  2. crop the detected face
  3. compute embedding (`embed_face`)
- Persisted outputs:
  - `storage.upsert_user(user_id, name)`
  - `storage.add_embeddings(user_id, [embedding])`

At this point, the embedding DB grows with each registered pose.

### 2) Data Preparation / Augmentation (Tab 1 · Data Preparation)
File: `app/tabs/data_preparation.py`
- Augmentation is performed at the **embedding level** (not re-capturing images).
- For each stored embedding, it generates multiple synthetic variants:
  - add Gaussian noise with chosen sigma
  - re-normalize with L2 norm
- The newly generated embeddings are appended back to the embeddings DB.

This design supports the demo constraint: “no re-capture required”.

### 3) Model Building (Classifier Training + Visualization)
File: `app/tabs/model_building.py` + `ml/model_builder.py`
- The UI offers classifier choice:
  - SVM
  - KNN
  - MLP
- Training inputs:
  - `ml/model_builder.collect_dataset()` flattens the embedding DB into (X, y)
    - X: embeddings
    - y: class index per user
- Training outputs:
  - a full-dimensional scikit-learn classifier used conceptually for separation
  - a 2D PCA projection model for plotting

Visualization:
- `decision_boundary_mesh()` computes a grid over PCA space.
- The model predicts a class index per grid point.
- The UI shows filled regions and scatter points, so the audience can “see” class separation.

### 4) Evaluation (Metrics + Threshold Behavior)
File: `app/tabs/evaluation.py` + `ml/evaluator.py`
- It requires enrolled identities (embeddings DB must not be empty).
- Test set construction:
  - For each enrolled user, compute an average embedding.
  - Create multiple “case” variants (Normal/Glasses/Mask/Side Face/Dark Lighting) by adding structured noise.
  - Add a few “Unknown Person” vectors (random normalized embeddings).
- Metrics:
  - confusion matrix
  - accuracy, precision, recall, F1
- Threshold sweep:
  - a slider controls the similarity threshold.
  - the tradeoff curve shows how raising/lowering the threshold changes precision vs recall vs F1.

This stage teaches that “Unknown” decisions can be preferable to wrong confident predictions.

### 5) Deployment (Live Inference + Attendance)
File: `app/tabs/deployment.py` + `ml/attendance_engine.py`
- Capture mode:
  - Browser camera (via `st.camera_input`) OR
  - Server webcam (OpenCV `VideoCapture`)
- Live pipeline:
  1. detect largest face
  2. extract embedding
  3. identify with `identify(emb, threshold=...)`
  4. mark attendance using `mark_attendance(...)` if the identity is known and not already logged
- Unknown behavior:
  - if similarity is below threshold, it refuses to guess and logs as unknown.

The tab also renders the attendance CSV as a table.

### 6) Monitoring & Feedback Loop (Human correction)
File: `app/tabs/monitoring_feedback.py` + `ml/feedback_engine.py` + `ml/storage.py`
- UI flow:
  1. capture/upload image
  2. preview model prediction at current threshold
  3. user chooses the correct identity (or Unknown)
  4. store feedback
  5. add corrected embedding back to embeddings DB
- Result:
  - the next training/evaluation/deployment can benefit from the new examples.

## Persistence & artifacts (handled by `ml/storage.py`)
`ml/storage.py` centralizes all disk I/O:
- **Users**: JSON (`load_users`, `save_users`, `upsert_user`)
- **Embeddings DB**: pickle (`load_embeddings_db`, `save_embeddings_db`, `add_embeddings`)
- **Trained classifier bundle**: pickle (`save_classifier`, `load_classifier`)
- **Attendance log**: CSV (`append_attendance`, `load_attendance`)
- **Feedback / Monitoring logs**: JSON list files (`append_feedback`, `load_feedback`, etc.)

Demo portability:
- `export_bundle()` / `import_bundle()` serializes:
  - users
  - embeddings
  - classifier (if present)
- Presenters can load the pre-enrolled dataset quickly offline.

Reset:
- `reset_all()` deletes persisted artifacts so the demo starts fresh.

## How the narration matches the UI
- `README.md` summarizes the lifecycle and the main interactions.
- `DEMO_SCRIPT.md` provides an ~8-minute guided narration tied to the lifecycle stage tabs.

