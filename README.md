---
title: Demo Face Recognition
emoji: 👁️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Demo Face Recognition

Ứng dụng demo nhận diện khuôn mặt sử dụng Streamlit + InsightFace.



# ML Lifecycle Demo (Face-Recognition Attendance)

Streamlit app that walks through an end-to-end machine-learning lifecycle:

- **Data Processing** (collect → prepare → extract embeddings)
- **Model Building** (train SVM/KNN/MLP)
- **Evaluation** (metrics, confusion matrix, thresholding)
- **Deployment** (browser camera → recognition → attendance log)
- **Monitoring & Feedback** (feedback loop)

> Inspired by teachable/interactive ML demos: everything is tab-driven so you can narrate each stage while the audience clicks along.

---

## Live demo / Run locally

### 1) Install dependencies
```bash
pip install -r requirements.txt
```

### 2) Start Streamlit
```bash
streamlit run app.py
```

### 3) Pre-download InsightFace models (recommended)
On first run, the app may download weights. If you want to avoid delays during a presentation, pre-download before the demo.

---

## Using the app

1. Open the sidebar.
2. Click **🚀 Load Sample Dataset** to instantly enroll a small set of historical scientists (Ada Lovelace, Alan Turing, Grace Hopper) with a trained classifier.
3. Navigate through the lifecycle tabs using the **Lifecycle stage** radio.

### Key interactions
- **Data Collection**: register one or more people by capturing poses.
- **Data Preparation**: visualize augmentation (brightness/contrast/blur/flip).
- **Feature Extraction**: inspect the 512-D identity embedding.
- **Model Training**: train a classifier (SVM/KNN/MLP).
- **Evaluation**: explore metrics and adjust the confidence/threshold behavior.
- **Deployment**: use your browser camera to fill an attendance log.
- **Application**: experience a classroom attendance system with photo capture and upload support.
- **Reset**: **🔄 Reset demo data** to start over.

---

## Project structure

- `app.py` – Streamlit entrypoint + tab routing.
- `app/tabs/*` – UI for each lifecycle stage.
- `ml/*` – ML utilities (embedding, storage, model building, evaluation, realtime inference).
- `datasets/`, `logs/`, `models/` – runtime artifacts.

## New Application architecture

```mermaid
flowchart LR
    AppPage["Application Page<br/>(app/tabs/application.py)"] -->|calls| RealtimeEngine["Realtime Engine<br/>(ml/realtime_engine.py)"]
    RealtimeEngine -->|uses| Camera["Webcam / Image Upload<br/>(st.camera_input / file_uploader)"]
    RealtimeEngine -->|uses| FaceDetector["Face Detection<br/>(ml/face_detector.py)"]
    RealtimeEngine -->|uses| Embedder["Embedding Extraction<br/>(ml/embedder.py)"]
    RealtimeEngine -->|uses| Classifier["Classifier Prediction<br/>(ml/attendance_engine.py)"]
    RealtimeEngine -->|manages| Tracker["FaceTracker<br/>(IoU bounding-box tracking)"]
    RealtimeEngine -->|manages| StateMachine["RecognitionStateMachine<br/>(Detecting → Verifying → Confirmed / Unknown)"]
    AppPage -->|stores state| Session["AttendanceSession<br/>(ml/attendance_session.py)"]
    Session -->|loads students| Storage["Storage<br/>(ml/storage.py)"]
    Session -->|contains| EventLog["EventLogger<br/>(last 100 events)"]
    AppPage -->|reads| Stats["SessionStatistics<br/>(dashboard metrics)"]
    AppPage -->|updates| UI["Checklist, Dashboard,<br/>Timeline, Export"]
```

## Application sequence

```mermaid
sequenceDiagram
    participant User
    participant UI as Streamlit UI
    participant Engine as RealtimeAttendanceEngine
    participant Tracker as FaceTracker
    participant Detector as Face Detector
    participant Embedder as Embedder
    participant Classifier as Classifier
    participant SM as RecognitionStateMachine
    participant Session as AttendanceSession
    participant Logger as EventLogger
    participant Stats as SessionStatistics

    User->>UI: Open Application tab
    UI->>Session: initialize students from storage
    UI->>Engine: load classifier
    User->>UI: Take photo or upload image
    UI->>Engine: process_photo(image)
    Engine->>Detector: detect_faces(image)
    Detector-->>Engine: detected faces
    Engine->>Tracker: update(bounding boxes)
    Tracker-->>Engine: face IDs (stable)
    loop each detected face
        Engine->>Embedder: embed_face(face)
        Embedder-->>Engine: embedding vector
        Engine->>Classifier: predict_identity(embedding)
        Classifier-->>Engine: RecognitionResult (name, confidence)
        Engine->>SM: update(prediction, is_already_present)
        SM-->>Engine: state (Detecting / Verifying / Confirmed / Unknown)
        alt state transitions to Confirmed
            Engine->>Session: mark_present(user_id, confidence)
            Engine->>Logger: log_event("✓ Present", green)
        else state transitions to Unknown
            Engine->>Logger: log_event("Rejected", red)
        else duplicate detection
            Engine->>Logger: log_event("Duplicate Ignored", gray)
        end
    end
    Engine->>Session: add_inference_time(ms)
    Engine-->>UI: annotated image
    UI->>Stats: get_stats(0.0)
    Stats-->>UI: dashboard metrics
    UI->>UI: render annotated image, checklist, timeline, dashboard
    User->>UI: Reset / Export CSV
```

---

## Testing

```bash
pytest -q
```

---

## Requirements

Python packages are listed in `requirements.txt`.

---

## Notes for presenters

See `DEMO_SCRIPT.md` for an ~8-minute narration guide with specific clicks and talking points for each tab.

