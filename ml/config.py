"""Central configuration for the ML Lifecycle Demo.

Keeps all tunable constants and filesystem paths in one place so the rest of
the codebase (ml/ + app/) can import them without duplicating magic values.
"""
from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Filesystem layout
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASETS_DIR = os.path.join(PROJECT_ROOT, "datasets")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")

# Persisted artifacts -------------------------------------------------------
EMBEDDINGS_DB_PATH = os.path.join(MODELS_DIR, "embeddings_db.pkl")
CLASSIFIER_PATH = os.path.join(MODELS_DIR, "classifier.pkl")
USERS_DB_PATH = os.path.join(DATASETS_DIR, "users.json")
ATTENDANCE_LOG_PATH = os.path.join(LOGS_DIR, "attendance.csv")
FEEDBACK_LOG_PATH = os.path.join(LOGS_DIR, "feedback.json")
MONITORING_LOG_PATH = os.path.join(LOGS_DIR, "monitoring.json")

# ---------------------------------------------------------------------------
# Model / pipeline parameters
# ---------------------------------------------------------------------------
EMBEDDING_DIM = 512                 # InsightFace buffalo_l output dimension
FACE_CROP_SIZE = (112, 112)         # standard ArcFace input size
DET_SIZE = (640, 640)               # InsightFace detector input size
REALTIME_TARGET_FPS = 5.0           # Cap live recognition to keep the UI responsive
REALTIME_CAMERA_WIDTH = 640         # Lower than full HD to keep latency low on CPU
REALTIME_CAMERA_HEIGHT = 480
REALTIME_CAMERA_FPS = 15            # Conservative webcam FPS for smoother live preview
REALTIME_DETECTION_INTERVAL = 2     # Run face detection every N recognition cycles
REALTIME_RESIZE_WIDTH = 640        # Downscale large frames before detection
REALTIME_UI_REFRESH_SECONDS = 0.25 # Limit dashboard/checklist refresh rate

# Cosine-similarity threshold above which a face is accepted as a known
# identity. Tunable live via the Evaluation tab slider.
DEFAULT_SIMILARITY_THRESHOLD = 0.45

# The canonical capture poses requested during data collection.
CAPTURE_POSES = ["Front", "Left", "Right", "Up", "Down"]

# Augmentation multiplier shown in the Data Preparation tab (10 -> 50).
AUGMENTATIONS_PER_IMAGE = 4

# Standard noise standard deviations to simulate environmental / appearance conditions
NOISE_NORMAL = 0.02
NOISE_GLASSES = 0.08
NOISE_LOW_LIGHT = 0.10
NOISE_SIDE_FACE = 0.15
NOISE_MASK = 0.18


# Educational copy reused across tabs ---------------------------------------
LIFECYCLE_STAGES = [
    "Data Collection",
    "Data Preparation",
    "Feature Extraction",
    "Model Building",
    "Evaluation",
    "Deployment",
    "Monitoring",
    "Feedback Loop",
    "Continuous Improvement",
]


def ensure_dirs() -> None:
    """Create the runtime directories if they do not already exist."""
    for path in (DATASETS_DIR, MODELS_DIR, LOGS_DIR):
        os.makedirs(path, exist_ok=True)
