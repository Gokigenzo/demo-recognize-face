"""Face detection.

Primary backend is InsightFace (SCRFD detector bundled with ``buffalo_l``).
If InsightFace cannot be loaded (e.g. missing models on an offline machine),
we transparently fall back to OpenCV's Haar cascade so the demo *always*
runs. The active backend is reported via :func:`backend_name` so the UI can
display it honestly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ml import config

# The InsightFace app is shared with the embedder (loading it twice is slow
# and wastes memory), so it lives in ml.insightface_app.
from ml.insightface_app import get_face_app, insightface_available


@dataclass
class DetectedFace:
    """A single detected face within an image."""

    bbox: Tuple[int, int, int, int]      # (x1, y1, x2, y2)
    crop: np.ndarray                      # BGR crop resized to FACE_CROP_SIZE
    confidence: float
    # InsightFace face object (None when using the Haar fallback). Carries the
    # embedding so the embedder can reuse it without re-running detection.
    raw: Optional[object] = None


# ---------------------------------------------------------------------------
# Haar fallback (loaded lazily)
# ---------------------------------------------------------------------------
_haar_cascade: Optional[cv2.CascadeClassifier] = None


def _get_haar() -> cv2.CascadeClassifier:
    global _haar_cascade
    if _haar_cascade is None:
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _haar_cascade = cv2.CascadeClassifier(path)
    return _haar_cascade


def backend_name() -> str:
    """Human-readable name of the active detection backend."""
    return "InsightFace (SCRFD)" if insightface_available() else "OpenCV Haar Cascade"


def _crop_and_resize(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        # Degenerate box -> return a blank crop rather than crashing.
        return np.zeros((*config.FACE_CROP_SIZE, 3), dtype=np.uint8)
    crop = image[y1:y2, x1:x2]
    return cv2.resize(crop, config.FACE_CROP_SIZE)


def detect_faces(image: np.ndarray) -> List[DetectedFace]:
    """Detect all faces in a BGR image.

    Parameters
    ----------
    image:
        A BGR ``np.ndarray`` (as returned by ``cv2.imread`` / VideoCapture).

    Returns
    -------
    list[DetectedFace]
        Ordered by detection confidence (highest first).
    """
    if image is None or image.size == 0:
        return []

    if insightface_available():
        app = get_face_app()
        faces = app.get(image)
        results: List[DetectedFace] = []
        for f in faces:
            x1, y1, x2, y2 = (int(v) for v in f.bbox)
            results.append(
                DetectedFace(
                    bbox=(x1, y1, x2, y2),
                    crop=_crop_and_resize(image, (x1, y1, x2, y2)),
                    confidence=float(getattr(f, "det_score", 1.0)),
                    raw=f,
                )
            )
        results.sort(key=lambda d: d.confidence, reverse=True)
        return results

    # ---- Haar fallback ----
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    boxes = _get_haar().detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    results = []
    for (x, y, w, h) in boxes:
        bbox = (int(x), int(y), int(x + w), int(y + h))
        results.append(
            DetectedFace(
                bbox=bbox,
                crop=_crop_and_resize(image, bbox),
                confidence=1.0,
                raw=None,
            )
        )
    return results


def detect_largest_face(image: np.ndarray) -> Optional[DetectedFace]:
    """Return the single largest detected face, or ``None``."""
    faces = detect_faces(image)
    if not faces:
        return None
    return max(faces, key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]))


def draw_detection(image: np.ndarray, face: DetectedFace, label: str = "") -> np.ndarray:
    """Return a copy of ``image`` with the face box (and optional label) drawn."""
    out = image.copy()
    x1, y1, x2, y2 = face.bbox
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 0), 2)
    if label:
        cv2.putText(
            out, label, (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2, cv2.LINE_AA,
        )
    return out
