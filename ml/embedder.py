"""Face embedding (feature extraction).

Turns a face crop into a fixed-length vector ("embedding"). With InsightFace
this is the real 512-D ArcFace embedding. Without it, we compute a
deterministic 512-D descriptor from a resized grayscale crop so the rest of
the pipeline (similarity search, clustering, evaluation) still works for the
demo. Both paths return L2-normalized vectors of length
:data:`ml.config.EMBEDDING_DIM`.
"""
from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np

from ml import config
from ml.face_detector import DetectedFace
from ml.insightface_app import insightface_available


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec.astype(np.float32)
    return (vec / norm).astype(np.float32)


def _fallback_embedding(crop: np.ndarray) -> np.ndarray:
    """Deterministic descriptor used when InsightFace is unavailable.

    Resizes the crop to a fixed grid, flattens grayscale intensities, then
    pads/truncates to EMBEDDING_DIM. Not accurate, but stable & repeatable so
    the same face yields the same vector -> similarity search behaves sanely.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    side = int(np.ceil(np.sqrt(config.EMBEDDING_DIM)))  # 23 -> 529 >= 512
    small = cv2.resize(gray, (side, side)).astype(np.float32).flatten()
    vec = small[: config.EMBEDDING_DIM]
    if vec.shape[0] < config.EMBEDDING_DIM:
        vec = np.pad(vec, (0, config.EMBEDDING_DIM - vec.shape[0]))
    return _l2_normalize(vec)


def embed_face(face: DetectedFace) -> np.ndarray:
    """Return the L2-normalized embedding for a detected face."""
    if insightface_available() and face.raw is not None:
        emb = getattr(face.raw, "normed_embedding", None)
        if emb is None:
            emb = getattr(face.raw, "embedding", None)
        if emb is not None:
            return _l2_normalize(np.asarray(emb, dtype=np.float32))
    return _fallback_embedding(face.crop)


def embed_crop(crop: np.ndarray) -> Optional[np.ndarray]:
    """Embed a standalone face crop (BGR np.ndarray).

    Used for augmented samples where we already have a tight crop. When
    InsightFace is active we re-detect inside the crop to obtain a real
    embedding; otherwise we use the deterministic fallback.
    """
    if crop is None or crop.size == 0:
        return None
    if insightface_available():
        # Local import avoids a circular import at module load time.
        from ml.face_detector import detect_largest_face

        face = detect_largest_face(crop)
        if face is not None and face.raw is not None:
            return embed_face(face)
    return _fallback_embedding(crop)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two (assumed normalized) vectors."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
    return float(np.dot(a, b) / denom)


def pca_project(embeddings: List[np.ndarray], n_components: int = 2) -> np.ndarray:
    """Project a list of embeddings to ``n_components`` dims via PCA.

    Returns an ``(n_samples, n_components)`` array. Gracefully handles the
    edge case of fewer samples than components.
    """
    from sklearn.decomposition import PCA

    if not embeddings:
        return np.empty((0, n_components), dtype=np.float32)
    matrix = np.vstack([np.asarray(e, dtype=np.float32) for e in embeddings])
    n_components = min(n_components, matrix.shape[0], matrix.shape[1])
    if n_components < 1:
        return matrix
    return PCA(n_components=n_components).fit_transform(matrix)
