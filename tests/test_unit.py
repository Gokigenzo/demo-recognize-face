"""Unit tests: face detection, embedding extraction, similarity search.

These tests are backend-agnostic — they pass whether InsightFace or the
OpenCV fallback is active, so they're safe to run on any machine / in CI.
"""
from __future__ import annotations

import numpy as np
import pytest

from ml import config
from ml.embedder import _fallback_embedding, cosine_similarity, embed_crop, pca_project
from ml.face_detector import detect_faces, detect_largest_face


def _synthetic_face(size: int = 160) -> np.ndarray:
    """A simple synthetic 'face-ish' BGR image (gradient + features)."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    yy, xx = np.mgrid[0:size, 0:size]
    img[..., 0] = (xx / size * 200).astype(np.uint8)
    img[..., 1] = (yy / size * 200).astype(np.uint8)
    img[..., 2] = 120
    return img


# ---------------------------------------------------------------------------
# Face detection
# ---------------------------------------------------------------------------
def test_detect_faces_empty_image_returns_empty():
    assert detect_faces(np.zeros((0, 0, 3), dtype=np.uint8)) == []


def test_detect_faces_none_returns_empty():
    assert detect_faces(None) == []


def test_detect_largest_face_handles_no_face():
    # A flat image typically has no detectable face; must not raise.
    result = detect_largest_face(_synthetic_face())
    assert result is None or hasattr(result, "crop")


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------
def test_fallback_embedding_dimension_and_norm():
    crop = _synthetic_face(112)
    emb = _fallback_embedding(crop)
    assert emb.shape == (config.EMBEDDING_DIM,)
    assert pytest.approx(np.linalg.norm(emb), rel=1e-4) == 1.0


def test_embed_crop_deterministic():
    crop = _synthetic_face(112)
    a = embed_crop(crop)
    b = embed_crop(crop)
    assert a is not None and b is not None
    assert np.allclose(a, b)


def test_embed_crop_empty_returns_none():
    assert embed_crop(np.zeros((0, 0, 3), dtype=np.uint8)) is None


# ---------------------------------------------------------------------------
# Similarity search
# ---------------------------------------------------------------------------
def test_cosine_similarity_identity():
    v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    assert pytest.approx(cosine_similarity(v, v), rel=1e-5) == 1.0


def test_cosine_similarity_orthogonal():
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    assert pytest.approx(cosine_similarity(a, b), abs=1e-6) == 0.0


def test_cosine_similarity_handles_zero_vector():
    a = np.zeros(4, dtype=np.float32)
    b = np.ones(4, dtype=np.float32)
    assert cosine_similarity(a, b) == 0.0


# ---------------------------------------------------------------------------
# PCA projection
# ---------------------------------------------------------------------------
def test_pca_projection_shape():
    embs = [np.random.rand(config.EMBEDDING_DIM).astype(np.float32) for _ in range(5)]
    coords = pca_project(embs, n_components=2)
    assert coords.shape == (5, 2)


def test_pca_empty_input():
    assert pca_project([], 2).shape == (0, 2)
