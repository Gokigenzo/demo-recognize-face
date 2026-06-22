"""Shared InsightFace ``FaceAnalysis`` singleton.

Both :mod:`ml.face_detector` and :mod:`ml.embedder` need the same model. We
load it once, lazily, and cache the result. ``insightface_available()`` lets
callers branch to a fallback without raising during import.
"""
from __future__ import annotations

import threading
from typing import Optional

from ml import config

_app: Optional[object] = None
_available: Optional[bool] = None
_lock = threading.Lock()


def _try_load() -> Optional[object]:
    """Attempt to construct and prepare a FaceAnalysis app."""
    try:
        from insightface.app import FaceAnalysis

        app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=config.DET_SIZE)
        return app
    except Exception:  # noqa: BLE001 - any failure means "use fallback"
        return None


def get_face_app() -> Optional[object]:
    """Return the cached FaceAnalysis app, loading it on first use."""
    global _app, _available
    if _available is None:
        with _lock:
            if _available is None:  # double-checked locking
                _app = _try_load()
                _available = _app is not None
    return _app


def insightface_available() -> bool:
    """True if the InsightFace backend loaded successfully."""
    get_face_app()
    return bool(_available)
