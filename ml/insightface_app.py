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
        # Monkey patch ONNX Runtime to force single-threaded execution and prevent segfaults
        try:
            import onnxruntime as ort
            original_init = ort.InferenceSession.__init__
            def patched_init(self, model_path, sess_options=None, *args, **kwargs):
                if sess_options is None:
                    sess_options = ort.SessionOptions()
                sess_options.intra_op_num_threads = 1
                sess_options.inter_op_num_threads = 1
                sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                original_init(self, model_path, sess_options, *args, **kwargs)
            ort.InferenceSession.__init__ = patched_init
        except Exception as patch_err:
            import logging
            logging.warning("Failed to monkey patch onnxruntime: %s", patch_err)

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
