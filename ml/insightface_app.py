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


def ensure_models() -> None:
    """Verify and robustly download/extract the buffalo_l models if missing or corrupted.

    This prevents onnxruntime from crashing with a segmentation fault when loading
    corrupted or partially downloaded ONNX files.
    """
    import os
    import shutil
    import urllib.request
    import zipfile
    import logging

    logger = logging.getLogger(__name__)

    root_dir = os.path.expanduser("~/.insightface")
    models_dir = os.path.join(root_dir, "models")
    target_dir = os.path.join(models_dir, "buffalo_l")
    zip_path = os.path.join(models_dir, "buffalo_l.zip")

    # Expected files and minimum valid size threshold (in bytes)
    expected_models = {
        "1k3d68.onnx": 100_000_000,
        "2d106det.onnx": 4_000_000,
        "det_10g.onnx": 15_000_000,
        "genderage.onnx": 1_000_000,
        "w600k_r50.onnx": 150_000_000,
    }

    is_valid = True
    if not os.path.exists(target_dir):
        is_valid = False
    else:
        for filename, min_size in expected_models.items():
            filepath = os.path.join(target_dir, filename)
            if not os.path.exists(filepath):
                logger.warning("Expected model file missing: %s", filepath)
                is_valid = False
                break
            file_size = os.path.getsize(filepath)
            if file_size < min_size:
                logger.warning(
                    "Model file corrupted/truncated: %s (size %d < %d)",
                    filepath, file_size, min_size
                )
                is_valid = False
                break

    if is_valid:
        logger.info("InsightFace buffalo_l models verified successfully.")
        return

    logger.warning("InsightFace models missing or invalid. Initiating clean download...")
    if os.path.exists(target_dir):
        try:
            shutil.rmtree(target_dir)
        except Exception as exc:
            logger.error("Failed to delete corrupted directory %s: %s", target_dir, exc)

    if os.path.exists(zip_path):
        try:
            os.remove(zip_path)
        except Exception as exc:
            logger.error("Failed to delete zip file %s: %s", zip_path, exc)

    os.makedirs(models_dir, exist_ok=True)

    url = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
    logger.info("Downloading buffalo_l.zip from %s...", url)
    
    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
        )
        with urllib.request.urlopen(req, timeout=120) as response, open(zip_path, "wb") as out_file:
            meta = response.info()
            file_size = int(meta.get("Content-Length", 0))
            logger.info("Target file size: %s bytes", file_size)

            downloaded = 0
            block_size = 1024 * 1024  # 1MB blocks
            last_reported = 0.0

            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                downloaded += len(buffer)
                out_file.write(buffer)
                
                if file_size > 0:
                    percent = (downloaded / file_size) * 100
                    if percent - last_reported >= 10.0:
                        logger.info("Download progress: %.1f%% (%d/%d bytes)", percent, downloaded, file_size)
                        last_reported = percent

        logger.info("Download completed. Verifying zip file size...")
        downloaded_size = os.path.getsize(zip_path)
        if downloaded_size < 200_000_000:
            raise ValueError(f"Downloaded zip file is too small ({downloaded_size} bytes)")

        logger.info("Extracting models to %s...", target_dir)
        os.makedirs(target_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(target_dir)

        logger.info("InsightFace models extracted successfully.")

    except Exception as exc:
        logger.error("Failed to download or extract buffalo_l models: %s", exc)
        if os.path.exists(target_dir):
            try:
                shutil.rmtree(target_dir)
            except Exception:
                pass
        raise
    finally:
        if os.path.exists(zip_path):
            try:
                os.remove(zip_path)
            except Exception:
                pass


def _try_load() -> Optional[object]:
    """Attempt to construct and prepare a FaceAnalysis app."""
    try:
        # Ensure the model files are verified and present
        ensure_models()

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
