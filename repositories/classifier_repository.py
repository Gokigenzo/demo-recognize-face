"""Classifier repository implementation.

Saves and loads scikit-learn classifier model pickle bundles and training
metadata to/from Supabase PostgreSQL + Storage with offline local fallback.
"""
from __future__ import annotations

import logging
import os
import pickle
import threading
from datetime import datetime
from typing import Any, Dict, Optional

from ml import config
from repositories.base_repository import get_manager

logger = logging.getLogger(__name__)


class ClassifierRepository:
    """Manages trained classifier artifacts and metadata in Supabase."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def save(self, payload: Dict[str, Any]) -> None:
        """Persist a trained classifier bundle (model + metadata).

        Writes locally as a pickle file and uploads to Supabase Storage,
        logging metadata in PostgreSQL.
        """
        with self._lock:
            # 1. Save local pickle file
            try:
                config.ensure_dirs()
                with open(config.CLASSIFIER_PATH, "wb") as fh:
                    pickle.dump(payload, fh)
            except Exception as exc:
                logger.error("Failed to write classifier locally: %s", exc)

            # 2. Upload to Supabase Storage and save metadata in DB
            manager = get_manager()
            if manager.is_online and manager.client is not None:
                try:
                    # Pickle payload as bytes
                    pickle_bytes = pickle.dumps(payload)
                    
                    # Ensure models bucket exists (ignore error if exists or no permission)
                    try:
                        manager.client.storage.create_bucket("models", options={"public": True})
                    except Exception:
                        pass
                    
                    # Upload to Storage
                    # Note: gotrue / storage3 client usage:
                    filename = "classifier.pkl"
                    manager.client.storage.from_("models").upload(
                        filename,
                        pickle_bytes,
                        file_options={"content-type": "application/octet-stream", "upsert": "true"},
                    )
                    
                    # Clean hyperparameters to be JSON serializable
                    raw_hyperparams = payload.get("hyperparameters", {})
                    serializable_hyperparams = {}
                    if isinstance(raw_hyperparams, dict):
                        for k, v in raw_hyperparams.items():
                            if isinstance(v, (str, int, float, bool, list, dict)) or v is None:
                                serializable_hyperparams[k] = v
                            else:
                                serializable_hyperparams[k] = str(v)

                    # Save metadata in DB
                    # Calculate metrics if not present
                    db_row = {
                        "model_name": payload.get("kind", "Unknown"),
                        "training_date": datetime.now().isoformat(),
                        "training_accuracy": float(payload.get("train_accuracy", 0.0)),
                        "precision": float(payload.get("precision", payload.get("train_accuracy", 0.0))),
                        "recall": float(payload.get("recall", payload.get("train_accuracy", 0.0))),
                        "f1": float(payload.get("f1", payload.get("train_accuracy", 0.0))),
                        "hyperparameters": serializable_hyperparams,
                        "model_version": "1.0",
                        "model_path": "models/classifier.pkl",
                    }
                    
                    manager.client.table("trained_classifiers").insert(db_row).execute()
                    logger.info("Classifier uploaded to Supabase Storage and metadata registered in DB.")
                except Exception as exc:
                    logger.warning("Failed to upload classifier to Supabase: %s", exc)

    def load(self) -> Optional[Dict[str, Any]]:
        """Load the persisted classifier bundle.

        Checks local file first; if missing, downloads from Supabase Storage.
        """
        with self._lock:
            # 1. Try to load local file
            if os.path.exists(config.CLASSIFIER_PATH):
                try:
                    with open(config.CLASSIFIER_PATH, "rb") as fh:
                        return pickle.load(fh)
                except Exception as exc:
                    logger.error("Failed to read local classifier pickle: %s", exc)

            # 2. Try to download from Supabase Storage if online
            manager = get_manager()
            if manager.is_online and manager.client is not None:
                try:
                    logger.info("Local classifier missing. Attempting to download from Supabase Storage …")
                    # Download bytes
                    pickle_bytes = manager.client.storage.from_("models").download("classifier.pkl")
                    if pickle_bytes:
                        # Save locally
                        config.ensure_dirs()
                        with open(config.CLASSIFIER_PATH, "wb") as fh:
                            fh.write(pickle_bytes)
                        # Load and return
                        return pickle.loads(pickle_bytes)
                except Exception as exc:
                    logger.warning("Failed to download classifier from Supabase: %s", exc)

            return None


_instance: Optional[ClassifierRepository] = None
_inst_lock = threading.Lock()


def get_classifier_repo() -> ClassifierRepository:
    """Get the singleton ClassifierRepository instance."""
    global _instance
    if _instance is None:
        with _inst_lock:
            if _instance is None:
                _instance = ClassifierRepository()
    return _instance
