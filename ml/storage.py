"""Lightweight persistence layer (JSON / CSV / Pickle).

This module isolates *all* disk I/O so the rest of the codebase never touches
file paths directly. It deliberately uses simple, human-inspectable formats
(JSON + CSV) plus pickle for the numeric embedding database, matching the
demo's "storage" learning objective.
"""
from __future__ import annotations

import csv
import json
import os
import pickle
from datetime import datetime
from typing import Any, Dict, List

import numpy as np

from ml import config


# ---------------------------------------------------------------------------
# Users (JSON)
# ---------------------------------------------------------------------------
def load_users() -> Dict[str, Dict[str, Any]]:
    """Return the registered-user metadata mapping {user_id: {...}}."""
    if not os.path.exists(config.USERS_DB_PATH):
        return {}
    with open(config.USERS_DB_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_users(users: Dict[str, Dict[str, Any]]) -> None:
    config.ensure_dirs()
    with open(config.USERS_DB_PATH, "w", encoding="utf-8") as fh:
        json.dump(users, fh, indent=2)


def upsert_user(user_id: str, name: str, extra: Dict[str, Any] | None = None) -> None:
    """Create or update a user record."""
    users = load_users()
    record = users.get(user_id, {"created_at": datetime.now().isoformat()})
    record.update({"user_id": user_id, "name": name})
    if extra:
        record.update(extra)
    users[user_id] = record
    save_users(users)


# ---------------------------------------------------------------------------
# Embedding database (Pickle)
# ---------------------------------------------------------------------------
def load_embeddings_db() -> Dict[str, List[np.ndarray]]:
    """Return {user_id: [embedding, ...]}.

    Embeddings are stored as a list per user so we can keep multiple poses /
    augmented samples and average / search across them.
    """
    if not os.path.exists(config.EMBEDDINGS_DB_PATH):
        return {}
    with open(config.EMBEDDINGS_DB_PATH, "rb") as fh:
        return pickle.load(fh)


def save_embeddings_db(db: Dict[str, List[np.ndarray]]) -> None:
    config.ensure_dirs()
    with open(config.EMBEDDINGS_DB_PATH, "wb") as fh:
        pickle.dump(db, fh)


def add_embeddings(user_id: str, embeddings: List[np.ndarray]) -> None:
    """Append one or more embeddings for a user and persist."""
    db = load_embeddings_db()
    db.setdefault(user_id, [])
    db[user_id].extend([np.asarray(e, dtype=np.float32) for e in embeddings])
    save_embeddings_db(db)


# ---------------------------------------------------------------------------
# Trained classifier (Pickle)
# ---------------------------------------------------------------------------
def save_classifier(payload: Dict[str, Any]) -> None:
    """Persist a trained classifier bundle (model + metadata)."""
    config.ensure_dirs()
    with open(config.CLASSIFIER_PATH, "wb") as fh:
        pickle.dump(payload, fh)


def load_classifier() -> Dict[str, Any] | None:
    """Return the persisted classifier bundle, or None if not trained yet."""
    if not os.path.exists(config.CLASSIFIER_PATH):
        return None
    with open(config.CLASSIFIER_PATH, "rb") as fh:
        return pickle.load(fh)


# ---------------------------------------------------------------------------
# Attendance log (CSV)
# ---------------------------------------------------------------------------
ATTENDANCE_FIELDS = ["timestamp", "user_id", "name", "confidence", "status"]


def append_attendance(
    user_id: str, name: str, confidence: float, status: str = "present"
) -> Dict[str, Any]:
    """Append a single attendance row and return the row written."""
    config.ensure_dirs()
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "user_id": user_id,
        "name": name,
        "confidence": round(float(confidence), 4),
        "status": status,
    }
    file_exists = os.path.exists(config.ATTENDANCE_LOG_PATH)
    with open(config.ATTENDANCE_LOG_PATH, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=ATTENDANCE_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    return row


def load_attendance() -> List[Dict[str, Any]]:
    if not os.path.exists(config.ATTENDANCE_LOG_PATH):
        return []
    with open(config.ATTENDANCE_LOG_PATH, "r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# ---------------------------------------------------------------------------
# Feedback / monitoring logs (JSON lists)
# ---------------------------------------------------------------------------
def _load_json_list(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as fh:
        try:
            return json.load(fh)
        except json.JSONDecodeError:
            return []


def _append_json_list(path: str, entry: Dict[str, Any]) -> None:
    config.ensure_dirs()
    data = _load_json_list(path)
    data.append(entry)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def append_feedback(entry: Dict[str, Any]) -> None:
    _append_json_list(config.FEEDBACK_LOG_PATH, entry)


def load_feedback() -> List[Dict[str, Any]]:
    return _load_json_list(config.FEEDBACK_LOG_PATH)


def append_monitoring(entry: Dict[str, Any]) -> None:
    _append_json_list(config.MONITORING_LOG_PATH, entry)


def load_monitoring() -> List[Dict[str, Any]]:
    return _load_json_list(config.MONITORING_LOG_PATH)


# ---------------------------------------------------------------------------
# Portable demo bundle (export / import for offline presentations)
# ---------------------------------------------------------------------------
# Bump this if the bundle layout ever changes so imports can validate it.
BUNDLE_VERSION = 1


def export_bundle() -> bytes:
    """Serialize the entire demo state into a single portable byte blob.

    Bundles the registered users, the embedding database, and (if present) the
    trained classifier so a presenter can reload a pre-enrolled dataset on any
    machine with no internet or re-enrollment required.
    """
    bundle = {
        "version": BUNDLE_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "users": load_users(),
        "embeddings": load_embeddings_db(),
        "classifier": load_classifier(),
    }
    return pickle.dumps(bundle)


def import_bundle(data: bytes, *, replace: bool = True) -> Dict[str, Any]:
    """Load a previously exported bundle and write it to disk.

    When ``replace`` is True the current users/embeddings/classifier are fully
    overwritten; otherwise embeddings are merged per-user. Returns a small
    summary describing what was imported.

    Raises ``ValueError`` if the blob is not a recognizable demo bundle.
    """
    try:
        bundle = pickle.loads(data)
    except Exception as exc:  # corrupt / wrong file type
        raise ValueError(f"Not a valid demo bundle: {exc}") from exc

    if not isinstance(bundle, dict) or "embeddings" not in bundle or "users" not in bundle:
        raise ValueError("File is not a recognizable ML Lifecycle demo bundle.")

    config.ensure_dirs()
    users = bundle.get("users", {}) or {}
    embeddings = bundle.get("embeddings", {}) or {}
    classifier = bundle.get("classifier")

    if replace:
        save_users(users)
        save_embeddings_db(
            {uid: [np.asarray(e, dtype=np.float32) for e in embs]
             for uid, embs in embeddings.items()}
        )
    else:
        merged_users = load_users()
        merged_users.update(users)
        save_users(merged_users)
        for uid, embs in embeddings.items():
            add_embeddings(uid, embs)

    if classifier is not None:
        save_classifier(classifier)

    return {
        "version": bundle.get("version"),
        "created_at": bundle.get("created_at"),
        "users": len(users),
        "embeddings": sum(len(v) for v in embeddings.values()),
        "has_classifier": classifier is not None,
    }


def generate_historical_scientists_bundle(file_path: str) -> None:
    """Generate a sample bundle containing historical scientists and save it to file_path.

    Constructs well-separated embedding clusters for Ada Lovelace, Alan Turing,
    and Grace Hopper, trains an initial SVM model, and exports the bundle.
    """
    current_users = load_users()
    current_embeddings = load_embeddings_db()
    current_classifier = load_classifier()

    try:
        rng = np.random.default_rng(42)

        def get_noisy_embs(centroid, n, std):
            embs = []
            for _ in range(n):
                noise = rng.normal(0, std, size=config.EMBEDDING_DIM)
                noisy = centroid + noise
                embs.append((noisy / (np.linalg.norm(noisy) or 1.0)).astype(np.float32))
            return embs

        ada_center = rng.normal(size=config.EMBEDDING_DIM)
        ada_center = ada_center / np.linalg.norm(ada_center)

        turing_center = rng.normal(size=config.EMBEDDING_DIM)
        turing_center -= np.dot(turing_center, ada_center) * ada_center
        turing_center = turing_center / np.linalg.norm(turing_center)

        hopper_center = rng.normal(size=config.EMBEDDING_DIM)
        hopper_center -= np.dot(hopper_center, ada_center) * ada_center
        hopper_center -= np.dot(hopper_center, turing_center) * turing_center
        hopper_center = hopper_center / np.linalg.norm(hopper_center)

        users = {
            "ada_lovelace": {
                "created_at": datetime.now().isoformat(),
                "user_id": "ada_lovelace",
                "name": "Ada Lovelace"
            },
            "alan_turing": {
                "created_at": datetime.now().isoformat(),
                "user_id": "alan_turing",
                "name": "Alan Turing"
            },
            "grace_hopper": {
                "created_at": datetime.now().isoformat(),
                "user_id": "grace_hopper",
                "name": "Grace Hopper"
            }
        }

        # Generate base poses (std = 0.03) + augmented (std = 0.08)
        embeddings_db = {
            "ada_lovelace": get_noisy_embs(ada_center, 5, 0.03) + get_noisy_embs(ada_center, 20, 0.08),
            "alan_turing": get_noisy_embs(turing_center, 5, 0.03) + get_noisy_embs(turing_center, 20, 0.08),
            "grace_hopper": get_noisy_embs(hopper_center, 5, 0.03) + get_noisy_embs(hopper_center, 20, 0.08),
        }

        save_users(users)
        save_embeddings_db(embeddings_db)

        from ml import model_builder
        model_builder.train(kind="SVM", persist=True)

        bundle_data = export_bundle()
        config.ensure_dirs()
        with open(file_path, "wb") as f:
            f.write(bundle_data)
    finally:
        save_users(current_users)
        save_embeddings_db(current_embeddings)
        if current_classifier is not None:
            save_classifier(current_classifier)
        elif os.path.exists(config.CLASSIFIER_PATH):
            os.remove(config.CLASSIFIER_PATH)


# ---------------------------------------------------------------------------
# Maintenance helpers (used by the UI "reset demo" button & tests)
# ---------------------------------------------------------------------------

def reset_all() -> None:
    """Delete all persisted demo artifacts. Used to reset between demos."""
    for path in (
        config.EMBEDDINGS_DB_PATH,
        config.CLASSIFIER_PATH,
        config.USERS_DB_PATH,
        config.ATTENDANCE_LOG_PATH,
        config.FEEDBACK_LOG_PATH,
        config.MONITORING_LOG_PATH,
    ):
        if os.path.exists(path):
            os.remove(path)
