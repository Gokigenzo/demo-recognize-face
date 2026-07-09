"""Persistence layer adapter wrapping the Supabase Repository Layer.

This module acts as a backward-compatible wrapper. The rest of the codebase
imports functions from this module as usual, but all storage operations are
internally dispatched to the clean Repository Layer backed by Supabase with
in-memory caching and offline fallbacks.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from ml import config
# Import concrete repositories and singleton accessors
from repositories import (
    get_manager,
    get_student_repo,
    get_embedding_repo,
    get_attendance_repo,
    get_feedback_repo,
    get_statistics_repo,
    get_configuration_repo,
    get_classifier_repo,
)

logger = logging.getLogger(__name__)

# Re-export BUNDLE_VERSION
BUNDLE_VERSION = 1


# ---------------------------------------------------------------------------
# Registered Students (legacy name: Users)
# ---------------------------------------------------------------------------
def load_users() -> Dict[str, Dict[str, Any]]:
    """Return the registered-user metadata mapping {user_id: {...}}."""
    return get_student_repo().load_all()


def save_users(users: Dict[str, Dict[str, Any]]) -> None:
    """Overwrite all registered student records."""
    get_student_repo().save_all(users)


def upsert_user(user_id: str, name: str, extra: Dict[str, Any] | None = None) -> None:
    """Create or update a single user record."""
    get_student_repo().upsert(user_id, name, extra)


# ---------------------------------------------------------------------------
# Embedding Database
# ---------------------------------------------------------------------------
def load_embeddings_db() -> Dict[str, List[np.ndarray]]:
    """Return the embeddings database {user_id: [embedding, ...]}."""
    return get_embedding_repo().load_all()


def save_embeddings_db(db: Dict[str, List[np.ndarray]]) -> None:
    """Overwrite the entire embeddings database."""
    get_embedding_repo().save_all(db)


def add_embeddings(user_id: str, embeddings: List[np.ndarray]) -> None:
    """Append one or more embeddings for a user and persist."""
    get_embedding_repo().add(user_id, embeddings)


# ---------------------------------------------------------------------------
# Trained Classifier
# ---------------------------------------------------------------------------
def save_classifier(payload: Dict[str, Any]) -> None:
    """Persist a trained classifier bundle (model + metadata)."""
    get_classifier_repo().save(payload)


def load_classifier() -> Dict[str, Any] | None:
    """Return the persisted classifier bundle, or None if not trained yet."""
    return get_classifier_repo().load()


# ---------------------------------------------------------------------------
# Attendance Log
# ---------------------------------------------------------------------------
def append_attendance(
    user_id: str, name: str, confidence: float, status: str = "present"
) -> Dict[str, Any]:
    """Append a single attendance row and return the row written."""
    return get_attendance_repo().append(user_id, name, confidence, status)


def load_attendance() -> List[Dict[str, Any]]:
    """Load all attendance records."""
    return get_attendance_repo().load_all()


# ---------------------------------------------------------------------------
# Feedback / Monitoring Logs
# ---------------------------------------------------------------------------
def append_feedback(entry: Dict[str, Any]) -> None:
    """Log a human correction feedback entry."""
    get_feedback_repo().append_feedback(entry)


def load_feedback() -> List[Dict[str, Any]]:
    """Load correction feedback logs."""
    return get_feedback_repo().load_feedback()


def append_monitoring(entry: Dict[str, Any]) -> None:
    """Log a system monitoring entry."""
    get_feedback_repo().append_monitoring(entry)


def load_monitoring() -> List[Dict[str, Any]]:
    """Load monitoring logs."""
    return get_feedback_repo().load_monitoring()


# ---------------------------------------------------------------------------
# Portable Demo Bundle (export / import)
# ---------------------------------------------------------------------------
def export_bundle() -> bytes:
    """Serialize the entire active demo state into a single portable byte blob."""
    import pickle
    bundle = {
        "version": BUNDLE_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "users": load_users(),
        "embeddings": load_embeddings_db(),
        "classifier": load_classifier(),
    }
    return pickle.dumps(bundle)


def import_bundle(data: bytes, *, replace: bool = True) -> Dict[str, Any]:
    """Load a previously exported bundle and write it to disk and Supabase."""
    import pickle
    try:
        bundle = pickle.loads(data)
    except Exception as exc:
        raise ValueError(f"Not a valid demo bundle: {exc}") from exc

    if not isinstance(bundle, dict) or "embeddings" not in bundle or "users" not in bundle:
        raise ValueError("File is not a recognizable ML Lifecycle demo bundle.")

    users = bundle.get("users", {}) or {}
    embeddings = bundle.get("embeddings", {}) or {}
    classifier = bundle.get("classifier")

    # Invalidate cache so they reload from DB/files
    get_student_repo().invalidate_cache()
    get_embedding_repo().invalidate_cache()

    if replace:
        save_users(users)
        save_embeddings_db(
            {uid: [np.asarray(e, dtype=np.float32) for e in embs]
             for uid, embs in embeddings.items()}
        )
    else:
        # Merge operation
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
    """Generate a sample bundle containing Ada Lovelace, Alan Turing, and Grace Hopper."""
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
# Reset & Maintenance Helpers
# ---------------------------------------------------------------------------
def reset_all() -> None:
    """Delete all persisted demo artifacts locally and in Supabase."""
    # 1. Clear local cache and files
    for path in (
        config.EMBEDDINGS_DB_PATH,
        config.CLASSIFIER_PATH,
        config.USERS_DB_PATH,
        config.ATTENDANCE_LOG_PATH,
        config.FEEDBACK_LOG_PATH,
        config.MONITORING_LOG_PATH,
    ):
        if os.path.exists(path):
            try:
                os.remove(path)
            except Exception as exc:
                logger.error("Failed to delete local file %s during reset: %s", path, exc)

    # Invalidate repository caches
    get_student_repo().invalidate_cache()
    get_embedding_repo().invalidate_cache()
    get_attendance_repo().invalidate_cache()
    get_feedback_repo().invalidate_cache()
    get_statistics_repo().invalidate_cache()
    get_configuration_repo().invalidate_cache()

    # 2. Reset Supabase tables if online
    manager = get_manager()
    if manager.is_online and manager.client is not None:
        try:
            logger.info("Resetting all Supabase tables …")
            # Truncating / deleting records in order of dependencies (leaves first, roots last)
            manager.client.table("embeddings").delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()
            manager.client.table("attendance").delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()
            manager.client.table("monitoring_feedback").delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()
            manager.client.table("students").delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()
            manager.client.table("trained_classifiers").delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()
            manager.client.table("session_statistics").delete().neq("session_id", "00000000-0000-0000-0000-000000000000").execute()
            
            # Reset application config to defaults
            manager.client.table("application_configuration").delete().eq("id", "default").execute()
            
            # Delete Storage classifier file if exists
            try:
                manager.client.storage.from_("models").remove(["classifier.pkl"])
            except Exception:
                pass
        except Exception as exc:
            logger.warning("Failed to reset Supabase tables: %s", exc)


# ---------------------------------------------------------------------------
# One-time Local -> Supabase Data Migration
# ---------------------------------------------------------------------------
def run_migration_if_needed() -> None:
    """Run one-time migration of local files to Supabase when first launched."""
    manager = get_manager()
    if not manager.is_online:
        return

    # Check if migration has already been recorded
    config_repo = get_configuration_repo()
    cfg = config_repo.load()
    if cfg.get("ui_preferences", {}).get("migration_completed") is True:
        return

    logger.info("Executing one-time local-to-Supabase migration …")
    try:
        # Load local files
        local_students = get_student_repo()._load_local_backup()
        local_embeddings = get_embedding_repo()._load_local_backup()
        local_attendance = get_attendance_repo()._load_local()
        local_feedback = get_feedback_repo()._load_json_list(config.FEEDBACK_LOG_PATH)
        local_classifier = get_classifier_repo().load()  # Try loading local pickle

        # 1. Migrate Students
        if local_students:
            logger.info("Migrating %d students …", len(local_students))
            get_student_repo().save_all(local_students)
            
            # Force load to establish uuid mappings
            get_student_repo().invalidate_cache()
            get_student_repo().load_all()

        # 2. Migrate Embeddings
        if local_embeddings:
            logger.info("Migrating embeddings database …")
            get_embedding_repo().save_all(local_embeddings)

        # 3. Migrate Attendance Logs
        if local_attendance:
            logger.info("Migrating %d attendance rows …", len(local_attendance))
            for row in local_attendance:
                # Format timestamp
                ts = row.get("timestamp")
                if ts:
                    try:
                        ts = datetime.fromisoformat(ts).isoformat()
                    except ValueError:
                        ts = datetime.now().isoformat()
                else:
                    ts = datetime.now().isoformat()

                student_uuid = get_student_repo().get_uuid(row.get("user_id"))
                db_row = {
                    "timestamp": ts,
                    "confidence": float(row.get("confidence", 0.0)),
                    "recognition_method": "InsightFace_Migration",
                    "status": row.get("status", "present"),
                }
                if student_uuid:
                    db_row["student_id"] = student_uuid
                
                if not manager.is_online or manager.client is None:
                    raise RuntimeError("Supabase offline during attendance migration")
                manager.client.table("attendance").insert(db_row).execute()

        # 4. Migrate Feedback logs
        if local_feedback:
            logger.info("Migrating %d feedback loop logs …", len(local_feedback))
            for entry in local_feedback:
                slug = entry.get("correct_user_id")
                student_uuid = get_student_repo().get_uuid(slug) if slug else None
                
                db_row = {
                    "predicted_name": entry.get("predicted"),
                    "correct_name": entry.get("corrected_to"),
                    "correct_student_id": slug,
                    "confidence": entry.get("confidence", 0.0),
                    "user_decision": "Correction_Migration",
                    "note": entry.get("note", ""),
                    "timestamp": entry.get("timestamp"),
                }
                if student_uuid:
                    db_row["student_id"] = student_uuid
                
                if not manager.is_online or manager.client is None:
                    raise RuntimeError("Supabase offline during feedback migration")
                manager.client.table("monitoring_feedback").insert(db_row).execute()

        # 5. Migrate Classifier Model Bundle
        if local_classifier:
            logger.info("Migrating trained model classifier …")
            get_classifier_repo().save(local_classifier)

        # Write migration completion flag
        prefs = cfg.get("ui_preferences", {})
        prefs["migration_completed"] = True
        cfg["ui_preferences"] = prefs
        
        if not manager.is_online or manager.client is None:
            raise RuntimeError("Supabase offline during migration completion registration")
        config_repo.save(cfg)
        logger.info("Migration completed successfully!")

    except Exception as exc:
        logger.error("Error during one-time migration to Supabase: %s", exc)



# Trigger auto-migration on start if online
run_migration_if_needed()

# Clear repository caches on module import/reload to ensure alignment with config paths
# (critical for preventing cache leakage between isolated pytest environments)
get_student_repo().invalidate_cache()
get_embedding_repo().invalidate_cache()
get_attendance_repo().invalidate_cache()
get_feedback_repo().invalidate_cache()
get_statistics_repo().invalidate_cache()
get_configuration_repo().invalidate_cache()

