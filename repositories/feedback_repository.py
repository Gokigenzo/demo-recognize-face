"""Feedback repository implementation.

Logs and retrieves human corrections and feedback loop inputs.
Handles mapping between legacy JSON representation and relational Supabase table.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from typing import Any, Dict, List, Optional

from ml import config
from repositories.base_repository import get_manager
from repositories.student_repository import get_student_repo

logger = logging.getLogger(__name__)


class FeedbackRepository:
    """Manages feedback correction records and monitoring logs."""

    def __init__(self) -> None:
        self._feedback_cache: Optional[List[Dict[str, Any]]] = None
        self._lock = threading.Lock()

    def invalidate_cache(self) -> None:
        with self._lock:
            self._feedback_cache = None

    def append_feedback(self, entry: Dict[str, Any]) -> None:
        """Log a human correction entry."""
        with self._lock:
            if self._feedback_cache is not None:
                self._feedback_cache.append(entry)
            self._append_json_list(config.FEEDBACK_LOG_PATH, entry)

        # Resolve student UUID in Supabase
        correct_slug = entry.get("correct_user_id")
        student_uuid = get_student_repo().get_uuid(correct_slug) if correct_slug else None

        db_row = {
            "predicted_name": entry.get("predicted", "Unknown"),
            "correct_name": entry.get("corrected_to", "Unknown"),
            "correct_student_id": correct_slug,
            "confidence": entry.get("confidence", 0.0),
            "user_decision": "Correction",
            "note": entry.get("note", ""),
            "timestamp": entry.get("timestamp"),
        }
        if student_uuid:
            db_row["student_id"] = student_uuid

        sync_entry = {
            "table": "monitoring_feedback",
            "operation": "insert",
            "data": db_row,
            "student_slug": correct_slug,
        }

        def op(client, r=db_row, slug=correct_slug):
            if "student_id" not in r and slug:
                uuid_val = get_student_repo().get_uuid(slug)
                if uuid_val:
                    r["student_id"] = uuid_val
            client.table("monitoring_feedback").insert(r).execute()

        def fallback():
            pass

        manager = get_manager()
        manager.execute(op, fallback, sync_entry)

    def load_feedback(self) -> List[Dict[str, Any]]:
        """Return all logged feedback entries in the legacy format."""
        with self._lock:
            if self._feedback_cache is not None:
                return list(self._feedback_cache)

            manager = get_manager()
            if manager.is_online and manager.client is not None:
                try:
                    response = (
                        manager.client.table("monitoring_feedback")
                        .select("*, students(student_id)")
                        .order("timestamp")
                        .execute()
                    )
                    rows = response.data or []
                    
                    cache = []
                    for row in rows:
                        student_info = row.get("students")
                        slug = student_info.get("student_id") if student_info else row.get("correct_student_id")
                        
                        cache.append({
                            "timestamp": row["timestamp"],
                            "predicted": row["predicted_name"],
                            "corrected_to": row["correct_name"],
                            "correct_user_id": slug,
                            "note": row.get("note") or "",
                        })
                    
                    self._feedback_cache = cache
                    self._overwrite_json_list(config.FEEDBACK_LOG_PATH, cache)
                    return list(cache)
                except Exception as exc:
                    logger.warning("Failed to fetch feedback from Supabase: %s. Loading local file.", exc)
                    manager.mark_offline()

            # Offline fallback
            cache = self._load_json_list(config.FEEDBACK_LOG_PATH)
            self._feedback_cache = cache
            return list(cache)

    def append_monitoring(self, entry: Dict[str, Any]) -> None:
        """Append a monitoring log entry (local file only as it's unused in the ML pipeline)."""
        with self._lock:
            self._append_json_list(config.MONITORING_LOG_PATH, entry)

    def load_monitoring(self) -> List[Dict[str, Any]]:
        """Load monitoring logs (local file fallback)."""
        with self._lock:
            return self._load_json_list(config.MONITORING_LOG_PATH)

    # -- Helper JSON list methods ---------------------------------------------
    def _load_json_list(self, path: str) -> List[Dict[str, Any]]:
        if not os.path.exists(path):
            return []
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
                return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError):
            return []

    def _append_json_list(self, path: str, entry: Dict[str, Any]) -> None:
        try:
            config.ensure_dirs()
            data = self._load_json_list(path)
            data.append(entry)
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, default=str)
        except Exception as exc:
            logger.error("Failed to append to JSON list %s: %s", path, exc)

    def _overwrite_json_list(self, path: str, data: List[Dict[str, Any]]) -> None:
        try:
            config.ensure_dirs()
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, default=str)
        except Exception as exc:
            logger.error("Failed to overwrite JSON list %s: %s", path, exc)


_instance: Optional[FeedbackRepository] = None
_inst_lock = threading.Lock()


def get_feedback_repo() -> FeedbackRepository:
    """Get the singleton FeedbackRepository instance."""
    global _instance
    if _instance is None:
        with _inst_lock:
            if _instance is None:
                _instance = FeedbackRepository()
    return _instance
