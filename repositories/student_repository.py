"""Student repository implementation.

Manages student records, caching, and offline fallback.
Exposes a legacy-compatible dict interface to avoid breaking existing code.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime
from typing import Any, Dict, Optional

from ml import config
from repositories.base_repository import get_manager

logger = logging.getLogger(__name__)


class StudentRepository:
    """Manages student data with in-memory caching and offline fallback."""

    def __init__(self) -> None:
        self._cache: Optional[Dict[str, Dict[str, Any]]] = None
        self._slug_to_uuid: Dict[str, str] = {}
        self._lock = threading.Lock()

    def invalidate_cache(self) -> None:
        with self._lock:
            self._cache = None
            self._slug_to_uuid.clear()

    def get_uuid(self, student_id: str) -> Optional[str]:
        """Get the database UUID for a given slug student_id."""
        with self._lock:
            # If cache hasn't been loaded, load it first
            if self._cache is None:
                self._load_unlocked()
            return self._slug_to_uuid.get(student_id)

    def load_all(self) -> Dict[str, Dict[str, Any]]:
        """Return the dictionary of registered students in the legacy format.

        Format:
            {
                "student_id_slug": {
                    "created_at": "...",
                    "user_id": "student_id_slug",
                    "name": "Full Name",
                    "avatar_url": "...",
                    "active": True
                }
            }
        """
        with self._lock:
            return self._load_unlocked()

    def _load_unlocked(self) -> Dict[str, Dict[str, Any]]:
        if self._cache is not None:
            return dict(self._cache)

        manager = get_manager()
        if manager.is_online and manager.client is not None:
            try:
                response = manager.client.table("students").select("*").execute()
                students_data = response.data or []
                
                cache = {}
                slug_to_uuid = {}
                for row in students_data:
                    slug = row["student_id"]
                    uuid_str = row["id"]
                    reg_date = row.get("registration_date") or row.get("created_at")
                    if reg_date:
                        # Make sure ISO format is uniform
                        try:
                            reg_date = datetime.fromisoformat(reg_date.replace("Z", "+00:00")).isoformat()
                        except ValueError:
                            pass
                    else:
                        reg_date = datetime.now().isoformat()
                    
                    cache[slug] = {
                        "user_id": slug,
                        "name": row["name"],
                        "created_at": reg_date,
                        "avatar_url": row.get("avatar_url"),
                        "active": row.get("active", True),
                    }
                    slug_to_uuid[slug] = uuid_str

                self._cache = cache
                self._slug_to_uuid = slug_to_uuid
                
                # Sync local JSON file with Supabase state as a backup
                self._write_local_backup(cache)
                return dict(cache)
            except Exception as exc:
                logger.warning("Failed to fetch students from Supabase: %s. Falling back to local file.", exc)
                manager.mark_offline()

        # Offline fallback
        cache = self._load_local_backup()
        self._cache = cache
        return dict(cache)

    def _load_local_backup(self) -> Dict[str, Dict[str, Any]]:
        if not os.path.exists(config.USERS_DB_PATH):
            return {}
        try:
            with open(config.USERS_DB_PATH, "r", encoding="utf-8") as fh:
                data = json.load(fh)
                if isinstance(data, dict):
                    return data
                return {}
        except Exception as exc:
            logger.error("Failed to read local users backup file: %s", exc)
            return {}

    def _write_local_backup(self, cache: Dict[str, Dict[str, Any]]) -> None:
        try:
            config.ensure_dirs()
            with open(config.USERS_DB_PATH, "w", encoding="utf-8") as fh:
                json.dump(cache, fh, indent=2)
        except Exception as exc:
            logger.error("Failed to write local users backup file: %s", exc)

    def save_all(self, users: Dict[str, Dict[str, Any]]) -> None:
        """Overwrite all student records."""
        with self._lock:
            self._cache = dict(users)
            self._write_local_backup(users)

            manager = get_manager()
            # Generate local sync queue entries and execute if online
            for slug, details in users.items():
                row = {
                    "student_id": slug,
                    "name": details.get("name", slug),
                    "registration_date": details.get("created_at") or datetime.now().isoformat(),
                    "avatar_url": details.get("avatar_url"),
                    "active": details.get("active", True),
                }
                sync_entry = {
                    "table": "students",
                    "operation": "upsert",
                    "data": row,
                }
                
                def op(client, r=row):
                    client.table("students").upsert(r, on_conflict="student_id").execute()
                
                def fallback():
                    pass

                manager.execute(op, fallback, sync_entry)

    def upsert(self, student_id: str, name: str, extra: Dict[str, Any] | None = None) -> None:
        """Create or update a single student."""
        with self._lock:
            if self._cache is None:
                self._load_unlocked()
            
            cache = self._cache if self._cache is not None else {}
            record = cache.get(student_id, {"created_at": datetime.now().isoformat()})
            record.update({"user_id": student_id, "name": name})
            if extra:
                record.update(extra)
            cache[student_id] = record
            self._cache = cache
            self._write_local_backup(cache)

            # Persist to Supabase
            row = {
                "student_id": student_id,
                "name": name,
                "registration_date": record.get("created_at"),
                "avatar_url": record.get("avatar_url"),
                "active": record.get("active", True),
            }
            sync_entry = {
                "table": "students",
                "operation": "upsert",
                "data": row,
            }

            def op(client, r=row):
                res = client.table("students").upsert(r, on_conflict="student_id").execute()
                # Update slug_to_uuid map with new id if returned
                if res.data and len(res.data) > 0:
                    self._slug_to_uuid[student_id] = res.data[0]["id"]

            def fallback():
                pass

            manager = get_manager()
            manager.execute(op, fallback, sync_entry)

    def delete(self, student_id: str) -> None:
        """Delete a student by slug."""
        with self._lock:
            if self._cache is None:
                self._load_unlocked()
            
            cache = self._cache if self._cache is not None else {}
            if student_id in cache:
                del cache[student_id]
            self._cache = cache
            self._write_local_backup(cache)

            uuid_val = self._slug_to_uuid.get(student_id)
            if uuid_val:
                del self._slug_to_uuid[student_id]

            # Sync delete to Supabase
            sync_entry = {
                "table": "students",
                "operation": "delete",
                "filter_col": "student_id",
                "filter_val": student_id,
            }

            def op(client, s_id=student_id):
                client.table("students").delete().eq("student_id", s_id).execute()

            def fallback():
                pass

            manager = get_manager()
            manager.execute(op, fallback, sync_entry)


_instance: Optional[StudentRepository] = None
_inst_lock = threading.Lock()


def get_student_repo() -> StudentRepository:
    """Get the singleton StudentRepository instance."""
    global _instance
    if _instance is None:
        with _inst_lock:
            if _instance is None:
                _instance = StudentRepository()
    return _instance
