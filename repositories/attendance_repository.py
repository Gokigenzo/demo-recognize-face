"""Attendance repository implementation.

Logs and retrieves attendance records, fallback to local CSV, and supports
uploading CSV reports to Supabase Storage.
"""
from __future__ import annotations

import csv
import logging
import os
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional

from ml import config
from repositories.base_repository import get_manager
from repositories.student_repository import get_student_repo

logger = logging.getLogger(__name__)


class AttendanceRepository:
    """Manages attendance logs in Supabase and local CSV fallback."""

    def __init__(self) -> None:
        self._cache: Optional[List[Dict[str, Any]]] = None
        self._lock = threading.Lock()
        self._fields = ["timestamp", "user_id", "name", "confidence", "status"]

    def invalidate_cache(self) -> None:
        with self._lock:
            self._cache = None

    def append(
        self, user_id: str, name: str, confidence: float, status: str = "present", session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Log a single attendance event.

        Returns the record in the legacy dict format.
        """
        timestamp = datetime.now().isoformat()
        row = {
            "timestamp": timestamp,
            "user_id": user_id,
            "name": name,
            "confidence": round(float(confidence), 4),
            "status": status,
        }

        with self._lock:
            if self._cache is not None:
                self._cache.append(row)
            self._append_local(row)

        # Resolve student UUID in Supabase
        student_uuid = get_student_repo().get_uuid(user_id)
        
        # Prepare Supabase insert payload
        db_row = {
            "timestamp": timestamp,
            "confidence": round(float(confidence), 4),
            "recognition_method": "InsightFace",
            "status": status,
            "session_id": session_id,
            "duplicate_detected": False,
        }
        if student_uuid:
            db_row["student_id"] = student_uuid

        sync_entry = {
            "table": "attendance",
            "operation": "insert",
            "data": db_row,
            "student_slug": user_id,  # Helper for resolving UUID during replay
        }

        def op(client, r=db_row, slug=user_id):
            if "student_id" not in r and slug != "Unknown":
                uuid_val = get_student_repo().get_uuid(slug)
                if uuid_val:
                    r["student_id"] = uuid_val
            client.table("attendance").insert(r).execute()

        def fallback():
            pass

        manager = get_manager()
        manager.execute(op, fallback, sync_entry)

        return row

    def _append_local(self, row: Dict[str, Any]) -> None:
        try:
            config.ensure_dirs()
            file_exists = os.path.exists(config.ATTENDANCE_LOG_PATH)
            
            # Format row timestamp for human inspection (CSV uses ISO/friendly representation)
            csv_row = dict(row)
            try:
                # Convert ISO string to slightly simplified format if possible, matching legacy
                dt = datetime.fromisoformat(row["timestamp"])
                csv_row["timestamp"] = dt.isoformat(timespec="seconds")
            except ValueError:
                pass

            with open(config.ATTENDANCE_LOG_PATH, "a", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=self._fields)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(csv_row)
        except Exception as exc:
            logger.error("Failed to write to local attendance CSV: %s", exc)

    def load_all(self) -> List[Dict[str, Any]]:
        """Load all attendance records as a list of dictionaries."""
        with self._lock:
            if self._cache is not None:
                return list(self._cache)

            manager = get_manager()
            if manager.is_online and manager.client is not None:
                try:
                    # Query attendance joining with students
                    response = (
                        manager.client.table("attendance")
                        .select("*, students(student_id, name)")
                        .order("timestamp")
                        .execute()
                    )
                    rows = response.data or []
                    
                    cache = []
                    for r in rows:
                        student_info = r.get("students")
                        slug = student_info.get("student_id") if student_info else "Unknown"
                        display_name = student_info.get("name") if student_info else "Unknown"
                        
                        reg_date = r["timestamp"]
                        try:
                            reg_date = datetime.fromisoformat(reg_date.replace("Z", "+00:00")).isoformat()
                        except ValueError:
                            pass

                        cache.append({
                            "timestamp": reg_date,
                            "user_id": slug,
                            "name": display_name,
                            "confidence": r["confidence"],
                            "status": r["status"],
                        })

                    self._cache = cache
                    # Write to local file as backup/sync
                    self._overwrite_local(cache)
                    return list(cache)
                except Exception as exc:
                    logger.warning("Failed to load attendance from Supabase: %s. Loading local CSV.", exc)
                    manager.mark_offline()

            # Offline fallback
            cache = self._load_local()
            self._cache = cache
            return list(cache)

    def _load_local(self) -> List[Dict[str, Any]]:
        if not os.path.exists(config.ATTENDANCE_LOG_PATH):
            return []
        try:
            with open(config.ATTENDANCE_LOG_PATH, "r", newline="", encoding="utf-8") as fh:
                return list(csv.DictReader(fh))
        except Exception as exc:
            logger.error("Failed to read local attendance CSV: %s", exc)
            return []

    def _overwrite_local(self, records: List[Dict[str, Any]]) -> None:
        try:
            config.ensure_dirs()
            with open(config.ATTENDANCE_LOG_PATH, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=self._fields)
                writer.writeheader()
                for r in records:
                    # Format timestamp
                    csv_row = dict(r)
                    try:
                        dt = datetime.fromisoformat(r["timestamp"])
                        csv_row["timestamp"] = dt.isoformat(timespec="seconds")
                    except ValueError:
                        pass
                    writer.writerow(csv_row)
        except Exception as exc:
            logger.error("Failed to overwrite local CSV backup: %s", exc)

    def upload_csv_report(self, csv_bytes: bytes, filename: str) -> Optional[str]:
        """Upload an attendance report CSV file to Supabase Storage.

        Returns the public URL if successful.
        """
        manager = get_manager()
        if not manager.is_online or manager.client is None:
            return None

        try:
            # Check or create bucket
            try:
                manager.client.storage.create_bucket("attendance", options={"public": True})
            except Exception:
                pass  # Bucket might already exist
            
            # Upload CSV
            res = manager.client.storage.from_("attendance").upload(
                filename,
                csv_bytes,
                file_options={"content-type": "text/csv", "upsert": "true"},
            )
            
            # Get public URL
            if res:
                return manager.client.storage.from_("attendance").get_public_url(filename)
        except Exception as exc:
            logger.warning("Failed to upload attendance CSV report to storage: %s", exc)
        
        return None


_instance: Optional[AttendanceRepository] = None
_inst_lock = threading.Lock()


def get_attendance_repo() -> AttendanceRepository:
    """Get the singleton AttendanceRepository instance."""
    global _instance
    if _instance is None:
        with _inst_lock:
            if _instance is None:
                _instance = AttendanceRepository()
    return _instance
