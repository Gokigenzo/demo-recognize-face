"""Statistics repository implementation.

Logs and retrieves in-stream recognition session statistics to/from Supabase.
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional

from repositories.base_repository import get_manager

logger = logging.getLogger(__name__)


class StatisticsRepository:
    """Manages stream session statistics in Supabase."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cache: Optional[List[Dict[str, Any]]] = None

    def invalidate_cache(self) -> None:
        with self._lock:
            self._cache = None

    def save_session(self, stats: Dict[str, Any]) -> None:
        """Insert a session statistics row.

        Args:
            stats: Dictionary containing start_time, end_time, present_count,
                   absent_count, unknown_count, average_confidence,
                   average_fps, recognition_time.
        """
        # Convert times to isoformat strings if they are floats or datetime objects
        start_time = stats.get("start_time")
        if isinstance(start_time, (int, float)):
            from datetime import datetime
            start_time = datetime.fromtimestamp(start_time).isoformat()
        elif hasattr(start_time, "isoformat"):
            start_time = start_time.isoformat()

        end_time = stats.get("end_time")
        if isinstance(end_time, (int, float)):
            from datetime import datetime
            end_time = datetime.fromtimestamp(end_time).isoformat()
        elif hasattr(end_time, "isoformat"):
            end_time = end_time.isoformat()
        elif end_time is None:
            from datetime import datetime
            end_time = datetime.now().isoformat()

        db_row = {
            "start_time": start_time,
            "end_time": end_time,
            "present_count": int(stats.get("present_count", 0)),
            "absent_count": int(stats.get("absent_count", 0)),
            "unknown_count": int(stats.get("unknown_count", 0)),
            "average_confidence": float(stats.get("average_confidence", 0.0)),
            "average_fps": float(stats.get("average_fps", 0.0)),
            "recognition_time": float(stats.get("recognition_time", 0.0)),
        }

        with self._lock:
            if self._cache is not None:
                self._cache.append(db_row)

        sync_entry = {
            "table": "session_statistics",
            "operation": "insert",
            "data": db_row,
        }

        def op(client, r=db_row):
            client.table("session_statistics").insert(r).execute()

        def fallback():
            pass

        manager = get_manager()
        manager.execute(op, fallback, sync_entry)

    def load_sessions(self) -> List[Dict[str, Any]]:
        """Load all historical session statistics."""
        with self._lock:
            if self._cache is not None:
                return list(self._cache)

            manager = get_manager()
            if manager.is_online and manager.client is not None:
                try:
                    response = manager.client.table("session_statistics").select("*").order("created_at").execute()
                    rows = response.data or []
                    self._cache = rows
                    return list(rows)
                except Exception as exc:
                    logger.warning("Failed to fetch session statistics from Supabase: %s", exc)
                    manager.mark_offline()
            
            return []


_instance: Optional[StatisticsRepository] = None
_inst_lock = threading.Lock()


def get_statistics_repo() -> StatisticsRepository:
    """Get the singleton StatisticsRepository instance."""
    global _instance
    if _instance is None:
        with _inst_lock:
            if _instance is None:
                _instance = StatisticsRepository()
    return _instance
