"""Configuration repository implementation.

Loads and saves runtime application configuration parameters to/from Supabase,
falling back to default ml.config constants when offline.
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Dict, Optional

from ml import config
from repositories.base_repository import get_manager

logger = logging.getLogger(__name__)


class ConfigurationRepository:
    """Manages application settings in Supabase with local default fallback."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cache: Optional[Dict[str, Any]] = None
        self._defaults = {
            "recognition_threshold": config.DEFAULT_SIMILARITY_THRESHOLD,
            "confirmation_frames": 5,
            "detection_interval": config.REALTIME_DETECTION_INTERVAL,
            "camera_resolution_width": config.REALTIME_CAMERA_WIDTH,
            "camera_resolution_height": config.REALTIME_CAMERA_HEIGHT,
            "recognition_fps": config.REALTIME_TARGET_FPS,
            "ui_preferences": {},
        }

    def invalidate_cache(self) -> None:
        with self._lock:
            self._cache = None

    def load(self) -> Dict[str, Any]:
        """Load the 'default' configuration settings."""
        with self._lock:
            if self._cache is not None:
                return dict(self._cache)

            manager = get_manager()
            if manager.is_online and manager.client is not None:
                try:
                    response = (
                        manager.client.table("application_configuration")
                        .select("*")
                        .eq("id", "default")
                        .execute()
                    )
                    rows = response.data or []
                    if rows:
                        db_row = rows[0]
                        # Map DB columns to our config dictionary keys
                        cfg = {
                            "recognition_threshold": db_row.get("recognition_threshold", self._defaults["recognition_threshold"]),
                            "confirmation_frames": db_row.get("confirmation_frames", self._defaults["confirmation_frames"]),
                            "detection_interval": db_row.get("detection_interval", self._defaults["detection_interval"]),
                            "camera_resolution_width": db_row.get("camera_resolution_width", self._defaults["camera_resolution_width"]),
                            "camera_resolution_height": db_row.get("camera_resolution_height", self._defaults["camera_resolution_height"]),
                            "recognition_fps": db_row.get("recognition_fps", self._defaults["recognition_fps"]),
                            "ui_preferences": db_row.get("ui_preferences", self._defaults["ui_preferences"]),
                        }
                        self._cache = cfg
                        return dict(cfg)
                except Exception as exc:
                    logger.warning("Failed to load configuration from Supabase: %s. Using default config.", exc)
                    manager.mark_offline()

            # Offline/default fallback
            return dict(self._defaults)

    def save(self, config_data: Dict[str, Any]) -> None:
        """Upsert application configuration."""
        with self._lock:
            self._cache = dict(config_data)

            # Map config dict keys to DB columns
            db_row = {
                "id": "default",
                "recognition_threshold": float(config_data.get("recognition_threshold", self._defaults["recognition_threshold"])),
                "confirmation_frames": int(config_data.get("confirmation_frames", self._defaults["confirmation_frames"])),
                "detection_interval": int(config_data.get("detection_interval", self._defaults["detection_interval"])),
                "camera_resolution_width": int(config_data.get("camera_resolution_width", self._defaults["camera_resolution_width"])),
                "camera_resolution_height": int(config_data.get("camera_resolution_height", self._defaults["camera_resolution_height"])),
                "recognition_fps": float(config_data.get("recognition_fps", self._defaults["recognition_fps"])),
                "ui_preferences": config_data.get("ui_preferences", self._defaults["ui_preferences"]),
                "updated_at": __import__("datetime").datetime.now().isoformat(),
            }

            sync_entry = {
                "table": "application_configuration",
                "operation": "upsert",
                "data": db_row,
            }

            def op(client, r=db_row):
                client.table("application_configuration").upsert(r).execute()

            def fallback():
                pass

            manager = get_manager()
            manager.execute(op, fallback, sync_entry)


_instance: Optional[ConfigurationRepository] = None
_inst_lock = threading.Lock()


def get_configuration_repo() -> ConfigurationRepository:
    """Get the singleton ConfigurationRepository instance."""
    global _instance
    if _instance is None:
        with _inst_lock:
            if _instance is None:
                _instance = ConfigurationRepository()
    return _instance
