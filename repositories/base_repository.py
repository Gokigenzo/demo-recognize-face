"""Base repository infrastructure: Supabase connection, offline queue, sync.

Provides :class:`SupabaseManager` — a thread-safe singleton that owns the
Supabase client, tracks online/offline state, and manages a persistent queue
of operations that failed while offline so they can be replayed when the
connection is restored.

Other repositories import :func:`get_manager` to obtain the shared instance.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, TypeVar

from dotenv import load_dotenv

# Load .env *before* anything reads env vars.
load_dotenv()

logger = logging.getLogger(__name__)

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Paths used by the offline sync queue
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LOGS_DIR = os.path.join(_PROJECT_ROOT, "logs")
_PENDING_SYNC_PATH = os.path.join(_LOGS_DIR, "pending_sync.json")


# ---------------------------------------------------------------------------
# Pending-sync queue (thread-safe, persisted to disk)
# ---------------------------------------------------------------------------
class PendingSyncQueue:
    """Stores operations that could not be sent to Supabase.

    Each entry is a dict with at least:
    - ``table``: target Supabase table name
    - ``operation``: one of ``insert``, ``upsert``, ``delete``
    - ``data``: the row payload (for insert/upsert) or filter dict (delete)
    - ``queued_at``: ISO timestamp
    """

    def __init__(self, path: Optional[str] = None) -> None:
        self._path = path or _PENDING_SYNC_PATH
        self._lock = threading.Lock()
        self._queue: List[Dict[str, Any]] = self._load()


    # -- persistence ---------------------------------------------------------
    def _load(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self._path):
            return []
        try:
            with open(self._path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
                return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError):
            return []

    def _persist(self) -> None:
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as fh:
            json.dump(self._queue, fh, indent=2, default=str)

    # -- public API ----------------------------------------------------------
    def add(self, entry: Dict[str, Any]) -> None:
        with self._lock:
            entry.setdefault("queued_at", datetime.now().isoformat())
            self._queue.append(entry)
            self._persist()

    def drain(self) -> List[Dict[str, Any]]:
        """Remove and return all queued entries."""
        with self._lock:
            items = list(self._queue)
            self._queue.clear()
            self._persist()
            return items

    def peek(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._queue)

    def __len__(self) -> int:
        with self._lock:
            return len(self._queue)

    @property
    def is_empty(self) -> bool:
        return len(self) == 0


# ---------------------------------------------------------------------------
# Supabase Manager (singleton)
# ---------------------------------------------------------------------------
class SupabaseManager:
    """Thread-safe singleton managing the Supabase client and offline state.

    Usage::

        mgr = get_manager()
        if mgr.is_online:
            result = mgr.client.table("students").select("*").execute()
    """

    _instance: Optional["SupabaseManager"] = None
    _init_lock = threading.Lock()

    def __new__(cls) -> "SupabaseManager":
        with cls._init_lock:
            if cls._instance is None:
                inst = super().__new__(cls)
                inst._initialized = False
                cls._instance = inst
            return cls._instance

    def __init__(self) -> None:
        if self._initialized:  # type: ignore[has-type]
            return
        self._initialized = True

        self._url: str = os.getenv("SUPABASE_URL", "")
        self._key: str = os.getenv("SUPABASE_KEY", "")
        self._client: Any = None  # supabase.Client
        self._is_online: bool = False
        self._lock = threading.Lock()
        self.sync_queue = PendingSyncQueue()

        self._sync_thread: Optional[threading.Thread] = None
        self._sync_stop = threading.Event()

        if self._url and self._key:
            self._try_connect()
        else:
            logger.info(
                "SUPABASE_URL / SUPABASE_KEY not set — running in offline mode."
            )

    # -- connection helpers ---------------------------------------------------
    def _try_connect(self) -> bool:
        """Attempt to create / verify a Supabase connection."""
        try:
            from supabase import create_client

            client = create_client(self._url, self._key)
            # Quick connectivity test — lightweight query.
            client.table("application_configuration").select("id").limit(1).execute()
            with self._lock:
                self._client = client
                self._is_online = True
            logger.info("Supabase connection established.")
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Supabase connection failed (%s). Offline mode.", exc)
            with self._lock:
                self._is_online = False
            return False

    def reconnect(self) -> bool:
        """Explicitly try to reconnect (e.g. after a transient failure)."""
        return self._try_connect()

    # -- properties -----------------------------------------------------------
    @property
    def client(self) -> Any:
        """Return the live Supabase client, or *None* when offline."""
        with self._lock:
            return self._client if self._is_online else None

    @property
    def is_online(self) -> bool:
        with self._lock:
            return self._is_online

    def mark_offline(self) -> None:
        with self._lock:
            self._is_online = False
        self._ensure_sync_thread()

    # -- execute with fallback ------------------------------------------------
    def execute(
        self,
        operation: Callable[..., T],
        fallback: Callable[..., T],
        sync_entry: Optional[Dict[str, Any]] = None,
    ) -> T:
        """Run *operation* against Supabase; on failure fall back locally.

        If *sync_entry* is provided it is enqueued so the change can be
        replayed when the connection is restored.
        """
        if self.is_online and self._client is not None:
            try:
                return operation(self._client)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Supabase operation failed (%s). Falling back.", exc)
                self.mark_offline()
                if sync_entry:
                    self.sync_queue.add(sync_entry)
                return fallback()
        else:
            if sync_entry:
                self.sync_queue.add(sync_entry)
            return fallback()

    # -- background sync thread -----------------------------------------------
    def _ensure_sync_thread(self) -> None:
        if self._sync_thread is not None and self._sync_thread.is_alive():
            return
        self._sync_stop.clear()
        t = threading.Thread(target=self._sync_loop, daemon=True, name="supabase-sync")
        t.start()
        self._sync_thread = t

    def _sync_loop(self) -> None:
        """Periodically try to reconnect and flush the pending queue."""
        while not self._sync_stop.is_set():
            time.sleep(30)
            if self.sync_queue.is_empty and self.is_online:
                continue
            if not self.is_online:
                if not self.reconnect():
                    continue
            # Connection restored — flush the queue.
            self._flush_queue()

    def _flush_queue(self) -> None:
        """Replay queued operations against Supabase."""
        entries = self.sync_queue.drain()
        if not entries:
            return
        logger.info("Flushing %d pending sync entries …", len(entries))
        failed: List[Dict[str, Any]] = []
        for entry in entries:
            try:
                self._replay(entry)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Replay failed for %s: %s", entry.get("table"), exc)
                failed.append(entry)
        # Re-queue anything that still failed.
        for f in failed:
            self.sync_queue.add(f)

    def _replay(self, entry: Dict[str, Any]) -> None:
        """Execute a single queued operation against Supabase."""
        client = self.client
        if client is None:
            raise RuntimeError("Not connected")
        table = entry["table"]
        op = entry["operation"]
        data = entry.get("data", {})
        if op == "insert":
            client.table(table).insert(data).execute()
        elif op == "upsert":
            client.table(table).upsert(data).execute()
        elif op == "delete":
            col = entry.get("filter_col", "id")
            val = entry.get("filter_val")
            if val is not None:
                client.table(table).delete().eq(col, val).execute()
        else:
            logger.warning("Unknown sync operation: %s", op)

    # -- cleanup --------------------------------------------------------------
    def shutdown(self) -> None:
        """Stop the background sync thread (call on app shutdown)."""
        self._sync_stop.set()
        if self._sync_thread is not None:
            self._sync_thread.join(timeout=5)

    # -- testing helpers ------------------------------------------------------
    @classmethod
    def _reset(cls) -> None:
        """Destroy the singleton (for unit tests only)."""
        global _manager
        if cls._instance is not None:
            cls._instance.shutdown()
        cls._instance = None
        _manager = None



# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------
_manager: Optional[SupabaseManager] = None
_mgr_lock = threading.Lock()


def get_manager() -> SupabaseManager:
    """Return the shared :class:`SupabaseManager` instance."""
    global _manager
    if _manager is None:
        with _mgr_lock:
            if _manager is None:
                _manager = SupabaseManager()
    return _manager
