"""Embedding repository implementation.

Manages student face embeddings, pgvector serialization, in-memory caching,
and offline fallback to local embeddings pickle database.
"""
from __future__ import annotations

import json
import logging
import os
import pickle
import threading
from typing import Any, Dict, List, Optional

import numpy as np

from ml import config
from repositories.base_repository import get_manager
from repositories.student_repository import get_student_repo

logger = logging.getLogger(__name__)


class EmbeddingRepository:
    """Manages face embeddings with pgvector storage and in-memory cache."""

    def __init__(self) -> None:
        self._cache: Optional[Dict[str, List[np.ndarray]]] = None
        self._lock = threading.Lock()

    def invalidate_cache(self) -> None:
        with self._lock:
            self._cache = None

    def load_all(self) -> Dict[str, List[np.ndarray]]:
        """Return the embeddings database in the legacy format:

        {
            "student_id_slug": [np.ndarray, np.ndarray, ...]
        }
        """
        with self._lock:
            return self._load_unlocked()

    def _load_unlocked(self) -> Dict[str, List[np.ndarray]]:
        if self._cache is not None:
            return {k: list(v) for k, v in self._cache.items()}

        manager = get_manager()
        if manager.is_online and manager.client is not None:
            try:
                # Query embeddings with joining students to get the student_id slug
                response = manager.client.table("embeddings").select("*, students!inner(student_id)").execute()
                rows = response.data or []
                
                cache: Dict[str, List[np.ndarray]] = {}
                for row in rows:
                    slug = row["students"]["student_id"]
                    vec_data = row["embedding_vector"]
                    
                    if isinstance(vec_data, str):
                        # pgvector can return string "[0.12,0.44,...]"
                        # Clean bracket symbols if necessary and parse
                        cleaned = vec_data.strip("[]")
                        vec = np.fromstring(cleaned, sep=",", dtype=np.float32)
                    elif isinstance(vec_data, list):
                        vec = np.array(vec_data, dtype=np.float32)
                    else:
                        logger.error("Unexpected embedding vector type: %s", type(vec_data))
                        continue
                    
                    cache.setdefault(slug, []).append(vec)

                self._cache = cache
                # Save a local backup copy
                self._write_local_backup(cache)
                return {k: list(v) for k, v in cache.items()}
            except Exception as exc:
                logger.warning("Failed to fetch embeddings from Supabase: %s. Falling back to local file.", exc)
                manager.mark_offline()

        # Offline fallback
        cache = self._load_local_backup()
        self._cache = cache
        return {k: list(v) for k, v in cache.items()}

    def _load_local_backup(self) -> Dict[str, List[np.ndarray]]:
        if not os.path.exists(config.EMBEDDINGS_DB_PATH):
            return {}
        try:
            with open(config.EMBEDDINGS_DB_PATH, "rb") as fh:
                data = pickle.load(fh)
                if isinstance(data, dict):
                    # Ensure numpy arrays are of float32
                    return {k: [np.asarray(e, dtype=np.float32) for e in v] for k, v in data.items()}
                return {}
        except Exception as exc:
            logger.error("Failed to read local embeddings pickle: %s", exc)
            return {}

    def _write_local_backup(self, cache: Dict[str, List[np.ndarray]]) -> None:
        try:
            config.ensure_dirs()
            with open(config.EMBEDDINGS_DB_PATH, "wb") as fh:
                pickle.dump(cache, fh)
        except Exception as exc:
            logger.error("Failed to write local embeddings pickle: %s", exc)

    def save_all(self, db: Dict[str, List[np.ndarray]]) -> None:
        """Overwrite the entire embeddings database."""
        with self._lock:
            self._cache = {k: list(v) for k, v in db.items()}
            self._write_local_backup(db)

            manager = get_manager()
            if not manager.is_online:
                # Store a clear operation and rebuild in sync queue
                # For simplicity, we enqueue deletion and inserts.
                pass
            
            # Rebuild all embeddings in Supabase
            def op(client):
                # 1. Delete all existing embeddings
                client.table("embeddings").delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()
                
                # 2. Insert all
                rows_to_insert = []
                student_repo = get_student_repo()
                for slug, embs in db.items():
                    uuid_val = student_repo.get_uuid(slug)
                    if not uuid_val:
                        continue
                    for emb in embs:
                        rows_to_insert.append({
                            "student_id": uuid_val,
                            "embedding_vector": emb.tolist(),
                            "pose": "Unknown",
                            "quality_score": 1.0,
                            "embedding_source": "SyncAll",
                        })
                if rows_to_insert:
                    # Batch inserts
                    client.table("embeddings").insert(rows_to_insert).execute()

            def fallback():
                pass

            # Since save_all is a heavy reset/restore operation, run it directly
            # or record it.
            manager.execute(op, fallback)

    def add(self, student_id: str, embeddings: List[np.ndarray], pose: Optional[str] = None, quality_score: Optional[float] = None) -> None:
        """Append embeddings for a student."""
        with self._lock:
            if self._cache is None:
                self._load_unlocked()
            
            cache = self._cache if self._cache is not None else {}
            cache.setdefault(student_id, [])
            
            new_embs = [np.asarray(e, dtype=np.float32) for e in embeddings]
            cache[student_id].extend(new_embs)
            self._cache = cache
            self._write_local_backup(cache)

            # Resolve student UUID
            student_uuid = get_student_repo().get_uuid(student_id)
            
            # Persist to Supabase
            manager = get_manager()
            for emb in new_embs:
                row = {
                    "pose": pose or "Unknown",
                    "quality_score": quality_score or 1.0,
                    "embedding_source": "Realtime",
                    "embedding_vector": emb.tolist(),
                }
                
                # If student_uuid is already resolved, add it.
                # Otherwise, the sync worker will have to look it up on retry.
                if student_uuid:
                    row["student_id"] = student_uuid

                sync_entry = {
                    "table": "embeddings",
                    "operation": "insert",
                    "data": row,
                    "student_slug": student_id,  # Helper field for offline sync resolution
                }

                def op(client, r=row, slug=student_id):
                    # Resolve UUID dynamic during sync/replay if not set
                    if "student_id" not in r:
                        uuid_val = get_student_repo().get_uuid(slug)
                        if not uuid_val:
                            raise RuntimeError(f"Cannot resolve student slug {slug}")
                        r["student_id"] = uuid_val
                    client.table("embeddings").insert(r).execute()

                def fallback():
                    pass

                manager.execute(op, fallback, sync_entry)


_instance: Optional[EmbeddingRepository] = None
_inst_lock = threading.Lock()


def get_embedding_repo() -> EmbeddingRepository:
    """Get the singleton EmbeddingRepository instance."""
    global _instance
    if _instance is None:
        with _inst_lock:
            if _instance is None:
                _instance = EmbeddingRepository()
    return _instance
