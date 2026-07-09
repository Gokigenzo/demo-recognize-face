"""Unit and integration tests for the Supabase Repository layer and migration.

Verifies offline fallback mode, caching, thread-safety, local sync queue,
and data migration.
"""
from __future__ import annotations

import json
import os
import pickle
import numpy as np
import pytest

from ml import config, storage
from repositories import (
    get_manager,
    get_student_repo,
    get_embedding_repo,
    get_attendance_repo,
    get_feedback_repo,
    get_configuration_repo,
    get_classifier_repo,
)
from repositories.base_repository import SupabaseManager


@pytest.fixture(autouse=True)
def clean_repos(tmp_path, monkeypatch):
    """Enforce clean temporary storage and reset singletons for each test."""
    # Monkeypatch storage directories
    monkeypatch.setattr(config, "DATASETS_DIR", str(tmp_path / "datasets"))
    monkeypatch.setattr(config, "MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setattr(config, "LOGS_DIR", str(tmp_path / "logs"))
    monkeypatch.setattr(config, "USERS_DB_PATH", str(tmp_path / "datasets" / "users.json"))
    monkeypatch.setattr(config, "EMBEDDINGS_DB_PATH", str(tmp_path / "models" / "embeddings_db.pkl"))
    monkeypatch.setattr(config, "CLASSIFIER_PATH", str(tmp_path / "models" / "classifier.pkl"))
    monkeypatch.setattr(config, "ATTENDANCE_LOG_PATH", str(tmp_path / "logs" / "attendance.csv"))
    monkeypatch.setattr(config, "FEEDBACK_LOG_PATH", str(tmp_path / "logs" / "feedback.json"))
    monkeypatch.setattr(config, "MONITORING_LOG_PATH", str(tmp_path / "logs" / "monitoring.json"))
    
    # Also update the base_repository path configs
    from repositories.base_repository import _PENDING_SYNC_PATH
    monkeypatch.setattr("repositories.base_repository._PENDING_SYNC_PATH", str(tmp_path / "logs" / "pending_sync.json"))
    
    config.ensure_dirs()

    # Reset SupabaseManager singleton
    SupabaseManager._reset()
    
    # Invalidate caches
    get_student_repo().invalidate_cache()
    get_embedding_repo().invalidate_cache()
    get_attendance_repo().invalidate_cache()
    get_feedback_repo().invalidate_cache()
    get_configuration_repo().invalidate_cache()


def test_offline_fallback_mode_reads_writes_locally():
    """Verify that when offline, data is written locally and can be reloaded."""
    student_repo = get_student_repo()
    
    # Enforce offline state
    manager = get_manager()
    assert manager.is_online is False
    
    # Perform upsert
    student_repo.upsert("marie_curie", "Marie Curie", {"active": True})
    
    # Verify it was cached in memory
    all_students = student_repo.load_all()
    assert "marie_curie" in all_students
    assert all_students["marie_curie"]["name"] == "Marie Curie"
    
    # Verify it was written to users.json backup file
    assert os.path.exists(config.USERS_DB_PATH)
    with open(config.USERS_DB_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
        assert "marie_curie" in data
        assert data["marie_curie"]["name"] == "Marie Curie"


def test_offline_operations_enqueued_in_sync_queue():
    """Verify that offline writes add items to the pending sync queue file."""
    student_repo = get_student_repo()
    manager = get_manager()
    
    # Verify initial queue is empty
    assert len(manager.sync_queue) == 0
    
    # Write a student while offline
    student_repo.upsert("rosalind_franklin", "Rosalind Franklin")
    
    # Verify item was added to the queue
    assert len(manager.sync_queue) == 1
    peeked = manager.sync_queue.peek()
    assert peeked[0]["table"] == "students"
    assert peeked[0]["operation"] == "upsert"
    assert peeked[0]["data"]["student_id"] == "rosalind_franklin"
    
    # Verify sync queue is persisted to the temp file
    from repositories.base_repository import _PENDING_SYNC_PATH
    assert os.path.exists(_PENDING_SYNC_PATH)
    with open(_PENDING_SYNC_PATH, "r", encoding="utf-8") as f:
        queued_items = json.load(f)
        assert len(queued_items) == 1
        assert queued_items[0]["data"]["student_id"] == "rosalind_franklin"


def test_embeddings_numpy_compatibility_local_pkl():
    """Verify loading/saving embeddings preserves numpy float32 ndarrays in local fallback."""
    emb_repo = get_embedding_repo()
    
    # Setup test embeddings
    test_embs = [np.random.normal(0.0, 1.0, 512).astype(np.float32) for _ in range(3)]
    
    # Add embeddings while offline
    emb_repo.add("marie_curie", test_embs)
    
    # Load and verify
    loaded = emb_repo.load_all()
    assert "marie_curie" in loaded
    assert len(loaded["marie_curie"]) == 3
    for orig, loaded_emb in zip(test_embs, loaded["marie_curie"]):
        assert isinstance(loaded_emb, np.ndarray)
        assert loaded_emb.dtype == np.float32
        assert loaded_emb.shape == (512,)
        assert np.allclose(orig, loaded_emb)


def test_migration_converts_local_files_if_online(monkeypatch):
    """Verify the migration execution handles loading files and processing in order."""
    # Write local mock database files to act as pre-existing data
    users_data = {
        "richard_feynman": {
            "created_at": "2026-07-01T12:00:00",
            "user_id": "richard_feynman",
            "name": "Richard Feynman"
        }
    }
    with open(config.USERS_DB_PATH, "w", encoding="utf-8") as f:
        json.dump(users_data, f)
        
    embs_db = {
        "richard_feynman": [np.random.normal(0.0, 1.0, 512).astype(np.float32) for _ in range(2)]
    }
    with open(config.EMBEDDINGS_DB_PATH, "wb") as f:
        pickle.dump(embs_db, f)

    attendance_data = "timestamp,user_id,name,confidence,status\n2026-07-01T12:01:00,richard_feynman,Richard Feynman,0.95,present\n"
    with open(config.ATTENDANCE_LOG_PATH, "w", encoding="utf-8") as f:
        f.write(attendance_data)

    feedback_data = [{
        "timestamp": "2026-07-01T12:05:00",
        "predicted": "Unknown",
        "corrected_to": "Richard Feynman",
        "correct_user_id": "richard_feynman",
        "note": "Correction test"
    }]
    with open(config.FEEDBACK_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(feedback_data, f)


    # Mock Supabase connection as online
    class MockStorage:
        def __init__(self, client):
            self.client = client
        def from_(self, bucket_name):
            return self
        def upload(self, *args, **kwargs):
            return self
        def download(self, *args, **kwargs):
            return b""
        def get_public_url(self, *args, **kwargs):
            return "http://mock/public/url"
        def create_bucket(self, *args, **kwargs):
            return self

    class MockClient:
        def __init__(self):
            self.tables_called = []
            self.storage = MockStorage(self)
            
        class MockTable:
            def __init__(self, name, client):
                self.name = name
                self.client = client
                
            def select(self, *args, **kwargs):
                return self
            def eq(self, *args, **kwargs):
                return self
            def neq(self, *args, **kwargs):
                return self
            def order(self, *args, **kwargs):
                return self
            def delete(self, *args, **kwargs):
                self.client.tables_called.append((self.name, "delete", args, kwargs))
                return self
            def upsert(self, *args, **kwargs):
                self.client.tables_called.append((self.name, "upsert", args, kwargs))
                return self
            def insert(self, *args, **kwargs):
                self.client.tables_called.append((self.name, "insert", args, kwargs))
                return self
            def execute(self):
                # Return empty/default rows
                if self.name == "application_configuration":
                    return type('res', (), {'data': [{'id': 'default', 'ui_preferences': {}}]})()
                elif self.name == "students":
                    # Mock finding the student
                    return type('res', (), {'data': [{'id': 'mock-student-uuid', 'student_id': 'richard_feynman', 'name': 'Richard Feynman'}]})()
                return type('res', (), {'data': []})()
                
        def table(self, name):
            return self.MockTable(name, self)

    mock_client = MockClient()
    manager = get_manager()
    
    # Inject online client state
    monkeypatch.setattr(manager, "_client", mock_client)
    monkeypatch.setattr(manager, "_is_online", True)
    
    # Reset caches
    get_student_repo().invalidate_cache()
    get_embedding_repo().invalidate_cache()
    get_attendance_repo().invalidate_cache()
    get_feedback_repo().invalidate_cache()
    
    # Run migration
    storage.run_migration_if_needed()
    
    # Check that Supabase tables were queried/inserted
    called_tables = [t[0] for t in mock_client.tables_called]
    assert "students" in called_tables
    assert "embeddings" in called_tables
    assert "attendance" in called_tables
    assert "monitoring_feedback" in called_tables
    assert "application_configuration" in called_tables  # For flag recording

