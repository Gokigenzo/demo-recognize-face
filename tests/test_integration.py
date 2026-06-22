"""Integration tests: registration → DB update → attendance → feedback.

Uses a temporary working directory (monkeypatched config paths) so tests
never touch real demo artifacts.
"""
from __future__ import annotations

import importlib

import numpy as np
import pytest

from ml import config


@pytest.fixture()
def isolated_storage(tmp_path, monkeypatch):
    """Point all persistence paths at a temp dir and reload storage."""
    monkeypatch.setattr(config, "DATASETS_DIR", str(tmp_path / "datasets"))
    monkeypatch.setattr(config, "MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setattr(config, "LOGS_DIR", str(tmp_path / "logs"))
    monkeypatch.setattr(config, "EMBEDDINGS_DB_PATH", str(tmp_path / "models" / "db.pkl"))
    monkeypatch.setattr(config, "USERS_DB_PATH", str(tmp_path / "datasets" / "users.json"))
    monkeypatch.setattr(config, "ATTENDANCE_LOG_PATH", str(tmp_path / "logs" / "att.csv"))
    monkeypatch.setattr(config, "FEEDBACK_LOG_PATH", str(tmp_path / "logs" / "fb.json"))
    monkeypatch.setattr(config, "MONITORING_LOG_PATH", str(tmp_path / "logs" / "mon.json"))
    config.ensure_dirs()

    from ml import storage
    importlib.reload(storage)
    return storage


def _unit_vec(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=config.EMBEDDING_DIM).astype(np.float32)
    return v / np.linalg.norm(v)


# ---------------------------------------------------------------------------
# Registration pipeline
# ---------------------------------------------------------------------------
def test_registration_pipeline(isolated_storage):
    storage = isolated_storage
    storage.upsert_user("ada", "Ada Lovelace")
    storage.add_embeddings("ada", [_unit_vec(1), _unit_vec(2)])

    users = storage.load_users()
    db = storage.load_embeddings_db()
    assert users["ada"]["name"] == "Ada Lovelace"
    assert len(db["ada"]) == 2


# ---------------------------------------------------------------------------
# Embedding database update
# ---------------------------------------------------------------------------
def test_embedding_db_update_appends(isolated_storage):
    storage = isolated_storage
    storage.add_embeddings("grace", [_unit_vec(3)])
    storage.add_embeddings("grace", [_unit_vec(4)])
    assert len(storage.load_embeddings_db()["grace"]) == 2


# ---------------------------------------------------------------------------
# Attendance pipeline
# ---------------------------------------------------------------------------
def test_attendance_pipeline_known_person(isolated_storage):
    storage = isolated_storage
    from ml import attendance_engine
    importlib.reload(attendance_engine)

    emb = _unit_vec(5)
    storage.upsert_user("turing", "Alan Turing")
    storage.add_embeddings("turing", [emb])

    # Identical embedding → high similarity → known.
    result = attendance_engine.recognize_and_log(emb, threshold=0.45, log=True)
    assert result.is_known
    assert result.user_id == "turing"
    assert len(storage.load_attendance()) == 1


def test_attendance_pipeline_unknown_person(isolated_storage):
    storage = isolated_storage
    from ml import attendance_engine
    importlib.reload(attendance_engine)

    storage.upsert_user("turing", "Alan Turing")
    storage.add_embeddings("turing", [_unit_vec(5)])

    # An orthogonal-ish random vector should fall below threshold.
    result = attendance_engine.recognize_and_log(_unit_vec(999), threshold=0.6, log=True)
    assert not result.is_known
    assert result.name == "Unknown"
    assert storage.load_attendance() == []


# ---------------------------------------------------------------------------
# Feedback loop closes
# ---------------------------------------------------------------------------
def test_feedback_loop_adds_data(isolated_storage):
    storage = isolated_storage
    from ml import attendance_engine, feedback_engine
    importlib.reload(attendance_engine)
    importlib.reload(feedback_engine)

    new_emb = _unit_vec(42)
    feedback_engine.record_correction(new_emb, "ada", "Ada Lovelace", "Unknown")

    db = storage.load_embeddings_db()
    assert "ada" in db and len(db["ada"]) == 1
    assert len(storage.load_feedback()) == 1

    # After correction the same embedding is now recognized.
    result = attendance_engine.identify(new_emb, threshold=0.45)
    assert result.is_known and result.user_id == "ada"
