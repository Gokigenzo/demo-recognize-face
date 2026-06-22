"""Unit tests for the model-building stage (SVM/KNN classifier training).

Uses isolated temp storage so the real demo artifacts are never touched, and
synthetic embeddings so the tests are fast and backend-agnostic.
"""
from __future__ import annotations

import importlib

import numpy as np
import pytest


@pytest.fixture()
def isolated_storage(tmp_path, monkeypatch):
    """Point all storage paths at a temp dir and reload dependent modules."""
    from ml import config

    monkeypatch.setattr(config, "DATASETS_DIR", str(tmp_path / "datasets"))
    monkeypatch.setattr(config, "MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setattr(config, "LOGS_DIR", str(tmp_path / "logs"))
    monkeypatch.setattr(config, "USERS_DB_PATH", str(tmp_path / "datasets" / "users.json"))
    monkeypatch.setattr(config, "EMBEDDINGS_DB_PATH", str(tmp_path / "models" / "embeddings_db.pkl"))
    monkeypatch.setattr(config, "CLASSIFIER_PATH", str(tmp_path / "models" / "classifier.pkl"))
    config.ensure_dirs()

    from ml import storage
    importlib.reload(storage)
    return config, storage


def _seed_two_people(storage, n: int = 6) -> None:
    rng = np.random.default_rng(0)
    dim = 16  # small dim is fine for the classifier tests
    a = rng.normal(2.0, 0.1, (n, dim)).astype(np.float32)
    b = rng.normal(-2.0, 0.1, (n, dim)).astype(np.float32)
    storage.upsert_user("u_a", "Alice")
    storage.upsert_user("u_b", "Bob")
    storage.add_embeddings("u_a", list(a))
    storage.add_embeddings("u_b", list(b))


def test_can_train_requires_two_classes(isolated_storage):
    config, storage = isolated_storage
    from ml import model_builder
    ok, _ = model_builder.can_train()
    assert ok is False

    _seed_two_people(storage)
    ok, _ = model_builder.can_train()
    assert ok is True


@pytest.mark.parametrize("kind", ["SVM", "KNN"])
def test_train_produces_separable_model(isolated_storage, kind):
    config, storage = isolated_storage
    from ml import model_builder

    _seed_two_people(storage)
    model = model_builder.train(kind, persist=True)

    assert model.kind == kind
    assert len(model.class_names) == 2
    # Well-separated synthetic clusters should be perfectly learnable.
    assert model.train_accuracy == pytest.approx(1.0, abs=1e-6)
    assert storage.load_classifier() is not None


def test_decision_boundary_mesh_shape(isolated_storage):
    config, storage = isolated_storage
    from ml import model_builder

    _seed_two_people(storage)
    model = model_builder.train("SVM", persist=False)
    xx, yy, zz = model_builder.decision_boundary_mesh(model, resolution=30)
    assert xx.shape == (30, 30)
    assert zz.shape == (30, 30)
    # Predicted class indices must be within the valid range.
    assert set(np.unique(zz)).issubset({0, 1})


def test_predict_uses_persisted_classifier(isolated_storage):
    config, storage = isolated_storage
    from ml import model_builder

    _seed_two_people(storage)
    model_builder.train("SVM", persist=True)

    rng = np.random.default_rng(1)
    alice_like = rng.normal(2.0, 0.1, 16).astype(np.float32)
    out = model_builder.predict(alice_like)
    assert out is not None
    assert out["name"] == "Alice"
    assert pytest.approx(sum(out["probabilities"].values()), rel=1e-3) == 1.0


def test_predict_without_training_returns_none(isolated_storage):
    config, storage = isolated_storage
    from ml import model_builder
    assert model_builder.predict(np.zeros(16, dtype=np.float32)) is None


# ---------------------------------------------------------------------------
# Export / import bundle round-trip
# ---------------------------------------------------------------------------
def test_export_import_roundtrip(isolated_storage):
    config, storage = isolated_storage
    from ml import model_builder

    _seed_two_people(storage)
    model_builder.train("SVM", persist=True)
    blob = storage.export_bundle()

    # Wipe everything, then restore from the exported bundle.
    storage.reset_all()
    assert storage.load_embeddings_db() == {}
    assert storage.load_classifier() is None

    summary = storage.import_bundle(blob, replace=True)
    assert summary["users"] == 2
    assert summary["has_classifier"] is True
    assert len(storage.load_users()) == 2
    assert len(storage.load_embeddings_db()) == 2
    assert storage.load_classifier() is not None


def test_import_rejects_garbage(isolated_storage):
    config, storage = isolated_storage
    with pytest.raises(ValueError):
        storage.import_bundle(b"not a pickle bundle")


def test_import_merge_appends_embeddings(isolated_storage):
    config, storage = isolated_storage
    _seed_two_people(storage)
    blob = storage.export_bundle()
    before = sum(len(v) for v in storage.load_embeddings_db().values())
    summary = storage.import_bundle(blob, replace=False)
    after = sum(len(v) for v in storage.load_embeddings_db().values())
    assert after == before * 2
    assert summary["users"] == 2
