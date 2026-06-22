"""System tests: Known Person, Unknown Person, Mask, Glasses, Low Light.

Ensures the end-to-end recognition pipelines (similarity search and classifier)
behave correctly under simulated production conditions.
"""
from __future__ import annotations

import importlib
import numpy as np
import pytest

from ml import config


@pytest.fixture()
def isolated_system_storage(tmp_path, monkeypatch):
    """Point all persistence paths at a temp dir and reload modules to isolate state."""
    monkeypatch.setattr(config, "DATASETS_DIR", str(tmp_path / "datasets"))
    monkeypatch.setattr(config, "MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setattr(config, "LOGS_DIR", str(tmp_path / "logs"))
    monkeypatch.setattr(config, "EMBEDDINGS_DB_PATH", str(tmp_path / "models" / "db.pkl"))
    monkeypatch.setattr(config, "USERS_DB_PATH", str(tmp_path / "datasets" / "users.json"))
    monkeypatch.setattr(config, "CLASSIFIER_PATH", str(tmp_path / "models" / "clf.pkl"))
    monkeypatch.setattr(config, "ATTENDANCE_LOG_PATH", str(tmp_path / "logs" / "att.csv"))
    config.ensure_dirs()

    from ml import storage, attendance_engine, model_builder
    importlib.reload(storage)
    importlib.reload(attendance_engine)
    importlib.reload(model_builder)

    return storage, attendance_engine, model_builder


def _unit_vec(seed: int) -> np.ndarray:
    """Generate a stable synthetic embedding (L2-normalized unit vector)."""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=config.EMBEDDING_DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def _perturb(emb: np.ndarray, std: float, seed: int = 100) -> np.ndarray:
    """Perturb an embedding with Gaussian noise and re-normalize."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, std, size=emb.shape)
    perturbed = emb + noise
    return perturbed / np.linalg.norm(perturbed)


def test_system_scenarios(isolated_system_storage):
    storage, attendance, model_builder = isolated_system_storage

    # 1. Setup the system database with 2 registered users
    # Alice (our target) and Bob (to make classifier training multi-class)
    alice_center = _unit_vec(10)
    bob_center = _unit_vec(20)

    # Make bob orthogonal to alice to ensure good separability
    bob_center -= np.dot(bob_center, alice_center) * alice_center
    bob_center = bob_center / np.linalg.norm(bob_center)

    storage.upsert_user("alice", "Alice Liddell")
    storage.upsert_user("bob", "Bob Smith")

    # Enroll 5 poses for both (adding slight noise to simulate poses)
    alice_poses = [_perturb(alice_center, 0.03, seed=i) for i in range(5)]
    bob_poses = [_perturb(bob_center, 0.03, seed=i+10) for i in range(5)]

    storage.add_embeddings("alice", alice_poses)
    storage.add_embeddings("bob", bob_poses)

    # Train the classifiers (SVM)
    model_builder.train(kind="SVM", persist=True)

    # Set default threshold
    threshold = 0.45

    # ----------------------------------------------------
    # Scenario A: Known Person (Normal condition)
    # ----------------------------------------------------
    normal_input = _perturb(alice_center, config.NOISE_NORMAL, seed=42)
    res_normal = attendance.identify(normal_input, threshold=0.45)
    
    assert res_normal.is_known is True
    assert res_normal.user_id == "alice"
    assert res_normal.confidence > 0.75  # Normal noise should yield very high similarity

    pred_normal = model_builder.predict(normal_input)
    assert pred_normal is not None
    assert pred_normal["user_id"] == "alice"

    # ----------------------------------------------------
    # Scenario B: Unknown Person
    # ----------------------------------------------------
    unknown_input = _unit_vec(999)
    unknown_input -= np.dot(unknown_input, alice_center) * alice_center
    unknown_input -= np.dot(unknown_input, bob_center) * bob_center
    unknown_input = unknown_input / np.linalg.norm(unknown_input)

    res_unknown = attendance.identify(unknown_input, threshold=0.45)
    assert res_unknown.is_known is False
    assert res_unknown.name == "Unknown"
    assert res_unknown.confidence < 0.45

    # ----------------------------------------------------
    # Scenario C: Glasses Condition
    # ----------------------------------------------------
    glasses_input = _perturb(alice_center, config.NOISE_GLASSES, seed=43)
    res_glasses = attendance.identify(glasses_input, threshold=0.35)
    
    assert res_glasses.is_known is True
    assert res_glasses.user_id == "alice"

    # ----------------------------------------------------
    # Scenario D: Low Light Condition
    # ----------------------------------------------------
    low_light_input = _perturb(alice_center, config.NOISE_LOW_LIGHT, seed=44)
    res_low_light = attendance.identify(low_light_input, threshold=0.30)
    
    assert res_low_light.is_known is True
    assert res_low_light.user_id == "alice"

    # ----------------------------------------------------
    # Scenario E: Mask Condition
    # ----------------------------------------------------
    mask_input = _perturb(alice_center, config.NOISE_MASK, seed=45)
    res_mask = attendance.identify(mask_input, threshold=0.45)
    
    assert res_mask.is_known is False
    assert res_mask.name == "Unknown"

    # Verify that confidence scores degrade in the expected order based on noise level
    assert res_normal.confidence > res_glasses.confidence
    assert res_glasses.confidence > res_low_light.confidence
    assert res_low_light.confidence > res_mask.confidence

