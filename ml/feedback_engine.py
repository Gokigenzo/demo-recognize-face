"""Feedback engine (feedback loop / continuous improvement stage).

When a human corrects a wrong/Unknown prediction, we (1) log the correction
and (2) fold the corrected embedding back into the database so the system
recognizes that appearance next time. This closes the ML lifecycle loop.
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

from ml.attendance_engine import identify


def record_correction(
    embedding: np.ndarray,
    correct_user_id: str,
    correct_name: str,
    predicted_name: str,
    note: str = "",
) -> Dict:
    """Log a human correction and add the embedding to the database.

    Returns the feedback entry that was stored.
    """
    from ml import storage
    storage.add_embeddings(correct_user_id, [np.asarray(embedding, dtype=np.float32)])
    # Ensure the user exists in the metadata store.
    storage.upsert_user(correct_user_id, correct_name)

    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "predicted": predicted_name,
        "corrected_to": correct_name,
        "correct_user_id": correct_user_id,
        "note": note,
    }
    storage.append_feedback(entry)
    return entry


def reverify(embedding: np.ndarray, threshold: float) -> Dict:
    """Re-run identification after a correction to show the improvement."""
    result = identify(embedding, threshold)
    return {
        "name": result.name,
        "confidence": result.confidence,
        "is_known": result.is_known,
    }


def improvement_summary() -> Dict:
    """Aggregate feedback stats for the lifecycle dashboard."""
    from ml import storage
    feedback = storage.load_feedback()
    db = storage.load_embeddings_db()
    total_embeddings = sum(len(v) for v in db.values())
    return {
        "total_corrections": len(feedback),
        "users_with_data": len(db),
        "total_embeddings": total_embeddings,
        "recent": feedback[-5:][::-1],
    }
