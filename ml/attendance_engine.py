"""Attendance engine (deployment stage).

Given a face embedding, search the embedding database for the closest known
identity. If the best cosine similarity clears the threshold, mark attendance;
otherwise return ``Unknown`` (the demo's "predicting Unknown is better than
guessing" lesson).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from ml import config, storage
from ml.embedder import cosine_similarity


@dataclass
class RecognitionResult:
    user_id: Optional[str]
    name: str
    confidence: float
    is_known: bool
    # Ranked (user_id, name, score) candidates for explainability in the UI.
    candidates: List[tuple]


def _user_name(user_id: str, users: Dict) -> str:
    return users.get(user_id, {}).get("name", user_id)


def identify(
    embedding: np.ndarray,
    threshold: float = config.DEFAULT_SIMILARITY_THRESHOLD,
) -> RecognitionResult:
    """Identify a single embedding against the stored database.

    The per-user score is the *maximum* cosine similarity across that user's
    stored embeddings (best matching pose / augmentation).
    """
    db = storage.load_embeddings_db()
    users = storage.load_users()

    if not db:
        return RecognitionResult(None, "Unknown", 0.0, False, [])

    scored: List[tuple] = []
    for user_id, embeddings in db.items():
        if not embeddings:
            continue
        best = max(cosine_similarity(embedding, e) for e in embeddings)
        scored.append((user_id, _user_name(user_id, users), best))

    scored.sort(key=lambda t: t[2], reverse=True)
    if not scored:
        return RecognitionResult(None, "Unknown", 0.0, False, [])

    top_id, top_name, top_score = scored[0]
    is_known = top_score >= threshold
    return RecognitionResult(
        user_id=top_id if is_known else None,
        name=top_name if is_known else "Unknown",
        confidence=top_score,
        is_known=is_known,
        candidates=scored[:5],
    )


def mark_attendance(result: RecognitionResult) -> Optional[Dict]:
    """Log an attendance row for a *known* recognition result."""
    if not result.is_known or result.user_id is None:
        return None
    return storage.append_attendance(
        result.user_id, result.name, result.confidence, status="present"
    )


def recognize_and_log(
    embedding: np.ndarray,
    threshold: float = config.DEFAULT_SIMILARITY_THRESHOLD,
    log: bool = True,
) -> RecognitionResult:
    """Convenience: identify then (optionally) log if known."""
    result = identify(embedding, threshold)
    if log and result.is_known:
        mark_attendance(result)
    return result
