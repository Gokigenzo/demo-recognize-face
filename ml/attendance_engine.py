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


def identify_with_classifier(
    embedding: np.ndarray,
    classifier_bundle: Dict[str, Any],
    threshold: float,
) -> RecognitionResult:
    """Identify a face embedding using a trained classifier and verify with similarity.

    Uses the classifier to predict the identity category and verifies the decision
    region matches the actual target profile by thresholding the cosine similarity score.
    """
    classifier = classifier_bundle.get("classifier")
    classes = list(classifier_bundle.get("classes", []))
    class_names = list(classifier_bundle.get("class_names", []))

    if not classifier or not classes:
        return identify(embedding, threshold=threshold)

    x = np.atleast_2d(np.asarray(embedding, dtype=np.float32))
    try:
        proba = classifier.predict_proba(x)
        top_idx = int(np.argmax(proba[0]))
    except Exception:
        try:
            prediction = classifier.predict(x)[0]
            top_idx = int(prediction)
        except Exception:
            return identify(embedding, threshold=threshold)

    if top_idx < 0 or top_idx >= len(classes):
        return identify(embedding, threshold=threshold)

    user_id = classes[top_idx]

    # Verify using cosine similarity
    db = storage.load_embeddings_db()
    users = storage.load_users()
    user_embs = db.get(user_id, [])
    if user_embs:
        similarity = max(cosine_similarity(embedding, e) for e in user_embs)
    else:
        similarity = 0.0

    is_known = similarity >= threshold
    name = users.get(user_id, {}).get("name", user_id)

    # Calculate top candidates for explainability UI
    scored = []
    for uid, embeddings in db.items():
        if not embeddings:
            continue
        best = max(cosine_similarity(embedding, e) for e in embeddings)
        scored.append((uid, users.get(uid, {}).get("name", uid), best))
    scored.sort(key=lambda t: t[2], reverse=True)

    return RecognitionResult(
        user_id=user_id if is_known else None,
        name=name if is_known else "Unknown",
        confidence=similarity,
        is_known=is_known,
        candidates=scored[:5],
    )
