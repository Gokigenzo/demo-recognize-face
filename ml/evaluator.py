"""Model evaluation (evaluation stage).

Builds a small controlled test set, runs identification at a given threshold,
and reports a confusion matrix plus accuracy / precision / recall / F1. Also
provides a threshold *sweep* so the UI can plot the precision-vs-recall
tradeoff and reinforce that "predicting Unknown is better than guessing".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from ml import config
from ml.attendance_engine import identify


@dataclass
class EvalSample:
    """A labelled evaluation sample."""

    embedding: np.ndarray
    true_user_id: Optional[str]   # None == genuinely Unknown person
    case: str                     # "Normal", "Glasses", "Mask", ...


@dataclass
class EvalResult:
    threshold: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    labels: List[str]
    confusion: np.ndarray
    per_case: Dict[str, float]    # case -> accuracy


def _prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def evaluate(
    samples: List[EvalSample],
    threshold: float = config.DEFAULT_SIMILARITY_THRESHOLD,
) -> EvalResult:
    """Evaluate identification over labelled samples at one threshold.

    Treats the task as binary at the decision level: a sample is a *positive*
    if it belongs to a known person. Precision/Recall are computed over
    correctly-identified known people, which is what matters for attendance.
    """
    tp = fp = fn = tn = 0
    correct = 0
    per_case_correct: Dict[str, int] = {}
    per_case_total: Dict[str, int] = {}

    # Confusion over {Known-correct, Known-wrong, Unknown}
    labels = ["Correct ID", "Wrong ID", "Unknown"]
    idx = {l: i for i, l in enumerate(labels)}
    confusion = np.zeros((2, 3), dtype=int)  # rows: actual known / actual unknown

    for s in samples:
        per_case_total[s.case] = per_case_total.get(s.case, 0) + 1
        result = identify(s.embedding, threshold)
        actual_known = s.true_user_id is not None
        pred_known = result.is_known

        if actual_known:
            if pred_known and result.user_id == s.true_user_id:
                tp += 1; correct += 1
                confusion[0, idx["Correct ID"]] += 1
                per_case_correct[s.case] = per_case_correct.get(s.case, 0) + 1
            elif pred_known:               # predicted someone, but wrong person
                fp += 1; fn += 1
                confusion[0, idx["Wrong ID"]] += 1
            else:                          # said Unknown for a known person
                fn += 1
                confusion[0, idx["Unknown"]] += 1
        else:  # actual unknown
            if pred_known:                 # false alarm
                fp += 1
                confusion[1, idx["Wrong ID"]] += 1
            else:                          # correctly rejected
                tn += 1; correct += 1
                confusion[1, idx["Unknown"]] += 1
                per_case_correct[s.case] = per_case_correct.get(s.case, 0) + 1

    precision, recall, f1 = _prf(tp, fp, fn)
    accuracy = correct / len(samples) if samples else 0.0
    per_case = {
        c: per_case_correct.get(c, 0) / per_case_total[c]
        for c in per_case_total
    }
    return EvalResult(
        threshold=threshold,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        labels=["Actual: Known", "Actual: Unknown"],
        confusion=confusion,
        per_case=per_case,
    )


def threshold_sweep(
    samples: List[EvalSample],
    thresholds: Optional[List[float]] = None,
) -> List[EvalResult]:
    """Evaluate across a range of thresholds for precision/recall curves."""
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.linspace(0.1, 0.9, 17)]
    return [evaluate(samples, t) for t in thresholds]
