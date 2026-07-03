"""Unit tests for the realtime attendance system components.

Tests cover:
- RecognitionStateMachine state transitions
- FaceTracker IoU-based tracking
- EventLogger event management
- SessionStatistics metric computation
- AttendanceSession mark_present and duplicate handling
"""
from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Dict, Optional

import numpy as np
import pytest

from ml.attendance_session import (
    AttendanceEvent,
    AttendanceSession,
    EventLogger,
    PredictionResult,
    RecognitionStateMachine,
    SessionStatistics,
)
from ml.realtime_engine import FaceTracker, RealtimeAttendanceEngine, compute_iou


# ---------------------------------------------------------------------------
# compute_iou
# ---------------------------------------------------------------------------
class TestComputeIoU:
    def test_identical_boxes(self):
        box = (10, 10, 50, 50)
        assert pytest.approx(compute_iou(box, box), rel=1e-5) == 1.0

    def test_no_overlap(self):
        a = (0, 0, 10, 10)
        b = (20, 20, 30, 30)
        assert compute_iou(a, b) == 0.0

    def test_partial_overlap(self):
        a = (0, 0, 10, 10)
        b = (5, 5, 15, 15)
        # Intersection: 5x5=25, Union: 100+100-25=175
        assert pytest.approx(compute_iou(a, b), rel=1e-4) == 25.0 / 175.0

    def test_one_inside_other(self):
        a = (0, 0, 20, 20)
        b = (5, 5, 15, 15)
        # Intersection: 10x10=100, Union: 400+100-100=400
        assert pytest.approx(compute_iou(a, b), rel=1e-4) == 100.0 / 400.0

    def test_zero_area_box(self):
        a = (5, 5, 5, 5)
        b = (0, 0, 10, 10)
        assert compute_iou(a, b) == 0.0


# ---------------------------------------------------------------------------
# FaceTracker
# ---------------------------------------------------------------------------
def test_realtime_engine_throttles_inference_processing():
    session = AttendanceSession(users={"u1": {"name": "Alice"}}, confirmation_frames=3)
    engine = RealtimeAttendanceEngine(session=session)
    engine._target_frame_interval = 0.1

    assert engine.should_process_frame(now=0.0) is True
    assert engine.should_process_frame(now=0.05) is False
    assert engine.should_process_frame(now=0.25) is True


def test_process_photo_marks_known_student_from_single_image(monkeypatch):
    session = AttendanceSession(users={"u1": {"name": "Alice"}}, confirmation_frames=3)
    engine = RealtimeAttendanceEngine(session=session)

    monkeypatch.setattr(engine, "_prepare_frame", lambda frame: frame)
    monkeypatch.setattr(
        "ml.realtime_engine.detect_faces",
        lambda frame: [SimpleNamespace(bbox=(0, 0, 40, 40), crop=np.zeros((40, 40, 3), dtype=np.uint8))],
    )
    monkeypatch.setattr("ml.realtime_engine.embed_face", lambda face: np.array([0.1, 0.2], dtype=np.float32))
    monkeypatch.setattr(
        engine,
        "_predict_identity",
        lambda embedding: PredictionResult(
            user_id="u1",
            name="Alice",
            confidence=0.96,
            is_known=True,
        ),
    )

    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    annotated, predictions, annotations = engine.process_photo(frame)

    assert annotated.shape == frame.shape
    assert len(predictions) == 1
    assert predictions[0].user_id == "u1"
    assert len(session.records) == 1
    assert session.students["u1"].present is True
    assert len(annotations) == 1


class TestFaceTracker:
    def test_assigns_new_ids_for_first_frame(self):
        tracker = FaceTracker()
        ids = tracker.update([(0, 0, 50, 50), (100, 100, 150, 150)])
        assert len(ids) == 2
        assert ids[0] != ids[1]

    def test_preserves_ids_for_same_positions(self):
        tracker = FaceTracker()
        ids1 = tracker.update([(10, 10, 60, 60)])
        ids2 = tracker.update([(12, 12, 62, 62)])  # Slight movement
        assert ids1[0] == ids2[0]

    def test_new_id_for_distant_face(self):
        tracker = FaceTracker()
        ids1 = tracker.update([(0, 0, 50, 50)])
        ids2 = tracker.update([(0, 0, 50, 50), (200, 200, 250, 250)])
        assert ids2[0] == ids1[0]
        assert ids2[1] != ids1[0]

    def test_empty_frame(self):
        tracker = FaceTracker()
        tracker.update([(10, 10, 50, 50)])
        ids = tracker.update([])
        assert ids == []
        assert tracker.tracked_faces == {}

    def test_face_disappears_and_reappears_gets_new_id(self):
        tracker = FaceTracker()
        ids1 = tracker.update([(10, 10, 50, 50)])
        tracker.update([])  # face disappears
        ids3 = tracker.update([(200, 200, 250, 250)])  # different location
        assert ids3[0] != ids1[0]


# ---------------------------------------------------------------------------
# RecognitionStateMachine
# ---------------------------------------------------------------------------
class TestRecognitionStateMachine:
    def _make_prediction(
        self,
        user_id: Optional[str] = "u1",
        name: str = "Alice",
        confidence: float = 0.95,
        is_known: bool = True,
    ) -> PredictionResult:
        return PredictionResult(
            user_id=user_id,
            name=name,
            confidence=confidence,
            is_known=is_known,
        )

    def test_initial_state_is_detecting(self):
        sm = RecognitionStateMachine(face_id=1, confirmation_frames=5)
        assert sm.state == "Detecting"

    def test_single_known_prediction_stays_detecting(self):
        sm = RecognitionStateMachine(face_id=1, confirmation_frames=5)
        state = sm.update(self._make_prediction(), is_already_present=False)
        assert state == "Detecting"
        assert sm.consecutive_count == 1

    def test_transitions_to_verifying_after_two(self):
        sm = RecognitionStateMachine(face_id=1, confirmation_frames=5)
        sm.update(self._make_prediction(), is_already_present=False)
        state = sm.update(self._make_prediction(), is_already_present=False)
        assert state == "Verifying"
        assert sm.consecutive_count == 2

    def test_transitions_to_confirmed_after_n_frames(self):
        sm = RecognitionStateMachine(face_id=1, confirmation_frames=3)
        sm.update(self._make_prediction(), is_already_present=False)
        sm.update(self._make_prediction(), is_already_present=False)
        state = sm.update(self._make_prediction(), is_already_present=False)
        assert state == "Confirmed"
        assert sm.consecutive_count == 3

    def test_identity_change_resets_counter(self):
        sm = RecognitionStateMachine(face_id=1, confirmation_frames=5)
        sm.update(self._make_prediction(user_id="u1", name="Alice"), is_already_present=False)
        sm.update(self._make_prediction(user_id="u1", name="Alice"), is_already_present=False)
        assert sm.consecutive_count == 2

        # Different identity resets
        state = sm.update(self._make_prediction(user_id="u2", name="Bob"), is_already_present=False)
        assert sm.consecutive_count == 1
        assert sm.consecutive_identity == "u2"
        assert state == "Detecting"

    def test_unknown_prediction_resets_to_unknown(self):
        sm = RecognitionStateMachine(face_id=1, confirmation_frames=5)
        sm.update(self._make_prediction(), is_already_present=False)
        sm.update(self._make_prediction(), is_already_present=False)
        assert sm.state == "Verifying"

        state = sm.update(
            self._make_prediction(user_id=None, name="Unknown", confidence=0.3, is_known=False),
            is_already_present=False,
        )
        assert state == "Unknown"
        assert sm.consecutive_count == 0
        assert sm.consecutive_identity is None

    def test_already_present_jumps_to_confirmed(self):
        sm = RecognitionStateMachine(face_id=1, confirmation_frames=5)
        state = sm.update(self._make_prediction(), is_already_present=True)
        assert state == "Confirmed"
        assert sm.consecutive_count == 5  # equals confirmation_frames


# ---------------------------------------------------------------------------
# EventLogger
# ---------------------------------------------------------------------------
class TestEventLogger:
    def test_log_event_adds_to_front(self):
        logger = EventLogger(max_events=10)
        logger.log_event("Alice", 0.95, "✓ Present", "green")
        logger.log_event("Bob", 0.90, "✓ Present", "green")
        assert len(logger.events) == 2
        assert logger.events[0].student_name == "Bob"
        assert logger.events[1].student_name == "Alice"

    def test_max_events_cap(self):
        logger = EventLogger(max_events=3)
        for i in range(5):
            logger.log_event(f"Student_{i}", 0.9, "✓ Present", "green")
        assert len(logger.events) == 3
        assert logger.events[0].student_name == "Student_4"
        assert logger.events[-1].student_name == "Student_2"

    def test_clear(self):
        logger = EventLogger()
        logger.log_event("Alice", 0.95, "✓ Present", "green")
        logger.clear()
        assert len(logger.events) == 0

    def test_event_has_timestamp(self):
        logger = EventLogger()
        logger.log_event("Alice", 0.95, "✓ Present", "green")
        assert logger.events[0].timestamp  # non-empty string


# ---------------------------------------------------------------------------
# AttendanceSession extensions
# ---------------------------------------------------------------------------
class TestAttendanceSessionExtensions:
    def _make_session(self) -> AttendanceSession:
        users = {
            "u1": {"name": "Alice"},
            "u2": {"name": "Bob"},
            "u3": {"name": "Charlie"},
        }
        return AttendanceSession(users=users, confirmation_frames=3)

    def test_mark_present(self):
        session = self._make_session()
        session.mark_present("u1", 0.98)
        assert session.students["u1"].present is True
        assert session.students["u1"].confidence == 0.98
        assert len(session.records) == 1
        assert session.records[0].name == "Alice"

    def test_mark_present_ignores_unknown_user(self):
        session = self._make_session()
        session.mark_present("unknown_id", 0.5)
        assert len(session.records) == 0

    def test_add_confidence_and_average(self):
        session = self._make_session()
        session.add_confidence(0.95)
        session.add_confidence(0.85)
        assert session.confidence_count == 2
        avg = session.confidence_sum / session.confidence_count
        assert pytest.approx(avg, rel=1e-4) == 0.90

    def test_add_inference_time(self):
        session = self._make_session()
        session.add_inference_time(50.0)
        session.add_inference_time(100.0)
        assert session.inference_time_count == 2
        avg = session.inference_time_sum / session.inference_time_count
        assert pytest.approx(avg, rel=1e-4) == 75.0

    def test_reset_clears_all_extended_state(self):
        session = self._make_session()
        session.mark_present("u1", 0.95)
        session.unknown_count = 5
        session.duplicate_count = 3
        session.add_confidence(0.9)
        session.add_inference_time(50)
        session.event_logger.log_event("Alice", 0.95, "✓ Present", "green")

        session.reset()

        assert session.students["u1"].present is False
        assert len(session.records) == 0
        assert session.unknown_count == 0
        assert session.duplicate_count == 0
        assert session.confidence_count == 0
        assert session.confidence_sum == 0.0
        assert session.inference_time_count == 0
        assert session.inference_time_sum == 0.0
        assert len(session.event_logger.events) == 0

    def test_duplicate_count_tracking(self):
        session = self._make_session()
        session.duplicate_count += 1
        session.duplicate_count += 1
        assert session.duplicate_count == 2

    def test_unknown_count_tracking(self):
        session = self._make_session()
        session.unknown_count += 1
        assert session.unknown_count == 1


# ---------------------------------------------------------------------------
# SessionStatistics
# ---------------------------------------------------------------------------
class TestSessionStatistics:
    def _make_session(self) -> AttendanceSession:
        users = {
            "u1": {"name": "Alice"},
            "u2": {"name": "Bob"},
            "u3": {"name": "Charlie"},
            "u4": {"name": "David"},
        }
        return AttendanceSession(users=users, confirmation_frames=3)

    def test_initial_stats(self):
        session = self._make_session()
        stats = SessionStatistics(session)
        result = stats.get_stats(current_fps=0.0)
        assert result["total_students"] == 4
        assert result["present"] == 0
        assert result["absent"] == 4
        assert result["attendance_rate"] == "0.0%"
        assert result["unknown_faces"] == 0
        assert result["duplicate_recognitions"] == 0

    def test_stats_after_attendance(self):
        session = self._make_session()
        session.mark_present("u1", 0.95)
        session.mark_present("u2", 0.90)
        session.add_confidence(0.95)
        session.add_confidence(0.90)
        session.unknown_count = 2
        session.duplicate_count = 1

        stats = SessionStatistics(session)
        result = stats.get_stats(current_fps=22.5)

        assert result["total_students"] == 4
        assert result["present"] == 2
        assert result["absent"] == 2
        assert result["attendance_rate"] == "50.0%"
        assert result["unknown_faces"] == 2
        assert result["duplicate_recognitions"] == 1
        assert result["current_fps"] == "22.5"
        assert "%" in result["average_confidence"]

    def test_elapsed_time_format(self):
        session = self._make_session()
        session.start_time = time.time() - 125  # 2 min 5 sec ago
        stats = SessionStatistics(session)
        result = stats.get_stats(current_fps=0.0)
        assert "min" in result["elapsed_session_time"]

    def test_average_recognition_time(self):
        session = self._make_session()
        session.add_inference_time(100.0)
        session.add_inference_time(200.0)
        stats = SessionStatistics(session)
        result = stats.get_stats(current_fps=0.0)
        assert result["average_recognition_time"] == "150 ms"
