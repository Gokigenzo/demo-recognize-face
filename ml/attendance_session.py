"""Runtime attendance session state for the application page.

Keeps all attendance state in memory during a live stream, including the
student checklist, confirmation counters, and finalized attendance records.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class PredictionResult:
    """A per-face inference outcome before session confirmation."""

    user_id: Optional[str]
    name: str
    confidence: float
    is_known: bool
    status: str = "unknown"
    confirmed: bool = False


@dataclass
class StudentAttendance:
    user_id: str
    name: str
    present: bool = False
    confidence: float = 0.0
    confirmation_count: int = 0


@dataclass
class AttendanceRecord:
    timestamp: str
    user_id: str
    name: str
    confidence: float
    status: str = "present"

    def as_dict(self) -> Dict[str, object]:
        return {
            "timestamp": self.timestamp,
            "user_id": self.user_id,
            "name": self.name,
            "confidence": round(float(self.confidence), 4),
            "status": self.status,
        }


@dataclass
class AttendanceEvent:
    """Represents a logged event in the live attendance timeline."""
    timestamp: str
    student_name: str
    confidence: float
    result: str
    color: str


class EventLogger:
    """Tracks and formats events in the live attendance timeline."""
    def __init__(self, max_events: int = 100) -> None:
        self.max_events = max_events
        self.events: List[AttendanceEvent] = []

    def log_event(self, student_name: str, confidence: float, result: str, color: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        event = AttendanceEvent(
            timestamp=timestamp,
            student_name=student_name,
            confidence=confidence,
            result=result,
            color=color,
        )
        self.events.insert(0, event)
        if len(self.events) > self.max_events:
            self.events.pop()

    def clear(self) -> None:
        self.events.clear()


class RecognitionStateMachine:
    """Manages the state transitions of a tracked face across frames."""

    def __init__(self, face_id: int, confirmation_frames: int = 5) -> None:
        self.face_id = face_id
        self.confirmation_frames = confirmation_frames
        self.state = "Detecting"  # "Detecting", "Verifying", "Confirmed", "Unknown"
        self.consecutive_identity: Optional[str] = None
        self.consecutive_count = 0
        self.last_confidence = 0.0

    def update(self, prediction: PredictionResult, is_already_present: bool) -> str:
        self.last_confidence = prediction.confidence

        if not prediction.is_known or prediction.user_id is None:
            self.state = "Unknown"
            self.consecutive_identity = None
            self.consecutive_count = 0
            return self.state

        user_id = prediction.user_id

        if is_already_present:
            self.state = "Confirmed"
            self.consecutive_identity = user_id
            self.consecutive_count = self.confirmation_frames
            return self.state

        if user_id == self.consecutive_identity:
            self.consecutive_count += 1
        else:
            self.consecutive_identity = user_id
            self.consecutive_count = 1

        if self.consecutive_count >= self.confirmation_frames:
            self.state = "Confirmed"
        elif self.consecutive_count > 1:
            self.state = "Verifying"
        else:
            self.state = "Detecting"

        return self.state


@dataclass
class AttendanceSession:
    users: Dict[str, Dict[str, object]]
    confirmation_frames: int = 5
    students: Dict[str, StudentAttendance] = field(init=False)
    records: List[AttendanceRecord] = field(init=False, default_factory=list)
    pending_counts: Dict[str, int] = field(init=False, default_factory=dict)
    frames_processed: int = 0
    predictions_seen: int = 0
    start_time: float = field(init=False)
    unknown_count: int = field(init=False, default=0)
    duplicate_count: int = field(init=False, default=0)
    confidence_sum: float = field(init=False, default=0.0)
    confidence_count: int = field(init=False, default=0)
    inference_time_sum: float = field(init=False, default=0.0)
    inference_time_count: int = field(init=False, default=0)
    event_logger: EventLogger = field(init=False)

    def __post_init__(self) -> None:
        self.students = {
            user_id: StudentAttendance(user_id=user_id, name=info.get("name", user_id))
            for user_id, info in self.users.items()
        }
        self.pending_counts = {user_id: 0 for user_id in self.users}
        self.start_time = time.time()
        self.event_logger = EventLogger()
        self.unknown_count = 0
        self.duplicate_count = 0
        self.confidence_sum = 0.0
        self.confidence_count = 0
        self.inference_time_sum = 0.0
        self.inference_time_count = 0

    def reset(self) -> None:
        self.records.clear()
        self.frames_processed = 0
        self.predictions_seen = 0
        self.pending_counts = {user_id: 0 for user_id in self.users}
        for student in self.students.values():
            student.present = False
            student.confidence = 0.0
            student.confirmation_count = 0
        self.start_time = time.time()
        self.event_logger.clear()
        self.unknown_count = 0
        self.duplicate_count = 0
        self.confidence_sum = 0.0
        self.confidence_count = 0
        self.inference_time_sum = 0.0
        self.inference_time_count = 0

    def mark_present(self, user_id: str, confidence: float) -> None:
        if user_id in self.students:
            student = self.students[user_id]
            student.present = True
            student.confidence = confidence
            student.confirmation_count = self.confirmation_frames
            record = AttendanceRecord(
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                user_id=user_id,
                name=student.name,
                confidence=confidence,
            )
            self.records.append(record)

    def add_confidence(self, confidence: float) -> None:
        self.confidence_sum += confidence
        self.confidence_count += 1

    def add_inference_time(self, inf_time_ms: float) -> None:
        self.inference_time_sum += inf_time_ms
        self.inference_time_count += 1

    def update_predictions(
        self, predictions: List[PredictionResult]
    ) -> List[AttendanceRecord]:
        """Update attendance state using the latest frame predictions."""
        self.frames_processed += 1
        self.predictions_seen += len(predictions)

        confirmed_records: List[AttendanceRecord] = []
        active_user_ids = set()

        for prediction in predictions:
            if not prediction.is_known or prediction.user_id is None:
                continue
            if prediction.user_id not in self.students:
                continue
            student = self.students[prediction.user_id]
            if student.present:
                prediction.confirmed = True
                prediction.status = "present"
                continue

            active_user_ids.add(prediction.user_id)
            self.pending_counts[prediction.user_id] += 1
            student.confidence = prediction.confidence
            student.confirmation_count = self.pending_counts[prediction.user_id]

            if self.pending_counts[prediction.user_id] >= self.confirmation_frames:
                student.present = True
                prediction.confirmed = True
                prediction.status = "present"
                record = AttendanceRecord(
                    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    user_id=student.user_id,
                    name=student.name,
                    confidence=student.confidence,
                )
                self.records.append(record)
                confirmed_records.append(record)
            else:
                prediction.status = "confirming"
                prediction.confirmed = False

        self._decay_pending(active_user_ids)
        return confirmed_records

    def _decay_pending(self, active_user_ids: set[str]) -> None:
        for user_id in self.pending_counts:
            if self.students[user_id].present:
                continue
            if user_id not in active_user_ids and self.pending_counts[user_id] > 0:
                self.pending_counts[user_id] = max(0, self.pending_counts[user_id] - 1)

    def student_checklist(self) -> List[Dict[str, object]]:
        return [
            {
                "user_id": student.user_id,
                "name": student.name,
                "present": student.present,
                "confidence": round(student.confidence, 4),
                "status": "present" if student.present else "absent",
            }
            for student in self.students.values()
        ]

    def present_count(self) -> int:
        return sum(1 for student in self.students.values() if student.present)

    def absent_count(self) -> int:
        return sum(1 for student in self.students.values() if not student.present)

    def recognition_success_rate(self) -> float:
        total = max(1, self.frames_processed)
        return round(100.0 * len(self.records) / total, 1)

    def attendance_table(self) -> List[Dict[str, object]]:
        return [record.as_dict() for record in self.records]

    def export_csv(self) -> bytes:
        import csv
        import io

        output = io.StringIO()
        writer = csv.DictWriter(
            output,
            fieldnames=["Student", "Time", "Confidence", "Status"],
        )
        writer.writeheader()

        record_map = {record.user_id: record for record in self.records}
        for student in self.students.values():
            if not student.present and student.user_id not in record_map:
                continue

            record = record_map.get(student.user_id)
            if record is None:
                record = AttendanceRecord(
                    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    user_id=student.user_id,
                    name=student.name,
                    confidence=student.confidence,
                    status="present" if student.present else "absent",
                )

            writer.writerow(
                {
                    "Student": record.name,
                    "Time": record.timestamp,
                    "Confidence": f"{record.confidence:.4f}",
                    "Status": record.status,
                }
            )
        return output.getvalue().encode("utf-8")


class SessionStatistics:
    """Computes session-wide metrics for the dashboard."""

    def __init__(self, session: AttendanceSession) -> None:
        self.session = session

    def get_stats(self, current_fps: float) -> Dict[str, object]:
        total = len(self.session.students)
        present = self.session.present_count()
        absent = self.session.absent_count()
        rate = (present / total * 100.0) if total > 0 else 0.0

        elapsed_sec = int(time.time() - self.session.start_time)
        mins = elapsed_sec // 60
        secs = elapsed_sec % 60
        elapsed_str = f"{mins} min {secs} sec" if mins > 0 else f"{secs} sec"

        avg_conf = 0.0
        if self.session.confidence_count > 0:
            avg_conf = (self.session.confidence_sum / self.session.confidence_count) * 100.0

        avg_rec_time = 0.0
        if self.session.inference_time_count > 0:
            avg_rec_time = self.session.inference_time_sum / self.session.inference_time_count

        return {
            "total_students": total,
            "present": present,
            "absent": absent,
            "attendance_rate": f"{rate:.1f}%",
            "unknown_faces": self.session.unknown_count,
            "duplicate_recognitions": self.session.duplicate_count,
            "average_confidence": f"{avg_conf:.1f}%",
            "average_recognition_time": f"{avg_rec_time:.0f} ms",
            "current_fps": f"{current_fps:.1f}",
            "elapsed_session_time": elapsed_str,
        }

