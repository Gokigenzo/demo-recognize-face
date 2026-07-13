"""Real-time inference engine for the attendance application.

This module isolates the webcam / detection / embedding / classification
pipeline from the Streamlit UI so the page remains focused on rendering.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import cv2
import numpy as np

from ml import config, storage
from ml.attendance_engine import RecognitionResult, identify, identify_with_classifier
from ml.attendance_session import AttendanceRecord, AttendanceSession, PredictionResult, RecognitionStateMachine
from ml.embedder import embed_face
from ml.face_detector import detect_faces

LOGGER = logging.getLogger(__name__)


def compute_iou(boxA: tuple[int, int, int, int], boxB: tuple[int, int, int, int]) -> float:
    """Compute the Intersection over Union (IoU) of two bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    unionArea = boxAArea + boxBArea - interArea
    if unionArea == 0:
        return 0.0
    return interArea / float(unionArea)


class FaceTracker:
    """Tracks detected face bounding boxes across frames using IoU."""

    def __init__(self, iou_threshold: float = 0.3) -> None:
        self.iou_threshold = iou_threshold
        self.next_id = 1
        self.tracked_faces: Dict[int, tuple[int, int, int, int]] = {}

    def update(self, bboxes: List[tuple[int, int, int, int]]) -> List[int]:
        assigned_ids = []
        new_tracked = {}
        for bbox in bboxes:
            best_id = None
            best_iou = -1.0
            for fid, last_bbox in self.tracked_faces.items():
                iou_val = compute_iou(bbox, last_bbox)
                if iou_val > best_iou and iou_val >= self.iou_threshold:
                    best_iou = iou_val
                    best_id = fid
            if best_id is not None:
                assigned_ids.append(best_id)
                new_tracked[best_id] = bbox
            else:
                assigned_ids.append(self.next_id)
                new_tracked[self.next_id] = bbox
                self.next_id += 1
        self.tracked_faces = new_tracked
        return assigned_ids



@dataclass
class FaceInference:
    face_id: int
    bbox: tuple[int, int, int, int]
    confidence: float
    label: str
    status: str
    color: tuple[int, int, int]


class RealtimeAttendanceEngine:
    """Handles live webcam frames and produces prediction updates."""

    def __init__(self, session: AttendanceSession, threshold: float = config.DEFAULT_SIMILARITY_THRESHOLD) -> None:
        self.session = session
        self.threshold = threshold
        self._classifier_loaded = False
        self._classifier: Optional[object] = None
        self._classes: List[str] = []
        self._class_names: List[str] = []
        self._last_frame_time = 0.0
        self._fps = 0.0
        self._face_counter = 0
        self._capture: Optional[cv2.VideoCapture] = None
        self._last_process_time: Optional[float] = None
        self._target_frame_interval = 1.0 / config.REALTIME_TARGET_FPS
        self._detection_interval = config.REALTIME_DETECTION_INTERVAL
        self._frame_counter = 0
        self._last_faces: List[object] = []
        self._last_predictions: Dict[int, PredictionResult] = {}
        self._dropped_frames = 0
        self.tracker = FaceTracker()
        self.state_machines: Dict[int, RecognitionStateMachine] = {}

    def load_classifier(self) -> None:
        bundle = storage.load_classifier()
        if bundle is None or "classifier" not in bundle:
            raise RuntimeError("No trained classifier available for Application page.")
        
        # Determine if the classifier has changed by checking metadata
        current_id = (bundle.get("kind"), bundle.get("train_accuracy"), bundle.get("n_samples"))
        if self._classifier_loaded and getattr(self, "_classifier_id", None) == current_id:
            return

        self._classifier = bundle["classifier"]
        self._classes = list(bundle.get("classes", []))
        self._class_names = list(bundle.get("class_names", []))
        if len(self._classes) != len(self._class_names):
            raise RuntimeError("Loaded classifier bundle is missing class metadata.")
        self._classifier_id = current_id
        self._classifier_loaded = True
        LOGGER.info("Realtime engine dynamically loaded/updated the classifier model.")



    def _predict_identity(self, embedding: np.ndarray) -> RecognitionResult:
        if self._classifier is None:
            return identify(embedding, threshold=self.threshold)

        bundle = {
            "classifier": self._classifier,
            "classes": self._classes,
            "class_names": self._class_names,
        }
        return identify_with_classifier(embedding, bundle, threshold=self.threshold)

    def open_camera(self, index: int = 0) -> None:
        if self._capture is not None and self._capture.isOpened():
            return
        self._capture = cv2.VideoCapture(index)
        if not self._capture.isOpened():
            raise RuntimeError("Unable to open webcam (camera index 0).")
        self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.REALTIME_CAMERA_WIDTH)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.REALTIME_CAMERA_HEIGHT)
        self._capture.set(cv2.CAP_PROP_FPS, config.REALTIME_CAMERA_FPS)

    def close_camera(self) -> None:
        if self._capture is not None:
            try:
                self._capture.release()
            except Exception:
                LOGGER.exception("Failed to release webcam.")
        self._capture = None

    def capture_frame(self) -> Optional[np.ndarray]:
        if self._capture is None:
            raise RuntimeError("Camera has not been opened.")
        ok, frame = self._capture.read()
        if not ok or frame is None or frame.size == 0:
            return None
        return frame

    def should_process_frame(self, now: Optional[float] = None) -> bool:
        if now is None:
            now = time.monotonic()
        if self._last_process_time is None:
            self._last_process_time = now
            return True
        if (now - self._last_process_time) >= self._target_frame_interval:
            self._last_process_time = now
            return True
        self._dropped_frames += 1
        return False

    def set_recognition_interval(self, interval_ms: int) -> None:
        interval_ms = max(100, int(interval_ms))
        self._target_frame_interval = interval_ms / 1000.0

    def set_detection_interval(self, interval: int) -> None:
        self._detection_interval = max(1, int(interval))

    def _prepare_frame(self, frame: np.ndarray) -> np.ndarray:
        if frame is None or frame.size == 0:
            return frame
        height, width = frame.shape[:2]
        if width <= config.REALTIME_RESIZE_WIDTH:
            return frame
        scale = config.REALTIME_RESIZE_WIDTH / float(width)
        resized_height = max(1, int(height * scale))
        return cv2.resize(frame, (config.REALTIME_RESIZE_WIDTH, resized_height), interpolation=cv2.INTER_AREA)

    def _update_fps(self) -> None:
        now = time.time()
        if self._last_frame_time > 0:
            elapsed = now - self._last_frame_time
            if elapsed > 0:
                self._fps = 0.9 * self._fps + 0.1 * (1.0 / elapsed)
        self._last_frame_time = now

    def fps(self) -> float:
        return round(self._fps, 1)

    def process_photo(self, frame: np.ndarray) -> tuple[np.ndarray, List[PredictionResult], List[FaceInference]]:
        """Process a single captured photo and mark attendance for any recognized faces."""
        t_start = time.perf_counter()
        self._update_fps()
        self._frame_counter += 1
        processed_frame = self._prepare_frame(frame)

        detect_start = time.perf_counter()
        faces = detect_faces(processed_frame)
        if faces:
            largest_face = max(faces, key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]))
            faces = [largest_face]
        self._last_faces = faces
        detect_ms = (time.perf_counter() - detect_start) * 1000.0

        bboxes = [face.bbox for face in faces]
        face_ids = self.tracker.update(bboxes)

        active_ids = set(face_ids)
        self.state_machines = {fid: sm for fid, sm in self.state_machines.items() if fid in active_ids}
        self._last_predictions = {fid: pred for fid, pred in self._last_predictions.items() if fid in active_ids}

        predictions: List[PredictionResult] = []
        annotations: List[FaceInference] = []

        for face, face_id in zip(faces, face_ids):
            cached_result = self._last_predictions.get(face_id)
            if cached_result is None:
                embed_start = time.perf_counter()
                embedding = embed_face(face)
                embed_ms = (time.perf_counter() - embed_start) * 1000.0
                classify_start = time.perf_counter()
                result = self._predict_identity(embedding)
                classify_ms = (time.perf_counter() - classify_start) * 1000.0
                self._last_predictions[face_id] = result
            else:
                result = cached_result
                embed_ms = 0.0
                classify_ms = 0.0

            if face_id not in self.state_machines:
                self.state_machines[face_id] = RecognitionStateMachine(face_id, self.session.confirmation_frames)
            sm = self.state_machines[face_id]

            if result.is_known and result.user_id is not None and result.user_id not in self.session.students:
                result.is_known = False
                result.user_id = None
                result.name = "Unknown"

            is_present = False
            if result.is_known and result.user_id is not None:
                is_present = self.session.students[result.user_id].present

            old_state = sm.state
            new_state = sm.update(result, is_present)
            if result.is_known and result.user_id is not None and not is_present:
                new_state = "Confirmed"
                sm.state = "Confirmed"
                sm.consecutive_identity = result.user_id
                sm.consecutive_count = max(1, self.session.confirmation_frames)

            if new_state == "Confirmed" and old_state != "Confirmed":
                if result.is_known and result.user_id is not None:
                    student = self.session.students[result.user_id]
                    if not student.present:
                        self.session.mark_present(result.user_id, result.confidence)
                        self.session.event_logger.log_event(
                            student_name=result.name,
                            confidence=result.confidence,
                            result="✓ Present",
                            color="green",
                        )
                    else:
                        if not getattr(sm, "_duplicate_logged", False):
                            self.session.event_logger.log_event(
                                student_name=result.name,
                                confidence=result.confidence,
                                result="Duplicate Ignored",
                                color="gray",
                            )
                            self.session.duplicate_count += 1
                            sm._duplicate_logged = True
            elif new_state == "Unknown" and old_state != "Unknown":
                if not getattr(sm, "_unknown_logged", False):
                    self.session.event_logger.log_event(
                        student_name="Unknown",
                        confidence=result.confidence,
                        result="Rejected",
                        color="red",
                    )
                    self.session.unknown_count += 1
                    sm._unknown_logged = True

            self.session.add_confidence(result.confidence)

            prediction = PredictionResult(
                user_id=result.user_id,
                name=result.name,
                confidence=result.confidence,
                is_known=result.is_known,
                status=new_state.lower(),
            )
            predictions.append(prediction)
            annotations.append(self._render_face_box_sm(face, face_id, sm, prediction))

        self.session.frames_processed += 1
        self.session.predictions_seen += len(predictions)

        annotated = self._draw_annotations_sm(processed_frame, faces, face_ids)

        inf_time_ms = (time.perf_counter() - t_start) * 1000.0
        self.session.add_inference_time(inf_time_ms)

        if self._frame_counter % 10 == 0 or LOGGER.isEnabledFor(logging.DEBUG):
            LOGGER.debug(
                "photo stats fps=%.1f detect=%.2fms embed=%.2fms classify=%.2fms total=%.2fms dropped=%d",
                self.fps(),
                detect_ms,
                embed_ms if 'embed_ms' in locals() else 0.0,
                classify_ms if 'classify_ms' in locals() else 0.0,
                inf_time_ms,
                self._dropped_frames,
            )

        return annotated, predictions, annotations

    def infer_frame(self, frame: np.ndarray) -> tuple[np.ndarray, List[PredictionResult], List[FaceInference]]:
        """Backward-compatible wrapper for the existing continuous camera flow."""
        return self.process_photo(frame)

        should_detect = self._frame_counter % self._detection_interval == 0
        if should_detect:
            detect_start = time.perf_counter()
            faces = detect_faces(processed_frame)
            if faces:
                largest_face = max(faces, key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]))
                faces = [largest_face]
            self._last_faces = faces
            detect_ms = (time.perf_counter() - detect_start) * 1000.0
        else:
            faces = self._last_faces
            detect_ms = 0.0

        # Track faces using IoU
        bboxes = [face.bbox for face in faces]
        face_ids = self.tracker.update(bboxes)

        # Clean up stale state machines
        active_ids = set(face_ids)
        self.state_machines = {fid: sm for fid, sm in self.state_machines.items() if fid in active_ids}
        self._last_predictions = {fid: pred for fid, pred in self._last_predictions.items() if fid in active_ids}

        predictions: List[PredictionResult] = []
        annotations: List[FaceInference] = []

        for face, face_id in zip(faces, face_ids):
            cached_result = self._last_predictions.get(face_id)
            if should_detect or cached_result is None:
                embed_start = time.perf_counter()
                embedding = embed_face(face)
                embed_ms = (time.perf_counter() - embed_start) * 1000.0
                classify_start = time.perf_counter()
                result = self._predict_identity(embedding)
                classify_ms = (time.perf_counter() - classify_start) * 1000.0
                self._last_predictions[face_id] = result
            else:
                result = cached_result
                embed_ms = 0.0
                classify_ms = 0.0

            # Get or create state machine for this face
            if face_id not in self.state_machines:
                self.state_machines[face_id] = RecognitionStateMachine(face_id, self.session.confirmation_frames)
            sm = self.state_machines[face_id]

            # Check if student is already marked present
            if result.is_known and result.user_id is not None and result.user_id not in self.session.students:
                result.is_known = False
                result.user_id = None
                result.name = "Unknown"

            is_present = False
            if result.is_known and result.user_id is not None:
                is_present = self.session.students[result.user_id].present

            # Update state machine
            old_state = sm.state
            new_state = sm.update(result, is_present)

            # Log events on transitions
            if new_state == "Confirmed" and old_state != "Confirmed":
                if result.is_known and result.user_id is not None:
                    student = self.session.students[result.user_id]
                    if not student.present:
                        self.session.mark_present(result.user_id, result.confidence)
                        self.session.event_logger.log_event(
                            student_name=result.name,
                            confidence=result.confidence,
                            result="✓ Present",
                            color="green",
                        )
                    else:
                        if not getattr(sm, "_duplicate_logged", False):
                            self.session.event_logger.log_event(
                                student_name=result.name,
                                confidence=result.confidence,
                                result="Duplicate Ignored",
                                color="gray",
                            )
                            self.session.duplicate_count += 1
                            sm._duplicate_logged = True
            elif new_state == "Unknown" and old_state != "Unknown":
                if not getattr(sm, "_unknown_logged", False):
                    self.session.event_logger.log_event(
                        student_name="Unknown",
                        confidence=result.confidence,
                        result="Rejected",
                        color="red",
                    )
                    self.session.unknown_count += 1
                    sm._unknown_logged = True

            # Add confidence stats
            self.session.add_confidence(result.confidence)

            # Create prediction result
            prediction = PredictionResult(
                user_id=result.user_id,
                name=result.name,
                confidence=result.confidence,
                is_known=result.is_known,
                status=new_state.lower(),
            )
            predictions.append(prediction)

            # Create face inference annotation metadata
            annotations.append(self._render_face_box_sm(face, face_id, sm, prediction))

        # Update session frame count
        self.session.frames_processed += 1
        self.session.predictions_seen += len(predictions)

        # Draw annotations on the frame
        annotated = self._draw_annotations_sm(processed_frame, faces, face_ids)

        # Log inference time
        inf_time_ms = (time.perf_counter() - t_start) * 1000.0
        self.session.add_inference_time(inf_time_ms)

        if self._frame_counter % 10 == 0 or LOGGER.isEnabledFor(logging.DEBUG):
            LOGGER.debug(
                "realtime stats fps=%.1f detect=%.2fms embed=%.2fms classify=%.2fms total=%.2fms dropped=%d",
                self.fps(),
                detect_ms,
                embed_ms if 'embed_ms' in locals() else 0.0,
                classify_ms if 'classify_ms' in locals() else 0.0,
                inf_time_ms,
                self._dropped_frames,
            )

        return annotated, predictions, annotations

    def _render_face_box_sm(self, face: object, face_id: int, sm: RecognitionStateMachine, prediction: PredictionResult) -> FaceInference:
        if sm.state == "Confirmed":
            color = (16, 185, 129)  # Green
            label = f"{prediction.name} (Confirmed)"
        elif sm.state == "Verifying":
            color = (234, 179, 8)   # Yellow
            label = f"Verifying... {prediction.name} ({sm.consecutive_count}/{sm.confirmation_frames})"
        elif sm.state == "Detecting":
            color = (234, 179, 8)   # Yellow
            label = "Detecting..."
        else:
            color = (239, 68, 68)   # Red
            label = "Unknown"

        return FaceInference(
            face_id=face_id,
            bbox=face.bbox,
            confidence=prediction.confidence,
            label=label,
            status=sm.state.lower(),
            color=color,
        )

    def _draw_annotations_sm(self, frame: np.ndarray, faces: List[object], face_ids: List[int]) -> np.ndarray:
        out = frame.copy()
        for face, face_id in zip(faces, face_ids):
            x1, y1, x2, y2 = face.bbox
            sm = self.state_machines.get(face_id)
            if sm is None:
                continue

            if sm.state == "Confirmed":
                color = (16, 185, 129)  # Green
            elif sm.state == "Verifying" or sm.state == "Detecting":
                color = (234, 179, 8)   # Yellow
            else:
                color = (239, 68, 68)   # Red

            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

            if sm.state == "Confirmed":
                student_name = self.session.students[sm.consecutive_identity].name if sm.consecutive_identity in self.session.students else sm.consecutive_identity
                label1 = student_name
                label2 = "Attendance Recorded"
            elif sm.state == "Verifying":
                student_name = self.session.students[sm.consecutive_identity].name if sm.consecutive_identity in self.session.students else sm.consecutive_identity
                label1 = "Verifying..."
                label2 = f"{student_name} ({sm.consecutive_count}/{sm.confirmation_frames})"
            elif sm.state == "Detecting":
                label1 = "Detecting..."
                label2 = ""
            else:
                label1 = "Unknown"
                label2 = ""

            cv2.putText(
                out,
                label1,
                (x1, max(y1 - 25 if label2 else y1 - 10, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )
            if label2:
                cv2.putText(
                    out,
                    label2,
                    (x1, max(y1 - 8, 25)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA,
                )
        return out

    def set_threshold(self, threshold: float) -> None:
        self.threshold = threshold

    def stop(self) -> None:
        self.close_camera()
