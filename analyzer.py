"""YOLOv8 person detection and analysis on camera frames."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_model = None
_PERSON_CLASS_ID = 0  # COCO class 0 = person


def _get_model():
    global _model
    if _model is None:
        from ultralytics import YOLO
        logger.info("Loading YOLOv8n model...")
        _model = YOLO("yolov8n.pt")
        logger.info("YOLOv8n model loaded.")
    return _model


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def center(self) -> tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    @property
    def area(self) -> int:
        return (self.x2 - self.x1) * (self.y2 - self.y1)


@dataclass
class AnalysisResult:
    camera_id: str
    detections: list[Detection] = field(default_factory=list)
    person_count: int = 0
    analyzed_at: float = field(default_factory=time.time)
    inference_ms: float = 0.0
    frame_width: int = 0
    frame_height: int = 0

    @property
    def has_people(self) -> bool:
        return self.person_count > 0


def analyze_frame(camera_id: str, frame: np.ndarray, confidence_threshold: float = 0.35) -> AnalysisResult:
    """Run YOLOv8 detection on a frame and return results."""
    model = _get_model()
    h, w = frame.shape[:2]

    t0 = time.time()
    results = model(frame, verbose=False, conf=confidence_threshold)
    inference_ms = (time.time() - t0) * 1000

    detections: list[Detection] = []
    person_count = 0

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            conf = float(boxes.conf[i].item())
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            class_name = result.names[cls_id] if result.names else str(cls_id)

            det = Detection(
                class_id=cls_id,
                class_name=class_name,
                confidence=conf,
                x1=int(x1),
                y1=int(y1),
                x2=int(x2),
                y2=int(y2),
            )
            detections.append(det)
            if cls_id == _PERSON_CLASS_ID:
                person_count += 1

    return AnalysisResult(
        camera_id=camera_id,
        detections=detections,
        person_count=person_count,
        inference_ms=inference_ms,
        frame_width=w,
        frame_height=h,
    )


def draw_detections(frame: np.ndarray, result: AnalysisResult) -> np.ndarray:
    """Draw bounding boxes and labels on a frame copy."""
    annotated = frame.copy()

    for det in result.detections:
        if det.class_id == _PERSON_CLASS_ID:
            color = (0, 255, 0)  # green for person
        else:
            color = (255, 165, 0)  # orange for other objects

        cv2.rectangle(annotated, (det.x1, det.y1), (det.x2, det.y2), color, 2)

        label = f"{det.class_name} {det.confidence:.0%}"
        font_scale = 0.6
        thickness = 2
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.rectangle(annotated, (det.x1, det.y1 - th - 8), (det.x1 + tw + 4, det.y1), color, -1)
        cv2.putText(
            annotated, label,
            (det.x1 + 2, det.y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness,
        )

    info_text = f"People: {result.person_count} | {result.inference_ms:.0f}ms"
    cv2.putText(
        annotated, info_text,
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
    )

    return annotated


def frame_to_jpeg(frame: np.ndarray, quality: int = 85) -> bytes:
    """Encode a numpy frame to JPEG bytes."""
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()
