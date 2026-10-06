from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

import db
from analyzer import analyze_frame, draw_detections, frame_to_jpeg
from models import CameraSource, StreamStatus, utc_now
from safety import SafetyEngine
from stream_capture import close_captures, CapturedFrame, FrameCache, StreamSource, grab_frame
from tracking import MultiCameraTracker, RuntimeDetection

logger = logging.getLogger(__name__)


@dataclass
class RuntimeCameraState:
    source: CameraSource
    status: StreamStatus = field(default_factory=lambda: StreamStatus(online=False, has_frame=False))
    last_error: str | None = None
    last_analyzed_at: float = 0.0


class CameraRuntime:
    def __init__(self) -> None:
        self._states: dict[str, RuntimeCameraState] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._tracker = MultiCameraTracker()
        self._rules = SafetyEngine()
        self._analysis_status: dict[str, str] = {}
        self._analysis_jpeg: dict[str, bytes] = {}

    def start(self) -> None:
        self.reload_sources()
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, daemon=True, name="camera-runtime")
            self._thread.start()
            logger.info("Camera runtime started for %d sources", len(self._states))

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        if not self._thread or not self._thread.is_alive():
            close_captures()

    def reload_sources(self) -> None:
        sources = db.list_camera_sources(enabled_only=True)
        with self._lock:
            self._states = {source.id: self._states.get(source.id, RuntimeCameraState(source=source)) for source in sources}
            for source in sources:
                self._states[source.id].source = source

    def status(self) -> dict[str, StreamStatus]:
        with self._lock:
            return {camera_id: state.status for camera_id, state in self._states.items()}

    def get_annotated_jpeg(self, camera_id: str) -> bytes | None:
        with self._lock:
            return self._analysis_jpeg.get(camera_id)

    def detections_payload(self, camera_id: str) -> dict[str, object]:
        tracks = self._tracker.active_tracks(camera_id)
        return {
            "camera_id": camera_id,
            "person_count": len(tracks),
            "status": self._analysis_status.get(camera_id, "warming_up"),
            "category": self._rules.categories.get(camera_id, "unknown"),
            "detections": [
                {
                    "class": "person",
                    "confidence": round(track.confidence, 3),
                    "bbox": track.bbox,
                    "track_id": track.local_id,
                    "zone": track.zone.name if track.zone else None,
                }
                for track in tracks
            ],
        }

    def _loop(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                sources = list(self._states.values())
            close_captures({s.source.id for s in sources})
            settings = db.load_settings()
            for state in sources:
                source = state.source
                now = time.monotonic()
                interval = 60.0 if source.source_type == "jpeg_snapshot" else 1 / max(source.fps_limit, 1)
                if now - state.last_analyzed_at < interval:
                    continue
                state.last_analyzed_at = now
                try:
                    stream_source = StreamSource(
                        camera_id=source.id,
                        name=source.name,
                        url=source.url,
                        stream_type="mjpeg" if source.source_type == "live_mjpeg" else source.source_type,
                    )
                    frame = grab_frame(stream_source)
                    if frame is None:
                        self._analysis_status[source.id] = "warming_up"
                        self._rules.categories[source.id] = "unknown"
                        self._mark_status(state, online=False, error="No frame available")
                        continue
                    FrameCache.get_instance().update(frame)
                    self._mark_status(state, online=True, error=None)
                    detections, annotated = self._analyze(source, frame)
                    tracks = self._tracker.update(source.id, detections, source.zones, (frame.width, frame.height))
                    for track in tracks:
                        db.upsert_track(track.to_model())
                    if self._analysis_status.get(source.id) == "running" and source.source_type != "jpeg_snapshot":
                        self._rules.evaluate(source, tracks, settings)
                    elif source.source_type == "retail_scene":
                        self._rules.categories[source.id] = "demo"
                    else:
                        self._rules.categories[source.id] = "unknown"
                    with self._lock:
                        self._analysis_jpeg[source.id] = annotated
                except Exception as exc:
                    self._analysis_status[source.id] = "warming_up"
                    self._rules.categories[source.id] = "unknown"
                    logger.warning("Camera runtime error for %s: %s", source.id, exc)
                    self._mark_status(state, online=False, error=str(exc))
            self._stop.wait(0.05)

    def _mark_status(self, state: RuntimeCameraState, *, online: bool, error: str | None) -> None:
        state.last_error = error
        state.status = StreamStatus(online=online, has_frame=online, last_frame_at=utc_now() if online else state.status.last_frame_at, error=error)
        db.update_camera_runtime(
            state.source.id,
            online=online,
            ai_status="running" if online and self._analysis_status.get(state.source.id) == "running" else "disabled",
            fps=state.source.fps_limit if online else 0,
            quality_score=88 if online else 40,
            error=error,
        )

    def _analyze(self, source: CameraSource, frame: CapturedFrame) -> tuple[list[RuntimeDetection], bytes]:
        result = analyze_frame(source.id, frame.numpy_frame, db.load_settings().confidence_threshold)
        self._analysis_status[source.id] = "demo" if source.source_type == "retail_scene" else ("running" if result.model_available else "model_unavailable")
        detections = [
            RuntimeDetection(
                bbox=[det.x1, det.y1, det.x2, det.y2],
                confidence=det.confidence,
                class_name=det.class_name,
            )
            for det in result.detections
            if det.class_name == "person" or det.class_id == 0
        ]
        if not detections and source.source_type == "retail_scene":
            detections = self._fallback_demo_detections(source.id, frame.width, frame.height)
        annotated = draw_detections(frame.numpy_frame, result) if result.detections else frame.numpy_frame
        return detections, frame_to_jpeg(annotated)

    @staticmethod
    def _fallback_demo_detections(camera_id: str, width: int, height: int) -> list[RuntimeDetection]:
        phase = int(time.time() * 30 + abs(hash(camera_id)) % 120) % max(width, 1)
        x = max(20, min(width - 90, phase))
        y1 = int(height * 0.48)
        y2 = int(height * 0.86)
        return [RuntimeDetection(bbox=[x, y1, x + 70, y2], confidence=0.64)]


runtime = CameraRuntime()
