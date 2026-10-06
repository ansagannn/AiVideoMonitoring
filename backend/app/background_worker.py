"""Background worker: periodically captures frames and runs YOLOv8 analysis."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

from analyzer import AnalysisResult, analyze_frame, draw_detections, frame_to_jpeg
from stream_capture import (
    CapturedFrame,
    FrameCache,
    StreamSource,
    grab_frame,
)

logger = logging.getLogger(__name__)

RETAIL_CAPTURE_INTERVAL = 1.0   # local scenes are cheap, animate ~1 Hz
REMOTE_CAPTURE_INTERVAL = 15.0  # remote MJPEG is expensive / rate-limited
ANALYSIS_INTERVAL = 5.0         # YOLO analysis cadence per camera
ANALYSIS_WARMUP_SECONDS = 2.0   # wait briefly so first frames exist


@dataclass
class AnalysisCache:
    """Thread-safe cache for analysis results + annotated frames."""

    _results: dict[str, AnalysisResult] = field(default_factory=dict)
    _annotated_jpeg: dict[str, bytes] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def update(self, result: AnalysisResult, annotated_jpeg: bytes) -> None:
        with self._lock:
            self._results[result.camera_id] = result
            self._annotated_jpeg[result.camera_id] = annotated_jpeg

    def get_result(self, camera_id: str) -> AnalysisResult | None:
        with self._lock:
            return self._results.get(camera_id)

    def get_annotated_jpeg(self, camera_id: str) -> bytes | None:
        with self._lock:
            return self._annotated_jpeg.get(camera_id)

    def get_all_results(self) -> dict[str, AnalysisResult]:
        with self._lock:
            return dict(self._results)


_analysis_cache = AnalysisCache()
_workers_started = False
_workers_lock = threading.Lock()


def get_analysis_cache() -> AnalysisCache:
    return _analysis_cache


def _capture_loop(sources: list[StreamSource], frame_cache: FrameCache) -> None:
    """Continuously capture frames from all sources.

    Synthetic retail scenes are rendered every RETAIL_CAPTURE_INTERVAL seconds
    so motion looks live; remote MJPEG cameras are polled every
    REMOTE_CAPTURE_INTERVAL seconds to avoid hammering public servers.
    """
    last_remote: dict[str, float] = {s.camera_id: 0.0 for s in sources}
    while True:
        now = time.time()
        for source in sources:
            is_remote = source.stream_type != "retail_scene"
            if is_remote and (now - last_remote.get(source.camera_id, 0.0) < REMOTE_CAPTURE_INTERVAL):
                continue
            try:
                frame = grab_frame(source)
                if frame is not None:
                    frame_cache.update(frame)
                    logger.debug("Captured frame from %s (%dx%d)", source.camera_id, frame.width, frame.height)
                else:
                    logger.debug("No frame from %s", source.camera_id)
            except Exception as exc:
                logger.error("Capture error for %s: %s", source.camera_id, exc)
            finally:
                if is_remote:
                    last_remote[source.camera_id] = time.time()
        time.sleep(RETAIL_CAPTURE_INTERVAL)


def _analysis_loop(sources: list[StreamSource], frame_cache: FrameCache, analysis_cache: AnalysisCache) -> None:
    """Continuously analyze cached frames with YOLOv8."""
    time.sleep(ANALYSIS_WARMUP_SECONDS)
    while True:
        for source in sources:
            try:
                cached = frame_cache.get(source.camera_id)
                if cached is None:
                    continue
                result = analyze_frame(source.camera_id, cached.numpy_frame)
                annotated = draw_detections(cached.numpy_frame, result)
                annotated_jpeg = frame_to_jpeg(annotated)
                analysis_cache.update(result, annotated_jpeg)
                logger.info(
                    "Analyzed %s: %d people, %.0fms",
                    source.camera_id, result.person_count, result.inference_ms,
                )
            except Exception as exc:
                logger.error("Analysis error for %s: %s", source.camera_id, exc)
        time.sleep(ANALYSIS_INTERVAL)


def start_workers(sources: list[StreamSource] | None = None) -> None:
    """Start capture and analysis threads."""
    global _workers_started
    from stream_capture import LIVE_STREAMS

    if sources is None:
        sources = LIVE_STREAMS

    with _workers_lock:
        if _workers_started:
            logger.info("Background workers already started; skipping duplicate start")
            return
        _workers_started = True

    frame_cache = FrameCache.get_instance()
    analysis_cache = get_analysis_cache()

    capture_thread = threading.Thread(
        target=_capture_loop,
        args=(sources, frame_cache),
        daemon=True,
        name="frame-capture",
    )
    analysis_thread = threading.Thread(
        target=_analysis_loop,
        args=(sources, frame_cache, analysis_cache),
        daemon=True,
        name="frame-analysis",
    )

    capture_thread.start()
    analysis_thread.start()
    logger.info("Background workers started: capture + analysis for %d cameras", len(sources))
