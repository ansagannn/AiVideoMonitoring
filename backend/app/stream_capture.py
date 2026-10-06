"""Capture frames from public MJPEG / JPEG webcam streams + synthetic retail scenes."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import ClassVar
from urllib.error import URLError
from urllib.request import Request, urlopen

import cv2
import numpy as np

from retail_scenes import render_scene_frame

logger = logging.getLogger(__name__)

_TIMEOUT = 8  # seconds per HTTP request
_USER_AGENT = "AiVideoMonitoring/1.0"


@dataclass
class StreamSource:
    camera_id: str
    name: str
    url: str
    stream_type: str = "mjpeg"  # mjpeg | jpeg_snapshot | retail_scene


@dataclass
class CapturedFrame:
    camera_id: str
    jpeg_bytes: bytes
    numpy_frame: np.ndarray
    captured_at: float = field(default_factory=time.time)
    width: int = 0
    height: int = 0

    def __post_init__(self) -> None:
        if self.numpy_frame is not None:
            self.height, self.width = self.numpy_frame.shape[:2]


# Public MJPEG sources are kept as opportunistic fallbacks. Many of them are
# flaky or geo-restricted, so the dashboard's primary cameras are now the
# synthetic retail scenes below.
PUBLIC_MJPEG_STREAMS: list[StreamSource] = [
    StreamSource(
        camera_id="cam-buffalo-trace",
        name="Buffalo Trace Factory (USA)",
        url="http://camera.buffalotrace.com/mjpg/video.mjpg",
        stream_type="mjpeg",
    ),
]

RETAIL_SCENE_STREAMS: list[StreamSource] = [
    StreamSource(
        camera_id="cam-hypermarket-frozen",
        name="Hypermarket — Frozen Aisle",
        url="retail-scene://cam-hypermarket-frozen",
        stream_type="retail_scene",
    ),
    StreamSource(
        camera_id="cam-supermarket-produce",
        name="Supermarket — Fresh Produce",
        url="retail-scene://cam-supermarket-produce",
        stream_type="retail_scene",
    ),
    StreamSource(
        camera_id="cam-supermarket-beverage",
        name="Supermarket — Beverages Aisle",
        url="retail-scene://cam-supermarket-beverage",
        stream_type="retail_scene",
    ),
    StreamSource(
        camera_id="cam-supermarket-checkout",
        name="Supermarket — Checkout Lanes",
        url="retail-scene://cam-supermarket-checkout",
        stream_type="retail_scene",
    ),
    StreamSource(
        camera_id="cam-warehouse-stock",
        name="Warehouse — Stock Backroom",
        url="retail-scene://cam-warehouse-stock",
        stream_type="retail_scene",
    ),
    StreamSource(
        camera_id="cam-mall-entrance",
        name="Shopping Mall — Main Entrance",
        url="retail-scene://cam-mall-entrance",
        stream_type="retail_scene",
    ),
]

LIVE_STREAMS: list[StreamSource] = RETAIL_SCENE_STREAMS + PUBLIC_MJPEG_STREAMS


def _grab_jpeg_snapshot(url: str) -> bytes | None:
    """Fetch a single JPEG frame from an HTTP JPEG snapshot URL."""
    try:
        req = Request(url, headers={"User-Agent": _USER_AGENT})
        with urlopen(req, timeout=_TIMEOUT) as resp:
            return resp.read()
    except (URLError, OSError, TimeoutError) as exc:
        logger.warning("Failed to grab snapshot from %s: %s", url, exc)
        return None


def _grab_mjpeg_frame(url: str) -> bytes | None:
    """Read exactly one JPEG frame from an MJPEG stream."""
    try:
        req = Request(url, headers={"User-Agent": _USER_AGENT})
        with urlopen(req, timeout=_TIMEOUT) as resp:
            buf = b""
            start_found = False
            while True:
                chunk = resp.read(4096)
                if not chunk:
                    break
                buf += chunk
                if not start_found:
                    soi = buf.find(b"\xff\xd8")
                    if soi >= 0:
                        buf = buf[soi:]
                        start_found = True
                if start_found:
                    eoi = buf.find(b"\xff\xd9")
                    if eoi >= 0:
                        return buf[: eoi + 2]
                if len(buf) > 2_000_000:
                    break
        return None
    except (URLError, OSError, TimeoutError) as exc:
        logger.warning("Failed to grab MJPEG frame from %s: %s", url, exc)
        return None


_captures: dict[str, tuple[str, object]] = {}


def close_captures(active_ids=None):
    for camera_id in list(_captures):
        if active_ids is None or camera_id not in active_ids:
            _captures.pop(camera_id)[1].release()


def _grab_video_frame(camera_id: str, url: str, loop: bool = False) -> bytes | None:
    entry = _captures.get(camera_id)
    if entry and entry[0] != url:
        entry[1].release()
        entry = None
    if entry is None:
        capture = cv2.VideoCapture()
        capture.open(url, cv2.CAP_FFMPEG, [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 3000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 3000])
        _captures[camera_id] = (url, capture)
    else:
        capture = entry[1]
    if not capture.isOpened():
        close_captures(set(_captures) - {camera_id})
        return None
    success, frame = capture.read()
    if not success and loop:
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        success, frame = capture.read()
    if not success or frame is None:
        close_captures(set(_captures) - {camera_id})
        return None
    _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return jpeg.tobytes()


def grab_frame(source: StreamSource) -> CapturedFrame | None:
    """Grab one frame from a stream source and return it."""
    if source.stream_type == "retail_scene" or source.url.startswith("retail-scene://"):
        # Allow using a retail-scene URL that names a different scene than the
        # camera id (e.g. url="retail-scene://cam-mall-entrance"). Extract the
        # scene id from the URL when present; fall back to the camera id.
        scene_id = source.url.split("://", 1)[1] if "//" in source.url else source.camera_id
        frame = render_scene_frame(scene_id)
        if frame is None:
            logger.warning("Retail scene %s returned no frame", source.camera_id)
            return None
        _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return CapturedFrame(
            camera_id=source.camera_id,
            jpeg_bytes=jpeg.tobytes(),
            numpy_frame=frame,
        )

    if source.stream_type in {"rtsp", "hls"} or source.url.startswith("rtsp://"):
        raw = _grab_video_frame(source.camera_id, source.url)
    elif source.stream_type in {"demo_video", "public_dataset", "public_webcam_archive"}:
        raw = _grab_video_frame(source.camera_id, source.url, loop=True)
    elif source.stream_type == "jpeg_snapshot":
        raw = _grab_jpeg_snapshot(source.url)
    else:
        raw = _grab_mjpeg_frame(source.url)

    if raw is None:
        return None

    arr = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        logger.warning("Failed to decode frame from %s", source.camera_id)
        return None

    _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return CapturedFrame(
        camera_id=source.camera_id,
        jpeg_bytes=jpeg.tobytes(),
        numpy_frame=frame,
    )


class FrameCache:
    """Thread-safe cache of the latest frame per camera."""

    _instance: ClassVar[FrameCache | None] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self) -> None:
        self._frames: dict[str, CapturedFrame] = {}
        self._frame_lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> FrameCache:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def update(self, frame: CapturedFrame) -> None:
        with self._frame_lock:
            self._frames[frame.camera_id] = frame

    def get(self, camera_id: str) -> CapturedFrame | None:
        with self._frame_lock:
            return self._frames.get(camera_id)

    def get_all(self) -> dict[str, CapturedFrame]:
        with self._frame_lock:
            return dict(self._frames)

    def is_online(self, camera_id: str) -> bool:
        frame = self.get(camera_id)
        if frame is None:
            return False
        return (time.time() - frame.captured_at) < 120
