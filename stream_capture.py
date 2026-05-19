"""Capture frames from public MJPEG / JPEG webcam streams."""

from __future__ import annotations

import io
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import ClassVar
from urllib.error import URLError
from urllib.request import Request, urlopen

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_TIMEOUT = 8  # seconds per HTTP request
_USER_AGENT = "AiVideoMonitoring/1.0"


@dataclass
class StreamSource:
    camera_id: str
    name: str
    url: str
    stream_type: str = "mjpeg"  # mjpeg | jpeg_snapshot


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


LIVE_STREAMS: list[StreamSource] = [
    StreamSource(
        camera_id="cam-buffalo-trace",
        name="Buffalo Trace Factory (USA)",
        url="http://camera.buffalotrace.com/mjpg/video.mjpg",
        stream_type="mjpeg",
    ),
    StreamSource(
        camera_id="cam-purdue-mall",
        name="Purdue Engineering Mall (USA)",
        url="http://webcam01.ecn.purdue.edu/mjpg/video.mjpg",
        stream_type="mjpeg",
    ),
    StreamSource(
        camera_id="cam-kirchhoff-physics",
        name="Kirchhoff Institute Physics (Germany)",
        url="http://pendelcam.kip.uni-heidelberg.de/mjpg/video.mjpg",
        stream_type="mjpeg",
    ),
    StreamSource(
        camera_id="cam-hotel-lobby",
        name="Hotel Lobby CCTV",
        url="http://158.58.130.148/mjpg/video.mjpg",
        stream_type="mjpeg",
    ),
    StreamSource(
        camera_id="cam-pajala-sweden",
        name="Soltorget Pajala (Sweden)",
        url="http://195.196.36.242/mjpg/video.mjpg",
        stream_type="mjpeg",
    ),
    StreamSource(
        camera_id="cam-piano-japan",
        name="Piano Factory (Japan)",
        url="http://takemotopiano.aa1.netvolante.jp:8190/nphMotionJpeg?Resolution=640x480&Quality=Standard&Framerate=30",
        stream_type="mjpeg",
    ),
]


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


def grab_frame(source: StreamSource) -> CapturedFrame | None:
    """Grab one frame from a stream source and return it."""
    if source.stream_type == "jpeg_snapshot":
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
