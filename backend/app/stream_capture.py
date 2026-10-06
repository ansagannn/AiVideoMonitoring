"""Capture frames from public MJPEG / JPEG webcam streams + synthetic retail scenes."""

from __future__ import annotations

import logging
import json
from urllib.parse import urlsplit
import subprocess
import re
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
    for camera_id in list(_hls_captures):
        if active_ids is None or camera_id not in active_ids:
            _hls_captures.pop(camera_id).release()
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

    if source.stream_type in {"hls", "live_mjpeg"}:
        raw = _grab_hls_frame(source.camera_id, source.url)
    elif source.stream_type == "rtsp" or source.url.startswith("rtsp://"):
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


class HlsCapture:
    """Drain FFmpeg continuously so low-rate inference always sees a recent frame."""
    def __init__(self, url):
        self.url, self.frame, self.at = url, None, 0.0
        self.error, self.started_at = None, time.monotonic()
        self.process = subprocess.Popen([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
            "-rw_timeout", "15000000", "-threads", "1", "-i", url, "-an", "-threads", "1", "-filter_threads", "1",
            "-vf", "fps=2,scale=640:-2", "-c:v", "mjpeg", "-q:v", "5",
            "-f", "image2pipe", "pipe:1"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        threading.Thread(target=self._read, daemon=True).start()
        threading.Thread(target=self._errors, daemon=True).start()

    def _read(self):
        buf = bytearray()
        while self.process.stdout:
            chunk = self.process.stdout.read1(16384)
            if not chunk:
                break
            buf.extend(chunk)
            while True:
                start = buf.find(b"\xff\xd8")
                end = buf.find(b"\xff\xd9", max(start, 0))
                if start < 0 or end < 0:
                    break
                self.frame, self.at = bytes(buf[start:end+2]), time.monotonic()
                del buf[:end+2]
            if len(buf) > 2_000_000:
                buf.clear()

    def _errors(self):
        for line in iter(self.process.stderr.readline, b""):
            # Do not log URLs, which can contain camera credentials.
            message = line.decode("utf-8", errors="replace")
            http = re.search(r"(?:HTTP error |Server returned )(\d{3})", message)
            reason = f"HTTP {http.group(1)}" if http else next((r for r in (
                "Connection timed out", "Connection refused", "Invalid data found",
                "Protocol not found", "Option not found", "Input/output error",
                "Error while opening decoder", "No route to host", "End of file",
                "TLS handshake failed", "Certificate verification failed") if r.lower() in message.lower()), None)
            self.error = reason or "FFmpeg could not decode the HLS stream"
            if reason:
                logger.warning("Public HLS capture: %s", reason)
        if self.process.poll() not in (None, 0):
            logger.warning("HLS decoder exited with code %s", self.process.returncode)

    def release(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2)


_hls_captures = {}
_hls_retry_at = {}


def _grab_hls_frame(camera_id, url):
    now = time.monotonic()
    if urlsplit(url).hostname == "cwwp2.dot.ca.gov" and url.endswith(".json"):
        url = _catalog_stream(camera_id, url)
        if not url:
            return None
    entry = _hls_captures.get(camera_id)
    if entry and (entry.url != url or entry.process.poll() is not None or
                  now - max(entry.at, entry.started_at) > 30):
        entry.release()
        _hls_captures.pop(camera_id, None)
        _hls_retry_at[camera_id] = now + 10
        entry = None
    if entry is None:
        if now < _hls_retry_at.get(camera_id, 0):
            return None
        entry = _hls_captures[camera_id] = HlsCapture(url)
    return entry.frame if now-entry.at < 5 else None


_catalogs = {}


def _catalog_stream(camera_id, catalog_url):
    """Resolve only official Caltrans URLs; retry another camera after decoder failure."""
    now = time.monotonic()
    state = _catalogs.get(camera_id)
    if state is None or now-state["loaded"] > 3600:
        try:
            with urlopen(Request(catalog_url, headers={"User-Agent": _USER_AGENT}), timeout=8) as response:
                document = json.loads(response.read(5_000_000))
            urls = []
            def visit(node):
                if isinstance(node, dict):
                    for key, value in node.items():
                        if key.lower() == "streamingvideourl" and isinstance(value, str):
                            parsed = urlsplit(value)
                            if parsed.scheme == "https" and parsed.hostname == "wzmedia.dot.ca.gov" and ".m3u8" in parsed.path:
                                urls.append(value)
                        elif isinstance(value, (dict, list)):
                            visit(value)
                elif isinstance(node, list):
                    for child in node:
                        visit(child)
            visit(document)
            state = _catalogs[camera_id] = {"loaded":now, "urls":list(dict.fromkeys(urls)), "index":0}
            logger.info("Caltrans public catalog resolved %s video streams", len(state["urls"]))
        except (URLError, OSError, ValueError) as exc:
            logger.warning("Caltrans public camera catalog unavailable: %s", type(exc).__name__)
            _catalogs[camera_id] = {"loaded":now-3540, "urls":[], "index":0}
            return None
    urls = state["urls"]
    if not urls:
        return None
    capture = _hls_captures.get(camera_id)
    if capture and (capture.process.poll() is not None or now-max(capture.at,capture.started_at)>30):
        capture.release()
        _hls_captures.pop(camera_id, None)
        state["index"] = (state["index"]+1) % len(urls)
        _hls_retry_at[camera_id] = now+2
    return urls[state["index"]]
