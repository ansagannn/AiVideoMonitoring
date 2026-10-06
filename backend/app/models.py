from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field, model_validator

EventType = Literal[
    "normal", "suspicious", "fight", "fall", "danger",
    "employee_absence",
    "employee_presence",
    "visitor_shelf_dwell",
    "hand_to_body",
    "back_to_camera",
    "system_stream_lost",
]
EventStatus = Literal["new", "confirmed", "dismissed"]
Severity = Literal["low", "medium", "high"]
CameraStatus = Literal["online", "unstable", "offline"]
AIStatus = Literal["running", "warming_up", "disabled"]
SourceType = Literal[
    "demo_video",
    "rtsp",
    "mock_rtsp",
    "public_dataset",
    "public_webcam_archive",
    "live_mjpeg",
    "retail_scene",
    "jpeg_snapshot",
    "hls",
]
ZoneKind = Literal["work_area", "shelf", "entrance", "checkout", "stock", "restricted"]

SUSPICIOUS_EVENT_TYPES: set[EventType] = {"suspicious", "visitor_shelf_dwell", "hand_to_body", "back_to_camera"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


class Zone(BaseModel):
    id: str
    name: str
    kind: ZoneKind
    polygon: list[tuple[float, float]] | None = None

    @model_validator(mode="after")
    def validate_polygon(self):
        if self.kind == "restricted" and not self.polygon:
            raise ValueError("Restricted zone needs a polygon")
        if self.polygon:
            if len(self.polygon) < 3 or any(not (0 <= x <= 1 and 0 <= y <= 1) for x,y in self.polygon):
                raise ValueError("Polygon needs at least three normalized points")
            area = abs(sum(self.polygon[i][0]*self.polygon[(i+1)%len(self.polygon)][1] - self.polygon[(i+1)%len(self.polygon)][0]*self.polygon[i][1] for i in range(len(self.polygon))))
            if area < 1e-6:
                raise ValueError("Polygon must have a non-zero area")
        return self


class Camera(BaseModel):
    id: str
    name: str
    location: str
    rtsp_url: str
    status: CameraStatus
    ai_status: AIStatus
    fps: int
    zones: list[Zone]
    last_seen_at: str
    source_type: SourceType
    quality_score: int
    uptime_minutes: int
    last_event_title: str | None = None
    last_event_at: str | None = None


class CameraSource(BaseModel):
    id: str
    name: str
    location: str = ""
    url: str
    source_type: SourceType = "rtsp"
    enabled: bool = True
    fps_limit: int = Field(default=8, ge=1, le=30)
    zones: list[Zone] = Field(default_factory=list)


class CameraSourceCreate(BaseModel):
    id: str
    name: str
    location: str = ""
    url: str
    source_type: SourceType = "rtsp"
    enabled: bool = True
    fps_limit: int = Field(default=8, ge=1, le=30)
    zones: list[Zone] = Field(default_factory=list)


class VideoEvent(BaseModel):
    id: str
    camera_id: str
    camera_name: str
    type: EventType
    severity: Severity
    title: str
    description: str
    zone: str
    detected_at: str
    snapshot_url: str
    status: EventStatus
    confidence: float
    feedback_note: str | None = None
    reviewed_by: str | None = None
    reviewed_at: str | None = None
    reaction_seconds: int | None = None
    telegram_sent: bool = False
    detection_mode: str = "legacy"
    analysis_summary: str
    evidence_tags: list[str]


class Track(BaseModel):
    id: str
    camera_id: str
    local_track_id: int
    first_seen_at: str
    last_seen_at: str
    zone_id: str | None = None
    zone_name: str | None = None
    confidence: float
    bbox: list[int]
    active: bool


class PresenceSession(BaseModel):
    id: str
    camera_id: str
    track_id: str
    zone_id: str
    started_at: str
    ended_at: str | None = None
    first_seen_at: str
    last_seen_at: str


class MonitoringSettings(BaseModel):
    absence_threshold_minutes: int = Field(default=5, ge=1, le=120)
    shelf_dwell_seconds: int = Field(default=45, ge=5, le=600)
    fall_hold_seconds: float = Field(default=2.0, ge=0.5, le=10)
    fight_motion_threshold: float = Field(default=0.8, ge=0.1, le=5)
    confidence_threshold: float = Field(default=0.6, ge=0.1, le=0.99)


class SimulateEventRequest(BaseModel):
    camera_id: str = "cam-hypermarket-frozen"
    event_type: EventType | None = None
    description: str | None = None


class FeedbackRequest(BaseModel):
    status: EventStatus
    reviewed_by: str = "operator"
    note: str | None = None


class TelegramButton(BaseModel):
    label: str
    action: EventStatus
    callback_data: str


class TelegramPreview(BaseModel):
    mode: Literal["telegram", "mock"]
    text: str
    buttons: list[TelegramButton]


class TelegramTestResponse(BaseModel):
    configured: bool
    sent: bool
    mode: Literal["telegram", "mock"]
    detail: str
    inline_feedback: bool
    preview: TelegramPreview


class ShiftAnalytics(BaseModel):
    report_date: str
    shift_started_at: str
    total_events: int
    open_events: int
    confirmed_events: int
    dismissed_events: int
    absence_events: int
    suspicious_events: int
    average_reaction_seconds: int | None
    cameras_online: int
    cameras_total: int
    telegram_configured: bool


class PublicVideoSource(BaseModel):
    id: str
    title: str
    camera_id: str
    source_url: str
    scenario: str
    license_note: str
    supported_signals: list[str]


class DetectionCapability(BaseModel):
    id: str
    title: str
    readiness: Literal["demo_ready", "heuristic_ready", "pilot_needed"]
    confidence: float
    what_it_checks: str
    evidence: list[str]
    current_limitations: str
    tz_mapping: str


class StreamStatus(BaseModel):
    online: bool
    has_frame: bool
    last_frame_at: str | None = None
    error: str | None = None
