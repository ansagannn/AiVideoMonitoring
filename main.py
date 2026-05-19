from datetime import datetime, timedelta, timezone
import csv
import html
import io
import json
import logging
import os
from pathlib import Path
import sqlite3
from typing import Literal
from urllib import request
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Video Monitoring MVP")

# Disable CORS. Do not remove this for full-stack development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if (FRONTEND_DIST / "assets").exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

DB_PATH = Path(os.getenv("AI_MONITORING_DB_PATH", Path(__file__).resolve().parents[1] / "monitoring.db"))

EventType = Literal[
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
SourceType = Literal["demo_video", "rtsp", "mock_rtsp", "public_dataset", "public_webcam_archive", "live_mjpeg"]

SUSPICIOUS_EVENT_TYPES: set[EventType] = {"visitor_shelf_dwell", "hand_to_body", "back_to_camera"}


def utc_now_static() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class Zone(BaseModel):
    id: str
    name: str
    kind: Literal["work_area", "shelf", "entrance", "checkout", "stock"]


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
    analysis_summary: str
    evidence_tags: list[str]


class MonitoringSettings(BaseModel):
    absence_threshold_minutes: int = Field(default=5, ge=1, le=120)
    shelf_dwell_seconds: int = Field(default=45, ge=5, le=600)
    confidence_threshold: float = Field(default=0.6, ge=0.1, le=0.99)


class SimulateEventRequest(BaseModel):
    camera_id: str = "cam-buffalo-trace"
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


DEFAULT_CAMERAS: list[Camera] = [
    Camera(
        id="cam-buffalo-trace",
        name="Buffalo Trace Factory (USA)",
        location="Завод / публичная live-камера",
        rtsp_url="http://camera.buffalotrace.com/mjpg/video.mjpg",
        status="online",
        ai_status="running",
        fps=15,
        zones=[
            Zone(id="zone-bt-area", name="Производственная зона", kind="work_area"),
            Zone(id="zone-bt-entrance", name="Вход", kind="entrance"),
        ],
        last_seen_at=utc_now_static(),
        source_type="live_mjpeg",
        quality_score=82,
        uptime_minutes=0,
    ),
    Camera(
        id="cam-purdue-mall",
        name="Purdue Engineering Mall (USA)",
        location="Университетский кампус / публичная live-камера",
        rtsp_url="http://webcam01.ecn.purdue.edu/mjpg/video.mjpg",
        status="online",
        ai_status="running",
        fps=10,
        zones=[
            Zone(id="zone-purdue-walk", name="Пешеходная зона", kind="entrance"),
            Zone(id="zone-purdue-area", name="Общая зона", kind="work_area"),
        ],
        last_seen_at=utc_now_static(),
        source_type="live_mjpeg",
        quality_score=78,
        uptime_minutes=0,
    ),
    Camera(
        id="cam-kirchhoff-physics",
        name="Kirchhoff Institute Physics (Germany)",
        location="Физический институт / публичная live-камера",
        rtsp_url="http://pendelcam.kip.uni-heidelberg.de/mjpg/video.mjpg",
        status="online",
        ai_status="running",
        fps=5,
        zones=[
            Zone(id="zone-ki-lab", name="Лабораторная зона", kind="work_area"),
        ],
        last_seen_at=utc_now_static(),
        source_type="live_mjpeg",
        quality_score=75,
        uptime_minutes=0,
    ),
    Camera(
        id="cam-hotel-lobby",
        name="Hotel Lobby CCTV",
        location="Лобби отеля / публичная CCTV-камера",
        rtsp_url="http://158.58.130.148/mjpg/video.mjpg",
        status="online",
        ai_status="running",
        fps=12,
        zones=[
            Zone(id="zone-lobby-entrance", name="Вход в лобби", kind="entrance"),
            Zone(id="zone-lobby-reception", name="Ресепшн", kind="work_area"),
        ],
        last_seen_at=utc_now_static(),
        source_type="live_mjpeg",
        quality_score=70,
        uptime_minutes=0,
    ),
    Camera(
        id="cam-pajala-sweden",
        name="Soltorget Pajala (Sweden)",
        location="Городская площадь / публичная live-камера",
        rtsp_url="http://195.196.36.242/mjpg/video.mjpg",
        status="online",
        ai_status="running",
        fps=8,
        zones=[
            Zone(id="zone-pajala-square", name="Площадь", kind="entrance"),
        ],
        last_seen_at=utc_now_static(),
        source_type="live_mjpeg",
        quality_score=72,
        uptime_minutes=0,
    ),
    Camera(
        id="cam-piano-japan",
        name="Piano Factory (Japan)",
        location="Фабрика пианино / публичная live-камера",
        rtsp_url="http://takemotopiano.aa1.netvolante.jp:8190/nphMotionJpeg?Resolution=640x480&Quality=Standard&Framerate=30",
        status="online",
        ai_status="running",
        fps=30,
        zones=[
            Zone(id="zone-piano-workshop", name="Мастерская", kind="work_area"),
            Zone(id="zone-piano-stock", name="Склад", kind="stock"),
        ],
        last_seen_at=utc_now_static(),
        source_type="live_mjpeg",
        quality_score=80,
        uptime_minutes=0,
    ),
]

PUBLIC_VIDEO_SOURCES: list[PublicVideoSource] = [
    PublicVideoSource(
        id="buffalo-trace-live",
        title="Buffalo Trace Factory — live MJPEG",
        camera_id="cam-buffalo-trace",
        source_url="http://camera.buffalotrace.com/mjpg/video.mjpg",
        scenario="Живая камера завода в США, YOLOv8 детекция людей.",
        license_note="Публичная камера.",
        supported_signals=["детекция людей", "подсчёт", "контроль присутствия"],
    ),
    PublicVideoSource(
        id="purdue-campus-live",
        title="Purdue Engineering Mall — live MJPEG",
        camera_id="cam-purdue-mall",
        source_url="http://webcam01.ecn.purdue.edu/mjpg/video.mjpg",
        scenario="Кампус Purdue, подсчёт людей и контроль зоны.",
        license_note="Публичная университетская камера.",
        supported_signals=["детекция людей", "crowd counting", "контроль зоны"],
    ),
    PublicVideoSource(
        id="kirchhoff-physics-live",
        title="Kirchhoff Institute — live MJPEG",
        camera_id="cam-kirchhoff-physics",
        source_url="http://pendelcam.kip.uni-heidelberg.de/mjpg/video.mjpg",
        scenario="Физический институт, камера с маятником.",
        license_note="Публичная камера университета.",
        supported_signals=["детекция объектов", "анализ движения"],
    ),
    PublicVideoSource(
        id="hotel-lobby-live",
        title="Hotel Lobby CCTV — live MJPEG",
        camera_id="cam-hotel-lobby",
        source_url="http://158.58.130.148/mjpg/video.mjpg",
        scenario="Лобби отеля, контроль входа, детекция людей.",
        license_note="Публичная CCTV-камера.",
        supported_signals=["детекция людей", "контроль входа", "анализ потока"],
    ),
    PublicVideoSource(
        id="pajala-sweden-live",
        title="Soltorget Pajala — live MJPEG",
        camera_id="cam-pajala-sweden",
        source_url="http://195.196.36.242/mjpg/video.mjpg",
        scenario="Городская площадь в Швеции, outdoor monitoring.",
        license_note="Публичная городская камера.",
        supported_signals=["детекция людей", "детекция транспорта", "outdoor monitoring"],
    ),
    PublicVideoSource(
        id="piano-japan-live",
        title="Piano Factory (Japan) — live MJPEG",
        camera_id="cam-piano-japan",
        source_url="http://takemotopiano.aa1.netvolante.jp:8190/nphMotionJpeg?Resolution=640x480&Quality=Standard&Framerate=30",
        scenario="Фабрика пианино, контроль присутствия.",
        license_note="Публичная камера фабрики.",
        supported_signals=["детекция людей", "контроль рабочей зоны", "подсчёт"],
    ),
]

DETECTION_CAPABILITIES: list[DetectionCapability] = [
    DetectionCapability(
        id="people-bbox",
        title="Детекция людей в кадре",
        readiness="demo_ready",
        confidence=0.88,
        what_it_checks="Есть ли человек в рабочей зоне или зоне полок.",
        evidence=["bounding box человека", "зона камеры", "confidence", "timestamp"],
        current_limitations="Используется YOLOv8n на реальных MJPEG-стримах; точность зависит от качества камеры.",
        tz_mapping="Недели 2-4: детекция людей и базовый трекинг внутри потока.",
    ),
    DetectionCapability(
        id="employee-absence",
        title="Отсутствие сотрудника",
        readiness="heuristic_ready",
        confidence=0.82,
        what_it_checks="Не появляется ли человек/сотрудник в рабочей зоне дольше заданного порога.",
        evidence=["последнее появление", "порог минут", "длительность отсутствия"],
        current_limitations="Без точной идентификации личности; ориентир на форму/зону/роль.",
        tz_mapping="Недели 4-6: логика присутствия, отсутствие N времени, первый/последний кадр.",
    ),
    DetectionCapability(
        id="shelf-dwell",
        title="Долгое нахождение у полки",
        readiness="heuristic_ready",
        confidence=0.76,
        what_it_checks="Посетитель находится у полки дольше порога и частично закрывает обзор.",
        evidence=["bbox в зоне shelf", "dwell seconds", "направление корпуса", "snapshot"],
        current_limitations="Требует проверки оператором; это событие-кандидат, не обвинение.",
        tz_mapping="Подозрительные действия: долго у полки, спиной к камере.",
    ),
    DetectionCapability(
        id="hand-to-body",
        title="Рука к телу/сумке",
        readiness="pilot_needed",
        confidence=0.69,
        what_it_checks="Повторяющееся движение руки от полки к телу/сумке как risk-сигнал.",
        evidence=["зона рук", "траектория движения", "2-3 последовательных кадра", "confidence"],
        current_limitations="Нужны реальные данные магазина и human feedback для снижения ложных срабатываний.",
        tz_mapping="Подозрительные действия: движение руки к телу или сумке.",
    ),
    DetectionCapability(
        id="multi-camera-check",
        title="Сверка с другой камерой",
        readiness="pilot_needed",
        confidence=0.64,
        what_it_checks="Есть ли похожее событие с другой камеры той же зоны в близкое время.",
        evidence=["camera_id", "zone", "timestamp window", "одинаковый тип события"],
        current_limitations="В MVP без re-identification; не смешивает треки между потоками.",
        tz_mapping="Работа с несколькими камерами без путаницы между людьми с разных камер.",
    ),
]

DEFAULT_SETTINGS = MonitoringSettings()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def db_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db() -> None:
    with db_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS cameras (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                location TEXT NOT NULL,
                rtsp_url TEXT NOT NULL,
                status TEXT NOT NULL,
                ai_status TEXT NOT NULL,
                fps INTEGER NOT NULL,
                zones TEXT NOT NULL,
                last_seen_at TEXT NOT NULL,
                source_type TEXT NOT NULL,
                quality_score INTEGER NOT NULL,
                uptime_minutes INTEGER NOT NULL,
                last_event_title TEXT,
                last_event_at TEXT
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id TEXT PRIMARY KEY,
                camera_id TEXT NOT NULL,
                camera_name TEXT NOT NULL,
                type TEXT NOT NULL,
                severity TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                zone TEXT NOT NULL,
                detected_at TEXT NOT NULL,
                snapshot_url TEXT NOT NULL,
                status TEXT NOT NULL,
                confidence REAL NOT NULL,
                feedback_note TEXT,
                reviewed_by TEXT,
                reviewed_at TEXT,
                reaction_seconds INTEGER,
                telegram_sent INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        for camera in DEFAULT_CAMERAS:
            connection.execute(
                """
                INSERT OR IGNORE INTO cameras (
                    id, name, location, rtsp_url, status, ai_status, fps, zones,
                    last_seen_at, source_type, quality_score, uptime_minutes,
                    last_event_title, last_event_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    camera.id,
                    camera.name,
                    camera.location,
                    camera.rtsp_url,
                    camera.status,
                    camera.ai_status,
                    camera.fps,
                    json.dumps([zone.model_dump() for zone in camera.zones], ensure_ascii=False),
                    camera.last_seen_at,
                    camera.source_type,
                    camera.quality_score,
                    camera.uptime_minutes,
                    camera.last_event_title,
                    camera.last_event_at,
                ),
            )
        for key, value in DEFAULT_SETTINGS.model_dump().items():
            connection.execute(
                "INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)",
                (key, str(value)),
            )
        connection.commit()
    seed_events_if_needed()


def row_to_camera(row: sqlite3.Row) -> Camera:
    zones = [Zone.model_validate(zone) for zone in json.loads(row["zones"])]
    return Camera(
        id=row["id"],
        name=row["name"],
        location=row["location"],
        rtsp_url=row["rtsp_url"],
        status=row["status"],
        ai_status=row["ai_status"],
        fps=row["fps"],
        zones=zones,
        last_seen_at=row["last_seen_at"],
        source_type=row["source_type"],
        quality_score=row["quality_score"],
        uptime_minutes=row["uptime_minutes"],
        last_event_title=row["last_event_title"],
        last_event_at=row["last_event_at"],
    )


def event_analysis(event_type: EventType) -> tuple[str, list[str]]:
    mapping: dict[EventType, tuple[str, list[str]]] = {
        "employee_absence": (
            "Система сверяет последнее появление сотрудника в рабочей зоне с порогом отсутствия.",
            ["рабочая зона", "last seen", "порог отсутствия", "timestamp"],
        ),
        "employee_presence": (
            "Зафиксировано появление человека в рабочей зоне; событие может использоваться как приход/возврат.",
            ["bbox человека", "рабочая зона", "первое появление", "confidence"],
        ),
        "visitor_shelf_dwell": (
            "Эвристика выделяет длительное нахождение у полки и отправляет событие на проверку оператору.",
            ["shelf zone", "dwell time", "bbox", "snapshot"],
        ),
        "hand_to_body": (
            "Risk-сигнал: повторное движение руки от полки к телу/сумке, требуется human feedback.",
            ["траектория руки", "зона тела/сумки", "2-3 кадра", "confidence"],
        ),
        "back_to_camera": (
            "Кандидат на подозрительное поведение: человек долго стоит спиной к камере у полки.",
            ["ориентация корпуса", "shelf zone", "длительность", "snapshot"],
        ),
        "system_stream_lost": (
            "Система отслеживает стабильность потока, FPS и качество изображения как обязательный этап ТЗ.",
            ["FPS", "quality score", "last frame", "RTSP status"],
        ),
    }
    return mapping[event_type]


def row_to_event(row: sqlite3.Row) -> VideoEvent:
    analysis_summary, evidence_tags = event_analysis(row["type"])
    return VideoEvent(
        id=row["id"],
        camera_id=row["camera_id"],
        camera_name=row["camera_name"],
        type=row["type"],
        severity=row["severity"],
        title=row["title"],
        description=row["description"],
        zone=row["zone"],
        detected_at=row["detected_at"],
        snapshot_url=row["snapshot_url"],
        status=row["status"],
        confidence=row["confidence"],
        feedback_note=row["feedback_note"],
        reviewed_by=row["reviewed_by"],
        reviewed_at=row["reviewed_at"],
        reaction_seconds=row["reaction_seconds"],
        telegram_sent=bool(row["telegram_sent"]),
        analysis_summary=analysis_summary,
        evidence_tags=evidence_tags,
    )


def camera_by_id(camera_id: str) -> Camera:
    with db_connection() as connection:
        row = connection.execute("SELECT * FROM cameras WHERE id = ?", (camera_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Camera not found")
    return row_to_camera(row)


def event_by_id(event_id: str) -> VideoEvent:
    with db_connection() as connection:
        row = connection.execute("SELECT * FROM events WHERE id = ?", (event_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Event not found")
    return row_to_event(row)


def event_title(event_type: EventType) -> tuple[str, Severity, float, str, list[str]]:
    titles: dict[EventType, tuple[str, Severity, float, str, list[str]]] = {
        "employee_absence": ("Сотрудник отсутствует в рабочей зоне", "high", 0.82,
            "YOLOv8: ни одного человека не обнаружено в рабочей зоне дольше порога.", ["person_absent", "yolov8", "threshold_exceeded"]),
        "employee_presence": ("Сотрудник появился в рабочей зоне", "low", 0.9,
            "YOLOv8: обнаружен человек (person) с confidence > порога.", ["person_detected", "yolov8", "bbox"]),
        "visitor_shelf_dwell": ("Длительное нахождение посетителя у полки", "medium", 0.72,
            "YOLOv8: человек в зоне полки дольше порогового времени.", ["dwell_time", "shelf_zone", "yolov8"]),
        "hand_to_body": ("Жест руки к телу или сумке", "medium", 0.64,
            "Эвристический анализ траектории руки.", ["hand_trajectory", "heuristic", "risk_signal"]),
        "back_to_camera": ("Посетитель стоит спиной к камере у полки", "medium", 0.68,
            "YOLOv8: человек обнаружен, но ориентирован спиной.", ["back_facing", "yolov8", "risk_signal"]),
        "system_stream_lost": ("Нестабильный RTSP-поток", "high", 0.95,
            "Поток недоступен или нестабилен.", ["stream_lost", "system", "connectivity"]),
    }
    return titles[event_type]


def default_zone(camera: Camera) -> str:
    if len(camera.zones) == 0:
        return "Без зоны"
    return camera.zones[0].name


def count_events() -> int:
    with db_connection() as connection:
        return connection.execute("SELECT COUNT(*) AS count FROM events").fetchone()["count"]


def create_event(camera_id: str, event_type: EventType, description: str | None) -> VideoEvent:
    camera = camera_by_id(camera_id)
    title, severity, confidence, analysis_summary, evidence_tags = event_title(event_type)
    event_id = str(uuid4())
    detected_at = utc_now()
    event = VideoEvent(
        id=event_id,
        camera_id=camera.id,
        camera_name=camera.name,
        type=event_type,
        severity=severity,
        title=title,
        description=description or title,
        zone=default_zone(camera),
        detected_at=detected_at,
        snapshot_url=f"/api/events/{event_id}/snapshot.svg",
        status="new",
        confidence=confidence,
        analysis_summary=analysis_summary,
        evidence_tags=evidence_tags,
    )
    with db_connection() as connection:
        connection.execute(
            """
            INSERT INTO events (
                id, camera_id, camera_name, type, severity, title, description, zone,
                detected_at, snapshot_url, status, confidence, feedback_note, reviewed_by,
                reviewed_at, reaction_seconds, telegram_sent
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.id,
                event.camera_id,
                event.camera_name,
                event.type,
                event.severity,
                event.title,
                event.description,
                event.zone,
                event.detected_at,
                event.snapshot_url,
                event.status,
                event.confidence,
                event.feedback_note,
                event.reviewed_by,
                event.reviewed_at,
                event.reaction_seconds,
                int(event.telegram_sent),
            ),
        )
        connection.execute(
            """
            UPDATE cameras
            SET last_event_title = ?, last_event_at = ?, last_seen_at = ?
            WHERE id = ?
            """,
            (event.title, event.detected_at, event.detected_at, event.camera_id),
        )
        connection.commit()
    return event


def seed_events_if_needed() -> None:
    if count_events() > 0:
        return
    create_event(
        "cam-buffalo-trace",
        "employee_absence",
        "Сотрудник не определяется в рабочей зоне дольше 7 минут.",
    )
    create_event(
        "cam-hotel-lobby",
        "employee_presence",
        "Появление человека в зоне лобби зафиксировано AI-анализом.",
    )
    create_event(
        "cam-purdue-mall",
        "visitor_shelf_dwell",
        "Посетитель находится в зоне дольше порога, YOLOv8 детекция.",
    )
    create_event(
        "cam-piano-japan",
        "back_to_camera",
        "Человек стоит спиной к камере, AI-анализ требует проверки оператором.",
    )


def list_events_from_db(
    event_type: EventType | None = None,
    status: EventStatus | None = None,
    camera_id: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    limit: int | None = None,
) -> list[VideoEvent]:
    query = "SELECT * FROM events WHERE 1 = 1"
    params: list[str | int] = []
    if event_type is not None:
        query += " AND type = ?"
        params.append(event_type)
    if status is not None:
        query += " AND status = ?"
        params.append(status)
    if camera_id is not None:
        query += " AND camera_id = ?"
        params.append(camera_id)
    if date_from is not None:
        query += " AND detected_at >= ?"
        params.append(date_from)
    if date_to is not None:
        query += " AND detected_at <= ?"
        params.append(date_to)
    query += " ORDER BY detected_at DESC"
    if limit is not None:
        query += " LIMIT ?"
        params.append(limit)
    with db_connection() as connection:
        rows = connection.execute(query, params).fetchall()
    return [row_to_event(row) for row in rows]


def load_settings() -> MonitoringSettings:
    with db_connection() as connection:
        rows = connection.execute("SELECT key, value FROM settings").fetchall()
    values = {row["key"]: row["value"] for row in rows}
    return MonitoringSettings(
        absence_threshold_minutes=int(values.get("absence_threshold_minutes", DEFAULT_SETTINGS.absence_threshold_minutes)),
        shelf_dwell_seconds=int(values.get("shelf_dwell_seconds", DEFAULT_SETTINGS.shelf_dwell_seconds)),
        confidence_threshold=float(values.get("confidence_threshold", DEFAULT_SETTINGS.confidence_threshold)),
    )


def save_settings(settings: MonitoringSettings) -> MonitoringSettings:
    with db_connection() as connection:
        for key, value in settings.model_dump().items():
            connection.execute(
                "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
                (key, str(value)),
            )
        connection.commit()
    return settings


def get_shift_bounds(day: str | None = None) -> tuple[str, str, str]:
    if day is None:
        date_value = datetime.now(timezone.utc).date().isoformat()
    else:
        date_value = day
    start = datetime.fromisoformat(date_value).replace(tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    return date_value, start.isoformat().replace("+00:00", "Z"), end.isoformat().replace("+00:00", "Z")


def build_shift_analytics(day: str | None = None) -> ShiftAnalytics:
    report_date, start, end = get_shift_bounds(day)
    events = list_events_from_db(date_from=start, date_to=end)
    with db_connection() as connection:
        camera_rows = connection.execute("SELECT * FROM cameras ORDER BY name").fetchall()
    cameras = [row_to_camera(row) for row in camera_rows]
    reacted = [event.reaction_seconds for event in events if event.reaction_seconds is not None]
    average_reaction = int(sum(reacted) / len(reacted)) if len(reacted) > 0 else None
    return ShiftAnalytics(
        report_date=report_date,
        shift_started_at=start,
        total_events=len(events),
        open_events=len([event for event in events if event.status == "new"]),
        confirmed_events=len([event for event in events if event.status == "confirmed"]),
        dismissed_events=len([event for event in events if event.status == "dismissed"]),
        absence_events=len([event for event in events if event.type == "employee_absence"]),
        suspicious_events=len([event for event in events if event.type in SUSPICIOUS_EVENT_TYPES]),
        average_reaction_seconds=average_reaction,
        cameras_online=len([camera for camera in cameras if camera.status == "online"]),
        cameras_total=len(cameras),
        telegram_configured=os.getenv("TELEGRAM_BOT_TOKEN") is not None and os.getenv("TELEGRAM_CHAT_ID") is not None,
    )


def build_telegram_preview(event: VideoEvent) -> TelegramPreview:
    text = (
        f"AI event: {event.title}\n"
        f"Камера: {event.camera_name}\n"
        f"Зона: {event.zone}\n"
        f"Confidence: {round(event.confidence * 100)}%\n"
        f"Время: {event.detected_at}"
    )
    buttons = [
        TelegramButton(label="Подтвердить", action="confirmed", callback_data=f"feedback:{event.id}:confirmed"),
        TelegramButton(label="Отклонить", action="dismissed", callback_data=f"feedback:{event.id}:dismissed"),
    ]
    mode: Literal["telegram", "mock"] = "telegram" if os.getenv("TELEGRAM_BOT_TOKEN") and os.getenv("TELEGRAM_CHAT_ID") else "mock"
    return TelegramPreview(mode=mode, text=text, buttons=buttons)


def send_telegram_message(text: str, event: VideoEvent | None = None) -> TelegramTestResponse:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    latest_events = list_events_from_db(limit=1)
    preview_event = event or latest_events[0]
    preview = build_telegram_preview(preview_event)
    if token is None or chat_id is None:
        return TelegramTestResponse(
            configured=False,
            sent=False,
            mode="mock",
            detail="TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID are not configured; inline message is shown as mock preview.",
            inline_feedback=True,
            preview=preview,
        )

    payload: dict[str, object] = {"chat_id": chat_id, "text": text}
    if event is not None:
        payload["reply_markup"] = {
                "inline_keyboard": [
                    [
                        {"text": "Подтвердить", "callback_data": f"feedback:{event.id}:confirmed"},
                        {"text": "Отклонить", "callback_data": f"feedback:{event.id}:dismissed"},
                    ]
                ]
            }
    telegram_request = request.Request(
        url=f"https://api.telegram.org/bot{token}/sendMessage",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(telegram_request, timeout=8) as response:
            sent = 200 <= response.status < 300
    except OSError as exc:
        return TelegramTestResponse(
            configured=True,
            sent=False,
            mode="telegram",
            detail=str(exc),
            inline_feedback=event is not None,
            preview=preview,
        )

    return TelegramTestResponse(
        configured=True,
        sent=sent,
        mode="telegram",
        detail="Telegram request completed.",
        inline_feedback=event is not None,
        preview=preview,
    )


def update_event_feedback(event_id: str, payload: FeedbackRequest) -> VideoEvent:
    event = event_by_id(event_id)
    reviewed_at = utc_now()
    reaction_seconds = max(0, int((parse_utc(reviewed_at) - parse_utc(event.detected_at)).total_seconds()))
    with db_connection() as connection:
        connection.execute(
            """
            UPDATE events
            SET status = ?, reviewed_by = ?, feedback_note = ?, reviewed_at = ?, reaction_seconds = ?
            WHERE id = ?
            """,
            (payload.status, payload.reviewed_by, payload.note, reviewed_at, reaction_seconds, event_id),
        )
        connection.commit()
    return event_by_id(event_id)


def build_csv_report(day: str | None = None) -> str:
    report_date, start, end = get_shift_bounds(day)
    events = list_events_from_db(date_from=start, date_to=end)
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Дата отчёта", report_date])
    writer.writerow([])
    writer.writerow([
        "Время",
        "Камера",
        "Зона",
        "Тип",
        "Статус",
        "Confidence",
        "Реакция, сек",
        "Описание",
    ])
    for event in events:
        writer.writerow([
            event.detected_at,
            event.camera_name,
            event.zone,
            event.type,
            event.status,
            round(event.confidence * 100),
            event.reaction_seconds or "",
            event.description,
        ])
    return output.getvalue()


def build_pdf_report(day: str | None = None) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfgen import canvas

    analytics = build_shift_analytics(day)
    _, start, end = get_shift_bounds(day)
    events = list_events_from_db(date_from=start, date_to=end)
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    if Path(font_path).exists():
        pdfmetrics.registerFont(TTFont("DejaVuSans", font_path))
        font_name = "DejaVuSans"
    else:
        font_name = "Helvetica"
    width, height = A4
    pdf.setFont(font_name, 16)
    pdf.drawString(40, height - 50, "AI Video Monitoring — отчёт смены")
    pdf.setFont(font_name, 10)
    y = height - 85
    summary_rows = [
        f"Дата: {analytics.report_date}",
        f"Всего событий: {analytics.total_events}",
        f"Подтверждено: {analytics.confirmed_events}",
        f"Отклонено: {analytics.dismissed_events}",
        f"Отсутствия сотрудников: {analytics.absence_events}",
        f"Подозрительные события: {analytics.suspicious_events}",
        f"Средняя реакция оператора: {analytics.average_reaction_seconds or 0} сек",
    ]
    for row in summary_rows:
        pdf.drawString(40, y, row)
        y -= 18
    y -= 10
    pdf.setFont(font_name, 11)
    pdf.drawString(40, y, "Последние события:")
    y -= 22
    pdf.setFont(font_name, 8)
    for event in events[:12]:
        text = f"{event.detected_at} | {event.camera_name} | {event.title} | {event.status}"
        pdf.drawString(40, y, text[:120])
        y -= 15
        if y < 60:
            pdf.showPage()
            pdf.setFont(font_name, 8)
            y = height - 50
    pdf.save()
    return buffer.getvalue()


def snapshot_svg(event: VideoEvent) -> str:
    color = {
        "high": "#ef4444",
        "medium": "#f59e0b",
        "low": "#10b981",
    }[event.severity]
    safe_title = html.escape(event.title)
    safe_camera = html.escape(event.camera_name)
    safe_zone = html.escape(event.zone)
    safe_time = html.escape(event.detected_at)
    safe_summary = html.escape(event.analysis_summary)
    safe_tags = " • ".join(html.escape(tag) for tag in event.evidence_tags[:4])
    bbox_x = 118 if event.type in {"employee_absence", "employee_presence"} else 368
    bbox_y = 92 if event.type != "back_to_camera" else 112
    return f"""
    <svg xmlns="http://www.w3.org/2000/svg" width="720" height="405" viewBox="0 0 720 405">
      <defs>
        <linearGradient id="bg" x1="0" x2="1" y1="0" y2="1">
          <stop offset="0%" stop-color="#0f172a"/>
          <stop offset="100%" stop-color="#1e293b"/>
        </linearGradient>
      </defs>
      <rect width="720" height="405" fill="url(#bg)"/>
      <rect x="26" y="28" width="668" height="349" rx="28" fill="#020617" opacity="0.64"/>
      <rect x="54" y="58" width="612" height="224" rx="18" fill="#111827" stroke="#334155"/>
      <rect x="84" y="88" width="210" height="150" rx="12" fill="#1f2937" stroke="#475569"/>
      <text x="104" y="116" fill="#94a3b8" font-size="13" font-family="Arial" font-weight="700">WORK / SHELF ZONE</text>
      <rect x="344" y="86" width="264" height="152" rx="12" fill="#172033" stroke="#475569"/>
      <text x="364" y="116" fill="#94a3b8" font-size="13" font-family="Arial" font-weight="700">PUBLIC/DEMO CAMERA FRAME</text>
      <circle cx="{bbox_x + 34}" cy="{bbox_y + 28}" r="20" fill="#cbd5e1"/>
      <rect x="{bbox_x + 12}" y="{bbox_y + 50}" width="44" height="78" rx="18" fill="#94a3b8"/>
      <rect x="{bbox_x}" y="{bbox_y}" width="92" height="148" rx="12" fill="none" stroke="{color}" stroke-width="4"/>
      <rect x="{bbox_x}" y="{bbox_y - 28}" width="174" height="24" rx="12" fill="{color}"/>
      <text x="{bbox_x + 12}" y="{bbox_y - 11}" fill="white" font-size="13" font-family="Arial" font-weight="700">AI bbox + эвристика</text>
      <path d="M430 176 C462 142, 496 142, 528 176" stroke="{color}" stroke-width="5" fill="none" stroke-linecap="round"/>
      <circle cx="430" cy="176" r="8" fill="{color}"/><circle cx="528" cy="176" r="8" fill="{color}"/>
      <rect x="54" y="298" width="612" height="50" rx="14" fill="#0f172a" stroke="#334155"/>
      <text x="74" y="318" fill="#e2e8f0" font-size="20" font-family="Arial" font-weight="700">{safe_title}</text>
      <text x="74" y="340" fill="#94a3b8" font-size="14" font-family="Arial">{safe_camera} • {safe_zone} • {safe_time}</text>
      <text x="74" y="368" fill="#cbd5e1" font-size="13" font-family="Arial">{safe_summary}</text>
      <text x="420" y="368" fill="#67e8f9" font-size="13" font-family="Arial">{safe_tags}</text>
    </svg>
    """


def live_camera_svg(camera: Camera) -> str:
    second = datetime.now(timezone.utc).second
    offset = (second % 18) * 5
    source_badge = "PUBLIC SOURCE" if camera.source_type.startswith("public") else "DEMO / MOCK RTSP"
    safe_name = html.escape(camera.name)
    safe_location = html.escape(camera.location)
    safe_source = html.escape(source_badge)
    safe_rtsp = html.escape(camera.rtsp_url)
    quality = f"{camera.quality_score}%"
    person_a_x = 116 + offset
    person_b_x = 438 - offset // 2
    return f"""
    <svg xmlns="http://www.w3.org/2000/svg" width="720" height="405" viewBox="0 0 720 405">
      <rect width="720" height="405" fill="#020617"/>
      <rect x="28" y="30" width="664" height="320" rx="22" fill="#111827" stroke="#334155"/>
      <rect x="58" y="74" width="252" height="214" rx="18" fill="#1f2937" stroke="#475569"/>
      <rect x="356" y="74" width="282" height="214" rx="18" fill="#172033" stroke="#475569"/>
      <text x="78" y="104" fill="#94a3b8" font-size="13" font-family="Arial" font-weight="700">ZONE A: рабочая зона</text>
      <text x="376" y="104" fill="#94a3b8" font-size="13" font-family="Arial" font-weight="700">ZONE B: полка / касса</text>
      <circle cx="{person_a_x + 24}" cy="158" r="18" fill="#e2e8f0"/>
      <rect x="{person_a_x + 4}" y="178" width="40" height="72" rx="16" fill="#94a3b8"/>
      <rect x="{person_a_x - 8}" y="130" width="76" height="130" rx="12" fill="none" stroke="#22c55e" stroke-width="4"/>
      <text x="{person_a_x - 4}" y="124" fill="#22c55e" font-size="12" font-family="Arial" font-weight="700">person 0.88</text>
      <circle cx="{person_b_x + 24}" cy="156" r="18" fill="#e2e8f0"/>
      <rect x="{person_b_x + 4}" y="176" width="40" height="74" rx="16" fill="#94a3b8"/>
      <path d="M{person_b_x + 42} 194 L{person_b_x + 86} 176" stroke="#f97316" stroke-width="7" stroke-linecap="round"/>
      <rect x="{person_b_x - 8}" y="128" width="112" height="132" rx="12" fill="none" stroke="#f97316" stroke-width="4"/>
      <text x="{person_b_x - 4}" y="122" fill="#f97316" font-size="12" font-family="Arial" font-weight="700">hand/shelf 0.71</text>
      <rect x="58" y="306" width="580" height="28" rx="14" fill="#0f172a"/>
      <text x="76" y="326" fill="#e2e8f0" font-size="14" font-family="Arial">LIVE PREVIEW • {safe_name} • FPS {camera.fps} • quality {quality} • {safe_source}</text>
      <text x="58" y="370" fill="#e2e8f0" font-size="22" font-family="Arial" font-weight="700">{safe_name}</text>
      <text x="58" y="392" fill="#94a3b8" font-size="14" font-family="Arial">{safe_location} • {safe_rtsp}</text>
    </svg>
    """


FRAMES_DIR = Path(__file__).resolve().parent / "frames"
FRAMES_DIR.mkdir(exist_ok=True)

init_db()


def _start_background_workers() -> None:
    from background_worker import get_analysis_cache, start_workers
    from stream_capture import LIVE_STREAMS
    start_workers(LIVE_STREAMS)


_start_background_workers()


@app.get("/healthz")
async def healthz():
    return {"status": "ok", "db_path": str(DB_PATH)}


@app.get("/api/cameras")
async def get_cameras() -> list[Camera]:
    with db_connection() as connection:
        rows = connection.execute("SELECT * FROM cameras ORDER BY name").fetchall()
    return [row_to_camera(row) for row in rows]


@app.get("/api/cameras/{camera_id}")
async def get_camera(camera_id: str) -> Camera:
    return camera_by_id(camera_id)


@app.get("/api/cameras/{camera_id}/live.svg")
async def get_camera_live(camera_id: str) -> Response:
    camera = camera_by_id(camera_id)
    return Response(content=live_camera_svg(camera), media_type="image/svg+xml")


@app.get("/api/cameras/{camera_id}/frame.jpg")
async def get_camera_frame(camera_id: str) -> Response:
    """Return the latest raw JPEG frame captured from a live stream."""
    from stream_capture import FrameCache
    frame_cache = FrameCache.get_instance()
    frame = frame_cache.get(camera_id)
    if frame is None:
        raise HTTPException(status_code=404, detail="No frame available yet for this camera")
    return Response(content=frame.jpeg_bytes, media_type="image/jpeg")


@app.get("/api/cameras/{camera_id}/frame_analyzed.jpg")
async def get_camera_frame_analyzed(camera_id: str) -> Response:
    """Return the latest JPEG frame with YOLO detection overlays."""
    from background_worker import get_analysis_cache
    analysis_cache = get_analysis_cache()
    jpeg = analysis_cache.get_annotated_jpeg(camera_id)
    if jpeg is None:
        from stream_capture import FrameCache
        frame = FrameCache.get_instance().get(camera_id)
        if frame is not None:
            return Response(content=frame.jpeg_bytes, media_type="image/jpeg")
        raise HTTPException(status_code=404, detail="No analyzed frame available yet")
    return Response(content=jpeg, media_type="image/jpeg")


@app.get("/api/cameras/{camera_id}/detections")
async def get_camera_detections(camera_id: str):
    """Return current YOLO detection results as JSON."""
    from background_worker import get_analysis_cache
    analysis_cache = get_analysis_cache()
    result = analysis_cache.get_result(camera_id)
    if result is None:
        return {"camera_id": camera_id, "detections": [], "person_count": 0, "status": "waiting"}
    return {
        "camera_id": camera_id,
        "person_count": result.person_count,
        "inference_ms": round(result.inference_ms, 1),
        "analyzed_at": result.analyzed_at,
        "frame_size": [result.frame_width, result.frame_height],
        "detections": [
            {
                "class": d.class_name,
                "confidence": round(d.confidence, 3),
                "bbox": [d.x1, d.y1, d.x2, d.y2],
            }
            for d in result.detections
        ],
    }


@app.get("/api/cameras/stream_status")
async def get_stream_status():
    """Return online/offline status for all cameras based on live frame cache."""
    from stream_capture import FrameCache
    frame_cache = FrameCache.get_instance()
    with db_connection() as connection:
        rows = connection.execute("SELECT id FROM cameras ORDER BY name").fetchall()
    statuses = {}
    for row in rows:
        cid = row["id"]
        statuses[cid] = {
            "online": frame_cache.is_online(cid),
            "has_frame": frame_cache.get(cid) is not None,
        }
    return statuses


@app.get("/api/public-sources")
async def get_public_sources() -> list[PublicVideoSource]:
    return PUBLIC_VIDEO_SOURCES


@app.get("/api/detection-capabilities")
async def get_detection_capabilities() -> list[DetectionCapability]:
    return DETECTION_CAPABILITIES


@app.get("/api/events")
async def get_events(
    event_type: EventType | None = Query(default=None),
    status: EventStatus | None = Query(default=None),
    camera_id: str | None = Query(default=None),
    date_from: str | None = Query(default=None),
    date_to: str | None = Query(default=None),
) -> list[VideoEvent]:
    return list_events_from_db(
        event_type=event_type,
        status=status,
        camera_id=camera_id,
        date_from=date_from,
        date_to=date_to,
    )


@app.get("/api/overview")
async def get_overview():
    analytics = build_shift_analytics()
    return {
        "active_cameras": analytics.cameras_online,
        "total_cameras": analytics.cameras_total,
        "open_events": analytics.open_events,
        "confirmed_events": analytics.confirmed_events,
        "dismissed_events": analytics.dismissed_events,
        "absence_events": analytics.absence_events,
        "suspicious_events": analytics.suspicious_events,
        "total_events": analytics.total_events,
        "average_reaction_seconds": analytics.average_reaction_seconds,
        "shift_started_at": analytics.shift_started_at,
        "report_date": analytics.report_date,
        "telegram_configured": analytics.telegram_configured,
    }


@app.get("/api/shift/analytics")
async def get_shift_analytics(day: str | None = Query(default=None)) -> ShiftAnalytics:
    return build_shift_analytics(day)


@app.get("/api/settings")
async def get_settings() -> MonitoringSettings:
    return load_settings()


@app.put("/api/settings")
async def update_settings(settings: MonitoringSettings) -> MonitoringSettings:
    return save_settings(settings)


@app.post("/api/events/simulate")
async def simulate_event(payload: SimulateEventRequest) -> VideoEvent:
    event_types: list[EventType] = [
        "employee_absence",
        "visitor_shelf_dwell",
        "hand_to_body",
        "back_to_camera",
        "system_stream_lost",
    ]
    event_type = payload.event_type or event_types[count_events() % len(event_types)]
    event = create_event(payload.camera_id, event_type, payload.description)
    telegram_result = send_telegram_message(
        f"{event.title}\nКамера: {event.camera_name}\nЗона: {event.zone}",
        event,
    )
    if telegram_result.sent:
        with db_connection() as connection:
            connection.execute("UPDATE events SET telegram_sent = 1 WHERE id = ?", (event.id,))
            connection.commit()
        return event_by_id(event.id)
    return event


@app.post("/api/events/{event_id}/feedback")
async def update_feedback(event_id: str, payload: FeedbackRequest) -> VideoEvent:
    return update_event_feedback(event_id, payload)


@app.get("/api/events/{event_id}/snapshot.svg")
async def get_event_snapshot(event_id: str) -> Response:
    event = event_by_id(event_id)
    return Response(content=snapshot_svg(event), media_type="image/svg+xml")


@app.get("/api/telegram/preview")
async def telegram_preview(event_id: str | None = Query(default=None)) -> TelegramPreview:
    event = event_by_id(event_id) if event_id is not None else list_events_from_db(limit=1)[0]
    return build_telegram_preview(event)


@app.post("/api/telegram/test")
async def test_telegram() -> TelegramTestResponse:
    event = list_events_from_db(limit=1)[0]
    return send_telegram_message("AI Video Monitoring MVP: test notification with inline feedback", event)


@app.get("/api/reports/day.csv")
async def report_csv(day: str | None = Query(default=None)) -> Response:
    csv_body = build_csv_report(day)
    headers = {"Content-Disposition": "attachment; filename=shift-report.csv"}
    return Response(content=csv_body, media_type="text/csv; charset=utf-8", headers=headers)


@app.get("/api/reports/day.pdf")
async def report_pdf(day: str | None = Query(default=None)) -> Response:
    pdf_body = build_pdf_report(day)
    headers = {"Content-Disposition": "attachment; filename=shift-report.pdf"}
    return Response(content=pdf_body, media_type="application/pdf", headers=headers)


@app.get("/")
async def dashboard() -> FileResponse:
    index_file = FRONTEND_DIST / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Frontend build not found")
    return FileResponse(index_file)


@app.get("/{full_path:path}")
async def dashboard_fallback(full_path: str) -> FileResponse:
    index_file = FRONTEND_DIST / "index.html"
    if full_path.startswith("api/") or not index_file.exists():
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(index_file)
