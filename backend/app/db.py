from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from defaults import DEFAULT_SETTINGS, default_sources
from models import (
    Camera,
    CameraSource,
    EventStatus,
    EventType,
    FeedbackRequest,
    MonitoringSettings,
    SUSPICIOUS_EVENT_TYPES,
    Severity,
    ShiftAnalytics,
    Track,
    VideoEvent,
    Zone,
    parse_utc,
    utc_now,
)

DB_PATH = Path(os.getenv("AI_MONITORING_DB_PATH", Path(__file__).resolve().parents[1] / "monitoring.db"))


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    return connection


def _execute_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS camera_sources (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            location TEXT NOT NULL,
            url TEXT NOT NULL,
            source_type TEXT NOT NULL,
            enabled INTEGER NOT NULL DEFAULT 1,
            fps_limit INTEGER NOT NULL DEFAULT 8,
            zones TEXT NOT NULL
        );

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
        );

        CREATE TABLE IF NOT EXISTS zones (
            id TEXT PRIMARY KEY,
            camera_id TEXT NOT NULL,
            name TEXT NOT NULL,
            kind TEXT NOT NULL,
            polygon TEXT
        );

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
        );

        CREATE TABLE IF NOT EXISTS rule_events (
            id TEXT PRIMARY KEY,
            event_id TEXT NOT NULL,
            rule_id TEXT NOT NULL,
            camera_id TEXT NOT NULL,
            track_id TEXT,
            zone_id TEXT,
            created_at TEXT NOT NULL,
            evidence TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS feedback (
            id TEXT PRIMARY KEY,
            event_id TEXT NOT NULL,
            status TEXT NOT NULL,
            reviewed_by TEXT NOT NULL,
            note TEXT,
            created_at TEXT NOT NULL,
            reaction_seconds INTEGER
        );

        CREATE TABLE IF NOT EXISTS tracks (
            id TEXT PRIMARY KEY,
            camera_id TEXT NOT NULL,
            local_track_id INTEGER NOT NULL,
            first_seen_at TEXT NOT NULL,
            last_seen_at TEXT NOT NULL,
            zone_id TEXT,
            zone_name TEXT,
            confidence REAL NOT NULL,
            bbox TEXT NOT NULL,
            active INTEGER NOT NULL DEFAULT 1
        );

        CREATE TABLE IF NOT EXISTS presence_sessions (
            id TEXT PRIMARY KEY,
            camera_id TEXT NOT NULL,
            track_id TEXT NOT NULL,
            zone_id TEXT NOT NULL,
            started_at TEXT NOT NULL,
            ended_at TEXT,
            first_seen_at TEXT NOT NULL,
            last_seen_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        """
    )


def init_db() -> None:
    with connect() as connection:
        _execute_schema(connection)
        connection.execute("CREATE TABLE IF NOT EXISTS event_media (event_id TEXT PRIMARY KEY, jpeg BLOB, detection_mode TEXT NOT NULL)")
        for source in default_sources():
            if not connection.execute("SELECT 1 FROM camera_sources WHERE id=?", (source.id,)).fetchone():
                upsert_camera_source(source, connection)
        for key, value in DEFAULT_SETTINGS.model_dump().items():
            connection.execute("INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)", (key, str(value)))
        connection.commit()


def upsert_camera_source(source: CameraSource, connection: sqlite3.Connection | None = None) -> CameraSource:
    owns_connection = connection is None
    connection = connection or connect()
    try:
        zones_json = json.dumps([zone.model_dump() for zone in source.zones], ensure_ascii=False)
        connection.execute(
            """
            INSERT OR REPLACE INTO camera_sources (id, name, location, url, source_type, enabled, fps_limit, zones)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (source.id, source.name, source.location, source.url, source.source_type, int(source.enabled), source.fps_limit, zones_json),
        )
        connection.execute(
            """
            INSERT OR REPLACE INTO cameras (
                id, name, location, rtsp_url, status, ai_status, fps, zones, last_seen_at,
                source_type, quality_score, uptime_minutes, last_event_title, last_event_at
            ) VALUES (
                ?, ?, ?, ?, ?,
                ?,
                ?, ?, COALESCE((SELECT last_seen_at FROM cameras WHERE id = ?), ?),
                ?, COALESCE((SELECT quality_score FROM cameras WHERE id = ?), 70),
                COALESCE((SELECT uptime_minutes FROM cameras WHERE id = ?), 0),
                (SELECT last_event_title FROM cameras WHERE id = ?),
                (SELECT last_event_at FROM cameras WHERE id = ?)
            )
            """,
            (
                source.id,
                source.name,
                source.location,
                source.url,
                "offline",
                "disabled" if not source.enabled else "warming_up",
                source.fps_limit,
                zones_json,
                source.id,
                utc_now(),
                source.source_type,
                source.id,
                source.id,
                source.id,
                source.id,
            ),
        )
        connection.execute("DELETE FROM zones WHERE camera_id = ?", (source.id,))
        for zone in source.zones:
            connection.execute(
                "INSERT OR REPLACE INTO zones (id, camera_id, name, kind, polygon) VALUES (?, ?, ?, ?, ?)",
                (zone.id, source.id, zone.name, zone.kind, json.dumps(zone.polygon) if zone.polygon else None),
            )
        if owns_connection:
            connection.commit()
    finally:
        if owns_connection:
            connection.close()
    return source


def row_to_source(row: sqlite3.Row) -> CameraSource:
    return CameraSource(
        id=row["id"],
        name=row["name"],
        location=row["location"],
        url=row["url"],
        source_type=row["source_type"],
        enabled=bool(row["enabled"]),
        fps_limit=row["fps_limit"],
        zones=[Zone.model_validate(zone) for zone in json.loads(row["zones"])],
    )


def list_camera_sources(enabled_only: bool = False) -> list[CameraSource]:
    query = "SELECT * FROM camera_sources"
    params: tuple[object, ...] = ()
    if enabled_only:
        query += " WHERE enabled = 1"
    query += " ORDER BY name"
    with connect() as connection:
        rows = connection.execute(query, params).fetchall()
    return [row_to_source(row) for row in rows]


def row_to_camera(row: sqlite3.Row) -> Camera:
    return Camera(
        id=row["id"],
        name=row["name"],
        location=row["location"],
        rtsp_url=row["rtsp_url"],
        status=row["status"],
        ai_status=row["ai_status"],
        fps=row["fps"],
        zones=[Zone.model_validate(zone) for zone in json.loads(row["zones"])],
        last_seen_at=row["last_seen_at"],
        source_type=row["source_type"],
        quality_score=row["quality_score"],
        uptime_minutes=row["uptime_minutes"],
        last_event_title=row["last_event_title"],
        last_event_at=row["last_event_at"],
    )


def list_cameras() -> list[Camera]:
    with connect() as connection:
        rows = connection.execute("SELECT * FROM cameras ORDER BY name").fetchall()
    return [row_to_camera(row) for row in rows]


def get_camera(camera_id: str) -> Camera | None:
    with connect() as connection:
        row = connection.execute("SELECT * FROM cameras WHERE id = ?", (camera_id,)).fetchone()
    return row_to_camera(row) if row else None


def update_camera_runtime(camera_id: str, *, online: bool, ai_status: str, fps: int, quality_score: int, error: str | None = None) -> None:
    status = "online" if online else "offline"
    with connect() as connection:
        connection.execute(
            """
            UPDATE cameras
            SET status = ?, ai_status = ?, fps = ?, quality_score = ?, last_seen_at = ?
            WHERE id = ?
            """,
            (status, ai_status, fps, quality_score, utc_now(), camera_id),
        )
        connection.commit()


def event_title(event_type: EventType) -> tuple[str, Severity, float, str, list[str]]:
    titles: dict[EventType, tuple[str, Severity, float, str, list[str]]] = {
        "employee_absence": ("Employee absent from work zone", "high", 0.82, "No employee-like person was present in a work zone longer than the configured threshold.", ["zone", "last_seen", "threshold"]),
        "employee_presence": ("Employee appeared in work zone", "low", 0.9, "A person was detected in a configured employee/work zone.", ["track_id", "zone", "first_seen"]),
        "visitor_shelf_dwell": ("Long dwell near shelf", "medium", 0.72, "A person stayed in a shelf zone longer than the dwell threshold.", ["track_id", "shelf_zone", "dwell_time"]),
        "hand_to_body": ("Experimental hand-to-body risk signal", "medium", 0.55, "Experimental heuristic; requires pose model and operator feedback before production use.", ["experimental", "human_feedback"]),
        "back_to_camera": ("Person facing away near shelf", "medium", 0.62, "A person stayed near a shelf with a back-facing/body-orientation heuristic.", ["track_id", "orientation_heuristic"]),
        "system_stream_lost": ("Camera stream unstable", "high", 0.95, "The camera stream is unavailable or unstable.", ["stream_status", "last_frame"]),
    }
    titles.update({
        "normal": ("Нормальная активность", "low", 0.8, "Обнаружен человек без сработавших правил риска.", ["person"]),
        "suspicious": ("Подозрительная активность", "medium", 0.6, "Длительное нахождение у полки. Требуется проверка; это не доказательство кражи.", ["zone", "dwell"]),
        "fight": ("Возможная драка или нападение", "high", 0.6, "Резкие повторяющиеся движения сближенных людей. Эвристика требует проверки.", ["motion", "proximity"]),
        "fall": ("Возможное падение", "high", 0.6, "Переход человека из вертикального положения в устойчивое горизонтальное.", ["posture_transition", "duration"]),
        "danger": ("Человек в опасной зоне", "high", 0.7, "Человек находится в настроенной запрещённой зоне.", ["restricted_polygon"]),
    })
    return titles[event_type]


def event_analysis(event_type: EventType) -> tuple[str, list[str]]:
    title, _severity, _confidence, summary, tags = event_title(event_type)
    return summary or title, tags


def row_to_event(row: sqlite3.Row) -> VideoEvent:
    summary, tags = event_analysis(row["type"])
    mode = row["persisted_mode"] if "persisted_mode" in row.keys() else None
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
        detection_mode=mode or "legacy",
        analysis_summary=summary,
        evidence_tags=tags,
    )


def create_event(
    camera_id: str,
    event_type: EventType,
    description: str | None = None,
    *,
    zone_name: str | None = None,
    confidence: float | None = None,
    rule_id: str | None = None,
    track_id: str | None = None,
    zone_id: str | None = None,
    evidence: dict[str, object] | None = None,
) -> VideoEvent:
    camera = get_camera(camera_id)
    if camera is None:
        raise ValueError(f"Camera not found: {camera_id}")
    title, severity, default_confidence, _summary, _tags = event_title(event_type)
    if rule_id and rule_id != "manual_simulate":
        cooldown_seconds = {
            "employee_presence": 300,
            "employee_absence": 600,
            "visitor_shelf_dwell": 300,
            "back_to_camera": 600,
            "hand_to_body": 600,
            "system_stream_lost": 300,
        }.get(event_type, 300)
        cutoff = (datetime.now(timezone.utc) - timedelta(seconds=cooldown_seconds)).isoformat().replace("+00:00", "Z")
        with connect() as connection:
            recent = connection.execute(
                """
                SELECT events.*, m.detection_mode AS persisted_mode FROM events LEFT JOIN event_media m ON m.event_id=events.id
                WHERE camera_id = ? AND type = ? AND zone = ? AND detected_at >= ?
                ORDER BY detected_at DESC
                LIMIT 1
                """,
                (camera_id, event_type, zone_name or (camera.zones[0].name if camera.zones else "Unconfigured zone"), cutoff),
            ).fetchone()
        if recent is not None:
            return row_to_event(recent)
    from stream_capture import FrameCache
    import time
    frame = FrameCache.get_instance().get(camera_id)
    jpeg = frame.jpeg_bytes if frame and time.time() - frame.captured_at < 10 else None
    mode = "demo" if rule_id == "manual_simulate" or camera.source_type == "retail_scene" else "heuristic"
    event_id = str(uuid4())
    detected_at = utc_now()
    zone = zone_name or (camera.zones[0].name if camera.zones else "Unconfigured zone")
    with connect() as connection:
        connection.execute(
            """
            INSERT INTO events (
                id, camera_id, camera_name, type, severity, title, description, zone,
                detected_at, snapshot_url, status, confidence, feedback_note, reviewed_by,
                reviewed_at, reaction_seconds, telegram_sent
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, 0)
            """,
            (
                event_id,
                camera.id,
                camera.name,
                event_type,
                severity,
                title,
                description or title,
                zone,
                detected_at,
                f"/api/events/{event_id}/snapshot.jpg",
                "new",
                confidence if confidence is not None else default_confidence,
            ),
        )
        connection.execute("INSERT INTO event_media VALUES (?, ?, ?)", (event_id, jpeg, mode))
        if rule_id:
            connection.execute(
                "INSERT INTO rule_events (id, event_id, rule_id, camera_id, track_id, zone_id, created_at, evidence) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (str(uuid4()), event_id, rule_id, camera_id, track_id, zone_id, detected_at, json.dumps(evidence or {}, ensure_ascii=False)),
            )
        connection.execute(
            "UPDATE cameras SET last_event_title = ?, last_event_at = ? WHERE id = ?",
            (title, detected_at, camera_id),
        )
        connection.commit()
    return get_event(event_id)  # type: ignore[return-value]


def get_event(event_id: str) -> VideoEvent | None:
    with connect() as connection:
        row = connection.execute("SELECT e.*, m.detection_mode AS persisted_mode FROM events e LEFT JOIN event_media m ON m.event_id=e.id WHERE e.id = ?", (event_id,)).fetchone()
    return row_to_event(row) if row else None


def list_events(
    event_type: EventType | None = None,
    status: EventStatus | None = None,
    camera_id: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    limit: int | None = None,
) -> list[VideoEvent]:
    query = "SELECT events.*, m.detection_mode AS persisted_mode FROM events LEFT JOIN event_media m ON m.event_id=events.id WHERE 1 = 1"
    params: list[object] = []
    if event_type:
        query += " AND type = ?"
        params.append(event_type)
    if status:
        query += " AND status = ?"
        params.append(status)
    if camera_id:
        query += " AND camera_id = ?"
        params.append(camera_id)
    if date_from:
        query += " AND detected_at >= ?"
        params.append(date_from)
    if date_to:
        query += " AND detected_at <= ?"
        params.append(date_to)
    query += " ORDER BY detected_at DESC"
    if limit:
        query += " LIMIT ?"
        params.append(limit)
    with connect() as connection:
        rows = connection.execute(query, params).fetchall()
    return [row_to_event(row) for row in rows]


def update_feedback(event_id: str, payload: FeedbackRequest) -> VideoEvent:
    event = get_event(event_id)
    if event is None:
        raise ValueError("Event not found")
    reviewed_at = utc_now()
    reaction_seconds = max(0, int((parse_utc(reviewed_at) - parse_utc(event.detected_at)).total_seconds()))
    with connect() as connection:
        connection.execute(
            "UPDATE events SET status = ?, reviewed_by = ?, feedback_note = ?, reviewed_at = ?, reaction_seconds = ? WHERE id = ?",
            (payload.status, payload.reviewed_by, payload.note, reviewed_at, reaction_seconds, event_id),
        )
        connection.execute(
            "INSERT INTO feedback (id, event_id, status, reviewed_by, note, created_at, reaction_seconds) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (str(uuid4()), event_id, payload.status, payload.reviewed_by, payload.note, reviewed_at, reaction_seconds),
        )
        connection.commit()
    return get_event(event_id)  # type: ignore[return-value]


def upsert_track(track: Track) -> None:
    with connect() as connection:
        connection.execute(
            """
            INSERT OR REPLACE INTO tracks (
                id, camera_id, local_track_id, first_seen_at, last_seen_at, zone_id,
                zone_name, confidence, bbox, active
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                track.id,
                track.camera_id,
                track.local_track_id,
                track.first_seen_at,
                track.last_seen_at,
                track.zone_id,
                track.zone_name,
                track.confidence,
                json.dumps(track.bbox),
                int(track.active),
            ),
        )
        connection.commit()


def list_tracks(camera_id: str | None = None, active: bool | None = None) -> list[Track]:
    query = "SELECT * FROM tracks WHERE 1 = 1"
    params: list[object] = []
    if camera_id:
        query += " AND camera_id = ?"
        params.append(camera_id)
    if active is not None:
        query += " AND active = ?"
        params.append(int(active))
    query += " ORDER BY last_seen_at DESC"
    with connect() as connection:
        rows = connection.execute(query, params).fetchall()
    return [
        Track(
            id=row["id"],
            camera_id=row["camera_id"],
            local_track_id=row["local_track_id"],
            first_seen_at=row["first_seen_at"],
            last_seen_at=row["last_seen_at"],
            zone_id=row["zone_id"],
            zone_name=row["zone_name"],
            confidence=row["confidence"],
            bbox=json.loads(row["bbox"]),
            active=bool(row["active"]),
        )
        for row in rows
    ]


def load_settings() -> MonitoringSettings:
    with connect() as connection:
        rows = connection.execute("SELECT key, value FROM settings").fetchall()
    values = {row["key"]: row["value"] for row in rows}
    return MonitoringSettings(
        absence_threshold_minutes=int(values.get("absence_threshold_minutes", DEFAULT_SETTINGS.absence_threshold_minutes)),
        shelf_dwell_seconds=int(values.get("shelf_dwell_seconds", DEFAULT_SETTINGS.shelf_dwell_seconds)),
        confidence_threshold=float(values.get("confidence_threshold", DEFAULT_SETTINGS.confidence_threshold)),
        fall_hold_seconds=float(values.get("fall_hold_seconds", DEFAULT_SETTINGS.fall_hold_seconds)),
        fight_motion_threshold=float(values.get("fight_motion_threshold", DEFAULT_SETTINGS.fight_motion_threshold)),
    )


def save_settings(settings: MonitoringSettings) -> MonitoringSettings:
    with connect() as connection:
        for key, value in settings.model_dump().items():
            connection.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (key, str(value)))
        connection.commit()
    return settings


def get_shift_bounds(day: str | None = None) -> tuple[str, str, str]:
    date_value = day or datetime.now(timezone.utc).date().isoformat()
    start = datetime.fromisoformat(date_value).replace(tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    return date_value, start.isoformat().replace("+00:00", "Z"), end.isoformat().replace("+00:00", "Z")


def build_shift_analytics(day: str | None = None) -> ShiftAnalytics:
    report_date, start, end = get_shift_bounds(day)
    events = list_events(date_from=start, date_to=end)
    cameras = list_cameras()
    reacted = [event.reaction_seconds for event in events if event.reaction_seconds is not None]
    average_reaction = int(sum(reacted) / len(reacted)) if reacted else None
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
