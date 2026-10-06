from __future__ import annotations

import html
import json
import logging
import os
import secrets
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

import db
import reports
import telegram
from auth import AuthLoginRequest, AuthLoginResponse, User, get_current_user, login_user, role_required
from camera_runtime import runtime
from defaults import DETECTION_CAPABILITIES, PUBLIC_VIDEO_SOURCES
from models import (
    Camera,
    CameraSource,
    CameraSourceCreate,
    EventStatus,
    EventType,
    FeedbackRequest,
    MonitoringSettings,
    ShiftAnalytics,
    SimulateEventRequest,
    TelegramPreview,
    TelegramTestResponse,
    Track,
    VideoEvent,
)
from stream_capture import FrameCache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Video Monitoring MVP")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIST = Path(os.getenv("AI_MONITOR_FRONTEND_DIST", str(Path(__file__).resolve().parents[2] / "frontend" / "dist")))
if not FRONTEND_DIST.exists():
    FRONTEND_DIST = Path(__file__).resolve().parents[1] / "frontend" / "dist"
if (FRONTEND_DIST / "assets").exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

PUBLIC_API_PATHS = {
    "/api/auth/login",
    "/api/auth/me",
    "/api/telegram/callback",
}


@app.middleware("http")
async def api_auth_middleware(request: Request, call_next):
    path = request.url.path
    if path.startswith("/api/") and path not in PUBLIC_API_PATHS:
        try:
            get_current_user(request)
        except HTTPException as exc:
            return Response(
                content=json.dumps({"detail": exc.detail}),
                status_code=exc.status_code,
                media_type="application/json",
            )
    return await call_next(request)


@app.on_event("startup")
async def startup() -> None:
    db.init_db()
    if os.getenv("AI_MONITOR_DISABLE_RUNTIME") != "1":
        runtime.start()
    from notifications import notifier
    notifier.start()


@app.on_event("shutdown")
async def shutdown() -> None:
    runtime.stop()
    from notifications import notifier
    notifier.stop()


def _require_camera(camera_id: str) -> Camera:
    camera = db.get_camera(camera_id)
    if camera is None:
        raise HTTPException(status_code=404, detail="Camera not found")
    return camera


def _require_event(event_id: str) -> VideoEvent:
    event = db.get_event(event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="Event not found")
    return event


def _event_snapshot_svg(event: VideoEvent) -> str:
    color = {"high": "#ef4444", "medium": "#f59e0b", "low": "#10b981"}[event.severity]
    title = html.escape(event.title)
    camera = html.escape(event.camera_name)
    zone = html.escape(event.zone)
    summary = html.escape(event.analysis_summary)
    return f"""
    <svg xmlns="http://www.w3.org/2000/svg" width="960" height="540" viewBox="0 0 960 540">
      <rect width="960" height="540" fill="#07111f"/>
      <rect x="42" y="42" width="876" height="456" rx="18" fill="#101827" stroke="#243348"/>
      <rect x="94" y="96" width="772" height="270" rx="14" fill="#172033" stroke="#334155"/>
      <rect x="390" y="142" width="145" height="185" rx="10" fill="none" stroke="{color}" stroke-width="5"/>
      <circle cx="462" cy="175" r="30" fill="#cbd5e1"/>
      <rect x="430" y="205" width="64" height="102" rx="24" fill="#94a3b8"/>
      <rect x="94" y="398" width="772" height="70" rx="12" fill="#0f172a" stroke="#334155"/>
      <text x="120" y="427" fill="#f8fafc" font-family="Arial" font-size="24" font-weight="700">{title}</text>
      <text x="120" y="454" fill="#94a3b8" font-family="Arial" font-size="16">{camera} | {zone} | {event.detected_at}</text>
      <text x="120" y="485" fill="#67e8f9" font-family="Arial" font-size="14">{summary}</text>
    </svg>
    """


def _live_svg(camera: Camera) -> str:
    safe_name = html.escape(camera.name)
    safe_location = html.escape(camera.location)
    safe_status = html.escape(camera.status)
    return f"""
    <svg xmlns="http://www.w3.org/2000/svg" width="960" height="540" viewBox="0 0 960 540">
      <rect width="960" height="540" fill="#06101d"/>
      <rect x="46" y="52" width="868" height="372" rx="22" fill="#111827" stroke="#334155"/>
      <line x1="46" y1="238" x2="914" y2="238" stroke="#243348"/>
      <line x1="480" y1="52" x2="480" y2="424" stroke="#243348"/>
      <circle cx="430" cy="198" r="32" fill="#cbd5e1"/>
      <rect x="394" y="230" width="72" height="122" rx="28" fill="#94a3b8"/>
      <rect x="376" y="154" width="132" height="210" rx="16" fill="none" stroke="#22c55e" stroke-width="5"/>
      <text x="70" y="92" fill="#67e8f9" font-family="Arial" font-size="20" font-weight="700">LIVE PREVIEW</text>
      <text x="70" y="458" fill="#f8fafc" font-family="Arial" font-size="28" font-weight="700">{safe_name}</text>
      <text x="70" y="488" fill="#94a3b8" font-family="Arial" font-size="16">{safe_location} | {safe_status}</text>
    </svg>
    """


@app.get("/healthz")
async def healthz():
    return {"status": "ok", "db_path": str(db.DB_PATH), "runtime_cameras": len(runtime.status())}


@app.post("/api/auth/login", response_model=AuthLoginResponse)
async def auth_login(payload: AuthLoginRequest):
    return login_user(payload)


@app.get("/api/auth/me", response_model=User)
async def auth_me(user: User = Depends(get_current_user)):
    return user


@app.get("/api/camera-sources", response_model=list[CameraSource])
async def get_camera_sources(_user: User = Depends(role_required(["admin", "manager", "operator"]))):
    return db.list_camera_sources()


@app.post("/api/camera-sources", response_model=CameraSource)
async def upsert_camera_source(payload: CameraSourceCreate, _user: User = Depends(role_required(["admin", "manager"]))):
    source = db.upsert_camera_source(CameraSource.model_validate(payload.model_dump()))
    runtime.reload_sources()
    return source


@app.get("/api/ai/status")
async def ai_readiness():
    from readiness import model_status
    return model_status()


@app.get("/api/camera-sources/{camera_id}/readiness")
async def camera_readiness(camera_id: str):
    from readiness import camera_status
    source = next((s for s in db.list_camera_sources() if s.id == camera_id), None)
    if source is None:
        raise HTTPException(status_code=404, detail="Camera source not found")
    return camera_status(source, runtime)


@app.get("/api/cameras/stream_status")
async def get_stream_status():
    statuses = runtime.status()
    return {
        camera.id: statuses.get(camera.id, {"online": False, "has_frame": False})
        for camera in db.list_cameras()
    }


@app.get("/api/tracks", response_model=list[Track])
async def get_tracks(camera_id: str | None = Query(default=None), active: bool | None = Query(default=True)):
    return db.list_tracks(camera_id=camera_id, active=active)


@app.get("/api/cameras", response_model=list[Camera])
async def get_cameras():
    return db.list_cameras()


@app.get("/api/cameras/{camera_id}", response_model=Camera)
async def get_camera(camera_id: str):
    return _require_camera(camera_id)


@app.get("/api/cameras/{camera_id}/live.svg")
async def get_camera_live(camera_id: str) -> Response:
    return Response(content=_live_svg(_require_camera(camera_id)), media_type="image/svg+xml")


@app.get("/api/cameras/{camera_id}/frame.jpg")
async def get_camera_frame(camera_id: str) -> Response:
    frame = FrameCache.get_instance().get(camera_id)
    if frame is None:
        raise HTTPException(status_code=404, detail="No frame available yet for this camera")
    return Response(content=frame.jpeg_bytes, media_type="image/jpeg")


@app.get("/api/cameras/{camera_id}/frame_analyzed.jpg")
async def get_camera_frame_analyzed(camera_id: str) -> Response:
    jpeg = runtime.get_annotated_jpeg(camera_id)
    if jpeg is not None:
        return Response(content=jpeg, media_type="image/jpeg")
    frame = FrameCache.get_instance().get(camera_id)
    if frame is not None:
        return Response(content=frame.jpeg_bytes, media_type="image/jpeg")
    raise HTTPException(status_code=404, detail="No analyzed frame available yet")


@app.get("/api/cameras/{camera_id}/detections")
async def get_camera_detections(camera_id: str):
    _require_camera(camera_id)
    return runtime.detections_payload(camera_id)


@app.get("/api/public-sources")
async def get_public_sources():
    return PUBLIC_VIDEO_SOURCES


@app.get("/api/detection-capabilities")
async def get_detection_capabilities():
    return DETECTION_CAPABILITIES


@app.get("/api/events", response_model=list[VideoEvent])
async def get_events(
    event_type: EventType | None = Query(default=None),
    status: EventStatus | None = Query(default=None),
    camera_id: str | None = Query(default=None),
    date_from: str | None = Query(default=None),
    date_to: str | None = Query(default=None),
):
    return db.list_events(event_type=event_type, status=status, camera_id=camera_id, date_from=date_from, date_to=date_to)


@app.get("/api/events/{event_id}", response_model=VideoEvent)
async def get_event(event_id: str):
    return _require_event(event_id)


@app.post("/api/events/simulate", response_model=VideoEvent)
async def simulate_event(payload: SimulateEventRequest, _user: User = Depends(role_required(["admin", "manager"]))):
    _require_camera(payload.camera_id)
    event_types: list[EventType] = ["normal", "suspicious", "fight", "fall", "danger"]
    event_type = payload.event_type or event_types[len(db.list_events(limit=1000)) % len(event_types)]
    event = db.create_event(payload.camera_id, event_type, payload.description, rule_id="manual_simulate")
    return event


@app.post("/api/events/{event_id}/feedback", response_model=VideoEvent)
async def update_feedback(event_id: str, payload: FeedbackRequest, user: User = Depends(get_current_user)):
    try:
        return db.update_feedback(event_id, payload.model_copy(update={"reviewed_by": user.username}))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/events/{event_id}/snapshot.jpg")
async def event_snapshot_jpeg(event_id: str):
    _require_event(event_id)
    with db.connect() as connection:
        media = connection.execute("SELECT jpeg FROM event_media WHERE event_id=?", (event_id,)).fetchone()
    if not media or not media["jpeg"]:
        raise HTTPException(status_code=404, detail="Event frame unavailable")
    return Response(content=media["jpeg"], media_type="image/jpeg", headers={"Cache-Control": "private, max-age=3600"})


@app.get("/api/events/{event_id}/snapshot.svg")
async def get_event_snapshot(event_id: str) -> Response:
    return Response(content=_event_snapshot_svg(_require_event(event_id)), media_type="image/svg+xml")


@app.get("/api/overview")
async def get_overview():
    analytics = db.build_shift_analytics()
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


@app.get("/api/shift/analytics", response_model=ShiftAnalytics)
async def get_shift_analytics(day: str | None = Query(default=None)):
    return db.build_shift_analytics(day)


@app.get("/api/settings", response_model=MonitoringSettings)
async def get_settings():
    return db.load_settings()


@app.put("/api/settings", response_model=MonitoringSettings)
async def update_settings(settings: MonitoringSettings, _user: User = Depends(role_required(["admin", "manager"]))):
    return db.save_settings(settings)


@app.get("/api/telegram/preview", response_model=TelegramPreview)
async def telegram_preview(event_id: str | None = Query(default=None)):
    event = _require_event(event_id) if event_id else (db.list_events(limit=1)[0] if db.list_events(limit=1) else None)
    if event is None:
        raise HTTPException(status_code=404, detail="No events available")
    return telegram.build_preview(event)


@app.post("/api/telegram/test", response_model=TelegramTestResponse)
async def test_telegram():
    events = db.list_events(limit=1)
    if not events:
        event = db.create_event("cam-hypermarket-frozen", "employee_presence", "Telegram test event", rule_id="telegram_test")
    else:
        event = events[0]
    return telegram.send_message("AI Video Monitoring MVP: test notification with inline feedback", event)


@app.post("/api/telegram/callback")
async def telegram_callback(request: Request):
    secret = os.getenv("TELEGRAM_WEBHOOK_SECRET", "")
    if not secret or not secrets.compare_digest(request.headers.get("X-Telegram-Bot-Api-Secret-Token", ""), secret):
        raise HTTPException(status_code=403, detail="Invalid webhook secret")
    payload = await request.json()
    data = payload.get("callback_query", {}).get("data", "")
    parts = data.split(":")
    if len(parts) != 3 or parts[0] != "feedback":
        return {"ok": False, "detail": "Unsupported callback"}
    _prefix, event_id, status = parts
    if status not in {"confirmed", "dismissed"}:
        return {"ok": False, "detail": "Unsupported status"}
    user_name = payload.get("callback_query", {}).get("from", {}).get("username", "telegram")
    event = db.update_feedback(event_id, FeedbackRequest(status=status, reviewed_by=user_name, note="Telegram feedback"))
    return {"ok": True, "event": event}


@app.get("/api/reports/day.csv")
async def report_csv(day: str | None = Query(default=None)) -> Response:
    return Response(
        content=reports.build_csv_report(day),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": "attachment; filename=shift-report.csv"},
    )


@app.get("/api/reports/day.pdf")
async def report_pdf(day: str | None = Query(default=None)) -> Response:
    return Response(
        content=reports.build_pdf_report(day),
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=shift-report.pdf"},
    )


@app.get("/api/reports/period.csv")
async def report_csv_period(start: str = Query(...), end: str = Query(...)) -> Response:
    return Response(
        content=reports.build_csv_report_range(start, end),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": "attachment; filename=shift-report-period.csv"},
    )


@app.get("/api/reports/period.pdf")
async def report_pdf_period(start: str = Query(...), end: str = Query(...)) -> Response:
    return Response(
        content=reports.build_pdf_report_range(start, end),
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=shift-report-period.pdf"},
    )


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
