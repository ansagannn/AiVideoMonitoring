from __future__ import annotations

import os

from models import CameraSource, DetectionCapability, MonitoringSettings, PublicVideoSource, Zone

DEFAULT_SETTINGS = MonitoringSettings()


def default_sources() -> list[CameraSource]:
    sources = [
        CameraSource(id="cam-public-rtsp-kz", name="RTSP.KZ — публичное демо / ашық демо",
            location="Публичный HLS-поток / Ашық HLS ағыны",
            url="https://rtsp.kz/hls/demo/stream.m3u8", source_type="hls", fps_limit=2),
        CameraSource(id="cam-public-yellowstone-arch", name="Yellowstone — северный вход / солтүстік кіреберіс",
            location="США · JPEG раз в минуту / АҚШ · минутына бір кадр",
            url="https://www.nps.gov/webcams-yell/mammoth_arch.jpg", source_type="jpeg_snapshot", fps_limit=1),
        CameraSource(id="cam-public-yellowstone-electric", name="Yellowstone — Electric Peak",
            location="США · JPEG раз в минуту / АҚШ · минутына бір кадр",
            url="https://www.nps.gov/webcams-yell/mammoth_electric.jpg", source_type="jpeg_snapshot", fps_limit=1),
        CameraSource(
            id="cam-hypermarket-frozen",
            name="Hypermarket - Frozen Aisle",
            location="Hypermarket / Frozen foods",
            url="retail-scene://cam-hypermarket-frozen",
            source_type="retail_scene",
            fps_limit=8,
            zones=[
                Zone(id="zone-hyp-frozen-shelf", name="Frozen shelves", kind="shelf"),
                Zone(id="zone-hyp-frozen-aisle", name="Aisle work area", kind="work_area"),
            ],
        ),
        CameraSource(
            id="cam-supermarket-produce",
            name="Supermarket - Fresh Produce",
            location="Supermarket / Produce",
            url="retail-scene://cam-supermarket-produce",
            source_type="retail_scene",
            fps_limit=8,
            zones=[
                Zone(id="zone-sm-produce-display", name="Produce display", kind="shelf"),
                Zone(id="zone-sm-produce-aisle", name="Produce aisle", kind="work_area"),
            ],
        ),
        CameraSource(
            id="cam-supermarket-beverage",
            name="Supermarket - Beverages Aisle",
            location="Supermarket / Beverages",
            url="retail-scene://cam-supermarket-beverage",
            source_type="retail_scene",
            fps_limit=8,
            zones=[
                Zone(id="zone-sm-bev-shelf", name="Beverage shelves", kind="shelf"),
                Zone(id="zone-sm-bev-aisle", name="Beverage aisle", kind="work_area"),
            ],
        ),
        CameraSource(
            id="cam-supermarket-checkout",
            name="Supermarket - Checkout Lanes",
            location="Supermarket / Checkout",
            url="retail-scene://cam-supermarket-checkout",
            source_type="retail_scene",
            fps_limit=8,
            zones=[
                Zone(id="zone-sm-checkout-desks", name="Checkout desks", kind="checkout"),
                Zone(id="zone-sm-checkout-queue", name="Queue area", kind="work_area"),
            ],
        ),
        CameraSource(
            id="cam-warehouse-stock",
            name="Warehouse - Stock Backroom",
            location="Warehouse / Stock",
            url="retail-scene://cam-warehouse-stock",
            source_type="retail_scene",
            fps_limit=8,
            zones=[
                Zone(id="zone-warehouse-racks", name="Storage racks", kind="stock"),
                Zone(id="zone-warehouse-aisle", name="Warehouse aisle", kind="work_area"),
            ],
        ),
        CameraSource(
            id="cam-mall-entrance",
            name="Shopping Mall - Main Entrance",
            location="Shopping mall / Entrance",
            url="retail-scene://cam-mall-entrance",
            source_type="retail_scene",
            fps_limit=8,
            zones=[
                Zone(id="zone-mall-entry", name="Entrance doors", kind="entrance"),
                Zone(id="zone-mall-security", name="Security post", kind="work_area"),
            ],
        ),
        CameraSource(
            id="cam-buffalo-trace",
            name="Buffalo Trace Factory - Public MJPEG",
            location="Public live camera fallback",
            url="http://camera.buffalotrace.com/mjpg/video.mjpg",
            source_type="live_mjpeg",
            enabled=False,
            fps_limit=2,
            zones=[
                Zone(id="zone-bt-area", name="Production area", kind="work_area"),
                Zone(id="zone-bt-entrance", name="Entrance", kind="entrance"),
            ],
        ),
    ]
    if os.getenv("AI_MONITOR_PUBLIC_ONLY") == "1":
        return [source for source in sources if source.id.startswith("cam-public-")]
    return sources


PUBLIC_SOURCE_PAGES = {
    "cam-public-rtsp-kz": "https://rtsp.kz/",
    "cam-public-yellowstone-arch": "https://www.nps.gov/media/webcam/view.htm?id=81B468BC-1DD8-B71B-0BBA4C383E179188",
    "cam-public-yellowstone-electric": "https://www.nps.gov/media/webcam/view.htm?id=81B468AB-1DD8-B71B-0BE84D8E8E0F1112",
}

PUBLIC_VIDEO_SOURCES: list[PublicVideoSource] = [
    PublicVideoSource(
        id=source.id,
        title=f"{source.name} ({source.source_type})",
        camera_id=source.id,
        source_url=source.url,
        scenario=source.location,
        license_note=("Официально опубликованная публичная камера. Страница владельца: " + PUBLIC_SOURCE_PAGES[source.id]
            if source.id in PUBLIC_SOURCE_PAGES else "Local synthetic scene or operator-configured stream."),
        supported_signals=(["person detection"] if source.source_type == "jpeg_snapshot"
            else ["person detection", "presence", "absence", "zone dwell"]),
    )
    for source in default_sources()
]

DETECTION_CAPABILITIES: list[DetectionCapability] = [
    DetectionCapability(
        id="people-bbox",
        title="Person detection in frame",
        readiness="demo_ready",
        confidence=0.86,
        what_it_checks="Detects people and stores bounding boxes per camera.",
        evidence=["bbox", "camera_id", "confidence", "timestamp"],
        current_limitations="Accuracy depends on camera angle, distance, lighting and local YOLO model availability.",
        tz_mapping="Weeks 2-4: detection and tracking inside every stream.",
    ),
    DetectionCapability(
        id="employee-absence",
        title="Employee presence and absence",
        readiness="heuristic_ready",
        confidence=0.78,
        what_it_checks="Tracks whether a person is present in configured work zones longer than threshold windows.",
        evidence=["first_seen", "last_seen", "zone", "threshold"],
        current_limitations="No exact identity recognition in MVP; employee is inferred from camera zone and visual rules.",
        tz_mapping="Weeks 4-6: first/last appearance and absence alerts.",
    ),
    DetectionCapability(
        id="shelf-dwell",
        title="Long dwell near shelf",
        readiness="heuristic_ready",
        confidence=0.68,
        what_it_checks="Flags a person staying in shelf zones for longer than configured dwell seconds.",
        evidence=["track_id", "zone", "dwell_seconds", "snapshot"],
        current_limitations="Operator confirmation is required; this is a risk candidate, not proof.",
        tz_mapping="Weeks 8-10: simple suspicious-behavior heuristics.",
    ),
    DetectionCapability(
        id="multi-camera-check",
        title="Cross-camera scene check",
        readiness="pilot_needed",
        confidence=0.46,
        what_it_checks="Checks whether another camera in the same location reports a related event in a close time window.",
        evidence=["camera_id", "location", "timestamp_window", "event_type"],
        current_limitations="No person re-identification between cameras in MVP.",
        tz_mapping="Weeks 10-12: second-angle validation without cross-camera identity mixing.",
    ),
]

# Capabilities exposed by the safety pipeline, replacing the retail-only list.
DETECTION_CAPABILITIES = [
    DetectionCapability(
        id=kind, title=title, readiness="heuristic_ready", confidence=0.6,
        what_it_checks=description, evidence=evidence,
        current_limitations="Сигнал-кандидат; требуется проверка оператором и пилот на реальных камерах.",
        tz_mapping="AI Camera Safety MVP",
    )
    for kind, title, description, evidence in [
        ("normal", "Қалыпты әрекет / Нормальная активность", "Успешный YOLO-анализ без сработавших правил риска.", ["person", "model_available"]),
        ("suspicious", "Күдікті әрекет / Подозрительная активность", "Длительное нахождение в зоне полки или склада; не доказательство кражи.", ["track", "zone", "duration"]),
        ("fight", "Төбелес / Возможная драка", "Устойчивые резкие движения двух сближенных людей.", ["motion", "proximity", "duration"]),
        ("fall", "Құлау / Возможное падение", "Переход из вертикального положения в горизонтальное и удержание положения.", ["bbox_transition", "duration"]),
        ("danger", "Қауіпті жағдай / Опасная ситуация", "Человек внутри настроенного полигона запрещённой зоны.", ["polygon", "track"]),
    ]
]
