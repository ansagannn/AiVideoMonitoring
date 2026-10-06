from __future__ import annotations

from dataclasses import dataclass, field

import db
from models import CameraSource, MonitoringSettings, parse_utc, utc_now
from tracking import MultiCameraTracker, RuntimeTrack


@dataclass
class CameraRuleState:
    last_employee_seen_at: str | None = None
    absence_open: bool = False
    emitted_presence_tracks: set[str] = field(default_factory=set)


class RuleEngine:
    def __init__(self) -> None:
        self._state: dict[str, CameraRuleState] = {}

    def evaluate(
        self,
        source: CameraSource,
        tracks: list[RuntimeTrack],
        settings: MonitoringSettings,
        tracker: MultiCameraTracker,
    ) -> None:
        state = self._state.setdefault(source.id, CameraRuleState())
        active_tracks = [track for track in tracks if track.missed_frames <= 2 and track.zone is not None]
        employee_tracks = [track for track in active_tracks if track.zone and track.zone.kind in {"work_area", "checkout", "entrance"}]

        if employee_tracks:
            state.last_employee_seen_at = utc_now()
            state.absence_open = False
            for track in employee_tracks:
                if track.id not in state.emitted_presence_tracks:
                    state.emitted_presence_tracks.add(track.id)
                    track.emitted_rules.add("employee_presence")
                    db.create_event(
                        source.id,
                        "employee_presence",
                        f"Track {track.local_id} appeared in {track.zone.name}.",
                        zone_name=track.zone.name,
                        confidence=track.confidence,
                        rule_id="employee_presence",
                        track_id=track.id,
                        zone_id=track.zone.id,
                        evidence={"track_id": track.id, "bbox": track.bbox},
                    )
        else:
            if state.last_employee_seen_at is None:
                state.last_employee_seen_at = utc_now()
            absence_minutes = (parse_utc(utc_now()) - parse_utc(state.last_employee_seen_at)).total_seconds() / 60
            if absence_minutes >= settings.absence_threshold_minutes and not state.absence_open:
                state.absence_open = True
                db.create_event(
                    source.id,
                    "employee_absence",
                    f"No employee-like person in configured work zones for {absence_minutes:.1f} minutes.",
                    zone_name="Work zone",
                    confidence=0.82,
                    rule_id="employee_absence",
                    evidence={"absence_minutes": round(absence_minutes, 1), "threshold": settings.absence_threshold_minutes},
                )

        for track in active_tracks:
            if not track.zone:
                continue
            if track.zone.kind in {"shelf", "stock"}:
                dwell = tracker.dwell_seconds(track, track.zone.id)
                if dwell >= settings.shelf_dwell_seconds and "visitor_shelf_dwell" not in track.emitted_rules:
                    track.emitted_rules.add("visitor_shelf_dwell")
                    db.create_event(
                        source.id,
                        "visitor_shelf_dwell",
                        f"Track {track.local_id} stayed near {track.zone.name} for {dwell:.0f} seconds.",
                        zone_name=track.zone.name,
                        confidence=max(0.5, track.confidence),
                        rule_id="visitor_shelf_dwell",
                        track_id=track.id,
                        zone_id=track.zone.id,
                        evidence={"track_id": track.id, "dwell_seconds": round(dwell, 1), "bbox": track.bbox},
                    )
                if dwell >= settings.shelf_dwell_seconds * 1.5 and "back_to_camera" not in track.emitted_rules:
                    track.emitted_rules.add("back_to_camera")
                    db.create_event(
                        source.id,
                        "back_to_camera",
                        f"Track {track.local_id} stayed near shelf with a low-confidence orientation heuristic.",
                        zone_name=track.zone.name,
                        confidence=0.58,
                        rule_id="back_to_camera",
                        track_id=track.id,
                        zone_id=track.zone.id,
                        evidence={"track_id": track.id, "heuristic": "long_shelf_dwell"},
                    )
