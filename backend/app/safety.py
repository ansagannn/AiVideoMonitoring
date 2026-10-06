"""Temporal candidates from real YOLO person tracks, never a theft verdict."""
from dataclasses import dataclass, field
from math import hypot
import time

import db
from models import CameraSource, MonitoringSettings
from tracking import RuntimeTrack, iou


@dataclass
class Motion:
    previous: list[int]
    at: float
    standing_at: float = -1e9
    horizontal_since: float | None = None
    fast_since: float | None = None


class SafetyEngine:
    def __init__(self):
        self.motion: dict[str, Motion] = {}
        self.categories: dict[str, str] = {}

    def evaluate(self, source: CameraSource, tracks: list[RuntimeTrack], settings: MonitoringSettings, now=None):
        now = time.monotonic() if now is None else now
        active = [t for t in tracks if t.missed_frames == 0 and t.confidence >= settings.confidence_threshold]
        category = 'normal'
        fast = set()
        candidates = []
        for track in active:
            x1, y1, x2, y2 = track.bbox
            width, height = max(1, x2-x1), max(1, y2-y1)
            state = self.motion.setdefault(track.id, Motion(track.bbox[:], now))
            dt = now-state.at
            if dt > 1.5:
                state.horizontal_since = state.fast_since = None
            if height / width > 1.3:
                state.standing_at = now
                state.horizontal_since = None
            elif width / height > 1.3 and now-state.standing_at < 8:
                if state.horizontal_since is None:
                    state.horizontal_since = now
                if now-state.horizontal_since >= settings.fall_hold_seconds:
                    candidates.append(('fall', track))
            else:
                state.horizontal_since = None
            px1, py1, px2, py2 = state.previous
            speed = hypot((x1+x2-px1-px2)/2, (y1+y2-py1-py2)/2) / max(height, py2-py1, 1) / max(dt, 0.01)
            if 0 < dt <= 1.5 and speed > settings.fight_motion_threshold:
                if state.fast_since is None:
                    state.fast_since = now
                if now-state.fast_since >= 0.7:
                    fast.add(track.id)
            else:
                state.fast_since = None
            if track.zone:
                if track.zone.kind == 'restricted' and track.zone.polygon:
                    candidates.append(('danger', track))
                elif track.zone.kind in {'shelf', 'stock'}:
                    from tracking import MultiCameraTracker
                    if MultiCameraTracker.dwell_seconds(track, track.zone.id) >= settings.shelf_dwell_seconds:
                        candidates.append(('suspicious', track))
            state.previous, state.at = track.bbox[:], now
        for a in active:
            for b in active:
                if a.id < b.id and a.id in fast and b.id in fast and iou(a.bbox,b.bbox) > 0.05:
                    candidates.append(('fight', a))
        rank = {'normal':0, 'suspicious':1, 'danger':2, 'fall':3, 'fight':4}
        for kind, track in candidates:
            if rank[kind] > rank[category]: category = kind
            db.create_event(source.id, kind, zone_name=track.zone.name if track.zone else 'Кадр / Кадр', confidence=track.confidence, rule_id=f'safety_{kind}', track_id=track.id, evidence={'bbox':track.bbox,'method':'temporal_bbox_heuristic'})
        self.categories[source.id] = category
        self.motion = {key:value for key,value in self.motion.items() if now-value.at < 15}
        return category
