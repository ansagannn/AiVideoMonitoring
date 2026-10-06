from __future__ import annotations

import itertools
from uuid import uuid4
from dataclasses import dataclass, field

from models import Track, Zone, parse_utc, utc_now


@dataclass
class RuntimeDetection:
    bbox: list[int]
    confidence: float
    class_name: str = "person"


@dataclass
class RuntimeTrack:
    local_id: int
    camera_id: str
    first_seen_at: str
    last_seen_at: str
    bbox: list[int]
    confidence: float
    missed_frames: int = 0
    zone: Zone | None = None
    dwell_by_zone: dict[str, str] = field(default_factory=dict)
    emitted_rules: set[str] = field(default_factory=set)
    session_id: str = field(default_factory=lambda: uuid4().hex[:8])

    @property
    def id(self) -> str:
        return f"{self.camera_id}:{self.session_id}:{self.local_id}"

    def to_model(self) -> Track:
        return Track(
            id=self.id,
            camera_id=self.camera_id,
            local_track_id=self.local_id,
            first_seen_at=self.first_seen_at,
            last_seen_at=self.last_seen_at,
            zone_id=self.zone.id if self.zone else None,
            zone_name=self.zone.name if self.zone else None,
            confidence=self.confidence,
            bbox=self.bbox,
            active=self.missed_frames <= 3,
        )


def iou(a: list[int], b: list[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    denom = area_a + area_b - inter_area
    return inter_area / denom if denom else 0.0


def _point_in_poly(x: float, y: float, polygon: list[tuple[float, float]]) -> bool:
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        intersects = ((yi > y) != (yj > y)) and x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi
        if intersects:
            inside = not inside
        j = i
    return inside


def zone_for_bbox(bbox: list[int], zones: list[Zone], frame_size: tuple[int, int]) -> Zone | None:
    if not zones:
        return None
    width, height = frame_size
    cx = ((bbox[0] + bbox[2]) / 2) / max(width, 1)
    cy = ((bbox[1] + bbox[3]) / 2) / max(height, 1)
    for zone in zones:
        if zone.polygon and _point_in_poly(cx, cy, zone.polygon):
            return zone
    zones = [zone for zone in zones if not zone.polygon]
    shelf_zones = [zone for zone in zones if zone.kind in {"shelf", "stock"}]
    work_zones = [zone for zone in zones if zone.kind in {"work_area", "checkout", "entrance"}]
    if shelf_zones and (cx < 0.34 or cx > 0.66 or cy < 0.58):
        return shelf_zones[0]
    if work_zones:
        return work_zones[0]
    return next((zone for zone in zones if zone.kind != "restricted" and not zone.polygon), None)


class MultiCameraTracker:
    def __init__(self, max_missed_frames: int = 6, iou_threshold: float = 0.25) -> None:
        self._tracks: dict[str, dict[int, RuntimeTrack]] = {}
        self._next_ids: dict[str, itertools.count] = {}
        self.max_missed_frames = max_missed_frames
        self.iou_threshold = iou_threshold

    def update(
        self,
        camera_id: str,
        detections: list[RuntimeDetection],
        zones: list[Zone],
        frame_size: tuple[int, int],
    ) -> list[RuntimeTrack]:
        now = utc_now()
        camera_tracks = self._tracks.setdefault(camera_id, {})
        next_ids = self._next_ids.setdefault(camera_id, itertools.count(1))
        unmatched = set(camera_tracks.keys())

        for detection in detections:
            best_track_id: int | None = None
            best_score = 0.0
            for track_id in unmatched:
                track = camera_tracks[track_id]
                score = iou(track.bbox, detection.bbox)
                if score < self.iou_threshold and score > 0.08:
                    ax, ay = (track.bbox[0]+track.bbox[2])/2, (track.bbox[1]+track.bbox[3])/2
                    bx, by = (detection.bbox[0]+detection.bbox[2])/2, (detection.bbox[1]+detection.bbox[3])/2
                    scale = max(track.bbox[3]-track.bbox[1], detection.bbox[3]-detection.bbox[1], 1)
                    if ((ax-bx)**2 + (ay-by)**2)**0.5 / scale < 0.6:
                        score = self.iou_threshold
                if score > best_score:
                    best_score = score
                    best_track_id = track_id
            if best_track_id is not None and best_score >= self.iou_threshold:
                track = camera_tracks[best_track_id]
                track.bbox = detection.bbox
                track.confidence = detection.confidence
                track.last_seen_at = now
                track.missed_frames = 0
                old_zone = track.zone
                track.zone = zone_for_bbox(detection.bbox, zones, frame_size)
                if old_zone and (not track.zone or old_zone.id != track.zone.id):
                    track.dwell_by_zone.pop(old_zone.id, None)
                if track.zone and track.zone.id not in track.dwell_by_zone:
                    track.dwell_by_zone[track.zone.id] = now
                unmatched.discard(best_track_id)
            else:
                local_id = next(next_ids)
                zone = zone_for_bbox(detection.bbox, zones, frame_size)
                track = RuntimeTrack(
                    local_id=local_id,
                    camera_id=camera_id,
                    first_seen_at=now,
                    last_seen_at=now,
                    bbox=detection.bbox,
                    confidence=detection.confidence,
                    zone=zone,
                )
                if zone:
                    track.dwell_by_zone[zone.id] = now
                camera_tracks[local_id] = track

        for track_id in list(unmatched):
            track = camera_tracks[track_id]
            track.missed_frames += 1
            if track.missed_frames > self.max_missed_frames:
                del camera_tracks[track_id]

        return list(camera_tracks.values())

    def active_tracks(self, camera_id: str | None = None) -> list[RuntimeTrack]:
        if camera_id:
            return [track for track in self._tracks.get(camera_id, {}).values() if track.missed_frames <= self.max_missed_frames]
        return [
            track
            for camera_tracks in self._tracks.values()
            for track in camera_tracks.values()
            if track.missed_frames <= self.max_missed_frames
        ]

    @staticmethod
    def dwell_seconds(track: RuntimeTrack, zone_id: str) -> float:
        started = track.dwell_by_zone.get(zone_id)
        if not started:
            return 0.0
        return max(0.0, (parse_utc(track.last_seen_at) - parse_utc(started)).total_seconds())
