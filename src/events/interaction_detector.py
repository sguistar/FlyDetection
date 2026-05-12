from __future__ import annotations

import math
from collections import defaultdict

from core.structures import Track
from events.id_utils import pair_track_event_fields


def _bbox_iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


class InteractionDetector:
    def __init__(self, distance_thres: float = 25.0, overlap_iou_thres: float = 0.55) -> None:
        self.distance_thres = distance_thres
        self.overlap_iou_thres = overlap_iou_thres

    def detect(self, tracks: list[Track], frame_idx: int) -> list[dict]:
        events: list[dict] = []
        visible = []
        for track in tracks:
            latest = track.latest_observation()
            if latest is not None and latest.frame_idx == frame_idx:
                visible.append((track, latest))
        for i in range(len(visible)):
            for j in range(i + 1, len(visible)):
                track_a, obs_a = visible[i]
                track_b, obs_b = visible[j]
                distance = math.hypot(obs_a.center[0] - obs_b.center[0], obs_a.center[1] - obs_b.center[1])
                overlap = _bbox_iou(obs_a.bbox, obs_b.bbox)
                if distance < self.distance_thres or overlap >= self.overlap_iou_thres:
                    event_type = "merged_state" if overlap >= self.overlap_iou_thres else "interaction"
                    events.append(
                        {
                            "frame_idx": frame_idx,
                            "type": event_type,
                            "distance": float(distance),
                            "iou": float(overlap),
                            **pair_track_event_fields(track_a, track_b),
                        }
                    )
        return events

    def detect_many(self, tracks: list[Track]) -> list[dict]:
        by_frame: dict[int, list[tuple[Track, object]]] = defaultdict(list)
        for track in tracks:
            for obs in track.trajectory:
                by_frame[obs.frame_idx].append((track, obs))

        events: list[dict] = []
        active_pairs: set[tuple[int, int, str]] = set()
        for frame_idx in sorted(by_frame):
            current_pairs: set[tuple[int, int, str]] = set()
            visible = by_frame[frame_idx]
            for i in range(len(visible)):
                for j in range(i + 1, len(visible)):
                    track_a, obs_a = visible[i]
                    track_b, obs_b = visible[j]
                    distance = math.hypot(obs_a.center[0] - obs_b.center[0], obs_a.center[1] - obs_b.center[1])
                    overlap = _bbox_iou(obs_a.bbox, obs_b.bbox)
                    if distance >= self.distance_thres and overlap < self.overlap_iou_thres:
                        continue
                    event_type = "merged_state" if overlap >= self.overlap_iou_thres else "interaction"
                    pair_key = (min(track_a.track_id, track_b.track_id), max(track_a.track_id, track_b.track_id), event_type)
                    current_pairs.add(pair_key)
                    if pair_key not in active_pairs:
                        events.append(
                            {
                                "frame_idx": frame_idx,
                                "type": event_type,
                                "distance": float(distance),
                                "iou": float(overlap),
                                **pair_track_event_fields(track_a, track_b),
                            }
                        )
            active_pairs = current_pairs
        return events
