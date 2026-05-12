from __future__ import annotations

import math

from core.states import TrackState
from core.structures import Track
from events.id_utils import named_track_event_fields, pair_track_event_fields


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


class SplitMergeResolver:
    def __init__(
        self,
        duplicate_iou_threshold: float = 0.90,
        merged_iou_threshold: float = 0.6,
        center_threshold: float = 12.0,
    ) -> None:
        self.duplicate_iou_threshold = duplicate_iou_threshold
        self.merged_iou_threshold = merged_iou_threshold
        self.center_threshold = center_threshold
        self.last_events: list[dict] = []

    def __call__(self, tracks: list[Track]) -> list[Track]:
        self.last_events = []
        remove_ids: set[int] = set()
        for i in range(len(tracks)):
            if tracks[i].track_id in remove_ids:
                continue
            obs_a = {obs.frame_idx: obs for obs in tracks[i].trajectory}
            for j in range(i + 1, len(tracks)):
                if tracks[j].track_id in remove_ids:
                    continue
                obs_b = {obs.frame_idx: obs for obs in tracks[j].trajectory}
                common_frames = sorted(set(obs_a) & set(obs_b))
                if len(common_frames) < 3:
                    continue

                ious = []
                distances = []
                first_delta = None
                last_delta = None
                for frame_idx in common_frames:
                    a = obs_a[frame_idx]
                    b = obs_b[frame_idx]
                    ious.append(_bbox_iou(a.bbox, b.bbox))
                    distances.append(math.hypot(a.center[0] - b.center[0], a.center[1] - b.center[1]))
                    delta = a.center[0] - b.center[0]
                    if first_delta is None:
                        first_delta = delta
                    last_delta = delta

                mean_iou = sum(ious) / len(ious)
                mean_dist = sum(distances) / len(distances)
                if mean_iou >= self.duplicate_iou_threshold and mean_dist <= self.center_threshold:
                    keep = tracks[i]
                    drop = tracks[j]
                    if (
                        len(drop.trajectory) > len(keep.trajectory)
                        or (drop.state == TrackState.CONFIRMED and keep.state != TrackState.CONFIRMED)
                    ):
                        keep, drop = drop, keep
                    remove_ids.add(drop.track_id)
                    self.last_events.append(
                        {
                            "type": "duplicate_track_suppressed",
                            "mean_iou": float(mean_iou),
                            "mean_distance": float(mean_dist),
                            **named_track_event_fields(keep, "keep"),
                            **named_track_event_fields(drop, "drop"),
                        }
                    )
                    continue

                if mean_iou >= self.merged_iou_threshold:
                    self.last_events.append(
                        {
                            "type": "merged_state",
                            "mean_iou": float(mean_iou),
                            "mean_distance": float(mean_dist),
                            **pair_track_event_fields(tracks[i], tracks[j]),
                        }
                    )

                if first_delta is not None and last_delta is not None and first_delta * last_delta < 0 and min(distances) <= self.center_threshold:
                    self.last_events.append(
                        {
                            "type": "possible_id_swap",
                            "min_distance": float(min(distances)),
                            **pair_track_event_fields(tracks[i], tracks[j]),
                        }
                    )

        return [track for track in tracks if track.track_id not in remove_ids]
