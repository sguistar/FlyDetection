from __future__ import annotations

import copy
import math

from core.structures import Track


def _sorted_obs(track: Track):
    return sorted(track.trajectory, key=lambda obs: obs.frame_idx)


def _anchor_center(track: Track, *, first: bool, window: int = 3) -> tuple[float, float] | None:
    observations = _sorted_obs(track)
    if not observations:
        return None
    items = observations[:window] if first else observations[-window:]
    x = sum(float(obs.center[0]) for obs in items) / max(len(items), 1)
    y = sum(float(obs.center[1]) for obs in items) / max(len(items), 1)
    return float(x), float(y)


def _trajectory_len(track: Track) -> int:
    return len(track.trajectory)


def _speed_between(
    left_center: tuple[float, float] | None,
    right_center: tuple[float, float] | None,
    gap: int,
) -> float:
    if left_center is None or right_center is None or gap <= 0:
        return math.inf
    return float(
        math.hypot(right_center[0] - left_center[0], right_center[1] - left_center[1]) / max(gap, 1)
    )


def apply_slot_stickiness(
    tracks: list[Track],
    *,
    max_fragment_len: int = 20,
    max_gap: int = 20,
    max_speed: float = 14.0,
    min_anchor_len: int = 24,
) -> list[Track]:
    """Relabel short A->B->A slot excursions back to the surrounding stable slot when motion is continuous.

    当运动连续时，将短暂的 A->B->A 槽位跳变重新标回周围稳定槽位。
    """
    updated = [copy.deepcopy(track) for track in tracks]
    if len(updated) < 3:
        return updated

    by_slot: dict[int, list[Track]] = {}
    for track in updated:
        if track.identity_slot is None:
            continue
        by_slot.setdefault(int(track.identity_slot), []).append(track)
    for slot_tracks in by_slot.values():
        slot_tracks.sort(key=lambda item: (item.start_frame, item.end_frame, item.track_id))

    ordered = sorted(updated, key=lambda track: (track.start_frame, track.end_frame, track.track_id))
    for mid in ordered:
        if mid.identity_slot is None:
            continue
        if _trajectory_len(mid) > max_fragment_len:
            continue

        previous_candidates = [
            track
            for track in updated
            if track.identity_slot is not None
            and track.identity_slot != mid.identity_slot
            and track.end_frame < mid.start_frame
        ]
        next_candidates = [
            track
            for track in updated
            if track.identity_slot is not None
            and track.identity_slot != mid.identity_slot
            and track.start_frame > mid.end_frame
        ]
        if not previous_candidates or not next_candidates:
            continue

        prev = max(previous_candidates, key=lambda track: (track.end_frame, track.track_id))
        next_track = min(next_candidates, key=lambda track: (track.start_frame, track.track_id))
        if prev.identity_slot != next_track.identity_slot:
            continue
        if _trajectory_len(prev) < min_anchor_len or _trajectory_len(next_track) < min_anchor_len:
            continue

        gap_before = mid.start_frame - prev.end_frame
        gap_after = next_track.start_frame - mid.end_frame
        if gap_before <= 0 or gap_after <= 0 or gap_before > max_gap or gap_after > max_gap:
            continue

        prev_end = _anchor_center(prev, first=False)
        mid_start = _anchor_center(mid, first=True)
        mid_end = _anchor_center(mid, first=False)
        next_start = _anchor_center(next_track, first=True)
        speed_before = _speed_between(prev_end, mid_start, gap_before)
        speed_after = _speed_between(mid_end, next_start, gap_after)
        direct_speed = _speed_between(prev_end, next_start, max(next_track.start_frame - prev.end_frame, 1))
        if max(speed_before, speed_after, direct_speed) > max_speed:
            continue

        mid.identity_slot = int(prev.identity_slot)

    return updated
