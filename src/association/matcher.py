from __future__ import annotations

import numpy as np

from core.states import TrackState
from core.structures import AssociationResult, Detection, Track

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover
    linear_sum_assignment = None


def hungarian_match(
    cost_matrix: np.ndarray,
    gate: float | None = None,
    *,
    track_indices: list[int] | None = None,
    detection_indices: list[int] | None = None,
    large_cost: float = 1e6,
) -> AssociationResult:
    if linear_sum_assignment is None:
        raise ImportError("scipy is required for Hungarian matching.")

    n_tracks, n_dets = cost_matrix.shape
    track_indices = list(range(n_tracks)) if track_indices is None else track_indices
    detection_indices = list(range(n_dets)) if detection_indices is None else detection_indices

    if n_tracks == 0:
        return AssociationResult([], [], detection_indices, cost_matrix)
    if n_dets == 0:
        return AssociationResult([], track_indices, [], cost_matrix)

    row_idx, col_idx = linear_sum_assignment(cost_matrix)
    matches: list[tuple[int, int]] = []
    matched_tracks = set()
    matched_dets = set()
    for row, col in zip(row_idx.tolist(), col_idx.tolist()):
        value = float(cost_matrix[row, col])
        if value >= large_cost:
            continue
        if gate is not None and value > gate:
            continue
        trk_idx = track_indices[row]
        det_idx = detection_indices[col]
        matches.append((trk_idx, det_idx))
        matched_tracks.add(trk_idx)
        matched_dets.add(det_idx)

    unmatched_tracks = [idx for idx in track_indices if idx not in matched_tracks]
    unmatched_detections = [idx for idx in detection_indices if idx not in matched_dets]
    return AssociationResult(matches, unmatched_tracks, unmatched_detections, cost_matrix)


def cascade_match(
    tracks: list[Track],
    detections: list[Detection],
    cost_matrix: np.ndarray,
    *,
    gate: float | None = None,
    large_cost: float = 1e6,
) -> AssociationResult:
    if len(tracks) == 0:
        return AssociationResult([], [], list(range(len(detections))), cost_matrix)
    if len(detections) == 0:
        return AssociationResult([], list(range(len(tracks))), [], cost_matrix)

    confirmed_indices = [
        idx
        for idx, track in enumerate(tracks)
        if track.state == TrackState.CONFIRMED
    ]
    lost_indices = [
        idx
        for idx, track in enumerate(tracks)
        if track.state == TrackState.LOST
    ]
    stage_two_indices = [
        idx
        for idx, track in enumerate(tracks)
        if track.state == TrackState.TENTATIVE
    ]
    remaining_detections = list(range(len(detections)))
    matches: list[tuple[int, int]] = []
    matched_tracks: set[int] = set()

    if confirmed_indices:
        sub_cost = cost_matrix[np.ix_(confirmed_indices, remaining_detections)]
        result = hungarian_match(
            sub_cost,
            gate=gate,
            track_indices=confirmed_indices,
            detection_indices=remaining_detections,
            large_cost=large_cost,
        )
        matches.extend(result.matches)
        matched_tracks.update(track_idx for track_idx, _ in result.matches)
        remaining_detections = result.unmatched_detections

    if lost_indices and remaining_detections:
        sub_cost = cost_matrix[np.ix_(lost_indices, remaining_detections)]
        result = hungarian_match(
            sub_cost,
            gate=gate,
            track_indices=lost_indices,
            detection_indices=remaining_detections,
            large_cost=large_cost,
        )
        matches.extend(result.matches)
        matched_tracks.update(track_idx for track_idx, _ in result.matches)
        remaining_detections = result.unmatched_detections

    if stage_two_indices and remaining_detections:
        sub_cost = cost_matrix[np.ix_(stage_two_indices, remaining_detections)]
        result = hungarian_match(
            sub_cost,
            gate=gate,
            track_indices=stage_two_indices,
            detection_indices=remaining_detections,
            large_cost=large_cost,
        )
        matches.extend(result.matches)
        matched_tracks.update(track_idx for track_idx, _ in result.matches)
        remaining_detections = result.unmatched_detections

    unmatched_tracks = [idx for idx in range(len(tracks)) if idx not in matched_tracks]
    return AssociationResult(matches, unmatched_tracks, remaining_detections, cost_matrix)


def apply_track_support_bias(
    tracks: list[Track],
    detections: list[Detection],
    cost_matrix: np.ndarray,
    *,
    score_matrix: np.ndarray | None = None,
    switch_risk_matrix: np.ndarray | None = None,
    motion_gate: float = 80.0,
    large_cost: float = 1e6,
    support_distance_thres: float = 42.0,
    support_reconnect_bonus: float = 0.12,
    lost_track_bonus: float = 0.06,
    fallback_cost_thres: float = 0.92,
    score_floor: float = 0.24,
    switch_risk_cap: float = 0.40,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Bias association toward track-supported or rescue detections without opening the gates too far.

    在不过度放宽门控的前提下，让关联更偏向轨迹支持或救援检测。
    """
    if cost_matrix.size == 0 or len(tracks) == 0 or len(detections) == 0:
        return cost_matrix, score_matrix, switch_risk_matrix

    track_index_by_id = {int(track.track_id): idx for idx, track in enumerate(tracks)}
    track_index_by_slot = {
        int(track.identity_slot): idx
        for idx, track in enumerate(tracks)
        if track.identity_slot is not None
    }

    for det_idx, det in enumerate(detections):
        candidate_indices: dict[int, float] = {}
        if det.is_track_supported and det.support_track_id is not None:
            support_idx = track_index_by_id.get(int(det.support_track_id))
            if support_idx is not None:
                candidate_indices[support_idx] = max(
                    candidate_indices.get(support_idx, 0.0),
                    support_reconnect_bonus,
                )
        if det.is_rescued and det.rescue_slot_id is not None:
            rescue_idx = track_index_by_slot.get(int(det.rescue_slot_id))
            if rescue_idx is not None:
                candidate_indices[rescue_idx] = max(
                    candidate_indices.get(rescue_idx, 0.0),
                    support_reconnect_bonus * 0.85,
                )

        for track_idx, base_bonus in candidate_indices.items():
            track = tracks[track_idx]
            predicted = track.predicted_center if track.predicted_center != (0.0, 0.0) else track.center
            center_distance = float(
                np.hypot(
                    det.center[0] - predicted[0],
                    det.center[1] - predicted[1],
                )
            )
            extra_radius = 6.0 * min(max(track.missed, 0), 4)
            if center_distance > float(support_distance_thres + extra_radius):
                continue

            total_bonus = base_bonus
            if track.state == TrackState.LOST:
                total_bonus += lost_track_bonus
            elif track.missed > 0:
                total_bonus += 0.5 * lost_track_bonus

            current_cost = float(cost_matrix[track_idx, det_idx])
            if current_cost >= large_cost:
                fallback_cost = center_distance / max(float(motion_gate), 1e-6)
                if det.is_rescued:
                    fallback_cost += 0.05
                if track.state == TrackState.LOST:
                    fallback_cost += 0.05 * min(track.missed, 3)
                if fallback_cost > fallback_cost_thres:
                    continue
                cost_matrix[track_idx, det_idx] = float(fallback_cost)
            else:
                cost_matrix[track_idx, det_idx] = float(max(current_cost - total_bonus, 0.0))

            if score_matrix is not None:
                distance_score = float(
                    np.clip(
                        1.0 - center_distance / max(float(support_distance_thres + extra_radius), 1e-6),
                        0.0,
                        1.0,
                    )
                )
                boosted_score = max(score_floor, 0.18 + 0.22 * distance_score)
                if track.state == TrackState.LOST:
                    boosted_score = max(boosted_score, score_floor + 0.04)
                score_matrix[track_idx, det_idx] = max(float(score_matrix[track_idx, det_idx]), float(boosted_score))

            if switch_risk_matrix is not None:
                switch_risk_matrix[track_idx, det_idx] = min(
                    float(switch_risk_matrix[track_idx, det_idx]),
                    float(switch_risk_cap),
                )

    return cost_matrix, score_matrix, switch_risk_matrix


def recover_track_supported_matches(
    tracks: list[Track],
    detections: list[Detection],
    assoc: AssociationResult,
    *,
    large_cost: float = 1e6,
    cost_margin: float = 0.08,
) -> AssociationResult:
    """Rewrite a small number of matches when a detection carries an explicit support target.

    当检测携带明确支持目标时，重写少量匹配结果。
    """
    if assoc.cost_matrix is None or len(tracks) == 0 or len(detections) == 0:
        return assoc

    track_index_by_id = {int(track.track_id): idx for idx, track in enumerate(tracks)}
    track_index_by_slot = {
        int(track.identity_slot): idx
        for idx, track in enumerate(tracks)
        if track.identity_slot is not None
    }
    matches = list(assoc.matches)
    unmatched_tracks = set(assoc.unmatched_tracks)
    unmatched_detections = set(assoc.unmatched_detections)
    det_to_match_pos = {det_idx: pos for pos, (_, det_idx) in enumerate(matches)}

    for det_idx, det in enumerate(detections):
        target_track_idx = None
        if det.is_track_supported and det.support_track_id is not None:
            target_track_idx = track_index_by_id.get(int(det.support_track_id))
        elif det.is_rescued and det.rescue_slot_id is not None:
            target_track_idx = track_index_by_slot.get(int(det.rescue_slot_id))
        if target_track_idx is None:
            continue

        target_cost = float(assoc.cost_matrix[target_track_idx, det_idx])
        if target_cost >= large_cost:
            continue

        matched_pos = det_to_match_pos.get(det_idx)
        if matched_pos is None:
            if target_track_idx not in unmatched_tracks:
                continue
            matches.append((target_track_idx, det_idx))
            det_to_match_pos[det_idx] = len(matches) - 1
            unmatched_tracks.discard(target_track_idx)
            unmatched_detections.discard(det_idx)
            continue

        current_track_idx, _ = matches[matched_pos]
        if current_track_idx == target_track_idx:
            continue
        if target_track_idx not in unmatched_tracks:
            continue

        current_cost = float(assoc.cost_matrix[current_track_idx, det_idx])
        target_track = tracks[target_track_idx]
        allowance = cost_margin
        if target_track.state == TrackState.LOST:
            allowance += 0.08 + 0.02 * min(target_track.missed, 3)
        elif target_track.missed > 0:
            allowance += 0.04
        if target_cost > current_cost + allowance:
            continue

        matches[matched_pos] = (target_track_idx, det_idx)
        unmatched_tracks.add(current_track_idx)
        unmatched_tracks.discard(target_track_idx)

    if (
        matches == assoc.matches
        and unmatched_tracks == set(assoc.unmatched_tracks)
        and unmatched_detections == set(assoc.unmatched_detections)
    ):
        return assoc
    return AssociationResult(
        matches=matches,
        unmatched_tracks=sorted(unmatched_tracks),
        unmatched_detections=sorted(unmatched_detections),
        cost_matrix=assoc.cost_matrix,
        score_matrix=assoc.score_matrix,
        switch_risk_matrix=assoc.switch_risk_matrix,
    )


def _slot_swap_eligible(track: Track, *, stable_hits: int, max_missed: int) -> bool:
    return (
        track.identity_slot is not None
        and track.identity_slot >= 0
        and track.state in {TrackState.CONFIRMED, TrackState.LOST}
        and track.hits >= stable_hits
        and track.missed <= max_missed
    )


def _preserves_local_order(track_a: Track, track_b: Track, det_a: Detection, det_b: Detection) -> bool:
    axis = np.asarray(
        [
            float(track_b.center[0] - track_a.center[0]),
            float(track_b.center[1] - track_a.center[1]),
        ],
        dtype=np.float32,
    )
    norm = float(np.linalg.norm(axis))
    if norm < 1e-6:
        return True
    axis /= max(norm, 1e-6)
    det_delta = np.asarray(
        [
            float(det_b.center[0] - det_a.center[0]),
            float(det_b.center[1] - det_a.center[1]),
        ],
        dtype=np.float32,
    )
    return float(det_delta @ axis) >= 0.0


def suppress_slot_swaps(
    tracks: list[Track],
    detections: list[Detection],
    assoc: AssociationResult,
    *,
    large_cost: float = 1e6,
    cost_margin: float = 0.08,
    distance_thres: float = 80.0,
    stable_hits: int = 6,
    max_missed: int = 2,
) -> AssociationResult:
    """Apply a conservative local rewrite when two stable slots appear to have swapped detections.

    当两个稳定槽位疑似交换检测结果时，执行保守的局部改写。
    """
    if assoc.cost_matrix is None or len(assoc.matches) < 2:
        return assoc

    matches = list(assoc.matches)
    changed = True
    while changed:
        changed = False
        for first_idx in range(len(matches)):
            track_idx_a, det_idx_a = matches[first_idx]
            track_a = tracks[track_idx_a]
            det_a = detections[det_idx_a]
            if not _slot_swap_eligible(track_a, stable_hits=stable_hits, max_missed=max_missed):
                continue

            for second_idx in range(first_idx + 1, len(matches)):
                track_idx_b, det_idx_b = matches[second_idx]
                track_b = tracks[track_idx_b]
                det_b = detections[det_idx_b]
                if not _slot_swap_eligible(track_b, stable_hits=stable_hits, max_missed=max_missed):
                    continue
                if track_a.identity_slot == track_b.identity_slot:
                    continue

                track_gap = float(
                    np.hypot(
                        track_a.center[0] - track_b.center[0],
                        track_a.center[1] - track_b.center[1],
                    )
                )
                det_gap = float(
                    np.hypot(
                        det_a.center[0] - det_b.center[0],
                        det_a.center[1] - det_b.center[1],
                    )
                )
                if track_gap > distance_thres or det_gap > distance_thres:
                    continue
                if _preserves_local_order(track_a, track_b, det_a, det_b):
                    continue

                alt_cost_a = float(assoc.cost_matrix[track_idx_a, det_idx_b])
                alt_cost_b = float(assoc.cost_matrix[track_idx_b, det_idx_a])
                cur_cost_a = float(assoc.cost_matrix[track_idx_a, det_idx_a])
                cur_cost_b = float(assoc.cost_matrix[track_idx_b, det_idx_b])
                if max(alt_cost_a, alt_cost_b) >= large_cost:
                    continue

                current_total = cur_cost_a + cur_cost_b
                alternate_total = alt_cost_a + alt_cost_b
                if alternate_total > current_total + cost_margin:
                    continue
                if not _preserves_local_order(track_a, track_b, det_b, det_a):
                    continue

                matches[first_idx] = (track_idx_a, det_idx_b)
                matches[second_idx] = (track_idx_b, det_idx_a)
                changed = True
                break
            if changed:
                break

    if matches == assoc.matches:
        return assoc
    return AssociationResult(
        matches=matches,
        unmatched_tracks=assoc.unmatched_tracks,
        unmatched_detections=assoc.unmatched_detections,
        cost_matrix=assoc.cost_matrix,
        score_matrix=assoc.score_matrix,
        switch_risk_matrix=assoc.switch_risk_matrix,
    )
