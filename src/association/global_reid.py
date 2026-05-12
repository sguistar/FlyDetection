from __future__ import annotations

from dataclasses import dataclass

import copy
import math

import numpy as np
try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover
    linear_sum_assignment = None

from core.structures import Track
from motion.kinematics import velocity_from_history


@dataclass
class TrackFragment:
    """A contiguous, trusted slice of a track used for offline slot reassignment.

    用于离线槽位重分配的一段连续且可信的轨迹片段。
    """
    parent_track_id: int
    fragment_index: int
    start_frame: int
    end_frame: int
    start_center: tuple[float, float]
    end_center: tuple[float, float]
    trajectory: list
    prototype_embedding: np.ndarray | None
    temporal_token: np.ndarray | None
    spatial_token: np.ndarray | None
    mean_shape: np.ndarray | None
    velocity: tuple[float, float]
    source_track: Track


@dataclass
class IdentitySlotState:
    """Running state for one global identity slot during offline reassignment.

    离线重分配过程中单个全局身份槽位的运行状态。
    """
    slot_id: int
    fragments: list[TrackFragment]
    prototype_embedding: np.ndarray | None
    temporal_token: np.ndarray | None
    spatial_token: np.ndarray | None
    mean_shape: np.ndarray | None
    last_end_frame: int
    last_center: tuple[float, float]


def _normalize(vector: np.ndarray | None) -> np.ndarray | None:
    if vector is None:
        return None
    vector = vector.astype(np.float32)
    return vector / (np.linalg.norm(vector) + 1e-8)


def _cosine_distance(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None and b is None:
        return 0.0
    if a is None or b is None:
        return 0.5
    na = np.linalg.norm(a) + 1e-8
    nb = np.linalg.norm(b) + 1e-8
    return float(1.0 - float(a @ b) / float(na * nb))


def _track_shape_distance(shape_a: np.ndarray | None, shape_b: np.ndarray | None) -> float:
    if shape_a is None and shape_b is None:
        return 0.0
    if shape_a is None or shape_b is None:
        return 0.25
    return float(np.linalg.norm(shape_a - shape_b) / np.sqrt(shape_a.shape[0]))


def _feature_items_between(
    track: Track,
    start_frame: int,
    end_frame: int,
    *,
    trusted_only: bool = False,
    trust_for: str = "temporal",
) -> list[dict]:
    """Extract feature snapshots inside a frame span, optionally keeping only trusted entries.

    提取指定帧范围内的特征快照，并可选择只保留可信条目。
    """
    filtered = [
        copy.deepcopy(item)
        for item in track.recent_feature_items(trusted_only=trusted_only, trust_for=trust_for)
        if item.get("frame_idx") is not None and start_frame <= int(item["frame_idx"]) <= end_frame
    ]
    if filtered:
        return filtered
    if trusted_only:
        return []
    return [copy.deepcopy(item) for item in track.feature_history]


def _compute_fragment_prototype(track: Track, *, start_frame: int, end_frame: int) -> np.ndarray | None:
    """Build a fragment prototype from trusted identity evidence before falling back to raw embeddings.

    优先使用可信身份证据构建片段原型，必要时再回退到原始嵌入。
    """
    trusted_items = _feature_items_between(
        track,
        start_frame,
        end_frame,
        trusted_only=True,
        trust_for="temporal",
    )
    trusted_vectors = [
        (
            float(item.get("reid_quality", 1.0)),
            item["identity"].astype(np.float32),
        )
        for item in trusted_items
        if item.get("identity") is not None
    ]
    if trusted_vectors:
        total_w = sum(weight for weight, _ in trusted_vectors)
        proto = sum(weight * vector for weight, vector in trusted_vectors) / max(total_w, 1e-8)
        return _normalize(proto)

    quality_by_frame = {
        int(item["frame_idx"]): float(item.get("reid_quality", 1.0))
        for item in track.feature_history
        if item.get("frame_idx") is not None and start_frame <= int(item["frame_idx"]) <= end_frame
    }
    weighted_vectors: list[tuple[float, np.ndarray]] = []
    for rec in track.embedding_records:
        if start_frame <= rec.frame_idx <= end_frame:
            weight = quality_by_frame.get(rec.frame_idx, 1.0)
            if weight > 0.0:
                weighted_vectors.append((weight, rec.vector.astype(np.float32)))
    if weighted_vectors:
        total_w = sum(weight for weight, _ in weighted_vectors)
        proto = sum(weight * vector for weight, vector in weighted_vectors) / max(total_w, 1e-8)
        return _normalize(proto)
    return _normalize(track.prototype_embedding if track.prototype_embedding is not None else track.mean_embedding())


def split_track_into_fragments(
    track: Track,
    *,
    min_fragment_len: int = 3,
    max_internal_gap: int = 3,
) -> list[tuple[int, int]]:
    """Cut one online track into offline fragments using long gaps and explicit breakpoints.

    根据长间隔和显式断点，将一条在线轨迹切分成离线片段。
    """
    observations = sorted(track.trajectory, key=lambda obs: obs.frame_idx)
    if not observations:
        return []

    breakpoints = set(track.fragment_breakpoints)
    spans: list[tuple[int, int]] = []
    start_idx = 0
    for idx in range(1, len(observations)):
        prev_obs = observations[idx - 1]
        curr_obs = observations[idx]
        force_break = curr_obs.frame_idx - prev_obs.frame_idx > max_internal_gap
        force_break = force_break or prev_obs.frame_idx in breakpoints or curr_obs.frame_idx in breakpoints
        if force_break:
            if idx - start_idx >= min_fragment_len:
                spans.append((start_idx, idx))
            start_idx = idx

    if len(observations) - start_idx >= min_fragment_len:
        spans.append((start_idx, len(observations)))
    if not spans:
        spans.append((0, len(observations)))
    return spans


def build_fragments_from_track(
    track: Track,
    *,
    min_fragment_len: int = 3,
    max_internal_gap: int = 3,
) -> list[TrackFragment]:
    """Convert a track into fragment objects with motion, shape, and identity summaries.

    将轨迹转换为带有运动、形状和身份摘要的片段对象。
    """
    observations = sorted(track.trajectory, key=lambda obs: obs.frame_idx)
    spans = split_track_into_fragments(track, min_fragment_len=min_fragment_len, max_internal_gap=max_internal_gap)
    fragments: list[TrackFragment] = []

    for fragment_index, (start_idx, end_idx) in enumerate(spans):
        fragment_obs = observations[start_idx:end_idx]
        if not fragment_obs:
            continue
        start_frame = fragment_obs[0].frame_idx
        end_frame = fragment_obs[-1].frame_idx
        feature_items = _feature_items_between(
            track,
            start_frame,
            end_frame,
            trusted_only=True,
            trust_for="spatial",
        )
        if not feature_items:
            feature_items = _feature_items_between(track, start_frame, end_frame)
        shape_vectors = [item["shape"] for item in feature_items if item.get("shape") is not None]
        mean_shape = np.mean(np.stack(shape_vectors, axis=0), axis=0).astype(np.float32) if shape_vectors else None
        fragments.append(
            TrackFragment(
                parent_track_id=track.track_id,
                fragment_index=fragment_index,
                start_frame=start_frame,
                end_frame=end_frame,
                start_center=fragment_obs[0].center,
                end_center=fragment_obs[-1].center,
                trajectory=copy.deepcopy(fragment_obs),
                prototype_embedding=_compute_fragment_prototype(track, start_frame=start_frame, end_frame=end_frame),
                temporal_token=_normalize(track.reid_state.temporal_token),
                spatial_token=_normalize(track.reid_state.spatial_token if track.reid_state.spatial_token is not None else track.mean_feature("spatial")),
                mean_shape=mean_shape,
                velocity=velocity_from_history([(obs.frame_idx, obs.center[0], obs.center[1]) for obs in fragment_obs]),
                source_track=track,
            )
        )
    return fragments


def _fragment_motion_distance(fragment: TrackFragment, slot: IdentitySlotState) -> float:
    gap = fragment.start_frame - slot.last_end_frame
    if gap <= 0:
        overlap_penalty = abs(gap) + 1
        return float(1.5 + 0.1 * overlap_penalty)
    vx, vy = slot.fragments[-1].velocity
    pred_x = slot.last_center[0] + vx * gap
    pred_y = slot.last_center[1] + vy * gap
    dist = math.hypot(pred_x - fragment.start_center[0], pred_y - fragment.start_center[1])
    return float(dist / max(gap, 1))


def _slot_cost(
    fragment: TrackFragment,
    slot: IdentitySlotState,
    *,
    max_link_gap: int,
) -> tuple[float, dict[str, float]]:
    """Score how well a fragment can continue an existing identity slot.

    评分一个片段续接到已有身份槽位的合理程度。
    """
    gap = fragment.start_frame - slot.last_end_frame
    appearance = _cosine_distance(fragment.prototype_embedding, slot.prototype_embedding)
    temporal = _cosine_distance(fragment.temporal_token, slot.temporal_token)
    spatial = _cosine_distance(fragment.spatial_token, slot.spatial_token)
    shape = _track_shape_distance(fragment.mean_shape, slot.mean_shape)
    motion = _fragment_motion_distance(fragment, slot)
    gap_penalty = 0.0
    if gap > max_link_gap:
        gap_penalty = 0.25 * float(gap - max_link_gap)
    total = 0.32 * motion + 0.28 * appearance + 0.18 * temporal + 0.12 * spatial + 0.10 * shape + gap_penalty
    return float(total), {"motion": motion, "appearance": appearance, "temporal": temporal, "spatial": spatial, "shape": shape}


def _append_fragment_to_slot(slot: IdentitySlotState, fragment: TrackFragment) -> None:
    """Attach a fragment to a slot and refresh the slot prototypes from merged evidence.

    将片段附加到槽位，并基于合并后的证据刷新槽位原型。
    """
    slot.fragments.append(fragment)
    slot.last_end_frame = fragment.end_frame
    slot.last_center = fragment.end_center

    embeddings = [frag.prototype_embedding for frag in slot.fragments if frag.prototype_embedding is not None]
    if embeddings:
        slot.prototype_embedding = _normalize(np.mean(np.stack(embeddings, axis=0), axis=0))

    temporal_tokens = [frag.temporal_token for frag in slot.fragments if frag.temporal_token is not None]
    if temporal_tokens:
        slot.temporal_token = _normalize(np.mean(np.stack(temporal_tokens, axis=0), axis=0))

    spatial_tokens = [frag.spatial_token for frag in slot.fragments if frag.spatial_token is not None]
    if spatial_tokens:
        slot.spatial_token = _normalize(np.mean(np.stack(spatial_tokens, axis=0), axis=0))

    shape_vectors = [frag.mean_shape for frag in slot.fragments if frag.mean_shape is not None]
    if shape_vectors:
        slot.mean_shape = np.mean(np.stack(shape_vectors, axis=0), axis=0).astype(np.float32)


def _allow_slot_merge(
    best_cost: float,
    best_terms: dict[str, float] | None,
    *,
    merge_threshold: float,
    appearance_threshold: float,
    shape_threshold: float,
    spatial_threshold: float,
    motion_threshold: float,
) -> bool:
    return (
        best_terms is not None
        and best_cost <= merge_threshold
        and best_terms["appearance"] <= appearance_threshold
        and best_terms["shape"] <= shape_threshold
        and best_terms["spatial"] <= spatial_threshold
        and best_terms["motion"] <= motion_threshold
    )


def _should_soft_assign_slot(
    fragment: TrackFragment,
    slot: IdentitySlotState | None,
    terms: dict[str, float] | None,
    *,
    appearance_threshold: float,
    shape_threshold: float,
) -> bool:
    if slot is None or terms is None:
        return False
    if fragment.start_frame <= slot.last_end_frame:
        return False
    return (
        terms["appearance"] <= min(appearance_threshold + 0.18, 0.55)
        and terms["temporal"] <= 0.40
        and terms["spatial"] <= 0.45
        and terms["shape"] <= min(shape_threshold + 0.15, 0.70)
    )


def _merge_slot_fragments(slot: IdentitySlotState) -> Track:
    """Rebuild one output track by stitching all fragments assigned to the same slot.

    拼接分配到同一槽位的所有片段，重建一条输出轨迹。
    """
    base = copy.deepcopy(slot.fragments[0].source_track)
    base.track_id = slot.slot_id
    base.identity_slot = slot.slot_id

    all_obs = []
    all_interpolated: set[int] = set()
    all_breakpoints: list[int] = []
    all_embedding_records: list = []
    all_feature_items: list[dict] = []
    all_embeddings: list[np.ndarray] = []
    all_quarantine: list[np.ndarray] = []

    for fragment in sorted(slot.fragments, key=lambda frag: frag.start_frame):
        source_track = fragment.source_track
        start_frame = fragment.start_frame
        end_frame = fragment.end_frame
        all_obs.extend(copy.deepcopy(fragment.trajectory))
        all_interpolated |= {frame for frame in source_track.interpolated_frames if start_frame <= frame <= end_frame}
        all_breakpoints.extend([frame for frame in source_track.fragment_breakpoints if start_frame <= frame <= end_frame])
        all_embedding_records.extend(
            copy.deepcopy(rec)
            for rec in source_track.embedding_records
            if start_frame <= rec.frame_idx <= end_frame
        )
        all_feature_items.extend(_feature_items_between(source_track, start_frame, end_frame))
        all_embeddings.extend([rec.vector.astype(np.float32) for rec in source_track.embedding_records if start_frame <= rec.frame_idx <= end_frame])
        if not source_track.embedding_records and source_track.embedding_history:
            all_embeddings.extend([emb.astype(np.float32) for emb in source_track.embedding_history])
        all_quarantine.extend(copy.deepcopy(source_track.reid_state.quarantine_embeddings))

    by_frame = {}
    for obs in all_obs:
        prev = by_frame.get(obs.frame_idx)
        if prev is None or obs.conf > prev.conf:
            by_frame[obs.frame_idx] = obs

    base.trajectory = sorted(by_frame.values(), key=lambda obs: obs.frame_idx)
    base.embedding_records = all_embedding_records
    base.embedding_history = all_embeddings
    base.feature_history = all_feature_items
    base.interpolated_frames = all_interpolated
    base.fragment_breakpoints = sorted(set(all_breakpoints))
    base.reid_state.quarantine_embeddings = all_quarantine[-6:]
    base.reid_state.short_term_embeddings = all_embeddings[-6:]
    base.reid_state.memory_reliability = max((fragment.source_track.reid_state.memory_reliability for fragment in slot.fragments), default=0.0)
    base.prototype_embedding = slot.prototype_embedding
    base.prototype_updates = len(all_embeddings)
    base.reid_state.temporal_token = slot.temporal_token
    base.reid_state.spatial_token = slot.spatial_token

    latest = base.latest_observation()
    if latest is not None:
        base.bbox = latest.bbox
        base.center = latest.center
        base.last_frame = latest.frame_idx
        base.conf = latest.conf
    base.stats_cache.clear()
    return base


def _slot_first_frame(slot: IdentitySlotState) -> int:
    return min(fragment.start_frame for fragment in slot.fragments)


def _consolidate_adjacent_slots_once(
    slots: list[IdentitySlotState],
    *,
    max_link_gap: int,
    merge_threshold: float,
    appearance_threshold: float,
    shape_threshold: float,
    spatial_threshold: float,
    motion_threshold: float,
) -> list[IdentitySlotState]:
    """Do one conservative pass to merge neighboring slots that are still obviously continuous.

    保守地执行一轮合并，将明显连续的相邻槽位合并起来。
    """
    if len(slots) < 2:
        return slots

    ordered = sorted(slots, key=lambda slot: (_slot_first_frame(slot), slot.slot_id))
    merged_slots: list[IdentitySlotState] = []
    idx = 0
    while idx < len(ordered):
        current = ordered[idx]
        if idx + 1 < len(ordered):
            nxt = ordered[idx + 1]
            next_fragment = min(nxt.fragments, key=lambda fragment: fragment.start_frame)
            cost, terms = _slot_cost(next_fragment, current, max_link_gap=max_link_gap)
            if _allow_slot_merge(
                cost,
                terms,
                merge_threshold=merge_threshold,
                appearance_threshold=appearance_threshold,
                shape_threshold=shape_threshold,
                spatial_threshold=spatial_threshold,
                motion_threshold=motion_threshold,
            ):
                for fragment in sorted(nxt.fragments, key=lambda fragment: (fragment.start_frame, fragment.fragment_index)):
                    _append_fragment_to_slot(current, fragment)
                merged_slots.append(current)
                idx += 2
                continue
        merged_slots.append(current)
        idx += 1
    return merged_slots


def _copy_track_fragment(
    fragment: TrackFragment,
    *,
    track_id: int,
    identity_slot: int | None,
) -> Track:
    """Materialize a single fragment as an output track when we choose not to merge it.

    当选择不合并片段时，将单个片段实体化为一条输出轨迹。
    """
    base = copy.deepcopy(fragment.source_track)
    start_frame = fragment.start_frame
    end_frame = fragment.end_frame

    base.track_id = track_id
    base.identity_slot = identity_slot
    base.trajectory = [
        copy.deepcopy(obs)
        for obs in base.trajectory
        if start_frame <= obs.frame_idx <= end_frame
    ]
    base.embedding_records = [
        copy.deepcopy(rec)
        for rec in base.embedding_records
        if start_frame <= rec.frame_idx <= end_frame
    ]
    if base.embedding_records:
        base.embedding_history = [rec.vector.astype(np.float32) for rec in base.embedding_records]
    base.feature_history = _feature_items_between(base, start_frame, end_frame)
    base.interpolated_frames = {
        frame_idx for frame_idx in base.interpolated_frames if start_frame <= frame_idx <= end_frame
    }
    base.fragment_breakpoints = [
        frame_idx for frame_idx in base.fragment_breakpoints if start_frame <= frame_idx <= end_frame
    ]
    base.reid_state.short_term_embeddings = [
        emb.astype(np.float32) for emb in base.embedding_history[-6:]
    ]
    base.reid_state.temporal_token = fragment.temporal_token
    base.reid_state.spatial_token = fragment.spatial_token
    base.prototype_embedding = fragment.prototype_embedding
    base.prototype_updates = len(base.embedding_history)
    latest = base.latest_observation()
    if latest is not None:
        base.bbox = latest.bbox
        base.center = latest.center
        base.last_frame = latest.frame_idx
        base.conf = latest.conf
    base.stats_cache.clear()
    return base


def global_reassign_ids(
    tracks: list[Track],
    *,
    max_link_gap: int = 20,
    merge_threshold: float = 1.15,
    appearance_threshold: float = 0.60,
    shape_threshold: float = 0.55,
    spatial_threshold: float = 0.55,
    motion_threshold: float = 0.85,
    fragment_min_len: int = 3,
    fragment_max_internal_gap: int = 3,
    offline_window: int = 24,
    max_identities: int | None = None,
    merge_fragments: bool = True,
    force_assign_when_full: bool = False,
) -> list[Track]:
    """Reassign fragment tracks into stable identity slots with windowed offline matching.

    通过窗口化离线匹配，将片段轨迹重新分配到稳定身份槽位。
    """
    if len(tracks) < 2:
        return tracks

    fragments: list[TrackFragment] = []
    for track in tracks:
        fragments.extend(
            build_fragments_from_track(
                track,
                min_fragment_len=fragment_min_len,
                max_internal_gap=fragment_max_internal_gap,
            )
        )
    if not fragments:
        return tracks

    fragments = sorted(fragments, key=lambda frag: (frag.start_frame, frag.parent_track_id, frag.fragment_index))
    max_slots = max_identities or len(fragments)
    slots: list[IdentitySlotState] = []
    fragment_outputs: list[Track] = []
    preserved_tracks: list[Track] = []
    next_fragment_track_id = 0
    window_size = max(int(offline_window), 1)
    grouped_fragments: list[list[TrackFragment]] = []
    current_group: list[TrackFragment] = []
    current_start: int | None = None
    for fragment in fragments:
        fragment_window = (fragment.start_frame // window_size) * window_size
        if current_start is None or fragment_window != current_start:
            if current_group:
                grouped_fragments.append(current_group)
            current_group = [fragment]
            current_start = fragment_window
        else:
            current_group.append(fragment)
    if current_group:
        grouped_fragments.append(current_group)

    for group in grouped_fragments:
        slot_candidates = [
            slot
            for slot in slots
            if any(fragment.start_frame > slot.last_end_frame for fragment in group)
        ]
        assignment_map: dict[int, int] = {}
        terms_map: dict[tuple[int, int], dict[str, float]] = {}

        if slot_candidates and linear_sum_assignment is not None:
            cost_matrix = np.full((len(slot_candidates), len(group)), fill_value=1e6, dtype=np.float32)
            for slot_idx, slot in enumerate(slot_candidates):
                for frag_idx, fragment in enumerate(group):
                    cost, terms = _slot_cost(fragment, slot, max_link_gap=max_link_gap)
                    cost_matrix[slot_idx, frag_idx] = cost
                    terms_map[(slot_idx, frag_idx)] = terms
            row_idx, col_idx = linear_sum_assignment(cost_matrix)
            assignment_map = {int(col): int(row) for row, col in zip(row_idx.tolist(), col_idx.tolist())}

        for fragment_idx, fragment in enumerate(group):
            best_slot = None
            best_cost = float("inf")
            best_terms: dict[str, float] | None = None
            for slot in slots:
                cost, terms = _slot_cost(fragment, slot, max_link_gap=max_link_gap)
                if cost < best_cost:
                    best_cost = cost
                    best_terms = terms
                    best_slot = slot

            if fragment_idx in assignment_map:
                assigned_slot = slot_candidates[assignment_map[fragment_idx]]
                assigned_cost, assigned_terms = _slot_cost(fragment, assigned_slot, max_link_gap=max_link_gap)
                if _allow_slot_merge(
                    assigned_cost,
                    assigned_terms,
                    merge_threshold=merge_threshold,
                    appearance_threshold=appearance_threshold,
                    shape_threshold=shape_threshold,
                    spatial_threshold=spatial_threshold,
                    motion_threshold=motion_threshold,
                ):
                    _append_fragment_to_slot(assigned_slot, fragment)
                    if not merge_fragments:
                        fragment_outputs.append(
                            _copy_track_fragment(
                                fragment,
                                track_id=next_fragment_track_id,
                                identity_slot=assigned_slot.slot_id,
                            )
                        )
                        next_fragment_track_id += 1
                    continue

            if len(slots) < max_slots:
                slot = IdentitySlotState(
                    slot_id=len(slots),
                    fragments=[fragment],
                    prototype_embedding=fragment.prototype_embedding,
                    temporal_token=fragment.temporal_token,
                    spatial_token=fragment.spatial_token,
                    mean_shape=fragment.mean_shape,
                    last_end_frame=fragment.end_frame,
                    last_center=fragment.end_center,
                )
                slots.append(slot)
                if not merge_fragments:
                    fragment_outputs.append(
                        _copy_track_fragment(
                            fragment,
                            track_id=next_fragment_track_id,
                            identity_slot=slot.slot_id,
                        )
                    )
                    next_fragment_track_id += 1
                continue

            if best_slot is not None and _allow_slot_merge(
                best_cost,
                best_terms,
                merge_threshold=merge_threshold,
                appearance_threshold=appearance_threshold,
                shape_threshold=shape_threshold,
                spatial_threshold=spatial_threshold,
                motion_threshold=motion_threshold,
            ):
                _append_fragment_to_slot(best_slot, fragment)
                if not merge_fragments:
                    fragment_outputs.append(
                        _copy_track_fragment(
                            fragment,
                            track_id=next_fragment_track_id,
                            identity_slot=best_slot.slot_id,
                        )
                    )
                    next_fragment_track_id += 1
                continue

            if best_slot is not None and (force_assign_when_full or max_identities is not None) and _should_soft_assign_slot(
                fragment,
                best_slot,
                best_terms,
                appearance_threshold=appearance_threshold,
                shape_threshold=shape_threshold,
            ):
                _append_fragment_to_slot(best_slot, fragment)
                if not merge_fragments:
                    fragment_outputs.append(
                        _copy_track_fragment(
                            fragment,
                            track_id=next_fragment_track_id,
                            identity_slot=best_slot.slot_id,
                        )
                    )
                    next_fragment_track_id += 1
                continue

            if not merge_fragments:
                assigned_slot_id = fragment.source_track.identity_slot
                if _should_soft_assign_slot(
                    fragment,
                    best_slot,
                    best_terms,
                    appearance_threshold=appearance_threshold,
                    shape_threshold=shape_threshold,
                ):
                    assigned_slot_id = best_slot.slot_id
                fragment_outputs.append(
                    _copy_track_fragment(
                        fragment,
                        track_id=next_fragment_track_id,
                        identity_slot=assigned_slot_id,
                    )
                )
                next_fragment_track_id += 1
                continue

            assigned_slot_id = fragment.source_track.identity_slot
            if _should_soft_assign_slot(
                fragment,
                best_slot,
                best_terms,
                appearance_threshold=appearance_threshold,
                shape_threshold=shape_threshold,
            ):
                assigned_slot_id = best_slot.slot_id
            preserved_tracks.append(
                _copy_track_fragment(
                    fragment,
                    track_id=next_fragment_track_id,
                    identity_slot=assigned_slot_id,
                )
            )
            next_fragment_track_id += 1

    if not merge_fragments:
        fragment_outputs.sort(key=lambda track: (track.start_frame, track.track_id))
        return fragment_outputs

    if max_identities is None:
        slots = _consolidate_adjacent_slots_once(
            slots,
            max_link_gap=max_link_gap,
            merge_threshold=merge_threshold,
            appearance_threshold=appearance_threshold,
            shape_threshold=shape_threshold,
            spatial_threshold=spatial_threshold,
            motion_threshold=motion_threshold,
        )

    corrected_tracks = [_merge_slot_fragments(slot) for slot in slots]
    corrected_tracks.extend(preserved_tracks)
    corrected_tracks.sort(key=lambda track: (track.start_frame, track.track_id))
    if preserved_tracks:
        for new_track_id, track in enumerate(corrected_tracks):
            track.track_id = new_track_id
    return corrected_tracks
