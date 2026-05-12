from __future__ import annotations

import copy

import numpy as np

from core.states import TrackState
from core.structures import AssociationResult, Detection, LatentSlotState, Track
from motion.kalman_filter import SimpleKalmanFilter
from motion.kinematics import velocity_from_history


class TrackBuilder:
    """Own the online track lifecycle, including slot reuse and cautious ReID memory updates.

    管理在线轨迹生命周期，包括槽位复用和谨慎的 ReID 记忆更新。
    """
    def __init__(
        self,
        confirm_hits: int = 3,
        max_missed: int = 15,
        remove_tentative_after: int = 1,
        max_tracks: int | None = None,
        reid_update_quality_thres: float = 0.55,
        suspicious_appearance_thres: float = 0.45,
        suspicious_hits: int = 2,
        reid_freeze_frames: int = 8,
        *,
        use_identity_slots: bool = True,
        recall_mode: bool = False,
        recovery_confirm_hits: int = 2,
        short_term_window: int = 6,
        long_term_momentum: float = 0.98,
        quarantine_min_quality: float = 0.45,
        trajectory_temporal=None,
        enable_latent_slot_reconnect: bool = True,
        latent_slot_max_age: int = 2400,
        latent_motion_gate: float = 36.0,
        latent_shape_ratio_tol: float = 0.90,
        latent_reconnect_min_reliability: float = 0.25,
        enable_weak_match_motion_blend: bool = True,
        weak_match_min_hits: int = 4,
        weak_match_score_thres: float = 0.40,
        weak_match_quality_thres: float = 0.55,
        weak_match_switch_risk_thres: float = 0.35,
        weak_match_position_alpha: float = 0.35,
    ) -> None:
        self.confirm_hits = confirm_hits
        self.max_missed = max_missed
        self.remove_tentative_after = remove_tentative_after
        self.max_tracks = max_tracks
        self.use_identity_slots = use_identity_slots
        self.recall_mode = recall_mode
        self.recovery_confirm_hits = max(1, recovery_confirm_hits)

        self.reid_update_quality_thres = reid_update_quality_thres
        self.suspicious_appearance_thres = suspicious_appearance_thres
        self.suspicious_hits = suspicious_hits
        self.reid_freeze_frames = reid_freeze_frames
        self.short_term_window = short_term_window
        self.long_term_momentum = long_term_momentum
        self.quarantine_min_quality = quarantine_min_quality
        self.trajectory_temporal = trajectory_temporal
        self.enable_latent_slot_reconnect = bool(enable_latent_slot_reconnect)
        self.latent_slot_max_age = (
            max(1, int(latent_slot_max_age))
            if self.enable_latent_slot_reconnect
            else 0
        )
        self.latent_motion_gate = float(latent_motion_gate)
        self.latent_shape_ratio_tol = float(latent_shape_ratio_tol)
        self.latent_reconnect_min_reliability = float(latent_reconnect_min_reliability)
        self.enable_weak_match_motion_blend = bool(enable_weak_match_motion_blend)
        self.weak_match_min_hits = max(1, int(weak_match_min_hits))
        self.weak_match_score_thres = float(weak_match_score_thres)
        self.weak_match_quality_thres = float(weak_match_quality_thres)
        self.weak_match_switch_risk_thres = float(weak_match_switch_risk_thres)
        self.weak_match_position_alpha = float(np.clip(weak_match_position_alpha, 0.05, 1.0))

        self.tracks: list[Track] = []
        self.finished_tracks: list[Track] = []
        self.latent_slots: dict[int, LatentSlotState] = {}

        self.next_track_id: int = 0
        self.available_track_ids: set[int] | None = (
            set(range(max_tracks)) if max_tracks is not None else None
        )

        self.kf = SimpleKalmanFilter()

    # ----------------------------
    # ID allocation
    # ----------------------------
    def _allocate_track_id(self, preferred_track_id: int | None = None) -> int | None:
        if self.max_tracks is None:
            if preferred_track_id is not None and preferred_track_id >= self.next_track_id:
                self.next_track_id = preferred_track_id + 1
                return preferred_track_id
            track_id = self.next_track_id
            self.next_track_id += 1
            return track_id

        if not self.available_track_ids:
            return None

        if preferred_track_id is not None and preferred_track_id in self.available_track_ids:
            self.available_track_ids.remove(preferred_track_id)
            return preferred_track_id

        track_id = min(self.available_track_ids)
        self.available_track_ids.remove(track_id)
        return track_id

    def _release_track_id(self, track_id: int) -> None:
        if self.available_track_ids is None:
            return
        still_used = any(
            track.track_id == track_id
            for track in self.tracks
            if track.state != TrackState.REMOVED
        )
        if not still_used:
            self.available_track_ids.add(track_id)

    # ----------------------------
    # Prediction
    # ----------------------------
    def predict(self) -> None:
        """Advance each active track one step with the Kalman model before frame association.

        在帧关联前使用卡尔曼模型将每条活跃轨迹向前预测一步。
        """
        for track in self.tracks:
            if track.kf_mean is not None and track.kf_cov is not None:
                track.kf_mean, track.kf_cov = self.kf.predict(
                    track.kf_mean, track.kf_cov
                )
                track.predicted_center = (
                    float(track.kf_mean[0]),
                    float(track.kf_mean[1]),
                )
            else:
                track.predicted_center = track.center
        if self.enable_latent_slot_reconnect:
            self._predict_latent_slots()

    def _predict_latent_slots(self) -> None:
        """Advance latent slot predictions so long-gap reconnect logic has a current position prior.

        推进潜在槽位预测，为长间隔重连逻辑提供当前位置先验。
        """
        stale_slots: list[int] = []
        for slot_id, latent in self.latent_slots.items():
            latent.predicted_center = (
                float(latent.predicted_center[0] + latent.velocity[0]),
                float(latent.predicted_center[1] + latent.velocity[1]),
            )
            latent.frames_since_seen += 1
            if latent.frames_since_seen > self.latent_slot_max_age:
                stale_slots.append(slot_id)
        for slot_id in stale_slots:
            self.latent_slots.pop(slot_id, None)

    # ----------------------------
    # ReID helpers
    # ----------------------------
    @staticmethod
    def _cosine_distance_np(
        a: np.ndarray | None,
        b: np.ndarray | None,
    ) -> float | None:
        if a is None or b is None:
            return None
        a = a.astype(np.float32)
        b = b.astype(np.float32)
        na = np.linalg.norm(a) + 1e-8
        nb = np.linalg.norm(b) + 1e-8
        return float(1.0 - float(a @ b) / float(na * nb))

    def _appearance_distance(self, track: Track, det: Detection) -> float | None:
        """Measure how far a detection is from the track's current identity references.

        度量检测结果与轨迹当前身份参考之间的距离。
        """
        if det.embedding is None:
            return None

        refs = [
            ref
            for ref in (
                track.prototype_embedding,
                track.short_term_embedding(recent=self.short_term_window),
                track.mean_embedding(recent=10),
            )
            if ref is not None
        ]
        if not refs:
            return None
        return min(self._cosine_distance_np(ref, det.embedding) for ref in refs)

    def _refresh_temporal_state(self, track: Track) -> None:
        """Refresh the cached temporal token from TT when possible, otherwise from trusted history.

        优先用 TT 刷新缓存时间 token，否则从可信历史中生成。
        """
        token = None
        if self.trajectory_temporal is not None:
            token = self.trajectory_temporal.encode_track(track)
        if token is None:
            token = track.short_term_embedding(recent=self.short_term_window)
        if token is None:
            token = track.mean_feature("identity", recent=self.short_term_window, trusted_only=True)
        if token is None:
            token = track.prototype_embedding
        if token is not None:
            track.update_temporal_token(token)

    def _refresh_spatial_state(self, track: Track) -> None:
        """Refresh the cached spatial token using trusted spatial history before weaker fallbacks.

        优先使用可信空间历史刷新缓存空间 token，再考虑较弱的回退方案。
        """
        spatial = track.mean_feature("spatial", trusted_only=True)
        if spatial is None:
            spatial = track.reid_state.spatial_token
        if spatial is None:
            spatial = track.mean_feature("spatial")
        if spatial is not None:
            track.update_spatial_token(spatial)

    @staticmethod
    def _latest_bbox_size(track: Track) -> tuple[float, float]:
        bbox = track.bbox
        return max(float(bbox[2] - bbox[0]), 1.0), max(float(bbox[3] - bbox[1]), 1.0)

    def _remember_latent_slot(self, track: Track) -> None:
        """Persist a removed confirmed slot as a latent trajectory so long reappearances can reclaim it.

        将已移除的确认槽位保留为潜在轨迹，使长时间消失后的目标可以重新认领。
        """
        if (
            not self.enable_latent_slot_reconnect
            or self.latent_slot_max_age <= 0
            or not self.use_identity_slots
            or track.identity_slot is None
            or track.hits < self.confirm_hits
            or track.state == TrackState.TENTATIVE
        ):
            return
        vx, vy = velocity_from_history(track.xy_history(include_interpolated=False))
        self.latent_slots[int(track.identity_slot)] = LatentSlotState(
            slot_id=int(track.identity_slot),
            predicted_center=tuple(track.predicted_center if track.predicted_center != (0.0, 0.0) else track.center),
            velocity=(float(vx), float(vy)),
            bbox_size=self._latest_bbox_size(track),
            frames_since_seen=0,
            prototype_embedding=None if track.prototype_embedding is None else track.prototype_embedding.astype(np.float32).copy(),
            temporal_token=None if track.reid_state.temporal_token is None else track.reid_state.temporal_token.astype(np.float32).copy(),
            spatial_token=None if track.reid_state.spatial_token is None else track.reid_state.spatial_token.astype(np.float32).copy(),
            short_term_embeddings=[emb.astype(np.float32).copy() for emb in track.reid_state.short_term_embeddings[-self.short_term_window:]],
            memory_reliability=float(track.reid_state.memory_reliability),
            last_conf=float(track.conf),
            source_track_id=int(track.track_id),
        )

    def _forget_latent_slot(self, slot_id: int | None) -> None:
        if slot_id is None:
            return
        self.latent_slots.pop(int(slot_id), None)

    def latent_slot_candidates(self) -> list[LatentSlotState]:
        """Return currently valid latent slots that are not occupied by a live track.

        返回当前有效且未被活动轨迹占用的潜在槽位。
        """
        active_slots = {
            int(track.identity_slot)
            for track in self.tracks
            if track.identity_slot is not None and track.state != TrackState.REMOVED
        }
        return [
            latent
            for slot_id, latent in self.latent_slots.items()
            if slot_id not in active_slots and latent.frames_since_seen <= self.latent_slot_max_age
        ]

    def _latent_motion_gate_for(self, latent: LatentSlotState) -> float:
        return float(self.latent_motion_gate + min(float(latent.frames_since_seen) * 0.12, 48.0))

    def _latent_appearance_distance(self, latent: LatentSlotState, det: Detection) -> float:
        if latent.prototype_embedding is None or det.embedding is None:
            return 0.35
        return float(self._cosine_distance_np(latent.prototype_embedding, det.embedding) or 0.35)

    def _latent_temporal_distance(self, latent: LatentSlotState, det: Detection) -> float:
        if latent.temporal_token is None or det.embedding is None:
            return 0.35
        return float(self._cosine_distance_np(latent.temporal_token, det.embedding) or 0.35)

    def recover_latent_slot_matches(
        self,
        detections: list[Detection],
        assoc: AssociationResult,
        *,
        large_cost: float = 1e6,
    ) -> AssociationResult:
        """Assign unmatched detections back to long-missing slot predictions when geometry and memory agree.

        当几何和记忆一致时，将未匹配检测分配回长期缺失的槽位预测。
        """
        if (
            not self.enable_latent_slot_reconnect
            or not self.use_identity_slots
            or not assoc.unmatched_detections
        ):
            return assoc

        latent_slots = self.latent_slot_candidates()
        if not latent_slots:
            return assoc

        unmatched_det_indices = list(assoc.unmatched_detections)
        latent_matches: list[tuple[int, int]] = []
        used_detections: set[int] = set()
        used_slots: set[int] = set()

        scored_pairs: list[tuple[float, int, int]] = []
        for latent in latent_slots:
            if latent.memory_reliability < self.latent_reconnect_min_reliability:
                continue
            gate = self._latent_motion_gate_for(latent)
            for det_idx in unmatched_det_indices:
                det = detections[det_idx]
                distance = float(
                    np.hypot(
                        det.center[0] - latent.predicted_center[0],
                        det.center[1] - latent.predicted_center[1],
                    )
                )
                if distance > gate:
                    continue
                if not self._shape_ratio_ok(latent, det):
                    continue
                motion_cost = distance / max(gate, 1e-6)
                appearance_cost = self._latent_appearance_distance(latent, det)
                temporal_cost = self._latent_temporal_distance(latent, det)
                total = 0.70 * motion_cost + 0.20 * appearance_cost + 0.10 * temporal_cost
                if det.is_track_supported:
                    total -= 0.05
                if det.is_rescued:
                    total += 0.03
                if total >= 0.95:
                    continue
                scored_pairs.append((float(total), int(latent.slot_id), int(det_idx)))

        for _, slot_id, det_idx in sorted(scored_pairs, key=lambda item: item[0]):
            if slot_id in used_slots or det_idx in used_detections:
                continue
            latent_matches.append((slot_id, det_idx))
            used_slots.add(slot_id)
            used_detections.add(det_idx)

        if not latent_matches:
            return assoc

        return AssociationResult(
            matches=assoc.matches,
            unmatched_tracks=assoc.unmatched_tracks,
            unmatched_detections=[idx for idx in assoc.unmatched_detections if idx not in used_detections],
            latent_matches=latent_matches,
            cost_matrix=assoc.cost_matrix,
            score_matrix=assoc.score_matrix,
            switch_risk_matrix=assoc.switch_risk_matrix,
        )

    def _shape_ratio_ok(self, latent: LatentSlotState, det: Detection) -> bool:
        width, height = latent.bbox_size
        det_width = max(float(det.bbox[2] - det.bbox[0]), 1.0)
        det_height = max(float(det.bbox[3] - det.bbox[1]), 1.0)
        width_ratio = abs(width - det_width) / max(width, det_width, 1.0)
        height_ratio = abs(height - det_height) / max(height, det_height, 1.0)
        return width_ratio <= self.latent_shape_ratio_tol and height_ratio <= self.latent_shape_ratio_tol

    def _seed_track_from_latent(self, track: Track, latent: LatentSlotState) -> None:
        """Restore slot-level memory when a long-missing identity reappears near its predicted path.

        当长期缺失身份在预测路径附近重新出现时，恢复槽位级记忆。
        """
        if latent.prototype_embedding is not None:
            track.prototype_embedding = latent.prototype_embedding.astype(np.float32).copy()
        if latent.temporal_token is not None:
            track.reid_state.temporal_token = latent.temporal_token.astype(np.float32).copy()
        if latent.spatial_token is not None:
            track.reid_state.spatial_token = latent.spatial_token.astype(np.float32).copy()
        if latent.short_term_embeddings:
            track.reid_state.short_term_embeddings = [
                emb.astype(np.float32).copy() for emb in latent.short_term_embeddings[-self.short_term_window:]
            ]
        track.reid_state.memory_reliability = max(track.reid_state.memory_reliability, float(latent.memory_reliability))
        track.reid_state.rescue_recovery_hits = self.recovery_confirm_hits

    @staticmethod
    def _trust_flags_for_detection(
        det: Detection,
        *,
        suspicious: bool,
    ) -> tuple[bool, bool]:
        """Decide whether the current match is safe to write into temporal and spatial memories.

        判断当前匹配是否足够安全，可写入时间和空间记忆。
        """
        safe_for_temporal = not suspicious and not det.is_merged_risk
        safe_for_spatial = not det.is_merged_risk
        return safe_for_temporal, safe_for_spatial

    def _record_association_feedback(
        self,
        track: Track,
        *,
        match_score: float | None,
        switch_risk: float | None,
    ) -> None:
        """Store association diagnostics so later debugging can inspect confidence and switch trends.

        保存关联诊断信息，便于后续调试检查置信度和切换趋势。
        """
        if match_score is not None:
            track.reid_state.match_confidence_history.append(float(match_score))
            track.reid_state.match_confidence_history = track.reid_state.match_confidence_history[-32:]
        if switch_risk is not None:
            track.reid_state.switch_risk_history.append(float(switch_risk))
            track.reid_state.switch_risk_history = track.reid_state.switch_risk_history[-32:]

    def _is_suspicious_match(
        self,
        track: Track,
        det: Detection,
        *,
        appearance_cost: float | None,
        match_score: float | None,
        switch_risk: float | None,
    ) -> bool:
        """Flag ambiguous matches that should preserve geometry but avoid polluting identity memory.

        标记有歧义的匹配，保留几何更新但避免污染身份记忆。
        """
        if track.hits < self.confirm_hits:
            return False
        if det.reid_quality < 0.40:
            return False
        if switch_risk is not None and switch_risk > 0.65:
            return True
        if match_score is not None and match_score < 0.20:
            return True
        if appearance_cost is not None and appearance_cost > self.suspicious_appearance_thres:
            return True
        return False

    def _should_update_reid(
        self,
        track: Track,
        det: Detection,
        frame_idx: int,
        *,
        suspicious: bool = False,
        match_score: float | None = None,
    ) -> bool:
        """Gate long-term prototype updates so only stable, high-quality matches can change identity memory.

        门控长期原型更新，确保只有稳定且高质量的匹配能改变身份记忆。
        """
        if det.embedding is None:
            return False
        if det.reid_quality < self.reid_update_quality_thres:
            return False
        if not track.can_update_reid(frame_idx):
            return False
        if det.is_crowded or det.is_merged_risk:
            return False
        if suspicious:
            return False
        if det.is_rescued and track.reid_state.rescue_recovery_hits < self.recovery_confirm_hits:
            return False
        if match_score is not None and match_score < 0.40:
            return False
        return True

    def _mark_suspicious(
        self,
        track: Track,
        frame_idx: int,
        *,
        appearance_cost: float | None,
        switch_risk: float | None = None,
    ) -> None:
        """Accumulate suspicion evidence and freeze ReID if ambiguity persists across frames.

        累积可疑证据，并在歧义跨帧持续时冻结 ReID 更新。
        """
        if appearance_cost is not None:
            track.appearance_cost_history.append(float(appearance_cost))
        if switch_risk is not None:
            track.reid_state.switch_risk_history.append(float(switch_risk))

        track.switch_suspect_count += 1
        track.suspicious_frames.append(frame_idx)

        if track.switch_suspect_count >= self.suspicious_hits:
            track.freeze_reid(frame_idx + self.reid_freeze_frames)
            if (
                not track.fragment_breakpoints
                or track.fragment_breakpoints[-1] != frame_idx
            ):
                track.fragment_breakpoints.append(frame_idx)

    def _relax_suspicion(self, track: Track) -> None:
        track.switch_suspect_count = max(0, track.switch_suspect_count - 1)

    @staticmethod
    def _bbox_with_center(
        bbox: tuple[float, float, float, float],
        center: tuple[float, float],
    ) -> tuple[float, float, float, float]:
        width = max(float(bbox[2] - bbox[0]), 1.0)
        height = max(float(bbox[3] - bbox[1]), 1.0)
        half_w = 0.5 * width
        half_h = 0.5 * height
        return (
            float(center[0] - half_w),
            float(center[1] - half_h),
            float(center[0] + half_w),
            float(center[1] + half_h),
        )

    def _use_weak_motion_blend(
        self,
        track: Track,
        det: Detection,
        *,
        suspicious: bool,
        match_score: float | None,
        switch_risk: float | None,
    ) -> bool:
        """Keep stable slots closer to prediction when only weak evidence is available.

        当只有弱证据可用时，让稳定槽位更贴近预测位置。
        """
        if not self.enable_weak_match_motion_blend:
            return False
        if track.state not in {TrackState.CONFIRMED, TrackState.LOST}:
            return False
        if track.hits < self.weak_match_min_hits:
            return False
        if track.predicted_center == (0.0, 0.0):
            return False
        if suspicious:
            return True
        if det.is_rescued:
            return True
        if det.reid_quality < self.weak_match_quality_thres:
            return True
        if match_score is not None and match_score < self.weak_match_score_thres:
            return True
        if switch_risk is not None and switch_risk > self.weak_match_switch_risk_thres:
            return True
        return False

    def _motion_blend_bbox(
        self,
        track: Track,
        det: Detection,
        *,
        predicted_center: tuple[float, float],
    ) -> tuple[tuple[float, float, float, float], tuple[float, float]]:
        """Blend weak-evidence detections toward the predicted trajectory instead of snapping to them.

        将弱证据检测向预测轨迹方向融合，而不是直接跳到检测位置。
        """
        alpha = self.weak_match_position_alpha * (0.88 ** max(track.reid_state.weak_match_streak, 0))
        alpha = float(np.clip(alpha, 0.12, self.weak_match_position_alpha))
        blended_center = (
            float((1.0 - alpha) * predicted_center[0] + alpha * det.center[0]),
            float((1.0 - alpha) * predicted_center[1] + alpha * det.center[1]),
        )
        return self._bbox_with_center(det.bbox, blended_center), blended_center

    def _update_reid_memory(
        self,
        track: Track,
        det: Detection,
        *,
        frame_idx: int,
        suspicious: bool,
        match_score: float | None,
    ) -> None:
        """Update short-term, quarantine, and long-term memories with rescue-aware trust rules.

        使用感知救援来源的信任规则更新短期、隔离和长期记忆。
        """
        if det.embedding is None:
            if det.is_rescued and not suspicious:
                track.reid_state.rescue_recovery_hits += 1
            elif suspicious:
                track.reid_state.rescue_recovery_hits = 0
            track.update_memory_reliability(matched=True, quality=det.reid_quality, suspicious=suspicious)
            self._refresh_spatial_state(track)
            self._refresh_temporal_state(track)
            return

        if det.is_rescued and not suspicious:
            track.push_short_term_embedding(det.embedding, maxlen=self.short_term_window)
            track.reid_state.rescue_recovery_hits += 1
        elif suspicious or det.reid_quality < self.quarantine_min_quality or det.is_merged_risk:
            track.quarantine_embedding(det.embedding, maxlen=self.short_term_window)
            if suspicious:
                track.reid_state.rescue_recovery_hits = 0
        else:
            track.push_short_term_embedding(det.embedding, maxlen=self.short_term_window)
            if not det.is_rescued:
                track.reid_state.rescue_recovery_hits = self.recovery_confirm_hits

        if self._should_update_reid(
            track,
            det,
            frame_idx,
            suspicious=suspicious,
            match_score=match_score,
        ):
            track.update_prototype(
                det.embedding,
                alpha=self.long_term_momentum,
                frame_idx=frame_idx,
            )

        track.update_memory_reliability(
            matched=True,
            quality=det.reid_quality,
            suspicious=suspicious,
        )
        self._refresh_spatial_state(track)
        self._refresh_temporal_state(track)

    # ----------------------------
    # Track creation
    # ----------------------------
    def _create_track(self, frame_idx: int, det: Detection) -> Track | None:
        """Spawn a new online track, optionally honoring a preferred identity slot from rescue logic.

        创建新的在线轨迹，并可优先采用救援逻辑提供的身份槽位。
        """
        preferred_track_id = None
        if self.use_identity_slots and det.rescue_slot_id is not None:
            preferred_track_id = int(det.rescue_slot_id)
        latent = None
        if preferred_track_id is not None:
            latent = self.latent_slots.get(int(preferred_track_id))

        track_id = self._allocate_track_id(preferred_track_id=preferred_track_id)
        if track_id is None:
            return None

        track = Track(
            track_id=track_id,
            identity_slot=track_id if self.use_identity_slots else None,
            bbox=det.bbox,
            center=det.center,
            predicted_center=det.center,
            state=TrackState.TENTATIVE,
            cls_id=det.cls_id,
            conf=det.conf,
            last_frame=frame_idx,
        )

        if latent is not None:
            self._seed_track_from_latent(track, latent)

        if det.raw_tid is not None:
            track.raw_tid_history.append(det.raw_tid)

        track.add_embedding(
            det.embedding,
            frame_idx=frame_idx,
            source=det.embedding_source,
        )
        safe_for_temporal, safe_for_spatial = self._trust_flags_for_detection(det, suspicious=False)
        track.add_feature_snapshot(
            det,
            safe_for_temporal=safe_for_temporal,
            safe_for_spatial=safe_for_spatial,
        )
        track.append_observation(
            frame_idx,
            det.bbox,
            det.conf,
            state=track.state,
            interpolated=False,
        )

        if det.embedding is not None:
            track.push_short_term_embedding(det.embedding, maxlen=self.short_term_window)
        if det.is_rescued:
            track.reid_state.rescue_recovery_hits = 1
        if det.embedding is not None and det.reid_quality >= self.reid_update_quality_thres and not det.is_rescued:
            proto_alpha = (
                self.long_term_momentum
                if latent is not None and track.prototype_embedding is not None
                else 0.0
            )
            track.update_prototype(
                det.embedding,
                alpha=proto_alpha,
                frame_idx=frame_idx,
            )

        track.kf_mean, track.kf_cov = self.kf.initiate(*det.center)
        track.update_memory_reliability(matched=True, quality=det.reid_quality, suspicious=False)
        self._refresh_spatial_state(track)
        self._refresh_temporal_state(track)
        return track

    # ----------------------------
    # Main update
    # ----------------------------
    def update(
        self,
        frame_idx: int,
        detections: list[Detection],
        assoc: AssociationResult,
    ) -> list[Track]:
        """Apply one frame of association results to the online tracker state machine.

        将单帧关联结果应用到在线跟踪器状态机。
        """
        removed_tracks: list[Track] = []

        for track_idx, det_idx in assoc.matches:
            track = self.tracks[track_idx]
            det = detections[det_idx]
            match_score = None
            switch_risk = None
            if assoc.score_matrix is not None:
                match_score = float(assoc.score_matrix[track_idx, det_idx])
            if assoc.switch_risk_matrix is not None:
                switch_risk = float(assoc.switch_risk_matrix[track_idx, det_idx])

            appearance_cost = self._appearance_distance(track, det)
            suspicious = self._is_suspicious_match(
                track,
                det,
                appearance_cost=appearance_cost,
                match_score=match_score,
                switch_risk=switch_risk,
            )

            if suspicious:
                self._mark_suspicious(
                    track,
                    frame_idx,
                    appearance_cost=appearance_cost,
                    switch_risk=switch_risk,
                )
            else:
                self._relax_suspicion(track)

            previous_predicted = track.predicted_center

            if track.state == TrackState.LOST:
                track.restore()

            weak_motion_blend = self._use_weak_motion_blend(
                track,
                det,
                suspicious=suspicious,
                match_score=match_score,
                switch_risk=switch_risk,
            )
            if weak_motion_blend:
                update_bbox, measurement_center = self._motion_blend_bbox(
                    track,
                    det,
                    predicted_center=previous_predicted if previous_predicted != (0.0, 0.0) else track.center,
                )
                track.reid_state.weak_match_streak += 1
            else:
                update_bbox = det.bbox
                measurement_center = det.center
                track.reid_state.weak_match_streak = 0

            track.update_position(
                frame_idx,
                update_bbox,
                det.conf,
                predicted_center=previous_predicted,
            )
            track.cls_id = det.cls_id

            if det.raw_tid is not None:
                track.raw_tid_history.append(det.raw_tid)

            track.add_embedding(
                det.embedding,
                frame_idx=frame_idx,
                source=det.embedding_source,
            )
            safe_for_temporal, safe_for_spatial = self._trust_flags_for_detection(
                det,
                suspicious=suspicious,
            )
            track.add_feature_snapshot(
                det,
                safe_for_temporal=safe_for_temporal,
                safe_for_spatial=safe_for_spatial,
            )
            self._record_association_feedback(track, match_score=match_score, switch_risk=switch_risk)

            if track.kf_mean is None or track.kf_cov is None:
                track.kf_mean, track.kf_cov = self.kf.initiate(*measurement_center)
            else:
                track.kf_mean, track.kf_cov = self.kf.update(
                    track.kf_mean,
                    track.kf_cov,
                    measurement_center,
                )

            self._update_reid_memory(
                track,
                det,
                frame_idx=frame_idx,
                suspicious=suspicious,
                match_score=match_score,
            )
            self._forget_latent_slot(track.identity_slot)

            if track.hits >= self.confirm_hits:
                track.state = TrackState.CONFIRMED
                if track.trajectory:
                    track.trajectory[-1].state = track.state.value

        for track_idx in assoc.unmatched_tracks:
            track = self.tracks[track_idx]
            track.mark_missed()
            track.reid_state.weak_match_streak = 0
            track.update_memory_reliability(matched=False, quality=0.0, suspicious=False)
            self._refresh_spatial_state(track)
            self._refresh_temporal_state(track)

            if track.state == TrackState.TENTATIVE:
                if track.missed >= self.remove_tentative_after:
                    track.state = TrackState.REMOVED
                    removed_tracks.append(track)
                else:
                    track.state = TrackState.TENTATIVE
                continue

            allowed_missed = self.max_missed + 4 if self.recall_mode else self.max_missed
            if track.missed > allowed_missed:
                self._remember_latent_slot(track)
                track.state = TrackState.REMOVED
                removed_tracks.append(track)
            else:
                track.state = TrackState.LOST

        for track in removed_tracks:
            self._release_track_id(track.track_id)

        latent_matched_detections: set[int] = set()
        for slot_id, det_idx in assoc.latent_matches:
            if det_idx in latent_matched_detections:
                continue
            if det_idx < 0 or det_idx >= len(detections):
                continue
            if int(slot_id) not in self.latent_slots:
                continue
            detections[det_idx].rescue_slot_id = int(slot_id)
            track = self._create_track(frame_idx, detections[det_idx])
            if track is None:
                continue
            track.state = TrackState.CONFIRMED
            if track.trajectory:
                track.trajectory[-1].state = track.state.value
            track.hits = max(track.hits, self.confirm_hits)
            self.tracks.append(track)
            self._forget_latent_slot(track.identity_slot)
            latent_matched_detections.add(det_idx)

        for det_idx in sorted(
            assoc.unmatched_detections,
            key=lambda idx: detections[idx].conf,
            reverse=True,
        ):
            if det_idx in latent_matched_detections:
                continue
            track = self._create_track(frame_idx, detections[det_idx])
            if track is not None:
                self.tracks.append(track)

        if removed_tracks:
            self.finished_tracks.extend(removed_tracks)

        self.tracks = [
            track for track in self.tracks
            if track.state != TrackState.REMOVED
        ]
        return self.tracks

    # ----------------------------
    # Export helpers
    # ----------------------------
    def _rebuild_track_memory(self, track: Track) -> None:
        """Recompute cached memory summaries after fragment export or offline merging.

        在片段导出或离线合并后重新计算缓存的记忆摘要。
        """
        if track.embedding_history:
            proto = np.mean(np.stack(track.embedding_history, axis=0), axis=0).astype(np.float32)
            proto /= np.linalg.norm(proto) + 1e-8
            track.prototype_embedding = proto
            track.prototype_updates = len(track.embedding_history)
            track.reid_state.short_term_embeddings = [emb.astype(np.float32) for emb in track.embedding_history[-self.short_term_window:]]
        track.reid_state.quarantine_embeddings = track.reid_state.quarantine_embeddings[-self.short_term_window:]
        self._refresh_spatial_state(track)
        self._refresh_temporal_state(track)

    def _merge_fragments_by_track_id(self, tracks: list[Track]) -> list[Track]:
        """Stitch fragments that reused the same online slot back into one export track.

        将复用同一在线槽位的片段拼接回一条导出轨迹。
        """
        grouped: dict[int, list[Track]] = {}
        for track in tracks:
            grouped.setdefault(track.track_id, []).append(track)

        merged_tracks: list[Track] = []
        for track_id, fragments in grouped.items():
            fragments = sorted(fragments, key=lambda item: item.start_frame)
            merged = copy.deepcopy(fragments[0])

            for fragment in fragments[1:]:
                merged.raw_tid_history.extend(fragment.raw_tid_history)
                merged.embedding_history.extend(fragment.embedding_history)
                merged.embedding_records.extend(copy.deepcopy(fragment.embedding_records))
                merged.feature_history.extend(copy.deepcopy(fragment.feature_history))
                merged.interpolated_frames |= set(fragment.interpolated_frames)
                merged.reid_state.quarantine_embeddings.extend(copy.deepcopy(fragment.reid_state.quarantine_embeddings))
                merged.identity_slot = merged.identity_slot if merged.identity_slot is not None else fragment.identity_slot

                merged.fragment_breakpoints = sorted(
                    set(getattr(merged, "fragment_breakpoints", []))
                    | set(getattr(fragment, "fragment_breakpoints", []))
                )

                by_frame = {obs.frame_idx: copy.deepcopy(obs) for obs in merged.trajectory}
                for obs in fragment.trajectory:
                    existing = by_frame.get(obs.frame_idx)
                    if existing is None or obs.conf > existing.conf:
                        by_frame[obs.frame_idx] = copy.deepcopy(obs)

                merged.trajectory = sorted(
                    by_frame.values(), key=lambda obs: obs.frame_idx)

                latest = fragment.latest_observation()
                if latest is not None and latest.frame_idx >= merged.last_frame:
                    merged.bbox = latest.bbox
                    merged.center = latest.center
                    merged.predicted_center = fragment.predicted_center
                    merged.conf = latest.conf
                    merged.last_frame = latest.frame_idx
                    merged.state = fragment.state

                merged.hits += fragment.hits
                merged.age += fragment.age
                merged.missed = fragment.missed
                merged.kf_mean = fragment.kf_mean
                merged.kf_cov = fragment.kf_cov
                merged.reid_state.memory_reliability = max(
                    merged.reid_state.memory_reliability,
                    fragment.reid_state.memory_reliability,
                )
                merged.stats_cache.clear()

            self._rebuild_track_memory(merged)
            merged_tracks.append(merged)

        merged_tracks.sort(key=lambda track: track.track_id)
        return merged_tracks

    def export_tracks(
        self,
        min_length: int = 1,
        *,
        merge_same_track_id: bool = True,
        reassign_track_ids: bool = False,
    ) -> list[Track]:
        """Export current and finished tracks, with optional fragment stitching and id reassignment.

        导出当前和已结束轨迹，并可执行片段拼接和 ID 重分配。
        """
        all_tracks = [copy.deepcopy(track) for track in (self.finished_tracks + self.tracks)]
        exported = (
            self._merge_fragments_by_track_id(all_tracks)
            if merge_same_track_id
            else sorted(all_tracks, key=lambda track: (track.start_frame, track.track_id))
        )
        exported = [track for track in exported if len(track.trajectory) >= min_length]
        if reassign_track_ids:
            for new_id, track in enumerate(exported):
                track.track_id = new_id
        exported.sort(key=lambda track: track.track_id)
        return exported
