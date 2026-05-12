from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .states import TrackState


@dataclass
class Detection:
    frame_idx: int
    bbox: tuple[float, float, float, float]
    conf: float
    cls_id: int = 0
    raw_tid: int | None = None
    crop: np.ndarray | None = None
    area: float = 0.0
    aspect: float = 0.0
    center: tuple[float, float] = (0.0, 0.0)
    frame_size: tuple[int, int] = (0, 0)
    blur_score: float | None = None
    embedding: np.ndarray | None = None
    identity_feature: np.ndarray | None = None
    spatial_feature: np.ndarray | None = None
    appearance_feature: np.ndarray | None = None
    shape_feature: np.ndarray | None = None
    quality_flags: list[str] = field(default_factory=list)
    is_border: bool = False
    duplicate_score: float | None = None
    
    reid_quality: float = 0.0
    embedding_source: str = "none"
    detector_source: str = "main"
    is_rescued: bool = False
    is_track_supported: bool = False
    support_track_id: int | None = None
    rescue_slot_id: int | None = None
    reid_quality_cap: float | None = None
    is_crowded: bool = False
    is_merged_risk: bool = False
    appearance_cost_cache: float | None = None
    context_feature: np.ndarray | None = None
    switch_risk_hint: float = 0.0
    association_score_cache: float | None = None


@dataclass
class FrameInfo:
    frame_idx: int
    timestamp_sec: float
    width: int
    height: int
    fps: float


@dataclass
class EmbeddingRecord:
    frame_idx: int
    track_id: int | None
    vector: np.ndarray
    source: str = "encoder"


@dataclass
class TrackObservation:
    frame_idx: int
    bbox: tuple[float, float, float, float]
    center: tuple[float, float]
    conf: float
    state: str
    interpolated: bool = False


@dataclass
class AssociationResult:
    matches: list[tuple[int, int]] = field(default_factory=list)
    unmatched_tracks: list[int] = field(default_factory=list)
    unmatched_detections: list[int] = field(default_factory=list)
    cost_matrix: np.ndarray | None = None
    score_matrix: np.ndarray | None = None
    switch_risk_matrix: np.ndarray | None = None
    latent_matches: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class LatentSlotState:
    slot_id: int
    predicted_center: tuple[float, float]
    velocity: tuple[float, float]
    bbox_size: tuple[float, float]
    frames_since_seen: int = 0
    prototype_embedding: np.ndarray | None = None
    temporal_token: np.ndarray | None = None
    spatial_token: np.ndarray | None = None
    short_term_embeddings: list[np.ndarray] = field(default_factory=list)
    memory_reliability: float = 0.0
    last_conf: float = 0.0
    source_track_id: int | None = None


@dataclass
class TrackReIDState:
    long_term_embedding: np.ndarray | None = None
    long_term_updates: int = 0
    short_term_embeddings: list[np.ndarray] = field(default_factory=list)
    quarantine_embeddings: list[np.ndarray] = field(default_factory=list)
    temporal_token: np.ndarray | None = None
    spatial_token: np.ndarray | None = None
    memory_reliability: float = 0.0
    last_good_frame: int = -1
    frozen_until: int = -1
    switch_suspect_count: int = 0
    appearance_cost_history: list[float] = field(default_factory=list)
    match_confidence_history: list[float] = field(default_factory=list)
    switch_risk_history: list[float] = field(default_factory=list)
    suspicious_frames: list[int] = field(default_factory=list)
    fragment_breakpoints: list[int] = field(default_factory=list)
    rescue_recovery_hits: int = 0
    weak_match_streak: int = 0

    def reset_diagnostics(self) -> None:
        self.appearance_cost_history = []
        self.match_confidence_history = []
        self.switch_risk_history = []
        self.suspicious_frames = []
        self.switch_suspect_count = 0
        self.rescue_recovery_hits = 0
        self.weak_match_streak = 0


@dataclass
class Track:
    track_id: int
    bbox: tuple[float, float, float, float]
    center: tuple[float, float]
    identity_slot: int | None = None
    predicted_center: tuple[float, float] = (0.0, 0.0)
    state: TrackState = TrackState.TENTATIVE
    hits: int = 1
    missed: int = 0
    age: int = 1
    cls_id: int = 0
    conf: float = 1.0
    raw_tid_history: list[int] = field(default_factory=list)
    embedding_history: list[np.ndarray] = field(default_factory=list)
    embedding_records: list[EmbeddingRecord] = field(default_factory=list)
    feature_history: list[dict[str, Any]] = field(default_factory=list)
    trajectory: list[TrackObservation] = field(default_factory=list)
    last_frame: int = -1
    interpolated_frames: set[int] = field(default_factory=set)
    stats_cache: dict[str, Any] = field(default_factory=dict)
    kf_mean: np.ndarray | None = None
    kf_cov: np.ndarray | None = None
    reid_state: TrackReIDState = field(default_factory=TrackReIDState)

    def append_observation(
        self,
        frame_idx: int,
        bbox: tuple[float, float, float, float],
        conf: float,
        state: TrackState | None = None,
        interpolated: bool = False,
    ) -> None:
        center = ((bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5)
        self.trajectory.append(
            TrackObservation(
                frame_idx=frame_idx,
                bbox=bbox,
                center=center,
                conf=conf,
                state=(state or self.state).value,
                interpolated=interpolated,
            )
        )
        self.last_frame = frame_idx
        if interpolated:
            self.interpolated_frames.add(frame_idx)

    def update_position(
        self,
        frame_idx: int,
        bbox: tuple[float, float, float, float],
        conf: float,
        predicted_center: tuple[float, float] | None = None,
        interpolated: bool = False,
    ) -> None:
        self.bbox = bbox
        self.center = ((bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5)
        if predicted_center is not None:
            self.predicted_center = predicted_center
        self.conf = conf
        self.append_observation(frame_idx, bbox, conf, state=self.state, interpolated=interpolated)
        self.hits += 1
        self.age += 1
        self.missed = 0
        self.stats_cache.clear()

    def mark_missed(self) -> None:
        self.missed += 1
        self.age += 1
        self.stats_cache.clear()

    def restore(self) -> None:
        self.missed = 0
        self.stats_cache.clear()

    def add_embedding(self,emb: np.ndarray | None, *, frame_idx: int = -1, source: str = "encoder",) -> None:
        if emb is not None:
            emb = emb.astype(np.float32)
            self.embedding_history.append(emb)
            self.embedding_records.append(
                EmbeddingRecord(
                    frame_idx=frame_idx,
                    track_id=self.track_id,
                    vector=emb,
                    source=source,
                )
            )

    def push_short_term_embedding(self, emb: np.ndarray | None, *, maxlen: int = 6) -> None:
        if emb is None:
            return
        emb = emb.astype(np.float32)
        self.reid_state.short_term_embeddings.append(emb)
        if maxlen > 0 and len(self.reid_state.short_term_embeddings) > maxlen:
            self.reid_state.short_term_embeddings = self.reid_state.short_term_embeddings[-maxlen:]

    def quarantine_embedding(self, emb: np.ndarray | None, *, maxlen: int = 6) -> None:
        if emb is None:
            return
        emb = emb.astype(np.float32)
        self.reid_state.quarantine_embeddings.append(emb)
        if maxlen > 0 and len(self.reid_state.quarantine_embeddings) > maxlen:
            self.reid_state.quarantine_embeddings = self.reid_state.quarantine_embeddings[-maxlen:]

    def add_feature_snapshot(
        self,
        det: Detection,
        *,
        safe_for_temporal: bool = True,
        safe_for_spatial: bool = True,
    ) -> None:
        self.feature_history.append(
            {
                "appearance": None if det.appearance_feature is None else det.appearance_feature.astype(np.float32),
                "identity": None if det.identity_feature is None else det.identity_feature.astype(np.float32),
                "spatial": None if det.spatial_feature is None else det.spatial_feature.astype(np.float32),
                "shape": None if det.shape_feature is None else det.shape_feature.astype(np.float32),   
                "conf": float(det.conf),
                "frame_idx": int(det.frame_idx),
                "area": float(det.area),
                "aspect": float(det.aspect),
                "reid_quality": float(det.reid_quality),
                "embedding_source": str(det.embedding_source),
                "detector_source": str(det.detector_source),
                "is_rescued": bool(det.is_rescued),
                "is_track_supported": bool(det.is_track_supported),
                "context": None if det.context_feature is None else det.context_feature.astype(np.float32),
                "is_crowded": bool(det.is_crowded),
                "is_merged_risk": bool(det.is_merged_risk),
                "switch_risk_hint": float(det.switch_risk_hint),
                "safe_for_temporal": bool(safe_for_temporal),
                "safe_for_spatial": bool(safe_for_spatial),
            }
        )
        self.stats_cache.clear()

    def mean_embedding(self, recent: int | None = None) -> np.ndarray | None:
        if not self.embedding_history:
            return None
        values = self.embedding_history[-recent:] if recent is not None else self.embedding_history
        return np.mean(np.stack(values, axis=0), axis=0)

    def short_term_embedding(self, recent: int | None = None) -> np.ndarray | None:
        values = self.reid_state.short_term_embeddings
        if not values:
            return None
        values = values[-recent:] if recent is not None else values
        return np.mean(np.stack(values, axis=0), axis=0)

    def recent_feature_items(
        self,
        recent: int | None = None,
        *,
        trusted_only: bool = False,
        trust_for: str = "temporal",
    ) -> list[dict[str, Any]]:
        items = self.feature_history[-recent:] if recent is not None else self.feature_history
        if not trusted_only:
            return items
        trust_key = "safe_for_spatial" if trust_for == "spatial" else "safe_for_temporal"
        return [entry for entry in items if bool(entry.get(trust_key, True))]

    def mean_feature(
        self,
        key: str,
        *,
        recent: int | None = None,
        trusted_only: bool = False,
    ) -> np.ndarray | None:
        trust_for = "spatial" if key == "spatial" else "temporal"
        values = [
            entry[key]
            for entry in self.recent_feature_items(recent=recent, trusted_only=trusted_only, trust_for=trust_for)
            if entry.get(key) is not None
        ]
        if not values:
            return None
        return np.mean(np.stack(values, axis=0), axis=0)

    def xy_history(self, include_interpolated: bool = True) -> list[tuple[int, float, float]]:
        points: list[tuple[int, float, float]] = []
        for obs in self.trajectory:
            if not include_interpolated and obs.interpolated:
                continue
            points.append((obs.frame_idx, obs.center[0], obs.center[1]))
        return points
    
    def can_update_reid(self, frame_idx: int) -> bool:
        return frame_idx > self.reid_state.frozen_until

    def freeze_reid(self, until_frame: int) -> None:
        self.reid_state.frozen_until = max(self.reid_state.frozen_until, until_frame)

    def update_memory_reliability(self, *, matched: bool, quality: float, suspicious: bool = False) -> None:
        quality = float(np.clip(quality, 0.0, 1.0))
        if suspicious:
            self.reid_state.memory_reliability *= 0.75
        elif matched:
            self.reid_state.memory_reliability = 0.85 * self.reid_state.memory_reliability + 0.15 * quality
        else:
            self.reid_state.memory_reliability *= 0.90
        self.reid_state.memory_reliability = float(np.clip(self.reid_state.memory_reliability, 0.0, 1.0))

    def update_temporal_token(self, token: np.ndarray | None) -> None:
        if token is None:
            return
        token = token.astype(np.float32)
        norm = np.linalg.norm(token) + 1e-8
        self.reid_state.temporal_token = (token / norm).astype(np.float32)

    def update_spatial_token(self, token: np.ndarray | None) -> None:
        if token is None:
            return
        token = token.astype(np.float32)
        norm = np.linalg.norm(token) + 1e-8
        self.reid_state.spatial_token = (token / norm).astype(np.float32)

    def update_prototype(
        self,
        emb: np.ndarray | None,
        *,
        alpha: float = 0.90,
        frame_idx: int = -1,
    ) -> None:
        if emb is None:
            return
        emb = emb.astype(np.float32)
        emb = emb / (np.linalg.norm(emb) + 1e-8)

        if self.reid_state.long_term_embedding is None:
            self.reid_state.long_term_embedding = emb.copy()
        else:
            self.reid_state.long_term_embedding = (
                alpha * self.reid_state.long_term_embedding + (1.0 - alpha) * emb
            ).astype(np.float32)
            self.reid_state.long_term_embedding = self.reid_state.long_term_embedding / (
                np.linalg.norm(self.reid_state.long_term_embedding) + 1e-8
            )

        self.reid_state.long_term_updates += 1
        self.reid_state.last_good_frame = frame_idx

    @property
    def start_frame(self) -> int:
        if not self.trajectory:
            return self.last_frame
        return min(obs.frame_idx for obs in self.trajectory)

    @property
    def end_frame(self) -> int:
        if not self.trajectory:
            return self.last_frame
        return max(obs.frame_idx for obs in self.trajectory)

    def latest_observation(self) -> TrackObservation | None:
        if not self.trajectory:
            return None
        return max(self.trajectory, key=lambda obs: obs.frame_idx)

    @property
    def prototype_embedding(self) -> np.ndarray | None:
        return self.reid_state.long_term_embedding

    @prototype_embedding.setter
    def prototype_embedding(self, value: np.ndarray | None) -> None:
        self.reid_state.long_term_embedding = value

    @property
    def prototype_updates(self) -> int:
        return self.reid_state.long_term_updates

    @prototype_updates.setter
    def prototype_updates(self, value: int) -> None:
        self.reid_state.long_term_updates = value

    @property
    def last_good_reid_frame(self) -> int:
        return self.reid_state.last_good_frame

    @last_good_reid_frame.setter
    def last_good_reid_frame(self, value: int) -> None:
        self.reid_state.last_good_frame = value

    @property
    def reid_frozen_until(self) -> int:
        return self.reid_state.frozen_until

    @reid_frozen_until.setter
    def reid_frozen_until(self, value: int) -> None:
        self.reid_state.frozen_until = value

    @property
    def switch_suspect_count(self) -> int:
        return self.reid_state.switch_suspect_count

    @switch_suspect_count.setter
    def switch_suspect_count(self, value: int) -> None:
        self.reid_state.switch_suspect_count = value

    @property
    def appearance_cost_history(self) -> list[float]:
        return self.reid_state.appearance_cost_history

    @appearance_cost_history.setter
    def appearance_cost_history(self, value: list[float]) -> None:
        self.reid_state.appearance_cost_history = value

    @property
    def suspicious_frames(self) -> list[int]:
        return self.reid_state.suspicious_frames

    @suspicious_frames.setter
    def suspicious_frames(self, value: list[int]) -> None:
        self.reid_state.suspicious_frames = value

    @property
    def fragment_breakpoints(self) -> list[int]:
        return self.reid_state.fragment_breakpoints

    @fragment_breakpoints.setter
    def fragment_breakpoints(self, value: list[int]) -> None:
        self.reid_state.fragment_breakpoints = value
