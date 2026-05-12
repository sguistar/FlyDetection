from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import copy
import hashlib
import json
import random
from typing import Any

import numpy as np
import torch


@dataclass
class DetectionConfig:
    model_path: str = "best.pt" # use yolov10n.pt for fly detection
    conf_thres: float = 0.05
    iou_thres: float = 0.80
    imgsz: int = 2560
    max_det: int = 32
    rescue_enabled: bool = True
    rescue_conf_thres: float = 0.001
    rescue_roi_scale: float = 1.8
    rescue_min_imgsz: int = 640
    rescue_imgsz_scale: float = 8.0
    rescue_when_full: bool = True
    rescue_max_rois_when_full: int = 2
    rescue_min_track_hits: int = 4
    rescue_coverage_radius_scale: float = 0.35
    rescue_coverage_radius_min: float = 2.0
    rescue_max_per_slot: int = 1
    rescue_center_gate_scale: float = 0.60
    rescue_center_gate_min: float = 8.0
    enhanced_rescue_enabled: bool = True
    bootstrap_tile_enabled: bool = False
    bootstrap_frames: int = 40
    bootstrap_grid_size: int = 2
    bootstrap_tile_overlap: float = 0.18
    class_ids: list[int] | None = None


@dataclass
class PreprocessConfig:
    crop_pad: int = 2
    min_area: float = 750.0
    max_area: float = 4000.0
    min_aspect: float = 0.10
    max_aspect: float = 10.0
    min_blur_score: float = 1.8
    border_margin: int = 2
    duplicate_iou_thres: float = 0.75
    duplicate_center_thres: float = 8.0
    keep_low_quality_border: bool = True
    keep_track_supported_low_quality: bool = True
    track_support_radius_scale: float = 1.6
    track_support_max_distance: float = 42.0


@dataclass
class FeatureConfig:
    encoder_backend: str = "cnn"
    embedding_dim: int = 128
    history_len: int = 16
    temporal_num_heads: int = 4
    short_term_window: int = 6
    long_term_momentum: float = 0.98
    quarantine_min_quality: float = 0.45
    appearance_weight: float = 0.15
    temporal_weight: float = 0.16
    shape_weight: float = 0.10
    spatial_weight: float = 0.18
    motion_weight: float = 1.05
    direction_weight: float = 0.10
    appearance_gate: float = 0.40
    temporal_gate: float = 0.62
    shape_gate: float = 0.65
    spatial_gate: float = 0.70
    recent_embedding_window: int = 20
    crop_size: tuple[int, int] = (112, 112)
    encoder_checkpoint: str = ""
    cnn_width: int = 32
    cnn_dropout: float = 0.10
    use_identity_memory: bool = True
    use_spacial_context: bool = True
    identity_hidden_dim: int = 128
    identity_dropout: float = 0.10
    spatial_hidden_dim: int = 96
    spatial_dropout: float = 0.10
    temporal_hidden_dim: int = 128
    temporal_num_layers: int = 1
    temporal_dropout: float = 0.10
    fallback_to_handcrafted_when_untrained: bool = True


@dataclass
class AssociationConfig:
    motion_gate: float = 110.0
    kf_gate: float = 9.0
    max_link_gap: int = 18
    max_interpolation_gap: int = 4
    large_cost: float = 1e6
    support_reconnect_bonus: float = 0.12
    support_lost_track_bonus: float = 0.06
    support_fallback_cost_thres: float = 0.92
    support_score_floor: float = 0.24
    support_switch_risk_cap: float = 0.40
    hard_conflict_identity_blend: float = 0.18
    hard_conflict_min_risk: float = 0.45
    hard_conflict_min_density: float = 0.18
    slot_swap_suppress_enabled: bool = True
    slot_swap_cost_margin: float = 0.08
    slot_swap_distance_thres: float = 80.0
    slot_swap_stable_hits: int = 6
    slot_swap_max_missed: int = 2
    use_learned_head: bool = True
    handcrafted_blend_weight: float = 0.55
    learned_blend_weight: float = 0.45
    low_match_penalty: float = 0.35
    high_switch_penalty: float = 0.25
    risk_adaptive_weights: bool = True
    switch_risk_weight: float = 0.50
    match_score_gate: float = 0.12
    association_hidden_dim: int = 64


@dataclass
class ReIDConfig:
    enabled: bool = True
    merge_threshold: float = 0.85
    appearance_threshold: float = 0.30
    shape_threshold: float = 0.50
    spatial_threshold: float = 0.55
    motion_threshold: float = 0.65
    use_trajectory_temporal: bool = True
    use_slot_reassign: bool = True
    slot_stickiness_enabled: bool = True
    slot_stickiness_max_fragment_len: int = 20
    slot_stickiness_max_gap: int = 20
    slot_stickiness_max_speed: float = 14.0
    slot_stickiness_min_anchor_len: int = 24
    fragment_min_len: int = 3
    fragment_max_internal_gap: int = 3
    offline_window: int = 24
    offline_force_assign_when_full: bool = False


@dataclass
class TrackConfig:
    num_flies: int = 6
    use_identity_slots: bool = True
    identity_slots: int = 6
    recall_mode: bool = True
    confirm_hits: int = 2
    max_missed: int = 8
    min_track_length: int = 1
    remove_tentative_after: int = 2
    reid_update_quality_thres: float = 0.45
    suspicious_appearance_thres: float = 0.35
    suspicious_hits: int = 2
    reid_freeze_frames: int = 10
    recovery_confirm_hits: int = 2
    enable_latent_slot_reconnect: bool = True
    latent_slot_max_age: int = 2400
    latent_motion_gate: float = 36.0
    latent_shape_ratio_tol: float = 0.90
    latent_reconnect_min_reliability: float = 0.25
    enable_weak_match_motion_blend: bool = True
    weak_match_min_hits: int = 4
    weak_match_score_thres: float = 0.40
    weak_match_quality_thres: float = 0.55
    weak_match_switch_risk_thres: float = 0.35
    weak_match_position_alpha: float = 0.35
    prune_short_low_conf_tracks: bool = True
    prune_rescue_heavy_tracks: bool = False
    low_conf_track_max_length: int = 12
    low_conf_track_mean_conf: float = 0.20
    rescue_ghost_min_feature_points: int = 12
    rescue_ghost_min_ratio: float = 0.90
    rescue_ghost_mean_conf: float = 0.06
    rescue_ghost_extreme_mean_conf: float = 0.03
    rescue_ghost_max_main_count: int = 2
    rescue_ghost_max_main_ratio: float = 0.15
    enable_global_reid: bool = True
    enable_long_gap_bridge: bool = True
    long_gap_bridge_min_gap: int = 5
    long_gap_bridge_max_gap: int = 2400
    long_gap_bridge_velocity_window: int = 6
    long_gap_bridge_endpoint_tol_per_frame: float = 2.5
    long_gap_bridge_max_step_per_frame: float = 14.0
    long_gap_bridge_shape_ratio_tol: float = 0.75
    long_gap_bridge_min_conf_scale: float = 0.12
    enable_interpolation: bool = True


@dataclass
class EventConfig:
    enable_crossing: bool = True
    enable_interaction: bool = True
    x_line: float | None = None
    y_line: float | None = None
    interaction_distance: float = 35.0
    merged_iou_threshold: float = 0.50
    duplicate_iou_threshold: float = 0.85
    split_merge_center_threshold: float = 16.0


@dataclass
class RenderConfig:
    trail_len: int = 24
    draw_labels: bool = True
    draw_hud: bool = True
    max_event_lines: int = 6
    bbox_thickness: int = 1


@dataclass
class EvaluationConfig:
    enabled: bool = True
    gt_csv_path: str = "coords/gt.csv"
    point_match_distance: float = 24.0
    compute_hota: bool = True
    ignore_unlabeled_frames: bool = True
    gt_frame_stride: int | None = 10
    gt_frame_offset: int = 0
    prediction_id_source: str = "auto"
    temporal_window_sec: float = 20.0
    detector_miss_long_segment_points: int = 5


@dataclass
class CacheConfig:
    enabled: bool = False
    use_detection_cache: bool = False
    write_detection_cache: bool = False


@dataclass
class TrainingConfig:
    data_root: str = ""
    train_subdir: str = "train"
    val_subdir: str = "val"
    epochs: int = 30
    batch_size: int = 16
    num_workers: int = 0
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    label_smoothing: float = 0.0
    triplet_margin: float = 0.3
    ce_weight: float = 1.0
    triplet_weight: float = 0.5
    supcon_weight: float = 0.5
    temporal_consistency_weight: float = 0.25
    temporal_token_consistency_weight: float = 0.35
    association_bce_weight: float = 0.5
    identity_frame_supcon_weight: float = 0.30
    identity_frame_triplet_weight: float = 0.20
    identity_anchor_weight: float = 0.20
    identity_center_margin_weight: float = 0.10
    temporal_center_margin_weight: float = 0.05
    representation_center_margin: float = 0.12
    identities_per_batch: int = 4
    samples_per_identity: int = 4
    min_images_per_identity: int = 2
    clip_len: int = 6
    clip_stride: int | None = None
    val_split: float = 0.2
    checkpoint_path: str | None = None
    resume_checkpoint: str | None = None
    max_batches_per_epoch: int | None = None
    save_best_only: bool = True
    random_seed: int = 42
    auto_prepare_from_gt: bool = True
    write_clip_metadata: bool = True
    association_hard_negative_topk: int = 1
    hard_sample_ratio: float = 0.60
    temporal_hard_mining: bool = True
    temporal_hard_threshold: float = 0.20
    temporal_hard_oversample: int = 2
    freeze_appearance_encoder: bool = False
    freeze_spacial_context: bool = False
    freeze_identity_memory: bool = False
    freeze_trajectory_temporal: bool = False
    freeze_association_head: bool = False
    temporal_predictive_weight: float = 0.45


@dataclass
class RuntimeConfig:
    video_path: str = "min_test.mp4"
    output_root: str = "outputs"
    seed: int = 42
    use_cuda: bool = True
    half_precision: bool = True
    max_frames: int | None = None
    log_every_n_frames: int = 50
    save_detection_csv: bool = True
    save_track_csv: bool = True
    save_event_csv: bool = True
    save_metrics_csv: bool = True
    save_video: bool = True
    save_cache: bool = False


@dataclass
class Paths:
    root: Path
    videos: Path = field(init=False)
    csv: Path = field(init=False)
    logs: Path = field(init=False)
    cache: Path = field(init=False)
    figures: Path = field(init=False)
    models: Path = field(init=False)

    def __post_init__(self) -> None:
        self.videos = self.root / "videos"
        self.csv = self.root / "csv"
        self.logs = self.root / "logs"
        self.cache = self.root / "cache"
        self.figures = self.root / "figures"
        self.models = self.root / "models"

    def mkdirs(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.videos.mkdir(parents=True, exist_ok=True)
        self.csv.mkdir(parents=True, exist_ok=True)
        self.logs.mkdir(parents=True, exist_ok=True)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.figures.mkdir(parents=True, exist_ok=True)
        self.models.mkdir(parents=True, exist_ok=True)


@dataclass
class Config:
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    association: AssociationConfig = field(default_factory=AssociationConfig)
    reid: ReIDConfig = field(default_factory=ReIDConfig)
    track: TrackConfig = field(default_factory=TrackConfig)
    events: EventConfig = field(default_factory=EventConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    @property
    def device(self) -> str:
        if torch is None:
            return "cpu"
        if self.runtime.use_cuda and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    @property
    def model_device(self) -> int | str:
        return 0 if self.device == "cuda" else "cpu"

    @property
    def paths(self) -> Paths:
        return Paths(Path(self.runtime.output_root))

    @property
    def config_hash(self) -> str:
        payload = asdict(self)
        payload["runtime"]["video_path"] = str(Path(self.runtime.video_path))
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
        return hashlib.sha1(encoded).hexdigest()[:12]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def get_config() -> Config:
    cfg = Config()
    cfg.cache.enabled = cfg.cache.enabled or cfg.runtime.save_cache
    cfg.cache.write_detection_cache = cfg.cache.write_detection_cache or cfg.runtime.save_cache
    set_random_seed(cfg.runtime.seed)
    cfg.paths.mkdirs()
    if not cfg.training.data_root:
        cfg.training.data_root = str(cfg.paths.root / "reid_data")
    if cfg.training.checkpoint_path is None:
        cfg.training.checkpoint_path = str(cfg.paths.models / "appearance_encoder.pt")
    if not cfg.feature.encoder_checkpoint:
        cfg.feature.encoder_checkpoint = cfg.training.checkpoint_path
    return cfg


def clone_config(cfg: Config) -> Config:
    return copy.deepcopy(cfg)


def apply_overrides(cfg: Config, overrides: dict[str, Any]) -> Config:
    for key, value in overrides.items():
        parts = key.split(".")
        target: Any = cfg
        for part in parts[:-1]:
            target = getattr(target, part)
        setattr(target, parts[-1], value)
    return cfg


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
