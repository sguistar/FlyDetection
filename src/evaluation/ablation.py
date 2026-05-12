from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import clone_config, get_config
from evaluation.benchmark import run_benchmark


def _motion_only_overrides() -> dict[str, object]:
    return {
        "reid.enabled": False,
        "reid.use_trajectory_temporal": False,
        "reid.use_slot_reassign": False,
        "reid.slot_stickiness_enabled": False,
        "feature.use_identity_memory": False,
        "feature.use_spacial_context": False,
        "feature.appearance_weight": 0.0,
        "feature.temporal_weight": 0.0,
        "feature.shape_weight": 0.0,
        "feature.spatial_weight": 0.0,
        "feature.direction_weight": 0.0,
        "association.use_learned_head": False,
        "association.risk_adaptive_weights": False,
        "association.hard_conflict_identity_blend": 0.0,
        "detection.rescue_enabled": False,
        "detection.enhanced_rescue_enabled": False,
        "detection.bootstrap_tile_enabled": False,
        "track.enable_global_reid": False,
        "track.enable_long_gap_bridge": False,
        "track.enable_interpolation": False,
        "track.enable_latent_slot_reconnect": False,
    }


def _with_reid() -> dict[str, object]:
    overrides = _motion_only_overrides()
    overrides.update(
        {
            "reid.enabled": True,
            "feature.use_identity_memory": True,
            "feature.appearance_weight": 0.15,
            "association.risk_adaptive_weights": True,
            "association.hard_conflict_identity_blend": 0.18,
        }
    )
    return overrides


def _with_trajectory_temporal() -> dict[str, object]:
    overrides = _with_reid()
    overrides.update(
        {
            "reid.use_trajectory_temporal": True,
            "feature.temporal_weight": 0.16,
        }
    )
    return overrides


def _with_rescue_detection() -> dict[str, object]:
    overrides = _with_trajectory_temporal()
    overrides.update(
        {
            "detection.rescue_enabled": True,
            "detection.enhanced_rescue_enabled": True,
            "detection.bootstrap_tile_enabled": False,
        }
    )
    return overrides


def _with_global_reid() -> dict[str, object]:
    overrides = _with_rescue_detection()
    overrides.update(
        {
            "reid.use_slot_reassign": True,
            "track.enable_global_reid": True,
        }
    )
    return overrides


def build_requested_ablation_override_sets() -> list[dict]:
    return [
        {
            "name": "baseline",
            "label": "Baseline",
            "setting": "YOLO + simple motion matching",
            "purpose": "Basic comparison",
            "overrides": _motion_only_overrides(),
        },
        {
            "name": "plus_reid",
            "label": "+ ReID",
            "setting": "Add appearance identity features",
            "purpose": "Verify ReID effect",
            "overrides": _with_reid(),
        },
        {
            "name": "plus_trajectory_temporal",
            "label": "+ Trajectory Temporal",
            "setting": "Add temporal features",
            "purpose": "Verify temporal modeling",
            "overrides": _with_trajectory_temporal(),
        },
        {
            "name": "plus_rescue_detection",
            "label": "+ Rescue Detection",
            "setting": "Add rescue detection",
            "purpose": "Verify FN improvement",
            "overrides": _with_rescue_detection(),
        },
        {
            "name": "plus_global_reid",
            "label": "+ Global ReID",
            "setting": "Add global identity reassignment",
            "purpose": "Verify IDSW improvement",
            "overrides": _with_global_reid(),
        },
        {
            "name": "full_system",
            "label": "Full System",
            "setting": "Complete system",
            "purpose": "Overall performance",
            "overrides": {},
        },
    ]


def run_ablation(
    base_cfg=None,
    *,
    output_root: str | Path | None = None,
    save_video: bool = False,
) -> list[dict]:
    base_cfg = clone_config(base_cfg or get_config())
    base_cfg.runtime.save_video = bool(save_video)
    if output_root is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_root = Path(base_cfg.runtime.output_root) / "ablation" / timestamp
    return run_benchmark(
        base_cfg=base_cfg,
        override_sets=build_requested_ablation_override_sets(),
        output_root=output_root,
        summary_name="ablation_summary.csv",
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the requested MOT ablation study.")
    parser.add_argument("--video-path", default="min_test.mp4", help="Video used for all ablation runs.")
    parser.add_argument("--gt-path", default=None, help="Optional ground-truth CSV path.")
    parser.add_argument("--output-root", default=None, help="Directory for isolated ablation runs.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional frame limit for quick checks.")
    parser.add_argument("--save-video", action="store_true", help="Render a result video for each run.")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    cfg = get_config()
    cfg.runtime.video_path = args.video_path
    cfg.runtime.save_video = args.save_video
    if args.max_frames is not None:
        cfg.runtime.max_frames = args.max_frames
    if args.gt_path:
        cfg.evaluation.gt_csv_path = args.gt_path

    rows = run_ablation(cfg, output_root=args.output_root, save_video=args.save_video)
    for row in rows:
        label = row.get("label", row["run_name"])
        metrics = {
            key: row.get(key)
            for key in ("idf1", "mota_like", "point_hota", "recall", "fn", "idsw")
            if key in row
        }
        print(label, metrics)


if __name__ == "__main__":
    main()
