from __future__ import annotations

import argparse
import csv
from datetime import datetime
import os
from pathlib import Path
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import apply_overrides, clone_config, get_config
from evaluation.benchmark import run_benchmark


def _write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_default_tt_sc_im_override_sets() -> list[dict]:
    return [
        {
            "name": "appearance_only",
            "overrides": {
                "feature.use_identity_memory": False,
                "feature.use_spacial_context": False,
                "reid.use_trajectory_temporal": False,
                "association.use_learned_head": False,
                "feature.appearance_weight": 0.20,
                "feature.spatial_weight": 0.0,
            },
        },
        {
            "name": "spatial_only",
            "overrides": {
                "feature.use_identity_memory": False,
                "feature.use_spacial_context": True,
                "reid.use_trajectory_temporal": False,
                "association.use_learned_head": False,
                "feature.appearance_weight": 0.0,
                "feature.spatial_weight": 0.22,
            },
        },
        {
            "name": "identity_only",
            "overrides": {
                "feature.use_identity_memory": True,
                "feature.use_spacial_context": False,
                "reid.use_trajectory_temporal": False,
                "association.use_learned_head": False,
                "feature.appearance_weight": 0.18,
                "feature.spatial_weight": 0.0,
            },
        },
        {
            "name": "temporal_only",
            "overrides": {
                "feature.use_identity_memory": False,
                "feature.use_spacial_context": False,
                "reid.use_trajectory_temporal": True,
                "association.use_learned_head": False,
                "feature.appearance_weight": 0.18,
                "feature.spatial_weight": 0.0,
            },
        },
        {
            "name": "full_tt_sc_im",
            "overrides": {
                "feature.use_identity_memory": True,
                "feature.use_spacial_context": True,
                "reid.use_trajectory_temporal": True,
                "association.use_learned_head": True,
                "feature.appearance_weight": 0.15,
                "feature.spatial_weight": 0.18,
            },
        },
        {
            "name": "full_tt_sc_im_no_learned_head",
            "overrides": {
                "feature.use_identity_memory": True,
                "feature.use_spacial_context": True,
                "reid.use_trajectory_temporal": True,
                "association.use_learned_head": False,
                "feature.appearance_weight": 0.15,
                "feature.spatial_weight": 0.18,
            },
        },
    ]


def build_default_training_overrides(*, preset: str = "standard") -> dict[str, object]:
    if preset == "quick":
        return {
            "training.epochs": 4,
            "training.batch_size": 8,
            "training.identities_per_batch": 4,
            "training.samples_per_identity": 2,
            "training.max_batches_per_epoch": 12,
            "training.save_best_only": True,
        }
    if preset == "full":
        return {
            "training.epochs": 12,
            "training.batch_size": 16,
            "training.identities_per_batch": 4,
            "training.samples_per_identity": 4,
            "training.max_batches_per_epoch": None,
            "training.save_best_only": True,
        }
    return {
        "training.epochs": 8,
        "training.batch_size": 12,
        "training.identities_per_batch": 4,
        "training.samples_per_identity": 3,
        "training.max_batches_per_epoch": None,
        "training.save_best_only": True,
    }


def _selection_score(row: dict) -> float:
    return (
        float(row.get("idf1", 0.0))
        + 0.35 * float(row.get("point_hota", 0.0))
        + 0.10 * float(row.get("idr", 0.0))
        + 0.05 * float(row.get("idp", 0.0))
        - 0.010 * float(row.get("idsw", 0.0))
        - 0.001 * float(row.get("fn", 0.0))
    )


def _rank_rows(rows: list[dict]) -> list[dict]:
    ranked = []
    ordered = sorted(rows, key=_selection_score, reverse=True)
    for rank, row in enumerate(ordered, start=1):
        ranked.append(
            {
                "rank": rank,
                "selection_score": _selection_score(row),
                **row,
            }
        )
    return ranked


def run_tt_sc_im_sweep(
    base_cfg=None,
    *,
    train_first: bool = False,
    training_overrides: dict[str, object] | None = None,
    override_sets: list[dict] | None = None,
    output_root: str | Path | None = None,
    preset: str = "standard",
) -> dict:
    base_cfg = clone_config(base_cfg or get_config())
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_root = Path(output_root) if output_root is not None else Path(base_cfg.runtime.output_root) / "sweeps" / f"tt_sc_im_{timestamp}"
    sweep_root.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(base_cfg.feature.encoder_checkpoint) if base_cfg.feature.encoder_checkpoint else sweep_root / "models" / "appearance_encoder.pt"
    training_summary = None
    if train_first:
        from training.train_encoder import train_encoder

        train_cfg = clone_config(base_cfg)
        train_cfg.runtime.output_root = str(sweep_root / "train")
        train_cfg.paths.mkdirs()
        train_cfg.training.checkpoint_path = str(sweep_root / "models" / "appearance_encoder.pt")
        train_cfg.feature.encoder_checkpoint = train_cfg.training.checkpoint_path
        apply_overrides(train_cfg, build_default_training_overrides(preset=preset))
        if training_overrides:
            apply_overrides(train_cfg, training_overrides)
        training_summary = train_encoder(train_cfg)
        checkpoint_path = Path(training_summary["checkpoint_path"])

    benchmark_cfg = clone_config(base_cfg)
    benchmark_cfg.feature.encoder_checkpoint = str(checkpoint_path)
    benchmark_cfg.training.checkpoint_path = str(checkpoint_path)
    benchmark_cfg.paths.mkdirs()

    rows = run_benchmark(
        base_cfg=benchmark_cfg,
        override_sets=override_sets or build_default_tt_sc_im_override_sets(),
        output_root=sweep_root / "runs",
        summary_name="benchmark_summary.csv",
    )
    ranked_rows = _rank_rows(rows)
    summary_path = sweep_root / "tt_sc_im_sweep_summary.csv"
    _write_rows(summary_path, ranked_rows)
    best_row = ranked_rows[0] if ranked_rows else None
    return {
        "sweep_root": str(sweep_root),
        "checkpoint_path": str(checkpoint_path),
        "training_summary": training_summary,
        "rows": ranked_rows,
        "best_row": best_row,
        "summary_path": str(summary_path),
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run TT/SC/IM training and sweep experiments.")
    parser.add_argument("--train-first", action="store_true", help="Train a fresh TT/SC/IM bundle before the sweep.")
    parser.add_argument("--preset", choices=["quick", "standard", "full"], default="standard", help="Training preset when --train-first is enabled.")
    parser.add_argument("--output-root", default=None, help="Optional sweep output root directory.")
    parser.add_argument("--video-path", default=None, help="Optional video path override.")
    parser.add_argument("--gt-path", default=None, help="Optional GT csv path override.")
    parser.add_argument("--checkpoint-path", default=None, help="Optional checkpoint path override for sweep runs.")
    parser.add_argument("--epochs", type=int, default=None, help="Optional training epoch override.")
    parser.add_argument("--max-batches", type=int, default=None, help="Optional training max batches per epoch.")
    parser.add_argument("--batch-size", type=int, default=None, help="Optional training batch size override.")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    cfg = get_config()
    if args.video_path:
        cfg.runtime.video_path = args.video_path
    if args.gt_path:
        cfg.evaluation.gt_csv_path = args.gt_path
    if args.checkpoint_path:
        cfg.feature.encoder_checkpoint = args.checkpoint_path
        cfg.training.checkpoint_path = args.checkpoint_path

    training_overrides: dict[str, object] = {}
    if args.epochs is not None:
        training_overrides["training.epochs"] = args.epochs
    if args.max_batches is not None:
        training_overrides["training.max_batches_per_epoch"] = args.max_batches
    if args.batch_size is not None:
        training_overrides["training.batch_size"] = args.batch_size

    summary = run_tt_sc_im_sweep(
        base_cfg=cfg,
        train_first=args.train_first,
        training_overrides=training_overrides or None,
        output_root=args.output_root,
        preset=args.preset,
    )
    print(summary)


if __name__ == "__main__":
    main()
