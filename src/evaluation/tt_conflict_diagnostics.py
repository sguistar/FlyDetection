from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from association.trajectory_temporal import TEMPORAL_SCALAR_DIM, build_trajectory_temporal
from association.trajectory_temporal import build_temporal_scalar_features
from config import clone_config, get_config
from identity.encoder import build_appearance_encoder
from identity.identity_memory import build_identity_memory
from identity.spacial_context import SPACIAL_INPUT_DIM, build_crop_spatial_input, build_spacial_context
from identity.transforms import build_reid_input
from io_utils.logger import setup_logger
from motion.kinematics import acceleration_from_history, velocity_from_history
from training.dataset import TEMPORAL_META_FIELDS

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


_META_INDEX = {name: idx for idx, name in enumerate(TEMPORAL_META_FIELDS)}


@dataclass(frozen=True)
class ConflictCase:
    frame_idx: int
    label: int
    history_paths: tuple[Path, ...]
    history_meta: tuple[np.ndarray, ...]
    query_path: Path
    query_meta: np.ndarray
    difficulty: float
    candidate_labels: tuple[int, ...]


def _normalize(vector: np.ndarray) -> np.ndarray:
    vector = vector.astype(np.float32)
    return vector / (np.linalg.norm(vector) + 1e-8)


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - float(np.dot(_normalize(a), _normalize(b))))


def _meta_value(meta_row: np.ndarray, key: str) -> float:
    return float(meta_row[_META_INDEX[key]])


def _rank_labels(score_by_label: dict[int, float]) -> list[int]:
    return [
        label
        for label, _ in sorted(
            score_by_label.items(),
            key=lambda item: (float(item[1]), int(item[0])),
        )
    ]


def _positive_margin(score_by_label: dict[int, float], positive_label: int) -> float:
    ordered = sorted(score_by_label.items(), key=lambda item: (float(item[1]), int(item[0])))
    if not ordered:
        return 0.0
    positive_score = float(score_by_label[positive_label])
    negative_scores = [float(score) for label, score in ordered if int(label) != int(positive_label)]
    best_negative = negative_scores[0] if negative_scores else positive_score
    return float(best_negative - positive_score)


def summarize_conflict_rows(rows: list[dict[str, object]]) -> dict[str, float]:
    if not rows:
        return {
            "num_cases": 0.0,
            "appearance_top1_acc": 0.0,
            "identity_top1_acc": 0.0,
            "temporal_top1_acc": 0.0,
            "ta_rescues_vs_identity": 0.0,
            "ta_hurts_vs_identity": 0.0,
            "ta_changes_top1": 0.0,
        }

    num_cases = float(len(rows))
    appearance_top1 = sum(1 for row in rows if int(row["appearance_rank"]) == 1)
    identity_top1 = sum(1 for row in rows if int(row["identity_rank"]) == 1)
    temporal_top1 = sum(1 for row in rows if int(row["temporal_rank"]) == 1)
    ta_rescues = sum(
        1
        for row in rows
        if int(row["temporal_rank"]) == 1 and int(row["identity_rank"]) != 1
    )
    ta_hurts = sum(
        1
        for row in rows
        if int(row["temporal_rank"]) != 1 and int(row["identity_rank"]) == 1
    )
    ta_changes_top1 = sum(
        1
        for row in rows
        if int(row["temporal_top1_label"]) != int(row["identity_top1_label"])
    )
    def _score_spread(key: str) -> float:
        spreads = []
        for row in rows:
            raw = row.get(key)
            if not raw:
                continue
            values = [float(item.split(":")[1]) for item in str(raw).split(";") if ":" in item]
            if values:
                spreads.append(max(values) - min(values))
        return float(np.mean(spreads)) if spreads else 0.0
    return {
        "num_cases": num_cases,
        "appearance_top1_acc": appearance_top1 / num_cases,
        "identity_top1_acc": identity_top1 / num_cases,
        "temporal_top1_acc": temporal_top1 / num_cases,
        "ta_rescues_vs_identity": float(ta_rescues),
        "ta_hurts_vs_identity": float(ta_hurts),
        "ta_changes_top1": float(ta_changes_top1),
        "appearance_mean_rank": float(np.mean([float(row["appearance_rank"]) for row in rows])),
        "identity_mean_rank": float(np.mean([float(row["identity_rank"]) for row in rows])),
        "temporal_mean_rank": float(np.mean([float(row["temporal_rank"]) for row in rows])),
        "appearance_mean_margin": float(np.mean([float(row["appearance_margin"]) for row in rows])),
        "identity_mean_margin": float(np.mean([float(row["identity_margin"]) for row in rows])),
        "temporal_mean_margin": float(np.mean([float(row["temporal_margin"]) for row in rows])),
        "appearance_mean_spread": _score_spread("appearance_scores"),
        "identity_mean_spread": _score_spread("identity_scores"),
        "temporal_mean_spread": _score_spread("temporal_scores"),
        "mean_difficulty": float(np.mean([float(row["difficulty"]) for row in rows])),
    }


def _load_checkpoint_models(cfg, checkpoint_path: str):
    if torch is None:
        raise ImportError("torch is required to run TT conflict diagnostics.")
    checkpoint = torch.load(checkpoint_path, map_location=cfg.device)

    appearance_model = build_appearance_encoder(
        embedding_dim=cfg.feature.embedding_dim,
        width=cfg.feature.cnn_width,
        dropout=cfg.feature.cnn_dropout,
    ).to(cfg.device)
    appearance_model.load_state_dict(
        checkpoint.get("appearance_state_dict", checkpoint.get("state_dict", checkpoint)),
        strict=False,
    )
    appearance_model.eval()

    spatial_model = build_spacial_context(
        input_dim=SPACIAL_INPUT_DIM,
        embedding_dim=cfg.feature.embedding_dim,
        hidden_dim=cfg.feature.spatial_hidden_dim,
        dropout=cfg.feature.spatial_dropout,
    ).to(cfg.device)
    spatial_state = checkpoint.get("spacial_context_state_dict")
    if spatial_state:
        spatial_model.load_state_dict(spatial_state, strict=False)
    spatial_model.eval()

    identity_model = build_identity_memory(
        embedding_dim=cfg.feature.embedding_dim,
        hidden_dim=cfg.feature.identity_hidden_dim,
        dropout=cfg.feature.identity_dropout,
    ).to(cfg.device)
    identity_state = checkpoint.get("identity_memory_state_dict")
    if identity_state:
        identity_model.load_state_dict(identity_state, strict=False)
    identity_model.eval()

    temporal_model = build_trajectory_temporal(
        input_dim=cfg.feature.embedding_dim * 2 + TEMPORAL_SCALAR_DIM,
        token_dim=cfg.feature.embedding_dim,
        hidden_dim=cfg.feature.temporal_hidden_dim,
        num_layers=cfg.feature.temporal_num_layers,
        num_heads=cfg.feature.temporal_num_heads,
        dropout=cfg.feature.temporal_dropout,
    ).to(cfg.device)
    temporal_state = checkpoint.get("trajectory_temporal_state_dict")
    if temporal_state:
        temporal_model.load_state_dict(temporal_state, strict=False)
    temporal_model.eval()
    return appearance_model, spatial_model, identity_model, temporal_model


def _load_metadata_rows(data_root: Path) -> list[dict[str, object]]:
    metadata_path = data_root / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata.csv under {data_root}")
    rows: list[dict[str, object]] = []
    with metadata_path.open("r", newline="", encoding="utf-8") as file_obj:
        reader = csv.DictReader(file_obj)
        for row in reader:
            path = data_root / str(row["image_path"])
            if not path.exists():
                continue
            rows.append(
                {
                    "path": path,
                    "label": int(row["identity"]),
                    "frame_idx": int(row["frame_idx"]),
                    "meta": np.asarray(
                        [float(row.get(field, 0.0) or 0.0) for field in TEMPORAL_META_FIELDS],
                        dtype=np.float32,
                    ),
                }
            )
    if not rows:
        raise RuntimeError(f"No metadata rows loaded from {metadata_path}")
    return rows


def _frame_difficulty(rows: list[dict[str, object]]) -> float:
    densities = [float(_meta_value(row["meta"], "local_density")) for row in rows]
    nearest_values = [
        float(np.clip((32.0 - _meta_value(row["meta"], "nearest_neighbor_dist")) / 32.0, 0.0, 1.0))
        for row in rows
    ]
    return float(
        np.clip(
            0.70 * float(np.mean(densities))
            + 0.30 * float(np.max(nearest_values) if nearest_values else 0.0),
            0.0,
            1.0,
        )
    )


def _build_conflict_cases(
    rows: list[dict[str, object]],
    *,
    history_len: int = 6,
    min_history: int = 3,
    hard_threshold: float = 0.20,
    max_cases: int = 180,
) -> list[ConflictCase]:
    rows_by_label: dict[int, list[dict[str, object]]] = defaultdict(list)
    rows_by_frame: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        rows_by_label[int(row["label"])].append(row)
        rows_by_frame[int(row["frame_idx"])].append(row)

    for label_rows in rows_by_label.values():
        label_rows.sort(key=lambda item: int(item["frame_idx"]))

    ranked_frames = sorted(
        [
            (frame_idx, _frame_difficulty(frame_rows))
            for frame_idx, frame_rows in rows_by_frame.items()
            if len(frame_rows) >= 2
        ],
        key=lambda item: (float(item[1]), int(item[0])),
        reverse=True,
    )

    cases: list[ConflictCase] = []
    for frame_idx, frame_score in ranked_frames:
        if frame_score < hard_threshold:
            continue
        frame_rows = rows_by_frame[frame_idx]
        candidate_labels = tuple(sorted(int(row["label"]) for row in frame_rows))
        for row in sorted(frame_rows, key=lambda item: int(item["label"])):
            label = int(row["label"])
            label_rows = rows_by_label[label]
            history_rows = [item for item in label_rows if int(item["frame_idx"]) < frame_idx][-history_len:]
            if len(history_rows) < min_history:
                continue
            row_difficulty = float(
                max(
                    frame_score,
                    float(_meta_value(row["meta"], "local_density")),
                )
            )
            cases.append(
                ConflictCase(
                    frame_idx=frame_idx,
                    label=label,
                    history_paths=tuple(Path(item["path"]) for item in history_rows),
                    history_meta=tuple(item["meta"] for item in history_rows),
                    query_path=Path(row["path"]),
                    query_meta=row["meta"],
                    difficulty=row_difficulty,
                    candidate_labels=candidate_labels,
                )
            )
            if len(cases) >= max_cases:
                return cases
    return cases


def _build_temporal_sequence_inputs(
    identity_embeddings: np.ndarray,
    spatial_embeddings: np.ndarray,
    frame_indices: np.ndarray,
    temporal_meta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    steps = identity_embeddings.shape[0]
    scalars = np.zeros((steps, TEMPORAL_SCALAR_DIM), dtype=np.float32)
    center_history: list[tuple[int, float, float]] = []
    prev_frame = None

    for step_idx in range(steps):
        meta_row = temporal_meta[step_idx]
        frame_idx = int(frame_indices[step_idx])
        cx = _meta_value(meta_row, "center_x")
        cy = _meta_value(meta_row, "center_y")
        center_history.append((frame_idx, cx, cy))
        vx, vy = velocity_from_history(center_history)
        ax, ay = acceleration_from_history(center_history)
        gap = 0.0 if prev_frame is None else float(max(frame_idx - prev_frame, 0))
        prev_frame = frame_idx
        frame_w = max(_meta_value(meta_row, "frame_w"), 1.0)
        frame_h = max(_meta_value(meta_row, "frame_h"), 1.0)
        local_density = _meta_value(meta_row, "local_density")
        border_flag = _meta_value(meta_row, "border_flag")
        neighbor_count = _meta_value(meta_row, "neighbor_count")
        scalars[step_idx] = build_temporal_scalar_features(
            vx=vx,
            vy=vy,
            ax=ax,
            ay=ay,
            area=_meta_value(meta_row, "area_proxy"),
            aspect=_meta_value(meta_row, "aspect_proxy"),
            reid_quality=float(np.clip(1.0 - 0.45 * border_flag - 0.35 * local_density, 0.10, 1.0)),
            is_crowded=neighbor_count >= 1.0 or local_density >= 0.20,
            is_merged_risk=local_density >= 0.45,
            interpolated=False,
            frame_gap=gap / max(float(steps), 1.0),
            memory_reliability=float(np.clip(1.0 - 0.20 * local_density - 0.15 * border_flag, 0.20, 1.0)),
            x_norm=cx / frame_w,
            y_norm=cy / frame_h,
        )
    sequence = np.concatenate([identity_embeddings, spatial_embeddings, scalars], axis=1).astype(np.float32)
    mask = np.zeros((steps,), dtype=np.float32)
    return sequence, mask


def _encode_crop_cache(
    paths: list[Path],
    cfg,
    appearance_model,
    spatial_model,
    identity_model,
) -> dict[str, dict[str, np.ndarray]]:
    cache: dict[str, dict[str, np.ndarray]] = {}
    for path in paths:
        key = str(path)
        if key in cache:
            continue
        image = cv2.imread(str(path))
        if image is None:
            continue
        crop_tensor = build_reid_input(image, backend="cnn", size=cfg.feature.crop_size)
        if not isinstance(crop_tensor, torch.Tensor):
            crop_tensor = torch.from_numpy(np.asarray(crop_tensor, dtype=np.float32))
        crop_tensor = crop_tensor.unsqueeze(0).to(cfg.device)
        spatial_input = torch.from_numpy(build_crop_spatial_input(image)).unsqueeze(0).to(cfg.device)
        with torch.no_grad():
            appearance_embedding, _ = appearance_model(crop_tensor)
            spatial_embedding = spatial_model(spatial_input)
            identity_embedding = identity_model(appearance_embedding, spatial_embedding)
        cache[key] = {
            "appearance": appearance_embedding[0].detach().cpu().numpy().astype(np.float32),
            "spatial": spatial_embedding[0].detach().cpu().numpy().astype(np.float32),
            "identity": identity_embedding[0].detach().cpu().numpy().astype(np.float32),
        }
    return cache


def _case_to_row(
    case: ConflictCase,
    *,
    frame_rows_by_frame: dict[int, list[dict[str, object]]],
    embedding_cache: dict[str, dict[str, np.ndarray]],
    temporal_model,
    cfg,
) -> dict[str, object] | None:
    history_keys = [str(path) for path in case.history_paths]
    if any(key not in embedding_cache for key in history_keys):
        return None

    history_identity = np.stack([embedding_cache[key]["identity"] for key in history_keys], axis=0)
    history_spatial = np.stack([embedding_cache[key]["spatial"] for key in history_keys], axis=0)
    history_appearance = np.stack([embedding_cache[key]["appearance"] for key in history_keys], axis=0)
    history_meta = np.stack(list(case.history_meta), axis=0).astype(np.float32)
    frame_indices = np.array([int(_meta_value(meta_row, "frame_idx")) if "frame_idx" in _META_INDEX else -1 for meta_row in history_meta], dtype=np.int64)
    if np.any(frame_indices < 0):
        frame_indices = np.array([_parse_frame_index_safe(path) for path in case.history_paths], dtype=np.int64)

    sequence, mask = _build_temporal_sequence_inputs(
        history_identity,
        history_spatial,
        frame_indices,
        history_meta,
    )
    with torch.no_grad():
        seq_tensor = torch.from_numpy(sequence).unsqueeze(0).to(cfg.device)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).to(cfg.device)
        temporal_token = temporal_model(seq_tensor, mask_tensor)[0].detach().cpu().numpy().astype(np.float32)

    appearance_anchor = _normalize(history_appearance.mean(axis=0))
    identity_anchor = _normalize(history_identity.mean(axis=0))
    temporal_anchor = _normalize(temporal_token)

    frame_rows = frame_rows_by_frame[case.frame_idx]
    candidate_map = {int(row["label"]): row for row in frame_rows if int(row["label"]) in case.candidate_labels}
    if case.label not in candidate_map or len(candidate_map) < 2:
        return None

    appearance_scores: dict[int, float] = {}
    identity_scores: dict[int, float] = {}
    temporal_scores: dict[int, float] = {}
    for label, row in sorted(candidate_map.items()):
        key = str(row["path"])
        if key not in embedding_cache:
            return None
        candidate_identity = embedding_cache[key]["identity"]
        candidate_appearance = embedding_cache[key]["appearance"]
        appearance_scores[label] = _cosine_distance(appearance_anchor, candidate_appearance)
        identity_scores[label] = _cosine_distance(identity_anchor, candidate_identity)
        temporal_scores[label] = _cosine_distance(temporal_anchor, candidate_identity)

    appearance_ranked = _rank_labels(appearance_scores)
    identity_ranked = _rank_labels(identity_scores)
    temporal_ranked = _rank_labels(temporal_scores)

    def _rank_of(ordered: list[int], label: int) -> int:
        return int(ordered.index(label) + 1)

    return {
        "frame_idx": case.frame_idx,
        "label": case.label,
        "difficulty": case.difficulty,
        "num_candidates": len(candidate_map),
        "appearance_rank": _rank_of(appearance_ranked, case.label),
        "identity_rank": _rank_of(identity_ranked, case.label),
        "temporal_rank": _rank_of(temporal_ranked, case.label),
        "appearance_top1_label": appearance_ranked[0],
        "identity_top1_label": identity_ranked[0],
        "temporal_top1_label": temporal_ranked[0],
        "appearance_margin": _positive_margin(appearance_scores, case.label),
        "identity_margin": _positive_margin(identity_scores, case.label),
        "temporal_margin": _positive_margin(temporal_scores, case.label),
        "appearance_scores": ";".join(f"{label}:{appearance_scores[label]:.4f}" for label in appearance_ranked),
        "identity_scores": ";".join(f"{label}:{identity_scores[label]:.4f}" for label in identity_ranked),
        "temporal_scores": ";".join(f"{label}:{temporal_scores[label]:.4f}" for label in temporal_ranked),
    }


def _parse_frame_index_safe(path: Path) -> int:
    stem = path.stem
    if "frame_" not in stem:
        return -1
    try:
        return int(stem.split("frame_", 1)[1].split("_", 1)[0])
    except Exception:
        return -1


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
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


def run_tt_conflict_diagnostics(
    cfg=None,
    *,
    checkpoint_path: str | None = None,
    output_root: str | Path | None = None,
    history_len: int = 6,
    min_history: int = 3,
    hard_threshold: float = 0.20,
    max_cases: int = 180,
) -> dict[str, object]:
    cfg = clone_config(cfg or get_config())
    checkpoint_path = checkpoint_path or cfg.feature.encoder_checkpoint
    if not checkpoint_path:
        raise ValueError("checkpoint_path is required for TT conflict diagnostics.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    diag_root = Path(output_root) if output_root is not None else cfg.paths.root / "diagnostics" / f"tt_conflict_{timestamp}"
    diag_root.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(diag_root / "tt_conflict.log")

    data_root = Path(cfg.training.data_root)
    rows = _load_metadata_rows(data_root)
    cases = _build_conflict_cases(
        rows,
        history_len=history_len,
        min_history=min_history,
        hard_threshold=hard_threshold,
        max_cases=max_cases,
    )
    logger.info("prepared conflict cases | %s", {"count": len(cases), "checkpoint_path": checkpoint_path})

    appearance_model, spatial_model, identity_model, temporal_model = _load_checkpoint_models(cfg, checkpoint_path)
    all_paths = sorted(
        {
            str(case.query_path)
            for case in cases
        }
        | {
            str(path)
            for case in cases
            for path in case.history_paths
        }
        | {
            str(row["path"])
            for row in rows
            if int(row["frame_idx"]) in {case.frame_idx for case in cases}
        }
    )
    embedding_cache = _encode_crop_cache([Path(path) for path in all_paths], cfg, appearance_model, spatial_model, identity_model)
    frame_rows_by_frame: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if int(row["frame_idx"]) in {case.frame_idx for case in cases}:
            frame_rows_by_frame[int(row["frame_idx"])].append(row)

    case_rows = []
    for case in cases:
        row = _case_to_row(
            case,
            frame_rows_by_frame=frame_rows_by_frame,
            embedding_cache=embedding_cache,
            temporal_model=temporal_model,
            cfg=cfg,
        )
        if row is not None:
            case_rows.append(row)

    summary = summarize_conflict_rows(case_rows)
    summary["checkpoint_path"] = str(checkpoint_path)
    summary["history_len"] = float(history_len)
    summary["hard_threshold"] = float(hard_threshold)

    case_rows_sorted = sorted(
        case_rows,
        key=lambda row: (
            abs(int(row["temporal_rank"]) - int(row["identity_rank"])),
            float(row["difficulty"]),
            int(row["frame_idx"]),
        ),
        reverse=True,
    )

    cases_path = diag_root / "tt_conflict_cases.csv"
    summary_path = diag_root / "tt_conflict_summary.csv"
    _write_rows(cases_path, case_rows_sorted)
    _write_rows(summary_path, [summary])
    logger.info("finished TT conflict diagnostics | %s", {"summary": summary, "cases_path": str(cases_path)})
    return {
        "diagnostic_root": str(diag_root),
        "cases_path": str(cases_path),
        "summary_path": str(summary_path),
        "summary": summary,
        "num_cases": len(case_rows_sorted),
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose whether TT changes identity ranking on hard conflict frames.")
    parser.add_argument("--checkpoint-path", default=None, help="Optional bundle checkpoint path.")
    parser.add_argument("--output-root", default=None, help="Optional diagnostic output root.")
    parser.add_argument("--history-len", type=int, default=6, help="Number of prior labeled crops to encode as TT history.")
    parser.add_argument("--min-history", type=int, default=3, help="Minimum history length required for a conflict case.")
    parser.add_argument("--hard-threshold", type=float, default=0.20, help="Minimum per-frame difficulty to include a case.")
    parser.add_argument("--max-cases", type=int, default=180, help="Maximum number of conflict cases to analyze.")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    summary = run_tt_conflict_diagnostics(
        checkpoint_path=args.checkpoint_path,
        output_root=args.output_root,
        history_len=args.history_len,
        min_history=args.min_history,
        hard_threshold=args.hard_threshold,
        max_cases=args.max_cases,
    )
    print(summary)


if __name__ == "__main__":
    main()
