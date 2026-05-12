from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from association.association_head import (
    ASSOCIATION_FEATURE_DIM,
    build_association_feature_vector,
    build_association_head,
)
from association.trajectory_temporal import (
    TEMPORAL_SCALAR_DIM,
    build_trajectory_temporal,
    build_temporal_scalar_features,
)
from config import get_config
from identity.encoder import build_appearance_encoder
from identity.identity_memory import build_identity_memory
from identity.spacial_context import SPACIAL_INPUT_DIM, build_spacial_context
from io_utils.logger import setup_logger
from motion.kinematics import acceleration_from_history, velocity_from_history
from training.dataset import (
    TEMPORAL_META_FIELDS,
    ReIDClipDataset,
    build_reid_clip_splits,
)
from training.losses import build_loss
from training.prepare_reid_dataset import ensure_reid_dataset
from training.sampler import IdentityBalancedSampler


def _to_tensor(value, *, device: str, dtype=None):
    """Move arbitrary batch payloads onto the target device with an optional dtype cast.

    将任意批量数据移动到目标设备，并可选择转换 dtype。
    """
    if torch is None:
        raise ImportError("torch is required to train the appearance encoder.")
    if isinstance(value, torch.Tensor):
        tensor = value.to(device)
    else:
        tensor = torch.as_tensor(value, device=device)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor


def _compute_accuracy(logits: torch.Tensor | None, labels: torch.Tensor) -> float:
    if logits is None or logits.numel() == 0:
        return 0.0
    preds = torch.argmax(logits, dim=1)
    return float((preds == labels).float().mean().item())


def _module_trainable(module) -> bool:
    return any(param.requires_grad for param in module.parameters())


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = (1.0 - mask).unsqueeze(-1)
    denom = valid.sum(dim=1).clamp(min=1.0)
    return (values * valid).sum(dim=1) / denom


def _masked_count(mask: torch.Tensor) -> torch.Tensor:
    return (1.0 - mask).sum(dim=1)


def _class_center_margin_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    margin: float = 0.12,
) -> torch.Tensor:
    if features.ndim != 2 or labels.ndim != 1 or features.shape[0] <= 1:
        return features.new_tensor(0.0)
    centers = []
    unique_labels = torch.unique(labels)
    if unique_labels.numel() <= 1:
        return features.new_tensor(0.0)
    normalized = F.normalize(features, dim=-1)
    for label in unique_labels.tolist():
        label_mask = labels == int(label)
        if not torch.any(label_mask):
            continue
        center = F.normalize(normalized[label_mask].mean(dim=0, keepdim=True), dim=-1)[0]
        centers.append(center)
    if len(centers) <= 1:
        return features.new_tensor(0.0)
    centers_tensor = torch.stack(centers, dim=0)
    distances = 1.0 - torch.matmul(centers_tensor, centers_tensor.T)
    pair_mask = torch.triu(torch.ones_like(distances, dtype=torch.bool), diagonal=1)
    if not torch.any(pair_mask):
        return features.new_tensor(0.0)
    penalties = F.relu(float(margin) - distances[pair_mask])
    return penalties.mean() if penalties.numel() > 0 else features.new_tensor(0.0)


def _anchor_alignment_loss(features: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    if features.ndim != 2 or anchors.ndim != 2 or features.shape[0] == 0 or anchors.shape[0] == 0:
        return features.new_tensor(0.0)
    return (1.0 - F.cosine_similarity(features, anchors, dim=-1)).mean()


def _last_valid_embeddings(frame_embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    outputs = []
    steps = frame_embeddings.shape[1]
    for batch_idx in range(frame_embeddings.shape[0]):
        valid_indices = torch.nonzero(mask[batch_idx] < 0.5, as_tuple=False).flatten()
        if valid_indices.numel() == 0:
            outputs.append(frame_embeddings[batch_idx, 0])
        else:
            outputs.append(frame_embeddings[batch_idx, int(valid_indices[-1].item())])
    return torch.stack(outputs, dim=0)


def _recent_mean_embeddings(frame_embeddings: torch.Tensor, mask: torch.Tensor, recent: int = 3) -> torch.Tensor:
    outputs = []
    for batch_idx in range(frame_embeddings.shape[0]):
        valid_indices = torch.nonzero(mask[batch_idx] < 0.5, as_tuple=False).flatten()
        if valid_indices.numel() == 0:
            outputs.append(frame_embeddings[batch_idx, 0])
            continue
        chosen = valid_indices[-recent:]
        outputs.append(frame_embeddings[batch_idx, chosen].mean(dim=0))
    return torch.stack(outputs, dim=0)


def _cosine_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return 1.0 - (a * b).sum(dim=-1)


_TEMPORAL_META_INDEX = {name: idx for idx, name in enumerate(TEMPORAL_META_FIELDS)}


def _temporal_meta_value(meta_row: torch.Tensor, key: str) -> float:
    return float(meta_row[_TEMPORAL_META_INDEX[key]].item())


def _build_temporal_sequence_inputs(
    frame_embeddings: torch.Tensor,
    spatial_embeddings: torch.Tensor,
    frame_indices: torch.Tensor,
    mask: torch.Tensor,
    temporal_meta: torch.Tensor,
) -> torch.Tensor:
    """Recreate the TT runtime input layout from batched training clips and stored metadata.

    根据训练批次片段和已存元数据重建 TT 运行时输入布局。
    """
    batch, steps, _ = frame_embeddings.shape
    scalars = frame_embeddings.new_zeros((batch, steps, TEMPORAL_SCALAR_DIM))

    for batch_idx in range(batch):
        prev_frame = None
        center_history: list[tuple[int, float, float]] = []
        for step_idx in range(steps):
            if mask[batch_idx, step_idx] >= 0.5:
                continue
            meta_row = temporal_meta[batch_idx, step_idx]
            frame_idx = int(frame_indices[batch_idx, step_idx].item())
            cx = _temporal_meta_value(meta_row, "center_x")
            cy = _temporal_meta_value(meta_row, "center_y")
            center_history.append((frame_idx, cx, cy))
            vx, vy = velocity_from_history(center_history)
            ax, ay = acceleration_from_history(center_history)
            gap = 0.0 if prev_frame is None or frame_idx < 0 else float(max(frame_idx - prev_frame, 0))
            prev_frame = frame_idx if frame_idx >= 0 else prev_frame

            frame_w = max(_temporal_meta_value(meta_row, "frame_w"), 1.0)
            frame_h = max(_temporal_meta_value(meta_row, "frame_h"), 1.0)
            local_density = _temporal_meta_value(meta_row, "local_density")
            neighbor_count = _temporal_meta_value(meta_row, "neighbor_count")
            border_flag = _temporal_meta_value(meta_row, "border_flag")
            scalar_row = build_temporal_scalar_features(
                vx=vx,
                vy=vy,
                ax=ax,
                ay=ay,
                area=_temporal_meta_value(meta_row, "area_proxy"),
                aspect=_temporal_meta_value(meta_row, "aspect_proxy"),
                reid_quality=float(np.clip(1.0 - 0.45 * border_flag - 0.35 * local_density, 0.10, 1.0)),
                is_crowded=neighbor_count >= 1.0 or local_density >= 0.20,
                is_merged_risk=local_density >= 0.45,
                interpolated=False,
                frame_gap=gap / max(float(steps), 1.0),
                memory_reliability=float(np.clip(1.0 - 0.20 * local_density - 0.15 * border_flag, 0.20, 1.0)),
                x_norm=cx / frame_w,
                y_norm=cy / frame_h,
            )
            scalars[batch_idx, step_idx] = frame_embeddings.new_tensor(scalar_row)

    return torch.cat([frame_embeddings, spatial_embeddings, scalars], dim=-1)


def _build_temporal_subsequence_masks(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prefix_mask = mask.clone()
    suffix_mask = mask.clone()
    valid_clip = torch.zeros((mask.shape[0],), device=mask.device, dtype=torch.float32)

    for batch_idx in range(mask.shape[0]):
        valid_indices = torch.nonzero(mask[batch_idx] < 0.5, as_tuple=False).flatten()
        if valid_indices.numel() < 4:
            continue
        split_idx = max(2, int(valid_indices.numel() // 2))
        prefix_keep = valid_indices[:split_idx]
        suffix_keep = valid_indices[-split_idx:]
        prefix_mask[batch_idx] = 1.0
        suffix_mask[batch_idx] = 1.0
        prefix_mask[batch_idx, prefix_keep] = 0.0
        suffix_mask[batch_idx, suffix_keep] = 0.0
        valid_clip[batch_idx] = 1.0

    return prefix_mask, suffix_mask, valid_clip


def _temporal_token_consistency_loss(
    temporal_tokens: torch.Tensor,
    prefix_tokens: torch.Tensor,
    suffix_tokens: torch.Tensor,
    valid_clip: torch.Tensor,
) -> torch.Tensor:
    if not torch.any(valid_clip > 0.0):
        return temporal_tokens.new_tensor(0.0)
    token_to_prefix = 1.0 - F.cosine_similarity(temporal_tokens, prefix_tokens, dim=-1)
    token_to_suffix = 1.0 - F.cosine_similarity(temporal_tokens, suffix_tokens, dim=-1)
    prefix_to_suffix = 1.0 - F.cosine_similarity(prefix_tokens, suffix_tokens, dim=-1)
    losses = token_to_prefix + token_to_suffix + 0.5 * prefix_to_suffix
    return (losses * valid_clip).sum() / valid_clip.sum().clamp(min=1.0)


def _cross_view_supcon_loss(
    anchors: torch.Tensor,
    positives: torch.Tensor,
    labels: torch.Tensor,
    valid_clip: torch.Tensor,
    *,
    sample_weights: torch.Tensor | None = None,
    temperature: float = 0.10,
) -> torch.Tensor:
    valid = valid_clip > 0.0
    if not torch.any(valid):
        return anchors.new_tensor(0.0)

    anchors = F.normalize(anchors[valid], dim=-1)
    positives = F.normalize(positives[valid], dim=-1)
    labels = labels[valid].view(-1)
    logits = torch.matmul(anchors, positives.T) / max(float(temperature), 1e-6)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    positive_mask = labels[:, None].eq(labels[None, :]).float()
    positive_count = positive_mask.sum(dim=1)
    if not torch.any(positive_count > 0):
        return anchors.new_tensor(0.0)

    log_prob = logits - torch.log(torch.exp(logits).sum(dim=1, keepdim=True) + 1e-8)
    losses = -(positive_mask * log_prob).sum(dim=1) / positive_count.clamp(min=1.0)
    if sample_weights is not None:
        weights = sample_weights[valid].clamp(min=0.0)
        return (losses * weights).sum() / weights.sum().clamp(min=1.0)
    return losses.mean()


def _temporal_predictive_loss(
    frame_embeddings: torch.Tensor,
    labels: torch.Tensor,
    prefix_tokens: torch.Tensor,
    suffix_tokens: torch.Tensor,
    prefix_mask: torch.Tensor,
    suffix_mask: torch.Tensor,
    valid_clip: torch.Tensor,
    clip_difficulty: torch.Tensor,
) -> torch.Tensor:
    if not torch.any(valid_clip > 0.0):
        return frame_embeddings.new_tensor(0.0)

    prefix_targets = _masked_mean(frame_embeddings, prefix_mask)
    suffix_targets = _masked_mean(frame_embeddings, suffix_mask)
    sample_weights = 1.0 + 1.5 * clip_difficulty.clamp(min=0.0, max=1.0)

    prefix_to_future = _cross_view_supcon_loss(
        prefix_tokens,
        suffix_targets,
        labels,
        valid_clip,
        sample_weights=sample_weights,
    )
    suffix_to_history = _cross_view_supcon_loss(
        suffix_tokens,
        prefix_targets,
        labels,
        valid_clip,
        sample_weights=sample_weights,
    )

    prefix_align = 1.0 - F.cosine_similarity(prefix_tokens, suffix_targets, dim=-1)
    suffix_align = 1.0 - F.cosine_similarity(suffix_tokens, prefix_targets, dim=-1)
    alignment = 0.5 * (prefix_align + suffix_align)
    alignment = (alignment * sample_weights * valid_clip).sum() / (sample_weights * valid_clip).sum().clamp(min=1.0)
    return 0.4 * (prefix_to_future + suffix_to_history) + 0.2 * alignment


def _last_valid_temporal_meta(temporal_meta: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    outputs = []
    for batch_idx in range(temporal_meta.shape[0]):
        valid_indices = torch.nonzero(mask[batch_idx] < 0.5, as_tuple=False).flatten()
        if valid_indices.numel() == 0:
            outputs.append(temporal_meta[batch_idx, 0])
        else:
            outputs.append(temporal_meta[batch_idx, int(valid_indices[-1].item())])
    return torch.stack(outputs, dim=0)


def _mean_valid_temporal_meta(
    temporal_meta: torch.Tensor,
    mask: torch.Tensor,
    *,
    recent: int | None = None,
) -> torch.Tensor:
    outputs = []
    for batch_idx in range(temporal_meta.shape[0]):
        valid_indices = torch.nonzero(mask[batch_idx] < 0.5, as_tuple=False).flatten()
        if valid_indices.numel() == 0:
            outputs.append(temporal_meta[batch_idx, 0])
            continue
        chosen = valid_indices[-recent:] if recent is not None else valid_indices
        outputs.append(temporal_meta[batch_idx, chosen].mean(dim=0))
    return torch.stack(outputs, dim=0)


def _association_terms_from_meta(
    anchor_meta_mean: torch.Tensor,
    anchor_meta_last: torch.Tensor,
    det_meta_last: torch.Tensor,
) -> tuple[float, float, float]:
    area_mean = max(_temporal_meta_value(anchor_meta_mean, "area_proxy"), 1.0)
    aspect_mean = max(_temporal_meta_value(anchor_meta_mean, "aspect_proxy"), 1e-3)
    area_last = _temporal_meta_value(det_meta_last, "area_proxy")
    aspect_last = _temporal_meta_value(det_meta_last, "aspect_proxy")
    shape_cost = float(
        np.clip(
            0.5 * abs(area_mean - area_last) / area_mean
            + 0.5 * abs(aspect_mean - aspect_last) / aspect_mean,
            0.0,
            1.5,
        )
    )

    frame_w = max(_temporal_meta_value(det_meta_last, "frame_w"), 1.0)
    frame_h = max(_temporal_meta_value(det_meta_last, "frame_h"), 1.0)
    diag = max((frame_w ** 2 + frame_h ** 2) ** 0.5, 1.0)
    dx = _temporal_meta_value(anchor_meta_last, "center_x") - _temporal_meta_value(det_meta_last, "center_x")
    dy = _temporal_meta_value(anchor_meta_last, "center_y") - _temporal_meta_value(det_meta_last, "center_y")
    motion_cost = float(np.clip((dx * dx + dy * dy) ** 0.5 / diag, 0.0, 1.5))
    return shape_cost, motion_cost, motion_cost


def _build_association_targets(
    frame_embeddings: torch.Tensor,
    spatial_embeddings: torch.Tensor,
    temporal_tokens: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    frame_indices: torch.Tensor,
    temporal_meta: torch.Tensor,
    *,
    hard_negative_topk: int = 1,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Synthesize positive and hard-negative pair features for training the association head.

    合成正样本和困难负样本的成对特征，用于训练关联头。
    """
    batch = frame_embeddings.shape[0]
    if batch == 0:
        return None, None, None

    long_term = _masked_mean(frame_embeddings, mask)
    short_term = _recent_mean_embeddings(frame_embeddings, mask, recent=3)
    last_embeddings = _last_valid_embeddings(frame_embeddings, mask)
    spatial_long_term = _masked_mean(spatial_embeddings, mask)
    spatial_last = _last_valid_embeddings(spatial_embeddings, mask)
    mean_temporal_meta = _mean_valid_temporal_meta(temporal_meta, mask)
    recent_temporal_meta = _mean_valid_temporal_meta(temporal_meta, mask, recent=3)
    last_temporal_meta = _last_valid_temporal_meta(temporal_meta, mask)

    feature_rows = []
    match_targets = []
    switch_targets = []
    topk = max(int(hard_negative_topk), 1)

    for batch_idx in range(batch):
        last_meta = last_temporal_meta[batch_idx]
        local_density = _temporal_meta_value(last_meta, "local_density")
        border_flag = _temporal_meta_value(last_meta, "border_flag")
        shape_cost, motion_cost, kf_cost = _association_terms_from_meta(
            mean_temporal_meta[batch_idx],
            recent_temporal_meta[batch_idx],
            last_meta,
        )
        positive_row = build_association_feature_vector(
            appearance_long=float(_cosine_distance(long_term[batch_idx:batch_idx + 1], last_embeddings[batch_idx:batch_idx + 1]).item()),
            appearance_short=float(_cosine_distance(short_term[batch_idx:batch_idx + 1], last_embeddings[batch_idx:batch_idx + 1]).item()),
            temporal_distance=float(_cosine_distance(temporal_tokens[batch_idx:batch_idx + 1], last_embeddings[batch_idx:batch_idx + 1]).item()),
            identity_distance=float(_cosine_distance(long_term[batch_idx:batch_idx + 1], last_embeddings[batch_idx:batch_idx + 1]).item()),
            spatial_distance=float(_cosine_distance(spatial_long_term[batch_idx:batch_idx + 1], spatial_last[batch_idx:batch_idx + 1]).item()),
            shape_cost=shape_cost,
            motion_cost=motion_cost,
            kf_cost=kf_cost,
            direction_cost=0.05,
            det_reid_quality=float(np.clip(1.0 - 0.45 * border_flag - 0.35 * local_density, 0.10, 1.0)),
            memory_reliability=float(np.clip(1.0 - 0.20 * local_density, 0.30, 1.0)),
            can_update_reid=True,
            is_crowded=_temporal_meta_value(last_meta, "neighbor_count") >= 1.0 or local_density >= 0.20,
            is_merged_risk=local_density >= 0.45,
            normalized_hits=float(np.clip(torch.sum(mask[batch_idx] < 0.5).item() / 10.0, 0.0, 1.0)),
            normalized_missed=0.0,
            has_identity_slot=True,
            switch_risk_hint=float(local_density),
            local_density=float(local_density),
            border_risk=float(border_flag),
        )
        feature_rows.append(positive_row.tolist())
        match_targets.append(1.0)
        switch_targets.append(0.0)

        negative_indices = torch.nonzero(labels != labels[batch_idx], as_tuple=False).flatten()
        if negative_indices.numel() == 0:
            continue

        candidates: list[tuple[float, int]] = []
        anchor_frame = int(torch.max(frame_indices[batch_idx][mask[batch_idx] < 0.5]).item())
        for neg_idx in negative_indices.tolist():
            appearance_distance = float(_cosine_distance(long_term[batch_idx:batch_idx + 1], last_embeddings[neg_idx:neg_idx + 1]).item())
            neg_frame = int(torch.max(frame_indices[neg_idx][mask[neg_idx] < 0.5]).item())
            frame_gap = min(abs(anchor_frame - neg_frame) / 32.0, 1.0)
            neg_last_meta = last_temporal_meta[neg_idx]
            frame_w = max(_temporal_meta_value(neg_last_meta, "frame_w"), 1.0)
            frame_h = max(_temporal_meta_value(neg_last_meta, "frame_h"), 1.0)
            diag = max((frame_w ** 2 + frame_h ** 2) ** 0.5, 1.0)
            spatial_gap = (
                (
                    (_temporal_meta_value(last_meta, "center_x") - _temporal_meta_value(neg_last_meta, "center_x")) ** 2
                    + (_temporal_meta_value(last_meta, "center_y") - _temporal_meta_value(neg_last_meta, "center_y")) ** 2
                ) ** 0.5
            ) / diag
            candidates.append((appearance_distance + 0.15 * frame_gap + 0.20 * spatial_gap, neg_idx))

        candidates.sort(key=lambda item: item[0])
        for confusion_score, hardest_idx in candidates[:topk]:
            neg_last_meta = last_temporal_meta[hardest_idx]
            neg_local_density = _temporal_meta_value(neg_last_meta, "local_density")
            neg_border_flag = _temporal_meta_value(neg_last_meta, "border_flag")
            shape_cost, motion_cost, kf_cost = _association_terms_from_meta(
                mean_temporal_meta[batch_idx],
                recent_temporal_meta[batch_idx],
                neg_last_meta,
            )
            negative_row = build_association_feature_vector(
                appearance_long=float(_cosine_distance(long_term[batch_idx:batch_idx + 1], last_embeddings[hardest_idx:hardest_idx + 1]).item()),
                appearance_short=float(_cosine_distance(short_term[batch_idx:batch_idx + 1], last_embeddings[hardest_idx:hardest_idx + 1]).item()),
                temporal_distance=float(_cosine_distance(temporal_tokens[batch_idx:batch_idx + 1], last_embeddings[hardest_idx:hardest_idx + 1]).item()),
                identity_distance=float(_cosine_distance(long_term[batch_idx:batch_idx + 1], last_embeddings[hardest_idx:hardest_idx + 1]).item()),
                spatial_distance=float(_cosine_distance(spatial_long_term[batch_idx:batch_idx + 1], spatial_last[hardest_idx:hardest_idx + 1]).item()),
                shape_cost=shape_cost,
                motion_cost=motion_cost,
                kf_cost=kf_cost,
                direction_cost=0.20,
                det_reid_quality=float(np.clip(1.0 - 0.45 * neg_border_flag - 0.35 * neg_local_density, 0.10, 1.0)),
                memory_reliability=float(np.clip(0.85 - 0.20 * neg_local_density, 0.20, 1.0)),
                can_update_reid=True,
                is_crowded=_temporal_meta_value(neg_last_meta, "neighbor_count") >= 1.0 or neg_local_density >= 0.20,
                is_merged_risk=neg_local_density >= 0.45,
                normalized_hits=float(np.clip(torch.sum(mask[batch_idx] < 0.5).item() / 10.0, 0.0, 1.0)),
                normalized_missed=0.10,
                has_identity_slot=True,
                switch_risk_hint=float(max(neg_local_density, 1.0 - confusion_score)),
                local_density=float(neg_local_density),
                border_risk=float(neg_border_flag),
            )
            feature_rows.append(negative_row.tolist())
            match_targets.append(0.0)
            switch_targets.append(1.0)

    if not feature_rows:
        return None, None, None

    features = frame_embeddings.new_tensor(feature_rows, dtype=torch.float32)
    match_tensor = frame_embeddings.new_tensor(match_targets, dtype=torch.float32)
    switch_tensor = frame_embeddings.new_tensor(switch_targets, dtype=torch.float32)
    return features, match_tensor, switch_tensor


def _run_epoch(
    appearance_model,
    spatial_model,
    identity_memory,
    temporal_model,
    association_head,
    loader: DataLoader,
    losses: dict,
    *,
    optimizer=None,
    device: str = "cpu",
    ce_weight: float = 1.0,
    triplet_weight: float = 0.5,
    supcon_weight: float = 0.5,
    temporal_consistency_weight: float = 0.25,
    temporal_token_consistency_weight: float = 0.35,
    temporal_predictive_weight: float = 0.45,
    association_bce_weight: float = 0.5,
    identity_frame_supcon_weight: float = 0.30,
    identity_frame_triplet_weight: float = 0.20,
    identity_anchor_weight: float = 0.20,
    identity_center_margin_weight: float = 0.10,
    temporal_center_margin_weight: float = 0.05,
    representation_center_margin: float = 0.12,
    hard_negative_topk: int = 1,
    max_batches: int | None = None,
) -> dict:
    """Run one training or validation epoch across appearance, SC, IM, TT, and association heads.

    在外观、SC、IM、TT 和关联头上运行一个训练或验证 epoch。
    """
    training = optimizer is not None
    appearance_trainable = _module_trainable(appearance_model)
    spatial_trainable = _module_trainable(spatial_model)
    identity_trainable = _module_trainable(identity_memory)
    temporal_trainable = _module_trainable(temporal_model)
    association_trainable = _module_trainable(association_head)
    appearance_model.train(training and appearance_trainable)
    spatial_model.train(training and spatial_trainable)
    identity_memory.train(training and identity_trainable)
    temporal_model.train(training and temporal_trainable)
    association_head.train(training and association_trainable)

    total_loss = 0.0
    total_acc = 0.0
    total_assoc_acc = 0.0
    total_token_loss = 0.0
    total_future_loss = 0.0
    total_identity_loss = 0.0
    total_center_loss = 0.0
    total_anchor_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images = _to_tensor(batch["images"], device=device, dtype=torch.float32)
        mask = _to_tensor(batch["mask"], device=device, dtype=torch.float32)
        labels = _to_tensor(batch["label"], device=device, dtype=torch.long)
        frame_indices = _to_tensor(batch["frame_indices"], device=device, dtype=torch.long)
        spatial_inputs = _to_tensor(batch["spatial_inputs"], device=device, dtype=torch.float32)
        temporal_meta = _to_tensor(batch["temporal_meta"], device=device, dtype=torch.float32)
        clip_difficulty = _to_tensor(batch.get("clip_difficulty", np.zeros((images.shape[0],), dtype=np.float32)), device=device, dtype=torch.float32)

        batch_size, steps, channels, height, width = images.shape
        flat_images = images.view(batch_size * steps, channels, height, width)
        flat_embeddings, flat_logits = appearance_model(flat_images)
        flat_spatial_inputs = spatial_inputs.view(batch_size * steps, -1)
        flat_spatial_embeddings = spatial_model(flat_spatial_inputs)
        flat_identity_embeddings = identity_memory(flat_embeddings, flat_spatial_embeddings)
        frame_embeddings = flat_identity_embeddings.view(batch_size, steps, -1)
        spatial_embeddings = flat_spatial_embeddings.view(batch_size, steps, -1)
        frame_logits = None if flat_logits is None else flat_logits.view(batch_size, steps, -1)

        seq_inputs = _build_temporal_sequence_inputs(
            frame_embeddings,
            spatial_embeddings,
            frame_indices,
            mask,
            temporal_meta,
        )
        prefix_mask, suffix_mask, valid_temporal_clips = _build_temporal_subsequence_masks(mask)
        if training and not temporal_trainable:
            with torch.no_grad():
                temporal_tokens = temporal_model(seq_inputs, mask)
                prefix_tokens = temporal_model(seq_inputs, prefix_mask)
                suffix_tokens = temporal_model(seq_inputs, suffix_mask)
            temporal_tokens = temporal_tokens.detach()
            prefix_tokens = prefix_tokens.detach()
            suffix_tokens = suffix_tokens.detach()
        else:
            temporal_tokens = temporal_model(seq_inputs, mask)
            prefix_tokens = temporal_model(seq_inputs, prefix_mask)
            suffix_tokens = temporal_model(seq_inputs, suffix_mask)
        valid_temporal_clips = valid_temporal_clips * (_masked_count(prefix_mask) >= 2).float() * (_masked_count(suffix_mask) >= 2).float()

        valid_mask = mask < 0.5
        valid_logits = None
        valid_labels = None
        if frame_logits is not None and torch.any(valid_mask):
            valid_logits = frame_logits[valid_mask]
            valid_labels = labels.unsqueeze(1).expand(-1, steps)[valid_mask]
        valid_identity_embeddings = frame_embeddings[valid_mask] if torch.any(valid_mask) else None
        valid_appearance_embeddings = flat_embeddings.view(batch_size, steps, -1)[valid_mask] if torch.any(valid_mask) else None

        ce_loss = frame_embeddings.new_tensor(0.0)
        if losses.get("ce") is not None and valid_logits is not None and valid_labels is not None and valid_logits.numel() > 0:
            ce_loss = losses["ce"](valid_logits, valid_labels)

        triplet_loss = frame_embeddings.new_tensor(0.0)
        if losses.get("triplet") is not None and temporal_trainable:
            triplet_loss = losses["triplet"](temporal_tokens, labels)

        supcon_loss = frame_embeddings.new_tensor(0.0)
        if losses.get("supcon") is not None and temporal_trainable:
            supcon_loss = losses["supcon"](temporal_tokens, labels)

        identity_triplet_loss = frame_embeddings.new_tensor(0.0)
        identity_supcon_loss = frame_embeddings.new_tensor(0.0)
        identity_anchor_loss = frame_embeddings.new_tensor(0.0)
        identity_center_loss = frame_embeddings.new_tensor(0.0)
        temporal_center_loss = frame_embeddings.new_tensor(0.0)
        if valid_identity_embeddings is not None and valid_labels is not None and valid_identity_embeddings.shape[0] > 1:
            if losses.get("triplet") is not None:
                identity_triplet_loss = losses["triplet"](valid_identity_embeddings, valid_labels)
            if losses.get("supcon") is not None:
                identity_supcon_loss = losses["supcon"](valid_identity_embeddings, valid_labels)
            if valid_appearance_embeddings is not None:
                identity_anchor_loss = _anchor_alignment_loss(valid_identity_embeddings, valid_appearance_embeddings)
            identity_center_loss = _class_center_margin_loss(
                valid_identity_embeddings,
                valid_labels,
                margin=representation_center_margin,
            )
        if temporal_trainable:
            temporal_center_loss = _class_center_margin_loss(
                temporal_tokens,
                labels,
                margin=representation_center_margin,
            )

        temporal_loss = frame_embeddings.new_tensor(0.0)
        if losses.get("temporal") is not None and temporal_trainable:
            temporal_loss = losses["temporal"](frame_embeddings, mask)
        temporal_token_loss = _temporal_token_consistency_loss(
            temporal_tokens,
            prefix_tokens,
            suffix_tokens,
            valid_temporal_clips,
        ) if temporal_trainable else frame_embeddings.new_tensor(0.0)
        temporal_future_loss = _temporal_predictive_loss(
            frame_embeddings,
            labels,
            prefix_tokens,
            suffix_tokens,
            prefix_mask,
            suffix_mask,
            valid_temporal_clips,
            clip_difficulty,
        ) if temporal_trainable else frame_embeddings.new_tensor(0.0)

        association_loss = frame_embeddings.new_tensor(0.0)
        assoc_accuracy = 0.0
        pair_features, match_targets, switch_targets = _build_association_targets(
            frame_embeddings,
            spatial_embeddings,
            temporal_tokens,
            labels,
            mask,
            frame_indices,
            temporal_meta,
            hard_negative_topk=hard_negative_topk,
        )
        if pair_features is not None and losses.get("bce") is not None:
            match_logits, switch_logits = association_head(pair_features)
            match_logits = match_logits.squeeze(-1)
            switch_logits = switch_logits.squeeze(-1)
            match_loss = losses["bce"](match_logits, match_targets)
            switch_loss = losses["bce"](switch_logits, switch_targets)
            association_loss = 0.5 * (match_loss + switch_loss)

            match_preds = (torch.sigmoid(match_logits) >= 0.5).float()
            switch_preds = (torch.sigmoid(switch_logits) >= 0.5).float()
            match_acc = (match_preds == match_targets).float().mean().item()
            switch_acc = (switch_preds == switch_targets).float().mean().item()
            assoc_accuracy = float(0.5 * (match_acc + switch_acc))

        loss = (
            ce_weight * ce_loss
            + triplet_weight * triplet_loss
            + supcon_weight * supcon_loss
            + temporal_consistency_weight * temporal_loss
            + temporal_token_consistency_weight * temporal_token_loss
            + temporal_predictive_weight * temporal_future_loss
            + association_bce_weight * association_loss
            + identity_frame_triplet_weight * identity_triplet_loss
            + identity_frame_supcon_weight * identity_supcon_loss
            + identity_anchor_weight * identity_anchor_loss
            + identity_center_margin_weight * identity_center_loss
            + temporal_center_margin_weight * temporal_center_loss
        )

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item())
        total_acc += _compute_accuracy(valid_logits, valid_labels) if valid_logits is not None and valid_labels is not None else 0.0
        total_assoc_acc += assoc_accuracy
        total_token_loss += float(temporal_token_loss.item())
        total_future_loss += float(temporal_future_loss.item())
        total_identity_loss += float((identity_triplet_loss + identity_supcon_loss).item())
        total_center_loss += float((identity_center_loss + temporal_center_loss).item())
        total_anchor_loss += float(identity_anchor_loss.item())
        num_batches += 1

    return {
        "loss": total_loss / max(num_batches, 1),
        "acc": total_acc / max(num_batches, 1),
        "assoc_acc": total_assoc_acc / max(num_batches, 1),
        "token_loss": total_token_loss / max(num_batches, 1),
        "future_loss": total_future_loss / max(num_batches, 1),
        "identity_loss": total_identity_loss / max(num_batches, 1),
        "center_loss": total_center_loss / max(num_batches, 1),
        "anchor_loss": total_anchor_loss / max(num_batches, 1),
    }


def train_encoder(cfg=None) -> dict:
    """Train the full appearance/SC/IM/TT/association bundle and save the best checkpoint.

    训练完整的外观、SC、IM、TT 和关联模块组合，并保存最佳检查点。
    """
    if torch is None or DataLoader is None or F is None:
        raise ImportError("torch is required to train the appearance encoder.")
    cfg = get_config() if cfg is None else cfg
    device = cfg.device
    checkpoint_path = Path(cfg.training.checkpoint_path or cfg.feature.encoder_checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(cfg.paths.logs / "train_encoder.log")

    prep_summary = ensure_reid_dataset(cfg)
    logger.info(
        "reid_data_root=%s prepared=%s reason=%s",
        prep_summary.get("data_root"),
        prep_summary.get("prepared"),
        prep_summary.get("reason"),
    )

    train_samples, val_samples, class_to_idx = build_reid_clip_splits(
        cfg.training.data_root,
        train_subdir=cfg.training.train_subdir,
        val_subdir=cfg.training.val_subdir,
        val_split=cfg.training.val_split,
        min_images_per_identity=cfg.training.min_images_per_identity,
        seed=cfg.training.random_seed,
        clip_len=cfg.training.clip_len,
        min_clip_len=2,
        clip_stride=cfg.training.clip_stride,
        metadata_root=cfg.training.data_root,
        temporal_hard_mining=cfg.training.temporal_hard_mining,
        temporal_hard_threshold=cfg.training.temporal_hard_threshold,
        temporal_hard_oversample=cfg.training.temporal_hard_oversample,
    )
    if not train_samples:
        raise RuntimeError("No clip-based training samples found for identity encoder.")

    train_dataset = ReIDClipDataset(
        train_samples,
        backend="cnn",
        crop_size=cfg.feature.crop_size,
        clip_len=cfg.training.clip_len,
        metadata_root=cfg.training.data_root,
    )
    val_dataset = (
        ReIDClipDataset(
            val_samples,
            backend="cnn",
            crop_size=cfg.feature.crop_size,
            clip_len=cfg.training.clip_len,
            metadata_root=cfg.training.data_root,
        )
        if val_samples
        else None
    )

    batch_size = max(cfg.training.batch_size, cfg.training.identities_per_batch * cfg.training.samples_per_identity)
    train_sampler = IdentityBalancedSampler(
        train_dataset,
        identities_per_batch=cfg.training.identities_per_batch,
        samples_per_identity=cfg.training.samples_per_identity,
        hard_sample_ratio=cfg.training.hard_sample_ratio,
        seed=cfg.training.random_seed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=cfg.training.num_workers,
        drop_last=False,
    )
    val_loader = None
    if val_dataset is not None and len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=cfg.training.num_workers,
            drop_last=False,
        )

    appearance_model = build_appearance_encoder(
        embedding_dim=cfg.feature.embedding_dim,
        width=cfg.feature.cnn_width,
        dropout=cfg.feature.cnn_dropout,
        num_classes=max(class_to_idx.values(), default=-1) + 1,
    ).to(device)
    spatial_model = build_spacial_context(
        input_dim=SPACIAL_INPUT_DIM,
        embedding_dim=cfg.feature.embedding_dim,
        hidden_dim=cfg.feature.spatial_hidden_dim,
        dropout=cfg.feature.spatial_dropout,
    ).to(device)
    identity_memory = build_identity_memory(
        embedding_dim=cfg.feature.embedding_dim,
        hidden_dim=cfg.feature.identity_hidden_dim,
        dropout=cfg.feature.identity_dropout,
    ).to(device)
    temporal_model = build_trajectory_temporal(
        input_dim=cfg.feature.embedding_dim * 2 + TEMPORAL_SCALAR_DIM,
        token_dim=cfg.feature.embedding_dim,
        hidden_dim=cfg.feature.temporal_hidden_dim,
        num_layers=cfg.feature.temporal_num_layers,
        num_heads=cfg.feature.temporal_num_heads,
        dropout=cfg.feature.temporal_dropout,
    ).to(device)
    association_head = build_association_head(
        input_dim=ASSOCIATION_FEATURE_DIM,
        hidden_dim=cfg.association.association_hidden_dim,
    ).to(device)

    def _set_trainable(module, trainable: bool) -> None:
        for param in module.parameters():
            param.requires_grad = bool(trainable)

    if cfg.training.resume_checkpoint:
        resume_path = Path(cfg.training.resume_checkpoint)
        if resume_path.exists():
            checkpoint = torch.load(resume_path, map_location=device)
            appearance_model.load_state_dict(
                checkpoint.get("appearance_state_dict", checkpoint.get("state_dict", checkpoint)),
                strict=False,
            )
            spatial_state = checkpoint.get("spacial_context_state_dict")
            if spatial_state:
                spatial_model.load_state_dict(spatial_state, strict=False)
            identity_state = checkpoint.get("identity_memory_state_dict")
            if identity_state:
                identity_memory.load_state_dict(identity_state, strict=False)
            temporal_state = checkpoint.get("trajectory_temporal_state_dict")
            if temporal_state:
                temporal_model.load_state_dict(temporal_state, strict=False)
            association_state = checkpoint.get("association_head_state_dict")
            if association_state:
                association_head.load_state_dict(association_state, strict=False)

    _set_trainable(appearance_model, not cfg.training.freeze_appearance_encoder)
    _set_trainable(spatial_model, not cfg.training.freeze_spacial_context)
    _set_trainable(identity_memory, not cfg.training.freeze_identity_memory)
    _set_trainable(temporal_model, not cfg.training.freeze_trajectory_temporal)
    _set_trainable(association_head, not cfg.training.freeze_association_head)

    trainable_params = [
        param
        for module in (appearance_model, spatial_model, identity_memory, temporal_model, association_head)
        for param in module.parameters()
        if param.requires_grad
    ]
    if not trainable_params:
        raise RuntimeError("No trainable parameters left after applying training freeze settings.")

    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(cfg.training.epochs, 1),
    )
    losses = build_loss(
        "combined",
        label_smoothing=cfg.training.label_smoothing,
        triplet_margin=cfg.training.triplet_margin,
    )

    best_score = float("-inf")
    history: list[dict] = []
    for epoch in range(cfg.training.epochs):
        train_sampler.set_epoch(epoch)
        train_metrics = _run_epoch(
            appearance_model,
            spatial_model,
            identity_memory,
            temporal_model,
            association_head,
            train_loader,
            losses,
            optimizer=optimizer,
            device=device,
            ce_weight=cfg.training.ce_weight,
            triplet_weight=cfg.training.triplet_weight,
            supcon_weight=cfg.training.supcon_weight,
            temporal_consistency_weight=cfg.training.temporal_consistency_weight,
            temporal_token_consistency_weight=cfg.training.temporal_token_consistency_weight,
            temporal_predictive_weight=cfg.training.temporal_predictive_weight,
            association_bce_weight=cfg.training.association_bce_weight,
            identity_frame_supcon_weight=cfg.training.identity_frame_supcon_weight,
            identity_frame_triplet_weight=cfg.training.identity_frame_triplet_weight,
            identity_anchor_weight=cfg.training.identity_anchor_weight,
            identity_center_margin_weight=cfg.training.identity_center_margin_weight,
            temporal_center_margin_weight=cfg.training.temporal_center_margin_weight,
            representation_center_margin=cfg.training.representation_center_margin,
            hard_negative_topk=cfg.training.association_hard_negative_topk,
            max_batches=cfg.training.max_batches_per_epoch,
        )
        val_metrics = (
            _run_epoch(
                appearance_model,
                spatial_model,
                identity_memory,
                temporal_model,
                association_head,
                val_loader,
                losses,
                optimizer=None,
                device=device,
                ce_weight=cfg.training.ce_weight,
                triplet_weight=cfg.training.triplet_weight,
                supcon_weight=cfg.training.supcon_weight,
                temporal_consistency_weight=cfg.training.temporal_consistency_weight,
                temporal_token_consistency_weight=cfg.training.temporal_token_consistency_weight,
                temporal_predictive_weight=cfg.training.temporal_predictive_weight,
                association_bce_weight=cfg.training.association_bce_weight,
                identity_frame_supcon_weight=cfg.training.identity_frame_supcon_weight,
                identity_frame_triplet_weight=cfg.training.identity_frame_triplet_weight,
                identity_anchor_weight=cfg.training.identity_anchor_weight,
                identity_center_margin_weight=cfg.training.identity_center_margin_weight,
                temporal_center_margin_weight=cfg.training.temporal_center_margin_weight,
                representation_center_margin=cfg.training.representation_center_margin,
                hard_negative_topk=cfg.training.association_hard_negative_topk,
                max_batches=cfg.training.max_batches_per_epoch,
            )
            if val_loader is not None
            else {
                "loss": train_metrics["loss"],
                "acc": train_metrics["acc"],
                "assoc_acc": train_metrics["assoc_acc"],
                "token_loss": train_metrics["token_loss"],
                "future_loss": train_metrics["future_loss"],
                "identity_loss": train_metrics["identity_loss"],
                "center_loss": train_metrics["center_loss"],
                "anchor_loss": train_metrics["anchor_loss"],
            }
        )
        scheduler.step()

        score = val_metrics["acc"] + 0.35 * val_metrics["assoc_acc"] - 0.1 * val_metrics["loss"]
        row = {
            "epoch": epoch + 1,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "train_assoc_acc": train_metrics["assoc_acc"],
            "train_token_loss": train_metrics["token_loss"],
            "train_future_loss": train_metrics["future_loss"],
            "train_identity_loss": train_metrics["identity_loss"],
            "train_center_loss": train_metrics["center_loss"],
            "train_anchor_loss": train_metrics["anchor_loss"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "val_assoc_acc": val_metrics["assoc_acc"],
            "val_token_loss": val_metrics["token_loss"],
            "val_future_loss": val_metrics["future_loss"],
            "val_identity_loss": val_metrics["identity_loss"],
            "val_center_loss": val_metrics["center_loss"],
            "val_anchor_loss": val_metrics["anchor_loss"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        logger.info(
            "epoch=%d train_loss=%.4f train_acc=%.4f train_assoc=%.4f train_token=%.4f train_future=%.4f train_identity=%.4f train_center=%.4f train_anchor=%.4f val_loss=%.4f val_acc=%.4f val_assoc=%.4f val_token=%.4f val_future=%.4f val_identity=%.4f val_center=%.4f val_anchor=%.4f",
            epoch + 1,
            train_metrics["loss"],
            train_metrics["acc"],
            train_metrics["assoc_acc"],
            train_metrics["token_loss"],
            train_metrics["future_loss"],
            train_metrics["identity_loss"],
            train_metrics["center_loss"],
            train_metrics["anchor_loss"],
            val_metrics["loss"],
            val_metrics["acc"],
            val_metrics["assoc_acc"],
            val_metrics["token_loss"],
            val_metrics["future_loss"],
            val_metrics["identity_loss"],
            val_metrics["center_loss"],
            val_metrics["anchor_loss"],
        )

        if score > best_score or not cfg.training.save_best_only:
            best_score = score
            torch.save(
                {
                    "state_dict": appearance_model.state_dict(),
                    "appearance_state_dict": appearance_model.state_dict(),
                    "spacial_context_state_dict": spatial_model.state_dict(),
                    "identity_memory_state_dict": identity_memory.state_dict(),
                    "trajectory_temporal_state_dict": temporal_model.state_dict(),
                    "association_head_state_dict": association_head.state_dict(),
                    "embedding_dim": cfg.feature.embedding_dim,
                    "cnn_width": cfg.feature.cnn_width,
                    "cnn_dropout": cfg.feature.cnn_dropout,
                    "identity_hidden_dim": cfg.feature.identity_hidden_dim,
                    "identity_dropout": cfg.feature.identity_dropout,
                    "spatial_hidden_dim": cfg.feature.spatial_hidden_dim,
                    "spatial_dropout": cfg.feature.spatial_dropout,
                    "crop_size": list(cfg.feature.crop_size),
                    "history_len": cfg.feature.history_len,
                    "short_term_window": cfg.feature.short_term_window,
                    "temporal_hidden_dim": cfg.feature.temporal_hidden_dim,
                    "temporal_num_layers": cfg.feature.temporal_num_layers,
                    "temporal_num_heads": cfg.feature.temporal_num_heads,
                    "temporal_dropout": cfg.feature.temporal_dropout,
                    "association_hidden_dim": cfg.association.association_hidden_dim,
                    "clip_len": cfg.training.clip_len,
                    "backend": "cnn_bundle",
                    "num_classes": max(class_to_idx.values(), default=-1) + 1,
                    "class_to_idx": class_to_idx,
                },
                checkpoint_path,
            )

    history_path = checkpoint_path.with_name(f"{checkpoint_path.stem}_history.csv")
    with history_path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    for handler in list(logger.handlers):
        handler.flush()
        handler.close()
        logger.removeHandler(handler)

    return {
        "data_root": str(cfg.training.data_root),
        "reid_dataset": prep_summary,
        "checkpoint_path": str(checkpoint_path),
        "history_path": str(history_path),
        "num_classes": max(class_to_idx.values(), default=-1) + 1,
        "num_train_clips": len(train_dataset),
        "num_val_clips": 0 if val_dataset is None else len(val_dataset),
        "best_score": best_score,
    }


def train_encoder_main() -> None:
    """CLI entrypoint for bundle training that prints a compact summary on completion.

    组合训练的 CLI 入口，训练完成后打印简洁摘要。
    """
    summary = train_encoder()
    print(summary)


if __name__ == "__main__":
    train_encoder_main()
