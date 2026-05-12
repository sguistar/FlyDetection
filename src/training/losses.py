from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class BatchHardTripletLoss(nn.Module if nn is not None else object):
    def __init__(self, margin: float = 0.3) -> None:
        if nn is None or torch is None or F is None:
            raise ImportError("torch is required to build training losses.")
        super().__init__()
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if embeddings.ndim != 2:
            raise ValueError("BatchHardTripletLoss expects embeddings with shape [N, D].")
        distances = torch.cdist(embeddings, embeddings, p=2)
        labels = labels.view(-1)
        same_identity = labels[:, None].eq(labels[None, :])
        eye_mask = torch.eye(labels.shape[0], device=labels.device, dtype=torch.bool)

        positive_mask = same_identity & (~eye_mask)
        negative_mask = ~same_identity

        if not torch.any(positive_mask) or not torch.any(negative_mask):
            return embeddings.new_tensor(0.0)

        positive_distances = distances.masked_fill(~positive_mask, float("-inf"))
        hardest_positive = positive_distances.max(dim=1).values

        negative_distances = distances.masked_fill(~negative_mask, float("inf"))
        hardest_negative = negative_distances.min(dim=1).values

        valid = torch.isfinite(hardest_positive) & torch.isfinite(hardest_negative)
        if not torch.any(valid):
            return embeddings.new_tensor(0.0)
        losses = F.relu(hardest_positive[valid] - hardest_negative[valid] + self.margin)
        return losses.mean() if losses.numel() > 0 else embeddings.new_tensor(0.0)


class SupervisedContrastiveLoss(nn.Module if nn is not None else object):
    def __init__(self, temperature: float = 0.10) -> None:
        if nn is None or torch is None or F is None:
            raise ImportError("torch is required to build training losses.")
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError("SupervisedContrastiveLoss expects features with shape [N, D].")
        if features.shape[0] <= 1:
            return features.new_tensor(0.0)

        labels = labels.view(-1)
        features = F.normalize(features, dim=1)
        logits = torch.matmul(features, features.T) / max(self.temperature, 1e-6)
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        identity_mask = labels[:, None].eq(labels[None, :]).float()
        eye = torch.eye(labels.shape[0], device=labels.device, dtype=torch.float32)
        positive_mask = identity_mask - eye
        denominator_mask = 1.0 - eye

        exp_logits = torch.exp(logits) * denominator_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

        positive_count = positive_mask.sum(dim=1)
        valid = positive_count > 0
        if not torch.any(valid):
            return features.new_tensor(0.0)

        mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / positive_count.clamp(min=1.0)
        return (-mean_log_prob_pos[valid]).mean()


class TemporalConsistencyLoss(nn.Module if nn is not None else object):
    def __init__(self) -> None:
        if nn is None or torch is None:
            raise ImportError("torch is required to build training losses.")
        super().__init__()

    def forward(self, frame_embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if frame_embeddings.ndim != 3:
            raise ValueError("TemporalConsistencyLoss expects frame embeddings with shape [B, T, D].")
        if mask.ndim != 2:
            raise ValueError("TemporalConsistencyLoss expects a mask with shape [B, T].")
        if frame_embeddings.shape[1] < 2:
            return frame_embeddings.new_tensor(0.0)

        normalized = F.normalize(frame_embeddings, dim=-1)
        deltas = 1.0 - (normalized[:, 1:] * normalized[:, :-1]).sum(dim=-1)
        valid = (1.0 - mask[:, 1:]) * (1.0 - mask[:, :-1])
        if not torch.any(valid > 0.0):
            return frame_embeddings.new_tensor(0.0)
        return (deltas * valid).sum() / valid.sum().clamp(min=1.0)


def build_loss(
    name: str = "combined",
    *,
    label_smoothing: float = 0.0,
    triplet_margin: float = 0.3,
    supcon_temperature: float = 0.10,
):
    if nn is None or torch is None:
        raise ImportError("torch is required to build identity training losses.")
    name = name.lower()
    if name == "ce":
        return {"ce": nn.CrossEntropyLoss(label_smoothing=label_smoothing), "triplet": None, "supcon": None, "temporal": None, "bce": None}
    if name == "triplet":
        return {"ce": None, "triplet": BatchHardTripletLoss(margin=triplet_margin), "supcon": None, "temporal": None, "bce": None}
    if name == "mse":
        return {"ce": nn.MSELoss(), "triplet": None, "supcon": None, "temporal": None, "bce": None}
    if name in {"combined", "ce_triplet"}:
        return {
            "ce": nn.CrossEntropyLoss(label_smoothing=label_smoothing),
            "triplet": BatchHardTripletLoss(margin=triplet_margin),
            "supcon": SupervisedContrastiveLoss(temperature=supcon_temperature),
            "temporal": TemporalConsistencyLoss(),
            "bce": nn.BCEWithLogitsLoss(),
        }
    raise ValueError(f"Unsupported loss: {name}")
