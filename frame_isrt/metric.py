from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class TripletLoss(nn.Module):
    """Batch-hard triplet loss over normalized embeddings."""

    def __init__(self, margin: float = 0.3) -> None:
        super().__init__()
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels = labels.view(-1)
        distances = torch.cdist(embeddings, embeddings, p=2)
        same = labels[:, None] == labels[None, :]
        eye = torch.eye(labels.numel(), dtype=torch.bool, device=labels.device)
        positive_mask = same & ~eye
        negative_mask = ~same
        if not positive_mask.any() or not negative_mask.any():
            return embeddings.sum() * 0.0
        hardest_positive = distances.masked_fill(~positive_mask, -1.0).max(dim=1).values
        hardest_negative = distances.masked_fill(~negative_mask, float("inf")).min(dim=1).values
        valid = torch.isfinite(hardest_negative) & (hardest_positive >= 0)
        if not valid.any():
            return embeddings.sum() * 0.0
        return F.relu(hardest_positive[valid] - hardest_negative[valid] + self.margin).mean()


class SupConLoss(nn.Module):
    """Supervised contrastive loss for one or more views per sample."""

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if embeddings.ndim == 2:
            embeddings = embeddings.unsqueeze(1)
        if embeddings.ndim != 3:
            raise ValueError(f"embeddings must be [B, D] or [B, V, D], got {tuple(embeddings.shape)}")
        batch, views, dim = embeddings.shape
        labels = labels.view(batch)
        features = F.normalize(embeddings.reshape(batch * views, dim), dim=-1)
        repeated_labels = labels.repeat_interleave(views)
        logits = features @ features.T / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        self_mask = torch.eye(batch * views, dtype=torch.bool, device=features.device)
        positive = (repeated_labels[:, None] == repeated_labels[None, :]) & ~self_mask
        logits_mask = ~self_mask
        exp_logits = torch.exp(logits) * logits_mask.float()
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
        positive_count = positive.float().sum(dim=1)
        valid = positive_count > 0
        if not valid.any():
            return embeddings.sum() * 0.0
        mean_log_prob_pos = (positive.float() * log_prob).sum(dim=1)[valid] / positive_count[valid]
        return -mean_log_prob_pos.mean()


class BatchHardTripletLoss(TripletLoss):
    """Explicit research-facing name for batch-hard triplet loss."""


class SupervisedContrastiveLoss(SupConLoss):
    """Explicit research-facing name for supervised contrastive loss."""
