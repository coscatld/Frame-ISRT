from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class ArcFaceLoss(nn.Module):
    """Additive angular margin loss for identity classification."""

    def __init__(
        self,
        num_classes: int,
        embedding_dim: int,
        scale: float = 30.0,
        margin: float = 0.5,
        easy_margin: bool = False,
    ) -> None:
        super().__init__()
        if num_classes <= 1:
            raise ValueError("ArcFaceLoss requires at least two classes.")
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.scale = scale
        self.margin = margin
        self.easy_margin = easy_margin
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if embeddings.ndim != 2:
            raise ValueError(f"embeddings must be [B, D], got {tuple(embeddings.shape)}")
        labels = labels.long()
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight)).clamp(-1.0, 1.0)
        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp_min(1e-9))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        logits = logits * self.scale
        return F.cross_entropy(logits, labels), logits


class SubCenterArcFaceLoss(nn.Module):
    """Sub-center ArcFace with K class centers per identity."""

    def __init__(
        self,
        num_classes: int,
        embedding_dim: int,
        num_subcenters: int = 3,
        scale: float = 30.0,
        margin: float = 0.5,
    ) -> None:
        super().__init__()
        if num_subcenters <= 0:
            raise ValueError("num_subcenters must be positive.")
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.num_subcenters = num_subcenters
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.empty(num_classes * num_subcenters, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        labels = labels.long()
        cosine_all = F.linear(F.normalize(embeddings), F.normalize(self.weight)).clamp(-1.0, 1.0)
        cosine = cosine_all.view(-1, self.num_classes, self.num_subcenters).max(dim=2).values
        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp_min(1e-9))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        logits = logits * self.scale
        return F.cross_entropy(logits, labels), logits
