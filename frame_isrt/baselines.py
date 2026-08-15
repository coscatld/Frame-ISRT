from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .model import KGBD_BONES, MotionGeometryBuilder


class TemporalSequenceBaseline(nn.Module):
    """Unified six-frame GRU, Transformer, or real Mamba comparison."""

    def __init__(
        self,
        backend: str,
        *,
        embedding_dim: int = 128,
        d_model: int = 128,
        dropout: float = 0.1,
        mamba_d_state: int = 16,
        num_joints: int = 20,
        root_joint: int = 11,
        bones: tuple[tuple[int, int], ...] = KGBD_BONES,
    ) -> None:
        super().__init__()
        self.backend = backend.lower()
        self.builder = MotionGeometryBuilder(
            num_joints=num_joints, root_joint=root_joint, bones=bones
        )
        self.input_projection = nn.Linear(num_joints * 3, d_model)
        self.position = nn.Parameter(torch.zeros(1, 6, d_model))
        nn.init.normal_(self.position, std=0.02)
        if self.backend == "gru":
            self.encoder = nn.GRU(
                d_model,
                d_model,
                num_layers=2,
                batch_first=True,
                dropout=dropout,
            )
        elif self.backend == "transformer":
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=4,
                dim_feedforward=d_model * 2,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=2, norm=nn.LayerNorm(d_model))
        elif self.backend == "mamba":
            try:
                from mamba_ssm import Mamba
            except ImportError as error:
                raise RuntimeError("The unified Mamba baseline requires mamba_ssm") from error
            self.encoder = nn.ModuleList(
                [
                    nn.ModuleDict(
                        {
                            "norm": nn.LayerNorm(d_model),
                            "mamba": Mamba(d_model=d_model, d_state=mamba_d_state, d_conv=4, expand=2),
                            "dropout": nn.Dropout(dropout),
                        }
                    )
                    for _ in range(2)
                ]
            )
        else:
            raise ValueError(f"Unknown temporal backend: {backend}")
        self.output = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, embedding_dim, bias=False))

    def forward(self, skeleton: torch.Tensor) -> torch.Tensor:
        normalized, _, _, _ = self.builder(skeleton)
        tokens = self.input_projection(normalized.flatten(2)) + self.position
        if self.backend == "gru":
            tokens, _ = self.encoder(tokens)
        elif self.backend == "transformer":
            tokens = self.encoder(tokens)
        else:
            for block in self.encoder:
                tokens = tokens + block["dropout"](block["mamba"](block["norm"](tokens)))
        return F.normalize(self.output(tokens.mean(dim=1)), dim=-1)


def normalized_joint_adjacency(
    num_joints: int = 20,
    bones: tuple[tuple[int, int], ...] = KGBD_BONES,
) -> torch.Tensor:
    adjacency = torch.eye(num_joints, dtype=torch.float32)
    for left, right in bones:
        adjacency[left, right] = 1.0
        adjacency[right, left] = 1.0
    degree = adjacency.sum(dim=1).clamp_min(1.0)
    inv_sqrt = degree.rsqrt()
    return inv_sqrt[:, None] * adjacency * inv_sqrt[None, :]


class STGCNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float) -> None:
        super().__init__()
        self.spatial = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.temporal = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=(3, 1),
            padding=(1, 0),
            bias=False,
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.residual = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        mixed = torch.einsum("vw,bctw->bctv", adjacency, x)
        output = self.norm(self.temporal(self.spatial(mixed)))
        return F.gelu(residual + self.dropout(output))


class STGCNBaseline(nn.Module):
    """Two-block spatial graph convolution with six-frame temporal kernels.

    With ``isrt=True`` a zero-initialized identity-scale residual input is added:
    the same graph blocks receive ``normalized_projection + sigmoid(g) *
    scale_residual_projection(root_centered)``, mirroring the ISR-GRU input
    combiner so the graph backbone becomes a single-variable control.
    """

    def __init__(
        self,
        *,
        embedding_dim: int = 128,
        dropout: float = 0.1,
        isrt: bool = False,
        num_joints: int = 20,
        root_joint: int = 11,
        bones: tuple[tuple[int, int], ...] = KGBD_BONES,
    ) -> None:
        super().__init__()
        self.isrt = bool(isrt)
        self.builder = MotionGeometryBuilder(
            num_joints=num_joints, root_joint=root_joint, bones=bones
        )
        self.register_buffer("adjacency", normalized_joint_adjacency(num_joints, bones))
        self.input_projection = nn.Conv2d(3, 64, kernel_size=1, bias=False)
        if self.isrt:
            self.scale_residual_projection = nn.Conv2d(3, 64, kernel_size=1, bias=False)
            nn.init.zeros_(self.scale_residual_projection.weight)
            self.residual_gate = nn.Parameter(torch.tensor(-1.5))
        self.blocks = nn.ModuleList((STGCNBlock(64, 64, dropout), STGCNBlock(64, 128, dropout)))
        self.output = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, embedding_dim, bias=False))

    def forward(self, skeleton: torch.Tensor) -> torch.Tensor:
        normalized, _, _, _ = self.builder(skeleton)
        features = self.input_projection(normalized.permute(0, 3, 1, 2))
        if self.isrt:
            centered = self.builder.root_center(skeleton)
            residual = self.scale_residual_projection(centered.permute(0, 3, 1, 2))
            features = features + torch.sigmoid(self.residual_gate) * residual
        for block in self.blocks:
            features = block(features, self.adjacency)
        pooled = features.mean(dim=(2, 3))
        return F.normalize(self.output(pooled), dim=-1)


class TranSGPyTorchReproduction(nn.Module):
    """Auditable PyTorch reproduction of TranSG's SGT, GPC, and STPR core."""

    def __init__(
        self,
        *,
        embedding_dim: int = 128,
        d_model: int = 128,
        num_heads: int = 8,
        dropout: float = 0.5,
        num_joints: int = 20,
        root_joint: int = 11,
        bones: tuple[tuple[int, int], ...] = KGBD_BONES,
        isrt: bool = False,
    ) -> None:
        super().__init__()
        if embedding_dim != d_model:
            raise ValueError("TranSG reproduction requires embedding_dim == d_model")
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        self.isrt = bool(isrt)
        self.builder = MotionGeometryBuilder(
            num_joints=num_joints, root_joint=root_joint, bones=bones
        )
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.head_dim = d_model // num_heads
        self.coordinate_encoder = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        if self.isrt:
            # Node-ISRT: zero-init per-joint scale residual projection + scalar gate.
            self.scale_residual_projection = nn.Linear(3, d_model, bias=False)
            nn.init.zeros_(self.scale_residual_projection.weight)
            self.residual_gate = nn.Parameter(torch.tensor(-1.5))
        laplacian = torch.eye(num_joints) - normalized_joint_adjacency(num_joints, bones)
        _, eigenvectors = torch.linalg.eigh(laplacian)
        # Sign-canonicalize each eigenvector: flip so the largest-magnitude entry is
        # positive. eigh eigenvectors are sign-non-unique (CPU vs GPU differ), which
        # made the learned PE weight device-dependent; this rule makes it deterministic.
        absmax_idx = eigenvectors.abs().argmax(dim=0)
        sign = torch.sign(eigenvectors[absmax_idx, torch.arange(num_joints)])
        sign[sign == 0] = 1
        eigenvectors = eigenvectors * sign[None, :]
        self.register_buffer("laplacian_position", eigenvectors[:, :10].contiguous())
        self.position_encoder = nn.Linear(10, d_model, bias=False)
        self.qkv_layers = nn.ModuleList(
            [nn.Linear(d_model, d_model * 3, bias=False) for _ in range(2)]
        )
        self.spatial_projection = nn.Linear(d_model, d_model)
        self.spatial_norm = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.output_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.structure_decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_joints * 3),
        )
        self.trajectory_decoder = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 6 * 3),
        )
        self._cached_nodes: torch.Tensor | None = None
        self._cached_frames: torch.Tensor | None = None
        self._cached_normalized: torch.Tensor | None = None

    def _spatial_attention(self, nodes: torch.Tensor, layer: nn.Linear) -> torch.Tensor:
        batch, frames, joints, channels = nodes.shape
        qkv = layer(nodes).reshape(
            batch, frames, joints, 3, self.num_heads, self.head_dim
        )
        query, key, value = qkv.permute(3, 0, 1, 4, 2, 5)
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.head_dim**0.5
        attention = torch.softmax(scores.clamp(-5.0, 5.0), dim=-1)
        mixed = torch.matmul(attention, value)
        return mixed.permute(0, 1, 3, 2, 4).reshape(batch, frames, joints, channels)

    def forward(self, skeleton: torch.Tensor) -> torch.Tensor:
        normalized, _, _, _ = self.builder(skeleton)
        nodes = self.coordinate_encoder(normalized)
        if self.isrt:
            centered = self.builder.root_center(skeleton)
            nodes = nodes + torch.sigmoid(self.residual_gate) * self.scale_residual_projection(
                centered
            )
        nodes = nodes + self.position_encoder(self.laplacian_position)[None, None]
        residual = nodes
        for qkv in self.qkv_layers:
            nodes = self._spatial_attention(nodes, qkv)
        nodes = self.spatial_norm(residual + self.dropout(self.spatial_projection(nodes)))
        nodes = self.output_norm(nodes + self.feed_forward(nodes))
        frames = nodes.mean(dim=2)
        sequence = frames.mean(dim=1)
        if self.training:
            self._cached_nodes = nodes
            self._cached_frames = frames
            self._cached_normalized = normalized
        return F.normalize(sequence, dim=-1)

    def training_objective(
        self,
        labels: torch.Tensor,
        class_prototypes: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self._cached_nodes is None or self._cached_frames is None or self._cached_normalized is None:
            raise RuntimeError("training_objective must follow forward")
        nodes = self._cached_nodes
        frames = self._cached_frames
        normalized = self._cached_normalized
        prototypes = F.normalize(class_prototypes, dim=-1)
        sequence = F.normalize(frames.mean(dim=1), dim=-1)
        sequence_logits = sequence @ prototypes.T / 0.07
        sequence_gpc = F.cross_entropy(sequence_logits, labels)
        frame_logits = F.normalize(frames, dim=-1) @ prototypes.T / 14.0
        frame_labels = labels[:, None].expand(-1, frames.shape[1]).reshape(-1)
        frame_gpc = F.cross_entropy(frame_logits.reshape(-1, len(prototypes)), frame_labels)
        gpc = 0.5 * (sequence_gpc + frame_gpc)

        kept_joints = torch.randperm(nodes.shape[2], device=nodes.device)[:10]
        structure_context = nodes.index_select(2, kept_joints).mean(dim=2)
        structure_prediction = self.structure_decoder(structure_context)
        structure_target = normalized.flatten(2)
        structure_loss = F.l1_loss(structure_prediction, structure_target)

        kept_frames = torch.randperm(6, device=nodes.device)[:4]
        trajectory_context = nodes.index_select(1, kept_frames).mean(dim=1)
        trajectory_prediction = self.trajectory_decoder(trajectory_context)
        trajectory_target = normalized.transpose(1, 2).flatten(2)
        trajectory_loss = F.l1_loss(trajectory_prediction, trajectory_target)
        reconstruction = 0.5 * (structure_loss + trajectory_loss)
        total = 0.5 * gpc + 0.5 * reconstruction
        return total, {
            "gpc": gpc.detach(),
            "structure": structure_loss.detach(),
            "trajectory": trajectory_loss.detach(),
        }
