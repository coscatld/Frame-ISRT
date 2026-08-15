from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


KGBD_BONES = (
    (11, 10),
    (10, 1),
    (1, 0),
    (1, 2),
    (2, 4),
    (4, 6),
    (6, 8),
    (1, 3),
    (3, 5),
    (5, 7),
    (7, 9),
    (11, 12),
    (12, 14),
    (14, 16),
    (16, 18),
    (11, 13),
    (13, 15),
    (15, 17),
    (17, 19),
)

# Standard Kinect v1 order used by the official TranSG IAS/BIWI/KGBD arrays:
# hip center, spine, shoulder center, head, left arm, right arm, left leg,
# right leg. The current project's duplicate-safe KGBD files use KGBD_BONES
# above, which is a different re-indexing despite having the same joint count.
TRANSG_20_BONES = (
    (2, 3),
    (2, 8),
    (8, 9),
    (9, 10),
    (10, 11),
    (2, 4),
    (4, 5),
    (5, 6),
    (6, 7),
    (2, 1),
    (1, 0),
    (0, 16),
    (16, 17),
    (17, 18),
    (18, 19),
    (0, 12),
    (12, 13),
    (13, 14),
    (14, 15),
)


class TemporalSelfAttentionBlock(nn.Module):
    """Small deterministic pre-norm self-attention block for six tokens."""

    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        self.num_heads = int(num_heads)
        self.head_dim = d_model // num_heads
        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.projection = nn.Linear(d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, frames, channels = tokens.shape
        qkv = self.qkv(self.norm1(tokens))
        qkv = qkv.reshape(batch, frames, 3, self.num_heads, self.head_dim)
        query, key, value = qkv.permute(2, 0, 3, 1, 4)
        attention = torch.softmax(
            torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim),
            dim=-1,
        )
        mixed = torch.matmul(attention, value).transpose(1, 2).reshape(batch, frames, channels)
        tokens = tokens + self.dropout(self.projection(mixed))
        return tokens + self.dropout(self.feed_forward(self.norm2(tokens)))


class MotionGeometryBuilder(nn.Module):
    """Root/scale normalization and deterministic geometric motion fields."""

    def __init__(
        self,
        eps: float = 1e-6,
        *,
        num_joints: int = 20,
        root_joint: int = 11,
        bones: tuple[tuple[int, int], ...] = KGBD_BONES,
    ) -> None:
        super().__init__()
        self.eps = float(eps)
        self.num_joints = int(num_joints)
        self.root_joint = int(root_joint)
        if not 0 <= self.root_joint < self.num_joints:
            raise ValueError("root_joint is outside the skeleton")
        if not bones or any(min(edge) < 0 or max(edge) >= self.num_joints for edge in bones):
            raise ValueError("bones contain an invalid joint index")
        self.register_buffer("parent", torch.tensor([item[0] for item in bones]))
        self.register_buffer("child", torch.tensor([item[1] for item in bones]))

    @staticmethod
    def _median(values: torch.Tensor) -> torch.Tensor:
        ordered = torch.sort(values, dim=1).values
        middle = ordered.shape[1] // 2
        if ordered.shape[1] % 2:
            return ordered[:, middle]
        return 0.5 * (ordered[:, middle - 1] + ordered[:, middle])

    def normalization_scale(self, skeleton: torch.Tensor) -> torch.Tensor:
        skeleton = skeleton.float()
        bone = skeleton.index_select(2, self.child) - skeleton.index_select(2, self.parent)
        length = torch.linalg.vector_norm(bone, dim=-1).clamp_min(self.eps)
        return self._median(length.flatten(1)).clamp_min(self.eps)

    def root_center(self, skeleton: torch.Tensor) -> torch.Tensor:
        skeleton = skeleton.float()
        return skeleton - skeleton[:, :, self.root_joint : self.root_joint + 1]

    def forward(
        self, skeleton: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        expected = (6, self.num_joints, 3)
        if skeleton.ndim != 4 or skeleton.shape[1:] != expected:
            raise ValueError(f"Expected [B,{expected[0]},{expected[1]},{expected[2]}], got {tuple(skeleton.shape)}")
        centered = self.root_center(skeleton)
        bone = centered.index_select(2, self.child) - centered.index_select(2, self.parent)
        length = torch.linalg.vector_norm(bone, dim=-1).clamp_min(self.eps)
        scale = self._median(length.flatten(1)).clamp_min(self.eps)
        normalized = centered / scale[:, None, None, None]
        normalized_length = length / scale[:, None, None]
        direction = bone / length.unsqueeze(-1)

        direction_change = torch.zeros_like(direction)
        direction_change[:, 1:] = direction[:, 1:] - direction[:, :-1]
        angular_speed = torch.zeros_like(length)
        cosine = (direction[:, 1:] * direction[:, :-1]).sum(dim=-1).clamp(-1.0, 1.0)
        angle = torch.acos(cosine)
        angle = torch.where(cosine >= 1.0 - self.eps, torch.zeros_like(angle), angle)
        angular_speed[:, 1:] = angle
        angular_acceleration = torch.zeros_like(length)
        angular_acceleration[:, 2:] = angular_speed[:, 2:] - angular_speed[:, 1:-1]
        dynamic = torch.cat(
            (
                direction_change,
                angular_speed.unsqueeze(-1),
                angular_acceleration.unsqueeze(-1),
            ),
            dim=-1,
        )
        median = torch.quantile(normalized_length, 0.50, dim=1)
        iqr = torch.quantile(normalized_length, 0.75, dim=1) - torch.quantile(
            normalized_length, 0.25, dim=1
        )
        geometry = torch.stack((median, iqr), dim=-1)
        return normalized, geometry, dynamic, direction


class ShortWindowGeometryMotionResidual(nn.Module):
    """Geometry-motion encoder with switchable residuals for exact ablations."""

    def __init__(
        self,
        embedding_dim: int = 128,
        dropout: float = 0.1,
        *,
        include_bone_direction: bool = True,
        use_sequence_residual: bool = True,
        use_geometry_condition: bool = True,
        use_handcrafted_residual: bool = True,
        num_joints: int = 20,
        root_joint: int = 11,
        bones: tuple[tuple[int, int], ...] = KGBD_BONES,
    ) -> None:
        super().__init__()
        self.include_bone_direction = bool(include_bone_direction)
        self.use_sequence_residual = bool(use_sequence_residual)
        self.use_geometry_condition = bool(use_geometry_condition)
        self.use_handcrafted_residual = bool(use_handcrafted_residual)
        if self.use_geometry_condition and not self.use_sequence_residual:
            raise ValueError("Geometry conditioning requires the sequence residual")
        self.num_joints = int(num_joints)
        self.builder = MotionGeometryBuilder(
            num_joints=self.num_joints,
            root_joint=root_joint,
            bones=bones,
        )
        bone_count = len(bones)
        static_dim = self.num_joints * 3 * 2 + bone_count * 2
        if self.include_bone_direction:
            static_dim += bone_count * 3 * 2
        temporal_dim = 2 * self.num_joints * 3 + bone_count * 5 * 2
        self.static_dim = int(static_dim)
        self.temporal_dim = int(temporal_dim)
        self.static_encoder = nn.Sequential(
            nn.Linear(static_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, embedding_dim),
        )
        if self.use_handcrafted_residual:
            self.temporal_encoder = nn.Sequential(
                nn.Linear(self.temporal_dim, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(256, embedding_dim, bias=False),
            )
            nn.init.zeros_(self.temporal_encoder[-1].weight)
            self.residual_gate = nn.Parameter(torch.tensor(-1.5))
        if self.use_sequence_residual:
            self.sequence_input = nn.Linear(self.num_joints * 3, embedding_dim)
            if self.use_geometry_condition:
                self.geometry_condition = nn.Linear(static_dim, embedding_dim)
            self.sequence_position = nn.Parameter(torch.zeros(1, 6, embedding_dim))
            nn.init.normal_(self.sequence_position, std=0.02)
            self.sequence_blocks = nn.ModuleList(
                [
                    TemporalSelfAttentionBlock(
                        embedding_dim, num_heads=4, dropout=dropout
                    )
                    for _ in range(2)
                ]
            )
            self.sequence_output = nn.Linear(embedding_dim, embedding_dim, bias=False)
            nn.init.zeros_(self.sequence_output.weight)
            self.sequence_gate = nn.Parameter(torch.tensor(-1.5))

    @staticmethod
    def dct_features(normalized: torch.Tensor) -> torch.Tensor:
        frames = normalized.shape[1]
        time = torch.arange(frames, device=normalized.device, dtype=normalized.dtype)
        frequencies = torch.arange(1, 3, device=normalized.device, dtype=normalized.dtype)
        basis = torch.cos(math.pi / frames * (time[None] + 0.5) * frequencies[:, None])
        basis = basis * math.sqrt(2.0 / frames)
        return torch.einsum("kt,btjc->bkjc", basis, normalized).flatten(1)

    def feature_vectors(self, skeleton: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        normalized, geometry, dynamic, direction = self.builder(skeleton)
        static_groups = [
            normalized.mean(dim=1).flatten(1),
            normalized.std(dim=1, unbiased=False).flatten(1),
            geometry.flatten(1),
        ]
        if self.include_bone_direction:
            static_groups.extend(
                (
                    direction.mean(dim=1).flatten(1),
                    direction.std(dim=1, unbiased=False).flatten(1),
                )
            )
        static = torch.cat(static_groups, dim=1)
        temporal = torch.cat(
            (
                self.dct_features(normalized),
                dynamic.mean(dim=1).flatten(1),
                dynamic.std(dim=1, unbiased=False).flatten(1),
            ),
            dim=1,
        )
        if static.shape[1] != self.static_dim or temporal.shape[1] != self.temporal_dim:
            raise RuntimeError("Unexpected SWGM feature contract")
        return static, temporal

    def forward(self, skeleton: torch.Tensor) -> torch.Tensor:
        static, temporal = self.feature_vectors(skeleton)
        output = self.static_encoder(static)
        if self.use_handcrafted_residual:
            motion_residual = self.temporal_encoder(temporal)
            output = output + torch.sigmoid(self.residual_gate) * motion_residual
        if self.use_sequence_residual:
            normalized, _, _, _ = self.builder(skeleton)
            frame_tokens = self.sequence_input(normalized.flatten(2)) + self.sequence_position
            if self.use_geometry_condition:
                condition_token = self.geometry_condition(static).unsqueeze(1)
                sequence_tokens = torch.cat((condition_token, frame_tokens), dim=1)
            else:
                sequence_tokens = frame_tokens
            for block in self.sequence_blocks:
                sequence_tokens = block(sequence_tokens)
            sequence_summary = (
                sequence_tokens[:, 0]
                if self.use_geometry_condition
                else sequence_tokens.mean(dim=1)
            )
            sequence_residual = self.sequence_output(sequence_summary)
            output = output + torch.sigmoid(self.sequence_gate) * sequence_residual
        return F.normalize(output, dim=-1)


class GeometryResidualGRU(nn.Module):
    """Strong recurrent motion backbone with a lightweight geometry residual."""

    def __init__(
        self,
        embedding_dim: int = 128,
        dropout: float = 0.1,
        *,
        d_model: int = 128,
        num_joints: int = 20,
        root_joint: int = 11,
        bones: tuple[tuple[int, int], ...] = KGBD_BONES,
    ) -> None:
        super().__init__()
        self.num_joints = int(num_joints)
        self.builder = MotionGeometryBuilder(
            num_joints=self.num_joints,
            root_joint=root_joint,
            bones=bones,
        )
        self.input_projection = nn.Linear(self.num_joints * 3, d_model)
        self.encoder = nn.GRU(
            d_model,
            d_model,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )
        self.temporal_output = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, embedding_dim, bias=False)
        )
        bone_count = len(bones)
        self.geometry_dim = self.num_joints * 3 * 2 + bone_count * 2 + bone_count * 3 * 2 + 1
        self.geometry_encoder = nn.Sequential(
            nn.Linear(self.geometry_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, embedding_dim, bias=False),
        )
        nn.init.zeros_(self.geometry_encoder[-1].weight)
        self.residual_gate = nn.Parameter(torch.tensor(-1.5))
        # Positional embedding matches TemporalSequenceBaseline exactly so the
        # geometry-residual combiner is a single-variable control.
        self.position = nn.Parameter(torch.zeros(1, 6, d_model))
        nn.init.normal_(self.position, std=0.02)

    def geometry_features(self, skeleton: torch.Tensor) -> torch.Tensor:
        normalized, geometry, _, direction = self.builder(skeleton)
        scale = self.builder.normalization_scale(skeleton).log().unsqueeze(1)
        features = torch.cat(
            (
                normalized.mean(dim=1).flatten(1),
                normalized.std(dim=1, unbiased=False).flatten(1),
                geometry.flatten(1),
                direction.mean(dim=1).flatten(1),
                direction.std(dim=1, unbiased=False).flatten(1),
                scale,
            ),
            dim=1,
        )
        if features.shape[1] != self.geometry_dim:
            raise RuntimeError("Unexpected GR-GRU geometry feature contract")
        return features

    def forward(self, skeleton: torch.Tensor) -> torch.Tensor:
        normalized, _, _, _ = self.builder(skeleton)
        tokens = self.input_projection(normalized.flatten(2)) + self.position
        tokens, _ = self.encoder(tokens)
        temporal = self.temporal_output(tokens.mean(dim=1))
        geometry = self.geometry_encoder(self.geometry_features(skeleton))
        output = temporal + torch.sigmoid(self.residual_gate) * geometry
        return F.normalize(output, dim=-1)


class RootCenteredGRU(nn.Module):
    """GRU baseline that preserves absolute body scale after root centering."""

    def __init__(
        self,
        embedding_dim: int = 128,
        dropout: float = 0.1,
        *,
        d_model: int = 128,
        num_joints: int = 20,
        root_joint: int = 11,
        bones: tuple[tuple[int, int], ...] = KGBD_BONES,
    ) -> None:
        super().__init__()
        self.num_joints = int(num_joints)
        self.builder = MotionGeometryBuilder(
            num_joints=self.num_joints,
            root_joint=root_joint,
            bones=bones,
        )
        self.input_projection = nn.Linear(self.num_joints * 3, d_model)
        self.encoder = nn.GRU(
            d_model,
            d_model,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )
        self.output = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, embedding_dim, bias=False)
        )
        # Positional embedding matches TemporalSequenceBaseline exactly so root
        # centering is the only difference (single-variable control).
        self.position = nn.Parameter(torch.zeros(1, 6, d_model))
        nn.init.normal_(self.position, std=0.02)

    def forward(self, skeleton: torch.Tensor) -> torch.Tensor:
        centered = self.builder.root_center(skeleton)
        tokens = self.input_projection(centered.flatten(2)) + self.position
        tokens, _ = self.encoder(tokens)
        return F.normalize(self.output(tokens.mean(dim=1)), dim=-1)


_RANDOM_RESIDUAL_SEED = 20260813


def _matched_random_residual(
    centered: torch.Tensor, holder: nn.Module
) -> torch.Tensor:
    """Per-sample RMS-matched Gaussian noise.

    Random shape, but each sample's noise is scaled so its root-mean-square
    equals the absolute residual input's (root-centered C) RMS for that sample.
    The generator advances across calls (fresh noise per forward) yet is seeded
    once per model instance, so a run is reproducible end to end.
    """
    device = centered.device
    generator = getattr(holder, "_noise_generator", None)
    if generator is None or generator.device != device:
        generator = torch.Generator(device=device).manual_seed(_RANDOM_RESIDUAL_SEED)
        holder._noise_generator = generator
    noise = torch.randn(
        centered.shape, dtype=centered.dtype, device=centered.device, generator=generator
    )
    dim = centered.shape[1] * centered.shape[2] * centered.shape[3]
    rms_centered = centered.flatten(1).norm(dim=1) / math.sqrt(dim)
    rms_noise = noise.flatten(1).norm(dim=1) / math.sqrt(dim)
    return noise * (rms_centered / rms_noise.clamp_min(1e-6))[:, None, None, None]


class IdentityScaleResidualGRU(nn.Module):
    """Scale-normalized GRU with a zero-initialized identity-scale token residual."""

    def __init__(
        self,
        embedding_dim: int = 128,
        dropout: float = 0.1,
        *,
        d_model: int = 128,
        num_joints: int = 20,
        root_joint: int = 11,
        bones: tuple[tuple[int, int], ...] = KGBD_BONES,
        residual_input_type: str = "absolute",
    ) -> None:
        super().__init__()
        if residual_input_type not in (
            "absolute",
            "normalized",
            "norm_matched",
            "random_matched",
        ):
            raise ValueError(f"Unknown residual_input_type: {residual_input_type}")
        self.num_joints = int(num_joints)
        self.builder = MotionGeometryBuilder(
            num_joints=self.num_joints,
            root_joint=root_joint,
            bones=bones,
        )
        frame_dim = self.num_joints * 3
        self.normalized_projection = nn.Linear(frame_dim, d_model)
        self.scale_residual_projection = nn.Linear(frame_dim, d_model, bias=False)
        nn.init.zeros_(self.scale_residual_projection.weight)
        self.residual_gate = nn.Parameter(torch.tensor(-1.5))
        self.residual_input_type = residual_input_type
        # Protocol-level constant for the norm_matched arm: N * factor lifts the
        # normalized residual's RMS to match the absolute input's. Set by the
        # training driver from training-set statistics; 1.0 (= normalized arm)
        # when left unset.
        self.register_buffer("residual_energy_factor", torch.tensor(1.0))
        self._noise_generator = None
        # Positional embedding matches TemporalSequenceBaseline exactly so the
        # only difference from the GRU baseline is the ISRT input combiner
        # (single-variable control).
        self.position = nn.Parameter(torch.zeros(1, 6, d_model))
        nn.init.normal_(self.position, std=0.02)
        self.encoder = nn.GRU(
            d_model,
            d_model,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )
        self.output = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, embedding_dim, bias=False)
        )

    def frame_tokens(self, skeleton: torch.Tensor) -> torch.Tensor:
        normalized, _, _, _ = self.builder(skeleton)
        centered = self.builder.root_center(skeleton)
        normalized_tokens = self.normalized_projection(normalized.flatten(2))
        # residual_input_type controls whether the residual projection reads
        # absolute root-centered coords C (default), the median-bone-length
        # normalized coords N = C/s, N re-scaled to absolute RMS/variance
        # (norm_matched), or per-sample RMS-matched Gaussian noise (random).
        # All from the SAME builder pass; shapes are identical, only content
        # and/or magnitude differ.
        residual_type = self.residual_input_type
        if residual_type == "normalized":
            residual_source = normalized
        elif residual_type == "norm_matched":
            residual_source = normalized * self.residual_energy_factor
        elif residual_type == "random_matched":
            residual_source = _matched_random_residual(centered, self)
        else:
            residual_source = centered
        scale_residual = self.scale_residual_projection(residual_source.flatten(2))
        return normalized_tokens + torch.sigmoid(self.residual_gate) * scale_residual

    def forward(self, skeleton: torch.Tensor) -> torch.Tensor:
        tokens = self.frame_tokens(skeleton) + self.position
        tokens, _ = self.encoder(tokens)
        return F.normalize(self.output(tokens.mean(dim=1)), dim=-1)


class IdentityScaleResidualSequence(nn.Module):
    """ISR token combiner feeding a configurable temporal backbone.

    Mirrors TemporalSequenceBaseline exactly except the input token combiner:
    scale-normalized projection plus a zero-initialized identity-scale residual
    instead of a single projection over normalized coordinates. The backbone
    (Transformer or Mamba) therefore receives the same positional encoding and
    per-frame token format, isolating the scale-residual mechanism.
    """

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
        residual_input_type: str = "absolute",
    ) -> None:
        super().__init__()
        if residual_input_type not in (
            "absolute",
            "normalized",
            "norm_matched",
            "random_matched",
        ):
            raise ValueError(f"Unknown residual_input_type: {residual_input_type}")
        self.backend = backend.lower()
        self.builder = MotionGeometryBuilder(
            num_joints=num_joints, root_joint=root_joint, bones=bones
        )
        frame_dim = num_joints * 3
        self.normalized_projection = nn.Linear(frame_dim, d_model)
        self.scale_residual_projection = nn.Linear(frame_dim, d_model, bias=False)
        nn.init.zeros_(self.scale_residual_projection.weight)
        self.residual_gate = nn.Parameter(torch.tensor(-1.5))
        self.residual_input_type = residual_input_type
        self.register_buffer("residual_energy_factor", torch.tensor(1.0))
        self._noise_generator = None
        self.position = nn.Parameter(torch.zeros(1, 6, d_model))
        nn.init.normal_(self.position, std=0.02)
        if self.backend == "transformer":
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=4,
                dim_feedforward=d_model * 2,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(
                layer, num_layers=2, norm=nn.LayerNorm(d_model)
            )
        elif self.backend == "mamba":
            try:
                from mamba_ssm import Mamba
            except ImportError as error:
                raise RuntimeError("The ISR-Mamba baseline requires mamba_ssm") from error
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
            raise ValueError(f"Unsupported ISR backbone: {backend}")
        self.output = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, embedding_dim, bias=False)
        )

    def frame_tokens(self, skeleton: torch.Tensor) -> torch.Tensor:
        normalized, _, _, _ = self.builder(skeleton)
        centered = self.builder.root_center(skeleton)
        normalized_tokens = self.normalized_projection(normalized.flatten(2))
        residual_type = self.residual_input_type
        if residual_type == "normalized":
            residual_source = normalized
        elif residual_type == "norm_matched":
            residual_source = normalized * self.residual_energy_factor
        elif residual_type == "random_matched":
            residual_source = _matched_random_residual(centered, self)
        else:
            residual_source = centered
        scale_residual = self.scale_residual_projection(residual_source.flatten(2))
        return normalized_tokens + torch.sigmoid(self.residual_gate) * scale_residual

    def forward(self, skeleton: torch.Tensor) -> torch.Tensor:
        tokens = self.frame_tokens(skeleton) + self.position
        if self.backend == "transformer":
            tokens = self.encoder(tokens)
        else:
            for block in self.encoder:
                tokens = tokens + block["dropout"](
                    block["mamba"](block["norm"](tokens))
                )
        return F.normalize(self.output(tokens.mean(dim=1)), dim=-1)


class ScalePreservingMotionGeometryGRU(nn.Module):
    """Single-stream GRU over scale-preserving structure and normalized motion."""

    def __init__(
        self,
        embedding_dim: int = 128,
        dropout: float = 0.1,
        *,
        d_model: int = 128,
        num_joints: int = 20,
        root_joint: int = 11,
        bones: tuple[tuple[int, int], ...] = KGBD_BONES,
    ) -> None:
        super().__init__()
        if d_model % 2:
            raise ValueError("d_model must be even")
        self.num_joints = int(num_joints)
        self.bone_count = len(bones)
        self.builder = MotionGeometryBuilder(
            num_joints=self.num_joints,
            root_joint=root_joint,
            bones=bones,
        )
        frame_dim = self.num_joints * 3 + self.bone_count
        self.structure_projection = nn.Sequential(
            nn.Linear(frame_dim, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.motion_projection = nn.Sequential(
            nn.Linear(frame_dim, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.encoder = nn.GRU(
            d_model,
            d_model,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )
        self.output = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, embedding_dim, bias=False)
        )
        # Positional embedding matches the baseline GRU exactly so the only
        # difference from the GRU baseline is the scale handling.
        self.position = nn.Parameter(torch.zeros(1, 6, d_model))
        nn.init.normal_(self.position, std=0.02)

    def frame_features(self, skeleton: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        centered = self.builder.root_center(skeleton)
        bone = centered.index_select(2, self.builder.child) - centered.index_select(
            2, self.builder.parent
        )
        length = torch.linalg.vector_norm(bone, dim=-1).clamp_min(self.builder.eps)
        scale = self.builder._median(length.flatten(1)).clamp_min(self.builder.eps)
        normalized = centered / scale[:, None, None, None]

        velocity = torch.zeros_like(normalized)
        velocity[:, 1:] = normalized[:, 1:] - normalized[:, :-1]
        direction = bone / length.unsqueeze(-1)
        angular_speed = torch.zeros_like(length)
        cosine = (direction[:, 1:] * direction[:, :-1]).sum(dim=-1).clamp(-1.0, 1.0)
        angle = torch.acos(cosine)
        angle = torch.where(
            cosine >= 1.0 - self.builder.eps, torch.zeros_like(angle), angle
        )
        angular_speed[:, 1:] = angle

        structure = torch.cat((centered.flatten(2), length), dim=-1)
        motion = torch.cat((velocity.flatten(2), angular_speed), dim=-1)
        expected = self.num_joints * 3 + self.bone_count
        if structure.shape[-1] != expected or motion.shape[-1] != expected:
            raise RuntimeError("Unexpected SPMG-GRU frame feature contract")
        return structure, motion

    def forward(self, skeleton: torch.Tensor) -> torch.Tensor:
        structure, motion = self.frame_features(skeleton)
        tokens = torch.cat(
            (self.structure_projection(structure), self.motion_projection(motion)), dim=-1
        ) + self.position
        tokens, _ = self.encoder(tokens)
        return F.normalize(self.output(tokens.mean(dim=1)), dim=-1)


def build_mva_model(
    name: str,
    *,
    embedding_dim: int = 128,
    dropout: float = 0.1,
    mamba_d_state: int = 16,
    layout: str = "kgbd_reindexed20",
) -> nn.Module:
    if layout == "kgbd_reindexed20":
        layout_kwargs = {"num_joints": 20, "root_joint": 11, "bones": KGBD_BONES}
    elif layout == "transg20":
        layout_kwargs = {"num_joints": 20, "root_joint": 0, "bones": TRANSG_20_BONES}
    else:
        raise ValueError(f"Unknown skeleton layout: {layout}")
    normalized = name.lower()
    if normalized == "isrgru":
        return IdentityScaleResidualGRU(
            embedding_dim=embedding_dim,
            dropout=dropout,
            d_model=128,
            **layout_kwargs,
        )
    if normalized == "isrgru_norm":
        return IdentityScaleResidualGRU(
            embedding_dim=embedding_dim,
            dropout=dropout,
            d_model=128,
            residual_input_type="normalized",
            **layout_kwargs,
        )
    if normalized in {"isrgru_normmatch", "isrgru_randommatch"}:
        residual_type = (
            "norm_matched" if normalized.endswith("normmatch") else "random_matched"
        )
        return IdentityScaleResidualGRU(
            embedding_dim=embedding_dim,
            dropout=dropout,
            d_model=128,
            residual_input_type=residual_type,
            **layout_kwargs,
        )
    if normalized == "rcgru":
        return RootCenteredGRU(
            embedding_dim=embedding_dim,
            dropout=dropout,
            d_model=128,
            **layout_kwargs,
        )
    if normalized == "spmg":
        return ScalePreservingMotionGeometryGRU(
            embedding_dim=embedding_dim,
            dropout=dropout,
            d_model=128,
            **layout_kwargs,
        )
    if normalized == "grgru":
        return GeometryResidualGRU(
            embedding_dim=embedding_dim,
            dropout=dropout,
            d_model=128,
            **layout_kwargs,
        )
    if normalized == "gctr":
        return ShortWindowGeometryMotionResidual(
            embedding_dim=embedding_dim,
            dropout=dropout,
            include_bone_direction=True,
            **layout_kwargs,
        )
    if normalized == "swgm_original":
        return ShortWindowGeometryMotionResidual(
            embedding_dim=embedding_dim,
            dropout=dropout,
            include_bone_direction=False,
            use_sequence_residual=False,
            use_geometry_condition=False,
            **layout_kwargs,
        )
    if normalized == "swgm_bonedir":
        return ShortWindowGeometryMotionResidual(
            embedding_dim=embedding_dim,
            dropout=dropout,
            include_bone_direction=True,
            use_sequence_residual=False,
            use_geometry_condition=False,
            **layout_kwargs,
        )
    if normalized == "gctr_unconditioned":
        return ShortWindowGeometryMotionResidual(
            embedding_dim=embedding_dim,
            dropout=dropout,
            include_bone_direction=True,
            use_sequence_residual=True,
            use_geometry_condition=False,
            **layout_kwargs,
        )
    if normalized in {"gru", "transformer", "mamba"}:
        from .baselines import TemporalSequenceBaseline

        return TemporalSequenceBaseline(
            normalized,
            embedding_dim=embedding_dim,
            d_model=128,
            dropout=dropout,
            mamba_d_state=mamba_d_state,
            **layout_kwargs,
        )
    if normalized in {"isr_transformer", "isr_mamba"}:
        return IdentityScaleResidualSequence(
            normalized.removeprefix("isr_"),
            embedding_dim=embedding_dim,
            d_model=128,
            dropout=dropout,
            mamba_d_state=mamba_d_state,
            **layout_kwargs,
        )
    if normalized in {"isr_transformer_norm", "isr_mamba_norm"}:
        return IdentityScaleResidualSequence(
            normalized.removeprefix("isr_").removesuffix("_norm"),
            embedding_dim=embedding_dim,
            d_model=128,
            dropout=dropout,
            mamba_d_state=mamba_d_state,
            residual_input_type="normalized",
            **layout_kwargs,
        )
    if normalized in {"isr_transformer_normmatch", "isr_transformer_randommatch"}:
        residual_type = (
            "norm_matched"
            if normalized.endswith("normmatch")
            else "random_matched"
        )
        return IdentityScaleResidualSequence(
            "transformer",
            embedding_dim=embedding_dim,
            d_model=128,
            dropout=dropout,
            mamba_d_state=mamba_d_state,
            residual_input_type=residual_type,
            **layout_kwargs,
        )
    if normalized == "stgcn":
        from .baselines import STGCNBaseline

        return STGCNBaseline(embedding_dim=embedding_dim, dropout=dropout, **layout_kwargs)
    if normalized == "isr_stgcn":
        from .baselines import STGCNBaseline

        return STGCNBaseline(
            embedding_dim=embedding_dim, dropout=dropout, isrt=True, **layout_kwargs
        )
    if normalized == "transg_reproduction":
        from .baselines import TranSGPyTorchReproduction

        return TranSGPyTorchReproduction(
            embedding_dim=embedding_dim,
            d_model=128,
            num_heads=8,
            dropout=0.5,
            **layout_kwargs,
        )
    if normalized == "node_isrt_transg":
        from .baselines import TranSGPyTorchReproduction

        return TranSGPyTorchReproduction(
            embedding_dim=embedding_dim,
            d_model=128,
            num_heads=8,
            dropout=0.5,
            isrt=True,
            **layout_kwargs,
        )
    raise ValueError(f"Unknown MVA model: {name}")
