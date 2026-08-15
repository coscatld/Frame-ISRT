"""Faithful PyTorch port of TranSG (Rao & Miao, CVPR 2023), joint level.

Ported from the official TF1.14 implementation (TranSG-main/TranSG.py,
utils/process_SG.py). Key fidelity points preserved from the official code:

- Input is ROOT-CENTERED ONLY (subtract root joint); TranSG does NOT apply any
  per-sequence scale normalization.
- SGT: 2 layers x 8 full-relation heads; per-head Q/K/V projections use
  random_normal std=1.0 (NOT Glorot); attention logits are scaled by
  1/sqrt(head_dim), clamped to [-5, 5], softmaxed, and the [J,J] score map is
  SHARED across all head_dim channels.
- Post-attention block runs ONCE: dropout -> dense -> residual against the
  pos-enc-added embedding -> BatchNorm -> FFN(H->2H->H) -> residual -> BatchNorm.
  BatchNorm over the channel dim with TF momentum 0.99 (=> torch momentum 0.01),
  eps=1e-3.
- GPC: sequence-level (t1=0.07, L2-normalized, mean over batch) and
  skeleton/frame-level (t2=14, NO L2 norm, per-frame CE SUMMED over frames then
  mean over batch). H_loss = 0.5*GPC_ske + 0.5*GPC_seq.
- STPR: structure-prompted (mask 10 joints) and trajectory-prompted (keep 4 of 6
  frames) reconstruction of the raw root-centered coordinates, L1 mean / batch.
  recon = 0.5*structure + 0.5*trajectory. total = 0.5*H_loss + 0.5*recon.
- Class prototypes are GROUND-TRUTH label means recomputed each epoch from a
  full no-grad feature pass (no faiss / k-means in the released code).

Node-ISRT injects a zero-init per-joint scale-residual projection of the
root-centered (un-normalized) coordinates at the input embedding, plus a scalar
gate. Because TranSG already preserves absolute scale, this is a gated
re-weighting rather than a scale-recovery path.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .model import TRANSG_20_BONES


def _normal_(t: torch.Tensor, std: float = 1.0) -> torch.Tensor:
    return nn.init.normal_(t, mean=0.0, std=std)


class TranSGFaithful(nn.Module):
    def __init__(
        self,
        *,
        embedding_dim: int = 128,
        d_model: int = 128,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.5,
        num_joints: int = 20,
        num_frames: int = 6,
        root_joint: int = 0,
        bones: tuple[tuple[int, int], ...] = TRANSG_20_BONES,
        enc_k: int = 10,
        isrt: bool = False,
        rand_flip: bool = True,
        residual_scale_mode: str = "absolute",
        residual_gate_fixed: float | None = None,
        residual_proj_init: str = "zero",
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.H = int(d_model)
        self.num_heads = int(num_heads)
        self.head_dim = self.H // self.num_heads
        self.num_layers = int(num_layers)
        self.num_joints = int(num_joints)
        self.num_frames = int(num_frames)
        self.root_joint = int(root_joint)
        self.isrt = bool(isrt)
        self.rand_flip = bool(rand_flip)
        if residual_scale_mode not in {"absolute", "normalized"}:
            raise ValueError(f"residual_scale_mode must be absolute or normalized, got {residual_scale_mode!r}")
        self.residual_scale_mode = str(residual_scale_mode)
        self.residual_gate_fixed = (
            None if residual_gate_fixed is None else float(residual_gate_fixed)
        )
        if residual_proj_init not in {"zero", "random"}:
            raise ValueError(
                f"residual_proj_init must be zero or random, got {residual_proj_init!r}"
            )
        self.residual_proj_init = str(residual_proj_init)

        # Input embedding: dense(3->H, relu) -> dense(H->H, None). Glorot + zero bias
        # matches tf.layers.dense defaults.
        self.input_fc1 = nn.Linear(3, self.H)
        self.input_fc2 = nn.Linear(self.H, self.H)

        # Laplacian positional encoding projection dense(k->H, None).
        self.pos_proj = nn.Linear(enc_k, self.H, bias=True)

        # FR attention heads: [L, H, heads, head_dim] Q/K/V, init std=1.0.
        self.W_Q = nn.Parameter(_normal_(torch.empty(num_layers, self.H, num_heads, self.head_dim)))
        self.W_K = nn.Parameter(_normal_(torch.empty(num_layers, self.H, num_heads, self.head_dim)))
        self.W_V = nn.Parameter(_normal_(torch.empty(num_layers, self.H, num_heads, self.head_dim)))

        # Post-attention block.
        self.post_dropout = nn.Dropout(dropout)
        self.post_dense = nn.Linear(self.H, self.H)
        self.bn1 = nn.BatchNorm1d(self.H, eps=1e-3, momentum=0.01)
        self.ffn_fc1 = nn.Linear(self.H, self.H * 2)
        self.ffn_drop = nn.Dropout(dropout)
        self.ffn_fc2 = nn.Linear(self.H * 2, self.H)
        self.bn2 = nn.BatchNorm1d(self.H, eps=1e-3, momentum=0.01)
        for layer in (
            self.input_fc1, self.input_fc2, self.pos_proj,
            self.post_dense, self.ffn_fc1, self.ffn_fc2,
        ):
            nn.init.xavier_uniform_(layer.weight)  # tf.layers.dense Glorot default
            nn.init.zeros_(layer.bias)

        # GPC_ske projections (std=1.0).
        self.gpc_f1 = nn.Parameter(_normal_(torch.empty(self.H, self.H)))
        self.gpc_f2 = nn.Parameter(_normal_(torch.empty(self.H, self.H)))

        # STPR decoders (std=1.0 weights, zero bias).
        self.struct_dec1 = nn.Linear(self.H, self.H)
        self.struct_dec2 = nn.Linear(self.H, num_joints * 3)
        self.traj_dec1 = nn.Linear(self.H, self.H // 2)
        self.traj_dec2 = nn.Linear(self.H // 2, self.num_frames * 3)
        for layer in (self.struct_dec1, self.struct_dec2, self.traj_dec1, self.traj_dec2):
            _normal_(layer.weight)
            nn.init.zeros_(layer.bias)

        if self.isrt:
            self.scale_residual_projection = nn.Linear(3, self.H, bias=False)
            if self.residual_proj_init == "random":
                # std=1.0 normal, matching tf.dense default used for the other
                # projection layers; zero-init is the faithful default.
                _normal_(self.scale_residual_projection.weight)
            else:
                nn.init.zeros_(self.scale_residual_projection.weight)
            if self.residual_gate_fixed is not None:
                # Fixed post-activation attenuation alpha (no learned gate).
                self.residual_gate_alpha = self.residual_gate_fixed
            else:
                self.residual_gate = nn.Parameter(torch.tensor(-1.5))

        # Buffers for adjacency / Laplacian positional encoding (filled lazily).
        self.register_buffer("pos_enc", torch.zeros(num_joints, enc_k), persistent=False)
        self._bones = bones
        self._enc_k = enc_k
        self._cached: dict[str, torch.Tensor] = {}

    # ---- preprocessing -------------------------------------------------
    def root_center(self, skeleton: torch.Tensor) -> torch.Tensor:
        skeleton = skeleton.float()
        return skeleton - skeleton[:, :, self.root_joint : self.root_joint + 1]

    def _residual_input(self, centered: torch.Tensor) -> torch.Tensor:
        """Input for the Node-ISRT residual branch.

        absolute (default): the root-centered coordinates C (TranSG preserves
        absolute scale -> gated re-weighting). normalized: C / s where s is the
        per-window median bone length (used by the Phase 2 parameter-matched
        control, which must NOT feed absolute scale to the residual).
        """
        if self.residual_scale_mode == "absolute":
            return centered
        left = torch.tensor(
            [edge[0] for edge in self._bones], device=centered.device
        )
        right = torch.tensor(
            [edge[1] for edge in self._bones], device=centered.device
        )
        bone_len = (centered[:, :, right] - centered[:, :, left]).norm(dim=-1)  # [B,T,E]
        s = bone_len.reshape(bone_len.shape[0], -1).median(dim=1).values  # [B]
        s = s.clamp_min(1e-6)
        return centered / s[:, None, None, None]

    def _laplacian_pos_enc(self, device: torch.device) -> torch.Tensor:
        # Official process_cme_SG.py: plain bone adjacency (NO self-loops),
        # L = I - D^-1/2 A D^-1/2, eigenvectors sorted by eigenvalue, cols 1..k.
        # The eigendecomposition is ALWAYS computed on CPU. Symmetric skeleton
        # limbs give repeated eigenvalues, and inside those degenerate subspaces
        # CPU (LAPACK) and GPU (MAGMA) eigh return different orthonormal bases
        # that a per-vector sign flip cannot reconcile; using GPU eigh therefore
        # changed the PE and eval mAP across devices. CPU LAPACK returns a
        # deterministic basis for a given library, so computing here on CPU and
        # moving the result pins the positional encoding across devices.
        work = torch.device("cpu")
        adj = torch.zeros(self.num_joints, self.num_joints, device=work)
        for left, right in self._bones:
            adj[left, right] = 1.0
            adj[right, left] = 1.0
        degree = adj.sum(dim=1).clamp_min(1e-12)
        inv_sqrt = degree.rsqrt()
        norm_adj = inv_sqrt[:, None] * adj * inv_sqrt[None, :]
        laplacian = torch.eye(self.num_joints, device=work) - norm_adj
        _, eigvecs = torch.linalg.eigh(laplacian)  # ascending eigenvalues
        # Sign-canonicalize each eigenvector: flip so the largest-magnitude entry
        # is positive. eigh eigenvectors are sign-non-unique across libraries, so
        # this rule makes the sign deterministic even when two solvers agree on
        # the degenerate-subspace basis.
        absmax_idx = eigvecs.abs().argmax(dim=0)
        sign = torch.sign(eigvecs[absmax_idx, torch.arange(self.num_joints)])
        sign[sign == 0] = 1
        eigvecs = eigvecs * sign[None, :]
        return eigvecs[:, 1 : self._enc_k + 1].to(device).contiguous()

    # ---- forward -------------------------------------------------------
    def encode(self, skeleton: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return intermediate representations. skeleton: [B, 6, J, 3] raw."""
        B, T, J, _ = skeleton.shape
        centered = self.root_center(skeleton)  # TranSG: root-center only, keep scale.
        h = self.input_fc2(F.relu(self.input_fc1(centered)))  # [B,T,J,H]
        if self.isrt:
            gate = (
                torch.as_tensor(self.residual_gate_alpha, device=h.device)
                if self.residual_gate_fixed is not None
                else torch.sigmoid(self.residual_gate)
            )
            h = h + gate * self.scale_residual_projection(
                self._residual_input(centered)
            )
        pos = self.pos_proj(self.pos_enc.to(h.device))  # [J,H]
        h = h + pos[None, None]
        seq_ftr_input = h  # official: residual source is the pos-enc-ADDED embedding.

        # FR attention layers (no residual/norm/FFN inside/between layers).
        for layer in range(self.num_layers):
            Q = torch.einsum("btjh,hnd->btjnd", h, self.W_Q[layer])
            K = torch.einsum("btjh,hnd->btjnd", h, self.W_K[layer])
            V = torch.einsum("btjh,hnd->btjnd", h, self.W_V[layer])
            scores = torch.einsum("btjnd,btknd->btnjk", Q, K) / (self.head_dim ** 0.5)
            scores = scores.clamp(-5.0, 5.0)
            attn = torch.softmax(scores, dim=-1)  # [B,T,heads,J,J] shared across channels
            aggr = torch.einsum("btnjk,btknd->btjnd", attn, V)  # [B,T,J,heads,head_dim]
            h = aggr.reshape(B, T, J, self.H)

        # Post-attention block (once).
        h = self.post_dropout(h)
        h = self.post_dense(h)
        h = h + seq_ftr_input
        h = self.bn1(h.reshape(B * T * J, self.H)).reshape(B, T, J, self.H)
        h_res2 = h
        h = self.ffn_fc2(self.ffn_drop(F.relu(self.ffn_fc1(h))))
        h = h_res2 + h
        h = self.bn2(h.reshape(B * T * J, self.H)).reshape(B, T, J, self.H)

        frame_ftr = h.mean(dim=2)  # [B,T,H] node->frame
        seq_ftr = frame_ftr.mean(dim=1)  # [B,H] sequence embedding
        return {
            "node_ftr": h,  # [B,T,J,H]
            "frame_ftr": frame_ftr,  # [B,T,H]
            "seq_ftr": seq_ftr,  # [B,H]
            "centered": centered,  # [B,T,J,3]
        }

    def forward(self, skeleton: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.encode(skeleton)["seq_ftr"], dim=-1)

    # ---- losses --------------------------------------------------------
    def gpc_seq(self, seq_ftr: torch.Tensor, labels: torch.Tensor, class_ftr: torch.Tensor) -> torch.Tensor:
        all_ftr = F.normalize(seq_ftr, dim=-1)
        cluster = F.normalize(class_ftr, dim=-1)
        logits = all_ftr @ cluster.t() / 0.07
        return F.cross_entropy(logits, labels)

    def gpc_ske(self, frame_ftr: torch.Tensor, labels: torch.Tensor, class_ftr: torch.Tensor) -> torch.Tensor:
        B, T, H = frame_ftr.shape
        all_trans = frame_ftr @ self.gpc_f1  # [B,T,H]
        cluster_trans = class_ftr @ self.gpc_f2  # [C,H]
        logits = all_trans @ cluster_trans.t() / 14.0  # [B,T,C]
        label_frames = labels[:, None].expand(-1, T)  # [B,T]
        ce = F.cross_entropy(
            logits.reshape(B * T, -1), label_frames.reshape(-1), reduction="none"
        ).reshape(B, T)
        return ce.sum(dim=-1).mean()  # sum over frames, mean over batch.

    def stpr(
        self,
        node_ftr: torch.Tensor,
        centered: torch.Tensor,
        node_mask: torch.Tensor,
        seq_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, J, H = node_ftr.shape
        gt_pos = centered.reshape(B, T, J * 3)
        # Structure: keep J - mask_num joints.
        kept = node_ftr[:, :, node_mask, :]  # [B,T,J-kept,H]
        g_h = kept.mean(dim=2)  # [B,T,H]
        pred = self.struct_dec2(F.relu(self.struct_dec1(g_h)))  # [B,T,J*3]
        struct_loss = (pred - gt_pos).abs().mean() / B
        # Trajectory: keep T - mask frames, per-joint reconstruct full trajectory.
        part = node_ftr[:, seq_mask, :, :]  # [B,T-kept,J,H]
        part = part.permute(0, 2, 1, 3)  # [B,J,T-kept,H]
        part = part.mean(dim=2)  # [B,J,H]
        pred_t = self.traj_dec2(F.relu(self.traj_dec1(part)))  # [B,J,T*3]
        gt_t = gt_pos.reshape(B, T, J, 3).permute(0, 2, 1, 3).reshape(B, J, T * 3)
        traj_loss = (pred_t - gt_t).abs().mean() / B
        return struct_loss, traj_loss

    def total_loss(
        self,
        skeleton: torch.Tensor,
        labels: torch.Tensor,
        class_ftr: torch.Tensor,
        node_mask: torch.Tensor,
        seq_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        rep = self.encode(skeleton)
        gpc = 0.5 * self.gpc_ske(rep["frame_ftr"], labels, class_ftr) + 0.5 * self.gpc_seq(
            rep["seq_ftr"], labels, class_ftr
        )
        struct_loss, traj_loss = self.stpr(rep["node_ftr"], rep["centered"], node_mask, seq_mask)
        recon = 0.5 * struct_loss + 0.5 * traj_loss
        total = 0.5 * gpc + 0.5 * recon
        return total, {
            "gpc": gpc.detach(),
            "recon": recon.detach(),
            "structure": struct_loss.detach(),
            "trajectory": traj_loss.detach(),
        }
