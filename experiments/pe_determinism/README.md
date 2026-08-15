# Deterministic Laplacian positional encoding

Fixes the P1-3 issue where the TranSG Laplacian positional encoding differed
between CPU and GPU.

## Cause

The skeleton graph has symmetric limbs that give repeated eigenvalues. Inside
those degenerate eigenspaces, CPU (LAPACK) and GPU (MAGMA) `torch.linalg.eigh`
return different orthonormal bases, so the learned PE projection became
device-dependent.

## Fix

In `frame_isrt/transg_faithful.py` (`_laplacian_pos_enc`):

1. the eigendecomposition always runs on **CPU** (fixed LAPACK basis), and
2. each eigenvector is sign-canonicalized so its largest-magnitude entry is
   positive.

CPU vs GPU now produce byte-identical positional encodings (max absolute
difference 0.0); the pre-fix raw `eigh` differed by up to 0.94.

## Re-verification

BIWI-S official-reproduction base retrained under the fixed PE, plus
`check_pe_device_identity` diagnostics (developer-local runs).
