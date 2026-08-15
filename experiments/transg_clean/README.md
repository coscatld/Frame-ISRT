# TranSG clean reproduction

Clean-reproduction protocol for TranSG (Rao & Miao, CVPR 2023) and the Node
(ISR-TranSG) boundary track, following the original TF1.14 training loop
(GPC + STPR objectives, class-balanced resampling, probe-Rank-1 model
selection).

The faithful PyTorch model is `frame_isrt/transg_faithful.py`; the training
loop is `scripts/train_transg_faithful.py`.

## Phase reports and audits

- `docs/audits/BIWI_S_REPRODUCTION_AUDIT.md` — bit-exact BIWI-S base
  reproduction.
- `docs/audits/CLEAN_PROTOCOL_AUDIT.md` — duplicate-safe protocol definition.
- `docs/audits/kgbd_data_count_audit_v2.json` — KGBD array/count audit.

## Deterministic positional encoding

The Laplacian positional encoding is pinned by a CPU eigendecomposition plus a
sign-canonicalization rule (see `../pe_determinism/`); CPU vs GPU produce
byte-identical encodings.
