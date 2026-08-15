# BIWI/IAS paired-view shared-training-set audit (2026-08-13)

Discovered while verifying the `norm_matched` arm's RMS-matching factor: the
release's two BIWI views and two IAS views share **byte-identical training
windows**, differing only in gallery/probe role assignment.

## Evidence (SHA-256, 16-hex prefixes)

| file | BIWI-S | BIWI-W | match |
|---|---|---|---|
| train_skeleton.npy | `4e4be9dd24f9ae28` | `4e4be9dd24f9ae28` | SAME |
| train_identity.npy | `435bca0e58ea36db` | `435bca0e58ea36db` | SAME |
| train_recording.npy | `01c69b1fe602f2c8` | `01c69b1fe602f2c8` | SAME |

| file | IAS-A | IAS-B | match |
|---|---|---|---|
| train_skeleton.npy | `21d1a6a260765d55` | `21d1a6a260765d55` | SAME |
| train_identity.npy | `c3e61227ab295137` | `c3e61227ab295137` | SAME |
| train_recording.npy | `13a8dcc1ef526c2c` | `13a8dcc1ef526c2c` | SAME |

Gallery/probe roles are swapped between the paired views (BIWI-S gallery =
Walking 822, probe = Still 531; BIWI-W gallery = Still 531, probe = Walking 822;
IAS-A/B likewise swap A/B roles). `protocol.json` differs because the role
assignment and split metadata differ.

## Consequence for the paper

- The six-row protocol set uses **four distinct training sets** (BIWI pair,
  IAS pair, KGBD official, KGBD duplicate-safe).
- The BIWI and IAS rows are paired evaluation protocols over a shared training
  set, not independent training sets. This strengthens the paired comparison
  (same trained features, different gallery/probe roles) but must be disclosed
  so "five protocols" is not read as five independent training corpora.
- It also explains why the `norm_matched` RMS-matching factor γ is identical
  within each pair.

## norm_matched RMS-matching factors γ = mean_RMS(C)/mean_RMS(N)

Recomputed on CPU with the training-driver code path
(`compute_norm_match_factor`, `scripts/mva_research/train_evaluate.py`):

| protocol | n_train | γ |
|---|---:|---:|
| biwi_s | 34,294 | 0.213295 |
| biwi_w | 34,294 | 0.213295 |
| ias_a | 14,831 | 0.253605 |
| ias_b | 14,831 | 0.253605 |
| kgbd (official) | 31,573 | 0.211182 |
| kgbd_dedup | 31,486 | 0.211209 |

Repro: `experiments/frame_scale_controls_20260813/scripts/verify_norm_match_factors.py`
and `verify_ias_kgbddedup.sh`.
