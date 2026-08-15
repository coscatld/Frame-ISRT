# Scale-content controls

The central experiment of the paper. Four parameter-identical arms differ only
in the *content* of the identity-scale residual input:

- `absolute` (C): root-centered coordinates — full absolute scale.
- `normalized` (C/s): median-bone-length normalized coordinates — scale removed.
- `norm_matched` (N·γ): normalized coordinates re-scaled to the training-set
  absolute RMS/variance — magnitude matched, no per-sample scale.
- `random_matched`: per-sample RMS-matched Gaussian noise — no scale content.

All arms train ISR-GRU / ISR-Transformer / ISR-Mamba over 6 protocols × 5 seeds
per backbone.

## Layout

- `scripts/` — run + analysis scripts:
  - `pooled_frame_stats_cluster.py` — **cluster-aware** pooled statistics
    (40 clusters / 60 cells; block bootstrap CI, sign-flip permutation p,
    cluster-mean t-test) — the numbers reported in the manuscript §4.5.1.
  - `pooled_frame_stats.py` — cell-iid pooled statistics (historical reference).
  - `cross_arm_paired_stats.py` — cross-arm paired deltas.
  - `arm_backbone_interaction.py` — arm × backbone interaction (ANOVA +
    permutation).
- `aggregate/` — generated outputs: `pooled_frame_stats_cluster.md`,
  `interaction_summary.md`, and the per-cell/per-summary CSVs.

## Regenerate

```bash
# reads results/per_seed_frame.csv, writes aggregate/pooled_frame_stats_cluster.md
python experiments/scale_content_controls/scripts/pooled_frame_stats_cluster.py
python experiments/scale_content_controls/scripts/cross_arm_paired_stats.py
python experiments/scale_content_controls/scripts/arm_backbone_interaction.py
```

## Main result

Only the `absolute` arm improves mAP: **+0.83 pp** (cluster-bootstrap 95% CI
[+0.54, +1.16], cluster permutation p < 0.0001, 52/60 cells). All three control
arms are ≈0.
