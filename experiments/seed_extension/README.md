# Seed extension

Extends the Frame track from 3 to 5 seeds. For every protocol × model type a
config copy whose only change is the extended seed list (2027, 2028) is
retrained, so the frozen Table 1 cells (42/123/2026) are untouched.

This raises the scale-content-control pool to **60 cells** per arm
(6 protocols × 2 backbones × 5 seeds). The cluster-aware analysis in
`../scale_content_controls/` consumes the combined 5-seed pool.

## Main effect of extension

Adding seeds 2027/2028 does not change the conclusions: pooled over 60 cells,
only the `absolute` arm is significant (+0.83 pp, p < 0.0001).

## Per-seed results

The combined 5-seed results live in `results/per_seed_frame.csv` at the
repository root.
