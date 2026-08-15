# Scale-normalization ablation

Ablates the `normalized` residual arm (C/s) against the `absolute` arm (C)
under the same ISR-GRU / ISR-Transformer architecture. This is the comparison
that isolates *per-sample scale content* from the shared coordinate
normalization.

Aggregates: `aggregate/` (per-cell results, per-arm summary, grand summary).

Combined with the scale-content controls, the four arms form the manuscript's
P1-1 control set:

| arm | residual input | pooled ΔmAP |
|---|---:|---:|
| absolute (C) | root-centered | +0.83 pp (p < 0.0001) |
| normalized (C/s) | median-bone-length normalized | +0.16 pp (p = 0.27) |
| norm_matched (N·γ) | RMS-matched normalized | +0.01 pp (p = 0.95) |
| random_matched | RMS-matched noise | +0.06 pp (p = 0.68) |
