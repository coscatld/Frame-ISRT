## Frame-track content-matched residual control — normalized vs absolute

Protocol×backbone×seed cells: 36. mAP in pp; Δ vs same-seed baseline.

| protocol | backbone | base | absolute | normalized | Δ_abs | Δ_norm | Δ_abs−Δ_norm |
|---|---|---|---|---|---|---|---|
| biwi_s | gru | 19.88 | 21.00 | 20.03 | +1.12±0.56 | +0.16±0.55 | +0.97±0.05 |
| biwi_s | transformer | 16.84 | 19.26 | 18.88 | +2.42±1.28 | +2.04±1.19 | +0.38±0.09 |
| biwi_w | gru | 24.00 | 25.30 | 23.77 | +1.30±0.94 | -0.23±0.79 | +1.53±0.16 |
| biwi_w | transformer | 20.45 | 21.72 | 21.08 | +1.26±2.19 | +0.62±1.86 | +0.64±0.34 |
| ias_a | gru | 23.60 | 23.82 | 23.61 | +0.22±0.61 | +0.01±0.78 | +0.21±0.19 |
| ias_a | transformer | 28.15 | 28.58 | 28.10 | +0.43±1.13 | -0.05±0.91 | +0.48±0.27 |
| ias_b | gru | 22.95 | 23.11 | 22.80 | +0.17±0.56 | -0.14±0.66 | +0.31±0.11 |
| ias_b | transformer | 27.26 | 27.42 | 26.99 | +0.16±0.67 | -0.27±0.61 | +0.43±0.14 |
| kgbd | gru | 16.36 | 17.02 | 16.20 | +0.66±0.32 | -0.16±0.30 | +0.82±0.04 |
| kgbd | transformer | 23.24 | 24.28 | 23.36 | +1.04±0.41 | +0.12±0.12 | +0.92±0.30 |
| kgbd_dedup | gru | 16.27 | 16.91 | 16.10 | +0.63±0.31 | -0.17±0.29 | +0.80±0.04 |
| kgbd_dedup | transformer | 23.04 | 24.13 | 23.20 | +1.09±0.44 | +0.16±0.19 | +0.93±0.29 |

**Grand mean over 36 cells:** Δ_abs +0.88±1.00pp (31/36 positive) | Δ_norm +0.17±0.92pp (21/36 positive) | Δ_abs−Δ_norm +0.70±0.40pp (36/36 cells abs>norm).

**Interpretation note:** Direction: absolute-scale content contributes. Parameter- and gate-matched, the absolute-scale residual (input C) beats the normalized residual (input C/s) in every cell (Δ_abs−Δ_norm > 0 across all cells); the gap is largest where ISR gains are largest (BIWI-S/W). The residual input content, not the extra parameterization alone, drives the benefit.

Gate sanity: learned σ(g) means are close across arms (absolute 0.203, normalized 0.195), i.e. the normalized residual is not inert — both branches are equally active and equally parameterized; only the residual input (C vs C/s) differs.
