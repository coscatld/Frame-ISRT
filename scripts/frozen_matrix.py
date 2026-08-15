"""Build the FROZEN Table 1 protocol x backbone x seed mAP matrix.

Enforces per-protocol protocol_sha256 so no cell mixes protocols.
  official BIWI-S  -> transg_official_biwi_s_f6 (02902688ec0d)
  official BIWI-W  -> transg_official_biwi_w_f6 (0f401ae7fd3c)
  official IAS-A   -> transg_official_ias_a_f6  (62c2a2d89f77)
  official IAS-B   -> transg_official_ias_b_f6  (41f7b981b980)
  official KGBD    -> transg_official_kgbd_f6_layoutfix_labels (7ce9607867ff)
  dedup KGBD       -> transg_official_kgbd_f6_dedup (5cc4c2c86ac5)
The audited BIWI protocol (biwi_ds6_audited, ac0f0b5921cb) is a SEPARATE
sequence-split audit protocol and is deliberately excluded from Table 1.
"""
from __future__ import annotations

import json
import os
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(os.environ.get("FRAME_ISRT_FROZEN_ROOT", "")) or Path(__file__).resolve().parents[1] / "results" / "frozen"
SCAN_ROOTS = [
    "isr_transfer_pilot_20260805",
    "isrgru_final_matrix_20260808",
    "isrgru_pos_fix_20260808",
    "transg_official_ias_confirm_20260808",
    "transg_official_kgbd_confirm_20260808",
    "transg_official_biwi_20260808",
    "transg_official_biwi_layoutfix_20260808",
    "biwi_audited_confirm_20260806",
    "kgbd_dedup_20260806/confirm_train_removal",
]
# protocol_label -> set of allowed protocol_sha256 prefixes (must match full 64-hex prefix)
ALLOWED_SHA = {
    "biwi_s": {"02902688ec0d"},
    "biwi_w": {"0f401ae7fd3c"},
    "ias_a": {"62c2a2d89f77"},
    "ias_b": {"41f7b981b980"},
    "kgbd": {"7ce9607867ff"},
    "kgbd_dedup": {"5cc4c2c86ac5"},
}
MODEL_ORDER = ["gru", "isrgru", "transformer", "isr_transformer", "mamba", "isr_mamba"]


SHA_TO_LABEL: dict[str, str] = {
    sha: label for label, shas in ALLOWED_SHA.items() for sha in shas
}

# Explicit source preference per (protocol, model): result dirs whose path
# contains the token win when multiple batches share the same protocol sha
# (e.g. biwi_s mamba exists in both transg_official_biwi_layoutfix and
# isr_transfer_pilot; layoutfix is the official confirm re-run).
PREFERRED_SOURCE: dict[tuple[str, str], str] = {
    ("biwi_s", "gru"): "biwi_layoutfix",
    ("biwi_s", "gctr"): "biwi_layoutfix",
    ("biwi_s", "stgcn"): "biwi_layoutfix",
    ("biwi_s", "mamba"): "biwi_layoutfix",
    ("biwi_s", "transformer"): "biwi_layoutfix",
    ("biwi_s", "isrgru"): "isrgru_pos_fix",
    ("biwi_s", "isr_transformer"): "isr_transfer_pilot",
    ("biwi_s", "isr_mamba"): "isr_transfer_pilot",
    ("biwi_w", "gru"): "biwi_layoutfix",
    ("biwi_w", "gctr"): "biwi_layoutfix",
    ("biwi_w", "stgcn"): "biwi_layoutfix",
    ("biwi_w", "mamba"): "biwi_layoutfix",
    ("biwi_w", "transformer"): "biwi_layoutfix",
    ("biwi_w", "isrgru"): "isrgru_pos_fix",
    ("biwi_w", "isr_transformer"): "isr_transfer_pilot",
    ("biwi_w", "isr_mamba"): "isr_transfer_pilot",
    ("ias_a", "gru"): "ias_confirm",
    ("ias_a", "grgru"): "ias_confirm",
    ("ias_a", "isrgru"): "isrgru_pos_fix",
    ("ias_b", "gru"): "ias_confirm",
    ("ias_b", "grgru"): "ias_confirm",
    ("ias_b", "isrgru"): "isrgru_pos_fix",
    ("kgbd", "gru"): "kgbd_confirm",
    ("kgbd", "rcgru"): "kgbd_confirm",
    ("kgbd", "isrgru"): "isrgru_pos_fix",
    ("kgbd_dedup", "isrgru"): "isrgru_pos_fix",
}


def main() -> None:
    cells: dict[str, dict[str, dict[int, dict]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(dict))
    )
    candidates: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for scan in SCAN_ROOTS:
        scan_root = ROOT / scan
        if not scan_root.exists():
            continue
        for result_path in scan_root.rglob("result.json"):
            if "_d64_" in str(result_path):
                continue  # d_state=64 discrimination runs, not Table 1 cells
            try:
                payload = json.loads(result_path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            model = payload.get("model")
            seed = payload.get("seed")
            layout = payload.get("model_layout")
            window = payload.get("window") or {}
            mAP = window.get("mAP")
            sha = (payload.get("protocol_sha256") or "").lower()[:12]
            if model is None or seed is None or mAP is None:
                continue
            label = SHA_TO_LABEL.get(sha)
            if label is None:
                continue
            rel = str(result_path.relative_to(ROOT))
            candidates[(label, model, seed)].append(
                {"mAP": mAP, "layout": layout, "path": rel}
            )

    for (label, model, seed), cands in candidates.items():
        preferred = PREFERRED_SOURCE.get((label, model))
        pick = None
        if preferred:
            match = [c for c in cands if preferred in c["path"]]
            if match:
                pick = match[0]
        if pick is None:
            pick = sorted(cands, key=lambda c: c["path"])[-1]  # deterministic fallback
        cells[label][model][seed] = pick

    # print per-protocol matrix
    print(f"{'protocol':12s} " + "  ".join(f"{m:20s}" for m in MODEL_ORDER))
    for proto in ["biwi_s", "biwi_w", "ias_a", "ias_b", "kgbd", "kgbd_dedup"]:
        line = f"{proto:12s} "
        for model in MODEL_ORDER:
            seeds = cells[proto].get(model, {})
            if not seeds:
                line += f"{'--':20s} "
                continue
            vals = [seeds[s]["mAP"] for s in sorted(seeds)]
            mean = statistics.mean(vals) * 100
            std = statistics.stdev(vals) * 100 if len(vals) > 1 else 0.0
            line += f"{mean:5.2f}±{std:<13.2f} "
        print(line)

    # ISRT deltas
    print("\nISRT deltas (ISR - base), mean mAP pp:")
    for proto in ["biwi_s", "biwi_w", "ias_a", "ias_b", "kgbd", "kgbd_dedup"]:
        deltas = []
        for base, isr in [("gru", "isrgru"), ("transformer", "isr_transformer"),
                          ("mamba", "isr_mamba")]:
            b = cells[proto].get(base)
            i = cells[proto].get(isr)
            if not b or not i:
                continue
            d = []
            for s in sorted(set(b) & set(i)):
                d.append((i[s]["mAP"] - b[s]["mAP"]) * 100)
            m = statistics.mean(d)
            pos = sum(1 for x in d if x > 0)
            deltas.append(f"{base}->{isr}: {m:+.2f}pp ({pos}/{len(d)})")
        print(f"{proto:12s} " + "; ".join(deltas))

    # seed detail for KGBD dedup row (audit trail)
    print("\nKGBD dedup seed detail:")
    for model in MODEL_ORDER:
        seeds = cells["kgbd_dedup"].get(model, {})
        if seeds:
            print(f"  {model:16s} " + ", ".join(
                f"s{s}={seeds[s]['mAP']*100:.2f}" for s in sorted(seeds)))
    print("\nOfficial KGBD seed detail:")
    for model in MODEL_ORDER:
        seeds = cells["kgbd"].get(model, {})
        if seeds:
            print(f"  {model:16s} " + ", ".join(
                f"s{s}={seeds[s]['mAP']*100:.2f}" for s in sorted(seeds)))

    print("\nCell provenance (protocol, model, seed -> result path):")
    for label in sorted(cells):
        for model in sorted(cells[label]):
            for seed in sorted(cells[label][model]):
                p = cells[label][model][seed]["path"]
                print(f"  {label:11s} {model:16s} s{seed:<5d} -> {p}")


if __name__ == "__main__":
    main()
