"""Export every Frame-track cell (protocol, arm, model, seed -> mAP) as CSV.

Reads from the same sources as pooled_frame_stats.py:
  frozen Table 1 (gru/isrgru/transformer/isr_transformer, seeds 42/123/2026)
  normalized arm (scale_normalization_ablation/results)
  norm_matched / random_matched arms (scale_content_controls/results)
  5-seed extension (seed_extension/results) for seeds 2027/2028

Every row includes the provenance result path so the listing is auditable.
Cells that are missing are omitted; the row count therefore reflects what has
completed. Used for the P1-4 "per-seed results" reproducible-package item.

Usage:
  python3 \
    scripts/export_per_seed_results.py [--out docs/per_seed_results_frame.csv]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from frozen_matrix import (  # noqa: E402
    PREFERRED_SOURCE,
    ROOT as FROZEN_ROOT,
    SCAN_ROOTS,
)

from frozen_matrix import ALLOWED_SHA  # noqa: E402

EXP = Path(os.environ.get("FRAME_ISRT_SCALE_CONTROLS_ROOT", str(Path(__file__).resolve().parents[1] / "experiments" / "scale_content_controls")))
NEW_ROOT = EXP / "results"
NORM_ROOT = Path(
    os.environ.get("FRAME_ISRT_NORM_ROOT", str(Path(__file__).resolve().parents[1] / "experiments" / "scale_normalization_ablation" / "results"))
)
SEED_EXTEND_ROOT = Path(
    os.environ.get("FRAME_ISRT_SEED_EXTEND_ROOT", str(Path(__file__).resolve().parents[1] / "experiments" / "seed_extension" / "results"))
)
PROTOS = ["biwi_s", "biwi_w", "ias_a", "ias_b", "kgbd", "kgbd_dedup"]
ARMS = [
    ("baseline", ("gru", "transformer")),
    ("absolute", ("isrgru", "isr_transformer")),
    ("normalized", ("isrgru_norm", "isr_transformer_norm")),
    ("norm_matched", ("isrgru_normmatch", "isr_transformer_normmatch")),
    ("random_matched", ("isrgru_randommatch", "isr_transformer_randommatch")),
]
OLD_SEEDS = [42, 123, 2026]
NEW_SEEDS = [2027, 2028]
SHA_TO_LABEL = {sha: label for label, shas in ALLOWED_SHA.items() for sha in shas}


def pick_frozen_cell(label, model, seed):
    """Same picker logic as frozen_matrix.py (preferred source, then sorted fallback)."""
    cands = []
    for scan in SCAN_ROOTS:
        scan_root = FROZEN_ROOT / scan
        if not scan_root.exists():
            continue
        for rp in scan_root.rglob("result.json"):
            if "_d64_" in str(rp):
                continue
            try:
                payload = json.loads(rp.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            if payload.get("model") != model or payload.get("seed") != seed:
                continue
            if SHA_TO_LABEL.get((payload.get("protocol_sha256") or "").lower()[:12]) != label:
                continue
            mAP = (payload.get("window") or {}).get("mAP")
            if mAP is None:
                continue
            cands.append({"mAP": mAP, "path": str(rp.relative_to(FROZEN_ROOT))})
    preferred = PREFERRED_SOURCE.get((label, model))
    if preferred:
        match = [c for c in cands if preferred in c["path"]]
        if match:
            return match[0]
    return sorted(cands, key=lambda c: c["path"])[-1] if cands else None


def load_new_cell(root, proto, model, seed):
    rp = root / proto / model / f"seed{seed}" / "result.json"
    if not rp.exists():
        return None
    try:
        payload = json.loads(rp.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    mAP = (payload.get("window") or {}).get("mAP")
    if mAP is None:
        return None
    return {"mAP": mAP, "path": str(rp.relative_to(root))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).resolve().parents[1] / "results" / "per_seed_frame.csv")
    args = parser.parse_args()

    rows = []
    for proto in PROTOS:
        for arm, (base_name, isr_name) in ARMS:
            for seed in OLD_SEEDS:
                for model in (base_name, isr_name):
                    if arm == "baseline":
                        cell = pick_frozen_cell(proto, model, seed)
                    elif arm == "absolute":
                        cell = pick_frozen_cell(proto, model, seed)
                    elif arm == "normalized":
                        cell = load_new_cell(NORM_ROOT, proto, model, seed)
                    elif arm == "norm_matched":
                        cell = load_new_cell(NEW_ROOT, proto, model, seed)
                    else:
                        cell = load_new_cell(NEW_ROOT, proto, model, seed)
                    if cell:
                        rows.append({
                            "protocol": proto, "arm": arm, "model": model,
                            "seed": seed, "mAP": f"{cell['mAP']*100:.4f}",
                            "path": cell["path"],
                        })
            for seed in NEW_SEEDS:
                for model in (base_name, isr_name):
                    cell = load_new_cell(SEED_EXTEND_ROOT, proto, model, seed)
                    if cell:
                        rows.append({
                            "protocol": proto, "arm": arm, "model": model,
                            "seed": seed, "mAP": f"{cell['mAP']*100:.4f}",
                            "path": cell["path"],
                        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["protocol", "arm", "model", "seed", "mAP", "path"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
