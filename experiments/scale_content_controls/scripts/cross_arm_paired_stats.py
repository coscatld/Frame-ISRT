"""5-seed paired cross-arm deltas for the scale-content controls (P1-2).

For every (protocol, backbone, seed) cell, computes each arm's delta vs the
same-cell baseline, then reports how often absolute beats norm_matched /
random_matched plus the paired deltas. Mirrors P1-1_REPORT.md §4.2 at n=60.
"""
from __future__ import annotations

import math
import statistics
import sys

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pooled_frame_stats as pfs  # noqa: E402


def cell_delta(proto, backbone_row, seed, idx):
    """Delta in pp for one (proto, backbone_row, seed, arm-index) cell; None if missing."""
    if seed in pfs.OLD_SEEDS:
        base = pfs.pick_frozen_cell(proto, backbone_row[0], seed)
        if not base:
            return None
        b = base["mAP"] * 100
    else:
        c = pfs.load_new_cell(pfs.SEED_EXTEND_ROOT, proto, backbone_row[0], seed)
        if c is None:
            return None
        b = c * 100
    model = backbone_row[idx]
    if seed in pfs.OLD_SEEDS:
        if idx == 1:
            cell = pfs.pick_frozen_cell(proto, model, seed)
            m = cell["mAP"] * 100 if cell else float("nan")
        elif idx == 2:
            cell = pfs.load_new_cell(pfs.NORM_ROOT, proto, model, seed)
            m = cell * 100 if cell is not None else float("nan")
        else:
            cell = pfs.load_new_cell(pfs.NEW_ROOT, proto, model, seed)
            m = cell * 100 if cell is not None else float("nan")
    else:
        cell = pfs.load_new_cell(pfs.SEED_EXTEND_ROOT, proto, model, seed)
        m = cell * 100 if cell is not None else float("nan")
    if not math.isfinite(m):
        return None
    return m - b


def main():
    cells = []
    for proto in pfs.PROTOS:
        for backbone_row in pfs.BACKBONES:
            for seed in pfs.SEEDS:
                ds = {}
                ok = True
                for arm, (idx,) in pfs.ARMS.items():
                    d = cell_delta(proto, backbone_row, seed, idx)
                    if d is None:
                        ok = False
                        break
                    ds[arm] = d
                if ok:
                    cells.append(ds)
    n = len(cells)
    if n == 0:
        print("no cells")
        return
    for pair, key_a, key_b in [
        ("abs vs normmatch", "abs", "normmatch"),
        ("abs vs random", "abs", "random"),
    ]:
        diff = [c[key_a] - c[key_b] for c in cells]
        pos = sum(1 for d in diff if d > 0)
        print(f"{pair}: mean_delta={statistics.mean(diff):+.3f}pp "
              f"sd={statistics.stdev(diff):.3f} pos={pos}/{n}")
    for arm in ("abs", "norm", "normmatch", "random"):
        d = [c[arm] for c in cells]
        print(f"{arm:9s} n={len(d)} mean={statistics.mean(d):+.3f}pp "
              f"sd={statistics.stdev(d):.3f} pos={sum(1 for x in d if x > 0)}/{len(d)}")
    print(f"total cells={n}")


if __name__ == "__main__":
    main()
