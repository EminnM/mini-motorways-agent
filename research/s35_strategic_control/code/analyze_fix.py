"""Planner-variant comparison on the 10 S35 seeds: planner B vs destination fix vs route objective lambda. Paired per seed; weeks survived; concrete left; game-over state
(unreached destinations at the end, overcrowded destination); roads built; network size.   python research/s35_strategic_control/code/analyze_fix.py"""
import glob
import gzip
import json
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
S35 = os.path.dirname(HERE)
RNG = np.random.default_rng(355)


def paired(a, b):
    seeds = sorted(set(a) & set(b))
    d = np.array([a[s] - b[s] for s in seeds], dtype=float)
    boot = [RNG.choice(d, len(d)).mean() for _ in range(10000)]
    return float(d.sum()), int((d > 0).sum()), int((d == 0).sum()), int((d < 0).sum()), [round(float(np.percentile(boot, 2.5)), 1), round(float(np.percentile(boot, 97.5)), 1)], [round(float(x)) for x in d]


def main():
    by = defaultdict(dict)
    for p in glob.glob(os.path.join(S35, "eval", "*.json.gz")):
        e = json.load(gzip.open(p, "rt"))
        if e["tag"].startswith("plannerB"):
            by[e["tag"]][e["seed"]] = e
    B = by["plannerB"]
    print(f"{'condition':18s} {'deliv':>6} {'len med':>7} {'weeks med':>9} {'surv':>4} {'roads built':>11} {'concrete left':>13} {'end: all houses conn':>20} {'end: unreached dests':>20} {'over-death':>10}")
    for tag, eps in sorted(by.items()):
        E = list(eps.values())
        over = sum(1 for e in E if any(d["over"] for d in e["dest_state_end"]))
        print(f"{tag:18s} {sum(e['score'] for e in E):6.0f} {np.median([e['t'] for e in E]):7.0f} {np.median([e['t'] for e in E]) / 140:9.1f} {sum(1 for e in E if not e['terminated']):4d} {sum(e['kinds']['ADD'] for e in E):11d} {np.mean([e['concrete_end'] for e in E]):13.1f} {sum(1 for e in E if e['connected_end'] == e['houses_end']):20d} {'n/a':>20} {over:10d}")
    print()
    for tag, eps in sorted(by.items()):
        if tag == "plannerB" or len(eps) < 10:
            continue
        s, h, eq, lo, ci, d = paired({k: e["score"] for k, e in eps.items()}, {k: e["score"] for k, e in B.items()})
        s2, h2, eq2, lo2, ci2, d2 = paired({k: e["t"] for k, e in eps.items()}, {k: e["t"] for k, e in B.items()})
        print(f"{tag:18s} vs plannerB: deliveries {s:+6.0f} H/E/L {h}/{eq}/{lo} CI {ci} diffs {d} | length {s2:+6.0f} H/E/L {h2}/{eq2}/{lo2} diffs {d2}")
    # attribution: lam4 alone vs fix alone vs both
    for a_, b_ in (("plannerBfix_lam4", "plannerBfix"), ("plannerBfix_lam1", "plannerBfix"), ("plannerB_lam4", "plannerB")):
        if a_ in by and b_ in by and len(by[a_]) >= 10:
            s, h, eq, lo, ci, d = paired({k: e["score"] for k, e in by[a_].items()}, {k: e["score"] for k, e in by[b_].items()})
            print(f"{a_:18s} vs {b_:14s}: deliveries {s:+6.0f} H/E/L {h}/{eq}/{lo} CI {ci} diffs {d}")
    # per-seed table
    tags = [t for t in ("plannerB", "plannerBfix", "plannerBfix_lam4", "plannerBfix_lam1", "plannerB_lam4") if t in by]
    print("\nseed      " + " ".join(f"{t[8:] or 'B':>14}" for t in tags))
    for s in sorted(B):
        print(f"{s}  " + " ".join((f"{by[t][s]['score']:4.0f}/{by[t][s]['t']:3d}/c{by[t][s]['concrete_end']:2d}" if s in by[t] else "            -") for t in tags))


if __name__ == "__main__":
    main()
