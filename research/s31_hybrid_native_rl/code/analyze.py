"""S31 analysis: paired per-seed comparison of frozen checkpoints vs planner B and the random strategy on the 12 held-out seeds; PLAN/EDIT/WAIT frequencies; whether deviations help.
   python research/s31_hybrid_native_rl/code/analyze.py"""
import glob
import gzip
import json
import os
import statistics as S
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from hybrid_env import EVAL_SEEDS, ROOT  # noqa: E402

OUT = os.path.join(ROOT, "research", "s31_hybrid_native_rl")
EV = os.path.join(OUT, "eval")


def boot_ci(v, n=10000, seed=0):
    v = np.asarray(v, dtype=float)
    if len(v) == 0:
        return None
    rng = np.random.default_rng(seed)
    return [float(np.percentile([v[rng.integers(len(v), size=len(v))].mean() for _ in range(n)], q)) for q in (2.5, 97.5)]


def main():
    tags = sorted({os.path.basename(p).rsplit("_", 1)[0] for p in glob.glob(os.path.join(EV, "*.json.gz"))})
    D = {t: {} for t in tags}
    for t in tags:
        for s in EVAL_SEEDS:
            p = os.path.join(EV, f"{t}_{s}.json.gz")
            if os.path.exists(p):
                D[t][s] = json.load(gzip.open(p, "rt"))
    summ = {}
    for t in tags:
        eps = list(D[t].values())
        if not eps:
            continue
        n_dec = sum(e["t"] for e in eps)
        summ[t] = {"n": len(eps), "score_total": sum(e["score"] for e in eps), "score_median": S.median(e["score"] for e in eps), "decisions_median": S.median(e["t"] for e in eps), "survived_600": sum(1 for e in eps if e["t"] >= 600), "terminated": sum(e["terminated"] for e in eps),
                   "illegal": sum(e["illegal"] for e in eps), "destroyed": sum(e["destroyed"] for e in eps), "PLAN": sum(e["plan"] for e in eps) / n_dec, "EDIT": sum(e["edit"] for e in eps) / n_dec, "WAIT": sum(e["wait"] for e in eps) / n_dec, "UPG": sum(e["upg"] for e in eps) / n_dec,
                   "return_total": sum(e["reward"] for e in eps), "connected_end_median": S.median(e["connected"] for e in eps)}
    paired = {}
    for t in tags:
        if t in ("plannerB", "random"):
            continue
        for ref in ("plannerB", "random"):
            common = [s for s in EVAL_SEEDS if s in D[t] and s in D.get(ref, {})]
            if not common:
                continue
            ds = [D[t][s]["score"] - D[ref][s]["score"] for s in common]
            dt = [D[t][s]["t"] - D[ref][s]["t"] for s in common]
            paired[f"{t}_minus_{ref}"] = {"n": len(common), "score_diffs": ds, "sum": sum(ds), "median": S.median(ds), "higher": sum(1 for x in ds if x > 0), "equal": sum(1 for x in ds if x == 0), "lower": sum(1 for x in ds if x < 0), "ci95_mean": boot_ci(ds),
                                          "decisions_diffs": dt, "decisions_median_diff": S.median(dt)}
    # do deviations help? per checkpoint: seeds where the policy deviated (EDIT+WAIT > 0) vs where it did not, score diff vs planner B
    dev = {}
    for t in tags:
        if t in ("plannerB", "random") or "plannerB" not in D:
            continue
        rows = [{"seed": s, "edit": D[t][s]["edit"], "wait": D[t][s]["wait"], "plan": D[t][s]["plan"], "d_score": D[t][s]["score"] - D["plannerB"][s]["score"], "d_t": D[t][s]["t"] - D["plannerB"][s]["t"], "destroyed": D[t][s]["destroyed"]} for s in EVAL_SEEDS if s in D[t] and s in D["plannerB"]]
        with_dev = [r for r in rows if r["edit"] + r["wait"] > 0]
        dev[t] = {"seeds_with_deviation": len(with_dev), "score_diff_when_deviating": [r["d_score"] for r in with_dev], "score_diff_when_not": [r["d_score"] for r in rows if r["edit"] + r["wait"] == 0], "rows": rows}
    res = {"summary": summ, "paired": paired, "deviations": dev}
    json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=1, default=str)
    for t, v in summ.items():
        print(t, json.dumps(v))
    for k, v in paired.items():
        print(k, json.dumps({kk: vv for kk, vv in v.items() if kk not in ("score_diffs", "decisions_diffs")}), "diffs", v["score_diffs"])
    for t, v in dev.items():
        print("deviation", t, {k: vv for k, vv in v.items() if k != "rows"})


if __name__ == "__main__":
    main()
