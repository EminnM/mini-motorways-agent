"""S13b fidelity probes against native logs (replicate episodes): (1) build-window / legal-ADD extent, (2) per-destination demand transition table check, (3) deliveries per connected house.
   python python/synthetic_rl/s13_fidelity.py -> research/s13_multi_house/fidelity_probe.json"""
import gzip
import json
import os
import statistics as S
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from synthetic_rl import layout as L  # noqa: E402
from synthetic_rl import s13_env as E  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "research", "s13_multi_house")


def my_window_and_count(env):
    m = env._mask
    xs, ys, n = [], [], 0
    for a in np.flatnonzero(m):
        if 0 < a < L.UPGRADE_BASE:
            k = L.decode_action(int(a))
            if k[0] == "ADD":
                n += 1
                for p in E.ends((k[1], k[2], k[3])):
                    xs.append(p[0])
                    ys.append(p[1])
    return ([min(xs), max(xs), min(ys), max(ys)] if xs else None), n


def main():
    sc = {s["id"]: i for i, s in enumerate(E.load_scenarios())}
    seeds = json.load(open(os.path.join(ROOT, "research", "s9_native_transfer", "seeds.json")))["seeds"]
    exact, tot, err, cnt_ratio = 0, 0, [], []
    per_edge = {"x0": [], "x1": [], "y0": [], "y1": []}
    for s in seeds:
        d = json.load(gzip.open(os.path.join(OUT, "raw", f"replicate_{s}.json.gz"), "rt"))
        env = E.MultiHouseEnv(E.load_scenarios(), seed=0, lam=0.0, mu=0.0)
        env.reset(options={"scenario_index": sc[f"{s}_replicate"]})
        for x in d["steps"]:
            i = x["i"]
            while env.t < i:
                env.step(0)
            if i % 5 == 0 and x["window_before"] and x["concrete_prev"] > 0 and not env.game_over:
                env.concrete = max(env.concrete, 1)
                env._compute_mask()
                w, n = my_window_and_count(env)
                if w:
                    tot += 1
                    exact += int(w == x["window_before"])
                    diffs = [a - b for a, b in zip(w, x["window_before"])]
                    for k, dv in zip(per_edge, diffs):
                        per_edge[k].append(dv)
                    err.append(max(abs(v) for v in diffs))
                    if x["n_legal_add_before"]:
                        cnt_ratio.append(n / x["n_legal_add_before"])
    pe = {k: {"mean_signed_error": round(float(np.mean(v)), 2), "exact_fraction": round(float(np.mean([a == 0 for a in v])), 3)} for k, v in per_edge.items()}
    # delivery rate per connected house per decision (native, replicate logs)
    num = den = 0
    for s in seeds:
        d = json.load(gzip.open(os.path.join(OUT, "raw", f"replicate_{s}.json.gz"), "rt"))
        for x in d["steps"]:
            num += x["scoreDelta"]
            den += x["connected"]
    res = {"window_compare": {"n_points": tot, "exact_window_fraction": round(exact / tot, 3), "median_max_edge_error_tiles": S.median(err), "p90_max_edge_error": float(np.percentile(err, 90)), "per_edge": pe,
                              "legal_add_count_ratio_mine_over_native_median": round(float(np.median(cnt_ratio)), 3), "ratio_p10_p90": [round(float(np.percentile(cnt_ratio, 10)), 3), round(float(np.percentile(cnt_ratio, 90)), 3)]},
           "native_deliveries_per_connected_house_decision": round(num / den, 4), "synthetic_MU": E.MU}
    json.dump(res, open(os.path.join(OUT, "fidelity_probe.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
