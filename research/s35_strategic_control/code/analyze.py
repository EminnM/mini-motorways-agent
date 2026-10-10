"""S35 analysis: per condition deliveries / length / resources / connectivity / game-over proxy; decision-point statistics (how often a strategic choice was live and taken);
network divergence from planner B at snapshot times (Jaccard of active-edge sets, paired by seed); paired per-seed differences vs planner B with bootstrap CIs.
   python research/s35_strategic_control/code/analyze.py"""
import glob
import gzip
import json
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
S35 = os.path.dirname(HERE)
RNG = np.random.default_rng(35)


def jacc(a, b):
    A, B = {tuple(x) for x in a}, {tuple(x) for x in b}
    return len(A & B) / max(1, len(A | B))


def paired(a, b):
    seeds = sorted(set(a) & set(b))
    d = np.array([a[s] - b[s] for s in seeds], dtype=float)
    boot = [RNG.choice(d, len(d)).mean() for _ in range(10000)] if len(d) else [0.0]
    return {"n": len(seeds), "sum": float(d.sum()), "higher": int((d > 0).sum()), "equal": int((d == 0).sum()), "lower": int((d < 0).sum()), "ci95_mean": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))], "diffs": d.tolist()}


def main():
    by = defaultdict(dict)
    for p in glob.glob(os.path.join(S35, "eval", "*.json.gz")):
        e = json.load(gzip.open(p, "rt"))
        by[e["tag"]][e["seed"]] = e
    out = {"summary": {}, "paired_vs_plannerB": {}, "network_divergence": {}, "decision_points": {}}
    B = by["plannerB"]
    for tag, eps in sorted(by.items()):
        E = list(eps.values())
        over_groups = [sorted({d["group"] for d in e["dest_state_end"] if d["over"]}) for e in E]
        out["summary"][tag] = {"n": len(E), "deliveries_total": sum(e["score"] for e in E), "deliveries_median": float(np.median([e["score"] for e in E])), "length_median": float(np.median([e["t"] for e in E])), "survived": sum(1 for e in E if not e["terminated"]),
                               "illegal": sum(e["illegal"] for e in E), "concrete_unspent_mean": float(np.mean([e["concrete_end"] for e in E])), "connected_end_mean": float(np.mean([e["connected_end"] for e in E])), "houses_end_mean": float(np.mean([e["houses_end"] for e in E])),
                               "all_houses_connected_at_end": sum(1 for e in E if e["connected_end"] == e["houses_end"]), "gameover_with_overcrowded_dest": sum(1 for g in over_groups if g), "abandoned": sum(e["exec_stats"]["abandoned"] for e in E), "destroy_vetoes": sum(e["exec_stats"]["destroy_vetoes"] for e in E),
                               "kinds": {k: sum(e["kinds"][k] for e in E) for k in ("WAIT", "ADD", "REMOVE", "UPG")}}
        pts = [p for e in E for p in e["points"]]
        if pts:
            out["decision_points"][tag] = {"points": len(pts), "d1_live": sum(p["d1_live"] for p in pts), "d2_live": sum(p["d2_live"] for p in pts), "chose_dedicated": sum(1 for p in pts if p["route"] == "dedicated"), "differs_from_planner": sum(p["differs_from_planner"] for p in pts),
                                           "episodes_with_any_difference": sum(1 for e in E if any(p["differs_from_planner"] for p in e["points"])), "dedicated_extra_cost_mean": float(np.mean([p["dedicated_cost"] - p["reuse_cost"] for p in pts if p["d2_live"]])) if any(p["d2_live"] for p in pts) else None}
        if tag != "plannerB":
            out["paired_vs_plannerB"][tag] = {"deliveries": paired({s: e["score"] for s, e in eps.items()}, {s: e["score"] for s, e in B.items()}), "length": paired({s: e["t"] for s, e in eps.items()}, {s: e["t"] for s, e in B.items()}),
                                              "identical_outcome_seeds": sum(1 for s in eps if s in B and (eps[s]["score"], eps[s]["t"]) == (B[s]["score"], B[s]["t"]))}
            div = {}
            for t in ("100", "200", "300", "400", "600"):
                js = [jacc(eps[s]["snapshots"][t], B[s]["snapshots"][t]) for s in eps if s in B and t in eps[s]["snapshots"] and t in B[s]["snapshots"]]
                if js:
                    div[t] = {"n": len(js), "jaccard_mean": float(np.mean(js)), "jaccard_min": float(min(js)), "seeds_identical": sum(1 for j in js if j == 1.0)}
            out["network_divergence"][tag] = div
    json.dump(out, open(os.path.join(S35, "results.json"), "w"), indent=1)
    for tag, s in out["summary"].items():
        print(f"{tag:12s} n={s['n']:2d} deliveries {s['deliveries_total']:6.0f} (med {s['deliveries_median']:.0f}) len med {s['length_median']:.0f} surv {s['survived']} illegal {s['illegal']} concrete left {s['concrete_unspent_mean']:.1f} connected {s['connected_end_mean']:.1f}/{s['houses_end_mean']:.1f} allconn {s['all_houses_connected_at_end']} overcrowded-death {s['gameover_with_overcrowded_dest']} abandoned {s['abandoned']} kinds {s['kinds']}")
    for tag, d in out["decision_points"].items():
        print("points", tag, d)
    for tag, v in out["paired_vs_plannerB"].items():
        print(f"paired {tag:12s} deliveries sum {v['deliveries']['sum']:+5.0f} H/E/L {v['deliveries']['higher']}/{v['deliveries']['equal']}/{v['deliveries']['lower']} CI {np.round(v['deliveries']['ci95_mean'],1).tolist()} | length sum {v['length']['sum']:+5.0f} H/E/L {v['length']['higher']}/{v['length']['equal']}/{v['length']['lower']} | identical seeds {v['identical_outcome_seeds']}; network jaccard {{ {', '.join(f'{t}: {x['jaccard_mean']:.2f} (ident {x['seeds_identical']}/{x['n']})' for t, x in out['network_divergence'][tag].items())} }}")


if __name__ == "__main__":
    main()
