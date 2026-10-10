"""S36 analysis. Benchmarks are never pooled. Historical planner B and historical hybrid rollouts are imported from S33/S34/replication folders where they exist on the same env/horizon
(dev: S33 postfix plannerB + S34-replication eval_dev; fresh: replication eval_fresh incl. 3 stochastic rollouts), new seeds come only from S36 runs.
Reports per benchmark: summary per policy, paired per-seed deliveries vs B+ (stochastic policies on the per-seed mean over rollouts), training-run vs rollout variation,
failure-mode breakdown (unreached destinations at end, overcrowding, connectivity, resources).   python research/s36_rebaseline/code/analyze36.py"""
import glob
import gzip
import json
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
S36 = os.path.join(ROOT, "research", "s36_rebaseline")
RNG = np.random.default_rng(36)


def load36(bench):
    out = defaultdict(lambda: defaultdict(dict))
    for p in glob.glob(os.path.join(S36, "eval", bench, "*.json.gz")):
        e = json.load(gzip.open(p, "rt"))
        out[e["policy"]][e["seed"]][e["rollout"]] = e
    return out


def load_hist(bench):
    """historical episodes on the same env version (postfix_v8) and horizon; normalised to the S36 record shape where fields exist."""
    out = defaultdict(lambda: defaultdict(dict))
    if bench == "dev":
        for p in glob.glob(os.path.join(ROOT, "research", "s33_playable_area_fix", "eval_postfix", "postfix_plannerB_*.json.gz")):
            e = json.load(gzip.open(p, "rt"))
            out["plannerB_hist"][e["seed"]][0] = e
        for p in glob.glob(os.path.join(ROOT, "research", "s34_replication", "eval_dev", "*_sto_*.json.gz")):
            e = json.load(gzip.open(p, "rt"))
            out["hyb_" + e["policy"]][e["seed"]][0] = e
        for p in glob.glob(os.path.join(ROOT, "research", "s33_playable_area_fix", "eval_postfix", "postfix_step_0049152_sto_*.json.gz")):
            e = json.load(gzip.open(p, "rt"))
            out["hyb_hybrid_s31__step_0049152_sto"][e["seed"]][0] = e
    elif bench == "fresh":
        for p in glob.glob(os.path.join(ROOT, "research", "s34_replication", "eval_fresh", "*.json.gz")):
            e = json.load(gzip.open(p, "rt"))
            if e["policy"] == "plannerB":
                out["plannerB_hist"][e["seed"]][0] = e
            elif e["policy"].endswith("_sto"):
                out["hyb_" + e["policy"]][e["seed"]][e["rollout"]] = e
    return out


def seed_means(grp):
    return {s: float(np.mean([e["score"] for e in r.values()])) for s, r in grp.items()}


def paired(a, b):
    seeds = sorted(set(a) & set(b))
    d = np.array([a[s] - b[s] for s in seeds], dtype=float)
    boot = [RNG.choice(d, len(d)).mean() for _ in range(10000)] if len(d) else [0.0]
    return {"n": len(seeds), "sum": float(d.sum()), "mean": float(d.mean()) if len(d) else None, "higher": int((d > 0).sum()), "equal": int((d == 0).sum()), "lower": int((d < 0).sum()), "ci95_mean": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))], "diffs": [round(float(x), 1) for x in d]}


def summ(grp):
    eps = [e for r in grp.values() for e in r.values()]
    s = {"episodes": len(eps), "seeds": len(grp), "rollouts_per_seed": sorted({len(r) for r in grp.values()}), "deliveries_total_of_seed_means": float(sum(seed_means(grp).values())), "length_median": float(np.median([e["t"] for e in eps])), "survived": sum(1 for e in eps if not e["terminated"]),
         "illegal": sum(e.get("illegal", 0) for e in eps), "boundary_violations": sum(e.get("boundary_violations", 0) for e in eps if "boundary_violations" in e), "destroyed_per_episode": float(np.mean([e["destroyed"] for e in eps if "destroyed" in e])) if any("destroyed" in e for e in eps) else None,
         "rollout_totals": [float(sum(e["score"] for s_, r in grp.items() for ro, e in r.items() if ro == k)) for k in sorted({ro for r in grp.values() for ro in r})]}
    if any("unreached_end" in e for e in eps):
        s["unreached_dests_at_end_mean"] = float(np.mean([e["unreached_end"]["n"] for e in eps]))
        s["unreached_demand_at_end_mean"] = float(np.mean([e["unreached_end"]["demand"] for e in eps]))
        s["episodes_ending_with_unreached_dest"] = sum(1 for e in eps if e["unreached_end"]["n"] > 0)
        s["concrete_end_mean"] = float(np.mean([e["concrete_end"] for e in eps]))
        s["bridges_end_mean"] = float(np.mean([e["bridges_end"] or 0 for e in eps]))
        s["connected_frac_end"] = float(np.mean([e["connected_end"] / max(1, e["houses_end"]) for e in eps]))
        s["episodes_ending_overcrowded"] = sum(1 for e in eps if any(d["over"] for d in e["dest_state_end"]))
        s["max_demand_colour_at_end_mean"] = float(np.mean([max([d["demand"] for d in e["dest_state_end"]] or [0]) for e in eps]))
    return s


def main():
    out = {}
    for bench in ("dev", "fresh", "new"):
        G = load36(bench)
        H = load_hist(bench)
        for k, v in H.items():
            if k not in G:
                G[k] = v
        if "plannerBplus" not in G:
            continue
        res = {"summary": {k: summ(v) for k, v in sorted(G.items())}, "paired_vs_Bplus": {}, "paired_vs_plannerB": {}}
        Bp = seed_means(G["plannerBplus"])
        Bh = seed_means(G["plannerB_hist"]) if "plannerB_hist" in G else (seed_means(G["plannerB"]) if "plannerB" in G else None)
        for k, v in G.items():
            if k != "plannerBplus":
                res["paired_vs_Bplus"][k] = paired(seed_means(v), Bp)
            if Bh is not None and k not in ("plannerB_hist", "plannerB"):
                res["paired_vs_plannerB"][k] = paired(seed_means(v), Bh)
        # training-run vs rollout variation for the hybrids at 100k
        hy = {k: v for k, v in G.items() if k.startswith("hyb_") and "0100352" in k}
        if hy:
            res["hybrid_100k_variation"] = {"advantage_over_Bplus_per_run": {k: res["paired_vs_Bplus"][k]["sum"] for k in hy}, "sd_across_runs": float(np.std([res["paired_vs_Bplus"][k]["sum"] for k in hy], ddof=1)) if len(hy) > 1 else None,
                                            "within_seed_rollout_sd_mean": {k: (float(np.mean([np.std([e["score"] for e in r.values()], ddof=1) for r in v.values() if len(r) > 1])) if any(len(r) > 1 for r in v.values()) else None) for k, v in hy.items()}}
        out[bench] = res
        print(f"=== {bench} (horizon {600 if bench == 'dev' else 1500})")
        for k, s in res["summary"].items():
            extra = f" unreached@end {s['episodes_ending_with_unreached_dest']}/{s['episodes']} (demand {s['unreached_demand_at_end_mean']:.1f}) concrete {s['concrete_end_mean']:.1f} conn {s['connected_frac_end']:.2f} overcrowded {s['episodes_ending_overcrowded']}" if "unreached_dests_at_end_mean" in s else ""
            print(f"  {k:42s} eps {s['episodes']:3d} total {s['deliveries_total_of_seed_means']:7.1f} len {s['length_median']:5.0f} surv {s['survived']:2d} illegal {s['illegal']} bviol {s['boundary_violations']} destroyed/ep {s['destroyed_per_episode']}{extra}")
        for k, v in res["paired_vs_Bplus"].items():
            print(f"    {k:40s} - B+ : {v['sum']:+8.1f} H/E/L {v['higher']}/{v['equal']}/{v['lower']} CI {np.round(v['ci95_mean'], 1).tolist()}")
        for k, v in res["paired_vs_plannerB"].items():
            print(f"    {k:40s} - B  : {v['sum']:+8.1f} H/E/L {v['higher']}/{v['equal']}/{v['lower']} CI {np.round(v['ci95_mean'], 1).tolist()}")
        if "hybrid_100k_variation" in res:
            print("    hybrid 100k variation:", res["hybrid_100k_variation"])
    json.dump(out, open(os.path.join(S36, "results.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
