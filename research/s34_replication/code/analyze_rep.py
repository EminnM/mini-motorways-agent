"""S34 analysis (preregistered): dev benchmark (12 designated seeds, 600 cap) and fresh benchmark (8 fresh seeds, 1,500 horizon, 3 stochastic rollouts).
Training-run variation (S31 / rep1 / rep2 at the same checkpoint) is reported separately from rollout variation (within-run SD over the 3 rollouts per seed); nothing is pooled.
   python research/s34_replication/code/analyze_rep.py"""
import glob
import gzip
import json
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
S34 = os.path.join(ROOT, "research", "s34_replication")
RNG = np.random.default_rng(34)


def load(d):
    eps = []
    for p in glob.glob(os.path.join(d, "*.json.gz")):
        eps.append(json.load(gzip.open(p, "rt")))
    return eps


def shares(eps):
    tot = sum(e["plan"] + e["edit"] + e["wait"] + e["upg"] for e in eps)
    return {k: round(sum(e[k] for e in eps) / tot, 3) for k in ("plan", "edit", "wait", "upg")} if tot else {}


def summary(eps):
    return {"n": len(eps), "deliveries_total": float(sum(e["score"] for e in eps)), "deliveries_median": float(np.median([e["score"] for e in eps])), "length_median": float(np.median([e["t"] for e in eps])),
            "survived": sum(1 for e in eps if not e["terminated"]), "destroyed": sum(e["destroyed"] for e in eps), "illegal": sum(e["illegal"] for e in eps), "shares": shares(eps), "env": sorted({e.get("env_version", "?") for e in eps})}


def paired(a, b, key="score"):
    """a, b: dict seed -> value. Returns paired stats of a - b over common seeds."""
    seeds = sorted(set(a) & set(b))
    d = np.array([a[s] - b[s] for s in seeds], dtype=float)
    boot = [RNG.choice(d, len(d)).mean() for _ in range(10000)] if len(d) else [0]
    return {"n": len(seeds), "sum": float(d.sum()), "mean": float(d.mean()) if len(d) else None, "median": float(np.median(d)) if len(d) else None, "higher": int((d > 0).sum()), "equal": int((d == 0).sum()), "lower": int((d < 0).sum()), "ci95_mean": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))], "diffs": d.tolist()}


def main():
    out = {}
    # ---------------- dev benchmark
    dev = load(os.path.join(S34, "eval_dev"))
    # planner B post-fix from S33 regression; S31 50k sto post-fix from S33 as well
    for p in glob.glob(os.path.join(ROOT, "research", "s33_playable_area_fix", "eval_postfix", "*.json.gz")):
        e = json.load(gzip.open(p, "rt"))
        e["policy"] = {"postfix_plannerB": "plannerB", "postfix_step_0049152_sto": "hybrid_s31__step_0049152_sto"}.get(e["policy"], e["policy"])
        e["bench"] = "dev"
        dev.append(e)
    by = defaultdict(dict)
    for e in dev:
        by[e["policy"]][e["seed"]] = e
    out["dev"] = {"summary": {k: summary(list(v.values())) for k, v in by.items()}, "paired_vs_plannerB": {}}
    B = {s: e["score"] for s, e in by["plannerB"].items()}
    for k, v in by.items():
        if k != "plannerB":
            out["dev"]["paired_vs_plannerB"][k] = paired({s: e["score"] for s, e in v.items()}, B)
    # ---------------- fresh benchmark
    fresh = load(os.path.join(S34, "eval_fresh"))
    fby = defaultdict(lambda: defaultdict(dict))   # policy -> seed -> rollout -> ep
    for e in fresh:
        fby[e["policy"]][e["seed"]][e["rollout"]] = e
    out["fresh"] = {"summary": {}, "rollout_variation": {}, "paired_vs_plannerB": {}, "training_run_variation": {}}
    FB = {s: list(r.values())[0]["score"] for s, r in fby["plannerB"].items()}
    per_policy_mean = {}
    for pol, seeds in fby.items():
        eps = [e for r in seeds.values() for e in r.values()]
        out["fresh"]["summary"][pol] = summary(eps)
        if pol.endswith("_sto"):
            sds = [float(np.std([e["score"] for e in r.values()], ddof=1)) for r in seeds.values() if len(r) > 1]
            means = {s: float(np.mean([e["score"] for e in r.values()])) for s, r in seeds.items()}
            out["fresh"]["rollout_variation"][pol] = {"rollouts_per_seed": sorted({len(r) for r in seeds.values()}), "within_seed_sd_mean": float(np.mean(sds)) if sds else None, "within_seed_sd_per_seed": [round(x, 1) for x in sds], "per_seed_mean": means, "total_of_means": float(sum(means.values()))}
            per_policy_mean[pol] = means
        else:
            per_policy_mean[pol] = {s: list(r.values())[0]["score"] for s, r in seeds.items()}
        if pol != "plannerB":
            out["fresh"]["paired_vs_plannerB"][pol] = paired(per_policy_mean[pol], FB)
    # training-run variation: same checkpoint/mode across runs
    for ck in ("step_0049152", "step_0100352"):
        for mode in ("sto", "det"):
            runs = {pol.split("__")[0]: out["fresh"]["paired_vs_plannerB"][pol] for pol in out["fresh"]["paired_vs_plannerB"] if pol.endswith(f"{ck}_{mode}")}
            if runs:
                sums = {r: v["sum"] for r, v in runs.items()}
                out["fresh"]["training_run_variation"][f"{ck}_{mode}"] = {"advantage_over_B_per_run": sums, "n_runs": len(sums), "min": min(sums.values()), "max": max(sums.values()), "sd_across_runs": float(np.std(list(sums.values()), ddof=1)) if len(sums) > 1 else None, "higher_seeds_per_run": {r: v["higher"] for r, v in runs.items()}}
    # preregistered decision rule on 100k sto
    tv = out["fresh"]["training_run_variation"].get("step_0100352_sto")
    if tv and tv["n_runs"] >= 3:
        adv = tv["advantage_over_B_per_run"]
        hi = tv["higher_seeds_per_run"]
        robust = all(v > 0 for v in adv.values()) and all(h >= 5 for h in hi.values()) and min(adv.values()) >= 0.5 * max(adv.values())
        out["decision"] = {"rule": "every run's 100k sto beats B in total with >=5/8 seeds higher, and worst run >= half of best run", "advantage": adv, "higher_seeds": hi, "robust": bool(robust)}
    json.dump(out, open(os.path.join(S34, "results.json"), "w"), indent=1)
    for bench in ("dev", "fresh"):
        print(f"=== {bench}")
        for k, v in sorted(out[bench]["summary"].items()):
            print(f"{k:45s} n={v['n']:3d} total={v['deliveries_total']:7.1f} med={v['deliveries_median']:6.1f} len={v['length_median']:6.0f} surv={v['survived']:2d} destroyed={v['destroyed']:4d} illegal={v['illegal']} {v['shares']} env={v['env']}")
        for k, v in sorted(out[bench]["paired_vs_plannerB"].items()):
            print(f"  {k:43s} vs B: sum {v['sum']:+7.1f} H/E/L {v['higher']}/{v['equal']}/{v['lower']} CI {np.round(v['ci95_mean'], 1).tolist()}")
    for k, v in out["fresh"]["rollout_variation"].items():
        print(f"rollout var {k:40s} within-seed SD {v['within_seed_sd_mean']} total_of_means {v['total_of_means']}")
    for k, v in out["fresh"]["training_run_variation"].items():
        print("run var", k, v)
    print("decision", out.get("decision"))


if __name__ == "__main__":
    main()
