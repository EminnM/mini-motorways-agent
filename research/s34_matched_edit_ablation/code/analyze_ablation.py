"""S34 ablation analysis. Conditions per training run (100k checkpoints; S31 50k secondary):
  A  = replication fresh-benchmark stochastic rollouts (research/s34_replication/eval_fresh, kinds trace only)
  A' = instrumented re-run of A with the same torch seeds (per-decision records)   -> A vs A' = pure rollout-noise null on identical seeds
  B  = randomized EDIT (policy's EDIT decisions kept, chosen edit replaced by a uniform legal EDIT)
  C  = planner B + matched-rate random EDIT/WAIT
Paired per-seed differences use the per-seed mean over the 3 rollouts; bootstrap CI on the mean of per-seed differences (n = 8 seeds). Variability is reported across seeds, rollouts and runs separately.
   python research/s34_matched_edit_ablation/code/analyze_ablation.py"""
import glob
import gzip
import json
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
ABL = os.path.join(ROOT, "research", "s34_matched_edit_ablation")
REP = os.path.join(ROOT, "research", "s34_replication", "eval_fresh")
RUNS = ["hybrid_s31", "rep1_seed131", "rep2_seed231"]
RNG = np.random.default_rng(340)


def load_abl():
    out = defaultdict(lambda: defaultdict(dict))    # key (cond, run, ckpt) -> seed -> rollout -> ep
    for p in glob.glob(os.path.join(ABL, "eval", "*.json.gz")):
        e = json.load(gzip.open(p, "rt"))
        cond = "A'" if e["cond"] == "A" else e["cond"]      # instrumented re-run of A is labelled A'
        out[(cond, e["run"], e["ckpt"])][e["seed"]][e["rollout"]] = e
    return out


def load_A():
    out = defaultdict(lambda: defaultdict(dict))
    for p in glob.glob(os.path.join(REP, "*_sto_r*_*.json.gz")):
        e = json.load(gzip.open(p, "rt"))
        run, rest = e["policy"].split("__")
        ck = rest.replace("_sto", "")
        out[("A", run, ck)][e["seed"]][e["rollout"]] = e
    for p in glob.glob(os.path.join(REP, "plannerB_r0_*.json.gz")):
        e = json.load(gzip.open(p, "rt"))
        out[("plannerB", "plannerB", "-")][e["seed"]][0] = e
    return out


def seed_means(group, key="score"):
    return {s: float(np.mean([e[key] for e in r.values()])) for s, r in group.items()}


def paired(a, b):
    seeds = sorted(set(a) & set(b))
    d = np.array([a[s] - b[s] for s in seeds])
    boot = [RNG.choice(d, len(d)).mean() for _ in range(10000)]
    return {"n_seeds": len(seeds), "sum": float(d.sum()), "mean": float(d.mean()), "higher": int((d > 0).sum()), "equal": int((d == 0).sum()), "lower": int((d < 0).sum()), "ci95_mean": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))], "diffs": [round(float(x), 1) for x in d]}


def summ(group):
    eps = [e for r in group.values() for e in r.values()]
    tot = sum(e["plan"] + e["edit"] + e["wait"] + e["upg"] for e in eps)
    s = {"episodes": len(eps), "deliveries_total_of_seed_means": float(sum(seed_means(group).values())), "deliveries_mean_per_episode": float(np.mean([e["score"] for e in eps])), "length_median": float(np.median([e["t"] for e in eps])), "survived": sum(1 for e in eps if not e["terminated"]), "destroyed_per_episode": float(np.mean([e["destroyed"] for e in eps])), "illegal": sum(e["illegal"] for e in eps),
         "shares": {k: round(sum(e[k] for e in eps) / tot, 3) for k in ("plan", "edit", "wait", "upg")}, "rollout_totals": [float(sum(e["score"] for s_, r in group.items() for ro, e in r.items() if ro == k)) for k in sorted({ro for r in group.values() for ro in r})], "within_seed_sd": float(np.mean([np.std([e["score"] for e in r.values()], ddof=1) for r in group.values() if len(r) > 1])) if any(len(r) > 1 for r in group.values()) else None}
    if any("per_decision" in e for e in eps):
        s["boundary_violations"] = sum(e.get("boundary_violations", 0) for e in eps)
        s["edits_sampled"] = sum(e.get("edits_sampled", 0) for e in eps)
        s["edits_replaced"] = sum(e.get("edits_replaced", 0) for e in eps)
        add, rem = sum(e["add"] for e in eps), sum(e["remove"] for e in eps)
        s["add_share_of_edits"] = round(add / max(1, add + rem), 3)
        s["destroyed_by_edit"] = sum(e["destroyed_by_edit"] for e in eps)
        s["destroyed_by_plan"] = sum(e["destroyed_by_plan"] for e in eps)
        s["destroyed_per_edit"] = round(s["destroyed_by_edit"] / max(1, add + rem), 4)
        s["destroyed_per_remove"] = round(sum(x["d"] for e in eps for x in e["per_decision"] if x.get("ar") == "R") / max(1, rem), 4)
        s["destroyed_per_add"] = round(sum(x["d"] for e in eps for x in e["per_decision"] if x.get("ar") == "A") / max(1, add), 4)
        if any(e.get("cond") == "B" for e in eps):
            oa = sum(1 for e in eps for x in e["per_decision"] if x.get("oar") == "A")
            orr = sum(1 for e in eps for x in e["per_decision"] if x.get("oar") == "R")
            s["policy_chosen_add_share_before_replacement"] = round(oa / max(1, oa + orr), 3)
    return s


def main():
    G = load_abl()
    G.update(load_A())
    out = {"summary": {}, "contrasts": {}, "run_variation": {}}
    for key, grp in sorted(G.items()):
        out["summary"]["|".join(key)] = summ(grp)
    B = seed_means(G[("plannerB", "plannerB", "-")])
    for run in RUNS:
        for ck in ("step_0100352", "step_0049152"):
            A = G.get(("A", run, ck))
            A2 = G.get(("A'", run, ck))
            Bc = G.get(("B", run, ck))
            Cc = G.get(("C", run, "matched")) if ck == "step_0100352" else None
            if not A or not Bc:
                continue
            c = {}
            mA, mB = seed_means(A), seed_means(Bc)
            c["A_minus_plannerB"] = paired(mA, B)
            c["B_minus_plannerB"] = paired(mB, B)
            c["A_minus_B (primary)"] = paired(mA, mB)
            if A2:
                mA2 = seed_means(A2)
                c["A_minus_Aprime (rollout-noise null)"] = paired(mA, mA2)
                c["Aprime_minus_B"] = paired(mA2, mB)
                c["Apooled_minus_B"] = paired({s: (mA[s] + mA2[s]) / 2 for s in mA}, mB)
            if Cc:
                mC = seed_means(Cc)
                c["C_minus_plannerB"] = paired(mC, B)
                c["A_minus_C"] = paired(mA, mC)
                c["B_minus_C"] = paired(mB, mC)
            out["contrasts"][f"{run}|{ck}"] = c
    for name in ("A_minus_B (primary)", "A_minus_C", "B_minus_C", "A_minus_plannerB", "B_minus_plannerB", "C_minus_plannerB"):
        vals = {k.split("|")[0]: v[name]["sum"] for k, v in out["contrasts"].items() if k.endswith("step_0100352") and name in v}
        if vals:
            out["run_variation"][name] = {"per_run_sum": vals, "sd_across_runs": float(np.std(list(vals.values()), ddof=1)) if len(vals) > 1 else None, "all_same_sign": all(v > 0 for v in vals.values()) or all(v < 0 for v in vals.values())}
    json.dump(out, open(os.path.join(ABL, "results.json"), "w"), indent=1)
    for k, v in out["summary"].items():
        print(f"{k:38s} eps={v['episodes']:2d} total(seed means)={v['deliveries_total_of_seed_means']:7.1f} len={v['length_median']:5.0f} surv={v['survived']} destroyed/ep={v['destroyed_per_episode']:5.1f} shares={v['shares']} rollouts={[round(x) for x in v['rollout_totals']]} sd_within={v['within_seed_sd']}" + (f" add={v['add_share_of_edits']} d/edit={v['destroyed_per_edit']} (add {v['destroyed_per_add']}, rem {v['destroyed_per_remove']}) replaced={v.get('edits_replaced')}/{v.get('edits_sampled')} bviol={v['boundary_violations']}" if "destroyed_per_edit" in v else ""))
    for k, c in out["contrasts"].items():
        print("==", k)
        for n, v in c.items():
            print(f"   {n:38s} sum {v['sum']:+7.1f} mean {v['mean']:+6.1f} H/E/L {v['higher']}/{v['equal']}/{v['lower']} CI {np.round(v['ci95_mean'], 1).tolist()} diffs {v['diffs']}")
    for n, v in out["run_variation"].items():
        print("run variation", n, v)


if __name__ == "__main__":
    main()
