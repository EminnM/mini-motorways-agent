"""Phase-1 spawn pilot analysis (predefined handling; seed-level pairing across anchors).   python research/spawn_harness/code/analyze_spawn_pilot.py -> pilot/results.json"""
import gzip
import json
import os
import statistics as S
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "s21_native_check", "code"))
import harness as H  # noqa: E402
import spawn_pilot as P  # noqa: E402
from native_adapter import NativeStateView, ROOT  # noqa: E402
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import s11_guard as G  # noqa: E402
from synthetic_rl import layout as L  # noqa: E402
from world_model import WorldModel  # noqa: E402
from candidates import connect_candidates  # noqa: E402

OUT = os.path.join(ROOT, "research", "spawn_harness", "pilot")
BR = ("base", "base2", "block_far", "block_near")


def connection_cost(raw):
    """planner's cheapest connection cost (concrete) summed over houses not yet connected and connectable; unconnectable houses counted separately."""
    mask = np.zeros(L.N_ACTIONS, dtype=bool)
    mask[:] = True                                            # structural estimate: legality not needed for a cost estimate (window rule applies)
    v = NativeStateView(raw, mask, 0, {})
    v.is_native = False                                       # use the geometric window rule (no live mask available in a checkpoint observation)
    wm = WorldModel(v)
    unconn = [h for h in wm.houses if h not in wm.connected]
    cands = connect_candidates(wm, k_alternatives=0, houses=unconn)
    best = {}
    for cnd in cands:
        h = cnd.targets["house"]
        best[h] = min(best.get(h, 1e9), cnd.concrete_cost)      # structural cost regardless of current concrete (feasible | needs_concrete)
    unreachable = [h for h in unconn if h not in best]
    return {"n_unconnected": len(unconn), "n_costed": len(best), "n_unreachable": len(unreachable), "total_cost": sum(best.values()), "per_house": {str(h): c for h, c in best.items()}}


def boot_ci(d, n=10000, seed=0):
    d = np.asarray([x for x in d if x is not None], dtype=float)
    if len(d) == 0:
        return None
    rng = np.random.default_rng(seed)
    bs = [d[rng.integers(len(d), size=len(d))].mean() for _ in range(n)]
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def main():
    forks = []
    skipped = []
    for seed in P.SEEDS:
        for anchor in P.ANCHORS:
            p = os.path.join(OUT, "raw", f"pilot_{seed}_{anchor}.json.gz")
            d = json.load(gzip.open(p, "rt"))
            if "fork" not in d or "error" in d.get("fork", {}):
                skipped.append({"seed": seed, "anchor": anchor, "reason": d.get("skipped") or d.get("fork", {}).get("error", "")[:120]})
                continue
            rec = d["fork"]
            br = H.parse_fork(rec)
            raw0 = d["anchor_raw"]
            occ = {n: P.occupied_tiles(d["plan"]["branches"][i]["actions"]) for i, n in enumerate(BR)}
            tiles0 = {(t["coord"][0], t["coord"][1]): t for t in raw0["tiles"]}
            cand_like = {n: sum(1 for p_ in occ[n] if p_ in tiles0 and tiles0[p_]["contentType"] == "None" and tiles0[p_]["buildable"] and not tiles0[p_]["overWater"] and all(v == "None" for v in tiles0[p_]["road"].values())) for n in ("block_far", "block_near")}
            f = {"seed": seed, "anchor": anchor, "far_dist": d["plan"]["far"]["dist_to_dest"], "near_dist": d["plan"]["near"]["dist_to_dest"], "n_segments": d["plan"]["n_segments"],
                 "equivalence_base_base2": H.equivalent_initial_state(br["base"], br["base2"]),
                 "legal": {n: rec["branches"][n]["editsAccepted"] == len(d["plan"]["branches"][i]["actions"]) for i, n in enumerate(BR)},
                 "concrete_cp10_delta": {n: rec["branches"][n]["concreteAtRestore"] - rec["branches"][n]["checkpoints"][0]["concrete"] for n in BR},
                 "empty_buildable_tiles_occupied": cand_like,
                 "prng": {n: {"after_edits_equal_base": rec["branches"][n]["prngAfterEdits"] == rec["branches"]["base"]["prngAfterEdits"], **H.rng_confounded(br["base"], br[n])} for n in ("base2", "block_far", "block_near")},
                 "spawns": {n: br[n]["spawns"] for n in BR}, "terminal": {n: br[n]["terminal_step"] for n in BR}}
            f["classification"] = {n: P.classify_spawns(br["base"]["spawns"], br[n]["spawns"], d["plan"]["branches"][i]["actions"]) for i, n in enumerate(BR) if n != "base"}
            f["n_spawn_diff"] = {n: len(br[n]["spawns"]) - len(br["base"]["spawns"]) for n in BR}
            f["type_timing_changes"] = {n: [(b[1:3], o[1:3], round(b[0]), round(o[0])) for b, o in zip(br["base"]["spawns"], br[n]["spawns"]) if b[1:3] != o[1:3] or abs(b[0] - o[0]) > 1] for n in ("block_far", "block_near")}
            dests0 = P.destination_tiles(raw0)
            def metric(spawns):
                hs = [s for s in spawns if s[1] == "H"]
                if not hs:
                    return None, "no_house_spawn"
                same_missing = sum(1 for s in hs if not any(dd[2] == s[2] for dd in dests0))
                return P.strategic_metric(hs, raw0), ("same_colour_destination_missing_for_%d" % same_missing if same_missing else "ok")
            f["metric"] = {n: metric(br[n]["spawns"]) for n in BR}
            # planner connection cost at checkpoint 90 (or last checkpoint)
            cc = {}
            for n in BR:
                cps = rec["branches"][n]["checkpoints"]
                obs = cps[-1].get("obs")
                cc[n] = connection_cost(obs) if obs else None
            f["connection_cost_cp_last"] = cc
            if all(cc[n] for n in BR):
                f["net_saving_after_3"] = {n: (cc["base"]["total_cost"] - cc[n]["total_cost"]) - 3 for n in ("block_far", "block_near")}
            forks.append(f)
    # effects per fork and seed-level aggregation
    def eff(f, n):
        m0, mn = f["metric"]["base"][0], f["metric"][n][0]
        return (mn - m0) if (m0 is not None and mn is not None) else None
    fork_rows = [{"seed": f["seed"], "anchor": f["anchor"], "d_far": eff(f, "block_far"), "d_near": eff(f, "block_near"), "d_base2": eff(f, "base2"), "far_minus_near": (eff(f, "block_far") - eff(f, "block_near")) if (eff(f, "block_far") is not None and eff(f, "block_near") is not None) else None,
                  "metric_base": f["metric"]["base"], "cls_far": f["classification"]["block_far"], "cls_near": f["classification"]["block_near"], "prng_far": f["prng"]["block_far"]["confounded"], "prng_near": f["prng"]["block_near"]["confounded"],
                  "net_far": f.get("net_saving_after_3", {}).get("block_far"), "net_near": f.get("net_saving_after_3", {}).get("block_near")} for f in forks]
    seed_rows = []
    for seed in P.SEEDS:
        rs = [r for r in fork_rows if r["seed"] == seed]
        def avg(k):
            v = [r[k] for r in rs if r[k] is not None]
            return (sum(v) / len(v)) if v else None
        seed_rows.append({"seed": seed, "forks": len(rs), "d_far": avg("d_far"), "d_near": avg("d_near"), "far_minus_near": avg("far_minus_near"), "net_far": avg("net_far"), "net_near": avg("net_near")})
    def summ(key):
        v = [r[key] for r in seed_rows if r[key] is not None]
        return {"n_seeds": len(v), "values": [round(x, 3) for x in v], "mean": float(np.mean(v)) if v else None, "median": float(np.median(v)) if v else None, "negative": sum(1 for x in v if x < 0), "zero": sum(1 for x in v if x == 0), "positive": sum(1 for x in v if x > 0), "ci95_mean": boot_ci(v)}
    res = {"forks_run": len(forks), "skipped": skipped, "checks": {"equivalent_base_base2": sum(f["equivalence_base_base2"]["equivalent"] for f in forks), "legal_all": sum(all(f["legal"].values()) for f in forks), "concrete_ok": sum(f["concrete_cp10_delta"]["block_far"] == 3 and f["concrete_cp10_delta"]["block_near"] == 3 and f["concrete_cp10_delta"]["base"] == 0 for f in forks)},
           "prng": {"far_after_edits_equal_base": sum(f["prng"]["block_far"]["after_edits_equal_base"] for f in forks), "near_after_edits_equal_base": sum(f["prng"]["block_near"]["after_edits_equal_base"] for f in forks), "far_confounded_before_first_spawn_diff": sum(f["prng"]["block_far"]["confounded"] for f in forks), "near_confounded": sum(f["prng"]["block_near"]["confounded"] for f in forks)},
           "classification_counts": {n: {c: sum(f["classification"][n].count(c) for f in forks) for c in ("UNCHANGED", "BLOCKED", "SHIFTED")} for n in ("block_far", "block_near", "base2")},
           "spawn_count_changes": {n: [f["n_spawn_diff"][n] for f in forks] for n in ("block_far", "block_near")}, "type_timing_changes": sum(len(f["type_timing_changes"]["block_far"]) + len(f["type_timing_changes"]["block_near"]) for f in forks),
           "metric_handling": {n: [f["metric"][n][1] for f in forks] for n in BR},
           "primary_seed_level": {"far_minus_near": summ("far_minus_near"), "far_minus_base": summ("d_far"), "near_minus_base": summ("d_near")},
           "net_saving_seed_level": {"block_far": summ("net_far"), "block_near": summ("net_near")},
           "fork_rows": fork_rows, "seed_rows": seed_rows, "forks": forks}
    json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=1, default=str)
    print("checks", res["checks"], "skipped", res["skipped"])
    print("prng", res["prng"])
    print("classification", res["classification_counts"], "spawn count changes", res["spawn_count_changes"], "type/timing changes", res["type_timing_changes"])
    print("metric handling", {n: {k: res["metric_handling"][n].count(k) for k in set(res["metric_handling"][n])} for n in BR})
    print("PRIMARY seed-level", json.dumps({k: {kk: vv for kk, vv in v.items()} for k, v in res["primary_seed_level"].items()}))
    print("NET saving seed-level", json.dumps(res["net_saving_seed_level"]))
    for r in fork_rows:
        print(r)


if __name__ == "__main__":
    main()
