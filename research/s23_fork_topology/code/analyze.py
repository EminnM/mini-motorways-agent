"""S23 analysis per PREREGISTRATION.md.   python research/s23_fork_topology/code/analyze.py -> research/s23_fork_topology/results.json"""
import gzip
import json
import os
import statistics as S
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "s21_native_check", "code"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "spawn_harness", "code"))
from native_adapter import ROOT  # noqa: E402
import harness as H  # noqa: E402
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import s11_guard as G  # noqa: E402

OUT = os.path.join(ROOT, "research", "s23_fork_topology")
SEEDS = json.load(open(os.path.join(ROOT, "research", "s9_native_transfer", "seeds.json")))["seeds"]
BR = ("base", "base2", "parallel", "dense")


def cp_stats(cp):
    o = cp.get("obs")
    if not o:
        return {}
    ctx = G.Ctx(o)
    func = G.edges_from_raw(o)
    conn = G.connected_house_set(ctx, func)
    dests = [t.get("destination") for t in o["tiles"] if t.get("destination")]
    return {"vehicles": sum(t.get("visibleVehicleCount", 0) for t in o["tiles"]), "connected": sorted(list(h) for h in conn), "n_connected": len(conn), "houses": len(ctx.houses), "n_func": len(func),
            "overcrowding": any(d.get("isOvercrowding") for d in dests), "max_demand": max([d.get("totalDemand", 0) for d in dests] or [0]), "max_overcrowding_time": max([float(d.get("overcrowdingTime", 0.0)) for d in dests] or [0.0]), "concrete": o["globals"]["concreteAvailable"]}


def boot_ci(d, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    d = np.asarray(d, dtype=float)
    if len(d) == 0:
        return None
    bs = [d[rng.integers(len(d), size=len(d))].mean() for _ in range(n)]
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def main():
    res = {"seeds": {}, "checks": {}, "primary": {}, "pre_divergence": {}, "secondary": {}}
    rows = []
    for s in SEEDS:
        p = os.path.join(OUT, "raw", f"fork_{s}.json.gz")
        if not os.path.exists(p):
            continue
        d = json.load(gzip.open(p, "rt"))
        rec = d["fork"]
        if "error" in rec:
            res["seeds"][s] = {"error": rec["error"][:200]}
            continue
        br = H.parse_fork(rec)
        rb = rec["branches"]
        score0 = d["anchor_state"]["score"]
        per = {}
        for n in BR:
            cps = rb[n]["checkpoints"]
            stats = {cp["t"]: {"score": cp["score"], "deliveries": cp["score"] - score0, **cp_stats(cp)} for cp in cps}
            spec = d["branch_specs"][n]
            per[n] = {"edits_requested": len(spec), "edits_accepted": rb[n]["editsAccepted"], "concrete_delta": rb[n]["concreteAtRestore"] - rb[n]["concreteAfterEdits"], "terminal": rb[n]["terminalStep"], "spawns": br[n]["spawns"],
                      "first_spawn_div_vs_base": H.first_spawn_difference(br["base"], br[n]) if n != "base" else None, "confound": H.rng_confounded(br["base"], br[n]) if n != "base" else None, "cp": stats,
                      "deliveries_60": stats[60]["deliveries"] if 60 in stats else None, "first_overcrowding_cp": next((t for t in sorted(stats) if stats[t].get("overcrowding")), None)}
        # checks
        chk = {n: {"legal_all": per[n]["edits_accepted"] == per[n]["edits_requested"], "concrete_ok": per[n]["concrete_delta"] == per[n]["edits_accepted"]} for n in BR}
        conn1 = {n: per[n]["cp"].get(1, {}).get("connected") for n in BR}
        chk["initial_connectivity_identical"] = len({json.dumps(v) for v in conn1.values()}) == 1
        chk["anchor_connected"] = d["anchor_state"]["connected"]
        chk["base_vs_base2"] = {**H.equivalent_initial_state(br["base"], br["base2"]), "vehicles_equal_all_cp": all(per["base"]["cp"][t].get("vehicles") == per["base2"]["cp"][t].get("vehicles") for t in per["base"]["cp"]), "deliveries_60_diff": per["base2"]["deliveries_60"] - per["base"]["deliveries_60"]}
        # pre-divergence window: last checkpoint <= first spawn divergence (or 60)
        pre = {}
        for n in ("base2", "parallel", "dense"):
            div = per[n]["first_spawn_div_vs_base"]
            tmax = max([t for t in per[n]["cp"] if div is None or t <= div] or [0])
            pre[n] = {"window_end": tmax, "deliveries_diff": (per[n]["cp"][tmax]["deliveries"] - per["base"]["cp"][tmax]["deliveries"]) if tmax else None, "vehicles_diff": (per[n]["cp"][tmax].get("vehicles", 0) - per["base"]["cp"][tmax].get("vehicles", 0)) if tmax else None,
                      "deliveries_base": per["base"]["cp"][tmax]["deliveries"] if tmax else None}
        res["seeds"][s] = {"anchor_state": {k: v for k, v in d["anchor_state"].items() if k != "dests"}, "branches": {n: {k: v for k, v in per[n].items() if k != "cp"} | {"cp_summary": {t: {kk: vv for kk, vv in c.items() if kk != "connected"} for t, c in per[n]["cp"].items()}} for n in BR}, "checks": chk, "pre_divergence": pre}
        rows.append({"seed": s, "d_full_par": per["parallel"]["deliveries_60"] - per["base"]["deliveries_60"], "d_full_den": per["dense"]["deliveries_60"] - per["base"]["deliveries_60"], "d_full_base2": per["base2"]["deliveries_60"] - per["base"]["deliveries_60"],
                     "base_60": per["base"]["deliveries_60"], "div_par": per["parallel"]["first_spawn_div_vs_base"], "div_den": per["dense"]["first_spawn_div_vs_base"], "div_base2": per["base2"]["first_spawn_div_vs_base"],
                     "pre_par": pre["parallel"]["deliveries_diff"], "pre_den": pre["dense"]["deliveries_diff"], "pre_end_par": pre["parallel"]["window_end"], "pre_end_den": pre["dense"]["window_end"],
                     "veh60_base": per["base"]["cp"][60].get("vehicles"), "veh60_par": per["parallel"]["cp"][60].get("vehicles"), "veh60_den": per["dense"]["cp"][60].get("vehicles"), "conf_par": per["parallel"]["confound"]["confounded"], "conf_den": per["dense"]["confound"]["confounded"],
                     "term": {n: per[n]["terminal"] for n in BR}, "over_cp": {n: per[n]["first_overcrowding_cp"] for n in BR}, "edits": {n: per[n]["edits_accepted"] for n in ("parallel", "dense")}})
    def summ(key):
        v = [r[key] for r in rows if r[key] is not None]
        return {"n": len(v), "values": v, "mean": float(np.mean(v)) if v else None, "median": float(np.median(v)) if v else None, "positive": sum(1 for x in v if x > 0), "zero": sum(1 for x in v if x == 0), "negative": sum(1 for x in v if x < 0), "ci95_mean": boot_ci(v)}
    res["primary"] = {"parallel_minus_base_full60": summ("d_full_par"), "dense_minus_base_full60": summ("d_full_den"), "base2_minus_base_full60": summ("d_full_base2"), "base_deliveries_60": summ("base_60")}
    res["pre_divergence"] = {"parallel": summ("pre_par"), "dense": summ("pre_den"), "window_end_parallel": [r["pre_end_par"] for r in rows], "window_end_dense": [r["pre_end_den"] for r in rows]}
    res["secondary"] = {"spawn_divergence_fraction": {n: sum(1 for r in rows if r[k] is not None) / len(rows) for n, k in (("parallel", "div_par"), ("dense", "div_den"), ("base2", "div_base2"))} if rows else None,
                        "rng_confounded_fraction": {"parallel": sum(r["conf_par"] for r in rows) / len(rows), "dense": sum(r["conf_den"] for r in rows) / len(rows)} if rows else None,
                        "vehicles_60": {"base": [r["veh60_base"] for r in rows], "parallel": [r["veh60_par"] for r in rows], "dense": [r["veh60_den"] for r in rows]},
                        "terminal_steps": [r["term"] for r in rows], "first_overcrowding_cp": [r["over_cp"] for r in rows], "edits_accepted": [r["edits"] for r in rows]}
    res["checks"] = {s: res["seeds"][s]["checks"] for s in res["seeds"] if "checks" in res["seeds"][s]}
    res["rows"] = rows
    json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=1, default=str)
    print("PRIMARY", json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "values"} for k, v in res["primary"].items()}))
    print("PRE-DIV", json.dumps({k: ({kk: vv for kk, vv in v.items() if kk != "values"} if isinstance(v, dict) else v) for k, v in res["pre_divergence"].items()}))
    print("SECONDARY", json.dumps(res["secondary"])[:1200])
    for r in rows:
        print(r)
    for s, c in res["checks"].items():
        print(s, "legal/concrete", {n: (c[n]["legal_all"], c[n]["concrete_ok"]) for n in BR}, "init_conn_identical", c["initial_connectivity_identical"], "base2:", {k: v for k, v in c["base_vs_base2"].items() if k in ("equivalent", "vehicles_equal_all_cp", "deliveries_60_diff")})


if __name__ == "__main__":
    main()
