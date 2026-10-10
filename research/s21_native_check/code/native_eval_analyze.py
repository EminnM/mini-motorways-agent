"""Native planner evaluation analysis: per-seed table for S8-C / planner A / planner B, paired differences, comparison with the S13a noise floor (three earlier S8-C control realisations per seed).
   python research/s21_native_check/code/native_eval_analyze.py -> research/s21_native_check/native_eval.json"""
import gzip
import json
import os
import statistics as S
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from native_adapter import ROOT  # noqa: E402

OUT = os.path.join(ROOT, "research", "s21_native_check")
SEEDS = json.load(open(os.path.join(ROOT, "research", "s9_native_transfer", "seeds.json")))["seeds"]
H = 60


def metrics(d):
    st = d["steps"]
    n = len(st)
    conn, conn_at, spawns = set(), [], []
    first_conn = None
    for x in st:
        if x["connected_houses"] is not None:
            conn = {tuple(h) for h in x["connected_houses"]}
        conn_at.append(set(conn))
        if first_conn is None and x["connected"] >= 1:
            first_conn = x["i"]
        for h in x["new_houses"]:
            spawns.append((x["i"], tuple(h)))
    ok = cen = 0
    lat = []
    for i0, h in spawns:
        if i0 + H >= n:
            cen += 1
        else:
            t = next((j for j in range(i0, i0 + H + 1) if h in conn_at[j]), None)
            if t is not None:
                ok += 1
                lat.append(t - i0)
    ex = d["end"].get("exec_stats") or {}
    return {"steps": n, "deliveries": st[-1]["score_cum"], "terminated": d["end"]["terminated"], "cut": d["end"]["cut"], "first_connection": first_conn, "connected_end": st[-1]["connected"], "houses_end": st[-1]["houses"],
            "peak_connected": max(x["connected"] for x in st), "frac_zero_concrete": sum(1 for x in st if x["concrete"] == 0) / n, "concrete_spent": sum(max(0, x["concrete_prev"] - x["concrete"]) for x in st if x["kind"] == "ADD"),
            "illegal": sum(1 for x in st if not x["legal"]), "destroyed": sum(len(x["lost"]) for x in st), "spawned": len(spawns), "spawned_connected_60": ok, "spawned_censored": cen, "latency_median": S.median(lat) if lat else None,
            "plans": ex.get("plans"), "completed": ex.get("completed"), "abandoned": ex.get("abandoned"), "abandon_reasons": ex.get("abandon_reasons"), "waits_for_concrete": ex.get("waits_for_concrete"), "destroy_vetoes": ex.get("destroy_vetoes"),
            "action_share": {k: sum(1 for x in st if x["kind"] == k) / n for k in ("WAIT", "ADD", "REMOVE", "UPGRADE")}}


def main():
    res = {"policies": {}, "per_seed": {}, "paired": {}, "noise_floor": {}}
    M = {}
    for pol in ("s8c", "A", "B"):
        M[pol] = {}
        for s in SEEDS:
            p = os.path.join(OUT, "raw", f"{pol}_{s}.json.gz")
            if os.path.exists(p):
                M[pol][s] = metrics(json.load(gzip.open(p, "rt")))
    # S13a control realisations of S8-C (S11 control, S12 control, S13 replicate) for the noise floor
    nf = json.load(open(os.path.join(ROOT, "research", "s13_multi_house", "noise_floor.json")))
    ctrl = {int(k): v for k, v in nf["control_realisations_per_seed_deliveries"].items()}
    for pol, m in M.items():
        v = list(m.values())
        if not v:
            continue
        res["policies"][pol] = {"episodes": len(v), "deliveries_total": sum(x["deliveries"] for x in v), "deliveries_median": S.median(x["deliveries"] for x in v), "steps_median": S.median(x["steps"] for x in v), "survived_600": sum(1 for x in v if x["steps"] >= 600),
                                "terminated": sum(1 for x in v if x["terminated"]), "cut_by_deadline": sum(1 for x in v if x["cut"]), "illegal_total": sum(x["illegal"] for x in v), "destroyed_total": sum(x["destroyed"] for x in v),
                                "spawned_connected_60": [sum(x["spawned_connected_60"] for x in v), sum(x["spawned"] - x["spawned_censored"] for x in v)], "latency_median_of_medians": S.median([x["latency_median"] for x in v if x["latency_median"] is not None] or [None]) if any(x["latency_median"] is not None for x in v) else None,
                                "frac_zero_concrete_median": S.median(x["frac_zero_concrete"] for x in v), "concrete_spent_median": S.median(x["concrete_spent"] for x in v), "connected_end_median": S.median(x["connected_end"] for x in v), "houses_end_median": S.median(x["houses_end"] for x in v),
                                "abandoned_total": sum((x["abandoned"] or 0) for x in v), "plans_total": sum((x["plans"] or 0) for x in v), "abandon_reasons": {k: sum((x["abandon_reasons"] or {}).get(k, 0) for x in v) for x_ in v for k in (x_["abandon_reasons"] or {})},
                                "action_share_mean": {k: sum(x["action_share"][k] for x in v) / len(v) for k in ("WAIT", "ADD", "REMOVE", "UPGRADE")}}
    for s in SEEDS:
        res["per_seed"][s] = {pol: ({k: M[pol][s][k] for k in ("deliveries", "steps", "connected_end", "houses_end", "spawned_connected_60", "spawned", "destroyed", "illegal", "abandoned", "frac_zero_concrete")} if s in M[pol] else None) for pol in M}
        res["per_seed"][s]["s8c_control_realisations_S11_S12_S13"] = ctrl.get(s)
    for a, b in (("A", "s8c"), ("B", "s8c"), ("A", "B")):
        d = [M[a][s]["deliveries"] - M[b][s]["deliveries"] for s in SEEDS if s in M[a] and s in M[b]]
        if d:
            res["paired"][f"{a}_minus_{b}"] = {"n": len(d), "sum": sum(d), "median": S.median(d), "higher": sum(1 for x in d if x > 0), "equal": sum(1 for x in d if x == 0), "lower": sum(1 for x in d if x < 0), "diffs": d,
                                                "steps_diff_median": S.median(M[a][s]["steps"] - M[b][s]["steps"] for s in SEEDS if s in M[a] and s in M[b])}
    # noise floor: planners vs the RANGE of the three earlier S8-C control realisations per seed
    for pol in ("A", "B", "s8c"):
        outside_above = [s for s in SEEDS if s in M[pol] and ctrl.get(s) and M[pol][s]["deliveries"] > max(ctrl[s])]
        outside_below = [s for s in SEEDS if s in M[pol] and ctrl.get(s) and M[pol][s]["deliveries"] < min(ctrl[s])]
        res["noise_floor"][pol] = {"seeds_above_all_three_S8C_controls": outside_above, "seeds_below_all_three": outside_below, "n": len(outside_above), "expected_above_if_exchangeable": "~3 of 12 (25 %)"}
    res["noise_floor"]["reference"] = {"S8C_control_totals_S11_S12_S13": [552, 503, 577], "replicate_vs_control_delivery_diff_sd_per_seed": 54.9, "sd_of_12_seed_sum": 190}
    json.dump(res, open(os.path.join(OUT, "native_eval.json"), "w"), indent=1, default=str)
    for pol, v in res["policies"].items():
        print(pol, json.dumps(v)[:600])
    print("paired", json.dumps(res["paired"]))
    print("noise", json.dumps(res["noise_floor"]))
    print("per seed deliveries [s8c, A, B | controls]")
    for s in SEEDS:
        r = res["per_seed"][s]
        print(s, [r[p]["deliveries"] if r.get(p) else None for p in ("s8c", "A", "B")], "|", r["s8c_control_realisations_S11_S12_S13"], "steps", [r[p]["steps"] if r.get(p) else None for p in ("s8c", "A", "B")], "conn", [f"{r[p]['connected_end']}/{r[p]['houses_end']}" if r.get(p) else None for p in ("s8c", "A", "B")])


if __name__ == "__main__":
    main()
