"""S22 analysis per PREREGISTRATION.md: equivalence checks (hash equality through T0, duplicate control), spawn divergence, matched-window primary outcome R = deliveries per connected-house-decision,
preregistered decision rule, secondary outcomes.   python research/s22_throughput/code/analyze.py -> research/s22_throughput/results.json"""
import gzip
import json
import os
import statistics as S
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from throughput_probe import OUT, SEEDS, T0  # noqa: E402

VARS = ("V0", "V0dup", "V1", "V2")


def load(v, s):
    p = os.path.join(OUT, "raw", f"{v}_{s}.json.gz")
    return json.load(gzip.open(p, "rt")) if os.path.exists(p) else None


def spawn_seq(d):
    return [(x["i"], tuple(n)) for x in d["steps"] for n in x["new_struct"] if x["i"] > 0]


def first_spawn_div(a, b):
    sa, sb = spawn_seq(a), spawn_seq(b)
    for x, y in zip(sa, sb):
        if x != y:
            return min(x[0], y[0])
    if len(sa) != len(sb):
        return (sa + sb)[min(len(sa), len(sb))][0]
    return None


def first_hash_div(a, b):
    for x, y in zip(a["steps"], b["steps"]):
        if x["hash"] != y["hash"]:
            return x["i"]
    return None


def window_stats(d, lo, hi):
    st = [x for x in d["steps"] if lo <= x["i"] <= hi]
    if not st:
        return None
    deliv = sum(x["scoreDelta"] for x in st)
    expo = sum(x["connected"] for x in st)
    veh = sum(x["vehicles"] for x in st)
    over = [x for x in st if any(dd[4] for dd in x["dests"])]
    return {"n": len(st), "deliveries": deliv, "exposure": expo, "R": (deliv / expo) if expo else None, "vehicles_per_decision": veh / len(st), "vehicles_per_connected": (veh / expo) if expo else None,
            "overcrowded_fraction": len(over) / len(st), "first_overcrowding": over[0]["i"] if over else None, "max_demand": max(max([dd[3] for dd in x["dests"]] or [0]) for x in st), "lost": sum(len(x["lost"]) for x in st),
            "concrete_end": st[-1]["concrete"], "n_func_mean": sum(x["n_func"] for x in st) / len(st), "connected_mean": expo / len(st)}


def episode_stats(d):
    st = d["steps"]
    return {"decisions": len(st), "deliveries": st[-1]["score_cum"], "terminated": d["end"]["terminated"], "survived_600": len(st) >= 600, "edits_legal": sum(e["legal"] for e in d["edits"]), "edits_total": len(d["edits"]),
            "lost_total": sum(len(x["lost"]) for x in st), "connected_end": st[-1]["connected"], "houses_end": st[-1]["houses"], "concrete_spent_after_T0": max(0, st[T0]["concrete"] - min(x["concrete"] for x in st[T0:])) if len(st) > T0 else None,
            "illegal": sum(not x["legal"] for x in st), "abandoned": d["end"]["exec_stats"].get("abandoned")}


def main():
    res = {"T0": T0, "seeds": {}, "excluded": {}, "primary": {}, "secondary": {}}
    rows = []
    for s in SEEDS:
        D = {v: load(v, s) for v in VARS}
        if any(D[v] is None for v in VARS):
            res["excluded"][s] = "missing episode(s): " + ",".join(v for v in VARS if D[v] is None)
            continue
        eq = {"V0dup_vs_V0_first_hash_divergence": first_hash_div(D["V0"], D["V0dup"]), **{f"{v}_vs_V0_first_hash_divergence": first_hash_div(D["V0"], D[v]) for v in ("V1", "V2")}}
        ok = all((x is None or x > T0) for x in eq.values())
        div = {v: first_spawn_div(D["V0"], D[v]) for v in ("V0dup", "V1", "V2")}
        ends = {v: len(D[v]["steps"]) - 1 for v in VARS}
        rec = {"equivalence_through_T0": ok, "hash_divergence": eq, "spawn_divergence_vs_V0": div, "episode": {v: episode_stats(D[v]) for v in VARS}}
        if not ok:
            res["excluded"][s] = "not identical through T0: " + json.dumps(eq)
            res["seeds"][s] = rec
            continue
        hi = {v: min(ends["V0"], ends[v], (div[v] if div[v] is not None else 10**9)) for v in ("V0dup", "V1", "V2")}
        rec["matched_window"] = {v: [T0 + 1, hi[v]] for v in hi}
        rec["window"] = {v: window_stats(D[v], T0 + 1, hi[v]) for v in ("V0dup", "V1", "V2")}
        rec["window"]["V0_vs_" + "V0dup"] = window_stats(D["V0"], T0 + 1, hi["V0dup"])
        rec["window"]["V0_vs_V1"] = window_stats(D["V0"], T0 + 1, hi["V1"])
        rec["window"]["V0_vs_V2"] = window_stats(D["V0"], T0 + 1, hi["V2"])
        res["seeds"][s] = rec
        R = lambda v, ref: (rec["window"][v]["R"], rec["window"][ref]["R"])  # noqa: E731
        r2, r0 = R("V2", "V0_vs_V2")
        r1, r0b = R("V1", "V0_vs_V1")
        rd, r0c = R("V0dup", "V0_vs_V0dup")
        rows.append({"seed": s, "dR_V2": (r2 - r0) if (r2 is not None and r0 is not None) else None, "dR_V1": (r1 - r0b) if (r1 is not None and r0b is not None) else None, "dR_dup": (rd - r0c) if (rd is not None and r0c is not None) else None,
                     "window_len_V2": hi["V2"] - T0, "window_len_V1": hi["V1"] - T0, "window_len_dup": hi["V0dup"] - T0, "R_V0": r0, "R_V2": r2, "R_V1": r1, "R_dup": rd})
    v2 = [r["dR_V2"] for r in rows if r["dR_V2"] is not None]
    v1 = [r["dR_V1"] for r in rows if r["dR_V1"] is not None]
    du = [abs(r["dR_dup"]) for r in rows if r["dR_dup"] is not None]
    pos2 = sum(1 for x in v2 if x > 0)
    neg2 = sum(1 for x in v2 if x < 0)
    med2 = S.median([abs(x) for x in v2]) if v2 else None
    meddu = S.median(du) if du else None
    rule_a = max(pos2, neg2) >= 9
    rule_b = (med2 is not None and meddu is not None and med2 > 2 * meddu)
    res["primary"] = {"matched_seeds": len(rows), "rows": rows, "V2_minus_V0_dR": {"values": v2, "median": S.median(v2) if v2 else None, "positive": pos2, "negative": neg2, "median_abs": med2},
                      "V1_minus_V0_dR": {"values": v1, "median": S.median(v1) if v1 else None, "positive": sum(1 for x in v1 if x > 0), "negative": sum(1 for x in v1 if x < 0), "median_abs": S.median([abs(x) for x in v1]) if v1 else None},
                      "duplicate_control_abs_dR": {"values": du, "median": meddu, "max": max(du) if du else None},
                      "decision_rule": {"(a)_sign_agreement_>=9": rule_a, "(b)_median_abs_>_2x_duplicate": rule_b, "topology_effect_claimed": rule_a and rule_b}}
    # secondary: whole episode
    sec = {}
    for v in VARS:
        eps = [res["seeds"][s]["episode"][v] for s in res["seeds"] if "episode" in res["seeds"][s]]
        if eps:
            sec[v] = {"n": len(eps), "deliveries_total": sum(e["deliveries"] for e in eps), "deliveries_median": S.median(e["deliveries"] for e in eps), "decisions_median": S.median(e["decisions"] for e in eps), "survived_600": sum(e["survived_600"] for e in eps),
                      "edits_legal_total": sum(e["edits_legal"] for e in eps), "lost_total": sum(e["lost_total"] for e in eps), "illegal_total": sum(e["illegal"] for e in eps), "concrete_spent_after_T0_median": S.median([e["concrete_spent_after_T0"] for e in eps if e["concrete_spent_after_T0"] is not None] or [0])}
    res["secondary"] = sec
    # paired whole-episode deliveries
    for v in ("V0dup", "V1", "V2"):
        d = [res["seeds"][s]["episode"][v]["deliveries"] - res["seeds"][s]["episode"]["V0"]["deliveries"] for s in res["seeds"] if "episode" in res["seeds"][s]]
        if d:
            res["secondary"][f"{v}_minus_V0_episode_deliveries"] = {"sum": sum(d), "median": S.median(d), "higher": sum(1 for x in d if x > 0), "lower": sum(1 for x in d if x < 0), "values": d}
    json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=1, default=str)
    print("excluded", res["excluded"])
    print("primary", json.dumps({k: v for k, v in res["primary"].items() if k != "rows"}))
    for r in rows:
        print(r)
    print("secondary", json.dumps(res["secondary"]))
    for s in res["seeds"]:
        rec = res["seeds"][s]
        print(s, "eq", rec["equivalence_through_T0"], "hashdiv", rec["hash_divergence"], "spawndiv", rec["spawn_divergence_vs_V0"], "ep", {v: (e["decisions"], e["deliveries"], e["lost_total"]) for v, e in rec["episode"].items()})


if __name__ == "__main__":
    main()
