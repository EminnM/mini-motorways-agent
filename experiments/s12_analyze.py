"""S12 analysis: paired control vs concrete-conservation guard on the 12 S9 seeds. Writes research/s12_concrete_conservation/s12_summary.json"""
import gzip
import hashlib
import json
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
O = os.path.join(ROOT, "research", "s12_concrete_conservation")
SEEDS = json.load(open(os.path.join(ROOT, "research", "s9_native_transfer", "seeds.json")))["seeds"]
PLAIN = ("cycle_closing", "extension", "joins_components", "new_segment")
H = 60


def load(c, s):
    return json.load(gzip.open(os.path.join(O, "raw", f"{c}_{s}.json.gz"), "rt"))


def spells(flags):
    res, cur = [], 0
    for f in flags:
        if f:
            cur += 1
        elif cur:
            res.append(cur)
            cur = 0
    if cur:
        res.append(cur)
    return res


def metrics(d):
    st, end = d["steps"], d["end"]
    n = len(st)
    conc = [x["concrete"] for x in st]
    zero = [c == 0 for c in conc]
    zs = spells(zero)
    adds = [x for x in st if x["kind"] == "ADD"]
    plain = [x for x in adds if x["exec_cls"] in PLAIN]
    # reconstruct functional graph states for repeated-configuration counts
    F = {tuple(e) for e in d["header"]["F0"]}
    seen, rev = set(), 0
    conn_at, conn = [], set()
    spawns = []
    for x in st:
        F -= {tuple(e) for e in x["fr"]}
        F |= {tuple(e) for e in x["fa"]}
        h = hash(frozenset(F))
        if h in seen:
            rev += 1
        seen.add(h)
        if x["connected_houses"] is not None:
            conn = {tuple(t) for t in x["connected_houses"]}
        conn_at.append(set(conn))
        for ns in x["new_struct"]:
            if ns[2] == "House":
                spawns.append((x["i"], (ns[0], ns[1]), x["concrete"]))
    adapt = []
    for i0, h, c in spawns:
        adapt.append(None if i0 + H >= n else any(h in conn_at[j] for j in range(i0, i0 + H + 1)))
    vet = [x for x in st if x["n_vetoable"] > 0]
    ex_cls = {}
    for x in adds:
        ex_cls[x["exec_cls"]] = ex_cls.get(x["exec_cls"], 0) + 1
    first_conn = next((x["i"] for x in st if x["connected"] >= 1), None)
    return {"steps": n, "deliveries": st[-1]["score_cum"], "ge1_delivery": st[-1]["score_cum"] > 0, "reached_horizon": bool(end["truncated"]), "penalty_missing": end["penalty_missing"],
            "frac_zero_concrete": round(sum(zero) / n, 3), "decisions_zero_concrete": sum(zero), "longest_zero_spell": max(zs) if zs else 0,
            "concrete_at_house_spawns": [c for _, _, c in spawns], "spawned": len(spawns), "spawn_connected_60": sum(1 for a in adapt if a), "spawn_censored": sum(1 for a in adapt if a is None),
            "connections_peak": max(x["connected"] for x in st), "unconnected_end": st[-1]["houses"] - st[-1]["connected"], "houses_end": st[-1]["houses"], "first_connection": first_conn,
            "ADDs": len(adds), "concrete_consuming_ADDs": len(plain), "executed_cycle_closing": ex_cls.get("cycle_closing", 0), "house_replacements": sum(v for k, v in ex_cls.items() if k and k.startswith("house_")),
            "concrete_recovered_by_REMOVE": sum(x["concrete"] - x["concrete_prev"] for x in st if x["kind"] == "REMOVE" and x["topup"] == 0 and x["concrete"] > x["concrete_prev"]),
            "weekly_topups_total": sum(x["topup"] for x in st), "topup_events": sum(1 for x in st if x["topup"] > 0), "active_edges_end": st[-1]["n_func"], "nonfunctional_edges_mean": round(statistics.fmean(x["n_nonfunc"] for x in st), 2),
            "connection_destruction_events": sum(len(x["lost"]) for x in st), "repeated_functional_graph_states": rev, "legal_add_median": statistics.median(x["n_legal_add_next"] for x in st),
            "steps_with_vetoable": len(vet), "mean_vetoable_mass": round(statistics.fmean(x["vetoable_mass"] for x in vet), 4) if vet else None, "modified_actions": sum(1 for x in st if x.get("modified")),
            "only_wait_remains_cases": sum(1 for x in st if x.get("only_wait_remains")), "unknown_exec_ADDs": sum(v for k, v in ex_cls.items() if k and k.startswith("unknown")), "exec_classes": ex_cls,
            "concrete_end": conc[-1]}


M = {c: {s: metrics(load(c, s)) for s in SEEDS} for c in ("control", "conserve")}
agg = {}


def col(c, k): return [M[c][s][k] for s in SEEDS]
def med(x):
    x = [v for v in x if v is not None]
    return statistics.median(x) if x else None


for k in ("deliveries", "steps", "frac_zero_concrete", "longest_zero_spell", "connections_peak", "unconnected_end", "ADDs", "concrete_consuming_ADDs", "executed_cycle_closing", "house_replacements", "concrete_recovered_by_REMOVE",
          "weekly_topups_total", "active_edges_end", "connection_destruction_events", "repeated_functional_graph_states", "legal_add_median", "concrete_end"):
    a, b = col("control", k), col("conserve", k)
    d = [y - x for x, y in zip(a, b)]
    agg[k] = {"control_median": med(a), "conserve_median": med(b), "control_total": round(sum(a), 3), "conserve_total": round(sum(b), 3), "paired_diff_median": med(d), "higher": sum(1 for v in d if v > 0), "equal": sum(1 for v in d if v == 0),
              "lower": sum(1 for v in d if v < 0), "diffs": [round(v, 3) for v in d]}
for c in M:
    sp = sum(M[c][s]["spawned"] for s in SEEDS)
    ok = sum(M[c][s]["spawn_connected_60"] for s in SEEDS)
    cen = sum(M[c][s]["spawn_censored"] for s in SEEDS)
    allc = [v for s in SEEDS for v in M[c][s]["concrete_at_house_spawns"]]
    agg[f"spawn_{c}"] = {"spawned": sp, "connected_within_60": ok, "censored": cen, "rate_uncensored": round(ok / max(sp - cen, 1), 3), "median_concrete_at_house_spawn": med(allc), "share_of_spawns_with_concrete<=1": round(sum(1 for v in allc if v <= 1) / len(allc), 3)}
for k in ("ge1_delivery", "reached_horizon", "penalty_missing"):
    agg[k] = {"control": sum(col("control", k)), "conserve": sum(col("conserve", k))}
agg["guard"] = {"steps_with_vetoable_total": {c: sum(col(c, "steps_with_vetoable")) for c in M}, "steps_total": {c: sum(col(c, "steps")) for c in M}, "modified_actions_total_conserve": sum(col("conserve", "modified_actions")),
                "only_wait_remains_total": sum(col("conserve", "only_wait_remains_cases")), "mean_vetoable_mass_median_over_seeds": {c: med(col(c, "mean_vetoable_mass")) for c in M}, "unknown_exec_ADDs": {c: sum(col(c, "unknown_exec_ADDs")) for c in M}}
exec_share = {}
for c in M:
    tot = {}
    for s in SEEDS:
        for k, v in M[c][s]["exec_classes"].items():
            tot[k] = tot.get(k, 0) + v
    n = sum(tot.values())
    exec_share[c] = {k: round(v / n, 4) for k, v in sorted(tot.items(), key=lambda kv: -kv[1])}
agg["executed_ADD_class_share"] = exec_share
json.dump({"per_seed": M, "aggregate": agg}, open(os.path.join(O, "s12_summary.json"), "w"), indent=1, default=str)
print("seed | steps C/T | deliveries C/T | zero-conc frac C/T | longest zero C/T | cycleADD C/T | plainADD C/T | spawn conc med | peak conn C/T | unconn end C/T")
for s in SEEDS:
    c, t = M["control"][s], M["conserve"][s]
    print(s, "|", c["steps"], t["steps"], "|", c["deliveries"], t["deliveries"], "|", c["frac_zero_concrete"], t["frac_zero_concrete"], "|", c["longest_zero_spell"], t["longest_zero_spell"], "|", c["executed_cycle_closing"], t["executed_cycle_closing"], "|",
          c["concrete_consuming_ADDs"], t["concrete_consuming_ADDs"], "|", c["connections_peak"], t["connections_peak"], "|", c["unconnected_end"], t["unconnected_end"])
for k, v in agg.items():
    if isinstance(v, dict) and "control_median" in v:
        print(k, {a: v[a] for a in ("control_median", "conserve_median", "control_total", "conserve_total", "paired_diff_median", "higher", "equal", "lower")})
    else:
        print(k, v)
