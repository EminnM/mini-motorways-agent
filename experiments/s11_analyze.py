"""S11 analysis: paired control vs guard on the 12 S9 seeds (functional connectivity = Active/VisiblyActive edges). Writes research/s11_connection_guard/s11_summary.json"""
import gzip
import json
import os
import statistics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
O = os.path.join(ROOT, "research", "s11_connection_guard")
SEEDS = json.load(open(os.path.join(ROOT, "research", "s9_native_transfer", "seeds.json")))["seeds"]
H_SPAWN = 60


def load(cond, s):
    return json.load(gzip.open(os.path.join(O, "raw", f"s9_{cond}_{s}.json.gz"), "rt"))


def spells(flags):
    res, cur = [], 0
    for f in flags:
        if f:
            cur += 1
        elif cur:
            res.append(cur); cur = 0
    if cur: res.append(cur)
    return res


def metrics(d):
    st, end = d["steps"], d["end"]
    n = len(st)
    cum = st[-1]["score_cum"]
    conn_first = next((x["i"] for x in st if x["connected"] >= 1), None)
    deliv_first = next((x["i"] for x in st if x["scoreDelta"] > 0), None)
    conc = [x["concrete"] for x in st]
    zero = [c == 0 for c in conc]
    zs = spells(zero)
    adds = [x for x in st if x["kind"] == "ADD"]
    rep = [x for x in adds if x["pred"] and x["pred"]["reason"] == "house_edge_replacement"]
    lost_events = sum(len(x["lost"]) for x in st)
    lost_steps = sum(1 for x in st if x["lost"])
    destructive_adds = sum(1 for x in adds if x["lost"])
    nondestr_rep = sum(1 for x in rep if not x["lost"])
    noop = sum(1 for x in adds if x["pred"] and x["pred"]["reason"] == "same_house_edge_noop")
    # connected-house set over time and spawn adaptation
    conn = set()
    conn_at = []
    for x in st:
        if x["connected_houses"] is not None:
            conn = {tuple(h) for h in x["connected_houses"]}
        conn_at.append(set(conn))
    spawns = []
    for x in st:
        for h in x["new_houses"]:
            spawns.append((x["i"], tuple(h)))
    adapt = []
    for i0, h in spawns:
        if i0 + H_SPAWN >= n:
            adapt.append(None)
            continue
        adapt.append(any(h in conn_at[j] for j in range(i0, i0 + H_SPAWN + 1)))
    ever_connected = set().union(*conn_at) if conn_at else set()
    vet_steps = [x for x in st if x["n_vetoed"] > 0]
    mod_steps = sum(1 for x in st if x.get("modified"))
    mass = [x["vetoed_mass"] for x in vet_steps if "vetoed_mass" in x]
    return {"steps": n, "deliveries": cum, "ge1_delivery": cum > 0, "first_connection": conn_first, "first_delivery": deliv_first, "reached_horizon": bool(end["truncated"]), "penalty_missing": end["penalty_missing"],
            "connections_peak": max(x["connected"] for x in st), "houses_ever_connected": len(ever_connected), "houses_end": st[-1]["houses"], "connected_end": st[-1]["connected"], "unconnected_end": st[-1]["houses"] - st[-1]["connected"],
            "connected_legacy_end": st[-1]["connected_legacy"], "connection_loss_events": lost_events, "steps_with_loss": lost_steps, "destructive_ADDs": destructive_adds,
            "ADDs": len(adds), "replacement_ADDs": len(rep), "nondestructive_replacements": nondestr_rep, "noop_ADDs": noop, "REMOVEs": sum(1 for x in st if x["kind"] == "REMOVE"),
            "concrete_at_first_connection": conc[conn_first] if conn_first is not None else None, "concrete_end": conc[-1], "decisions_zero_concrete": sum(zero), "fraction_zero_concrete": round(sum(zero) / n, 3),
            "longest_zero_spell": max(zs) if zs else 0, "exhausted_ge10": (max(zs) if zs else 0) >= 10, "legal_add_median": statistics.median(x["n_legal_add_next"] for x in st), "legal_remove_median": statistics.median(x["n_legal_remove_next"] for x in st),
            "spawned_houses": len(spawns), "spawned_connected_within_60": sum(1 for a in adapt if a), "spawned_censored": sum(1 for a in adapt if a is None),
            "guard_steps_with_vetoes": len(vet_steps), "guard_modified_actions": mod_steps, "mean_vetoed_prob_mass": round(statistics.fmean(mass), 3) if mass else None, "max_vetoed_prob_mass": round(max(mass), 3) if mass else None,
            "unknown_counterfactual_steps": sum(1 for x in st if x["n_unknown"] > 0), "pred_mismatch_adds": sum(1 for x in adds if x["pred"] and x["pred"]["known"] and x["pred"]["match"] is False and not x["pred"]["spawn_in_step"]),
            "score_by_quarter": [round(st[min(int(n * q / 4), n) - 1]["score_cum"], 0) for q in (1, 2, 3, 4)]}


M = {c: {s: metrics(load(c, s)) for s in SEEDS} for c in ("control", "guard")}
summ = {"per_seed": {c: M[c] for c in M}}
agg = {}
def col(c, k): return [M[c][s][k] for s in SEEDS]
def med(x):
    x = [v for v in x if v is not None]; return statistics.median(x) if x else None
for k in ("deliveries", "steps", "connection_loss_events", "destructive_ADDs", "connections_peak", "houses_ever_connected", "unconnected_end", "ADDs", "replacement_ADDs", "decisions_zero_concrete", "fraction_zero_concrete", "concrete_end", "first_connection", "first_delivery", "legal_add_median"):
    a, b = col("control", k), col("guard", k)
    d = [y - x for x, y in zip(a, b) if x is not None and y is not None]
    agg[k] = {"control_median": med(a), "guard_median": med(b), "control_total": sum(v for v in a if v is not None), "guard_total": sum(v for v in b if v is not None), "paired_diff_median": med(d),
              "seeds_guard_higher": sum(1 for v in d if v > 0), "seeds_equal": sum(1 for v in d if v == 0), "seeds_guard_lower": sum(1 for v in d if v < 0), "paired_diffs": d}
for k in ("ge1_delivery", "reached_horizon", "exhausted_ge10", "penalty_missing"):
    agg[k] = {"control": sum(col("control", k)), "guard": sum(col("guard", k))}
for c in M:
    sp = sum(M[c][s]["spawned_houses"] for s in SEEDS); ok = sum(M[c][s]["spawned_connected_within_60"] for s in SEEDS); cen = sum(M[c][s]["spawned_censored"] for s in SEEDS)
    agg[f"spawn_adaptation_{c}"] = {"spawned": sp, "connected_within_60": ok, "censored(<60 decisions left)": cen, "rate_among_uncensored": round(ok / max(sp - cen, 1), 3)}
agg["guard_stats"] = {"steps_with_vetoes_total": sum(col("guard", "guard_steps_with_vetoes")), "modified_actions_total": sum(col("guard", "guard_modified_actions")), "unknown_steps_total": sum(col("guard", "unknown_counterfactual_steps")),
                      "control_steps_where_guard_would_veto_something": sum(col("control", "guard_steps_with_vetoes")), "mean_vetoed_mass_guard_median_over_seeds": med(col("guard", "mean_vetoed_prob_mass")),
                      "pred_mismatch_adds_control": sum(col("control", "pred_mismatch_adds")), "pred_mismatch_adds_guard": sum(col("guard", "pred_mismatch_adds"))}
summ["aggregate"] = agg
json.dump(summ, open(os.path.join(O, "s11_summary.json"), "w"), indent=1, default=str)
print("seed  | steps C/G | deliveries C/G | lossEvents C/G | peak conn C/G | unconnected_end C/G | zero-concrete frac C/G | first conn C/G | hist")
for s in SEEDS:
    c, g = M["control"][s], M["guard"][s]
    print(s, "|", c["steps"], g["steps"], "|", c["deliveries"], g["deliveries"], "|", c["connection_loss_events"], g["connection_loss_events"], "|", c["connections_peak"], g["connections_peak"], "|", c["unconnected_end"], g["unconnected_end"], "|", c["fraction_zero_concrete"], g["fraction_zero_concrete"], "|", c["first_connection"], g["first_connection"])
for k, v in agg.items():
    if isinstance(v, dict) and "control_median" in v:
        print(k, {a: v[a] for a in ("control_median", "guard_median", "control_total", "guard_total", "paired_diff_median", "seeds_guard_higher", "seeds_equal", "seeds_guard_lower")})
    else:
        print(k, v)
