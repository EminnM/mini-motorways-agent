"""S9: per-seed tables, paired comparisons, legality and failure-mode indicators (reads research/s9_native_transfer/rows/*.jsonl)."""
import gzip, json, os, statistics, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "research", "s9_native_transfer")
W = ["p1det", "p1sto", "p2det", "p2sto"]
rows = {w: {json.loads(l)["seed"]: json.loads(l) for l in open(os.path.join(OUT, "rows", w + ".jsonl"))} for w in W}
seeds = json.load(open(os.path.join(OUT, "seeds.json")))["seeds"]
def med(x): x = [v for v in x if v is not None]; return statistics.median(x) if x else None
def q(x):
    x = sorted(v for v in x if v is not None)
    return None if not x else {"n": len(x), "min": x[0], "median": statistics.median(x), "mean": round(statistics.fmean(x), 2), "max": x[-1]}
summ = {}
for w in W:
    r = [rows[w][s] for s in seeds if s in rows[w]]
    n = len(r)
    summ[w] = {"episodes": n, "ge1_connection": sum(1 for x in r if x["diag_connections_max"] > 0), "ge1_delivery": sum(1 for x in r if x["diag_deliveries"] > 0),
               "connections_max": q([x["diag_connections_max"] for x in r]), "deliveries": q([x["diag_deliveries"] for x in r]), "first_connection_step": q([x["diag_first_connection_step"] for x in r]),
               "adds_before_first_connection": q([x["n_add_before_first_connection"] for x in r]), "irrelevant_adds_before_first_connection": q([x["irrelevant_adds_before_first_connection"] for x in r]),
               "first_delivery_step": q([x["diag_first_delivery_step"] for x in r]), "survival": q([x["n_steps"] for x in r]), "native_score": q([x["native_score"] for x in r]),
               "penalty_missing": sum(1 for x in r if x["diag_terminal_penalty_missing"]), "distinct_edges": q([x["diag_distinct_edges_added"] for x in r]), "n_add": q([x["n_add"] for x in r]),
               "n_remove": q([x["n_remove"] for x in r]), "house_swaps": q([x["n_house_swaps"] for x in r]), "noop_adds": q([x["n_noop_adds"] for x in r]), "water_endpoint_adds": q([x["n_water_endpoint_adds"] for x in r]),
               "dest_adjacent_adds": q([x["n_dest_adjacent_adds"] for x in r]), "oscillation_steps": q([x["road_state_stats"]["two_state_oscillation_steps"] for x in r]), "two_state_cycles": q([x["road_state_stats"]["two_state_cycles"] for x in r]),
               "revisits": q([x["road_state_stats"]["revisits"] for x in r]), "mean_unmasked_mass_on_illegal": q([x["mean_unmasked_mass_on_illegal"] for x in r]),
               "frac_steps_unmasked_top1_illegal": q([x["frac_steps_unmasked_top1_illegal"] for x in r]), "houses_max": q([x["n_houses_max"] for x in r]), "k_required": q([x["k_required_native_mask_oracle"] for x in r]),
               "remaining_min_reached": q([x["remaining_min_reached"] for x in r]), "legal_mean": q([x["legal_mean"] for x in r]), "cut": sum(1 for x in r if x["cut_by_wallclock_or_steplimit"]),
               "action_fractions": {k: sum(x[k + "_count"] for x in r) / max(sum(x["length_decisions"] for x in r), 1) for k in ("wait", "add", "remove", "upgrade")},
               "policy_entropy_mean": q([x["diag_policy_entropy_mean"] for x in r]), "wall_s_total": round(sum(x["wall_s"] for x in r), 1)}
paired = {}
for mode in ("det", "sto"):
    a, b = rows["p1" + mode], rows["p2" + mode]
    def d(key, f=lambda x: x):
        out = []
        for s in seeds:
            if s in a and s in b and a[s][key] is not None and b[s][key] is not None: out.append(f(b[s][key]) - f(a[s][key]))
        return out
    paired[mode] = {"connections_max_diff_P2_minus_P1": d("diag_connections_max"), "deliveries_diff": d("diag_deliveries"), "first_connection_step_diff_where_both": d("diag_first_connection_step"),
                    "adds_before_conn_diff_where_both": d("n_add_before_first_connection"), "survival_diff": d("n_steps"), "distinct_edges_diff": d("diag_distinct_edges_added"),
                    "seeds_P2_connects_P1_not": [s for s in seeds if b[s]["diag_connections_max"] > 0 and a[s]["diag_connections_max"] == 0], "seeds_P1_connects_P2_not": [s for s in seeds if a[s]["diag_connections_max"] > 0 and b[s]["diag_connections_max"] == 0],
                    "seeds_P2_delivers_P1_not": [s for s in seeds if b[s]["diag_deliveries"] > 0 and a[s]["diag_deliveries"] == 0], "seeds_P1_delivers_P2_not": [s for s in seeds if a[s]["diag_deliveries"] > 0 and b[s]["diag_deliveries"] == 0]}
    paired[mode]["median_diffs"] = {k: med(v) for k, v in paired[mode].items() if k.endswith("diff") or k.endswith("both")}
per_seed = []
for s in seeds:
    rec = {"seed": s}
    for w in W:
        x = rows[w][s]
        rec[w] = {"k_req": x["k_required_native_mask_oracle"], "conn": x["diag_connections_max"], "first_conn": x["diag_first_connection_step"], "deliv": x["diag_deliveries"], "first_deliv": x["diag_first_delivery_step"], "steps": x["n_steps"],
                  "adds": x["n_add"], "swaps": x["n_house_swaps"], "distinct": x["diag_distinct_edges_added"], "osc": x["road_state_stats"]["two_state_oscillation_steps"], "houses_max": x["n_houses_max"], "pen_missing": x["diag_terminal_penalty_missing"],
                  "rem_min": x["remaining_min_reached"]}
    per_seed.append(rec)
json.dump({"summary": summ, "paired": paired, "per_seed": per_seed}, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
print("policy  ep  >=1conn  >=1deliv  conn(med,max)  deliv(med,max)  firstconn(med)  survival(med)  adds(med)  swaps(med)  osc(med)  houses_max(med)  illegal-top1(med)")
for w in W:
    s = summ[w]
    print(w, s["episodes"], s["ge1_connection"], s["ge1_delivery"], (s["connections_max"]["median"], s["connections_max"]["max"]), (s["deliveries"]["median"], s["deliveries"]["max"]), s["first_connection_step"] and s["first_connection_step"]["median"], s["survival"]["median"], s["n_add"]["median"], s["house_swaps"]["median"], s["oscillation_steps"]["median"], s["houses_max"]["median"], s["frac_steps_unmasked_top1_illegal"] and round(s["frac_steps_unmasked_top1_illegal"]["median"], 2))
print(json.dumps(paired, indent=0)[:1800])
print("k_required (native mask oracle):", summ["p1det"]["k_required"])
for r in per_seed: print(r["seed"], {w: (r[w]["k_req"], r[w]["conn"], r[w]["first_conn"], r[w]["deliv"], r[w]["steps"]) for w in W})
