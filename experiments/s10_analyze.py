"""S10 analysis: replay measurements (4 episodes, n=4) + S9 trajectories (24 S8-C episodes) -> research/s10_post_connection/s10_summary.json and printed tables."""
import gzip
import json
import os
import statistics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
O = os.path.join(ROOT, "research", "s10_post_connection")
S9 = os.path.join(ROOT, "research", "s9_native_transfer")
SEEDS = [563352, 523757, 539595, 587109]
R = {s: json.load(gzip.open(os.path.join(O, f"replay_{s}.json.gz"), "rt")) for s in SEEDS}
S9ROWS = {w: {json.loads(l)["seed"]: json.loads(l) for l in open(os.path.join(S9, "rows", w + ".jsonl"))} for w in ("p2det", "p2sto", "p1det", "p1sto")}
s9seeds = json.load(open(os.path.join(S9, "seeds.json")))["seeds"]
out = {"replay": {}}


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


allrows = []
for s, d in R.items():
    st = d["steps"]
    n = len(st)
    first = d["final"]["first"]
    conc = [x["concrete"] for x in st]
    tot = [x["concrete_total"] for x in st]
    topups = [(st[i]["i"], tot[i] - tot[i - 1], conc[i - 1], conc[i]) for i in range(1, n) if tot[i] > tot[i - 1]]
    spawn_h = [st[i]["i"] for i in range(1, n) if st[i]["houses"] > st[i - 1]["houses"]]
    spawn_d = [st[i]["i"] for i in range(1, n) if st[i]["dests"] > st[i - 1]["dests"]]
    zero = [c == 0 for c in conc]
    zsp = spells(zero)
    adds = [x for x in st if x["kind"] == "ADD"]
    rep = [x for x in adds if x["removed"] >= 1]
    noop = [x for x in adds if x["added"] == 0 and x["removed"] == 0]
    prod = [x for x in adds if x["removed"] == 0 and x["added"] >= 1]
    lost = [(x["i"], x["kind"], x["touched_house"], x["conn_lost"]) for x in st if x["conn_lost"] > 0]
    la_zero = [x["n_legal_add_next"] for x in st if x["concrete"] == 0]
    la_pos = [x["n_legal_add_next"] for x in st if x["concrete"] > 0 and x["n_legal_add_next"] > 0]
    unconn_steps_pos_conc = sum(1 for x in st if x["houses"] - x["connected"] > 0 and x["concrete"] > 0)
    unconn_steps_zero_conc = sum(1 for x in st if x["houses"] - x["connected"] > 0 and x["concrete"] == 0)
    last = st[-1]
    rc = d["route_checks"]
    no_route_zero = [c for c in rc if c["concrete"] == 0 for v in c["min_adds_per_unconnected_house_under_current_mask"]]
    ev = {"steps": n, "first_connection": first["conn"], "first_delivery": first["deliv"], "deliveries": last["score_cum"],
          "concrete_at_first_connection": conc[first["conn"]] if first["conn"] is not None else None, "concrete_at_first_delivery": conc[first["deliv"]] if first["deliv"] is not None else None,
          "concrete_at_house_spawns": [conc[i] for i in spawn_h], "house_spawn_steps": spawn_h, "destination_spawn_steps": spawn_d, "concrete_at_destination_spawns": [conc[i] for i in spawn_d],
          "concrete_end": conc[-1], "concrete_min": min(conc), "concrete_total_start_end": [tot[0], tot[-1]], "concrete_topups(step,+total,before,after)": topups,
          "decisions_with_zero_concrete": sum(zero), "fraction_zero_concrete": round(sum(zero) / n, 3), "zero_concrete_spells": zsp, "longest_zero_spell": max(zsp) if zsp else 0,
          "legal_add_median_when_concrete_0": statistics.median(la_zero) if la_zero else None, "legal_add_median_when_concrete_pos": statistics.median(la_pos) if la_pos else None,
          "ADDs": len(adds), "productive_ADDs(net growth)": len(prod), "replacement_ADDs(removed>=1)": len(rep), "noop_ADDs": len(noop), "REMOVEs": sum(1 for x in st if x["kind"] == "REMOVE"),
          "edges_start_end": [st[0]["edges"], last["edges"]], "connections_lost_events(step,kind,touched_house,n)": lost,
          "houses_end": last["houses"], "connected_end": last["connected"], "destinations_end": last["dests"], "overcrowding_destinations_end": [(c["group"], c["demand"], c["overcrowding"]) for c in d["final"]["dest_state_end"]],
          "decisions_with_unconnected_houses_and_concrete>0": unconn_steps_pos_conc, "decisions_with_unconnected_houses_and_concrete=0": unconn_steps_zero_conc,
          "route_checks": rc, "final_reward": st[-1]["reward"], "terminated": d["final"]["terminated"], "last_delivery": max([x["i"] for x in st if x["scoreDelta"] > 0], default=None),
          "decisions_first_connection_to_termination": (n - 1 - first["conn"]) if first["conn"] is not None else None, "vehicles_max": max(x["vehicles"] for x in st), "demand_max_over_episode": max(x["dest_demand_max"] for x in st)}
    out["replay"][s] = ev
    allrows += [(x["concrete"], x["n_legal"], x["n_legal_add_next"]) for x in st]
# proxy validation: legal total in [5,40] <-> concrete == 0  (decisions pooled over the 4 replays; not independent samples)
prox = [(5 <= nl <= 40, c == 0) for c, nl, na in allrows]
tp = sum(1 for p, z in prox if p and z); fp = sum(1 for p, z in prox if p and not z); fn = sum(1 for p, z in prox if not p and z)
out["proxy_validation"] = {"decisions": len(prox), "legal_total_in_5_40_and_concrete0": tp, "legal_in_5_40_but_concrete_positive": fp, "concrete0_but_legal_outside_5_40": fn}
# proxy applied to the 24 S9 S8-C episodes (INFERRED, not measured)
proxy = {}
for w in ("p2det", "p2sto"):
    for s in s9seeds:
        steps = json.load(gzip.open(os.path.join(S9, "raw", f"{w}_{s}.json.gz"), "rt"))["steps"]
        n = len(steps)
        coll = [5 <= x[6] <= 40 for x in steps]
        adds = [x for x in steps if x[2] == "ADD"]
        rep = [x for x in adds if x[11] >= 1]
        r = S9ROWS[w][s]
        fc = r["diag_first_connection_step"]
        proxy[f"{w}_{s}"] = {"steps": n, "proxy_zero_concrete_decisions": sum(coll), "proxy_zero_concrete_fraction": round(sum(coll) / n, 3), "first_connection": fc, "deliveries": r["diag_deliveries"],
                             "houses_end": steps[-1][12], "connected_end": steps[-1][13], "ADDs": len(adds), "replacement_ADD_share": round(len(rep) / len(adds), 3) if adds else None,
                             "late_first_connection(>=100 or none)": (fc is None or fc >= 100), "oscillation_steps": r["road_state_stats"]["two_state_oscillation_steps"],
                             "penalty_missing": r["diag_terminal_penalty_missing"], "terminal_on_entry": r["diag_terminal_on_entry"], "final_applyInfo": (r["diag_final_status"] or {}).get("applyInfo"),
                             "last_delivery_to_end": (n - 1 - max([x[0] for x in steps if x[8] > 0])) if any(x[8] > 0 for x in steps) else None}
out["s9_p2_proxy"] = proxy
json.dump(out, open(os.path.join(O, "s10_summary.json"), "w"), indent=1, default=str)
for s, e in out["replay"].items():
    print("=== seed", s)
    for k, v in e.items():
        if k not in ("route_checks",):
            print("  ", k, ":", v)
    print("   route_checks (step, concrete, min ADDs per unconnected house under current mask):", [(c["i"], c["concrete"], c["min_adds_per_unconnected_house_under_current_mask"]) for c in e["route_checks"]][:14])
print("proxy validation", out["proxy_validation"])
