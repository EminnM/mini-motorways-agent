"""S13a: native noise floor. Compares four sets of 12 paired native episodes (S8-C stochastic, same 12 game seeds): S11 control, S12 control (same sampling seeds as S11), the S13 re-seeded replicate (independent sampling seeds),
and the treated arms (S11 guard, S12 conserve) for reassessment.   python experiments/s13_noise.py"""
import gzip
import json
import os
import statistics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEEDS = json.load(open(os.path.join(ROOT, "research", "s9_native_transfer", "seeds.json")))["seeds"]
OUT = os.path.join(ROOT, "research", "s13_multi_house")
SRC = {"S11_control": ("research/s11_connection_guard/raw/s9_control_{s}.json.gz", 11), "S11_guard": ("research/s11_connection_guard/raw/s9_guard_{s}.json.gz", 11),
       "S12_control": ("research/s12_concrete_conservation/raw/control_{s}.json.gz", 12), "S12_conserve": ("research/s12_concrete_conservation/raw/conserve_{s}.json.gz", 12),
       "S13_replicate": ("research/s13_multi_house/raw/replicate_{s}.json.gz", 13)}
H = 60


def load(name, s):
    return json.load(gzip.open(os.path.join(ROOT, SRC[name][0].format(s=s)), "rt"))


def metrics(d, kind):
    st = d["steps"]
    n = len(st)
    conn_first = next((x["i"] for x in st if x["connected"] >= 1), None)
    conn, conn_at, spawns = set(), [], []
    for x in st:
        if x["connected_houses"] is not None:
            conn = {tuple(h) for h in x["connected_houses"]}
        conn_at.append(set(conn))
        news = [tuple(h) for h in x["new_houses"]] if kind == 11 else [(ns[0], ns[1]) for ns in x["new_struct"] if ns[2] == "House"]
        for h in news:
            spawns.append((x["i"], h))
    ok = cen = 0
    for i0, h in spawns:
        if i0 + H >= n:
            cen += 1
        elif any(h in conn_at[j] for j in range(i0, i0 + H + 1)):
            ok += 1
    return {"steps": n, "deliveries": st[-1]["score_cum"], "first_connection": conn_first, "peak_connections": max(x["connected"] for x in st), "frac_zero_concrete": sum(1 for x in st if x["concrete"] == 0) / n,
            "spawned": len(spawns), "spawned_connected_60": ok, "spawned_censored": cen, "actions": [x["action"] for x in st]}


def first_div(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


M = {name: {s: metrics(load(name, s), SRC[name][1]) for s in SEEDS} for name in SRC}
res = {"sampling_seeds": {"S11_S12": "1000 + i (i = 0..11)", "S13_replicate": "6000 + i (i = 0..11)"}, "native_init": {"start_mode": "NORMAL_START", "max_decisions": 600, "decision_delta": 1.0, "game_seed": "S9 seeds 500000 + 7919 i"}}
pairs = {"S12ctrl_vs_S11ctrl(same sampling seed)": ("S12_control", "S11_control"), "replicate_vs_S12ctrl(indep. sampling seed)": ("S13_replicate", "S12_control"), "replicate_vs_S11ctrl(indep. sampling seed)": ("S13_replicate", "S11_control")}
pc = {}
for label, (a, b) in pairs.items():
    rows = []
    for s in SEEDS:
        ma, mb = M[a][s], M[b][s]
        fd = first_div(ma["actions"], mb["actions"])
        rows.append({"seed": s, "deliveries_diff": ma["deliveries"] - mb["deliveries"], "steps_diff": ma["steps"] - mb["steps"], "first_conn_a": ma["first_connection"], "first_conn_b": mb["first_connection"],
                     "peak_conn_diff": ma["peak_connections"] - mb["peak_connections"], "zero_conc_diff": round(ma["frac_zero_concrete"] - mb["frac_zero_concrete"], 3), "spawned_connected_diff": ma["spawned_connected_60"] - mb["spawned_connected_60"],
                     "identical_action_sequence": fd is None, "first_divergence_decision": fd})
    d = [r["deliveries_diff"] for r in rows]
    pc[label] = {"per_seed": rows, "deliveries_diff_sum": sum(d), "deliveries_diff_median": statistics.median(d), "deliveries_diff_median_abs": statistics.median(abs(x) for x in d), "deliveries_diff_sd": round(statistics.pstdev(d), 1),
                 "steps_diff_median_abs": statistics.median(abs(r["steps_diff"]) for r in rows), "identical_sequences": sum(r["identical_action_sequence"] for r in rows),
                 "seeds_a_higher": sum(1 for x in d if x > 0), "seeds_equal": sum(1 for x in d if x == 0), "seeds_a_lower": sum(1 for x in d if x < 0),
                 "median_first_divergence_where_diverged": statistics.median([r["first_divergence_decision"] for r in rows if r["first_divergence_decision"] is not None] or [None])}
res["pairs"] = pc
tot = {name: {"deliveries_total": sum(M[name][s]["deliveries"] for s in SEEDS), "deliveries_median": statistics.median(M[name][s]["deliveries"] for s in SEEDS), "steps_total": sum(M[name][s]["steps"] for s in SEEDS),
              "steps_median": statistics.median(M[name][s]["steps"] for s in SEEDS), "peak_connections_median": statistics.median(M[name][s]["peak_connections"] for s in SEEDS), "frac_zero_concrete_median": round(statistics.median(M[name][s]["frac_zero_concrete"] for s in SEEDS), 3),
              "spawn_connected_60": [sum(M[name][s]["spawned_connected_60"] for s in SEEDS), sum(M[name][s]["spawned"] - M[name][s]["spawned_censored"] for s in SEEDS)]} for name in SRC}
res["set_summaries"] = tot
# reassessment: treated arms vs each control realisation and vs the spread of control realisations
re = {}
for treat, ctrls in (("S11_guard", ("S11_control", "S13_replicate")), ("S12_conserve", ("S12_control", "S13_replicate"))):
    for c in ctrls:
        d = [M[treat][s]["deliveries"] - M[c][s]["deliveries"] for s in SEEDS]
        top3 = sorted(d, reverse=True)[:3]
        re[f"{treat} - {c}"] = {"sum": sum(d), "median": statistics.median(d), "higher": sum(1 for x in d if x > 0), "equal": sum(1 for x in d if x == 0), "lower": sum(1 for x in d if x < 0), "sum_without_top3_gains": sum(d) - sum(top3), "diffs": d}
    ctrl_vals = {s: [M[c][s]["deliveries"] for c in ("S11_control", "S12_control", "S13_replicate")] for s in SEEDS}
    outside = [s for s in SEEDS if M[treat][s]["deliveries"] > max(ctrl_vals[s]) or M[treat][s]["deliveries"] < min(ctrl_vals[s])]
    re[f"{treat}: seeds whose treated deliveries fall OUTSIDE the range of the 3 control realisations"] = {"n": len(outside), "seeds": outside}
res["reassessment"] = re
res["control_realisations_per_seed_deliveries"] = {s: [M[c][s]["deliveries"] for c in ("S11_control", "S12_control", "S13_replicate")] for s in SEEDS}
json.dump(res, open(os.path.join(OUT, "noise_floor.json"), "w"), indent=1, default=str)
for k, v in pc.items():
    print(k, {a: v[a] for a in v if a != "per_seed"})
print("set summaries", json.dumps(tot, indent=0)[:1500])
print("per seed deliveries [S11ctrl, S12ctrl, replicate | S11guard, S12conserve]")
for s in SEEDS:
    print(s, [M[c][s]["deliveries"] for c in ("S11_control", "S12_control", "S13_replicate")], "|", M["S11_guard"][s]["deliveries"], M["S12_conserve"][s]["deliveries"], "| steps", [M[c][s]["steps"] for c in ("S11_control", "S12_control", "S13_replicate")])
for k, v in re.items():
    print(k, v if "n" in v else {a: v[a] for a in v if a != "diffs"})
for r in pc["replicate_vs_S12ctrl(indep. sampling seed)"]["per_seed"]:
    print(r["seed"], "dDeliv", r["deliveries_diff"], "dSteps", r["steps_diff"], "firstconn", r["first_conn_a"], r["first_conn_b"], "divergence", r["first_divergence_decision"])
for r in pc["S12ctrl_vs_S11ctrl(same sampling seed)"]["per_seed"]:
    if not r["identical_action_sequence"]:
        print("same-seed non-identical:", r["seed"], "dDeliv", r["deliveries_diff"], "first divergence", r["first_divergence_decision"])
