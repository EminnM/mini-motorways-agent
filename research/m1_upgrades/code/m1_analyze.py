"""M1 analysis: Stage A validation (export reliability, choice frequency, pair diversity, apply-type agreement) and Stage B paired comparison (slot0 / random / ruleR).
   python research/m1_upgrades/code/m1_analyze.py -> research/m1_upgrades/results.json"""
import collections
import gzip
import json
import os
import statistics as S
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "s21_native_check", "code"))
from native_adapter import ROOT  # noqa: E402

OUT = os.path.join(ROOT, "research", "m1_upgrades")
SEEDS = json.load(open(os.path.join(ROOT, "research", "s9_native_transfer", "seeds.json")))["seeds"]
ARMS = ("slot0", "random", "ruleR")
H = 60


def ep_metrics(d):
    st = d["steps"]
    n = len(st)
    spawns = [(x["i"], tuple(h)) for x in st for h in x["new_houses"]]
    # latency: first decision at which 'connected' count increased after spawn is not house-specific; use connected/houses at end + within-60 by count growth proxy is unreliable -> report connected_end only
    return {"decisions": n, "deliveries": st[-1]["score_cum"], "terminated": d["end"]["terminated"], "survived_600": n >= 600, "connected_end": st[-1]["connected"], "houses_end": st[-1]["houses"], "illegal": sum(1 for x in st if not x["legal"]),
            "n_choices": len(d["choices"]), "min_concrete": min(x["concrete"] for x in st), "zero_concrete_frac": sum(1 for x in st if x["concrete"] == 0) / n, "abandoned": d["end"]["exec_stats"].get("abandoned"),
            "inventory_end": d["choices"][-1]["inventory_after"] if d["choices"] else None}


def boot_ci(v, n=10000, seed=0):
    v = np.asarray(v, dtype=float)
    if len(v) == 0:
        return None
    rng = np.random.default_rng(seed)
    return [float(np.percentile([v[rng.integers(len(v), size=len(v))].mean() for _ in range(n)], q)) for q in (2.5, 97.5)]


def main():
    D = {a: {} for a in ARMS}
    for a in ARMS:
        for s in SEEDS:
            p = os.path.join(OUT, "raw", f"{a}_{s}.json.gz")
            if os.path.exists(p):
                D[a][s] = json.load(gzip.open(p, "rt"))
    # ---- Stage A (all arms' choice logs)
    all_choices = [c for a in ARMS for s in D[a] for c in D[a][s]["choices"]]
    n_eps = sum(len(D[a]) for a in ARMS)
    export_ok = all(len(c["offered"]) == 2 for c in all_choices)
    apply_match = sum(1 for c in all_choices if c["chosen_type_apply"] == c["chosen_type_export"])
    differ = sum(1 for c in all_choices if c["offered"][0]["type"] != c["offered"][1]["type"])
    pairs = collections.Counter(tuple(sorted(o["type"] for o in c["offered"])) for c in all_choices)
    types = collections.Counter(o["type"] for c in all_choices for o in c["offered"])
    inv_check = []
    for c in all_choices:
        t = c["chosen_type_apply"] or ""
        key = {"Concrete": "concrete", "Bridge": "bridges", "Motorway": "motorways", "TrafficLight": "trafficLights", "Roundabout": "roundabouts", "Tunnel": "tunnels"}.get(t)
        before, after = c["inventory_before"], c["inventory_after"]
        inv_check.append({"type": t, "amount": c["offered"][c["slot"]]["amount"], "addConcrete": c["offered"][c["slot"]]["additionalConcrete"], "concrete_total_delta": after["concrete"][1] - before["concrete"][1], "type_total_delta": (after[key][1] - before[key][1]) if key else None})
    per_choice_slot0_choices = [c for s in D["slot0"] for c in D["slot0"][s]["choices"]]
    stageA = {"episodes": n_eps, "choices_total": len(all_choices), "choices_per_episode_mean": len(all_choices) / max(1, n_eps), "export_two_options_always": export_ok, "apply_type_matches_export": [apply_match, len(all_choices)],
              "offered_types_differ": [differ, len(all_choices)], "pair_frequency": {" | ".join(k): v for k, v in pairs.most_common()}, "type_frequency": dict(types), "inventory_delta_samples": inv_check[:12],
              "choice_decisions_slot0": sorted({c["i"] for c in per_choice_slot0_choices}),
              "pass": export_ok and apply_match == len(all_choices) and len(all_choices) / max(1, n_eps) >= 2 and differ / max(1, len(all_choices)) >= 0.5}
    # ---- Stage B paired
    M = {a: {s: ep_metrics(D[a][s]) for s in D[a]} for a in ARMS}
    summ = {a: {"n": len(M[a]), "deliveries_total": sum(m["deliveries"] for m in M[a].values()), "deliveries_median": S.median(m["deliveries"] for m in M[a].values()) if M[a] else None, "decisions_median": S.median(m["decisions"] for m in M[a].values()) if M[a] else None,
                "survived_600": sum(m["survived_600"] for m in M[a].values()), "illegal": sum(m["illegal"] for m in M[a].values()), "choices": sum(m["n_choices"] for m in M[a].values()), "min_concrete_median": S.median(m["min_concrete"] for m in M[a].values()) if M[a] else None,
                "chosen_types": dict(collections.Counter(c["chosen_type_apply"] for s in D[a] for c in D[a][s]["choices"]))} for a in ARMS}
    paired = {}
    for a, b in (("ruleR", "slot0"), ("random", "slot0"), ("ruleR", "random")):
        common = [s for s in SEEDS if s in M[a] and s in M[b]]
        dd = [M[a][s]["deliveries"] - M[b][s]["deliveries"] for s in common]
        ds = [M[a][s]["decisions"] - M[b][s]["decisions"] for s in common]
        same_choices = sum(1 for s in common if [c["slot"] for c in D[a][s]["choices"]] == [c["slot"] for c in D[b][s]["choices"]])
        paired[f"{a}_minus_{b}"] = {"n": len(common), "deliveries": {"values": dd, "sum": sum(dd), "median": S.median(dd) if dd else None, "higher": sum(1 for x in dd if x > 0), "equal": sum(1 for x in dd if x == 0), "lower": sum(1 for x in dd if x < 0), "ci95_mean": boot_ci(dd)},
                                    "decisions": {"values": ds, "median": S.median(ds) if ds else None, "higher": sum(1 for x in ds if x > 0), "equal": sum(1 for x in ds if x == 0), "lower": sum(1 for x in ds if x < 0)}, "seeds_with_identical_choice_sequence": same_choices}
    per_seed = {s: {a: ({"deliveries": M[a][s]["deliveries"], "decisions": M[a][s]["decisions"], "choices": [(c["i"], [o["type"] for o in c["offered"]], c["slot"], c["chosen_type_apply"]) for c in D[a][s]["choices"]]} if s in M[a] else None) for a in ARMS} for s in SEEDS}
    res = {"stageA": stageA, "arms": summ, "paired": paired, "per_seed": per_seed}
    json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=1, default=str)
    print("STAGE A", json.dumps({k: v for k, v in stageA.items() if k != "inventory_delta_samples"}))
    print("inventory samples", stageA["inventory_delta_samples"][:6])
    print("ARMS", json.dumps(summ))
    print("PAIRED", json.dumps(paired))
    for s in SEEDS:
        print(s, {a: (v["deliveries"], v["decisions"], [(c[0], c[3]) for c in v["choices"]]) if v else None for a, v in per_seed[s].items()})


if __name__ == "__main__":
    main()
