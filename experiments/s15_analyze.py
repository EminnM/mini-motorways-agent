"""S15 analysis: retention gate, preregistered checkpoint selection (identical rule for both arms), paired A-vs-B and vs-step-0 differences on the matched S13 validation episodes, Arm A group-channel
ablation (inference only), forgetting timeline. Validation data only.   python experiments/s15_analyze.py -> research/s15_mixed_training/analysis.json"""
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sb3_contrib import MaskablePPO  # noqa: E402
from synthetic_rl import s15_eval as EV  # noqa: E402
from synthetic_rl.models_v2 import SpatialMaskablePolicyV2  # noqa: E402,F401

OUT = os.path.join(ROOT, "research", "s15_mixed_training")
GATE = {"k7_det": 0.75, "k10_det": 0.65, "wait_fraction_max": 0.5}


def load_run(arm):
    d = os.path.join(OUT, "runs", f"arm_{arm}_s15")
    run = json.load(open(os.path.join(d, "run.json")))
    rows = {e["decisions"]: json.load(open(os.path.join(d, "rows", f"step_{e['decisions']}.json"))) for e in run["evaluations"]}
    return run, rows


def passes_gate(ev):
    s7 = ev["s7"]
    return s7["k7_det"]["success"] >= GATE["k7_det"] and s7["k10_det"]["success"] >= GATE["k10_det"] and s7["k7_det"]["wait_fraction"] < GATE["wait_fraction_max"] and s7["k7_sto"]["wait_fraction"] < GATE["wait_fraction_max"]


def select(run):
    """1 reject failing retention; 2 strongest S13 validation: primary = (new_rate_60, deliveries_median) lexicographic with a tie band of 0.02 / 2 deliveries; 3 ties -> earlier checkpoint. Zero-decision checkpoint is a candidate too."""
    cands = [e for e in run["evaluations"] if passes_gate(e)]
    if not cands:
        return None, "no checkpoint passes the S7 retention gate"
    best = None
    for e in sorted(cands, key=lambda e: e["decisions"]):
        if best is None:
            best = e
            continue
        s, b = e["s13"], best["s13"]
        better = (s["new_rate_60"] > b["new_rate_60"] + 0.02) or (abs(s["new_rate_60"] - b["new_rate_60"]) <= 0.02 and s["deliveries_median"] > b["deliveries_median"] + 2)
        if better:
            best = e
    return best, "retention-passing checkpoint with the highest new-house connection rate (tie band 0.02), then median deliveries (tie band 2), ties -> earlier"


def paired(rows_a, rows_b, keys=("deliveries", "return", "new_connected_60", "destroyed", "decisions", "frac_zero_concrete", "plain_adds", "misrouted_house_steps")):
    ka = {(r["scenario"], r["env_seed"]): r for r in rows_a}
    kb = {(r["scenario"], r["env_seed"]): r for r in rows_b}
    out = {}
    for k in keys:
        d = [ka[x][k] - kb[x][k] for x in ka if x in kb]
        out[k] = {"n": len(d), "median_diff": float(np.median(d)), "mean_diff": round(float(np.mean(d)), 3), "a_higher": sum(1 for v in d if v > 0), "equal": sum(1 for v in d if v == 0), "a_lower": sum(1 for v in d if v < 0), "iqr": [float(np.percentile(d, 25)), float(np.percentile(d, 75))]}
    return out


def main():
    torch.set_num_threads(2)
    res = {"gate": GATE}
    runs, rows = {}, {}
    for arm in "AB":
        runs[arm], rows[arm] = load_run(arm)
    # retention timeline + selection
    for arm in "AB":
        run = runs[arm]
        res[f"arm_{arm}"] = {"timeline": [{"decisions": e["decisions"], "k7_det": e["s7"]["k7_det"]["success"], "k7_sto": e["s7"]["k7_sto"]["success"], "k10_det": e["s7"]["k10_det"]["success"], "k10_sto": e["s7"]["k10_sto"]["success"],
                                           "wait_frac_k7_det": e["s7"]["k7_det"]["wait_fraction"], "wait_prob_fixed_obs": e["wait_prob_fixed_s7_obs"], "passes_gate": passes_gate(e),
                                           "s13_new_rate_60": e["s13"]["new_rate_60"], "s13_deliveries_median": e["s13"]["deliveries_median"], "s13_return_mean": e["s13"]["return_mean"], "s13_survived": e["s13"]["survived"],
                                           "s13_destroyed_median": e["s13"]["destroyed_median"], "s13_zero_concrete": e["s13"]["frac_zero_concrete_median"], "s13_wait_while_disconnected": e["s13"]["wait_frac_while_disconnected_mean"],
                                           "group_w_abs_mean": e["group_channel_weights"]["abs_mean_group"]} for e in run["evaluations"]],
                             "mixture": run.get("mixture_stats_final"), "throughput": run.get("throughput"), "wall_s": run.get("wall_s"), "stopped": run.get("stopped"), "init_bitwise_equal": run["init"]["parameters_bitwise_equal_to_init"]}
        sel, why = select(run)
        res[f"arm_{arm}"]["selected"] = {"decisions": sel["decisions"] if sel else None, "rationale": why}
        if sel:
            base = run["evaluations"][0]
            res[f"arm_{arm}"]["selected_vs_step0_paired"] = paired(rows[arm][sel["decisions"]]["s13"], rows[arm][0]["s13"])
    # A vs B at matched checkpoints
    res["A_vs_B_matched"] = {}
    for e in runs["A"]["evaluations"]:
        d = e["decisions"]
        if d in rows["B"]:
            res["A_vs_B_matched"][d] = paired(rows["A"][d]["s13"], rows["B"][d]["s13"])
    # ablation of Arm A's selected checkpoint (inference only, same seeds)
    selA = res["arm_A"]["selected"]["decisions"]
    if selA is not None:
        ck = next(c["path"] for c in runs["A"]["checkpoints"] if c["decisions"] == selA) if selA > 0 else os.path.join(ROOT, "research", "s14_group_observation", "s8c_migrated_v2_step0.zip")
        m = MaskablePPO.load(ck, device="cpu")
        t0 = time.time()
        pol = EV.Policy(m, blind=True)
        s7_sum, _ = EV.eval_s7(pol)
        s13_sum, s13_rows = EV.eval_s13(pol)
        res["ablation_arm_A_selected_groups_zeroed"] = {"checkpoint_decisions": selA, "s7": {k: {"success": v["success"], "wait_fraction": v["wait_fraction"]} for k, v in s7_sum.items()}, "s13": s13_sum,
                                                        "paired_intact_minus_ablated": paired(rows["A"][selA]["s13"], s13_rows), "eval_s": round(time.time() - t0, 1)}
        json.dump(s13_rows, open(os.path.join(OUT, "ablation_rows.json"), "w"))
    json.dump(res, open(os.path.join(OUT, "analysis.json"), "w"), indent=1, default=str)
    for arm in "AB":
        print(arm, "selected", res[f"arm_{arm}"]["selected"])
        for t in res[f"arm_{arm}"]["timeline"]:
            print("  ", t)
    print("A_vs_B", json.dumps(res["A_vs_B_matched"])[:1500])
    if "ablation_arm_A_selected_groups_zeroed" in res:
        print("ablation", json.dumps(res["ablation_arm_A_selected_groups_zeroed"]["paired_intact_minus_ablated"])[:1200])


if __name__ == "__main__":
    main()
