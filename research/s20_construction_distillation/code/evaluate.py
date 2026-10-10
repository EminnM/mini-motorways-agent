"""S20 autonomous evaluation on held-out TEST scenes (paired: the same task instances for every policy).
Policies: A planner (goal-conditioned S18-H executor, reference), B learned executor (greedy and sampling), B0 BC-only model if a DAgger
checkpoint exists, C frozen S8-C (v1 observation via obs_v2.strip_to_v1 semantics = env._obs; NO goal -> caveat), RANDOM legal ADD/REMOVE.
Budget = max(6, 3 x planner steps).  Metrics: completion, legal-action rate, concrete per completed connection, steps vs planner, destroyed
connections, breakdowns (route length, existing edges, valid-set size, prefix kind, region), paired bootstrap CIs, failure examples."""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import rollout as R  # noqa: E402
from model import GoalPolicy  # noqa: E402

torch.set_num_threads(2)
DATA = os.path.join(C.S20, "data")
RUN = os.path.join(C.S20, "runs")
S8C = os.path.join(C.ROOT, "research", "s8_long_route_training", "runs", "arm_C_s11", "checkpoints", "step_0099840.zip")


def budget_of(sc):
    return max(6, 3 * sc["meta"]["n_adds"])


def run_policy(name, make, scenes, query_done):
    pol = make()
    rows = []
    for i, sc in enumerate(scenes):
        if isinstance(pol, R.PlannerPolicy):
            pol = R.PlannerPolicy()
        r = R.run_task(sc, pol, budget_of(sc), query_done=query_done)
        r["scene"] = i
        rows.append(r)
    extra = {}
    if isinstance(pol, R.LearnedPolicy):
        extra["raw_argmax_legal_rate"] = pol.raw_legal / max(1, pol.calls)
    return rows, extra


def boot_ci(x, n=2000, seed=0):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def summarize(rows, scenes):
    s = np.array([r["success"] for r in rows], dtype=float)
    steps = np.array([r["steps"] for r in rows])
    plan = np.array([sc["meta"]["n_adds"] for sc in scenes])
    ok = s > 0
    out = {"n": len(rows), "completion": float(s.mean()), "completion_ci": boot_ci(s), "legal_rate": float(1 - sum(r["illegal"] for r in rows) / max(1, steps.sum() + sum(r["illegal"] for r in rows))),
           "illegal_total": int(sum(r["illegal"] for r in rows)), "destroyed_total": int(sum(r["destroyed"] for r in rows)), "tasks_with_destroyed": int(sum(r["destroyed"] > 0 for r in rows)),
           "premature_done": int(sum(r["premature_done"] for r in rows)), "water_adds": int(sum(r["water_adds"] for r in rows)), "removes": int(sum(r["removes"] for r in rows)),
           "other_house_swaps": int(sum(r["other_house_swaps"] for r in rows))}
    if ok.any():
        cs = np.array([r["concrete_spent"] for r in rows])[ok]
        out["concrete_per_completed_mean"] = float(cs.mean())
        out["planner_cost_on_completed_mean"] = float(np.array([sc["meta"]["planner_cost"] for sc in scenes])[ok].mean())
        out["steps_on_completed_mean"] = float(steps[ok].mean())
        out["planner_steps_on_completed_mean"] = float(plan[ok].mean())
        out["steps_ratio_vs_planner"] = float(steps[ok].sum() / max(1, plan[ok].sum()))
        out["steps_eq_planner_frac"] = float((steps[ok] == plan[ok]).mean())
    d = [r["done_at_completion"] for r in rows if r["done_at_completion"] is not None]
    if d:
        out["done_at_completion_rate"] = float(np.mean(d))
    return out


def breakdown(rows, scenes, key_fn, name):
    groups = {}
    for r, sc in zip(rows, scenes):
        groups.setdefault(key_fn(sc), []).append(r["success"])
    return {name: {str(k): {"n": len(v), "completion": float(np.mean(v)), "ci": boot_ci(v)} for k, v in sorted(groups.items())}}


BINS = {
    "route_len": lambda sc: {True: "2-4", False: None}[sc["meta"]["route_len"] <= 4] or ("5-7" if sc["meta"]["route_len"] <= 7 else ("8-10" if sc["meta"]["route_len"] <= 10 else "11-14")),
    "planner_steps": lambda sc: "1-2" if sc["meta"]["n_adds"] <= 2 else ("3-5" if sc["meta"]["n_adds"] <= 5 else ("6-8" if sc["meta"]["n_adds"] <= 8 else "9-14")),
    "existing_edges": lambda sc: "0-9" if sc["meta"]["n_edges_existing"] <= 9 else ("10-19" if sc["meta"]["n_edges_existing"] <= 19 else ("20-29" if sc["meta"]["n_edges_existing"] <= 29 else "30+")),
    "valid_set_size": lambda sc: "1" if sc["meta"]["valid_size"] == 1 else ("2-3" if sc["meta"]["valid_size"] <= 3 else "4+"),
    "prefix": lambda sc: sc["meta"]["prefix_kind"],
    "region": lambda sc: sc["meta"]["region"],
    "n_dest": lambda sc: str(sc["meta"]["n_dest"]),
}


def main():
    t0 = time.time()
    scenes = json.load(open(os.path.join(DATA, "scenes_test.json")))
    tl = json.load(open(os.path.join(RUN, "train_log.json")))["info"]
    final = tl["final_checkpoint"]
    model = GoalPolicy()
    model.load_state_dict(torch.load(os.path.join(RUN, final)))
    model.eval()
    policies = {"A_planner": (lambda: R.PlannerPolicy(), False),
                "B_learned_greedy": (lambda: R.LearnedPolicy(model, "greedy", seed=11), True),
                "B_learned_sample": (lambda: R.LearnedPolicy(model, "sample", seed=12), True),
                "Bdiag_greedy_DONE_masked": (lambda: R.LearnedPolicy(model, "greedy", seed=14, allow_done=False), False)}
    if final != "model_bc.pt":
        model0 = GoalPolicy()
        model0.load_state_dict(torch.load(os.path.join(RUN, "model_bc.pt")))
        model0.eval()
        policies["B0_bc_only_greedy"] = (lambda: R.LearnedPolicy(model0, "greedy", seed=13), True)
    from sb3_contrib import MaskablePPO
    s8c = MaskablePPO.load(S8C, device="cpu")
    policies["C_s8c_frozen_nogoal"] = (lambda: R.S8CPolicy(s8c), False)
    policies["RANDOM_legal"] = (lambda: R.RandomLegalPolicy(seed=5), False)
    results, per_scene = {}, {}
    for name, (make, qd) in policies.items():
        t = time.time()
        rows, extra = run_policy(name, make, scenes, qd)
        per_scene[name] = rows
        results[name] = {**summarize(rows, scenes), **extra, "wall_s": round(time.time() - t, 1)}
        for bname, fn in BINS.items():
            results[name].update(breakdown(rows, scenes, fn, f"by_{bname}"))
        print(name, json.dumps({k: v for k, v in results[name].items() if not k.startswith("by_")}), flush=True)
    # paired comparisons (per-scene success differences, bootstrap over scenes)
    paired = {}
    sA = np.array([r["success"] for r in per_scene["A_planner"]], dtype=float)
    for name in policies:
        if name == "A_planner":
            continue
        sB = np.array([r["success"] for r in per_scene[name]], dtype=float)
        diff = sB - sA
        paired[f"{name}_minus_A"] = {"mean_diff": float(diff.mean()), "ci": boot_ci(diff), "wins": int((diff > 0).sum()), "losses": int((diff < 0).sum())}
    for a, b in (("B_learned_greedy", "C_s8c_frozen_nogoal"), ("B_learned_greedy", "RANDOM_legal"), ("B_learned_greedy", "B_learned_sample"), ("B_learned_greedy", "B0_bc_only_greedy"), ("Bdiag_greedy_DONE_masked", "B_learned_greedy")):
        if a in per_scene and b in per_scene:
            d = np.array([r["success"] for r in per_scene[a]], dtype=float) - np.array([r["success"] for r in per_scene[b]], dtype=float)
            paired[f"{a}_minus_{b}"] = {"mean_diff": float(d.mean()), "ci": boot_ci(d), "wins": int((d > 0).sum()), "losses": int((d < 0).sum())}
    # failure examples of the learned greedy policy
    fails = []
    for r, sc in zip(per_scene["B_learned_greedy"], scenes):
        if not r["success"] and len(fails) < 12:
            fails.append({"seed": sc["meta"]["seed"], "region": sc["meta"]["region"], "route_len": sc["meta"]["route_len"], "planner_steps": sc["meta"]["n_adds"], "existing_edges": sc["meta"]["n_edges_existing"],
                          "prefix": sc["meta"]["prefix_kind"], "valid_size": sc["meta"]["valid_size"], "steps_used": r["steps"], "budget": r["budget"], "premature_done": r["premature_done"], "destroyed": r["destroyed"],
                          "removes": r["removes"], "other_house_swaps": r["other_house_swaps"], "water_adds": r["water_adds"]})
    out = {"n_test_scenes": len(scenes), "final_checkpoint": final, "results": results, "paired": paired, "failures_B_greedy": fails, "wall_s": round(time.time() - t0, 1)}
    json.dump(out, open(os.path.join(RUN, "eval_test.json"), "w"), indent=1)
    json.dump(per_scene, open(os.path.join(RUN, "eval_test_per_scene.json"), "w"))
    print("paired", json.dumps(paired), flush=True)


if __name__ == "__main__":
    main()
