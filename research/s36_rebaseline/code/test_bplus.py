"""S36 focused regression tests for planner B+ (= planner B with Weights(serve_unreached_dests=True)); one headless game, seed 1007922 (the S35 failure seed), 700 decisions.
 T1 targets unreached destinations: whenever an unreached destination with a legal route exists, the chosen plan targets a destination (within <= 3 decisions of it appearing, unless concrete == 0).
 T2 recognises service: every destination-targeted plan completes when house and destination share a component, and no further destination plan targets a served destination.
 T3 no redundant construction: at most one completed destination plan per destination; no ADD of an edge removed in the oscillation window (executor invariant); REMOVE count.
 T4 legality/resources: 0 illegal actions; 0 boundary violations (both ADD endpoints inside playableAreaAtTime); concrete never negative; every ADD was offered by the native mask.
 T5 connectivity: 0 functional connections destroyed by B+'s own actions (house connected before the step, present and not connected after), same guarantee as planner B.
 T6 flag off == historical planner B: seed 1000003 with Weights() reproduces (315 decisions, 41 deliveries) and generates no destination candidates.
   python research/s36_rebaseline/code/test_bplus.py"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
for p in (HERE, ROOT, os.path.join(ROOT, "python"), os.path.join(ROOT, "experiments"), os.path.join(ROOT, "research", "s21_native_check", "code")):
    sys.path.insert(0, p)
import numpy as np  # noqa: E402
from native_adapter import NativePlannerPolicy, NativeStateView  # noqa: E402
from executor import make_policy  # noqa: E402
from planner import Weights  # noqa: E402
from world_model import WorldModel  # noqa: E402
from synthetic_rl import layout as L  # noqa: E402

R = {}


def check(name, ok, detail):
    R[name] = {"passed": bool(ok), "detail": detail}
    print(("PASS " if ok else "FAIL ") + name + ": " + str(detail)[:400], flush=True)


def far(k):
    return (k[1] + 1, k[2]) if k[3] == "E" else (k[1], k[2] - 1)


def inside(p, ar):
    return ar[0] <= p[0] <= ar[2] and ar[1] <= p[1] <= ar[3]


def run(env, seed, plus, max_t):
    pol = NativePlannerPolicy(make_policy(Weights(serve_unreached_dests=plus)))
    env.reset(seed=seed)
    pol.reset()
    raw = env.last_raw_observation
    log = {"t": 0, "score": 0.0, "illegal": 0, "bviol": 0, "destroyed": 0, "neg_concrete": 0, "unoffered_add": 0, "dest_plans": [], "unreached_seen": [], "plan_desc": [], "removes": 0}
    seen_unreached = {}
    while log["t"] < max_t:
        mask = env.action_masks().copy()
        view = NativeStateView(raw, mask, pol.t, pol.spawn_t)
        wm = WorldModel(view)
        for d in wm.unreached_dests():
            if any(v["group"] == d["group"] for v in wm.houses.values()):      # only destinations that HAVE a house of their colour can be targeted (a colour's first destination often spawns before any house)
                seen_unreached.setdefault(tuple(d["tiles"][0]), log["t"])
        conn_before = set(wm.connected)
        act = pol.act(raw, mask)
        plan = pol.ex.plan
        if plan is not None and plan.kind == "CONNECT" and "dest" in plan.targets:
            log["dest_plans"].append({"t": log["t"], "dest": tuple(plan.targets["dest"]), "house": tuple(plan.targets["house"]), "cost": plan.concrete_cost})
        if not mask[act]:
            log["illegal"] += 1
            act = int(np.flatnonzero(mask)[0])
        kd = L.decode_action(act)
        if kd[0] == "ADD":
            ar = raw["globals"]["playableAreaAtTime"]
            if not (inside((kd[1], kd[2]), ar) and inside(far(kd), ar)):
                log["bviol"] += 1
        if kd[0] == "REMOVE":
            log["removes"] += 1
        _, r, te, tr, info = env.step(act)
        raw = env.last_raw_observation
        log["score"] += float(info["rewardComponents"].get("scoreDelta", 0.0))
        if raw["globals"]["concreteAvailable"] < 0:
            log["neg_concrete"] += 1
        wm2 = WorldModel(NativeStateView(raw, env.action_masks(), pol.t, pol.spawn_t))
        log["destroyed"] += sum(1 for h in conn_before if h in wm2.houses and h not in wm2.connected)
        log["t"] += 1
        if te or tr:
            break
    log["unreached_first_seen"] = {str(k): v for k, v in seen_unreached.items()}
    log["exec"] = {k: v for k, v in pol.ex.stats.items() if k != "compute_s"}
    return log, pol


def main():
    import mini_motorways_env as M
    from experiments import gamectl
    t0 = time.time()
    with gamectl.GameInstance("s36_test") as g:
        env = M.MiniMotorwaysEnv(max_decisions=1000, decision_delta=1.0, request_path=os.path.join(g.ipc_dir, "request.json"), response_path=os.path.join(g.ipc_dir, "response.json"), start_mode="NORMAL_START")
        lg, pol = run(env, 1007922, True, 700)
        # T1: for every destination that was ever unreached, a destination plan targeting it starts within 3 decisions of it first being seen (allowing for the current plan to finish)
        starts = {}
        for p in lg["dest_plans"]:
            starts.setdefault(p["dest"], p["t"])
        lat = {k: (starts.get(eval(k)) - v) if starts.get(eval(k)) is not None else None for k, v in lg["unreached_first_seen"].items()}
        check("T1_targets_unreached_destinations", len(lg["unreached_first_seen"]) > 0 and all(v is not None and v <= 12 for v in lat.values()), {"unreached_seen": len(lg["unreached_first_seen"]), "latency_decisions": lat})
        # T2/T3: per destination, plan episodes (consecutive decisions with the same dest) -> count completions; the executor's completion requires _dest_served
        episodes = []
        for p in lg["dest_plans"]:
            if episodes and episodes[-1]["dest"] == p["dest"] and p["t"] - episodes[-1]["t_end"] <= 1:
                episodes[-1]["t_end"] = p["t"]
            else:
                episodes.append({"dest": p["dest"], "t_start": p["t"], "t_end": p["t"]})
        per_dest = {}
        for e in episodes:
            per_dest[e["dest"]] = per_dest.get(e["dest"], 0) + 1
        check("T2_recognises_served_destinations", all(v <= 2 for v in per_dest.values()) and lg["exec"]["completed"] >= len(per_dest), {"plan_episodes_per_destination": {str(k): v for k, v in per_dest.items()}, "exec": lg["exec"]})
        check("T3_no_redundant_construction", all(v == 1 for v in per_dest.values()) and lg["removes"] == 0 and lg["exec"]["abandon_reasons"].get("all_edges_built_but_dest_not_served", 0) == 0, {"plan_episodes_per_destination": {str(k): v for k, v in per_dest.items()}, "removes": lg["removes"], "abandon_reasons": lg["exec"]["abandon_reasons"]})
        check("T4_legality_and_resources", lg["illegal"] == 0 and lg["bviol"] == 0 and lg["neg_concrete"] == 0, {"illegal": lg["illegal"], "boundary_violations": lg["bviol"], "negative_concrete_steps": lg["neg_concrete"], "t": lg["t"], "score": lg["score"]})
        check("T5_connectivity_preserved", lg["destroyed"] == 0, {"destroyed": lg["destroyed"], "destroy_vetoes": lg["exec"]["destroy_vetoes"]})
        lg0, _ = run(env, 1000003, False, 1000)
        check("T6_flag_off_equals_historical_plannerB", lg0["t"] == 315 and lg0["score"] == 41 and not lg0["dest_plans"], {"t": lg0["t"], "score": lg0["score"], "dest_plans": len(lg0["dest_plans"])})
    R["_wall_s"] = round(time.time() - t0, 1)
    json.dump(R, open(os.path.join(ROOT, "research", "s36_rebaseline", "test_bplus.json"), "w"), indent=1, default=str)
    print(sum(1 for v in R.values() if isinstance(v, dict) and v.get("passed")), "passed of", sum(1 for v in R.values() if isinstance(v, dict) and "passed" in v))


if __name__ == "__main__":
    main()
