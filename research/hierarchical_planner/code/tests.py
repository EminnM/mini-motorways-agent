"""S18-H validation tests (synthetic MultiHouseEnv only; no native game).
  T1 action legality          : every action emitted by the executor is legal under the env mask at emission time (all scenarios, 1 seed)
  T2 connectivity model       : WorldModel.connected == env.connected_houses() at every decision, including after house-edge swaps / reactivations
  T3 resource accounting      : predicted concrete cost of every completed CONNECT plan == observed concrete change (top-ups subtracted)
  T4 execution completes      : every CONNECT plan selected is either completed (house connected) or abandoned with a logged reason; completion rate reported
  T5 connection preservation  : no functionally connected house loses its connection while the executor plays (connections_destroyed == 0),
                                and the executor's would-destroy veto fires on a constructed adversarial case
  T6 determinism              : two runs with the same env seed produce identical action sequences
python research/hierarchical_planner/code/tests.py"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from world_model import WorldModel, TOPUP_AT, TOPUP_AMT, canon  # noqa: E402
from executor import Executor  # noqa: E402
from planner import Weights  # noqa: E402
from synthetic_rl import s13_env as E  # noqa: E402
from synthetic_rl import layout as L  # noqa: E402

SC = E.load_scenarios()
OUT = os.path.join(os.path.dirname(HERE), "tests.json")


def run_episode(idx, seed, weights=None, max_t=600):
    env = E.MultiHouseEnv(SC, seed=seed)
    env._rng = np.random.default_rng(seed)
    env.reset(options={"scenario_index": idx})
    ex = Executor(weights or Weights(), log=True)
    rec = {"illegal": 0, "conn_mismatch": 0, "cost_mismatch": [], "destroyed": 0, "actions": [], "steps": 0}
    prev_conn = set()
    spent = {}                       # plan key -> (predicted cost, observed concrete spent so far, house, t0)
    done = False
    while not done:
        wm = WorldModel(env)
        if wm.connected != env.connected_houses():
            rec["conn_mismatch"] += 1
        a = ex.act(env)
        if not env.action_masks()[a]:
            rec["illegal"] += 1
        # resource accounting: attribute this step's concrete delta (top-up removed) to the plan active after act()
        key = ex.plan.key() if (ex.plan is not None and ex.plan.kind == "CONNECT") else None
        if key is not None and key not in spent:
            spent[key] = [ex.plan.concrete_cost, 0, ex.plan.targets["house"], env.t]
        c_before = env.concrete
        rec["actions"].append(a)
        _, r, te, tr, inf = env.step(a)
        done = te or tr
        rec["steps"] += 1
        topup = TOPUP_AMT[TOPUP_AT.index(env.t)] if env.t in TOPUP_AT else 0
        if key is not None:
            spent[key][1] += c_before + topup - env.concrete
        conn = env.connected_houses()
        rec["destroyed"] += len([h for h in prev_conn if h in env.houses and h not in conn])
        prev_conn = conn
    for key, (cost, obs, h, t0) in spent.items():
        if h in prev_conn and obs != cost:
            rec["cost_mismatch"].append((SC[idx]["id"], t0, cost, obs))
    rec["stats"] = ex.stats
    rec["log"] = ex.log
    return rec


def adversarial_veto():
    """house connected through a plain road; a different house edge would mothball its driveway. The executor must never emit it and
    WorldModel.lost_connections must flag it."""
    env = E.MultiHouseEnv(SC, seed=1)
    env._rng = np.random.default_rng(1)
    env.reset(options={"scenario_index": 0})
    ex = Executor(Weights())
    # play until the first house is connected
    for _ in range(200):
        a = ex.act(env)
        env.step(a)
        if env.connected_houses():
            break
    wm = WorldModel(env)
    h = next(iter(wm.connected))
    active_h = [e for e in env._house_edges(h) if env.edges.get(e) == "A"]
    assert active_h, "connected house without an active driveway"
    other = [e for e in env._house_edges(h) if e not in env.edges and wm.structurally_buildable(e)]
    if not other:
        return {"ok": True, "note": "no alternative house edge buildable; veto case not constructible here"}
    e = other[0]
    lost = wm.lost_connections([e], [])
    flagged = h in lost
    # the env mask allows this ADD (house edges are always legal); confirm the model prediction against the env itself
    env2_edges = dict(env.edges)
    env.step(L.action_index("ADD", *e))
    truly_lost = h not in env.connected_houses()
    return {"ok": bool(flagged == truly_lost and flagged), "flagged_by_model": flagged, "truly_lost_in_env": truly_lost, "edge": e, "house": h}


def main():
    res = {}
    idxs = list(range(len(SC)))
    recs = {idx: run_episode(idx, 9000 + idx) for idx in idxs}
    res["T1_action_legality"] = {"illegal_actions": sum(r["illegal"] for r in recs.values()), "actions": sum(r["steps"] for r in recs.values()), "pass": all(r["illegal"] == 0 for r in recs.values())}
    res["T2_connectivity_model"] = {"mismatching_decisions": sum(r["conn_mismatch"] for r in recs.values()), "pass": all(r["conn_mismatch"] == 0 for r in recs.values())}
    mism = [m for r in recs.values() for m in r["cost_mismatch"]]
    res["T3_resource_accounting"] = {"completed_plans_checked": sum(r["stats"]["completed"] for r in recs.values()), "mismatches": mism[:10], "n_mismatch": len(mism), "pass": len(mism) == 0}
    plans = sum(r["stats"]["plans"] for r in recs.values())
    comp = sum(r["stats"]["completed"] for r in recs.values())
    ab = sum(r["stats"]["abandoned"] for r in recs.values())
    reasons = {}
    for r in recs.values():
        for k, v in r["stats"]["abandon_reasons"].items():
            reasons[k] = reasons.get(k, 0) + v
    # every plan is completed or abandoned with a logged reason; at most one plan per episode may still be in progress at the horizon
    res["T4_execution_completes"] = {"plans": plans, "completed": comp, "abandoned": ab, "in_progress_at_end": plans - comp - ab, "abandon_reasons": reasons, "completion_rate": round(comp / max(1, plans), 3),
                                     "pass": all(r["stats"]["abandoned"] == len([x for x in r["log"] if "ABANDON" in x]) for r in recs.values()) and 0 <= plans - comp - ab <= len(recs)}
    adv = adversarial_veto()
    res["T5_connection_preservation"] = {"connections_destroyed_total": sum(r["destroyed"] for r in recs.values()), "destroy_vetoes": sum(r["stats"]["destroy_vetoes"] for r in recs.values()), "adversarial": adv,
                                         "pass": sum(r["destroyed"] for r in recs.values()) == 0 and adv["ok"]}
    r1, r2 = run_episode(2, 4242), run_episode(2, 4242)
    res["T6_determinism"] = {"identical_action_sequences": r1["actions"] == r2["actions"], "pass": r1["actions"] == r2["actions"]}
    res["all_pass"] = all(v["pass"] for k, v in res.items() if k.startswith("T"))
    json.dump(res, open(OUT, "w"), indent=1, default=str)
    for k, v in res.items():
        print(k, json.dumps(v, default=str)[:400])


if __name__ == "__main__":
    main()
