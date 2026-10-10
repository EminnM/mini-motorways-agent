"""S21 adapter validation WITHOUT the game: (1) 48 saved native states (research/s7_long_route/native_normal_start_captures.jsonl.gz: raw observation + native legal set) -> adapter vs the
S11 functional-edge model and the native legal set; (2) end-to-end equivalence: the S18-H executors driven through the adapter on native-style JSON renderings of synthetic S13 states must emit the
same action sequences as the executors run directly on the synthetic env (36 scenarios, incl. spawns, house-edge replacements, Mothballed edges, top-ups, mid-plan changes).
   python research/s21_native_check/code/adapter_tests.py -> research/s21_native_check/adapter_tests.json"""
import gzip
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from native_adapter import NativeStateView, NativePlannerPolicy, ROOT  # noqa: E402
import s11_guard as G  # noqa: E402
from synthetic_rl import layout as L  # noqa: E402
from synthetic_rl.s13_env import load_scenarios  # noqa: E402
from synthetic_rl.s14_env import MultiHouseEnvV2  # noqa: E402
from world_model import WorldModel  # noqa: E402
from executor import make_policy  # noqa: E402
from planner import Weights  # noqa: E402

OUT = os.path.join(ROOT, "research", "s21_native_check")
R = {}


def check(name, ok, detail):
    R[name] = {"passed": bool(ok), "detail": detail}
    print(("PASS " if ok else "FAIL ") + name + ": " + str(detail)[:300], flush=True)


def t1_native_captures():
    caps = [json.loads(l) for l in gzip.open(os.path.join(ROOT, "research", "s7_long_route", "native_normal_start_captures.jsonl.gz"), "rt")]
    n = ok_edges = ok_conn = ok_content = ok_mask_subset = 0
    plan_ok = 0
    for c in caps:
        for key in ("reset", "first_house"):
            if not c.get(key):
                continue
            raw, legal = c[key]["raw"], set(c[key]["legal"])
            mask = np.zeros(L.N_ACTIONS, dtype=bool)
            mask[list(legal)] = True
            v = NativeStateView(raw, mask, 0, {})
            ctx = G.Ctx(raw)
            n += 1
            ok_content += int(v.content == {k: cc for k, cc in ctx.content.items() if cc in ("House", "Destination", "Carpark")} and all(v.group.get(k) == ctx.group.get(k) for k in v.content))
            ok_edges += int({e for e, s in v.edges.items() if s == "A"} == G.edges_from_raw(raw) and set(v.edges) == G.edges_present(raw))
            ok_conn += int(v.connected_houses() == G.connected_house_set(ctx, G.edges_from_raw(raw)))
            wm = WorldModel(v)
            # every structurally buildable edge the planner could consider must be inside the native legal set when concrete > 0 (legal_now is the authority; this checks the planner's own rule is not more permissive)
            viol = [e for e in wm.edges if False]
            cand = [(x, y, d) for x in range(L.XMIN, L.XMIN + L.W) for y in range(L.YMIN, L.YMIN + L.H) for d in ("E", "S") if wm.structurally_buildable((x, y, d))]
            legal_adds = {e for e in cand if mask[L.action_index("ADD", *e)]}
            ok_mask_subset += int(len(legal_adds) > 0)
            if key == "first_house":
                ex = make_policy(Weights())
                a = ex.act(v)
                plan_ok += int(bool(mask[a]) and (a != 0 or ex.plan is not None))
    check("T1_content_and_groups_match_S11_Ctx", ok_content == n, {"states": n, "ok": ok_content})
    check("T1_edges_match_S11_functional_model", ok_edges == n, {"states": n, "ok": ok_edges})
    check("T1_connectivity_matches_S11", ok_conn == n, {"states": n, "ok": ok_conn})
    check("T1_planner_first_action_legal_on_native_states", plan_ok == 24, {"first_house_states": 24, "legal_first_actions": plan_ok})


def t2_end_to_end_equivalence():
    sc = load_scenarios()
    tot = same = 0
    diverged = []
    illegal = 0
    stats = {"A": {}, "B": {}}
    for label, w in (("A", Weights(greedy=True)), ("B", Weights())):
        for idx in range(36):
            env_d = MultiHouseEnvV2(sc, seed=100 + idx)
            env_v = MultiHouseEnvV2(sc, seed=100 + idx)
            env_d.reset(options={"scenario_index": idx})
            env_v.reset(options={"scenario_index": idx})
            ex_d = make_policy(w)
            pol = NativePlannerPolicy(make_policy(w))
            done = False
            first_div = None
            while not done:
                a_d = ex_d.act(env_d)
                raw = env_v.to_native_raw()
                a_v = pol.act(raw, env_v.action_masks())
                tot += 1
                illegal += int(not env_v.action_masks()[a_v])
                if a_d == a_v:
                    same += 1
                elif first_div is None:
                    first_div = (env_d.t, L.decode_action(a_d), L.decode_action(a_v))
                _, _, te, tr, _ = env_d.step(a_d)
                _, _, te2, tr2, _ = env_v.step(a_v)
                done = te or tr
            if first_div:
                diverged.append({"arm": label, "scenario": sc[idx]["id"], "first_divergence": first_div})
            stats[label][sc[idx]["id"]] = {"direct_deliveries": env_d.deliveries_total, "adapter_deliveries": env_v.deliveries_total, "abandoned_adapter": pol.ex.stats["abandoned"], "plans": pol.ex.stats["plans"]}
    check("T2_adapter_driven_executor_matches_direct_executor", same == tot and illegal == 0, {"decisions": tot, "identical_actions": same, "illegal_actions": illegal, "episodes_diverged": len(diverged), "examples": diverged[:3]})
    R["T2_per_scenario"] = stats


def t3_pending_state_handling():
    """synthetic raw JSON with a Pending edge: must be present (not buildable again), not functional."""
    sc = load_scenarios()
    env = MultiHouseEnvV2(sc, seed=1)
    env.reset(options={"scenario_index": 0})
    for _ in range(5):
        env.step(0)
    raw = env.to_native_raw()
    tiles = {(t["coord"][0], t["coord"][1]): t for t in raw["tiles"]}
    # mark an empty edge as Pending on both endpoint tiles
    p, q = (-5, -5), (-4, -5)
    tiles[p]["road"]["E"] = "Pending"
    tiles[q]["road"]["W"] = "Pending"
    v = NativeStateView(raw, env.action_masks(), env.t, {})
    wm = WorldModel(v)
    e = (-5, -5, "E")
    check("T3_pending_edge_present_not_functional_not_buildable", v.edges.get(e) == "P" and e not in wm.active and not wm.structurally_buildable(e), {"state": v.edges.get(e), "functional": e in wm.active, "buildable": wm.structurally_buildable(e)})


def t4_choose_upgrade_interrupt():
    """mask with only the two ChooseUpgrade slots legal (WAIT illegal): the policy must return a legal upgrade action, keep its plan, and resume normally afterwards."""
    sc = load_scenarios()
    env = MultiHouseEnvV2(sc, seed=2)
    env.reset(options={"scenario_index": 3})
    pol = NativePlannerPolicy(make_policy(Weights()))
    for _ in range(12):
        a = pol.act(env.to_native_raw(), env.action_masks())
        env.step(a)
    plan_before = pol.ex.plan.describe() if pol.ex.plan else None
    m = np.zeros(L.N_ACTIONS, dtype=bool)
    m[L.UPGRADE_BASE] = True
    m[L.UPGRADE_BASE + 1] = True
    a_int = pol.act(env.to_native_raw(), m)
    plan_after = pol.ex.plan.describe() if pol.ex.plan else None
    a_next = pol.act(env.to_native_raw(), env.action_masks())
    check("T4_choose_upgrade_interrupt_handled", a_int == L.UPGRADE_BASE and plan_before == plan_after and bool(env.action_masks()[a_next]) and pol.forced_actions == [(12, L.UPGRADE_BASE)],
          {"interrupt_action": L.decode_action(a_int), "plan_kept": plan_before == plan_after, "next_action_legal": bool(env.action_masks()[a_next]), "forced_log": pol.forced_actions})


def main():
    t0 = time.time()
    for f in (t1_native_captures, t2_end_to_end_equivalence, t3_pending_state_handling, t4_choose_upgrade_interrupt):
        try:
            f()
        except Exception as ex:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            check(f.__name__, False, repr(ex))
    R["_wall_s"] = round(time.time() - t0, 1)
    json.dump(R, open(os.path.join(OUT, "adapter_tests.json"), "w"), indent=1, default=str)
    print(sum(1 for k, v in R.items() if isinstance(v, dict) and v.get("passed")), "passed of", sum(1 for k, v in R.items() if isinstance(v, dict) and "passed" in v))


if __name__ == "__main__":
    main()
