"""Overnight step 1b (second half): LIVE Mothballed-handling probe. Planner B plays; once a house is connected, ONE scripted house-driveway replacement (ADD of another legal edge at that house)
is injected, which natively Mothballs the old driveway and breaks the connection. Checks that the adapter's view shows the Mothballed edge, that the S11 model agrees per step, and that the
planner reconnects the house (by re-adding the Mothballed edge or building a new route) without illegal actions.   python research/s21_native_check/code/live_mothballed_probe.py --seed-idx 1"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from native_adapter import NativePlannerPolicy, NativeStateView, ROOT  # noqa: E402
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import numpy as np  # noqa: E402
import s11_guard as G  # noqa: E402
from synthetic_rl import layout as L  # noqa: E402
from executor import make_policy  # noqa: E402
from planner import Weights  # noqa: E402

OUT = os.path.join(ROOT, "research", "s21_native_check")
SEEDS = json.load(open(os.path.join(ROOT, "research", "s9_native_transfer", "seeds.json")))["seeds"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-idx", type=int, default=1)
    ap.add_argument("--decisions", type=int, default=120)
    a = ap.parse_args()
    import mini_motorways_env as M
    from experiments import gamectl
    seed = SEEDS[a.seed_idx]
    pol = NativePlannerPolicy(make_policy(Weights()))
    res = {"seed": seed, "injected": None, "events": [], "edge_model_mismatch": 0, "connectivity_mismatch": 0, "illegal": 0, "reconnected_at": None, "reconnect_mode": None}
    t0 = time.time()
    with gamectl.GameInstance("s21_moth") as g:
        env = M.MiniMotorwaysEnv(max_decisions=a.decisions, decision_delta=1.0, request_path=os.path.join(g.ipc_dir, "request.json"), response_path=os.path.join(g.ipc_dir, "response.json"), start_mode="NORMAL_START")
        env.reset(seed=seed)
        pol.reset()
        raw = env.last_raw_observation
        done = False
        prev_moth = set()
        while not done:
            mask = env.action_masks().copy()
            view = NativeStateView(raw, mask, pol.t, pol.spawn_t)
            ctx = G.Ctx(raw)
            func = G.edges_from_raw(raw)
            conn = G.connected_house_set(ctx, func)
            res["edge_model_mismatch"] += int({e for e, s in view.edges.items() if s == "A"} != func or set(view.edges) != G.edges_present(raw))
            res["connectivity_mismatch"] += int(view.connected_houses() != conn)
            moth = {e for e, s in view.edges.items() if s == "M"}
            if moth != prev_moth:
                res["events"].append({"t": pol.t, "moth_added": sorted(moth - prev_moth), "moth_removed": sorted(prev_moth - moth), "connected": sorted(conn), "plan": pol.ex.plan.describe() if pol.ex.plan else None})
            prev_moth = moth
            act = None
            if res["injected"] is None and conn and pol.t >= 10:
                h = sorted(conn)[0]
                drive = [e for e in G.edges_at(func, h)]
                alts = [L.action_index("ADD", *e) for e in G.house_edges(h)] if hasattr(G, "house_edges") else []
                if not alts:
                    hx, hy = h
                    alts = [L.action_index("ADD", *e) for e in ((hx, hy, "E"), (hx, hy, "S"), (hx - 1, hy, "E"), (hx, hy + 1, "S"))]
                legal_alts = [x for x in alts if mask[x] and L.decode_action(x)[1:] not in [tuple(d) for d in drive]]
                if legal_alts:
                    act = legal_alts[0]
                    res["injected"] = {"t": pol.t, "house": list(h), "old_driveway": [list(d) for d in drive], "action": L.decode_action(act)}
                    pol.t += 1                                     # keep the adapter's clock aligned with the env
            if act is None:
                act = pol.act(raw, mask)
            res["illegal"] += int(not mask[act])
            _, r, te, tr, inf = env.step(act)
            raw = env.last_raw_observation
            done = te or tr
            if res["injected"] and res["reconnected_at"] is None and pol.t > res["injected"]["t"] + 1:
                ctx2 = G.Ctx(raw)
                if tuple(res["injected"]["house"]) in G.connected_house_set(ctx2, G.edges_from_raw(raw)):
                    res["reconnected_at"] = pol.t
                    old = {tuple(d) for d in res["injected"]["old_driveway"]}
                    res["reconnect_mode"] = "old_driveway_reactivated" if old & G.edges_from_raw(raw) else "new_route"
        res["executor_stats"] = dict(pol.ex.stats)
        res["forced_actions"] = pol.forced_actions
        env.close()
    res["wall_s"] = round(time.time() - t0, 1)
    json.dump(res, open(os.path.join(OUT, "live_mothballed_probe.json"), "w"), indent=1, default=str)
    print(json.dumps(res, default=str)[:2500])


if __name__ == "__main__":
    main()
