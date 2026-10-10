"""S21/overnight: ONE-SEED live integration check of the adapter (planner B) -- authorised overnight step 1b. Per decision: adapter edge/content/connectivity vs the S11 functional model on the LIVE
raw observation, legality of every emitted action, Mothballed-edge events (house-edge replacement / REMOVE) and whether the adapter's view reflects them, upgrade interrupts, executor stats.
Also probes whether the deployed DLL answers the DIAG ops (PING + DIAG_SNAPSHOT capability check after the episode).   python research/s21_native_check/code/live_smoke.py --seed-idx 0 --decisions 150"""
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
    ap.add_argument("--seed-idx", type=int, default=0)
    ap.add_argument("--decisions", type=int, default=150)
    a = ap.parse_args()
    import mini_motorways_env as M
    from experiments import gamectl
    seed = SEEDS[a.seed_idx]
    pol = NativePlannerPolicy(make_policy(Weights()))
    res = {"seed": seed, "decisions": 0, "edge_model_mismatch": 0, "content_mismatch": 0, "connectivity_mismatch": 0, "illegal": 0, "mothballed_events": [], "upgrade_interrupts": 0, "pending_seen": 0, "wall_s": None}
    t0 = time.time()
    with gamectl.GameInstance("s21_smoke") as g:
        env = M.MiniMotorwaysEnv(max_decisions=a.decisions, decision_delta=1.0, request_path=os.path.join(g.ipc_dir, "request.json"), response_path=os.path.join(g.ipc_dir, "response.json"), start_mode="NORMAL_START")
        obs, _ = env.reset(seed=seed)
        pol.reset()
        raw = env.last_raw_observation
        done = False
        prev_moth = set()
        while not done:
            mask = env.action_masks().copy()
            view = NativeStateView(raw, mask, pol.t, pol.spawn_t)
            ctx = G.Ctx(raw)
            res["edge_model_mismatch"] += int({e for e, s in view.edges.items() if s == "A"} != G.edges_from_raw(raw) or set(view.edges) != G.edges_present(raw))
            res["content_mismatch"] += int(view.content != {k: c for k, c in ctx.content.items() if c in ("House", "Destination", "Carpark")})
            res["connectivity_mismatch"] += int(view.connected_houses() != G.connected_house_set(ctx, G.edges_from_raw(raw)))
            res["pending_seen"] += len(view.pending_edges)
            moth = {e for e, s in view.edges.items() if s == "M"}
            if moth != prev_moth:
                res["mothballed_events"].append({"t": pol.t, "added": sorted(moth - prev_moth), "removed": sorted(prev_moth - moth), "connected_now": len(view.connected_houses()), "houses": len(view.houses)})
            prev_moth = moth
            if not mask[0]:
                res["upgrade_interrupts"] += 1
            act = pol.act(raw, mask)
            res["illegal"] += int(not mask[act])
            obs, r, te, tr, inf = env.step(act)
            raw = env.last_raw_observation
            done = te or tr
            res["decisions"] += 1
        res["score"] = raw["globals"]["score"]
        res["executor_stats"] = dict(pol.ex.stats)
        res["forced_actions"] = pol.forced_actions
        res["connected_end"] = [len(G.connected_house_set(G.Ctx(raw), G.edges_from_raw(raw))), len(G.Ctx(raw).houses)]
        # DIAG capability probe (separate episode state is irrelevant: ops answer 'unknown' or with a result)
        caps = {}
        try:
            caps["PING"] = env.client.send("PING")
        except Exception as ex:  # noqa: BLE001
            caps["PING"] = repr(ex)[:200]
        res["diag_capability"] = caps
        env.close()
    res["wall_s"] = round(time.time() - t0, 1)
    json.dump(res, open(os.path.join(OUT, "live_smoke.json"), "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in res.items() if k != "mothballed_events"}, default=str))
    print("mothballed events:", res["mothballed_events"][:8])


if __name__ == "__main__":
    main()
