"""S35 runner (postfix_v8): one game instance, the native env, planner B's adapter wrapping a StrategicExecutor with a given rule.
  --cond plannerB | wrapper (PlannerRule: must reproduce plannerB) | default (demand+dedicated) | d1only (demand+reuse) | d2only (planner+dedicated) | random (--rollout r)
Seeds 1000003 + 7919*i (i < 10; unused before S35), horizon 1,000 decisions. Per episode: score, length, termination, concrete used, connected/houses over time, decision-point log,
active-edge snapshots at t = 100/200/300/400 (for network comparison), final destination demand state (game-over cause proxy). Checks: 0 illegal, playableAreaAtTime present.
   python research/s35_strategic_control/code/run.py --cond default [--idx 0,1,...] [--rollout 0]"""
import argparse
import gzip
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import numpy as np  # noqa: E402
from strategic import StrategicExecutor, PlannerRule, FixedRule, RandomRule  # noqa: E402
from native_adapter import NativePlannerPolicy  # noqa: E402
from executor import make_policy  # noqa: E402
from planner import Weights  # noqa: E402
from synthetic_rl import layout as L  # noqa: E402

SEEDS = [1000003 + 7919 * i for i in range(10)]
HORIZON = 1000
SNAP_T = (100, 200, 300, 400, 600)
DLL = r"C:\Program Files (x86)\Steam\steamapps\common\Mini Motorways\Bootstrap\MiniMotorwaysBootstrap.dll"
V8 = "2aa42367cbb14babce31a6063038f50be9b3535bd7bc23d8dbbdbb65533aa00a"
OUT = os.path.join(ROOT, "research", "s35_strategic_control", "eval")


def make(cond, rollout):
    if cond == "plannerB":
        return NativePlannerPolicy(make_policy(Weights()))
    if cond == "plannerBfix":                      # S35 fix: planner B + destination-targeted CONNECT candidates for unreached destinations
        return NativePlannerPolicy(make_policy(Weights(serve_unreached_dests=True)))
    if cond.startswith("plannerBfix_lam"):          # S35 geometry: + route search cost = length + lambda * new concrete
        return NativePlannerPolicy(make_policy(Weights(serve_unreached_dests=True, route_lambda=float(cond.split("lam")[1]))))
    if cond.startswith("plannerB_lam"):             # geometry only (no destination fix), for attribution
        return NativePlannerPolicy(make_policy(Weights(route_lambda=float(cond.split("lam")[1]))))
    rule = {"wrapper": lambda: PlannerRule(), "default": lambda: FixedRule("demand", "dedicated"), "d1only": lambda: FixedRule("demand", "reuse"), "d2only": lambda: FixedRule("planner", "dedicated"),
            "random": lambda: RandomRule(35000 + rollout)}[cond]()
    return NativePlannerPolicy(StrategicExecutor(rule, Weights()))


def dest_state(raw):
    out = []
    for t in raw["tiles"]:
        if t.get("contentType") == "Destination" and t.get("destination"):
            d = t["destination"]
            out.append({"coord": t["coord"], "group": t.get("groupIndex"), "demand": d.get("totalDemand", 0), "over": bool(d.get("isOvercrowding"))})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", required=True)
    ap.add_argument("--idx", default=None)
    ap.add_argument("--rollout", type=int, default=0)
    ap.add_argument("--visible", action="store_true", help="open a game window and pace decisions in real time (viewing only; output tag prefixed vis_)")
    a = ap.parse_args()
    assert hashlib.sha256(open(DLL, "rb").read()).hexdigest() == V8, "deployed DLL is not postfix_v8"
    import mini_motorways_env as M
    from experiments import gamectl
    idx = [int(x) for x in a.idx.split(",")] if a.idx else list(range(len(SEEDS)))
    os.makedirs(OUT, exist_ok=True)
    tag = a.cond if a.cond != "random" else f"random_r{a.rollout}"
    if a.visible:
        tag = "vis_" + tag
    t0 = time.time()
    with gamectl.GameInstance(f"s35_{tag}_{''.join(map(str, idx))}"[:40], headless=not a.visible) as g:
        env = M.MiniMotorwaysEnv(max_decisions=HORIZON, decision_delta=1.0, request_path=os.path.join(g.ipc_dir, "request.json"), response_path=os.path.join(g.ipc_dir, "response.json"), start_mode="NORMAL_START")
        if a.visible:
            _send = env.client.send
            env.client.send = lambda op, **kw: _send(op, **(dict(kw, realtime=True) if op == "STEP" else kw))
        for k in idx:
            seed = SEEDS[k]
            pol = make(a.cond, a.rollout)
            env.reset(seed=seed)
            pol.reset()
            raw = env.last_raw_observation
            assert raw["globals"].get("playableAreaAtTime") is not None, "environment-version mismatch"
            score = 0.0
            illegal = 0
            kinds = {"WAIT": 0, "ADD": 0, "REMOVE": 0, "UPG": 0}
            snaps, timeline = {}, []
            adds_log = []
            t = 0
            te = tr = False
            while not (te or tr):
                mask = env.action_masks().copy()
                act = pol.act(raw, mask)
                if not mask[act]:
                    illegal += 1
                    act = int(np.flatnonzero(mask)[0])
                kd = L.decode_action(act)
                kinds["UPG" if kd[0] == "UPGRADE" else kd[0]] += 1
                if kd[0] == "ADD":                                   # boundary bookkeeping: where every road goes relative to the native playable rectangle at this decision
                    ar = raw["globals"].get("playableAreaAtTime")
                    fx, fy = (kd[1] + 1, kd[2]) if kd[3] == "E" else (kd[1], kd[2] - 1)
                    margin = min(kd[1] - ar[0], ar[2] - kd[1], kd[2] - ar[1], ar[3] - kd[2], fx - ar[0], ar[2] - fx, fy - ar[1], ar[3] - fy) if ar else None
                    adds_log.append({"t": t, "edge": [kd[1], kd[2], kd[3]], "area": ar, "margin": margin})
                    if a.visible:
                        print(f"   t={t:4d} ADD ({kd[1]},{kd[2]},{kd[3]}) area x[{ar[0]},{ar[2]}] y[{ar[1]},{ar[3]}] margin {margin}{'  <-- ON BOUNDARY' if margin == 0 else ''}", flush=True)
                _, r, te, tr, info = env.step(act)
                raw = env.last_raw_observation
                score += float(info["rewardComponents"].get("scoreDelta", 0.0))
                t += 1
                view = pol.last_view
                if a.visible and t % 25 == 0:          # on-screen cross-check of the HUD road counter against the simulation's exported inventory
                    gg = raw["globals"]
                    print(f"   t={t:4d} day {gg.get('day')} week {gg.get('week')} | roads {gg['concreteAvailable']}/{gg['concreteTotal']} bridges {gg.get('bridgesAvailable')}/{gg.get('bridgesTotal')} | score {score:.0f} | unreached dests {[d['tiles'][0] for d in (pol.last_view and __import__('world_model').WorldModel(pol.last_view).unreached_dests() or [])]}", flush=True)
                if t in SNAP_T or t % 50 == 0:
                    v = view if view is not None else None
                    timeline.append({"t": t, "score": score, "concrete": int(raw["globals"]["concreteAvailable"]), "houses": len(v.houses) if v else None, "connected": len(v.connected_houses()) if v else None, "active_edges": sum(1 for s in v.edges.values() if s == "A") if v else None})
                if t in SNAP_T and view is not None:
                    snaps[str(t)] = sorted([list(e) for e, s in view.edges.items() if s == "A"])
            ex = pol.ex
            ep = {"cond": a.cond, "tag": tag, "rollout": a.rollout, "seed": seed, "env_version": "postfix_v8", "horizon": HORIZON, "t": t, "score": score, "terminated": bool(te), "illegal": illegal, "kinds": kinds,
                  "concrete_end": int(raw["globals"]["concreteAvailable"]), "concrete_total_end": int(raw["globals"]["concreteTotal"]), "houses_end": len(pol.last_view.houses), "connected_end": len(pol.last_view.connected_houses()),
                  "dest_state_end": dest_state(raw), "exec_stats": {k: v for k, v in ex.stats.items() if k != "compute_s"}, "forced_actions": len(pol.forced_actions),
                  "points": getattr(ex, "points", []), "timeline": timeline, "snapshots": snaps, "adds": adds_log,
                  "adds_on_boundary": sum(1 for x in adds_log if x["margin"] == 0), "adds_outside_area": sum(1 for x in adds_log if x["margin"] is not None and x["margin"] < 0)}
            with gzip.open(os.path.join(OUT, f"{tag}_{seed}.json.gz"), "wt") as f:
                json.dump(ep, f)
            pts = ep["points"]
            print(f"[{tag}] seed {seed}: t {t} score {score:.0f} term {te} illegal {illegal} kinds {kinds} connected {ep['connected_end']}/{ep['houses_end']} concrete {ep['concrete_end']} points {len(pts)} live_d1 {sum(p['d1_live'] for p in pts)} live_d2 {sum(p['d2_live'] for p in pts)} differs {sum(p['differs_from_planner'] for p in pts)} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
