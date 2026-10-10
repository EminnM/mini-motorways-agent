"""Focused diagnostic (read-only): reproduce planner B on one seed headless and dump, at a chosen decision, what the planner sees and decides:
destinations (group, demand, overcrowding, component, whether any house of that colour can reach it), houses (group, connected), every candidate with score terms / feasibility /
rejection reason, the executor's plan and stats, globals (concrete, bridges...), and a mask-level reachability probe from each house of the destination's colour to the destination.
   python research/s35_strategic_control/code/diag_446.py --seed 1007922 --at 446 [--window 20]"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
sys.path.insert(0, os.path.join(ROOT, "research", "s21_native_check", "code"))
import numpy as np  # noqa: E402
from native_adapter import NativePlannerPolicy, NativeStateView  # noqa: E402
from executor import make_policy  # noqa: E402
from planner import Weights, score_all  # noqa: E402
from candidates import generate, dijkstra_route  # noqa: E402
from world_model import WorldModel, ends  # noqa: E402
from synthetic_rl import layout as L  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1007922)
    ap.add_argument("--at", type=int, default=446)
    ap.add_argument("--window", type=int, default=20)
    a = ap.parse_args()
    import mini_motorways_env as M
    from experiments import gamectl
    out = {"seed": a.seed, "at": a.at, "trace": []}
    with gamectl.GameInstance("s35_diag") as g:
        env = M.MiniMotorwaysEnv(max_decisions=a.at + 5, decision_delta=1.0, request_path=os.path.join(g.ipc_dir, "request.json"), response_path=os.path.join(g.ipc_dir, "response.json"), start_mode="NORMAL_START")
        pol = NativePlannerPolicy(make_policy(Weights()))
        env.reset(seed=a.seed)
        pol.reset()
        raw = env.last_raw_observation
        t = 0
        score = 0.0
        while True:
            mask = env.action_masks().copy()
            if t >= a.at - a.window:
                view = NativeStateView(raw, mask, pol.t, pol.spawn_t)
                wm = WorldModel(view)
                g_ = raw["globals"]
                rec = {"t": t, "score": score, "concrete": wm.concrete, "houses": len(wm.houses), "connected": len(wm.connected), "plan": pol.ex.plan.describe() if pol.ex.plan else None,
                       "dests": [{"tiles": d["tiles"][:1], "group": d["group"], "demand": d["demand"], "over": d["demand"] >= 7, "houses_of_group_in_component": sum(1 for h, v in wm.houses.items() if v["group"] == d["group"] and wm.component(h) == wm.component(d["tiles"][0]))} for d in wm.dests]}
                out["trace"].append(rec)
                if t == a.at:
                    cands = generate(wm, 2, frozenset(), frozenset(), include_motorway=False)
                    score_all(wm, cands, Weights())
                    out["globals"] = {k: v for k, v in g_.items() if not isinstance(v, (list, dict))}
                    out["globals_lists"] = {k: v for k, v in g_.items() if isinstance(v, (list, dict))}
                    out["houses"] = [{"tile": list(h), "group": v["group"], "connected": h in wm.connected, "component": str(wm.component(h))} for h, v in sorted(wm.houses.items())]
                    out["dests_full"] = [{"tiles": d["tiles"], "group": d["group"], "demand": d["demand"], "over": d["demand"] >= 7, "carparks": d["carparks"], "component": str(wm.component(d["tiles"][0])), "groups_in_component": sorted(wm.dest_comp_groups.get(wm.component(d["tiles"][0]), []), key=str)} for d in wm.dests]
                    out["candidates"] = [{"kind": c.kind, "desc": c.describe(), "score": c.score, "terms": c.score_terms, "feasibility": c.feasibility, "confidence": c.confidence, "would_destroy": c.would_destroy_functional_connection, "prereq": c.prerequisites} for c in cands]
                    out["executor"] = {"stats": {k: v for k, v in pol.ex.stats.items()}, "plan": pol.ex.plan.describe() if pol.ex.plan else None, "forced_actions": pol.forced_actions[-5:]}
                    # reachability probe: for every destination, can a house of its colour reach ANY tile adjacent to the destination block through buildable edges (mask-authoritative)?
                    probes = []
                    for d in wm.dests:
                        adj = {n for p in d["tiles"] for n in [(p[0] + 1, p[1]), (p[0] - 1, p[1]), (p[0], p[1] + 1), (p[0], p[1] - 1)] if n not in wm.blocked and n not in wm.house_set}
                        for h, v in wm.houses.items():
                            if v["group"] != d["group"]:
                                continue
                            path = dijkstra_route(wm, h, adj, {}, set())
                            probes.append({"dest": d["tiles"][0], "group": d["group"], "house": list(h), "house_connected": h in wm.connected, "path_len": len(path) - 1 if path else None, "new_edges": (sum(1 for u, v_ in zip(path[:-1], path[1:]) if wm.edges.get(__import__('world_model').canon(u, v_)) != 'A') if path else None)})
                    out["reach_probes"] = probes
                    # water around each destination and legal ADDs touching water tiles
                    water = raw_water = [tuple(tt["coord"]) for tt in raw["tiles"] if tt.get("overWater")]
                    out["n_water_tiles"] = len(water)
                    legal_adds = [L.decode_action(int(i)) for i in np.flatnonzero(mask) if 0 < i < L.UPGRADE_BASE and L.decode_action(int(i))[0] == "ADD"]
                    ws = set(water)
                    out["legal_adds_touching_water"] = sum(1 for k in legal_adds if (k[1], k[2]) in ws or ((k[1] + 1, k[2]) if k[3] == "E" else (k[1], k[2] - 1)) in ws)
                    out["legal_adds_total"] = len(legal_adds)
                    out["tiles_sample_keys"] = sorted(raw["tiles"][0].keys())
            act = pol.act(raw, mask)
            if not mask[act]:
                act = int(np.flatnonzero(mask)[0])
            if t >= a.at - a.window:
                out["trace"][-1]["action"] = L.decode_action(act)
            _, r, te, tr, info = env.step(act)
            raw = env.last_raw_observation
            score += float(info["rewardComponents"].get("scoreDelta", 0.0))
            t += 1
            if te or tr or t > a.at + 3:
                break
    json.dump(out, open(os.path.join(HERE, "..", f"diag_{a.seed}_{a.at}.json"), "w"), indent=1, default=str)
    for r in out["trace"]:
        print(r["t"], "score", r["score"], "concrete", r["concrete"], "conn", f"{r['connected']}/{r['houses']}", "plan", r["plan"], "action", r.get("action"), "dests", [(d["group"], d["demand"], d["over"], d["houses_of_group_in_component"]) for d in r["dests"]])
    print("GLOBALS", out.get("globals"))
    print("DESTS", json.dumps(out.get("dests_full"), default=str))
    print("HOUSES", out.get("houses"))
    print("CANDIDATES", json.dumps(out.get("candidates"), default=str)[:3000])
    print("EXEC", out.get("executor"))
    print("PROBES", json.dumps(out.get("reach_probes"), default=str))
    print("WATER", out.get("n_water_tiles"), "legal adds touching water", out.get("legal_adds_touching_water"), "/", out.get("legal_adds_total"))


if __name__ == "__main__":
    main()
