"""Route-geometry diagnostic (read-only): reproduce planner B headless on one seed to decision --at and, for every connected house, compare
  existing  = driving distance over the ACTIVE road network from the house to the nearest road tile adjacent to a destination of its colour (BFS on active edges)
  shortest  = shortest buildable path length allowing new roads (Dijkstra with ROUTE_LAMBDA = 0, i.e. pure length; native mask = buildability), its new-tile cost,
              and whether building it would break any functional connection.
Also reports, for the planner's own objective (lexicographic concrete-first) vs lambda = 4, which route it would pick for each disconnected house / unreached destination.
   python research/s35_strategic_control/code/geom.py --seed 1007922 --at 327"""
import argparse
import json
import os
import sys
from collections import deque

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
for p in (HERE, ROOT, os.path.join(ROOT, "python"), os.path.join(ROOT, "experiments"), os.path.join(ROOT, "research", "s21_native_check", "code")):
    sys.path.insert(0, p)
import numpy as np  # noqa: E402
from native_adapter import NativePlannerPolicy, NativeStateView  # noqa: E402
from executor import make_policy  # noqa: E402
from planner import Weights  # noqa: E402
import candidates as C  # noqa: E402
from world_model import WorldModel, canon, neighbours  # noqa: E402


def adj_road_tiles_of_dest(wm, d):
    out = set()
    for p in d["tiles"] + d["carparks"]:
        for n in neighbours(p):
            if n not in wm.blocked and n not in wm.house_set and n in wm.uf.p:
                out.add(n)
    return out


def bfs_active(wm, start, goals):
    """driving distance over active edges from a house tile to any goal tile (house edges included)."""
    seen = {start: 0}
    q = deque([start])
    while q:
        u = q.popleft()
        if u in goals:
            return seen[u]
        for v in neighbours(u):
            e = canon(u, v)
            if wm.edges.get(e) == "A" and v not in seen and (v not in wm.house_set or v == start):
                seen[v] = seen[u] + 1
                q.append(v)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1007922)
    ap.add_argument("--at", type=int, default=327)
    a = ap.parse_args()
    import mini_motorways_env as M
    from experiments import gamectl
    with gamectl.GameInstance("s35_geom") as g:
        env = M.MiniMotorwaysEnv(max_decisions=a.at + 2, decision_delta=1.0, request_path=os.path.join(g.ipc_dir, "request.json"), response_path=os.path.join(g.ipc_dir, "response.json"), start_mode="NORMAL_START")
        pol = NativePlannerPolicy(make_policy(Weights()))
        env.reset(seed=a.seed)
        pol.reset()
        raw = env.last_raw_observation
        t = 0
        while t < a.at:
            mask = env.action_masks().copy()
            act = pol.act(raw, mask)
            if not mask[act]:
                act = int(np.flatnonzero(mask)[0])
            _, r, te, tr, info = env.step(act)
            raw = env.last_raw_observation
            t += 1
            if te or tr:
                break
        mask = env.action_masks().copy()
        view = NativeStateView(raw, mask, pol.t, pol.spawn_t)
        wm = WorldModel(view)
    rows = []
    for h, v in sorted(wm.houses.items()):
        g_ = v["group"]
        dests = [d for d in wm.dests if d["group"] == g_]
        goals = set().union(*[adj_road_tiles_of_dest(wm, d) for d in dests]) if dests else set()
        existing = bfs_active(wm, h, goals) if goals else None
        C.ROUTE_LAMBDA = 0.0
        path = C.dijkstra_route(wm, h, goals, {}, set()) if goals else None
        C.ROUTE_LAMBDA = None
        if path:
            adds = [canon(u, w_) for u, w_ in zip(path[:-1], path[1:]) if wm.edges.get(canon(u, w_)) != "A"]
            cost = sum(0 if wm.is_house_edge(e) else 1 for e in adds)
            lost = sorted(wm.lost_connections(adds, []))
        rows.append({"house": list(h), "group": g_, "connected": h in wm.connected, "existing_drive": existing, "shortest_len": (len(path) - 1) if path else None, "shortcut_new_tiles": cost if path else None, "shortcut_breaks": len(lost) if path else None, "detour_ratio": (round(existing / (len(path) - 1), 2) if (path and existing) else None)})
    unreached = [{"dest": d["tiles"][0], "group": d["group"], "demand": d["demand"]} for d in wm.unreached_dests()]
    out = {"seed": a.seed, "at": a.at, "concrete": wm.concrete, "bridges": raw["globals"].get("bridgesAvailable"), "houses": rows, "unreached_dests": unreached, "active_edges": len(wm.active)}
    json.dump(out, open(os.path.join(HERE, "..", f"geom_{a.seed}_{a.at}.json"), "w"), indent=1, default=str)
    print(f"t={a.at} concrete {wm.concrete} bridges {out['bridges']} active edges {len(wm.active)} unreached dests {unreached}")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
