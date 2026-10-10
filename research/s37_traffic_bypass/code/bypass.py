"""S37 traffic-aware bypass, native causal test by paired deterministic replay (postfix_v8, planner B+ = Weights(serve_unreached_dests=True)).
  replay : B+ plays seed to the end; every decision logs, for colour --group: destination pins (totalDemand, overcrowding), its houses, each house's driving distance over the
           ACTIVE network to the nearest road tile adjacent to one of its destinations (BFS; the S25-validated static shortest-path proxy for the native router), the shortest LEGAL
           buildable path (Dijkstra, pure length, native mask) with its new-tile cost, the junction tiles (degree >= 3) on the current driving path and their visibleVehicleCount,
           and how many other colours' houses route through those junction tiles. Prints candidate intervention decisions (detour ratio >= 2, shortcut affordable, destroys nothing).
  bypass : B+ plays to --t0, then the shortcut edges computed AT t0 for the worst-detour house of --group are injected one ADD per decision (only if legal under the native mask
           at that decision and affordable), then B+ continues to the end. Same logging. The two runs share seed, stepping and upgrade handling; they diverge only through the
           injected roads (and whatever spawn/PRNG re-alignment road occupancy causes, S27) -- that divergence is reported (house counts / spawn times), not assumed away.
   python research/s37_traffic_bypass/code/bypass.py --mode replay --seed 1007922 --group 3
   python research/s37_traffic_bypass/code/bypass.py --mode bypass --seed 1007922 --group 3 --t0 330"""
import argparse
import gzip
import hashlib
import json
import os
import sys
import time
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
from world_model import WorldModel, canon, neighbours, ends  # noqa: E402
from synthetic_rl import layout as L  # noqa: E402

DLL = r"C:\Program Files (x86)\Steam\steamapps\common\Mini Motorways\Bootstrap\MiniMotorwaysBootstrap.dll"
V8 = "2aa42367cbb14babce31a6063038f50be9b3535bd7bc23d8dbbdbb65533aa00a"
OUT = os.path.join(ROOT, "research", "s37_traffic_bypass", "runs")


def adj_road_tiles(wm, dests):
    out = set()
    for d in dests:
        for p in d["tiles"] + d["carparks"]:
            for n in neighbours(p):
                if n not in wm.blocked and n not in wm.house_set and n in wm.uf.p:
                    out.add(n)
    return out


def bfs_path(wm, start, goals):
    prev = {start: None}
    q = deque([start])
    while q:
        u = q.popleft()
        if u in goals:
            path = [u]
            while prev[path[-1]] is not None:
                path.append(prev[path[-1]])
            return path[::-1]
        for v in neighbours(u):
            if wm.edges.get(canon(u, v)) == "A" and v not in prev and (v not in wm.house_set):
                prev[v] = u
                q.append(v)
    return None


def shortcut(wm, house, goals):
    C.ROUTE_LAMBDA = 0.0
    path = C.dijkstra_route(wm, house, goals, {}, set())
    C.ROUTE_LAMBDA = None
    if not path:
        return None
    adds = [canon(u, v) for u, v in zip(path[:-1], path[1:]) if wm.edges.get(canon(u, v)) != "A"]
    cost = sum(0 if wm.is_house_edge(e) else 1 for e in adds)
    return {"len": len(path) - 1, "adds": adds, "cost": cost, "breaks": len(wm.lost_connections(adds, []))}


def analyse(wm, raw, group):
    tiles = {(t["coord"][0], t["coord"][1]): t for t in raw["tiles"]}
    dests = [d for d in wm.dests if d["group"] == group]
    goals = adj_road_tiles(wm, dests)
    degree = {}
    for e in wm.active:
        for p in ends(e):
            degree[p] = degree.get(p, 0) + 1
    rec = {"dest_demand": sum(d["demand"] for d in dests), "dest_over": any(d["demand"] >= 7 for d in dests), "n_dests": len(dests), "houses": []}
    all_paths_tiles = {}
    for h, v in wm.houses.items():
        if v["group"] == group or not goals:
            continue
        og = adj_road_tiles(wm, [d for d in wm.dests if d["group"] == v["group"]])
        pth = bfs_path(wm, h, og) if og else None
        if pth:
            for p in pth:
                all_paths_tiles.setdefault(p, set()).add(v["group"])
    for h, v in wm.houses.items():
        if v["group"] != group:
            continue
        pth = bfs_path(wm, h, goals) if goals else None
        sc = shortcut(wm, h, goals) if goals else None
        junctions = [p for p in (pth or []) if degree.get(p, 0) >= 3 and p not in wm.house_set]
        rec["houses"].append({"house": list(h), "connected": h in wm.connected, "drive": (len(pth) - 1) if pth else None, "shortest": sc["len"] if sc else None, "cost": sc["cost"] if sc else None, "breaks": sc["breaks"] if sc else None,
                              "ratio": round((len(pth) - 1) / sc["len"], 2) if (pth and sc and sc["len"]) else None, "junctions": len(junctions),
                              "junction_vehicles": sum((tiles.get(p) or {}).get("visibleVehicleCount", 0) for p in junctions), "junction_other_colours": len({g for p in junctions for g in all_paths_tiles.get(p, set())} - {group}),
                              "path_vehicles": sum((tiles.get(p) or {}).get("visibleVehicleCount", 0) for p in (pth or []))})
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["replay", "bypass"])
    ap.add_argument("--seed", type=int, default=1007922)
    ap.add_argument("--group", type=int, default=3)
    ap.add_argument("--t0", type=int, default=None)
    ap.add_argument("--horizon", type=int, default=1500)
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--policy", default="bplus", choices=["bplus", "bypass"], help="bypass = B+ with Weights(bypass=True): the planner itself generates and builds shortcut candidates")
    ap.add_argument("--ratio", type=float, default=2.0)
    ap.add_argument("--max-cost", type=int, default=10)
    a = ap.parse_args()
    assert hashlib.sha256(open(DLL, "rb").read()).hexdigest() == V8
    import mini_motorways_env as M
    from experiments import gamectl
    os.makedirs(OUT, exist_ok=True)
    tag = f"{a.mode}_{a.policy}_{a.seed}_g{a.group}" + (f"_t{a.t0}" if a.t0 is not None else "")
    traj, inject, injected = [], [], []
    bypass_plans, usage = [], []          # planner-generated bypasses (policy=bypass): plan records and, after completion, vehicles on the new tiles vs the old driving path
    t0w = time.time()
    with gamectl.GameInstance(f"s37_{tag}"[:40]) as g:
        env = M.MiniMotorwaysEnv(max_decisions=a.horizon, decision_delta=1.0, request_path=os.path.join(g.ipc_dir, "request.json"), response_path=os.path.join(g.ipc_dir, "response.json"), start_mode="NORMAL_START")
        pol = NativePlannerPolicy(make_policy(Weights(serve_unreached_dests=True, bypass=(a.policy == "bypass"), bypass_ratio=a.ratio, bypass_max_cost=a.max_cost)))
        env.reset(seed=a.seed)
        pol.reset()
        raw = env.last_raw_observation
        score, t = 0.0, 0
        te = tr = False
        while not (te or tr):
            mask = env.action_masks().copy()
            view = NativeStateView(raw, mask, pol.t, pol.spawn_t)
            wm = WorldModel(view)
            if a.mode == "bypass" and t == a.t0:
                dests = [d for d in wm.dests if d["group"] == a.group]
                goals = adj_road_tiles(wm, dests)
                worst, best = None, None
                for h, v in wm.houses.items():
                    if v["group"] != a.group:
                        continue
                    pth = bfs_path(wm, h, goals)
                    sc = shortcut(wm, h, goals)
                    if pth and sc and sc["cost"] > 0 and sc["breaks"] == 0:
                        r = (len(pth) - 1) / max(1, sc["len"])
                        if worst is None or r > worst:
                            worst, best = r, (h, sc)
                assert best is not None, "no shortcut candidate at t0"
                inject = list(best[1]["adds"])
                print(f"t0={t}: bypass for house {best[0]} ratio {worst:.2f} len {best[1]['len']} cost {best[1]['cost']} adds {inject} (concrete {wm.concrete})", flush=True)
            if t % a.every == 0 or (a.mode == "bypass" and inject):
                rec = analyse(wm, raw, a.group)
                rec.update({"t": t, "score": score, "concrete": wm.concrete, "houses_total": len(wm.houses), "connected_total": len(wm.connected), "active_edges": len(wm.active), "unreached": len(wm.unreached_dests())})
                traj.append(rec)
            act = None
            if inject:
                e = inject[0]
                idx = L.action_index("ADD", *e)
                if wm.edges.get(e) == "A":
                    inject.pop(0)
                elif mask[idx] and (wm.concrete > 0 or wm.is_house_edge(e)):
                    if wm.lost_connections([e], []):
                        print(f"t={t}: injection of {e} would destroy a connection -> skipped", flush=True)
                        inject.pop(0)
                    else:
                        act = idx
                        injected.append({"t": t, "edge": e})
                        inject.pop(0)
                        pol.t += 1                                   # keep the adapter clock aligned (planner bypassed this decision)
                else:
                    print(f"t={t}: injection of {e} not legal/affordable now (mask {bool(mask[idx])}, concrete {wm.concrete}) -> skipped", flush=True)
                    inject.pop(0)
            if act is None:
                act = pol.act(raw, mask)
                if not mask[act]:
                    act = int(np.flatnonzero(mask)[0])
                pl = pol.ex.plan
                if pl is not None and pl.kind == "CONNECT" and pl.targets.get("bypass") and (not bypass_plans or bypass_plans[-1]["adds"] != [list(e) for e in pl.adds]):
                    h = tuple(pl.targets["house"])
                    goals_h = adj_road_tiles(wm, [d for d in wm.dests if d["group"] == wm.houses[h]["group"]])
                    old_path = bfs_path(wm, h, goals_h) or []
                    bypass_plans.append({"t": t, "house": list(h), "group": wm.houses[h]["group"], "drive": pl.targets["drive"], "shortest": pl.targets["shortest"], "cost": pl.concrete_cost, "adds": [list(e) for e in pl.adds],
                                         "new_tiles": sorted({p for e in pl.adds for p in ends(e) if p not in wm.house_set and p not in {q for f in wm.active for q in ends(f)}}), "old_path": [list(p) for p in old_path], "score_terms": pl.score_terms, "concrete": wm.concrete})
                    print(f"t={t}: BYPASS plan house {h} drive {pl.targets['drive']} -> {pl.targets['shortest']} cost {pl.concrete_cost} score {pl.score:.1f} new tiles {bypass_plans[-1]['new_tiles']}", flush=True)
            if bypass_plans:
                tiles_now = {(tt["coord"][0], tt["coord"][1]): tt for tt in raw["tiles"]}
                bp = bypass_plans[-1]
                built = all(wm.edges.get(tuple(e)) == "A" for e in bp["adds"])
                if built and (t % a.every == 0):
                    h = tuple(bp["house"])
                    goals_h = adj_road_tiles(wm, [d for d in wm.dests if d["group"] == bp["group"]])
                    pth = bfs_path(wm, h, goals_h) or []
                    newset = {tuple(p) for p in bp["new_tiles"]}
                    usage.append({"t": t, "drive_now": len(pth) - 1 if pth else None, "uses_new_tiles": sum(1 for p in pth if tuple(p) in newset), "path": [list(p) for p in pth], "vehicles_new_tiles": sum((tiles_now.get(tuple(p)) or {}).get("visibleVehicleCount", 0) for p in newset),
                                  "vehicles_old_path": sum((tiles_now.get(tuple(p)) or {}).get("visibleVehicleCount", 0) for p in bp["old_path"] if tuple(p) not in newset)})
            _, r, te, tr, info = env.step(act)
            raw = env.last_raw_observation
            score += float(info["rewardComponents"].get("scoreDelta", 0.0))
            t += 1
        final = {"t": t, "score": score, "terminated": bool(te), "concrete_end": raw["globals"]["concreteAvailable"], "concrete_total_end": raw["globals"]["concreteTotal"], "houses_end": len(wm.houses), "injected": injected,
                 "dest_state_end": [{"coord": tt["coord"], "group": tt.get("groupIndex"), "demand": tt["destination"].get("totalDemand", 0), "over": bool(tt["destination"].get("isOvercrowding"))} for tt in raw["tiles"] if tt.get("contentType") == "Destination" and tt.get("destination")]}
    out = {"tag": tag, "seed": a.seed, "group": a.group, "t0": a.t0, "policy": a.policy, "final": final, "traj": traj, "bypass_plans": bypass_plans, "usage": usage}
    if bypass_plans:
        print("bypass plans:", json.dumps(bypass_plans, default=str)[:1500])
        print("usage after completion (t, drive_now, path uses new tiles, vehicles on new tiles, vehicles on old path):", [(u["t"], u["drive_now"], u["uses_new_tiles"], u["vehicles_new_tiles"], u["vehicles_old_path"]) for u in usage[:40]])
    with gzip.open(os.path.join(OUT, tag + ".json.gz"), "wt") as f:
        json.dump(out, f, default=str)
    print(f"[{tag}] t {t} score {score:.0f} term {te} concrete {final['concrete_end']}/{final['concrete_total_end']} injected {len(injected)} ({time.time() - t0w:.0f}s)")
    if a.mode == "replay":
        cands = [(r["t"], max((h["ratio"] or 0) for h in r["houses"]) if r["houses"] else 0, min((h["cost"] for h in r["houses"] if h["cost"]), default=None), r["concrete"], r["dest_demand"]) for r in traj if r["houses"]]
        firsts = [c for c in cands if c[1] >= 2.0 and c[2] is not None and c[2] <= c[3]]
        print("candidate intervention decisions (t, worst ratio, min cost, concrete, group demand):", firsts[:5], "... first overcrowding t:", next((r["t"] for r in traj if r["dest_over"]), None))


if __name__ == "__main__":
    main()
