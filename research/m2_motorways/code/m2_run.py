"""M2 Stage B runner: planner B + preregistered rule M (research/m2_motorways/PREREGISTERED_RULE.md) vs planner B without placement; both arms prefer the Motorway upgrade when offered.
   python research/m2_motorways/code/m2_run.py --arm ruleM|noMW --idx 0,...,11"""
import argparse
import gzip
import hashlib
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "s21_native_check", "code"))
from motorway_env import MotorwayEnv, ROOT  # noqa: E402
from native_adapter import NativePlannerPolicy  # noqa: E402
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import numpy as np  # noqa: E402
import s11_guard as G  # noqa: E402
from synthetic_rl import layout as L  # noqa: E402
from executor import make_policy  # noqa: E402
from planner import Weights  # noqa: E402

OUT = os.path.join(ROOT, "research", "m2_motorways")
SEEDS = json.load(open(os.path.join(ROOT, "research", "s9_native_transfer", "seeds.json")))["seeds"]
DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))
RAMP_DIRS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
MIN_SAVING, MIN_CHEB = 6, 4


def obs_hash(raw):
    g = {k: v for k, v in raw["globals"].items() if k not in ("pendingUpgradeTypes", "motorways", "motorwaysAvailableOrDraft")}
    return hashlib.sha256(json.dumps(raw["tiles"], sort_keys=True).encode() + json.dumps(g, sort_keys=True).encode()).hexdigest()[:16]


def graph(raw):
    ctx = G.Ctx(raw)
    func = G.edges_from_raw(raw)
    adj = {}
    for e in func:
        a, b = G.ends(e)
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    carparks = {k for k, c in ctx.content.items() if c == "Carpark"}
    destg = {k: ctx.group.get(k) for k, c in ctx.content.items() if c == "Destination"}
    stubs = {}
    for e in func:
        a, b = G.ends(e)
        for cp, o in ((a, b), (b, a)):
            if cp in carparks:
                for dx, dy in DIRS:
                    g = destg.get((cp[0] + dx, cp[1] + dy))
                    if g is not None:
                        stubs.setdefault(g, set()).add(o)
    return ctx, func, adj, stubs


def bfs(adj, src):
    dist, prev = {src: 0}, {src: None}
    dq = [src]
    while dq:
        u = dq.pop(0)
        for v in adj.get(u, ()):
            if v not in dist:
                dist[v] = dist[u] + 1
                prev[v] = u
                dq.append(v)
    return dist, prev


def evaluate_rule(raw):
    """returns (best placement dict or None, summary of the opportunity evaluation)"""
    ctx, func, adj, stubs = graph(raw)
    conn = G.connected_house_set(ctx, func)
    routes = {}
    for h in conn:
        g = ctx.group.get(h)
        goals = stubs.get(g, set())
        dist, prev = bfs(adj, h)
        reach = [t for t in goals if t in dist]
        if not reach:
            continue
        tgt = min(reach, key=lambda t: dist[t])
        path = [tgt]
        while prev[path[-1]] is not None:
            path.append(prev[path[-1]])
        routes[h] = (path[::-1], dist[tgt], g)
    if not routes:
        return None, {"connected_with_route": 0}
    stub_near = {}
    for g, goals in stubs.items():
        for s in goals:
            d, _ = bfs(adj, s)
            for t, dd in d.items():
                if dd <= 2 and ctx.content.get(t) is None:
                    stub_near.setdefault(g, {})[t] = min(dd, stub_near.get(g, {}).get(t, 99))
    best = None
    n_pairs = 0
    for h, (path, T_road, g) in routes.items():
        road_tiles = [t for t in path if ctx.content.get(t) is None]
        for ai, a in enumerate(road_tiles):
            hops_h_a = path.index(a)
            if T_road - hops_h_a < 2:
                continue
            for b, hops_b_stub in stub_near.get(g, {}).items():
                cheb = max(abs(a[0] - b[0]), abs(a[1] - b[1]))
                if cheb < MIN_CHEB or a == b:
                    continue
                n_pairs += 1
                saving = 0
                for h2, (p2, T2, g2) in routes.items():
                    if g2 == g and a in p2:
                        t_mw = p2.index(a) + 2 + math.ceil(cheb / 3) + hops_b_stub
                        saving += max(0, T2 - t_mw)
                key = (saving, -a[0], -a[1], -b[0], -b[1])
                if saving >= MIN_SAVING and (best is None or key > best[0]):
                    best = (key, {"a": list(a), "b": list(b), "cheb": cheb, "saving": saving, "house": list(h), "T_road": T_road})
    return (best[1] if best else None), {"connected_with_route": len(routes), "pairs_evaluated": n_pairs, "best_saving": best[1]["saving"] if best else 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["ruleM", "noMW"])
    ap.add_argument("--idx", default=",".join(str(i) for i in range(12)))
    ap.add_argument("--deadline-min", type=float, default=14.0)
    a = ap.parse_args()
    from experiments import gamectl
    os.makedirs(os.path.join(OUT, "raw"), exist_ok=True)
    t0 = time.time()
    with gamectl.GameInstance(f"m2_{a.arm}") as g:
        env = MotorwayEnv(max_decisions=600, decision_delta=1.0, request_path=os.path.join(g.ipc_dir, "request.json"), response_path=os.path.join(g.ipc_dir, "response.json"), start_mode="NORMAL_START")
        for k in [int(x) for x in a.idx.split(",")]:
            seed = SEEDS[k]
            pol = NativePlannerPolicy(make_policy(Weights()))
            env.reset(seed=seed)
            pol.reset()
            raw = env.last_raw_observation
            steps, placements, opps, choices = [], [], [], []
            te = tr = False
            i, cum = 0, 0.0
            mw_tiles = set()
            while not (te or tr) and time.time() < t0 + 60 * a.deadline_min:
                mask = env.action_masks().copy()
                gl = raw["globals"]
                ctx = G.Ctx(raw)
                conn = G.connected_house_set(ctx, G.edges_from_raw(raw))
                rec = {"i": i, "hash": obs_hash(raw), "houses": len(ctx.houses), "connected": len(conn), "concrete": gl["concreteAvailable"], "mw_avail": gl.get("motorwaysAvailableOrDraft", 0), "n_mw": len(gl.get("motorways", [])),
                       "veh_total": sum(t.get("visibleVehicleCount", 0) for t in raw["tiles"]), "veh_mw_tiles": sum(t.get("visibleVehicleCount", 0) for t in raw["tiles"] if (t["coord"][0], t["coord"][1]) in mw_tiles),
                       "new_houses": [list(h) for h in ctx.houses if h not in {tuple(x) for x in steps[-1]["houses_list"]}] if steps else [], "houses_list": [list(h) for h in ctx.houses]}
                motorway_req = None
                if not mask[0] and gl.get("pendingUpgradeChoices", 0) > 0:
                    offered = gl.get("pendingUpgradeTypes", [])
                    slot = next((j for j, o in enumerate(offered) if o["type"] == "Motorway"), 0)
                    act = L.UPGRADE_BASE + slot if mask[L.UPGRADE_BASE + slot] else int(np.flatnonzero(mask)[0])
                    pol.t += 1
                    choices.append({"i": i, "offered": [o["type"] for o in offered], "slot": slot})
                elif a.arm == "ruleM" and gl.get("motorwaysAvailableOrDraft", 0) >= 1:
                    best, summ = evaluate_rule(raw)
                    opps.append({"i": i, **summ, "placed": best is not None})
                    if best is not None:
                        req = None
                        for d1 in RAMP_DIRS:
                            for d2 in RAMP_DIRS:
                                m = {"op": "add", "x1": best["a"][0], "y1": best["a"][1], "d1": d1, "x2": best["b"][0], "y2": best["b"][1], "d2": d2}
                                if env.motorway_check(m).get("ok"):
                                    req = m
                                    break
                            if req:
                                break
                        if req:
                            motorway_req = req
                            placements.append({"i": i, **best, "request": req})
                            mw_tiles |= {tuple(best["a"]), tuple(best["b"])}
                        else:
                            opps[-1]["placed"] = False
                            opps[-1]["no_legal_ramp"] = True
                    act = pol.act(raw, mask)
                else:
                    act = pol.act(raw, mask)
                if motorway_req:
                    pol.t += 1
                    _, r, te, tr, inf = env.step_motorway(motorway_req, action=0)
                    placements[-1]["motorwayOk"] = inf["motorwayOk"]
                    placements[-1]["motorwayInfo"] = inf["motorwayInfo"]
                    act = 0
                else:
                    _, r, te, tr, inf = env.step(act)
                raw = env.last_raw_observation
                rc = inf.get("rewardComponents", {})
                cum += float(rc.get("scoreDelta", 0.0))
                rec.update({"action": int(act), "legal": bool(mask[act]), "scoreDelta": float(rc.get("scoreDelta", 0.0)), "score_cum": cum, "motorways": raw["globals"].get("motorways", [])})
                rec.pop("houses_list_prev", None)
                steps.append(rec)
                i += 1
            for s_ in steps:
                s_.pop("houses_list", None)
            out = {"arm": a.arm, "seed": seed, "steps": steps, "placements": placements, "opportunities": opps, "choices": choices, "end": {"terminated": bool(te), "truncated": bool(tr), "cut": not (te or tr), "exec_stats": dict(pol.ex.stats)}}
            with gzip.open(os.path.join(OUT, "raw", f"{a.arm}_{seed}.json.gz"), "wt") as f:
                json.dump(out, f)
            print(f"[{a.arm}] seed {seed}: steps {len(steps)} deliveries {cum} conn {steps[-1]['connected']}/{steps[-1]['houses']} choices {[(c['i'], c['offered'], c['slot']) for c in choices]} placements {[(p['i'], p['a'], p['b'], p['saving'], p.get('motorwayOk')) for p in placements]} opps {len(opps)} ({time.time() - t0:.0f}s)", flush=True)
        env.close()


if __name__ == "__main__":
    main()
