"""S22 native throughput probe runner (approved). Variants V0 (planner B), V0dup (identical), V1 parallel, V2 dense -- see ../PREREGISTRATION.md. Per-decision logging incl. raw-observation hashes
for the equivalence check, spawns, destinations (demand/overcrowding), vehicles, connectivity, concrete.
   python research/s22_throughput/code/throughput_probe.py --variant V2 --idx 0 --decisions 150 --smoke"""
import argparse
import gzip
import hashlib
import json
import os
import sys
import time
from collections import deque

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "s21_native_check", "code"))
from native_adapter import NativePlannerPolicy, NativeStateView, ROOT  # noqa: E402
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import numpy as np  # noqa: E402
import s11_guard as G  # noqa: E402
from synthetic_rl import layout as L  # noqa: E402
from executor import make_policy  # noqa: E402
from planner import Weights  # noqa: E402

OUT = os.path.join(ROOT, "research", "s22_throughput")
SEEDS = json.load(open(os.path.join(ROOT, "research", "s9_native_transfer", "seeds.json")))["seeds"]
T0, K = 60, 8
DIRS = {"N": (0, 1), "E": (1, 0), "S": (0, -1), "W": (-1, 0)}


def obs_hash(raw):
    return hashlib.sha256(json.dumps(raw["tiles"], sort_keys=True).encode() + json.dumps(raw["globals"], sort_keys=True).encode()).hexdigest()[:16]


def dest_info(raw):
    tiles = {(t["coord"][0], t["coord"][1]): t for t in raw["tiles"] if t["contentType"] == "Destination"}
    seen, out = set(), []
    for k in sorted(tiles):
        if k in seen:
            continue
        comp, st = [], [k]
        seen.add(k)
        g = tiles[k].get("groupIndex")
        while st:
            u = st.pop()
            comp.append(u)
            for dx, dy in DIRS.values():
                v = (u[0] + dx, u[1] + dy)
                if v in tiles and v not in seen and tiles[v].get("groupIndex") == g:
                    seen.add(v)
                    st.append(v)
        d = [tiles[c].get("destination") or {} for c in comp]
        out.append([min(comp)[0], min(comp)[1], g, max(x.get("totalDemand", 0) for x in d), int(any(x.get("isOvercrowding") for x in d)), max(float(x.get("overcrowdingTime", 0.0)) for x in d)])
    return out


# ---------------------------------------------------------------- deterministic topology edit sets (fixed before the first run)
def _adj_tiles(func):
    adj = {}
    for e in func:
        a, b = G.ends(e)
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    return adj


def _stub_tiles(raw, func):
    ctx = G.Ctx(raw)
    carparks = {k for k, c in ctx.content.items() if c == "Carpark"}
    return sorted({(b if a in carparks else a) for e in func for a, b in [G.ends(e)] if (a in carparks) != (b in carparks)})


def _legal_add(mask, e):
    return bool(mask[L.action_index("ADD", *e)])


def edits_dense(raw, mask):
    """mesh around the trunk: BFS from the stub tiles over functional roads; for each road tile in BFS order, add legal plain edges to its 4 neighbours, preferring edges that close a loop
    (other endpoint already a road tile), then edges to free tiles; stop at K. Never touches house tiles."""
    func = G.edges_from_raw(raw)
    ctx = G.Ctx(raw)
    adj = _adj_tiles(func)
    road_tiles = set(adj)
    order, seen = [], set()
    dq = deque(_stub_tiles(raw, func))
    for s in dq:
        seen.add(s)
    while dq:
        u = dq.popleft()
        order.append(u)
        for v in sorted(adj.get(u, ())):
            if v not in seen:
                seen.add(v)
                dq.append(v)
    chosen, chosen_set = [], set()
    for pref_loop in (True, False):
        for u in order:
            if u in ctx.house_set:
                continue
            for d, (dx, dy) in sorted(DIRS.items()):
                v = (u[0] + dx, u[1] + dy)
                if v in ctx.house_set or ctx.content.get(v) in ("Destination", "Carpark"):
                    continue
                e = G.canon(u, v)
                if e in func or e in chosen_set or not _legal_add(mask, e):
                    continue
                if (v in road_tiles) != pref_loop:
                    continue
                chosen.append(e)
                chosen_set.add(e)
                if len(chosen) >= K:
                    return chosen
    return chosen


def edits_parallel(raw, mask):
    """offset copy of the longest trunk: path (over functional roads) from a stub tile to the farthest connected house; interior tiles shifted by one tile to the side with more legal
    edges, plus connectors at both ends; up to K edges in path order."""
    func = G.edges_from_raw(raw)
    ctx = G.Ctx(raw)
    adj = _adj_tiles(func)
    conn = G.connected_house_set(ctx, func)
    stubs = _stub_tiles(raw, func)
    best = None
    for s in stubs:
        prev, dist = {s: None}, {s: 0}
        dq = deque([s])
        while dq:
            u = dq.popleft()
            for v in adj.get(u, ()):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    prev[v] = u
                    dq.append(v)
        for h in conn:
            if h in dist and (best is None or dist[h] > best[0]):
                path = [h]
                while prev[path[-1]] is not None:
                    path.append(prev[path[-1]])
                best = (dist[h], path[::-1])
    if best is None or best[0] < 3:
        return []
    path = best[1][:-1]                                        # exclude the house tile
    cands = []
    for side in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        edges, used = [], set()
        for u, v in zip(path[:-1], path[1:]):
            if (v[0] - u[0], v[1] - u[1]) in (side, (-side[0], -side[1])):
                continue                                       # path segment runs along the offset direction: no parallel copy there
            pu, pv = (u[0] + side[0], u[1] + side[1]), (v[0] + side[0], v[1] + side[1])
            e = G.canon(pu, pv)
            if e not in func and e not in used and _legal_add(mask, e) and pu not in ctx.house_set and pv not in ctx.house_set:
                edges.append(e)
                used.add(e)
        for u in (path[0], path[-1]):                          # connectors
            pu = (u[0] + side[0], u[1] + side[1])
            e = G.canon(u, pu)
            if e not in func and e not in used and _legal_add(mask, e) and pu not in ctx.house_set:
                edges.append(e)
                used.add(e)
        cands.append(edges)
    # execution-order fix (smoke run 523757: with K=8 the 7 line edges consumed the budget and only one connector fitted -> a spur, not a loop):
    # connectors first, then at most K-2 line edges, so both ends attach whenever the connectors are legal.
    edges = max(cands, key=len)
    conn_edges = [e for e in edges if any(u in G.ends(e) for u in (path[0], path[-1]))]
    line_edges = [e for e in edges if e not in conn_edges]
    return (conn_edges + line_edges[: K - len(conn_edges)])[:K]


def run_episode(env, variant, seed, deadline):
    pol = NativePlannerPolicy(make_policy(Weights()))
    env.reset(seed=seed)
    pol.reset()
    raw = env.last_raw_observation
    steps, edits, edit_log = [], None, []
    te = tr = False
    i, cum = 0, 0.0
    prev_houses = set()
    while not (te or tr):
        if time.time() > deadline:
            break
        mask = env.action_masks().copy()
        ctx = G.Ctx(raw)
        func = G.edges_from_raw(raw)
        conn = G.connected_house_set(ctx, func)
        rec = {"i": i, "hash": obs_hash(raw), "houses": len(ctx.houses), "connected": len(conn), "connected_houses": [list(h) for h in sorted(conn)], "n_func": len(func), "concrete": raw["globals"]["concreteAvailable"],
               "vehicles": sum(t.get("visibleVehicleCount", 0) for t in raw["tiles"]), "dests": dest_info(raw), "new_struct": [[k[0], k[1], c, ctx.group.get(k)] for k, c in ctx.content.items() if c in ("House", "Destination") and k not in prev_houses]}
        prev_houses = {k for k, c in ctx.content.items() if c in ("House", "Destination")}
        a = None
        if variant in ("V1", "V2") and i >= T0:
            if edits is None:
                edits = list(edits_parallel(raw, mask) if variant == "V1" else edits_dense(raw, mask))
                rec["edit_set"] = [list(e) for e in edits]
            while edits:
                e = edits.pop(0)
                idx = L.action_index("ADD", *e)
                if mask[idx]:
                    a = idx
                    edit_log.append({"i": i, "edge": list(e), "legal": True})
                    pol.t += 1                                 # adapter clock stays aligned
                    break
                edit_log.append({"i": i, "edge": list(e), "legal": False})
        if a is None:
            a = pol.act(raw, mask)
        rec["action"] = int(a)
        rec["kind"] = L.decode_action(int(a))[0]
        rec["legal"] = bool(mask[a])
        _, r, te, tr, inf = env.step(a)
        raw = env.last_raw_observation
        rc = inf.get("rewardComponents", {})
        rec["scoreDelta"] = float(rc.get("scoreDelta", 0.0))
        cum += rec["scoreDelta"]
        rec["score_cum"] = cum
        ctx_a = G.Ctx(raw)
        conn_a = G.connected_house_set(ctx_a, G.edges_from_raw(raw))
        rec["lost"] = [list(h) for h in sorted(conn) if h in ctx_a.house_set and h not in conn_a]
        steps.append(rec)
        i += 1
    return {"variant": variant, "seed": seed, "T0": T0, "K": K, "steps": steps, "edits": edit_log, "end": {"terminated": bool(te), "truncated": bool(tr), "cut": not (te or tr), "exec_stats": dict(pol.ex.stats), "forced": pol.forced_actions}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["V0", "V0dup", "V1", "V2"])
    ap.add_argument("--idx", default="0")
    ap.add_argument("--decisions", type=int, default=600)
    ap.add_argument("--deadline-min", type=float, default=12.0)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    import torch
    torch.set_num_threads(1)
    import mini_motorways_env as M
    from experiments import gamectl
    os.makedirs(os.path.join(OUT, "raw"), exist_ok=True)
    t0 = time.time()
    with gamectl.GameInstance(f"s22_{a.variant}_{a.idx.replace(',', '')}") as g:
        env = M.MiniMotorwaysEnv(max_decisions=a.decisions, decision_delta=1.0, request_path=os.path.join(g.ipc_dir, "request.json"), response_path=os.path.join(g.ipc_dir, "response.json"), start_mode="NORMAL_START")
        for k in [int(x) for x in a.idx.split(",")]:
            seed = SEEDS[k]
            rec = run_episode(env, a.variant, seed, t0 + 60 * a.deadline_min)
            name = f"{'smoke_' if a.smoke else ''}{a.variant}_{seed}.json.gz"
            with gzip.open(os.path.join(OUT, "raw", name), "wt") as f:
                json.dump(rec, f)
            st = rec["steps"]
            print(f"[{a.variant}] seed {seed}: steps {len(st)} deliveries {st[-1]['score_cum']} connected_end {st[-1]['connected']}/{st[-1]['houses']} edits {sum(e['legal'] for e in rec['edits'])}/{len(rec['edits'])} illegal {sum(not s['legal'] for s in st)} lost {sum(len(s['lost']) for s in st)} ({time.time() - t0:.0f}s)", flush=True)
        env.close()


if __name__ == "__main__":
    main()
