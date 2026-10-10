"""S24 pass 1: anchor inspection (no fork). Planner-B main line to decision 200 per seed; records the anchor observation + mask, candidate edit sets (S22 rules, unchanged), per-tile vehicle counts over
decisions 170-200, and evaluates the preregistered eligibility rules E1-E4 structurally.   python research/s24_loaded_anchor/code/inspect_anchors.py -> raw/anchor_<seed>.json.gz, eligibility.json"""
import gzip
import heapq
import json
import os
import sys
import time
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(R, "s22_throughput", "code"))
sys.path.insert(0, os.path.join(R, "s21_native_check", "code"))
from native_adapter import NativePlannerPolicy, ROOT  # noqa: E402
import throughput_probe as TP  # noqa: E402
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import s11_guard as G  # noqa: E402
from synthetic_rl import layout as L  # noqa: E402
from executor import make_policy  # noqa: E402
from planner import Weights  # noqa: E402

OUT = os.path.join(ROOT, "research", "s24_loaded_anchor")
SEEDS = json.load(open(os.path.join(ROOT, "research", "s9_native_transfer", "seeds.json")))["seeds"]
ANCHOR, TRAFFIC_FROM = 200, 170


def adj_of(edges):
    adj = {}
    for e in edges:
        a, b = G.ends(e)
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    return adj


def components(adj):
    comp, seen = {}, set()
    for s in adj:
        if s in seen:
            continue
        st, cid = [s], s
        seen.add(s)
        while st:
            u = st.pop()
            comp[u] = cid
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    st.append(v)
    return comp


def cyclomatic(edges):
    adj = adj_of(edges)
    n = len(adj)
    c = len(set(components(adj).values()))
    return len(edges) - n + c


def shortest(adj, src, goals, forbid=frozenset()):
    """BFS hop count from src to any goal tile over the functional graph; returns (dist, path)"""
    prev, dist = {src: None}, {src: 0}
    dq = [src]
    while dq:
        u = dq.pop(0)
        if u in goals and u != src:
            p = [u]
            while prev[p[-1]] is not None:
                p.append(prev[p[-1]])
            return dist[u], p[::-1]
        for v in adj.get(u, ()):
            if v not in dist and v not in forbid:
                dist[v] = dist[u] + 1
                prev[v] = u
                dq.append(v)
    return None, None


def structural_checks(raw, mask, edits):
    ctx = G.Ctx(raw)
    func = G.edges_from_raw(raw)
    conn = G.connected_house_set(ctx, func)
    legal = all(bool(mask[L.action_index("ADD", *e)]) for e in edits)
    after = set(func) | set(edits)
    adj0, adj1 = adj_of(func), adj_of(after)
    road0 = set(adj0)
    # distinct route: added-edge subgraph connects two distinct existing road tiles through >= 3 added edges (cycle through the old graph)
    adj_add = adj_of(edits)
    attach = {t for t in adj_add if t in road0}
    comp_add = components(adj_add)
    distinct = False
    for cid in set(comp_add.values()):
        members = [t for t, c in comp_add.items() if c == cid]
        att = [t for t in members if t in road0]
        n_edges = sum(1 for e in edits if comp_add.get(G.ends(e)[0]) == cid)
        if len(att) >= 2 and n_edges >= 3:
            distinct = True
    dcyc = cyclomatic(after) - cyclomatic(func)
    # usability: per connected house, hop distance to any own-colour destination stub tile (tile adjacent to a carpark of a same-group destination)
    carparks = {k for k, c in ctx.content.items() if c == "Carpark"}
    dest_group = {k: ctx.group.get(k) for k, c in ctx.content.items() if c == "Destination"}
    stub_tiles = {}
    for e in func:
        a, b = G.ends(e)
        for cp, other in ((a, b), (b, a)):
            if cp in carparks:
                groups = {dest_group.get((cp[0] + dx, cp[1] + dy)) for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)) if (cp[0] + dx, cp[1] + dy) in dest_group}
                for g in groups:
                    stub_tiles.setdefault(g, set()).add(other)
    usable, per_house = False, {}
    added_tiles = {t for e in edits for t in G.ends(e)}
    for h in conn:
        g = ctx.group.get(h)
        goals = stub_tiles.get(g, set())
        d0, p0 = shortest(adj0, h, goals)
        d1, p1 = shortest(adj1, h, goals)
        on_added = bool(p1) and any(G.canon(u, v) in set(edits) for u, v in zip(p1[:-1], p1[1:]))
        # equal-length alternative using added edges: search again forbidding nothing but requiring an added edge -> approximate by checking all equal-length paths via a second BFS that forbids the first path's interior
        alt_equal = False
        if d1 is not None and d0 is not None and d1 == d0 and p0:
            d2, p2 = shortest(adj1, h, goals, forbid=set(p0[1:-1]))
            alt_equal = d2 is not None and d2 == d0 and any(G.canon(u, v) in set(edits) for u, v in zip(p2[:-1], p2[1:]))
        per_house[str(h)] = {"group": g, "hops_before": d0, "hops_after": d1, "shortest_after_uses_added": on_added, "equal_length_alternative_via_added": alt_equal}
        if (d1 is not None and d0 is not None and d1 < d0) or on_added or alt_equal:
            usable = True
    return {"n_edits": len(edits), "legal_all": legal, "distinct_route": distinct, "cyclomatic_increase": dcyc, "usable": usable, "per_house": per_house, "attach_tiles": sorted(attach), "added_tiles": sorted(added_tiles)}


def main():
    import mini_motorways_env as M
    from experiments import gamectl
    os.makedirs(os.path.join(OUT, "raw"), exist_ok=True)
    t0 = time.time()
    elig = {}
    with gamectl.GameInstance("s24_inspect") as g:
        env = M.MiniMotorwaysEnv(max_decisions=100000, decision_delta=1.0, request_path=os.path.join(g.ipc_dir, "request.json"), response_path=os.path.join(g.ipc_dir, "response.json"), start_mode="NORMAL_START")
        for seed in SEEDS:
            pol = NativePlannerPolicy(make_policy(Weights()))
            env.reset(seed=seed)
            pol.reset()
            raw = env.last_raw_observation
            veh = Counter()
            total_veh = 0
            hashes = []
            terminated = False
            for i in range(ANCHOR):
                mask = env.action_masks().copy()
                hashes.append(TP.obs_hash(raw))
                if i >= TRAFFIC_FROM:
                    for t in raw["tiles"]:
                        c = t.get("visibleVehicleCount", 0)
                        if c:
                            veh[(t["coord"][0], t["coord"][1])] += c
                            total_veh += c
                _, _, te, tr, _ = env.step(pol.act(raw, mask))
                raw = env.last_raw_observation
                if te or tr:
                    terminated = True
                    break
            mask = env.action_masks().copy()
            par = TP.edits_parallel(raw, mask) if not terminated else []
            den = TP.edits_dense(raw, mask) if not terminated else []
            ctx = G.Ctx(raw)
            func = G.edges_from_raw(raw)
            conn = G.connected_house_set(ctx, func)
            bott = [list(k) for k, _ in veh.most_common(5)]
            rec = {"seed": seed, "terminated_before_anchor": terminated, "decisions_played": len(hashes), "houses": len(ctx.houses), "connected": [list(h) for h in sorted(conn)], "n_func": len(func), "concrete": raw["globals"]["concreteAvailable"], "score": raw["globals"]["score"],
                   "traffic_sum_170_200": total_veh, "bottleneck_tiles": bott, "vehicle_counts_by_tile": {f"{k[0]},{k[1]}": v for k, v in veh.items()}, "edits": {"parallel": [list(e) for e in par], "dense": [list(e) for e in den]},
                   "structure": {"parallel": structural_checks(raw, mask, par) if not terminated else None, "dense": structural_checks(raw, mask, den) if not terminated else None}}
            E1 = len(conn) >= 2 and not terminated
            E2 = total_veh >= 10
            st = rec["structure"]
            valid = {n: bool(st and st[n] and st[n]["n_edits"] > 0 and st[n]["legal_all"] and (st[n]["distinct_route"] if n == "parallel" else st[n]["cyclomatic_increase"] >= 2)) for n in ("parallel", "dense")}
            usable = {n: bool(st and st[n] and st[n]["usable"]) for n in ("parallel", "dense")}
            rec["eligibility"] = {"E1_connected>=2": E1, "E2_traffic>=10": E2, "E3_structurally_valid": valid, "E4_usable": usable, "eligible": E1 and E2 and (valid["parallel"] or valid["dense"]), "primary_branches": [n for n in ("parallel", "dense") if valid[n] and usable[n]]}
            elig[seed] = rec["eligibility"] | {"connected": len(conn), "houses": len(ctx.houses), "traffic": total_veh, "n_edits": {n: len(rec["edits"][n]) for n in rec["edits"]}, "terminated": terminated}
            rec["anchor_raw"], rec["anchor_mask"] = raw, [int(x) for x in mask.nonzero()[0]]
            with gzip.open(os.path.join(OUT, "raw", f"anchor_{seed}.json.gz"), "wt") as f:
                json.dump(rec, f)
            print(f"seed {seed}: played {len(hashes)} conn {len(conn)}/{len(ctx.houses)} traffic {total_veh} edits par {len(par)} den {len(den)} valid {valid} usable {usable} eligible {rec['eligibility']['eligible']} primary {rec['eligibility']['primary_branches']} ({time.time() - t0:.0f}s)", flush=True)
        env.close()
    json.dump(elig, open(os.path.join(OUT, "eligibility.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
