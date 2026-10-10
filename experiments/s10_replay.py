"""S10 measurement-only native replay of the frozen S8-C checkpoint (deterministic). The action is exactly model.predict(obs, native mask); everything else is read-only
logging from the raw observation: concrete, legal ADD/REMOVE split, edges, houses, destinations, connected houses, demand / overcrowding, vehicles, connection losses.
   python experiments/s10_replay.py --seed 563352
   python experiments/s10_replay.py --twin"""
import argparse
import gzip
import hashlib
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from experiments import paths  # noqa: E402,F401
import numpy as np  # noqa: E402
import torch  # noqa: E402

OUT = os.path.join(ROOT, "research", "s10_post_connection")
CK = os.path.join(ROOT, "research", "s8_long_route_training", "runs", "arm_C_s11", "checkpoints", "step_0099840.zip")
DIRS = {"N": (0, 1), "E": (1, 0), "S": (0, -1), "W": (-1, 0)}


def connected_and_unconnected_houses(raw):
    """same union-find as mini_motorways_env.connectivity_potential, but returns the house tiles (read-only diagnostic)"""
    nodes = {(t["coord"][0], t["coord"][1]): t for t in raw["tiles"]}
    parent = {}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for k, t in nodes.items():
        if any(v != "None" for v in t["road"].values()) or t["contentType"] != "None":
            parent[k] = k
    for k, t in nodes.items():
        if k not in parent:
            continue
        for d, (dx, dy) in DIRS.items():
            if t["road"][d] == "None":
                continue
            n = (k[0] + dx, k[1] + dy)
            if n in parent:
                union(k, n)
    for k, t in nodes.items():
        if k not in parent or t["contentType"] not in ("Destination", "Carpark"):
            continue
        for dx, dy in DIRS.values():
            n = (k[0] + dx, k[1] + dy)
            if n in parent and nodes[n]["contentType"] in ("Destination", "Carpark") and (t["contentType"] == "Destination" or nodes[n]["contentType"] == "Destination"):
                union(k, n)
    dg = {}
    for k, t in nodes.items():
        if k in parent and t["contentType"] == "Destination" and t.get("groupIndex") is not None:
            dg.setdefault(find(k), set()).add(t["groupIndex"])
    conn, unconn = [], []
    for k, t in nodes.items():
        if t["contentType"] == "House":
            (conn if (k in parent and t.get("groupIndex") in dg.get(find(k), ())) else unconn).append((k, t.get("groupIndex")))
    return conn, unconn


def dest_clusters(raw):
    tiles = {(t["coord"][0], t["coord"][1]): t for t in raw["tiles"] if t["contentType"] == "Destination"}
    seen, cl = set(), []
    for k in tiles:
        if k in seen:
            continue
        comp, st = [], [k]
        seen.add(k)
        while st:
            u = st.pop()
            comp.append(u)
            for dx, dy in DIRS.values():
                v = (u[0] + dx, u[1] + dy)
                if v in tiles and v not in seen:
                    seen.add(v)
                    st.append(v)
        g = tiles[comp[0]].get("groupIndex")
        dem = max((tiles[c].get("destination") or {}).get("totalDemand", 0) for c in comp)
        over = any((tiles[c].get("destination") or {}).get("isOvercrowding") for c in comp)
        cl.append({"tiles": comp, "group": g, "demand": dem, "overcrowding": bool(over)})
    return cl


def edges_of(raw):
    return {(t["coord"][0], t["coord"][1], d) for t in raw["tiles"] for d in "ES" if t["road"][d] != "None"}


def run(env, model, seed, deadline, max_steps=None, instrument=True, hashes=False):
    import networkx as nx
    from synthetic_rl import layout as L
    from synthetic_rl import s7_oracle as O
    obs, info = env.reset(seed=seed)
    st, hs = [], []
    first = {"conn": None, "deliv": None}
    te = tr = False
    i = 0
    cum = 0.0
    route_checks = []
    cut = False
    while not (te or tr):
        if time.time() > deadline or (max_steps is not None and i >= max_steps):
            cut = True
            break
        mask = env.action_masks().copy()
        raw_b = env.last_raw_observation
        a = int(model.predict(obs, action_masks=mask, deterministic=True)[0])
        dec = L.decode_action(a)
        if instrument:
            eb = edges_of(raw_b)
            conn_b, _ = connected_and_unconnected_houses(raw_b)
        obs, r, te, tr, inf = env.step(a)
        raw_a = env.last_raw_observation
        if hashes:
            hs.append((a, float(r), hashlib.sha1(obs.tobytes()).hexdigest()))
        if instrument:
            ea = edges_of(raw_a)
            conn_a, unconn_a = connected_and_unconnected_houses(raw_a)
            cl = dest_clusters(raw_a)
            g = raw_a["globals"]
            m2 = env.action_masks()
            kinds = [L.decode_action(int(x))[0] for x in np.flatnonzero(m2) if 0 < x < L.UPGRADE_BASE]
            rc = inf.get("rewardComponents", {})
            cum += float(rc.get("scoreDelta", 0.0))
            n_h = sum(1 for t in raw_a["tiles"] if t["contentType"] == "House")
            veh = sum(t.get("visibleVehicleCount", 0) for t in raw_a["tiles"])
            touched_house = False
            if dec[0] == "ADD":
                e = (dec[1], dec[2], dec[3])
                tm = {(t["coord"][0], t["coord"][1]): t for t in raw_b["tiles"]}
                touched_house = any(tm.get(p, {}).get("contentType") == "House" for p in [(e[0], e[1]), L.neighbor(*e)])
            rec = {"i": i, "kind": dec[0], "action": a, "touched_house": touched_house, "n_legal": int(mask.sum()), "n_legal_add_next": kinds.count("ADD"), "n_legal_remove_next": kinds.count("REMOVE"),
                   "n_legal_upgrade_next": int((np.flatnonzero(m2) >= L.UPGRADE_BASE).sum()), "concrete": g["concreteAvailable"], "concrete_total": g["concreteTotal"], "bridges": g["bridgesAvailable"],
                   "edges": len(ea), "added": len(ea - eb), "removed": len(eb - ea), "houses": n_h, "dests": len(cl), "connected": len(conn_a), "connected_before": len(conn_b),
                   "conn_lost": max(len(conn_b) - len(conn_a), 0), "vehicles": veh, "dest_demand_max": max([c["demand"] for c in cl] or [0]), "dest_overcrowding": sum(1 for c in cl if c["overcrowding"]),
                   "reward": float(r), "scoreDelta": float(rc.get("scoreDelta", 0.0)), "score_cum": cum, "sim_time": raw_a.get("simulationTime"), "accepted": inf.get("accepted"),
                   "unconnected_houses": [list(h[0]) for h in unconn_a]}
            if len(conn_a) >= 1 and first["conn"] is None:
                first["conn"] = i
            if rec["scoreDelta"] > 0 and first["deliv"] is None:
                first["deliv"] = i
            if i % 10 == 9 and unconn_a:        # can each unconnected house still be connected under the CURRENT native mask (budget + geometry)?
                G = O._graph(obs, m2)
                tm = {(t["coord"][0], t["coord"][1]): t for t in raw_a["tiles"]}
                carparks = {}
                for t in raw_a["tiles"]:
                    if t["contentType"] == "Carpark":
                        c = (t["coord"][0], t["coord"][1])
                        carparks[c] = None
                        for dx, dy in DIRS.values():
                            n = (c[0] + dx, c[1] + dy)
                            if n in tm and tm[n]["contentType"] == "Destination":
                                carparks[c] = tm[n].get("groupIndex")
                per = []
                for (hk, hg) in unconn_a:
                    tg = [c for c, gg in carparks.items() if gg == hg and c in G]
                    best = None
                    if hk in G:
                        for c in tg:
                            try:
                                v = int(nx.shortest_path_length(G, hk, c, weight="w"))
                                best = v if best is None else min(best, v)
                            except nx.NetworkXNoPath:
                                pass
                    per.append(best)
                route_checks.append({"i": i, "concrete": g["concreteAvailable"], "min_adds_per_unconnected_house_under_current_mask": per})
            st.append(rec)
        i += 1
    return st, first, route_checks, hs, cut, (te, tr), (env.last_raw_observation if instrument else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int)
    ap.add_argument("--twin", action="store_true")
    ap.add_argument("--deadline-min", type=float, default=9.0)
    a = ap.parse_args()
    torch.set_num_threads(1)
    from sb3_contrib import MaskablePPO
    import mini_motorways_env as M
    from experiments import gamectl
    model = MaskablePPO.load(CK, device="cpu")
    t0 = time.time()
    tag = "s10twin" if a.twin else f"s10_{a.seed}"
    with gamectl.GameInstance(tag) as g:
        env = M.MiniMotorwaysEnv(max_decisions=600, decision_delta=1.0, request_path=os.path.join(g.ipc_dir, "request.json"),
                                 response_path=os.path.join(g.ipc_dir, "response.json"), start_mode="NORMAL_START")
        if a.twin:
            h1 = run(env, model, 563352, time.time() + 300, 60, False, True)[3]
            h2 = run(env, model, 563352, time.time() + 300, 60, True, True)[3]
            h3 = run(env, model, 563352, time.time() + 300, 60, False, True)[3]
            res = {"steps": len(h1), "plain_vs_instrumented_identical": h1 == h2, "plain_vs_plain_identical": h1 == h3, "wall_s": round(time.time() - t0, 1)}
            json.dump(res, open(os.path.join(OUT, "twin_noninterference.json"), "w"), indent=1)
            print(json.dumps(res))
            env.close()
            return
        st, first, rc, _, cut, (te, tr), raw_end = run(env, model, a.seed, t0 + 60 * a.deadline_min)
        final = {"terminated": te, "truncated": tr, "cut": cut, "first": first, "dest_state_end": dest_clusters(raw_end), "wall_s": round(time.time() - t0, 1)}
        with gzip.open(os.path.join(OUT, f"replay_{a.seed}.json.gz"), "wt") as f:
            json.dump({"seed": a.seed, "final": final, "route_checks": rc, "steps": st}, f)
        print(f"seed {a.seed}: steps {len(st)} first_conn {first['conn']} first_deliv {first['deliv']} deliveries {st[-1]['score_cum']} houses {st[-1]['houses']} connected {st[-1]['connected']} "
              f"concrete_end {st[-1]['concrete']} min_concrete {min(x['concrete'] for x in st)} ({final['wall_s']}s)", flush=True)
        env.close()


if __name__ == "__main__":
    main()
