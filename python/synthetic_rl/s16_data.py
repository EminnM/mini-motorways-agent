"""S16 diagnostic dataset: colour-aware next-edge selection on the S13 multi-house environment with the v2 observation (unchanged encoders, unchanged envs).
A SCENE = one geometry seed: a land region of the S13 map, 2-3 destination blocks (distinct colours), 2-3 houses (>= 2 colours), all non-query houses pre-connected by the privileged controller,
the query house optionally partially built (frontier away from the house). LABEL = the set of legal ADD actions leaving the query house's current Active component (frontier) that lie on SOME shortest (concrete-cost) path from that component to a destination of the query house's colour (oracle = Dijkstra with ground-truth groups; the model only sees the v2 observation). COUNTERFACTUAL twin = same geometry, the
query house takes the colour of the nearest (decoy) destination; the v1 observation is identical, v2 differs, and the label set must change (checked automatically; scenes failing the check are
excluded from the colour-dependence evaluation). Splits are by geometry seed."""
import heapq
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from synthetic_rl import layout as L  # noqa: E402
from synthetic_rl import obs_v2 as V  # noqa: E402
from synthetic_rl import s13_controller as CT  # noqa: E402
from synthetic_rl.s13_env import DIRS, canon, ends  # noqa: E402
from synthetic_rl.s14_env import MultiHouseEnvV2  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "research", "s16_colour_routing")
RF_RADIUS = 3                                     # existing CNN: three 3x3 convs -> 7x7 receptive field


def _dest_block(x, y, g):
    """carpark column at x, 2x2 destination at x+1..x+2, rows y-1..y (matches native blocks); stub edge from the lower carpark going west"""
    st = [[x, y, "Carpark", None], [x, y - 1, "Carpark", None], [x + 1, y, "Destination", g], [x + 2, y, "Destination", g], [x + 1, y - 1, "Destination", g], [x + 2, y - 1, "Destination", g]]
    return st, [x - 1, y - 1, "E"], [(x + dx, y + dy) for dx in range(-1, 3) for dy in (-1, 0)]


def make_scene(rng, water, n_dest, n_house, region=((-9, 9), (-6, 4))):
    (x0, x1), (y0, y1) = region
    for _ in range(200):
        used = set()
        structs, free = [], []
        events = []                                            # the S13 env merges all destination tiles of one struct list into one block -> each extra block is its own t=1 event
        groups = list(rng.permutation(4)[:n_dest])            # native ids: arbitrary labels
        dests = []
        ok = True
        for g in groups:
            for _t in range(50):
                x, y = int(rng.integers(x0 + 1, x1 - 2)), int(rng.integers(y0 + 1, y1))
                st, stub, foot = _dest_block(x, y, int(g))
                pad = {(a + dx, b + dy) for a, b in foot for dx in (-1, 0, 1) for dy in (-1, 0, 1)}
                if any(p in water for p in foot) or (stub[0], stub[1]) in water or pad & used:
                    continue
                used |= pad
                if not dests:
                    structs += st
                    free.append(stub)
                else:
                    events.append({"t": 1, "struct": st, "free_edges": [stub]})
                dests.append((int(g), foot))
                break
            else:
                ok = False
        if not ok:
            continue
        houses = []
        hg = [int(groups[0]), int(groups[1])] + [int(rng.choice(groups)) for _ in range(n_house - 2)]
        rng.shuffle(hg)
        for g in hg:
            for _t in range(50):
                x, y = int(rng.integers(x0, x1 - 1)), int(rng.integers(y0, y1 + 1))
                if (x, y) in water or (x + 1, y) in water or (x, y) in used or (x + 1, y) in used:
                    continue
                used |= {(x, y), (x + 1, y)}
                structs.append([x, y, "House", g])
                free.append([x, y, "E"])
                houses.append(((x, y), g))
                break
            else:
                ok = False
        if ok:
            return {"id": "s16", "structures0": structs, "free_edges0": free, "events": events}, houses, dests
    raise RuntimeError("could not place scene")


def oracle(env, house, group):
    """all legal ADD edges on some shortest-cost path from the house's current Active component to a destination of `group` (cost: plain edge 1, Active 0, house edge 0)."""
    goals = set()
    for d in env.dests:
        if d["group"] == group:
            for e in env.free:
                a, b = ends(e)
                for cp in d["carparks"]:
                    if cp in (a, b):
                        goals.add(b if a == cp else a)
    mask = env.action_masks()
    blocked = {p for p, c in env.content.items() if c in ("Destination", "Carpark")}

    def nbrs(u):
        for dx, dy in DIRS.values():
            v = (u[0] + dx, u[1] + dy)
            if not L.in_bounds(*v) or v in blocked or (env.content.get(v) == "House" and v != house):
                continue
            e = canon(u, v)
            if env.edges.get(e) == "A":
                yield v, 0, e
            elif mask[L.action_index("ADD", *e)] and house not in (u, v) and env.content.get(u) != "House":
                yield v, 1, e                                    # plain edges only: house-edge swaps are excluded from the label (they are free for every target and carry no colour decision)

    def dijkstra(src_set, reverse=False):
        dist = {s: 0 for s in src_set}
        pq = [(0, s) for s in src_set]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            for v, c, _ in nbrs(u):
                if d + c < dist.get(v, 1e9):
                    dist[v] = d + c
                    heapq.heappush(pq, (d + c, v))
        return dist
    dh = dijkstra({house})
    dg = dijkstra(goals)
    best = min((dh.get(g, 1e9) for g in goals), default=1e9)
    if best >= 1e9:
        return None, None
    valid = set()
    for u in dh:
        if dh[u] != 0:
            continue                                             # FRONTIER only: edges leaving the house's current Active component (cost-0 reachable set)
        for v, c, e in nbrs(u):
            if env.edges.get(e) == "A":
                continue
            if dh[u] + c + dg.get(v, 1e9) == best:
                valid.add(L.action_index("ADD", *e))
    return sorted(valid), int(best)


def connect_house(env, h, max_steps=60):
    for _ in range(max_steps):
        if h in env.connected_houses():
            return True
        p = CT._plan(env, h)
        if p is None:
            return False
        for u, v in zip(p[:-1], p[1:]):
            e = canon(u, v)
            if env.edges.get(e) != "A":
                a = L.action_index("ADD", *e)
                if not env.action_masks()[a]:
                    return False
                env.step(a)
                break
    return h in env.connected_houses()


def build_state(scen, houses, query, partial_steps, rng):
    env = MultiHouseEnvV2([scen], seed=int(rng.integers(1 << 30)), lam=0.0, mu=0.0, horizon=10000)
    env.reset()
    env.step(0)                                                  # applies the t=1 destination events
    env.concrete = 200
    env.total = 200                                              # resources never bind in the diagnostic
    for (h, g) in houses:
        if h != query and not connect_house(env, h):
            return None
    # partial construction of the query route (frontier moves away from the house)
    for _ in range(partial_steps):
        p = CT._plan(env, query)
        if p is None or len(p) < 3:
            break
        e = canon(p[0], p[1]) if env.edges.get(canon(p[0], p[1])) != "A" else next((canon(u, v) for u, v in zip(p[:-1], p[1:]) if env.edges.get(canon(u, v)) != "A"), None)
        if e is None:
            break
        a = L.action_index("ADD", *e)
        if not env.action_masks()[a] or query in env.connected_houses():
            break
        env.step(a)
    env._compute_mask()
    env._build_obs()
    return env


def rf_category(env, query, valid, group):
    """within RF if some valid edge tile is within RF_RADIUS of the query house AND of a destination tile of the right colour (the 7x7 window can see both colours)."""
    dest_tiles = [p for d in env.dests if d["group"] == group for p in d["tiles"]]
    within = False
    dmin = 99
    for a in valid:
        k = L.decode_action(a)
        for p in ends((k[1], k[2], k[3])):
            dh = max(abs(p[0] - query[0]), abs(p[1] - query[1]))
            dd = min(max(abs(p[0] - q[0]), abs(p[1] - q[1])) for q in dest_tiles)
            dmin = min(dmin, max(dh, dd))
            if dh <= RF_RADIUS and dd <= RF_RADIUS:
                within = True
    return within, dmin


def generate(n_geom, seed, split):
    rng = np.random.default_rng(seed)
    water = MultiHouseEnvV2([{"id": "x", "structures0": [[0, 0, "House", 0]], "free_edges0": [[0, 0, "E"]], "events": []}]).water
    samples = []
    stats = {"geometries": 0, "rejected": 0, "cf_excluded": 0}
    g = 0
    while stats["geometries"] < n_geom:
        g += 1
        n_dest = int(rng.choice([2, 3], p=[0.6, 0.4]))
        n_house = int(rng.choice([2, 3]))
        try:
            scen, houses, dests = make_scene(rng, water, n_dest, n_house)
        except RuntimeError:
            stats["rejected"] += 1
            continue
        query, qg = houses[int(rng.integers(len(houses)))]
        partial = int(rng.choice([0, 0, 1, 2, 3, 4]))
        env = build_state(scen, houses, query, partial, rng)
        if env is None or query in env.connected_houses():
            stats["rejected"] += 1
            continue
        valid, cost = oracle(env, query, qg)
        if not valid:
            stats["rejected"] += 1
            continue
        # decoy = nearest destination of another colour (by oracle cost); counterfactual twin: query house takes the decoy colour
        others = {}
        for dg_ in {d["group"] for d in env.dests if d["group"] != qg}:
            v2, c2 = oracle(env, query, dg_)
            if v2:
                others[dg_] = (v2, c2)
        if not others:
            stats["rejected"] += 1
            continue
        decoy_g = min(others, key=lambda k: others[k][1])
        decoy_valid, decoy_cost = others[decoy_g]
        obs = env._obs_v2.copy()
        mask = env.action_masks().copy()
        saved = env.houses[query]["group"]
        env.houses[query]["group"] = decoy_g
        env._build_obs()
        obs_cf = env._obs_v2.copy()
        env.houses[query]["group"] = saved
        env._build_obs()
        v1_same = np.array_equal(V.strip_to_v1(obs), V.strip_to_v1(obs_cf))
        v2_diff = not np.array_equal(obs, obs_cf)
        disjoint = not (set(valid) & set(decoy_valid))
        colour_dependent = v1_same and v2_diff and disjoint
        if not colour_dependent:
            stats["cf_excluded"] += 1
        within, dmin = rf_category(env, query, valid, qg)
        slot_map = dict(env.tracker.slots)
        base = {"geom": g, "split": split, "query": list(query), "query_group": int(qg), "query_slot": slot_map.get(qg), "decoy_group": int(decoy_g), "decoy_slot": slot_map.get(decoy_g), "n_dest": n_dest, "n_house": n_house,
                "partial": partial, "cost_correct": cost, "cost_decoy": decoy_cost, "decoy_nearer": decoy_cost < cost, "within_rf": within, "min_rf_dist": dmin, "colour_dependent": colour_dependent, "n_legal_add": int(mask[1:L.UPGRADE_BASE:2].sum())}
        samples.append({**base, "cf": False, "obs": obs, "mask": mask, "valid": valid, "other_valid": decoy_valid})
        if colour_dependent:
            samples.append({**base, "cf": True, "query_group": int(decoy_g), "query_slot": slot_map.get(decoy_g), "decoy_group": int(qg), "decoy_slot": slot_map.get(qg), "cost_correct": decoy_cost, "cost_decoy": cost, "decoy_nearer": cost < decoy_cost,
                            "obs": obs_cf, "mask": mask, "valid": decoy_valid, "other_valid": valid})
        stats["geometries"] += 1
    return samples, stats


def save(samples, path):
    meta = [{k: v for k, v in s.items() if k not in ("obs", "mask", "valid", "other_valid")} | {"valid": s["valid"], "other_valid": s["other_valid"]} for s in samples]
    np.savez_compressed(path, obs=np.stack([s["obs"] for s in samples]).astype(np.float16), mask=np.stack([s["mask"] for s in samples]))
    json.dump(meta, open(path.replace(".npz", ".json"), "w"))


if __name__ == "__main__":
    import time
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    allstats = {}
    for split, n, seed in (("train", 900, 1601), ("val", 150, 1602), ("test", 400, 1603)):
        s, st = generate(n, seed, split)
        save(s, os.path.join(OUT, f"{split}.npz"))
        allstats[split] = {**st, "samples": len(s), "colour_dependent_pairs": sum(1 for x in s if x["cf"]), "within_rf": sum(1 for x in s if x["within_rf"]), "decoy_nearer": sum(1 for x in s if x["decoy_nearer"]),
                           "mean_valid_set_size": float(np.mean([len(x["valid"]) for x in s])), "slots_of_query": {str(k): int(v) for k, v in zip(*np.unique([x["query_slot"] for x in s], return_counts=True))}}
        print(split, allstats[split], f"{time.time() - t0:.0f}s", flush=True)
    allstats["generation_s"] = round(time.time() - t0, 1)
    json.dump(allstats, open(os.path.join(OUT, "dataset_stats.json"), "w"), indent=1)
