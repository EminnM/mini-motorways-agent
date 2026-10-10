"""S20 flat-map curriculum generator (NO water / bridges / motorways / upgrades / traffic; no delayed-removal or pending mechanics).
A scene = one map seed -> a land region, 2-4 destination blocks (distinct colours), 2-6 houses (>= 2 colours, every house colour has a destination),
pre-existing networks built by the S18-H executor (GoalExecutor, one house at a time) for a random subset of the non-goal houses, 0-3 legal
redundant road stubs, and an optional partial prefix of the goal route (0-5 edges; 'correct' = planner prefix for the true colour, 'wrong' =
planner prefix built while the goal house temporarily carried the decoy colour).  Scenes whose build window contains a water tile are rejected,
so every ADD in the dataset is a plain land road.  Deterministic from the map seed; splits are disjoint seed ranges (common.SEEDS)."""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
from common import L, canon, ends, MultiHouseEnvV2, GoalExecutor, goal_route, oracle_valid, save_state, water_in_window  # noqa: E402

OUT = os.path.join(C.S20, "data")
REGIONS = {"A": ((-2, 7), (-6, 10)), "B": ((-14, -4), (-5, 10)), "C": ((-17, 18), (3, 10))}   # water-free rectangles of the S13 map (structures); the
# build window (min 14x8) may still reach the river column x=-3: scenes with any road on a water tile (pre-built, prefix or goal route) are rejected
MAX_PRE_STEPS = 40


def _dest_block(x, y, g):
    """carpark column at x, 2x2 destination at x+1..x+2, rows y-1..y; stub edge from the lower carpark going west (as S16 / native blocks)."""
    st = [[x, y, "Carpark", None], [x, y - 1, "Carpark", None], [x + 1, y, "Destination", g], [x + 2, y, "Destination", g], [x + 1, y - 1, "Destination", g], [x + 2, y - 1, "Destination", g]]
    return st, [x - 1, y - 1, "E"], [(x + dx, y + dy) for dx in range(-1, 3) for dy in (-1, 0)]


def make_scene(rng, water, n_dest, n_house, region):
    (x0, x1), (y0, y1) = region
    for _ in range(100):
        used = set()
        structs, free, events = [], [], []
        groups = [int(g) for g in rng.permutation(4)[:n_dest]]
        dests, ok = [], True
        for g in groups:
            for _t in range(60):
                x, y = int(rng.integers(x0 + 1, x1 - 2)), int(rng.integers(y0 + 1, y1))
                st, stub, foot = _dest_block(x, y, g)
                pad = {(a + dx, b + dy) for a, b in foot for dx in (-1, 0, 1) for dy in (-1, 0, 1)}
                if any(p in water for p in foot) or (stub[0], stub[1]) in water or pad & used or stub[0] < x0:
                    continue
                used |= pad
                if not dests:
                    structs += st
                    free.append(stub)
                else:
                    events.append({"t": 1, "struct": st, "free_edges": [stub]})
                dests.append((g, foot))
                break
            else:
                ok = False
        if not ok:
            continue
        houses = []
        hg = [groups[0], groups[1]] + [int(rng.choice(groups)) for _ in range(n_house - 2)]
        rng.shuffle(hg)
        for g in hg:
            for _t in range(60):
                x, y = int(rng.integers(x0, x1 - 1)), int(rng.integers(y0, y1 + 1))
                if (x, y) in water or (x + 1, y) in water or (x, y) in used or (x + 1, y) in used:
                    continue
                used |= {(x, y), (x + 1, y)}
                structs.append([x, y, "House", int(g)])
                free.append([x, y, "E"])
                houses.append(((x, y), int(g)))
                break
            else:
                ok = False
        if ok:
            return {"id": "s20", "structures0": structs, "free_edges0": free, "events": events}, houses, dests
    return None


def run_goal(env, house, max_steps):
    """run the S18-H executor for ONE house until connected (or max_steps); returns (connected, steps)."""
    ex = GoalExecutor(house)
    for s in range(max_steps):
        if house in env.connected_houses():
            return True, s
        a = ex.act(env)
        if a == 0:
            return False, s
        env.step(a)
    return house in env.connected_houses(), max_steps


def add_random_stubs(env, rng, n):
    """n legal plain ADDs (not touching a house, not adjacent to a destination/carpark) placed at random: redundant / disconnected road stubs."""
    placed = 0
    for _ in range(n * 20):
        if placed >= n:
            break
        m = env.action_masks()
        legal = np.flatnonzero(m[1:L.UPGRADE_BASE:2]) * 2 + 1
        if len(legal) == 0:
            break
        a = int(rng.choice(legal))
        d = L.decode_action(a)
        e = (d[1], d[2], d[3])
        pa, pb = ends(e)
        if env.content.get(pa) or env.content.get(pb):
            continue
        if any(env.content.get((p[0] + dx, p[1] + dy)) in ("House",) for p in (pa, pb) for dx, dy in C.DIRS.values()):
            continue
        env.step(a)
        placed += 1
    return placed


def generate_scene(seed, water):
    rng = np.random.default_rng(seed)
    region_name = str(rng.choice(list(REGIONS)))
    n_dest = int(rng.choice([2, 3, 4], p=[0.4, 0.4, 0.2]))
    n_house = int(rng.integers(2, 7))
    made = make_scene(rng, water, n_dest, n_house, REGIONS[region_name])
    if made is None:
        return None, "placement"
    scen, houses, dests = made
    env = MultiHouseEnvV2([scen], seed=seed, lam=0.0, mu=0.0, horizon=10 ** 6)
    env.reset()
    env.step(0)                                                   # applies the t=1 destination events
    if env.skipped_events or len(env.dests) != n_dest or len(env.houses) != n_house:
        return None, "spawn_shift"
    env.concrete = env.total = C.START_CONCRETE

    def road_on_water():
        return any(p in env.water for e in env.edges for p in ends(e))
    goal, gg = houses[int(rng.integers(len(houses)))]
    others = [h for h, _ in houses if h != goal]
    n_pre = int(rng.integers(0, len(others) + 1))
    pre = [others[i] for i in rng.permutation(len(others))[:n_pre]]
    pre_ok = 0
    for h in pre:
        ok, _ = run_goal(env, h, MAX_PRE_STEPS)
        pre_ok += int(ok)
    n_stubs = int(rng.choice([0, 0, 1, 2, 3]))
    stubs = add_random_stubs(env, rng, n_stubs)
    if goal in env.connected_houses():
        return None, "goal_connected_by_prebuild"
    if road_on_water():
        return None, "water_road_prebuild"
    # partial prefix of the goal route
    kind = str(rng.choice(["none", "correct", "wrong"], p=[0.5, 0.3, 0.2]))
    partial = 0
    if kind != "none":
        want = int(rng.integers(1, 6))
        if kind == "wrong":
            decoys = [g for g in {d["group"] for d in env.dests} if g != gg]
            if not decoys:
                kind = "correct"
            else:
                saved = env.houses[goal]["group"]
                env.houses[goal]["group"] = int(rng.choice(decoys))
                r = goal_route(env, goal)
                k = min(want, (r["n_adds"] - 1) if r else 0)
                ex = GoalExecutor(goal)
                for _ in range(k):
                    a = ex.act(env)
                    if a == 0:
                        break
                    env.step(a)
                    partial += 1
                env.houses[goal]["group"] = saved
                env._build_obs()
        if kind == "correct":
            r = goal_route(env, goal)
            k = min(want, (r["n_adds"] - 1) if r else 0)
            ex = GoalExecutor(goal)
            for _ in range(k):
                a = ex.act(env)
                if a == 0:
                    break
                env.step(a)
                partial += 1
    env._compute_mask()
    env._build_obs()
    if goal in env.connected_houses():
        return None, "goal_connected_after_prefix"
    r = goal_route(env, goal)
    if r is None or r["n_adds"] < 1 or not (2 <= r["route_len"] <= 14):
        return None, "route_len"
    if env.concrete < r["n_adds"] + 5:
        return None, "concrete"
    if road_on_water() or any(p in env.water for p in r["path"]):
        return None, "water_route"
    valid, best = oracle_valid(env, goal)
    if not valid:
        return None, "oracle_unreachable"
    conn = env.connected_houses()
    meta = {"seed": seed, "region": region_name, "n_dest": n_dest, "n_house": n_house, "goal": list(goal), "goal_group": gg, "n_pre_connected": pre_ok, "n_pre_tried": n_pre,
            "stubs": stubs, "prefix_kind": kind if partial else "none", "prefix_edges": partial, "route_len": r["route_len"], "n_adds": r["n_adds"], "reused": r["reused"],
            "n_edges_existing": len(env.edges), "n_plain_edges_existing": len([e for e in env.edges if e not in env.free]), "others_connected": len(conn),
            "valid_size": len(valid), "multi_route": len(valid) >= 2, "oracle_cost": best, "planner_cost": r["cost"], "water_tiles_in_window": water_in_window(env)}
    return {"meta": meta, "state": save_state(env)}, "ok"


def generate_split(split, water):
    scenes, rej = [], {}
    for seed in C.seed_range(split):
        sc, why = generate_scene(seed, water)
        if sc is None:
            rej[why] = rej.get(why, 0) + 1
        else:
            scenes.append(sc)
    return scenes, rej


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    water = MultiHouseEnvV2([{"id": "x", "structures0": [[0, 0, "House", 0]], "free_edges0": [[0, 0, "E"]], "events": []}]).water
    t0 = time.time()
    stats = {}
    for split in ("train", "val", "test"):
        scenes, rej = generate_split(split, water)
        json.dump(scenes, open(os.path.join(OUT, f"scenes_{split}.json"), "w"))
        m = [s["meta"] for s in scenes]
        stats[split] = {"seeds": C.SEEDS[split], "scenes": len(scenes), "rejected": rej,
                        "route_len_hist": {str(k): int(v) for k, v in zip(*np.unique([x["route_len"] for x in m], return_counts=True))},
                        "n_adds_hist": {str(k): int(v) for k, v in zip(*np.unique([x["n_adds"] for x in m], return_counts=True))},
                        "prefix_kind": {str(k): int(v) for k, v in zip(*np.unique([x["prefix_kind"] for x in m], return_counts=True))},
                        "region": {str(k): int(v) for k, v in zip(*np.unique([x["region"] for x in m], return_counts=True))},
                        "existing_edges_mean": float(np.mean([x["n_edges_existing"] for x in m])), "existing_edges_max": int(max(x["n_edges_existing"] for x in m)),
                        "multi_route_frac": float(np.mean([x["multi_route"] for x in m])), "mean_valid_size": float(np.mean([x["valid_size"] for x in m])),
                        "pre_connected_mean": float(np.mean([x["n_pre_connected"] for x in m])), "houses_mean": float(np.mean([x["n_house"] for x in m])),
                        "water_in_window_frac": float(np.mean([x["water_tiles_in_window"] > 0 for x in m]))}
        print(split, json.dumps(stats[split]), f"{time.time() - t0:.0f}s", flush=True)
    stats["generation_s"] = round(time.time() - t0, 1)
    json.dump(stats, open(os.path.join(OUT, "scene_stats.json"), "w"), indent=1)
