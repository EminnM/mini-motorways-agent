"""Validation suite for the synthetic sandbox. Run BEFORE any training:  python -m synthetic_rl.tests"""
import json
import os
import random
import sys
import time
from collections import deque

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from synthetic_rl import layout as L  # noqa: E402
from synthetic_rl.env import SyntheticGapEnv, load_tasks  # noqa: E402
import mini_motorways_env as native  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MAN = os.path.join(ROOT, "research", "synthetic_lab", "manifests")
OUT = os.path.join(ROOT, "research", "synthetic_lab")
RESULTS = {}


def check(name, ok, detail):
    RESULTS[name] = {"passed": bool(ok), "detail": detail}
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)


def tasks_for(n, split):
    return load_tasks(os.path.join(MAN, f"gap{n}_{split}.jsonl"))


# ---------------------------------------------------------------- native-format roundtrip (mapping test)
_ROAD_NAME = {0.0: "None", L.ROAD_ACTIVE: "Active"}


def native_style_observation(env):
    """Build the native PolicyObservation JSON dict from the synthetic state, using the NATIVE field names, so the REAL
    encode_observation() can produce the vector. This is the explicit, tested feature mapping."""
    t = env.task
    house = tuple(t["house"])
    foot = {tuple(f) for f in t["footprint"]}
    tiles = []
    for y in range(L.YMIN, L.YMIN + L.H):
        for x in range(L.XMIN, L.XMIN + L.W):
            road = {}
            for d in "NESW":
                road[d] = "Active" if L.edge_key(x, y, d) in env.edges else "None"
            ct = "House" if (x, y) == house else "Destination" if (x, y) in foot else ("Carpark" if (x, y) == tuple(t["carpark"]) else "None")
            tiles.append({"coord": [x, y], "buildable": True, "overWater": False, "underMountain": False, "overRail": False,
                          "road": road, "hasTrafficLight": False, "hasRoundabout": False, "contentType": ct,
                          "groupIndex": 0 if (x, y) == house or (x, y) in foot else None,
                          "destination": ({"totalDemand": 0, "isOvercrowding": False} if (x, y) in foot else None),
                          "visibleVehicleCount": 0})
    hours = env.t * 1.2
    g = {"hour": int(hours) % 24, "day": int(hours) // 24, "week": 0, "score": env.score,
         "concreteAvailable": env.concrete, "concreteTotal": env.concrete_total,
         "bridgesAvailable": 0, "bridgesTotal": 0, "tunnelsAvailable": 0, "tunnelsTotal": 0, "motorwaysAvailable": 0, "motorwaysTotal": 0,
         "trafficLightsAvailable": 0, "trafficLightsTotal": 0, "roundaboutsAvailable": 0, "roundaboutsTotal": 0, "pendingUpgradeChoices": 0}
    return {"tiles": tiles, "globals": g}


def obs_only_solver(obs, mask):
    """Chooses an action from the OBSERVATION VECTOR and the legal mask only (no env internals): contract road
    components; breadth-first search from the house component to the destination component over legal new edges."""
    tiles = obs[: L.N_TILES * L.NUM_TILE_FEATURES].reshape(L.H, L.W, L.NUM_TILE_FEATURES)
    road = {d: tiles[:, :, L.F_ROAD[d]] > 0 for d in "NESW"}
    content = tiles[:, :, L.F_CONTENT]
    parent = {}

    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for yi in range(L.H):
        for xi in range(L.W):
            for d, (dx, dy) in L.DIRS.items():
                if road[d][yi, xi]:
                    nx, ny = xi + dx, yi + dy
                    if 0 <= nx < L.W and 0 <= ny < L.H:
                        parent[find((xi, yi))] = find((nx, ny))
    house = [(xi, yi) for yi in range(L.H) for xi in range(L.W) if abs(content[yi, xi] - L.CONTENT_HOUSE) < 1e-6]
    foot = [(xi, yi) for yi in range(L.H) for xi in range(L.W) if abs(content[yi, xi] - L.CONTENT_DEST) < 1e-6]
    if not house or not foot:
        return 0
    fs = set(foot)
    carp = None
    for (xi, yi) in foot:
        for d, (dx, dy) in L.DIRS.items():
            c = (xi + dx, yi + dy)
            if c in fs or not (0 <= c[0] < L.W and 0 <= c[1] < L.H) or content[c[1], c[0]] != 0:
                continue
            if any(road[dd][c[1], c[0]] for dd in "NESW"):
                carp = c
    if carp is None:
        return 0
    src, dst = find(house[0]), find(carp)
    if src == dst:
        return 0
    # legal ADD edges between components
    edges = {}
    for yi in range(L.H):
        for xi in range(L.W):
            for d in ("E", "S"):
                a = 1 + (yi * L.W + xi) * 4 + (0 if d == "E" else 1) * 2
                if mask[a]:
                    nx, ny = (xi + 1, yi) if d == "E" else (xi, yi - 1)
                    cu, cv = find((xi, yi)), find((nx, ny))
                    if cu != cv:
                        edges.setdefault(cu, []).append((cv, a))
                        edges.setdefault(cv, []).append((cu, a))
    prev = {src: None}
    q = deque([src])
    while q:
        u = q.popleft()
        if u == dst:
            break
        for v, a in edges.get(u, []):
            if v not in prev:
                prev[v] = (u, a)
                q.append(v)
    if dst not in prev:
        return 0
    node = dst
    first = None
    while prev[node] is not None:
        u, a = prev[node]
        if u == src:
            first = a
        node = u
    return first if first is not None else 0


def run_suite():
    g1 = {s: tasks_for(1, s) for s in ("train", "validation", "test")}
    # ------------------------------------------------------------------ 1. oracle completes every task (Gap-1/2/3, all splits)
    tot = ok = 0
    worst = 0
    for n in (1, 2, 3):
        for s in ("train", "validation", "test"):
            ts = tasks_for(n, s)
            env = SyntheticGapEnv(ts, sampling="sequential")
            for i in range(len(ts)):
                env.reset(options={"task_index": i})
                done = False
                steps = 0
                while not done:
                    _, r, te, tr, info = env.step(env.oracle_action())
                    steps += 1
                    done = te or tr
                tot += 1
                ok += bool(info["synthetic"]["success"] and r == 1.0 and te and steps == n)
                worst = max(worst, steps)
    check("1_oracle_completes_every_generated_task", ok == tot, f"{ok}/{tot} tasks (Gap-1/2/3, train+validation+test); oracle uses exactly N actions (max steps {worst})")
    # ------------------------------------------------------------------ 2. random is much worse
    rng = np.random.default_rng(0)
    res = {}
    for n in (1, 2, 3):
        ts = tasks_for(n, "test")
        env = SyntheticGapEnv(ts, sampling="sequential")
        succ = 0
        for i in range(len(ts)):
            env.reset(options={"task_index": i})
            done = False
            while not done:
                _, r, te, tr, info = env.step(int(rng.choice(np.flatnonzero(env.action_masks()))))
                done = te or tr
            succ += info["synthetic"]["success"]
        res[f"gap{n}"] = succ / len(ts)
    check("2_random_policy_substantially_worse_than_oracle", res["gap1"] < 0.30 and res["gap2"] < res["gap1"] + 1e-9 and res["gap3"] <= res["gap2"] + 0.02,
          f"random-legal completion on TEST: {res} (oracle 1.0)")
    # ------------------------------------------------------------------ 3. intended actions are legal
    bad = 0
    legal_counts = []
    for n in (1, 2, 3):
        for s in ("train", "validation", "test"):
            ts = tasks_for(n, s)
            env = SyntheticGapEnv(ts, sampling="sequential")
            for i in range(len(ts)):
                env.reset(options={"task_index": i})
                m = env.action_masks()
                legal_counts.append(int(m.sum()))
                if not all(m[a] for a in env.task["solution_action_indices"]):
                    bad += 1
    check("3_every_intended_correct_action_is_legal", bad == 0, f"{bad} violations over all tasks; legal actions per state: mean {np.mean(legal_counts):.0f}, min {min(legal_counts)}, max {max(legal_counts)} (native Gap-1 starts: mean 151)")
    # ------------------------------------------------------------------ 4. observation / action mappings
    env = SyntheticGapEnv(g1["test"], sampling="sequential")
    mism = 0
    for i in range(40):
        obs, _ = env.reset(options={"task_index": i})
        for _ in range(3):
            m = env.action_masks()
            obs, *_ = env.step(int(np.random.default_rng(i).choice(np.flatnonzero(m))))
            ref = native.encode_observation(native_style_observation(env))
            if not np.array_equal(ref, obs):
                mism += 1
    bij = all(
        (lambda dec: (dec[0] == "WAIT" and a == 0) or (dec[0] in ("ADD", "REMOVE") and L.action_index(dec[0], dec[1], dec[2], dec[3]) == a) or (dec[0] == "UPGRADE" and L.UPGRADE_BASE + dec[1] == a))(L.decode_action(a))
        for a in range(L.N_ACTIONS))
    check("4_observation_and_action_mappings_consistent", mism == 0 and bij and L.OBS_DIM == 11666 and L.N_ACTIONS == 3111,
          f"synthetic obs == real encode_observation(native-style dict) in {120 - mism}/120 states; action index<->descriptor bijection over all {L.N_ACTIONS} indices: {bij}; "
          f"derived sizes obs {L.OBS_DIM}, actions {L.N_ACTIONS}")
    # ------------------------------------------------------------------ 5. rewards / termination semantics
    problems = []
    env = SyntheticGapEnv(g1["test"], sampling="sequential")
    env.reset(options={"task_index": 0})
    _, r, te, tr, info = env.step(0)
    if not (r == 0 and not te and not tr):
        problems.append("wait")
    env.reset(options={"task_index": 0})
    a_star = env.task["solution_action_indices"][0]
    _, r, te, tr, info = env.step(a_star)
    if not (r == 1.0 and te and not tr and info["synthetic"]["success"]):
        problems.append("completion")
    env.reset(options={"task_index": 1})
    legal_add = [a for a in np.flatnonzero(env.action_masks()) if a != env.task["solution_action_indices"][0] and L.decode_action(a)[0] == "ADD"]
    _, r, te, tr, info = env.step(int(legal_add[0]))
    if not (r == 0 and not te and not tr):
        problems.append("wrong_add")
    # destructive: remove an original edge where legal
    done_destr = False
    for i in range(60):
        env.reset(options={"task_index": i})
        m = env.action_masks()
        rem = [a for a in np.flatnonzero(m) if L.decode_action(a)[0] == "REMOVE" and L.edge_key(*L.decode_action(a)[1:]) in env.original]
        if rem:
            _, r, te, tr, info = env.step(int(rem[0]))
            if not (r == 0 and te and not tr and info["synthetic"]["destructive_failure"] and not info["synthetic"]["success"]):
                problems.append("destructive")
            done_destr = True
            break
    if not done_destr:
        problems.append("no_destructive_case_found")
    env.reset(options={"task_index": 2})
    for k in range(20):
        _, r, te, tr, info = env.step(0)
    if not (tr and not te and r == 0 and info["synthetic"]["timeout"] and k == 19):
        problems.append("timeout")
    for mode, expect_len in (("delayed", 1 + 12), ("native_like", 1 + 13 * 3)):
        e2 = SyntheticGapEnv(g1["test"], reward_mode=mode, sampling="sequential")
        e2.reset(options={"task_index": 0})
        rs = []
        step = 0
        done = False
        _, r, te, tr, info = e2.step(e2.oracle_action())
        rs.append(r)
        while not (te or tr):
            if not (e2.action_masks()[0] and e2.action_masks().sum() == 1):
                problems.append(f"{mode}_mask_not_wait_only")
                break
            _, r, te, tr, info = e2.step(0)
            rs.append(r)
        if len(rs) != expect_len or sum(rs) != (1.0 if mode == "delayed" else 3.0):
            problems.append(f"{mode}: len {len(rs)} sum {sum(rs)}")
    check("5_rewards_and_termination_semantics", not problems, f"WAIT=0, completion +1 & terminates, wrong ADD=0, destructive terminates with 0, timeout truncates at 20, "
          f"delayed reward arrives {12} WAIT steps after connecting and terminates, native-like stream delivers 3 sparse +1; problems={problems}")
    # ------------------------------------------------------------------ 6. disjoint layouts
    leaks = []
    for n in (1, 2, 3):
        sig = {s: {t["base_signature"] for t in tasks_for(n, s)} for s in ("train", "validation", "test")}
        for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
            if sig[a] & sig[b]:
                leaks.append((n, a, b, len(sig[a] & sig[b])))
        full = {s: {t["signature"] for t in tasks_for(n, s)} for s in ("train", "validation", "test")}
        for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
            if full[a] & full[b]:
                leaks.append((n, a, b, "config"))
    check("6_train_and_test_layouts_disjoint", not leaks, f"overlaps={leaks} (checked base layouts and full configurations for Gap-1/2/3)")
    # ------------------------------------------------------------------ 7. no label leakage
    env = SyntheticGapEnv(g1["test"], sampling="sequential")
    obs_a, _ = env.reset(options={"task_index": 5})
    saved = list(env.task["solution_edges"]), list(env.task["solution_action_indices"])
    env.task["solution_edges"] = [[0, 0, "E"]]
    env.task["solution_action_indices"] = [1]
    env.solution = [(0, 0, "E")]
    obs_b = env._obs.copy()
    leak_struct = not np.array_equal(obs_a, obs_b)
    # also: observation is a pure function of (built edges, house/destination placement, counters)
    env2 = SyntheticGapEnv(g1["test"], sampling="sequential")
    obs_c, _ = env2.reset(options={"task_index": 5})
    check("7_no_correct_action_labels_in_observation", (not leak_struct) and obs_a.shape == (L.OBS_DIM,) and np.array_equal(obs_a, obs_c),
          f"obs unchanged when the stored solution is overwritten: {not leak_struct}; shape {obs_a.shape} (native-identical); the solution only exists in env.task/info metadata")
    # ------------------------------------------------------------------ 8. enough information to infer the construction
    solved = tot = 0
    for n in (1, 2, 3):
        ts = tasks_for(n, "test")
        env = SyntheticGapEnv(ts, sampling="sequential")
        for i in range(len(ts)):
            obs, _ = env.reset(options={"task_index": i})
            done = False
            while not done:
                a = obs_only_solver(obs, env.action_masks())
                obs, r, te, tr, info = env.step(a)
                done = te or tr
            tot += 1
            solved += info["synthetic"]["success"]
    check("8_observation_contains_enough_information", solved == tot, f"a solver that sees ONLY the observation vector + legal mask (component contraction + BFS) solves {solved}/{tot} test tasks (Gap-1/2/3)")
    # informational: how often is the UNCONSTRAINED generator observation-ambiguous (native-faithful carpark invisibility)?
    from synthetic_rl import generator as GEN
    rng_a = random.Random(5)
    amb_tasks = []
    for _ in range(300):
        t = GEN.make_task(rng_a, 1, clear_footprint=False)
        t["task_id"] = "amb"
        amb_tasks.append(t)
    env_a = SyntheticGapEnv(amb_tasks, sampling="sequential")
    amb_ok = 0
    for i in range(len(amb_tasks)):
        obs, _ = env_a.reset(options={"task_index": i})
        a = obs_only_solver(obs, env_a.action_masks())
        amb_ok += a in env_a.task["solution_action_indices"]
    RESULTS["_info_unconstrained_generator_obs_only_solver_gap1"] = {"solved": amb_ok, "of": len(amb_tasks),
        "note": "when roads may pass beside the destination, the carpark is not identifiable from the native-compatible observation"}
    print(f"[INFO] unconstrained generator (roads may touch the destination): observation-only solver solves {amb_ok}/{len(amb_tasks)} Gap-1 tasks", flush=True)
    # ------------------------------------------------------------------ 9. adapter tested over feature groups (explicit table)
    env = SyntheticGapEnv(g1["test"], sampling="sequential")
    obs, _ = env.reset(options={"task_index": 0})
    tiles = obs[: L.N_TILES * L.NUM_TILE_FEATURES].reshape(L.N_TILES, L.NUM_TILE_FEATURES)
    glob = obs[L.N_TILES * L.NUM_TILE_FEATURES:]
    table = {
        "buildable (0)": "1 on every tile (neutral; no terrain modelled)",
        "water/mountain/rail (1-3)": "0 (not modelled)",
        "road N/E/S/W (4-7)": "Active/3 = 0.667 on live edges, 0 otherwise (both endpoint tiles)",
        "traffic light / roundabout (8-9)": "0 (not modelled)",
        "contentType (10)": "House 0.5, Destination 1.0 (4 footprint tiles); carpark tile is content 0 exactly as native (unknown content -> 0)",
        "hasGroupIndex (11)": "1 on house and destination tiles",
        "dest demand / overcrowding (12-13)": "0 (not modelled)",
        "visibleVehicleCount (14)": "0 (no traffic)",
        "globals hour/day/week/score": "hour=1.2*t mod 24 /24, day, week=0, score/1000",
        "globals concrete ratio": "(25 - built edges - added edges)/25",
        "globals other resources, pending upgrades": "0 (not modelled)",
    }
    ok = (tiles[:, L.F_BUILDABLE] == 1).all() and (tiles[:, [L.F_WATER, L.F_MOUNTAIN, L.F_RAIL, L.F_LIGHT, L.F_ROUNDABOUT, L.F_DEMAND, L.F_OVERCROWD, L.F_VEHICLES]] == 0).all() \
        and set(np.unique(tiles[:, L.F_CONTENT]).round(3).tolist()) == {0.0, 0.5, 1.0} and glob[5:].sum() == 0
    check("9_synthetic_to_native_adapter_has_explicit_tested_mappings", ok, "mapping table below verified against the real encode_observation (check 4); neutral fields asserted constant: " + json.dumps(table))
    # ------------------------------------------------------------------ 10. throughput
    env = SyntheticGapEnv(g1["train"], reward_mode="completion")
    env.reset(seed=0)
    rng = np.random.default_rng(1)
    n = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 3.0:
        m = env.action_masks()
        a = int(rng.choice(np.flatnonzero(m)))
        _, r, te, tr, info = env.step(a)
        n += 1
        if te or tr:
            env.reset()
    sps = n / (time.perf_counter() - t0)
    check("10_throughput_without_the_game", sps > 2000, f"{sps:.0f} env steps/s (single process, random legal actions incl. mask + 11,666-float observation per step); native game ~4-5 steps/s")
    RESULTS["_throughput_steps_per_second"] = sps


def hand_example():
    tasks = tasks_for(1, "test")
    env = SyntheticGapEnv(tasks, sampling="sequential")
    obs, _ = env.reset(options={"task_index": 3})
    t = env.task
    m = env.action_masks()
    sol = t["solution_action_indices"][0]
    lines = []
    lines.append("HAND-INSPECTABLE EXAMPLE (Gap-1, test task " + t["task_id"] + ")")
    lines.append("window (reach) x=%d..%d y=%d..%d ; H=house tile, D=destination footprint, C=carpark, '+'=road tile" % (t["window"][0], t["window"][2], t["window"][1], t["window"][3]))
    lines.append(env.render_ascii())
    lines.append(f"house {t['house']} driveway {t['driveway']} -> start tile {t['start']};  carpark {t['carpark']} stub {t['stub_dir']} -> goal tile {t['goal']}")
    lines.append(f"path ({t['path_length_edges']} edges, {t['n_gaps']} left out): " + " ".join(f"({p[0]},{p[1]})" for p in t["path"]))
    ex = t["solution_edges"][0]
    lines.append(f"missing edge: ADD_ROAD({ex[0]},{ex[1]},{ex[2]}) = action index {sol}  (decode -> {L.decode_action(sol)}); legal now: {bool(m[sol])}")
    lines.append(f"legal actions: {int(m.sum())} of {L.N_ACTIONS} (WAIT, {int(sum(1 for a in np.flatnonzero(m) if L.decode_action(a)[0]=='ADD'))} ADD, {int(sum(1 for a in np.flatnonzero(m) if L.decode_action(a)[0]=='REMOVE'))} REMOVE)")
    lines.append("non-zero observation features around the gap (tile: feature=value):")
    tiles = obs[: L.N_TILES * L.NUM_TILE_FEATURES].reshape(L.N_TILES, L.NUM_TILE_FEATURES)
    for (x, y) in [(ex[0], ex[1]), L.neighbor(ex[0], ex[1], ex[2])]:
        nz = {i: round(float(v), 3) for i, v in enumerate(tiles[L.tile_index(x, y)]) if v}
        lines.append(f"  tile ({x},{y}) idx {L.tile_index(x, y)}: {nz}   [4..7 = road N/E/S/W, 10 = content, 11 = group flag]")
    _, r, te, tr, info = env.step(sol)
    lines.append(f"after ADD_ROAD({ex[0]},{ex[1]},{ex[2]}): reward {r}, terminated {te}, connected {env.connected()}")
    lines.append(env.render_ascii())
    return "\n".join(lines)


if __name__ == "__main__":
    run_suite()
    ex = hand_example()
    print("\n" + ex)
    os.makedirs(OUT, exist_ok=True)
    json.dump(RESULTS, open(os.path.join(OUT, "validation_results.json"), "w"), indent=2)
    open(os.path.join(OUT, "hand_example.txt"), "w", encoding="utf-8").write(ex + "\n")
    sys.exit(0 if all(v["passed"] for k, v in RESULTS.items() if not k.startswith("_")) else 1)
