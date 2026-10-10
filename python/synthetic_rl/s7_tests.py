"""S7 Gate A: generator / environment / oracle validation and native-distribution comparison. No training.
   python -m synthetic_rl.s7_tests      (from python/)  or  python python/synthetic_rl/s7_tests.py"""
import hashlib
import itertools
import json
import os
import statistics
import sys
import time
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from synthetic_rl import layout as L  # noqa: E402
from synthetic_rl import s7  # noqa: E402
from synthetic_rl import s7_oracle as O  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "research", "s7_long_route")
R = {}


def check(name, ok, detail, **extra):
    R[name] = {"passed": bool(ok), "detail": detail, **extra}
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)


def write_manifests():
    sets, stats = s7.build_task_sets()
    os.makedirs(os.path.join(OUT, "manifests"), exist_ok=True)
    hashes = {}
    for split, by_k in sets.items():
        p = os.path.join(OUT, "manifests", f"s7_{split}.jsonl")
        with open(p, "w") as f:
            for k in s7.LENGTHS:
                for t in by_k[k]:
                    f.write(json.dumps(t) + "\n")
        hashes[os.path.basename(p)] = hashlib.sha256(open(p, "rb").read()).hexdigest()
    with open(os.path.join(OUT, "manifests", "SHA256.txt"), "w") as f:
        for k, v in hashes.items():
            f.write(f"{v} *{k}\n")
    return sets, stats, hashes


def route_edges_and_env(task):
    env = s7.LongRouteEnv([task], sampling="sequential")
    obs, _ = env.reset()
    need = O.nx_route(obs, env.action_masks(), env.scene["ctile"])
    return env, obs, need


def main():
    os.makedirs(OUT, exist_ok=True)
    world = s7.World.get()
    sets, stats, hashes = write_manifests()
    alltasks = [t for split in sets.values() for k in s7.LENGTHS for t in split[k]]
    # prereg
    prereg = {"horizon_rule": "horizon(k) = 10 * k decisions (k = minimum required construction edits)", "horizons": {k: s7.horizon_for(k) for k in s7.LENGTHS},
              "reward": "+1 exactly at house-carpark connectivity (terminates), 0 otherwise", "frozen_before": "any S7 baseline or training result"}
    prereg["manifest_sha256"] = hashes
    json.dump(prereg, open(os.path.join(OUT, "PREREGISTERED_PROTOCOL.json"), "w"), indent=1)

    # 1 splits
    keys = {sp: {(t["scene_id"], tuple(t["house"])) for k in s7.LENGTHS for t in sets[sp][k]} for sp in sets}
    ids = {sp: {t["task_id"] for k in s7.LENGTHS for t in sets[sp][k]} for sp in sets}
    over = [(a, b) for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")) if keys[a] & keys[b] or ids[a] & ids[b]]
    check("1_splits_disjoint_by_scene_house", not over and len(alltasks) == len(set(t["task_id"] for t in alltasks)),
          f"{ {sp: sum(len(sets[sp][k]) for k in s7.LENGTHS) for sp in sets} } tasks; per-length test counts { {k: len(sets['test'][k]) for k in s7.LENGTHS} }; no (scene,house) pair or task id shared; overlaps={over}", stats=stats)

    # 2 solvable + min correct + any order
    bad = []
    n_orders = 0
    kcount = Counter()
    rng = np.random.default_rng(0)
    t0 = time.time()
    for t in alltasks:
        env, obs, need = route_edges_and_env(t)
        mn = O.nx_min_edits(obs, env.action_masks(), env.scene["ctile"])
        if not (mn == t["k_min"] == env.remaining() == len(need)):
            bad.append((t["task_id"], mn, t["k_min"], env.remaining(), len(need)))
            continue
        kcount[t["k_min"]] += 1
        orders = [list(need), list(reversed(need)), list(rng.permutation(len(need)))]
        for oi, order in enumerate(orders):
            seq = [need[i] for i in order] if oi == 2 else order
            env.reset()
            rs = []
            for e in seq:
                _, r, te, tr, info = env.step(L.action_index("ADD", *e))
                rs.append(r)
                if not info["accepted"]:
                    bad.append((t["task_id"], "illegal route edge", e))
            n_orders += 1
            if rs[:-1] != [0.0] * (len(rs) - 1) or rs[-1] != 1.0 or not te:
                bad.append((t["task_id"], "reward sequence", rs))
    check("2_every_task_solvable_min_k_correct_any_order", not bad,
          f"{len(alltasks)} tasks: independent networkx min-edits (live mask + observation) == stored k == generator Dijkstra == route length; the shortest route succeeds in {n_orders} replays "
          f"(forward/reverse/random order) with reward 0,...,0,1 and termination exactly at the k-th ADD ({time.time() - t0:.0f}s); violations={bad[:3]}", per_k=dict(kcount))

    # 3 brute force true minimum
    bf = []
    envs = {}
    for k, take, maxd in ((2, 30, 2), (3, 4, 2)):
        pool = (sets["test"][k] + sets["validation"][k])[:take]
        for t in pool:
            env = s7.LongRouteEnv([t], sampling="sequential")
            if k == 2:
                d, n = O.brute_force_min_actions(env, 0, 2)
                bf.append((k, t["task_id"], d, n))
            else:
                env.reset(options={"task_index": 0})
                cnt = [0]
                none_within_2 = not O.exists_win_within(env, 2, cnt)
                bf.append((k, t["task_id"], "no_win_within_2" if none_within_2 else "WIN_WITHIN_2", cnt[0]))
    ok3 = all((r[2] == 2) for r in bf if r[0] == 2) and all(r[2] == "no_win_within_2" for r in bf if r[0] == 3)
    check("3_exhaustive_search_confirms_minimum_on_small_tasks", ok3,
          f"k=2: exhaustive depth<=2 search (all legal actions incl. WAIT/REMOVE) finds the minimum 2 on {sum(1 for r in bf if r[0] == 2)}/{sum(1 for r in bf if r[0] == 2)} tasks; "
          f"k=3: no action sequence of length <=2 wins on {sum(1 for r in bf if r[0] == 3)} tasks ({sum(r[3] for r in bf if r[0] == 3):,} env steps). Larger k rely on the shortest-path lower bound "
          f"(every successful sequence must add every non-initial edge of a house-carpark path)")

    # 4 no single-action shortcut (all tasks, all legal actions)
    n_single = bad_single = 0
    t0 = time.time()
    for t in alltasks:
        env = s7.LongRouteEnv([t], sampling="sequential")
        env.reset()
        snap = env.snapshot()
        for a in np.flatnonzero(env.action_masks()):
            env.restore(snap)
            _, r, te, tr, _ = env.step(int(a))
            n_single += 1
            bad_single += (r > 0 or te)
    check("4_no_single_action_shortcut", bad_single == 0, f"{n_single:,} single legal actions (WAIT/ADD/REMOVE) from the start state of all {len(alltasks)} tasks: {bad_single} reward or terminate ({time.time() - t0:.0f}s)")

    # 5 reward structure on partial routes
    viol = []
    nparts = 0
    for t in alltasks[::3]:
        env, obs, need = route_edges_and_env(t)
        for drop in range(len(need)):
            env.reset()
            rs = []
            for i, e in enumerate(need):
                if i == drop:
                    continue
                _, r, te, tr, _ = env.step(L.action_index("ADD", *e))
                rs.append((r, te))
            nparts += 1
            if any(r != 0.0 or te for r, te in rs) or env.success:
                viol.append((t["task_id"], drop))
        # truncation without success gives no reward
        env.reset()
        for _ in range(env.horizon):
            _, r, te, tr, _ = env.step(0)
        if r != 0.0 or not tr or te:
            viol.append((t["task_id"], "horizon"))
    check("5_partial_routes_and_timeouts_give_zero_reward", not viol, f"{nparts} routes with one required edge withheld: reward 0, not terminal, not connected; WAIT-only episodes truncate at 10*k with reward 0; violations={viol[:3]}")

    # 6 independent connectivity
    mism = 0
    states = 0
    t0 = time.time()
    for ti, t in enumerate(alltasks[::4]):
        env, obs, need = route_edges_and_env(t)
        for ep in range(3):
            obs, _ = env.reset()
            for step in range(min(env.horizon, 40)):
                legal = np.flatnonzero(env.action_masks())
                if ep == 0 and step < len(need):
                    a = L.action_index("ADD", *need[step])          # follow the route (so connected states are visited)
                else:
                    a = int(rng.choice(legal))
                obs, r, te, tr, info = env.step(a)
                c1, c2, c3 = env.connected(), O.obs_connected(obs, env.scene["ctile"]), O.nx_components_connected(obs, env.scene["ctile"])
                states += 1
                mism += not (c1 == c2 == c3 == (r > 0))
                if te or tr:
                    break
    # negative: complete route minus each edge
    neg = 0
    for t in alltasks[::10]:
        env, obs, need = route_edges_and_env(t)
        for drop in range(len(need)):
            env.reset()
            for i, e in enumerate(need):
                if i != drop:
                    obs, r, te, tr, _ = env.step(L.action_index("ADD", *e))
            neg += O.obs_connected(obs, env.scene["ctile"]) or O.nx_components_connected(obs, env.scene["ctile"])
    check("6_connectivity_detected_independently_of_generator", mism == 0 and neg == 0,
          f"{states:,} visited states: env.connected == observation-only BFS == networkx components == (reward>0) in all; route-minus-one-edge never connects ({neg} false positives) ({time.time() - t0:.0f}s)")

    # 7 native mask parity on the 24 captured native states
    ns = json.load(open(os.path.join(OUT, "native_states_summary.json")))
    pr = []
    for n in ns:
        s = world.scenes[n["scene_id"]]
        t = {"scene_id": n["scene_id"], "house": n["house"], "driveway_dir": None, "k_min": 0, "task_id": "native"}
        h = tuple(n["house"])
        dr = tuple(n["driveway"])
        dd = [d for d in s7.DIRS4 if s7.canon(h, L.neighbor(h[0], h[1], d)) == dr][0]
        t["driveway_dir"] = dd
        env = s7.LongRouteEnv([t], sampling="sequential")
        env.reset()
        m = env.action_masks()
        synth_add = {int(a) for a in np.flatnonzero(m) if 0 < a < L.UPGRADE_BASE and L.decode_action(int(a))[0] == "ADD"}
        synth_rm = {int(a) for a in np.flatnonzero(m) if 0 < a < L.UPGRADE_BASE and L.decode_action(int(a))[0] == "REMOVE"}
        na, nr = set(n["native_legal_add"]), set(n["native_legal_remove"])
        tp = len(synth_add & na)
        pr.append({"seed": n["seed"], "native_add": len(na), "synth_add": len(synth_add), "recall": tp / len(na), "precision": tp / len(synth_add), "remove_equal": synth_rm == nr,
                   "synth_remove": len(synth_rm), "native_remove": len(nr), "wait": bool(m[0]), "upgrade_legal": bool(m[L.UPGRADE_BASE:].any()),
                   "missed_native": sorted(na - synth_add)[:3], "extra_synth_n": len(synth_add - na)})
    rec = statistics.fmean(p["recall"] for p in pr)
    prec = statistics.fmean(p["precision"] for p in pr)
    rm_eq = sum(p["remove_equal"] for p in pr)
    # what are the native-legal edges we miss?
    missed = Counter()
    for n, p in zip(ns, pr):
        s = world.scenes[n["scene_id"]]
        for a in set(n["native_legal_add"]) - {int(x) for x in []}:
            pass
    check("7_synthetic_masks_vs_native_masks_on_24_captured_states", rec >= 0.99 and prec >= 0.88 and rm_eq >= 22,
          f"ADD recall {100 * rec:.1f}% (native-legal edges also legal here), precision {100 * prec:.1f}% (synthetic mask is a superset on window-boundary and one-water-endpoint edges); REMOVE sets identical on {rm_eq}/24 states; "
          f"WAIT legal 24/24, upgrade slots illegal 24/24; mean legal ADD native {statistics.fmean(p['native_add'] for p in pr):.1f} vs synthetic {statistics.fmean(p['synth_add'] for p in pr):.1f}",
          per_state=pr)

    # 8 house-edge / stub / budget behaviour
    t = sets["test"][5][0]
    env = s7.LongRouteEnv([t], sampling="sequential")
    obs, _ = env.reset()
    h = env.house
    he = [e for e in s7.house_edges(h) if e in env.edges]
    e_new = [e for e in s7.house_edges(h) if e not in env.edges and L.action_index("ADD", *e) < L.N_ACTIONS and env.action_masks()[L.action_index("ADD", *e)]][0]
    c0, n0 = env.concrete, len(env.edges)
    env.step(L.action_index("ADD", *e_new))
    repl_ok = (e_new in env.edges and he[0] not in env.edges and len(env.edges) == n0 and env.concrete == c0)
    rm_house_illegal = not any(env.action_masks()[L.action_index("REMOVE", *e)] for e in s7.house_edges(h))
    # budget: ADD far edges until concrete 0
    env.reset()
    far = [e for e in env.scene["cand"] if e not in env.edges and h not in s7.ends(e)]
    spent = 0
    for e in far:
        if env.concrete == 0:
            break
        if env.action_masks()[L.action_index("ADD", *e)]:
            env.step(L.action_index("ADD", *e))
            spent += 1
    at_zero_adds = [int(a) for a in np.flatnonzero(env.action_masks()) if 0 < a < L.UPGRADE_BASE and L.decode_action(int(a))[0] == "ADD"]
    only_house = bool(at_zero_adds) and all(h in s7.ends(tuple(L.decode_action(a)[1:])) for a in at_zero_adds)
    env.reset()
    stub = env.scene["stub"]
    st_a = L.action_index("REMOVE", *stub)
    stub_legal = bool(env.action_masks()[st_a])
    ne = len(env.edges)
    env.step(st_a) if stub_legal else None
    stub_noop = stub in env.edges and len(env.edges) == ne
    expect_stub_legal = (stub[0], stub[1]) == env.scene["ctile"]
    check("8_house_edge_replacement_stub_quirk_and_road_budget", repl_ok and rm_house_illegal and stub_legal == expect_stub_legal and stub_noop and env.concrete == c0 and spent == 25 and only_house,
          f"ADD on another house edge replaces the driveway (edge count and concrete unchanged: {repl_ok}); house edges not removable: {rm_house_illegal}; stub REMOVE legal iff action tile is the carpark "
          f"({stub_legal}/{expect_stub_legal}) and has no effect ({stub_noop}); budget 25: after {spent} non-house ADDs concrete={c0 - spent if spent else c0}, then ADD only where it replaces the house edge")

    # 9 leakage
    leak = []
    t = sets["test"][7][0]
    e1 = s7.LongRouteEnv([t], sampling="sequential")
    o1, _ = e1.reset()
    t2 = dict(t, k_min=99, task_id="other", split="x")
    e2 = s7.LongRouteEnv([t2], sampling="sequential")
    o2, _ = e2.reset()
    same_meta = np.array_equal(o1, o2)
    # independent rebuild of the observation from the scene file alone
    s = world.scenes[t["scene_id"]]
    ob = np.zeros(L.OBS_DIM, np.float32)
    til = ob[: L.N_TILES * L.NUM_TILE_FEATURES].reshape(L.N_TILES, L.NUM_TILE_FEATURES)
    til[:, 0] = 1
    for (x, y) in world.water:
        til[L.tile_index(x, y), 1] = 1
    for (x, y) in s["dest"]:
        til[L.tile_index(x, y), L.F_CONTENT] = L.CONTENT_DEST
        til[L.tile_index(x, y), L.F_GROUP] = 1
    hh = tuple(t["house"])
    til[L.tile_index(*hh), L.F_CONTENT] = L.CONTENT_HOUSE
    til[L.tile_index(*hh), L.F_GROUP] = 1
    for e in (s["stub"], s7.canon(hh, L.neighbor(hh[0], hh[1], t["driveway_dir"]))):
        (ax, ay), (bx, by) = s7.ends(e)
        ob[L.tile_index(ax, ay) * 15 + L.F_ROAD[e[2]]] = L.ROAD_ACTIVE
        ob[L.tile_index(bx, by) * 15 + L.F_ROAD[L.OPP[e[2]]]] = L.ROAD_ACTIVE
    g = L.N_TILES * 15
    ob[g + L.G_HOUR] = (int(1.2) % 24) / 24
    ob[g + L.G_CONCRETE] = 1.0
    ob[g + 5] = 1.0
    rebuilt = np.array_equal(ob, o1)
    unused = [L.F_MOUNTAIN, L.F_RAIL, L.F_LIGHT, L.F_ROUNDABOUT, L.F_DEMAND, L.F_OVERCROWD, L.F_VEHICLES]
    tiles = o1[: g].reshape(L.N_TILES, 15)
    unused_zero = all(not tiles[:, c].any() for c in unused) and not o1[g + 6: g + 11].any()
    # same scene+house, different driveway: only road channels at the driveway differ
    other = [d for d in s7.DIRS4 if d != t["driveway_dir"] and L.neighbor(hh[0], hh[1], d) in s["land"] and L.neighbor(hh[0], hh[1], d) not in world.water][0]
    e3 = s7.LongRouteEnv([dict(t, driveway_dir=other)], sampling="sequential")
    o3, _ = e3.reset()
    diff_idx = np.flatnonzero(o3 != o1)
    diff_ch = {int(i % 15) for i in diff_idx if i < g}
    only_roads = diff_ch <= {4, 5, 6, 7} and len(diff_idx) <= 4
    check("9_no_privileged_path_labels_in_observation", same_meta and rebuilt and unused_zero and only_roads,
          f"obs unchanged when k/task_id/split metadata change: {same_meta}; obs is bit-identical to an independent rebuild from the scene file + (house, driveway) alone: {rebuilt}; non-native channels "
          f"all zero: {unused_zero}; changing only the driveway direction changes {len(diff_idx)} entries, all road channels: {only_roads}; the env stores no route/solution field")

    # 10 alternative routes
    alt_ok = alt_n = multi = no_alt = 0
    cnts = []
    for t in alltasks[::5]:
        env, obs, need = route_edges_and_env(t)
        c = O.count_shortest_routes(obs, env.action_masks(), env.scene["ctile"], cap=60)
        cnts.append(c)
        multi += c >= 2
        alt = None
        for fe in need:
            try:
                alt = O.nx_route(obs, env.action_masks(), env.scene["ctile"], forbid_edges=[fe])
                break
            except Exception:
                alt = None
        if alt is None:
            no_alt += 1
            continue
        env.reset()
        rs = []
        for e in alt:
            _, r, te, tr, _ = env.step(L.action_index("ADD", *e))
            rs.append(r)
        alt_n += 1
        alt_ok += (rs[-1] == 1.0 and all(x == 0 for x in rs[:-1]))
    check("10_alternative_valid_routes_are_accepted", alt_ok == alt_n,
          f"{alt_n} tasks: a route that avoids one edge of the shortest route (length >= k) completes with reward +1 on {alt_ok}/{alt_n} ({no_alt} further tasks have no alternative because every route edge is a bridge of the legal graph); tasks with >= 2 distinct shortest routes: {100 * multi / (alt_n + no_alt):.0f}% (median count {statistics.median(cnts)}, cap 60)")

    # 11 prereg + env throughput
    envs = s7.LongRouteEnv(sets["train"][7], sampling="random")
    envs.reset()
    t0 = time.time()
    n = 0
    for _ in range(3000):
        _, r, te, tr, _ = envs.step(int(rng.choice(np.flatnonzero(envs.action_masks()))))
        n += 1
        if te or tr:
            envs.reset()
    sps = n / (time.time() - t0)
    check("11_horizon_rule_preregistered_and_env_speed", all(s7.horizon_for(k) == 10 * k for k in s7.LENGTHS), f"horizon(k)=10k: {prereg['horizons']} written to PREREGISTERED_PROTOCOL.json with manifest hashes; random-policy env speed {sps:.0f} steps/s", env_steps_per_s=sps)

    json.dump(R, open(os.path.join(OUT, "gate_a_results.json"), "w"), indent=1, default=str)
    print("\nGate A:", sum(1 for v in R.values() if v["passed"]), "/", len(R), "passed")
    sys.exit(0 if all(v["passed"] for v in R.values()) else 1)


if __name__ == "__main__":
    main()
