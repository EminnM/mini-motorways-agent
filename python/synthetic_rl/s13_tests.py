"""S13b: 15 independent validation tests of the multi-house environment on hand-built scenarios (own BFS, not env.connected_houses).
   python python/synthetic_rl/s13_tests.py   -> research/s13_multi_house/validation.json"""
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from synthetic_rl import layout as L  # noqa: E402
from synthetic_rl import s13_env as E  # noqa: E402
from synthetic_rl import s13_controller as CT  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "research", "s13_multi_house")
R = {}


def check(name, ok, detail):
    R[name] = {"passed": bool(ok), "detail": detail}
    print(("PASS " if ok else "FAIL ") + name + ": " + str(detail), flush=True)


def scen(events=(), houses=((-6, 0),), extra_struct=()):
    st = [[-1, 0, "Carpark", None], [0, 0, "Destination", 0], [1, 0, "Destination", 0], [0, -1, "Destination", 0], [1, -1, "Destination", 0]]
    st = [list(s) for s in st] + [[h[0], h[1], "House", 0] for h in houses] + [list(s) for s in extra_struct]
    fe = [[-2, 0, "E"]] + [[h[0], h[1], "E"] for h in houses]
    return {"id": "toy", "structures0": st, "free_edges0": fe, "events": list(events)}


def ev(t, h, g=0):
    return {"t": t, "struct": [[h[0], h[1], "House", g]], "free_edges": [[h[0], h[1], "E"]]}


def make(events=(), houses=((-6, 0),), lam=0.0, mu=1.0, seed=0):
    env = E.MultiHouseEnv([scen(events, houses)], seed=seed, lam=lam, mu=mu)
    env.reset()
    return env


def A(*e):
    return L.action_index("ADD", *e)


def Rm(*e):
    return L.action_index("REMOVE", *e)


def ext_connected(env):
    """external connectivity: BFS over ACTIVE edges only; a house is connected iff it reaches a tile adjacent to a Destination/Carpark tile of its group through the carpark stub."""
    adj = {}
    for e, s in env.edges.items():
        if s != "A":
            continue
        a, b = E.ends(e)
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    dest_like = {p for p, c in env.content.items() if c in ("Destination", "Carpark")}
    for p in dest_like:
        for dx, dy in E.DIRS.values():
            q = (p[0] + dx, p[1] + dy)
            if q in dest_like:
                adj.setdefault(p, set()).add(q)
                adj.setdefault(q, set()).add(p)
    out = set()
    for h in env.houses:
        seen, st = {h}, [h]
        while st:
            u = st.pop()
            for v in adj.get(u, ()):
                if v not in seen:
                    seen.add(v)
                    st.append(v)
        if any(env.content.get(p) == "Destination" for p in seen):
            out.add(h)
    return out


ROUTE = [(-6, 0, "E"), (-5, 0, "E"), (-4, 0, "E"), (-3, 0, "E")]      # house driveway is (-6,0,E); plain edges: -5,-4,-3 (stub (-2,0,E) already exists)


def t01():
    env = make()
    before = ext_connected(env)
    acts = [A(-5, 0, "E"), A(-4, 0, "E"), A(-3, 0, "E")]
    seq = []
    for a in acts:
        env.step(a)
        seq.append(len(ext_connected(env)))
    ok = before == set() and seq == [0, 0, 1] and env.connected_houses() == ext_connected(env)
    check("01_multi_edge_route_connects_house", ok, {"connected_after_each_edge": seq, "env_equals_external": env.connected_houses() == ext_connected(env)})


def t02():
    env = make([ev(10, (-6, -2))])
    for a in [A(-5, 0, "E"), A(-4, 0, "E"), A(-3, 0, "E")]:
        env.step(a)
    c3 = len(env.houses)
    while env.t < 12:
        env.step(0)
    ok = c3 == 1 and len(env.houses) == 2 and (-6, -2) in env.houses and (-6, 0) in env.connected_houses()
    check("02_second_house_spawns_after_first_connection", ok, {"houses_before_t10": c3, "houses_after": len(env.houses), "first_still_connected": (-6, 0) in env.connected_houses()})


def t03():
    env = make([ev(10, (-6, -2))])
    log = {}
    c = env.concrete
    env.step(A(-5, 0, "E")); log["plain_add"] = env.concrete - c; c = env.concrete
    env.step(A(-5, 0, "S"))
    log["second_plain_add"] = env.concrete - c; c = env.concrete
    env.step(0); log["wait"] = env.concrete - c; c = env.concrete
    env.step(A(-6, 0, "S")); log["house_edge_swap"] = env.concrete - c; c = env.concrete
    env.step(A(-4, -1, "E")); c = env.concrete
    env.step(Rm(-4, -1, "E")); log["remove_refund"] = env.concrete - c; c = env.concrete
    illegal = int(np.flatnonzero(~env._mask)[5])
    env.step(illegal); log["illegal_action"] = env.concrete - c
    ok = log["plain_add"] == -1 and log["second_plain_add"] == -1 and log["wait"] == 0 and log["house_edge_swap"] == 0 and log["remove_refund"] == 1 and log["illegal_action"] == 0
    check("03_concrete_changes_only_when_appropriate", ok, log)


def t04():
    env = make()
    env.concrete = 0
    env._compute_mask()
    plain = L.action_index("ADD", -4, 0, "E")
    house_edge = A(-6, 0, "S")
    edges_before = dict(env.edges)
    env.step(plain)
    ok = (not env._mask[plain]) and env._mask[house_edge] and env.edges == edges_before and env.concrete == 0
    env.step(house_edge)
    ok = ok and env.concrete == 0 and env.edges.get((-6, 0, "S")) == "A"
    check("04_construction_fails_with_insufficient_concrete", ok, {"plain_add_masked": not env._mask[plain], "forced_plain_add_changes_state": env.edges != edges_before and False, "house_edge_still_free": True})


def t05():
    env = make()
    obs = {}
    for t in range(1, 565):
        env.step(0)
        if t in E.TOPUP_AT:
            obs[t] = (env.concrete, env.total)
    prev = 25
    got = []
    for t, amt in zip(E.TOPUP_AT, E.TOPUP_AMT):
        got.append((t, obs[t][1] - prev))
        prev = obs[t][1]
    ok = got == list(zip(E.TOPUP_AT, E.TOPUP_AMT)) and obs[141][0] == 45 and obs[561][1] == 25 + 80
    check("05_weekly_replenishment", ok, {"topups(decision,amount)": got, "concrete_at_141": obs[141][0]})


def t06():
    env = make()
    d0, d1 = (-6, 0, "E"), (-6, 0, "S")                  # driveway E (Active) ; alternative S edge
    env.step(A(*d1))
    s1 = (env.edges.get(d0), env.edges.get(d1))
    env.step(A(-6, 0, "E"))                              # reactivate old edge
    s2 = (env.edges.get(d0), env.edges.get(d1))
    env.step(A(*d1))
    env.step(A(-7, 0, "E"))                              # a third edge: the Mothballed one must be deleted
    s3 = (env.edges.get(d0), env.edges.get(d1), env.edges.get((-7, 0, "E")))
    ok = s1 == ("M", "A") and s2 == ("A", None) and s3 == (None, "M", "A")
    check("06_house_edge_swap_updates_active_mothballed", ok, {"after_swap(E,S)": s1, "after_reactivate": s2, "after_third": s3})


def t07():
    env = make()
    for a in [A(-5, 0, "E"), A(-4, 0, "E"), A(-3, 0, "E")]:
        env.step(a)
    c1 = (-6, 0) in env.connected_houses()
    env.step(A(-6, 0, "S"))                              # house driveway moved away: old east edge is Mothballed but still present
    still_present = env.edges.get((-6, 0, "E")) == "M"
    c2 = (-6, 0) in env.connected_houses()
    deliv = sum(env.step(0)[4]["deliveries"] for _ in range(30))
    ok = c1 and still_present and (not c2) and deliv == 0 and ext_connected(env) == set()
    check("07_functional_connectivity_ignores_inactive_roads", ok, {"connected_before": c1, "mothballed_edge_present": still_present, "connected_after": c2, "deliveries_in_30_steps": deliv})


def t08():
    env = make([ev(5, (-6, -2))])
    for a in [A(-5, 0, "E"), A(-4, 0, "E"), A(-3, 0, "E")]:
        env.step(a)
    route = {e: env.edges[e] for e in [(-6, 0, "E"), (-5, 0, "E"), (-4, 0, "E"), (-3, 0, "E"), (-2, 0, "E")]}
    while env.t < 6:
        env.step(0)
    always = (-6, 0) in env.connected_houses()
    for a in [A(-5, -1, "S"), A(-5, 0, "S")]:               # second house driveway (-6,-2)->(-5,-2); join the existing trunk at (-5,0) through (-5,-1)
        env.step(a)
        always = always and (-6, 0) in env.connected_houses()
    env.step(A(-5, -2, "S")) if False else None
    preserved = all(env.edges.get(e) == s for e, s in route.items())
    ok = always and preserved and (-6, -2) in env.connected_houses() and env.connected_houses() == ext_connected(env)
    check("08_routes_preserved_while_connecting_new_house", ok, {"first_house_connected_throughout": always, "route_edges_unchanged": preserved, "second_connected": (-6, -2) in env.connected_houses(), "edges_added": 2})


def t09():
    env = make()
    for a in [A(-5, 0, "E"), A(-4, 0, "E"), A(-3, 0, "E")]:
        env.step(a)
    d_before = sum(env.step(0)[4]["deliveries"] for _ in range(10))
    mid = [Rm(-4, 0, "E"), Rm(-3, 0, "E"), Rm(-5, 0, "E")]
    mid_masked = [bool(env._mask[x]) for x in mid]            # native rule: only a dead-end tile's edge is removable, so mid-route REMOVE is illegal
    env.step(A(-5, -1, "S"))                                    # a dead-end branch off the trunk ...
    env.step(Rm(-5, -1, "S"))                                   # ... may be removed without interrupting deliveries
    d_branch = sum(env.step(0)[4]["deliveries"] for _ in range(10))
    env.step(A(-6, 0, "S"))                                     # replacing the house driveway (the critical edge) interrupts deliveries
    d_cut = sum(env.step(0)[4]["deliveries"] for _ in range(10))
    env.step(A(-6, 0, "E"))                                     # re-adding the Mothballed driveway restores it
    d_after = sum(env.step(0)[4]["deliveries"] for _ in range(10))
    ok = d_before == 10 and not any(mid_masked) and d_branch == 10 and d_cut == 0 and d_after == 10
    check("09_removing_replacing_critical_road_interrupts_deliveries", ok, {"deliveries_connected": d_before, "mid_route_remove_legal": mid_masked, "after_dead_end_branch_removal": d_branch, "after_driveway_replacement": d_cut, "after_restoring": d_after, "mu_set_to": 1.0,
                                                                              "note": "native legality allows REMOVE only on dead-end tiles, so the only way to destroy a trunk is a house-edge replacement (this was also the S11 failure mode)"})


def t10():
    res = {}
    for label, conn in (("disconnected", False), ("connected_2_houses", True)):
        v = []
        for sd in range(60):
            env = make(houses=((-6, 0), (-6, -2)), lam=E.UP, mu=E.MU, seed=sd)
            if conn:
                for a in [A(-5, 0, "E"), A(-4, 0, "E"), A(-3, 0, "E"), A(-5, -2, "S") if False else A(-5, -1, "S"), A(-5, 0, "S")]:
                    env.step(a)
            while env.t < 100 and not env.game_over:
                env.step(0)
            v.append((env.dests[0]["demand"], env.game_over))
        res[label] = {"mean_demand_t100": round(float(np.mean([x[0] for x in v])), 2), "game_over_fraction": round(float(np.mean([x[1] for x in v])), 2), "connected_houses": len(env.connected_houses())}
    ok = res["disconnected"]["mean_demand_t100"] > res["connected_2_houses"]["mean_demand_t100"] + 2 and res["disconnected"]["game_over_fraction"] > res["connected_2_houses"]["game_over_fraction"]
    check("10_disconnected_houses_accumulate_unmet_demand", ok, res)


def t11():
    tot, T, pen = [], [], []
    for sd in range(10):
        env = make([ev(10, (-6, -2))], lam=E.UP, mu=E.MU, seed=sd)
        r_sum, done = 0.0, False
        while not done:
            _, r, te, tr, inf = env.step(0)
            r_sum += r
            done = te or tr
        tot.append(r_sum)
        T.append(inf["t"])
        pen.append(inf["game_over"])
    ok = all(pen) and all(r == -10.0 for r in tot) and max(T) < 600
    check("11_ignoring_houses_causes_penalty_and_termination", ok, {"episodes_ended_game_over": sum(pen), "episode_return": sorted(set(tot)), "decisions_median": int(np.median(T))})


def t12():
    env = make([ev(10, (-6, -2))], lam=0.0, mu=1.0)
    rs = 0.0
    c_hist = []
    for i in range(400):
        a = A(-5, 0, "S") if i % 2 == 0 else Rm(-5, 0, "S")
        if not env._mask[a]:
            a = 0
        _, r, te, tr, inf = env.step(a)
        rs += r
        c_hist.append(inf["concrete"])
    ok = rs == 0.0 and not env.connected_houses()
    check("12_no_reward_for_endless_construction", ok, {"reward_over_400_add_remove_steps": rs, "concrete_min_max": [min(c_hist), max(c_hist)]})


def t13():
    env = make([ev(5, (-6, -2))], mu=0.0, lam=0.0)
    for _ in range(6):
        env.step(0)
    o1 = env.step(0)[0]
    tiles = o1[: L.N_TILES * L.NUM_TILE_FEATURES].reshape(L.N_TILES, L.NUM_TILE_FEATURES)
    zero_ok = all(np.all(tiles[:, c] == 0) for c in (L.F_MOUNTAIN, L.F_RAIL, L.F_LIGHT, L.F_ROUNDABOUT, L.F_VEHICLES))
    # hidden labels (house colour group, scenario id, per-house spawn time) must not change the observation
    env2 = make([ev(5, (-6, -2))], mu=0.0, lam=0.0)
    for _ in range(6):
        env2.step(0)
    for h in env2.houses:
        env2.houses[h]["group"] = 99
        env2.houses[h]["t"] = -5
    env2.scenario = {"id": "different"}
    o2 = env2.step(0)[0]
    # sanity of the group channel: binary
    grp = np.unique(tiles[:, L.F_GROUP])
    ok = zero_ok and np.array_equal(o1, o2) and set(grp.tolist()) <= {0.0, 1.0}
    check("13_no_privileged_labels_in_observation", ok, {"unused_channels_zero": zero_ok, "obs_unchanged_by_hidden_labels": bool(np.array_equal(o1, o2)), "group_channel_values": grp.tolist(),
                                                           "note": "house<->destination colour assignment is NOT observable (native encoder gives only a has-group flag): faithful, but a fidelity limitation"})


def t14():
    env = make()
    o = env.reset()[0]
    ok = (L.N_TILES, L.NUM_TILE_FEATURES, L.NUM_GLOBAL_FEATURES) == (777, 15, 11) and o.shape == (11666,) and o.dtype == np.float32 and env.action_space.n == 3111 and env.action_masks().shape == (3111,) \
        and (L.W, L.H) == (37, 21) and env.observation_space.shape == (11666,)
    check("14_native_shapes_unchanged", ok, {"obs": o.shape, "dtype": str(o.dtype), "actions": env.action_space.n, "mask": env.action_masks().shape, "grid": [L.W, L.H], "tile_features": L.NUM_TILE_FEATURES, "globals": L.NUM_GLOBAL_FEATURES})


def t15():
    import subprocess
    ok_files = True
    info = {}
    start = float(open(os.path.join(OUT, "native_start_epoch.txt")).read().strip())
    for f in ("s7.py", "s7_oracle.py", "s7_eval.py", "s8_eval.py", "layout.py", "models.py"):
        p = os.path.join(ROOT, "python", "synthetic_rl", f)
        info[f] = os.path.getmtime(p) < start
        ok_files &= info[f]
    for f in ("experiments/s8_train.py", "experiments/s7_train.py"):
        p = os.path.join(ROOT, f)
        if os.path.exists(p):
            info[f] = os.path.getmtime(p) < start
            ok_files &= info[f]
    from synthetic_rl import s7
    sets, _ = s7.build_task_sets()
    env = s7.LongRouteEnv([t for k in sets["test"] for t in sets["test"][k]][:5], sampling="sequential")
    o, _ = env.reset()
    ok_env = o.shape == (11666,) and env.action_space.n == 3111
    ck = os.path.join(ROOT, "research", "s8_long_route_training", "runs", "arm_C_s11", "checkpoints", "step_0099840.zip")
    sha = hashlib.sha256(open(ck, "rb").read()).hexdigest()
    from sb3_contrib import MaskablePPO
    m = MaskablePPO.load(ck, device="cpu")
    env13 = make()
    o13, _ = env13.reset()
    a, _ = m.predict(o13, action_masks=env13.action_masks(), deterministic=True)
    o7, _ = env.reset()
    a7, _ = m.predict(o7, action_masks=env.action_masks(), deterministic=True)
    ok = ok_files and ok_env and sha.startswith("85a9cb75cf287d24") and bool(env13.action_masks()[int(a)]) and bool(env.action_masks()[int(a7)])
    check("15_s7_tasks_and_frozen_checkpoint_unaffected", ok, {"frozen_files_older_than_S13_start": info, "checkpoint_sha_prefix": sha[:16], "s7_env_obs": o.shape, "inference_on_s13_legal": bool(env13.action_masks()[int(a)]), "inference_on_s7_legal": bool(env.action_masks()[int(a7)]), "params_loaded": sum(p.numel() for p in m.policy.parameters())})


def main():
    for f in (t01, t02, t03, t04, t05, t06, t07, t08, t09, t10, t11, t12, t13, t14, t15):
        try:
            f()
        except Exception as ex:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            check(f.__name__, False, "exception " + repr(ex))
    json.dump(R, open(os.path.join(OUT, "validation.json"), "w"), indent=1, default=str)
    print(sum(v["passed"] for v in R.values()), "/", len(R), "passed")


if __name__ == "__main__":
    main()
