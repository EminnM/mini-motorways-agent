"""S13b inference-only baselines, exploit tests and throughput benchmark (NO training, no optimizer).
   python python/synthetic_rl/s13_baselines.py -> research/s13_multi_house/baselines.json, exploits.json, throughput.json"""
import json
import os
import statistics as S
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from synthetic_rl import layout as L  # noqa: E402
from synthetic_rl import s13_env as E  # noqa: E402
from synthetic_rl import s13_controller as CT  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "research", "s13_multi_house")
CK = os.path.join(ROOT, "research", "s8_long_route_training", "runs", "arm_C_s11", "checkpoints", "step_0099840.zip")
SC = E.load_scenarios()
ENV_SEEDS = 3


def episode(policy, idx, seed, env=None):
    env = env or E.MultiHouseEnv(SC, seed=seed)
    env._rng = np.random.default_rng(seed)
    env.reset(options={"scenario_index": idx})
    prev_conn, ever, first_conn, spawn_t = set(), {}, {}, {}
    destroyed = zero_c = plain_adds = removes = swaps = 0
    ret, done = 0.0, False
    conn_curve = []
    while not done:
        a = policy(env)
        dec = L.decode_action(a)
        ok_before = bool(env.action_masks()[a])
        if ok_before and dec[0] == "ADD":
            x = (dec[1], dec[2], dec[3])
            ends_ = E.ends(x)
            if any(env.content.get(p) == "House" for p in ends_):
                swaps += 1
            else:
                plain_adds += 1
        elif ok_before and dec[0] == "REMOVE":
            removes += 1
        _, r, te, tr, inf = env.step(a)
        done = te or tr
        ret += r
        conn = env.connected_houses()
        destroyed += len([h for h in prev_conn if h in env.houses and h not in conn])
        for h in env.houses:
            spawn_t.setdefault(h, env.t)
            if h in conn:
                first_conn.setdefault(h, env.t)
        prev_conn = conn
        zero_c += int(env.concrete == 0)
        conn_curve.append((len(conn), len(env.houses)))
    n = env.t
    new = [h for h, t0 in spawn_t.items() if t0 > 1 and t0 + 60 < n + 1]
    ok60 = sum(1 for h in new if h in first_conn and first_conn[h] - spawn_t[h] <= 60)
    quart = [np.mean([c / max(1, hh) for c, hh in conn_curve[int(len(conn_curve) * q0):int(len(conn_curve) * (q0 + .25))] or [(0, 1)]]) for q0 in (0, .25, .5, .75)]
    return {"scenario": SC[idx]["id"], "split": SC[idx]["split"], "return": ret, "deliveries": env.deliveries_total, "decisions": n, "survived_600": bool(not env.game_over), "game_over": bool(env.game_over),
            "houses": len(env.houses), "connected_end": len(prev_conn), "frac_connected_by_quarter": [round(float(q), 3) for q in quart], "new_houses_eligible": len(new), "new_connected_60": ok60,
            "frac_zero_concrete": zero_c / n, "connections_destroyed": destroyed, "plain_adds": plain_adds, "swaps": swaps, "removes": removes, "deliveries_per_plain_add": env.deliveries_total / max(1, plain_adds)}


def summarize(rows):
    def med(k):
        return round(float(np.median([r[k] for r in rows])), 3)
    elig = sum(r["new_houses_eligible"] for r in rows)
    return {"episodes": len(rows), "return_median": med("return"), "return_mean": round(float(np.mean([r["return"] for r in rows])), 2), "deliveries_median": med("deliveries"), "decisions_median": med("decisions"),
            "survived_600": sum(r["survived_600"] for r in rows), "houses_median": med("houses"), "connected_end_median": med("connected_end"),
            "frac_connected_by_quarter_mean": [round(float(np.mean([r["frac_connected_by_quarter"][q] for r in rows])), 3) for q in range(4)],
            "new_houses_connected_within_60": [sum(r["new_connected_60"] for r in rows), elig], "frac_zero_concrete_median": med("frac_zero_concrete"), "connections_destroyed_median": med("connections_destroyed"),
            "connections_destroyed_total": sum(r["connections_destroyed"] for r in rows), "plain_adds_median": med("plain_adds"), "deliveries_per_plain_add_median": med("deliveries_per_plain_add")}


def run(name, policy_factory, scen_idx, rows_out):
    pol = policy_factory()
    rows = []
    for idx in scen_idx:
        for s in range(ENV_SEEDS):
            rows.append(episode(pol, idx, 7000 + 100 * idx + s))
    rows_out[name] = {"all": summarize(rows), "test_split": summarize([r for r in rows if r["split"] == "test"]), "train_split": summarize([r for r in rows if r["split"] == "train"]), "episodes": rows}
    print(name, json.dumps(rows_out[name]["all"]), flush=True)


def main():
    import torch
    torch.set_num_threads(1)
    from sb3_contrib import MaskablePPO
    model = MaskablePPO.load(CK, device="cpu")
    rng = np.random.default_rng(123)
    idxs = list(range(len(SC)))
    res = {}
    run("WAIT", lambda: (lambda env: 0), idxs, res)
    run("RANDOM_uniform_over_legal", lambda: (lambda env: int(rng.choice(np.flatnonzero(env.action_masks())))), idxs, res)

    def s8c():
        torch.manual_seed(8008)
        return lambda env: int(model.predict(env._obs, action_masks=env.action_masks(), deterministic=False)[0])
    t0 = time.time()
    run("S8C_frozen_stochastic", s8c, idxs, res)
    res["S8C_frozen_stochastic"]["wall_s"] = round(time.time() - t0, 1)
    run("SCRIPTED_reference_controller", lambda: CT.act, idxs, res)
    json.dump({k: {kk: vv for kk, vv in v.items() if kk != "episodes"} for k, v in res.items()}, open(os.path.join(OUT, "baselines.json"), "w"), indent=1)
    json.dump({k: v["episodes"] for k, v in res.items()}, open(os.path.join(OUT, "baselines_episodes.json"), "w"))

    # ---------------- exploit tests: policies that try to score without serving the growing network
    ex = {}

    def first_house_only():
        def pol(env):
            first = min(env.houses, key=lambda h: env.houses[h]["t"]) if env.houses else None
            if first is None or first in env.connected_houses():
                return 0
            p = CT._plan(env, first)
            if p is None:
                return 0
            for u, v in zip(p[:-1], p[1:]):
                e = E.canon(u, v)
                if env.edges.get(e) != "A":
                    a = L.action_index("ADD", *e)
                    return a if env.action_masks()[a] else 0
            return 0
        return pol

    def endless_construction():
        r2 = np.random.default_rng(5)

        def pol(env):
            m = env.action_masks().copy()
            m[0] = False
            leg = np.flatnonzero(m)
            return int(r2.choice(leg)) if len(leg) else 0
        return pol

    def swap_churn():
        r3 = np.random.default_rng(6)

        def pol(env):
            hs = list(env.houses)
            if not hs:
                return 0
            h = hs[int(r3.integers(len(hs)))]
            acts = [L.action_index("ADD", *e) for e in env._house_edges(h) if e[2] in ("E", "S") and env.action_masks()[L.action_index("ADD", *e)]]
            return int(r3.choice(acts)) if acts else 0
        return pol

    for name, fac in (("first_house_only_then_WAIT", first_house_only), ("endless_random_construction_no_WAIT", endless_construction), ("house_edge_swap_churn", swap_churn), ("full_controller", lambda: CT.act), ("WAIT", lambda: (lambda env: 0))):
        rows = [episode(fac(), idx, 9100 + idx) for idx in idxs]
        ex[name] = summarize(rows)
        print("EXPLOIT", name, json.dumps(ex[name]), flush=True)
    json.dump(ex, open(os.path.join(OUT, "exploits.json"), "w"), indent=1)

    # ---------------- throughput (no optimizer): environment-only and env + frozen S8-C inference
    env = E.MultiHouseEnv(SC, seed=1)
    rr = np.random.default_rng(9)
    env.reset()
    n, t0 = 0, time.time()
    while n < 20000:
        a = int(rr.choice(np.flatnonzero(env.action_masks())))
        _, r, te, tr, _ = env.step(a)
        n += 1
        if te or tr:
            env.reset()
    env_sps = n / (time.time() - t0)
    env.reset()
    n, t0 = 0, time.time()
    while n < 4000:
        a = int(model.predict(env._obs, action_masks=env.action_masks(), deterministic=False)[0])
        _, r, te, tr, _ = env.step(a)
        n += 1
        if te or tr:
            env.reset()
    inf_sps = n / (time.time() - t0)
    # batched inference-only speed of the frozen policy (what a rollout of 512 steps would need), no gradients
    obs = np.stack([env._obs] * 64)
    masks = np.stack([env.action_masks()] * 64)
    t0 = time.time()
    for _ in range(20):
        model.predict(obs, action_masks=masks, deterministic=False)
    batch_sps = 20 * 64 / (time.time() - t0)
    tp = {"env_steps_per_s_random_policy": round(env_sps, 1), "env_plus_frozen_S8C_inference_steps_per_s": round(inf_sps, 1), "batched_inference_only_obs_per_s": round(batch_sps, 1), "threads": 1,
          "projected_rollout_wall_for_100k_decisions_s": round(100000 / inf_sps, 0), "note": "optimizer time (4 epochs x 512 samples / batch 256 per update) NOT included;"}
    json.dump(tp, open(os.path.join(OUT, "throughput.json"), "w"), indent=1)
    print("THROUGHPUT", tp)


if __name__ == "__main__":
    main()
