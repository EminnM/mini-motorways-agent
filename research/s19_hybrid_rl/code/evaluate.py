"""S19 evaluation on the synthetic MultiHouseEnv: all 36 scenarios x 3 env seeds (env seeds 7000+100*idx+s, identical to S13 / S18-H), with the
HELD-OUT TEST split (game seeds 515838/539595/563352/587109; 12 scenarios x 3 = 36 episodes) as the primary table.
Policies: WAIT | A greedy planner (S18-H) | B resource-aware planner (S18-H) | C learned (selected checkpoint; stochastic and greedy/argmax) |
          D random strategic (uniform over legal candidates through the same executor).
Metrics: episode_ext() is a VERBATIM copy of s13_baselines.episode() accounting (deliveries, survival, new-house-within-60, destroyed connections,
plain ADDs, zero-concrete fraction, ...) extended with: new-house connection latency, WAIT-primitive fraction, illegal primitive actions
(env mask), and the strategic controller summary (WAIT frequency, candidate-kind counts, mean duration, abandoned plans).  summarize() is
s13_baselines.summarize() plus the extensions.  Paired per-episode comparisons C-A, C-B, C-D with bootstrap 95 % CIs; per-game-seed medians.
python research/s19_hybrid_rl/code/evaluate.py [checkpoint] -> research/s19_hybrid_rl/evaluation.json, evaluation_episodes.json"""
import json
import os
import sys
import time
from typing import Any, Callable, Dict, List

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import strategic_env as S  # noqa: E402
from policy import CandidatePolicy  # noqa: E402
from executor import Executor  # noqa: E402
from planner import Weights  # noqa: E402
from synthetic_rl import layout as L  # noqa: E402
from synthetic_rl import s13_baselines as B  # noqa: E402
from synthetic_rl import s13_env as E  # noqa: E402

OUT_DIR = os.path.dirname(HERE)
RUN_DIR = os.path.join(OUT_DIR, "runs", "s19")
SC = S.SC
ENV_SEEDS = 3


def episode_ext(policy: Callable, idx: int, seed: int) -> Dict[str, Any]:
    """s13_baselines.episode() accounting (copied verbatim) + latency / WAIT-primitive fraction / illegal-primitive count."""
    env = E.MultiHouseEnv(SC, seed=seed)
    env._rng = np.random.default_rng(seed)
    env.reset(options={"scenario_index": idx})
    prev_conn, first_conn, spawn_t = set(), {}, {}
    destroyed = zero_c = plain_adds = removes = swaps = 0
    wait_prim = illegal_prim = 0
    ret, done = 0.0, False
    conn_curve = []
    while not done:
        a = policy(env)
        dec = L.decode_action(a)
        ok_before = bool(env.action_masks()[a])
        illegal_prim += int(not ok_before)
        wait_prim += int(a == 0)
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
    lat = [first_conn[h] - spawn_t[h] for h in new if h in first_conn]
    quart = [np.mean([c / max(1, hh) for c, hh in conn_curve[int(len(conn_curve) * q0):int(len(conn_curve) * (q0 + .25))] or [(0, 1)]]) for q0 in (0, .25, .5, .75)]
    return {"scenario": SC[idx]["id"], "split": SC[idx]["split"], "split3": S.split3(idx), "game_seed": S.game_seed(idx), "env_seed": seed, "return": ret, "deliveries": env.deliveries_total, "decisions": n,
            "survived_600": bool(not env.game_over), "game_over": bool(env.game_over), "houses": len(env.houses), "connected_end": len(prev_conn),
            "frac_connected_by_quarter": [round(float(q), 3) for q in quart], "new_houses_eligible": len(new), "new_connected_60": ok60, "new_connected_any": len(lat),
            "latency_sum": int(sum(lat)), "latency_mean": (float(np.mean(lat)) if lat else None), "frac_zero_concrete": zero_c / n, "connections_destroyed": destroyed,
            "plain_adds": plain_adds, "swaps": swaps, "removes": removes, "deliveries_per_plain_add": env.deliveries_total / max(1, plain_adds), "wait_primitive_frac": wait_prim / n, "illegal_primitive": illegal_prim}


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    out = B.summarize(rows)
    lat = [r["latency_sum"] for r in rows]
    nlat = sum(r["new_connected_any"] for r in rows)
    out["new_house_latency_mean"] = (sum(lat) / nlat) if nlat else None
    out["new_house_rate_60"] = (out["new_houses_connected_within_60"][0] / out["new_houses_connected_within_60"][1]) if out["new_houses_connected_within_60"][1] else None
    out["wait_primitive_frac_mean"] = float(np.mean([r["wait_primitive_frac"] for r in rows]))
    out["illegal_primitive_total"] = int(sum(r["illegal_primitive"] for r in rows))
    out["deliveries_mean"] = float(np.mean([r["deliveries"] for r in rows]))
    out["frac_zero_concrete_mean"] = float(np.mean([r["frac_zero_concrete"] for r in rows]))
    if "strategic_actions" in rows[0]:
        tot = sum(r["strategic_actions"] for r in rows)
        kinds = {k: sum(r["kind_counts"][k] for r in rows) for k in S.KINDS}
        out["wait_frequency"] = kinds["WAIT"] / max(1, tot)
        out["kind_frequency"] = {k: v / max(1, tot) for k, v in kinds.items()}
        out["kind_counts"] = kinds
        out["mean_duration"] = float(np.mean([r["mean_duration"] for r in rows]))
        out["mean_duration_connect"] = float(np.mean([r["mean_duration_connect"] for r in rows if r["kind_counts"]["CONNECT"] > 0] or [0]))
        out["alt_selected_total"] = int(sum(r["alt_selected"] for r in rows))
        out["repeated_keys_total"] = int(sum(r["repeated_keys"] for r in rows))
        out["illegal_selected_total"] = int(sum(r["illegal_selected"] for r in rows))
        out["truncated_candidates_total"] = int(sum(r["truncated_candidates"] for r in rows))
    if "abandoned_plans" in rows[0]:
        out["abandoned_plans_total"] = int(sum(r["abandoned_plans"] for r in rows))
        out["plans_total"] = int(sum(r["plans"] for r in rows))
        out["waits_for_concrete_total"] = int(sum(r["waits_for_concrete"] for r in rows))
        reasons = {}
        for r in rows:
            for k, v in r.get("abandon_reasons", {}).items():
                reasons[k] = reasons.get(k, 0) + v
        out["abandon_reasons"] = reasons
    return out


def load_policy(path: str) -> CandidatePolicy:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    pol = CandidatePolicy(hidden=ck["config"]["hidden"])
    pol.load_state_dict(ck["state_dict"])
    pol.eval()
    return pol


def learned_chooser(pol: CandidatePolicy, deterministic: bool, seed: int = 0):
    gen = torch.Generator().manual_seed(seed)

    def ch(obs, ctrl):
        return pol.act(obs, deterministic=deterministic, gen=gen)[0]
    return ch


def run_policy(name: str, factory: Callable[[], Any], idxs: List[int]) -> List[Dict[str, Any]]:
    """factory() -> (policy callable env->action, stats_fn() -> dict | None); fresh instance per episode."""
    rows = []
    t0 = time.time()
    for idx in idxs:
        for s in range(ENV_SEEDS):
            pol, stats_fn = factory()
            r = episode_ext(pol, idx, S.env_seed(idx, s))
            if stats_fn is not None:
                r.update(stats_fn())
            r["policy"] = name
            rows.append(r)
    print(f"{name}: {len(rows)} episodes in {time.time() - t0:.1f} s; test deliveries median {np.median([r['deliveries'] for r in rows if r['split3'] == 'test']):.1f}", flush=True)
    return rows


def policy_factories(pol: CandidatePolicy, seed: int = 2024) -> Dict[str, Callable]:
    def exec_stats(ex):
        return lambda: {"abandoned_plans": ex.stats["abandoned"], "plans": ex.stats["plans"], "waits_for_concrete": ex.stats["waits_for_concrete"], "abandon_reasons": dict(ex.stats["abandon_reasons"]),
                        "illegal_suppressed": ex.stats["illegal_suppressed"]}

    def A():
        ex = Executor(Weights(greedy=True))
        return ex.act, exec_stats(ex)

    def Bp():
        ex = Executor(Weights())
        return ex.act, exec_stats(ex)

    counter = {"c": 0}

    def strat(chooser_factory):
        def f():
            counter["c"] += 1
            sp = S.StrategicPolicy(chooser_factory(counter["c"]))
            return sp.act, sp.ctrl.summary
        return f
    return {"WAIT": lambda: ((lambda env: 0), None), "A_greedy_planner": A, "B_resource_aware_planner": Bp,
            "C_learned_stochastic": strat(lambda i: learned_chooser(pol, False, seed + i)), "C_learned_greedy": strat(lambda i: learned_chooser(pol, True)),
            "D_random_strategic": strat(lambda i: S.random_chooser(np.random.default_rng(seed + 1000 + i)))}


def bootstrap_diff(x: np.ndarray, y: np.ndarray, n: int = 10000, seed: int = 0) -> Dict[str, float]:
    """paired per-episode differences x - y: mean, median, 95 % bootstrap CIs (paired resampling of episodes)."""
    d = np.asarray(x, float) - np.asarray(y, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    means = d[idx].mean(1)
    meds = np.median(d[idx], 1)
    return {"n": int(len(d)), "mean_diff": float(d.mean()), "mean_ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))], "median_diff": float(np.median(d)),
            "median_ci95": [float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))], "wins": int((d > 0).sum()), "losses": int((d < 0).sum()), "ties": int((d == 0).sum())}


def paired(rows_by_policy: Dict[str, List[Dict]], split: str = "test") -> Dict[str, Any]:
    out = {}
    key = lambda r: (r["scenario"], r["env_seed"])
    for c in ("C_learned_stochastic", "C_learned_greedy"):
        for b in ("A_greedy_planner", "B_resource_aware_planner", "D_random_strategic"):
            rc = {key(r): r for r in rows_by_policy[c] if r["split3"] == split}
            rb = {key(r): r for r in rows_by_policy[b] if r["split3"] == split}
            ks = sorted(set(rc) & set(rb))
            assert len(ks) == len(rc) == len(rb), "unpaired episodes"
            out[f"{c} - {b}"] = {m: bootstrap_diff([rc[k][m] for k in ks], [rb[k][m] for k in ks]) for m in ("deliveries", "return", "survived_600", "new_connected_60", "plain_adds")}
    return out


def per_game_seed(rows_by_policy: Dict[str, List[Dict]], split: str = "test") -> Dict[str, Dict[str, Any]]:
    out = {}
    for p, rows in rows_by_policy.items():
        d = {}
        for r in rows:
            if r["split3"] == split:
                d.setdefault(r["game_seed"], []).append(r)
        out[p] = {str(gs): {"episodes": len(v), "deliveries_median": float(np.median([r["deliveries"] for r in v])), "survived": int(sum(r["survived_600"] for r in v))} for gs, v in sorted(d.items())}
    return out


def main(ckpt: str = None):
    ckpt = ckpt or os.path.join(RUN_DIR, "best.pt")
    pol = load_policy(ckpt)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    idxs = list(range(len(SC)))
    t0 = time.time()
    rows_by = {name: run_policy(name, fac, idxs) for name, fac in policy_factories(pol).items()}
    res = {"checkpoint": os.path.relpath(ckpt, OUT_DIR), "checkpoint_iter": ck.get("iter"), "checkpoint_val": ck.get("val"), "env_seeds": "7000 + 100*idx + s, s in 0..2 (identical to S13 / S18-H)",
           "metric_code": "evaluate.episode_ext = verbatim s13_baselines.episode accounting + latency / WAIT-primitive / illegal-primitive / strategic summary; summarize = s13_baselines.summarize + extensions",
           "splits": {}, "paired_test": paired(rows_by, "test"), "per_game_seed_test": per_game_seed(rows_by, "test"), "wall_s": round(time.time() - t0, 1)}
    for split in ("test", "validation", "train", "all"):
        res["splits"][split] = {p: summarize([r for r in rows if split == "all" or r["split3"] == split]) for p, rows in rows_by.items()}
    json.dump(res, open(os.path.join(OUT_DIR, "evaluation.json"), "w"), indent=1, default=str)
    json.dump(rows_by, open(os.path.join(OUT_DIR, "evaluation_episodes.json"), "w"), default=str)
    for p, s in res["splits"]["test"].items():
        print(p, json.dumps({k: s[k] for k in ("deliveries_median", "deliveries_mean", "survived_600", "new_houses_connected_within_60", "connections_destroyed_total", "plain_adds_median", "deliveries_per_plain_add_median",
                                                "frac_zero_concrete_median", "wait_primitive_frac_mean", "illegal_primitive_total") if k in s} | {k: s[k] for k in ("wait_frequency", "abandoned_plans_total", "mean_duration") if k in s}))
    for k, v in res["paired_test"].items():
        print(k, "deliveries", json.dumps(v["deliveries"]), "survival", json.dumps(v["survived_600"]["mean_diff"]))
    return res


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
