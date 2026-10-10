"""S21 native evaluation runner -- PREPARED, NOT EXECUTED (requires explicit approval). Mirrors experiments/s13_runner.py (same game instance handling, same per-decision logging) with a
policy switch: frozen S8-C stochastic (v1 observation), S18-H planner A (greedy) or B (resource-aware) through the raw-observation adapter. One episode per (policy, seed); budget guard.
   python research/s21_native_check/code/native_eval_runner.py --policy A|B|s8c --idx 0,1,... --deadline-min 10
Seeds: the 12 S9/S13 seeds (research/s9_native_transfer/seeds.json). Nothing here tunes a policy; the planners' weights are frozen as in research/hierarchical_planner."""
import argparse
import gzip
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from native_adapter import NativePlannerPolicy, ROOT  # noqa: E402
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import numpy as np  # noqa: E402
import s11_guard as G  # noqa: E402
from synthetic_rl import layout as L  # noqa: E402
from executor import make_policy  # noqa: E402
from planner import Weights  # noqa: E402

OUT = os.path.join(ROOT, "research", "s21_native_check")
CK = os.path.join(ROOT, "research", "s8_long_route_training", "runs", "arm_C_s11", "checkpoints", "step_0099840.zip")
SEEDS = json.load(open(os.path.join(ROOT, "research", "s9_native_transfer", "seeds.json")))["seeds"]
SAMPLING_SEED_BASE = 7000                          # fresh, independent torch sampling seeds for the S8-C arm (S11/S12 used 1000+i, S13 6000+i)


def run_episode(env, policy_name, policy, seed, deadline):
    import torch
    if policy_name == "s8c":
        torch.manual_seed(SAMPLING_SEED_BASE + SEEDS.index(seed))
    obs, info = env.reset(seed=seed)
    if policy_name != "s8c":
        policy.reset()
    raw = env.last_raw_observation
    steps, te, tr, cum, cut, i = [], False, False, 0.0, False, 0
    prev_conc = raw["globals"]["concreteAvailable"]
    while not (te or tr):
        if time.time() > deadline:
            cut = True
            break
        mask = env.action_masks().copy()
        ctx = G.Ctx(raw)
        func_b = G.edges_from_raw(raw)
        conn_b = G.connected_house_set(ctx, func_b)
        if policy_name == "s8c":
            a = int(policy.predict(obs, action_masks=mask, deterministic=False)[0])
        else:
            a = policy.act(raw, mask)
        legal = bool(mask[a])
        obs, r, te, tr, inf = env.step(a)
        raw = env.last_raw_observation
        ctx_a = G.Ctx(raw)
        func_a = G.edges_from_raw(raw)
        conn_a = G.connected_house_set(ctx_a, func_a)
        g = raw["globals"]
        rc = inf.get("rewardComponents", {})
        cum += float(rc.get("scoreDelta", 0.0))
        lost = sorted(h for h in conn_b if h in ctx_a.house_set and h not in conn_a)
        dec = L.decode_action(a)
        steps.append({"i": i, "kind": dec[0], "action": a, "legal": legal, "concrete_prev": prev_conc, "concrete": g["concreteAvailable"], "concrete_total": g["concreteTotal"], "houses": len(ctx_a.houses), "connected": len(conn_a),
                      "lost": [list(h) for h in lost], "scoreDelta": float(rc.get("scoreDelta", 0.0)), "score_cum": cum, "connected_houses": [list(h) for h in sorted(conn_a)] if conn_a != conn_b else None,
                      "new_houses": [list(h) for h in ctx_a.houses if h not in ctx.house_set], "n_func": len(func_a),
                      "plan": (policy.ex.plan.describe() if policy_name != "s8c" and policy.ex.plan else None), "exec_stats": (dict(policy.ex.stats) if policy_name != "s8c" else None)})
        prev_conc = g["concreteAvailable"]
        i += 1
    end = {"terminated": bool(te), "truncated": bool(tr), "cut": cut, "exec_stats": (dict(policy.ex.stats) if policy_name != "s8c" else None)}
    return steps, end


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True, choices=["A", "B", "s8c"])
    ap.add_argument("--idx", default="0")
    ap.add_argument("--deadline-min", type=float, default=10.0)
    ap.add_argument("--i-have-approval", action="store_true", help="refuses to start the game without this flag (S21: native runs need explicit approval)")
    a = ap.parse_args()
    if not a.i_have_approval:
        raise SystemExit("S21 protocol: native evaluation not approved yet; re-run with --i-have-approval after approval.")
    import torch
    torch.set_num_threads(1)
    from sb3_contrib import MaskablePPO
    import mini_motorways_env as M
    from experiments import gamectl
    policy = MaskablePPO.load(CK, device="cpu") if a.policy == "s8c" else NativePlannerPolicy(make_policy(Weights(greedy=(a.policy == "A"))))
    t0 = time.time()
    os.makedirs(os.path.join(OUT, "raw"), exist_ok=True)
    with gamectl.GameInstance(f"s21_{a.policy}_{a.idx.replace(',', '')}") as g:
        env = M.MiniMotorwaysEnv(max_decisions=600, decision_delta=1.0, request_path=os.path.join(g.ipc_dir, "request.json"), response_path=os.path.join(g.ipc_dir, "response.json"), start_mode="NORMAL_START")
        for k in [int(x) for x in a.idx.split(",")]:
            seed = SEEDS[k]
            steps, end = run_episode(env, a.policy, policy, seed, t0 + 60 * a.deadline_min)
            with gzip.open(os.path.join(OUT, "raw", f"{a.policy}_{seed}.json.gz"), "wt") as f:
                json.dump({"policy": a.policy, "seed": seed, "idx": k, "end": end, "steps": steps}, f)
            print(f"[{a.policy}] seed {seed}: steps {len(steps)} deliveries {steps[-1]['score_cum'] if steps else 0} connected_end {steps[-1]['connected'] if steps else 0}/{steps[-1]['houses'] if steps else 0} illegal {sum(not s['legal'] for s in steps)} ({time.time() - t0:.0f}s)", flush=True)
        env.close()


if __name__ == "__main__":
    main()
