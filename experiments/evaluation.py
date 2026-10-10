"""Evaluation harness: WAIT / RANDOM / PPO-checkpoint under identical seeds, initialization, horizon and
metrics. Never trains or updates a model. Policy semantics deliberately mirror python/phase11a_runner.py so
results are comparable with the already-frozen baselines:
  wait    = WAIT every decision; when a weekly upgrade claim is pending (WAIT illegal) claim option 0
  random  = uniform over the currently legal actions (numpy default_rng(random_seed))
  ppo     = MaskablePPO.predict with the native legal mask (deterministic per config)"""
import csv
import json
import math
import os
import statistics
from typing import Any, Dict, List, Optional

import numpy as np

from . import paths
from .recording import CATEGORIES, EpisodeAccumulator
from . import postprocess as pp
from .train_worker import sha256_file

LAST_SHAPES = {}
W, H = 37, 21
UPGRADE_BASE = 1 + W * H * 4          # first upgrade-choice action index (existing action indexing)


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return [None, None]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0.0, c - h), min(1.0, c + h)]


def mean_ci(xs: List[float]):
    n = len(xs)
    if n == 0:
        return {"mean": None, "std": None, "ci95": [None, None]}
    m = statistics.fmean(xs)
    sd = statistics.stdev(xs) if n > 1 else 0.0
    half = 1.96 * sd / math.sqrt(n) if n > 1 else 0.0
    return {"mean": m, "std": sd, "ci95": [m - half, m + half]}


def make_eval_env(env_cfg: Dict[str, Any], ipc_dir: str):
    import mini_motorways_env as M
    shaped = env_cfg.get("reward") == "connection_shaped"
    kwargs = dict(max_decisions=env_cfg["episode_horizon_decisions"], decision_delta=env_cfg["decision_interval_seconds"],
                  request_path=os.path.join(ipc_dir, "request.json"), response_path=os.path.join(ipc_dir, "response.json"),
                  start_mode=env_cfg["start_mode"], start_gap=env_cfg["start_gap"])
    if shaped:
        kwargs.update(shaping=float(env_cfg["shaping_coefficient"]))
    return M.MiniMotorwaysEnv(**kwargs)


def policy_entropy(model, obs, mask):
    """Entropy (nats) of the masked policy distribution at this state. Read-only forward pass (no gradient, no RNG use)."""
    import torch
    with torch.no_grad():
        t, _ = model.policy.obs_to_tensor(obs)
        d = model.policy.get_distribution(t, action_masks=mask[None, :])
        return float(d.entropy().mean().item())


def run_policy(policy: str, model_path: Optional[str], env_cfg: Dict[str, Any], eval_cfg: Dict[str, Any],
               ipc_dir: str, progress=print) -> List[Dict[str, Any]]:
    ev = eval_cfg["evaluation"]
    model = None
    if policy == "ppo":
        from sb3_contrib import MaskablePPO
        import train_ppo     # registers GridExtractor for CNN checkpoints
        model = MaskablePPO.load(model_path, custom_objects={"GridExtractor": train_ppo.GridExtractor})
    rng = np.random.default_rng(ev.get("random_seed", 0))
    env = make_eval_env(env_cfg, ipc_dir)
    LAST_SHAPES.update(observation_shape=list(env.observation_space.shape), action_space_size=int(env.action_space.n))
    meta = {"initialization": env_cfg["initialization"], "start_gap": env_cfg["start_gap"],
            "reward_mode": env_cfg.get("reward", "native_score"),
            "decision_interval_seconds": env_cfg["decision_interval_seconds"],
            "diagnostics": bool((eval_cfg.get("logging") or {}).get("diagnostics", False))}
    acc = EpisodeAccumulator(meta)
    rows = []
    n = 0
    try:
        for seed in ev["seeds"]:
            for rep in range(ev["episodes_per_seed"]):
                obs, info = env.reset(seed=seed)
                acc.reset_episode(info.get("seed", seed), None, env.last_raw_observation if acc.diag is not None else None)
                terminated = truncated = False
                while not (terminated or truncated):
                    mask = env.action_masks()
                    legal = np.flatnonzero(mask)
                    if policy == "wait":
                        ups = [int(a) for a in legal if a >= UPGRADE_BASE]
                        action = 0 if mask[0] else (ups[0] if ups else int(legal[0]))
                    elif policy == "random":
                        action = int(rng.choice(legal))
                    else:
                        a, _ = model.predict(obs, action_masks=mask, deterministic=ev["deterministic"])
                        action = int(a)
                    legal_sel = bool(mask[action])
                    ent = None
                    if acc.diag is not None and model is not None:
                        ent = policy_entropy(model, obs, mask)
                    mask_before = mask.copy() if acc.diag is not None else None
                    obs, reward, terminated, truncated, info = env.step(action)
                    acc.add_step(action, legal_sel, reward, info, terminated, truncated,
                                 raw_after=(env.last_raw_observation if acc.diag is not None else None), mask_before=mask_before, policy_entropy=ent)
                n += 1
                row = acc.row(n, None)
                row["repeat"] = rep
                row["requested_seed"] = seed
                row["action_index_counts"] = acc.action_index_rows()
                rows.append(row)
                progress(f"  eval episode {n}: seed={seed} native_score={row['native_score']} return={row['return']} "
                         f"length={row['length_decisions']} terminal={row['terminal']}")
    finally:
        try:
            env.close()
        except Exception:
            pass
    return rows


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    score = mean_ci([r["native_score"] for r in rows])
    ret = mean_ci([r["return"] for r in rows])
    ln = mean_ci([r["length_decisions"] for r in rows])
    term = sum(1 for r in rows if r["terminal"])
    tot = {c: 0 for c in CATEGORIES}
    key = {"WAIT": "wait_count", "ADD_ROAD": "add_count", "REMOVE_ROAD": "remove_count", "UPGRADE": "upgrade_count"}
    for r in rows:
        for c in CATEGORIES:
            tot[c] += r[key[c]]
    steps = max(sum(tot.values()), 1)
    return {
        "num_episodes": n,
        "mean_native_score": score["mean"], "median_native_score": statistics.median([r["native_score"] for r in rows]) if n else None,
        "std_native_score": score["std"], "native_score_ci95": score["ci95"],
        "mean_return": ret["mean"], "std_return": ret["std"], "return_ci95": ret["ci95"],
        "terminal_rate": term / n if n else None, "terminal_rate_ci95_wilson": wilson(term, n),
        "mean_episode_length": ln["mean"], "episode_length_ci95": ln["ci95"],
        "action_counts": tot, "action_fractions": {c: tot[c] / steps for c in CATEGORIES},
        "ci_note": "normal-approximation 95% CI of the mean over episodes (episodes sharing a seed are not independent)",
    }


def write_outputs(out_dir: str, rows: List[Dict[str, Any]], results: Dict[str, Any], logging_cfg: Dict[str, Any]):
    os.makedirs(out_dir, exist_ok=True)
    pp.write_csv(os.path.join(out_dir, "episodes.csv"), pp.EPISODE_COLUMNS + ["repeat", "requested_seed"], rows)
    pp.write_actions(out_dir, rows)
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(out_dir, "episodes_full.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def evaluate_to_dir(policy: str, model_path: Optional[str], env_cfg: Dict[str, Any], eval_cfg: Dict[str, Any],
                    out_dir: str, ipc_dir: str, run_id: Optional[str], checkpoint_entry: Optional[Dict[str, Any]],
                    extra: Optional[Dict[str, Any]] = None, progress=print) -> Dict[str, Any]:
    import datetime
    started = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    rows = run_policy(policy, model_path, env_cfg, eval_cfg, ipc_dir, progress)
    res = summarize(rows)
    res.update({
        "policy": policy,
        "checkpoint": os.path.abspath(model_path) if model_path else None,
        "checkpoint_sha256": sha256_file(model_path) if model_path else None,
        "checkpoint_training_timestep": checkpoint_entry["training_timestep"] if checkpoint_entry else None,
        "checkpoint_run_id": checkpoint_entry["run_id"] if checkpoint_entry else None,
        "evaluated_in_run_id": run_id,
        "evaluation_seeds": eval_cfg["evaluation"]["seeds"],
        "episodes_per_seed": eval_cfg["evaluation"]["episodes_per_seed"],
        "deterministic": eval_cfg["evaluation"]["deterministic"] if policy == "ppo" else None,
        "random_seed": eval_cfg["evaluation"].get("random_seed", 0) if policy == "random" else None,
        "environment": env_cfg, "started_utc": started, **LAST_SHAPES,
        "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    })
    if extra:
        res.update(extra)
    write_outputs(out_dir, rows, res, eval_cfg.get("logging", {}))
    return res
