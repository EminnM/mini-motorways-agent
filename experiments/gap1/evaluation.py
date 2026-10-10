"""Gap-1 evaluation on a frozen manifest split: WAIT, RANDOM LEGAL, or a PPO checkpoint. Identical tasks, horizon,
environment and termination for every policy. Never trains. Writes per-configuration rows (episodes.csv), summary
(results.json) and breakdowns."""
import csv
import json
import math
import os
import statistics
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

import numpy as np

from .. import paths
from ..evaluation import wilson, mean_ci
from ..train_worker import sha256_file
from . import gap1_env as G

UPGRADE_BASE = 1 + 37 * 21 * 4


def build_env(ipc_dir: str, delta: float, k: int, cont: int, reward_variant: str = "native", beta: float = 5.0):
    import mini_motorways_env as M
    env = M.MiniMotorwaysEnv(
        max_decisions=k + cont + 40, decision_delta=delta,
        request_path=os.path.join(ipc_dir, "request.json"), response_path=os.path.join(ipc_dir, "response.json"),
        start_mode="CONNECTED_START", start_gap=0)
    return env


def a_star_stats(model, obs, mask, a_star: int):
    """Probability and rank the policy assigns to a* (read-only; does not change inference)."""
    import torch
    with torch.no_grad():
        obs_t, _ = model.policy.obs_to_tensor(obs)
        dist = model.policy.get_distribution(obs_t, action_masks=mask[None, :])
        probs = dist.distribution.probs[0].cpu().numpy()
    p = float(probs[a_star])
    rank = int((probs > p).sum()) + 1
    return p, rank


def _run_task(env, model, policy, rng, deterministic, i, t):
    obs, info = env.reset(options={"task_index": i})
    terminated = truncated = False
    ret = 0.0
    first_p = first_rank = None
    n_steps = 0
    last_gap = {}
    while not (terminated or truncated):
        mask = env.action_masks()
        legal = np.flatnonzero(mask)
        if n_steps == 0 and model is not None:
            first_p, first_rank = a_star_stats(model, obs, mask, t["a_star"]["action_index"])
        if policy == "wait":
            ups = [int(a) for a in legal if a >= UPGRADE_BASE]
            action = 0 if mask[0] else (ups[0] if ups else int(legal[0]))
        elif policy == "random":
            action = int(rng.choice(legal))
        else:
            a, _ = model.predict(obs, action_masks=mask, deterministic=deterministic)
            action = int(a)
        obs, r, terminated, truncated, info = env.step(action)
        ret += r
        n_steps += 1
        last_gap = info["gap1"]
    return {
        "task_id": t["task_id"], "game_seed": t["game_seed"], "a_star_index": t["a_star"]["action_index"],
        "gap_x": t["a_star"]["x"], "gap_y": t["a_star"]["y"], "orientation": t["gap_orientation"],
        "path_length_edits": t["path_length_edits"], "gap_position_in_path": t["gap_position_in_path"],
        "success": last_gap["success"], "steps_to_success": last_gap["steps_to_success"],
        "first_action_correct": last_gap["first_action_correct"], "destructive_failure": last_gap["destructive_failure"],
        "timeout": last_gap["timeout"], "native_terminal": last_gap["native_terminal"],
        "construction_decisions": last_gap["construction_decisions"],
        "native_score_after_continuation": last_gap["native_score_total"], "return_native_variant": round(ret, 4),
        "n_wait": last_gap["n_WAIT"], "n_add": last_gap["n_ADD"], "n_remove": last_gap["n_REMOVE"], "n_upgrade": last_gap["n_UPGRADE"],
        "n_add_correct": last_gap["n_ADD_correct"], "n_add_incorrect": last_gap["n_ADD_incorrect"],
        "n_remove_before_success": last_gap["n_REMOVE_before_success"],
        "a_star_prob_first_decision": first_p, "a_star_rank_first_decision": first_rank,
        "a_star_added_at": last_gap["a_star_added_at"],
    }


def run_split(policy: str, model_path: Optional[str], manifest: List[Dict[str, Any]], ipc_dir: Optional[str], delta: float,
              k: int, cont: int, deterministic: bool = True, random_seed: int = 0, continue_after_success: bool = True,
              progress=print, instance=None, restart_every: int = 60, partial_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """`instance` (a gamectl.GameInstance) enables automatic restarts: every `restart_every` configurations (the game
    process hangs after a few hundred RESETs) and after any IPC failure; completed rows are appended to `partial_path`
    so an interrupted evaluation resumes where it stopped."""
    model = None
    if policy == "ppo":
        from sb3_contrib import MaskablePPO
        import train_ppo
        model = MaskablePPO.load(model_path, custom_objects={"GridExtractor": train_ppo.GridExtractor})
    rng = np.random.default_rng(random_seed)
    done = {}
    if partial_path and os.path.exists(partial_path):
        for line in open(partial_path, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                done[r["task_id"]] = r

    def make_env():
        base = build_env(instance.ipc_dir if instance else ipc_dir, delta, k, cont)
        return G.Gap1Env(base, manifest, construction_window=k, continuation=cont, reward_variant="native", sampling="sequential",
                         continue_after_success=continue_after_success)

    env = make_env()
    rows = []
    since_restart = 0
    pf = open(partial_path, "a", encoding="utf-8") if partial_path else None
    try:
        for i, t in enumerate(manifest):
            if t["task_id"] in done:
                rows.append(done[t["task_id"]])
                continue
            for attempt in range(4):
                try:
                    row = _run_task(env, model, policy, rng, deterministic, i, t)
                    break
                except (TimeoutError, RuntimeError, OSError, EOFError) as e:
                    if instance is None or attempt == 3:
                        raise
                    progress(f"  [restart] {t['task_id']}: {e!r}")
                    try:
                        env.close()
                    except Exception:
                        pass
                    instance.start()
                    env = make_env()
                    since_restart = 0
            rows.append(row)
            if pf:
                pf.write(json.dumps(row) + "\n")
                pf.flush()
            since_restart += 1
            if instance is not None and since_restart >= restart_every and i + 1 < len(manifest):
                try:
                    env.close()
                except Exception:
                    pass
                instance.start()
                env = make_env()
                since_restart = 0
            if (i + 1) % 40 == 0:
                progress(f"  {i + 1}/{len(manifest)} configurations")
        progress(f"  evaluated {len(rows)} configurations")
    finally:
        if pf:
            pf.close()
        try:
            env.close()
        except Exception:
            pass
    return rows


def _rate(rows, key):
    xs = [1 if r[key] else 0 for r in rows]
    k, n = sum(xs), len(xs)
    return {"rate": k / n if n else None, "count": k, "n": n, "ci95_wilson": wilson(k, n)}


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    succ = [r for r in rows if r["success"]]
    steps = [r["steps_to_success"] for r in succ]
    cons_total = sum(max(r["construction_decisions"], 1) for r in rows)
    sums = {k: sum(r[k] for r in rows) for k in ("n_wait", "n_add", "n_remove", "n_upgrade", "n_add_correct", "n_add_incorrect", "n_remove_before_success")}
    tot = max(sums["n_wait"] + sums["n_add"] + sums["n_remove"] + sums["n_upgrade"], 1)
    pr = [r["a_star_prob_first_decision"] for r in rows if r["a_star_prob_first_decision"] is not None]
    rk = [r["a_star_rank_first_decision"] for r in rows if r["a_star_rank_first_decision"] is not None]
    # dominant-action diagnostic: what fraction would a policy that always picks the single most common a* index solve?
    cnt = Counter(r["a_star_index"] for r in rows)
    top_idx, top_n = cnt.most_common(1)[0]
    by = {}
    for name, keyf in (("orientation", lambda r: r["orientation"]),
                       ("gap_x_bin", lambda r: f"x[{(r['gap_x'] + 18) // 6 * 6 - 18},{(r['gap_x'] + 18) // 6 * 6 - 12})"),
                       ("path_length", lambda r: (r["path_length_edits"] // 3) * 3)):
        groups = defaultdict(list)
        for r in rows:
            groups[keyf(r)].append(r)
        by[name] = {str(g): {"n": len(v), "completion": sum(1 for r in v if r["success"]) / len(v),
                             "first_action_accuracy": sum(1 for r in v if r["first_action_correct"]) / len(v)}
                    for g, v in sorted(groups.items(), key=lambda kv: str(kv[0]))}
    return {
        "num_configurations": n,
        "gap_completion_rate": _rate(rows, "success"),
        "first_action_accuracy": _rate(rows, "first_action_correct"),
        "median_steps_to_success": statistics.median(steps) if steps else None,
        "mean_steps_to_success": statistics.fmean(steps) if steps else None,
        "destructive_failure_rate": _rate(rows, "destructive_failure"),
        "timeout_rate": _rate(rows, "timeout"),
        "native_terminal_rate": _rate(rows, "native_terminal"),
        "native_score_after_fixed_continuation": mean_ci([r["native_score_after_continuation"] for r in rows]),
        "action_fractions": {"WAIT": sums["n_wait"] / tot, "ADD": sums["n_add"] / tot, "REMOVE": sums["n_remove"] / tot, "UPGRADE": sums["n_upgrade"] / tot},
        "correct_edge_add_fraction": sums["n_add_correct"] / tot, "incorrect_add_fraction": sums["n_add_incorrect"] / tot,
        "remove_before_success_fraction": sums["n_remove_before_success"] / tot,
        "a_star_prob_first_decision_mean": statistics.fmean(pr) if pr else None,
        "a_star_rank_first_decision_median": statistics.median(rk) if rk else None,
        "a_star_in_top5_first_decision": (sum(1 for x in rk if x <= 5) / len(rk)) if rk else None,
        "dominant_a_star_index": {"index": top_idx, "share_of_configs": top_n / n},
        "breakdown": by,
    }


def write(out_dir: str, rows: List[Dict[str, Any]], results: Dict[str, Any]):
    os.makedirs(out_dir, exist_ok=True)
    cols = list(rows[0].keys())
    with open(os.path.join(out_dir, "episodes.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader(); [w.writerow(r) for r in rows]
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def evaluate_to_dir(policy: str, model_path: Optional[str], split: str, manifest_dir: str, ipc_dir: str, out_dir: str,
                    delta: float = 1.0, k: int = 20, cont: int = 40, deterministic: bool = True, random_seed: int = 0,
                    continue_after_success: bool = True, extra: Optional[Dict[str, Any]] = None, progress=print,
                    instance=None, restart_every: int = 60) -> Dict[str, Any]:
    mpath = os.path.join(manifest_dir, f"{split}.jsonl")
    manifest = G.load_manifest(mpath)
    os.makedirs(out_dir, exist_ok=True)
    rows = run_split(policy, model_path, manifest, ipc_dir, delta, k, cont, deterministic, random_seed, continue_after_success, progress,
                     instance=instance, restart_every=restart_every, partial_path=os.path.join(out_dir, "partial_rows.jsonl"))
    res = summarize(rows)
    res.update({"policy": policy, "split": split, "manifest": mpath, "manifest_sha256": sha256_file(mpath),
                "checkpoint": os.path.abspath(model_path) if model_path else None,
                "checkpoint_sha256": sha256_file(model_path) if model_path else None,
                "deterministic": deterministic if policy == "ppo" else None, "construction_window": k,
                "continuation_decisions": cont if continue_after_success else 0, "decision_interval_seconds": delta})
    if extra:
        res.update(extra)
    write(out_dir, rows, res)
    return res
