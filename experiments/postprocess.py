"""Turn the raw per-worker episode logs and SB3's progress.csv into the documented CSVs.
Raw files stay in metrics/raw/ and metrics/sb3_progress.csv; the CSVs here are derived and re-creatable."""
import csv
import glob
import json
import os
from collections import Counter, defaultdict
from typing import Any, Dict, List

from .recording import CATEGORIES

EPISODE_COLUMNS = ["episode", "training_timestep", "game_seed", "initialization", "start_gap", "reward_mode", "return",
                   "native_score", "length_decisions", "length_sim_seconds", "terminal", "truncated",
                   "wait_count", "add_count", "remove_count", "upgrade_count",
                   "wait_fraction", "add_fraction", "remove_fraction", "upgrade_fraction", "wall_seconds"]
from .recording import GAP1_FIELDS
DIAG_COLUMNS = ["diag_deliveries", "diag_first_delivery_step", "diag_connections_max", "diag_connections_final", "diag_first_connection_step",
                "diag_connection_to_first_delivery_steps", "diag_n_add_accepted", "diag_n_remove_accepted", "diag_churn", "diag_repeated_edits",
                "diag_n_nonzero_nonterminal_reward", "diag_road_edges_max", "diag_legal_mean", "diag_empirical_action_entropy", "diag_policy_entropy_mean",
                "diag_terminal_penalty_missing", "diag_terminal_on_entry"]
EXTRA_EPISODE_COLUMNS = ["worker_pid", "unix_time"] + [f"gap1_{k}" for k in GAP1_FIELDS] + DIAG_COLUMNS

TRAINING_COLUMNS = ["timestep", "wall_time", "simulated_game_time", "episodes_completed", "episode_return",
                    "episode_native_score", "episode_length", "SPS", "policy_loss", "value_loss", "entropy",
                    "approx_kl", "clip_fraction", "explained_variance"]
SB3_MAP = {"timestep": "time/total_timesteps", "wall_time": "time/time_elapsed", "SPS": "time/fps",
           "episode_return": "rollout/ep_rew_mean", "episode_length": "rollout/ep_len_mean",
           "policy_loss": "train/policy_gradient_loss", "value_loss": "train/value_loss",
           "approx_kl": "train/approx_kl", "clip_fraction": "train/clip_fraction",
           "explained_variance": "train/explained_variance"}


def read_raw_episodes(raw_dir: str) -> List[Dict[str, Any]]:
    rows = []
    for p in sorted(glob.glob(os.path.join(raw_dir, "episodes_*.jsonl"))):
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows.sort(key=lambda r: (r["unix_time"], r["worker_pid"]))
    return rows


def write_csv(path: str, columns: List[str], rows: List[Dict[str, Any]]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: ("" if r.get(c) is None else r.get(c)) for c in columns})


def write_actions(metrics_dir: str, episodes: List[Dict[str, Any]], prefix: str = ""):
    """actions.csv: aggregate by category; action_indices.csv: per action index with legality at selection."""
    cat = Counter()
    idx_count, idx_legal = Counter(), Counter()
    key = {"WAIT": "wait_count", "ADD_ROAD": "add_count", "REMOVE_ROAD": "remove_count", "UPGRADE": "upgrade_count"}
    for e in episodes:
        for c in CATEGORIES:
            cat[c] += int(e.get(key[c], 0) or 0)
        for i, (n, legal) in (e.get("action_index_counts") or {}).items():
            idx_count[int(i)] += n
            idx_legal[int(i)] += legal
    total = max(sum(cat.values()), 1)
    with open(os.path.join(metrics_dir, f"{prefix}actions.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["category", "count", "fraction"])
        for c in CATEGORIES:
            w.writerow([c, cat[c], round(cat[c] / total, 6)])
    with open(os.path.join(metrics_dir, f"{prefix}action_indices.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["action_index", "count", "legal_when_selected", "fraction"])
        for i in sorted(idx_count):
            w.writerow([i, idx_count[i], idx_legal[i], round(idx_count[i] / total, 6)])
    return {c: cat[c] for c in CATEGORIES}


def build_episodes_csv(seed_dir: str, n_envs: int, write_actions_files: bool = True) -> List[Dict[str, Any]]:
    metrics = os.path.join(seed_dir, "metrics")
    eps = read_raw_episodes(os.path.join(metrics, "raw"))
    cum = 0
    for i, e in enumerate(eps, 1):
        e["episode"] = i
        cum += e["length_decisions"]
        # exact for one env (worker-local step counter == global timestep); with several envs it is the
        # cumulative decisions of episodes ordered by completion time (approximate, flagged in metadata)
        e["training_timestep"] = e["worker_steps_total"] if n_envs == 1 else cum
    write_csv(os.path.join(metrics, "episodes.csv"), EPISODE_COLUMNS + EXTRA_EPISODE_COLUMNS, eps)
    if write_actions_files:
        write_actions(metrics, eps)
    return eps


def build_training_csv(seed_dir: str, decision_interval: float, episodes: List[Dict[str, Any]]) -> int:
    metrics = os.path.join(seed_dir, "metrics")
    src = os.path.join(metrics, "sb3_progress.csv")
    if not os.path.exists(src):
        return 0
    out = []
    ep_sorted = sorted(episodes, key=lambda e: e["training_timestep"])
    with open(src, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            def g(k):
                v = row.get(SB3_MAP[k], "")
                return float(v) if v not in ("", None) else None
            ts = g("timestep")
            if ts is None:
                continue
            done = [e for e in ep_sorted if e["training_timestep"] <= ts]
            recent = done[-100:]                      # same window SB3's Monitor uses for ep_rew_mean
            ent = row.get("train/entropy_loss", "")
            out.append({
                "timestep": int(ts), "wall_time": g("wall_time"), "simulated_game_time": round(ts * decision_interval, 3),
                "episodes_completed": len(done), "episode_return": g("episode_return"),
                "episode_native_score": (sum(e["native_score"] for e in recent) / len(recent)) if recent else None,
                "episode_length": g("episode_length"), "SPS": g("SPS"), "policy_loss": g("policy_loss"),
                "value_loss": g("value_loss"), "entropy": (-float(ent)) if ent not in ("", None) else None,
                "approx_kl": g("approx_kl"), "clip_fraction": g("clip_fraction"),
                "explained_variance": g("explained_variance"),
            })
    write_csv(os.path.join(metrics, "training.csv"), TRAINING_COLUMNS, out)
    return len(out)
