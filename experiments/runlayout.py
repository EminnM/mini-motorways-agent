"""Run directories: creation (never overwrite), seed sub-layout, README manifest."""
import csv
import datetime
import json
import os
from typing import Any, Dict, List, Optional

import yaml

from . import paths


def new_run_dir(name: str, runs_root: str = None, kind: str = "") -> (str, str):
    runs_root = runs_root or paths.RUNS_DIR
    os.makedirs(runs_root, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    base = f"{stamp}_{kind}{name}"
    run_id, n = base, 1
    while True:
        path = os.path.join(runs_root, run_id)
        try:
            os.makedirs(path)               # exclusive: fails if it exists
            break
        except FileExistsError:
            n += 1
            run_id = f"{base}_{n}"
    for sub in ("checkpoints", "metrics", "evaluation", "plots", "logs"):
        os.makedirs(os.path.join(path, sub))
    return run_id, path


def seed_dir(run_dir: str, seeds: List[int], seed: int) -> str:
    """Flat layout for a single training seed (as documented); per-seed sub-runs when several seeds are given."""
    if len(seeds) == 1:
        return run_dir
    d = os.path.join(run_dir, "seeds", f"seed_{seed}")
    for sub in ("checkpoints", "metrics", "evaluation", "plots", "logs"):
        os.makedirs(os.path.join(d, sub), exist_ok=True)
    return d


def write_yaml(path: str, obj: Any):
    with open(path, "x", encoding="utf-8") as f:       # 'x': refuse to overwrite
        yaml.safe_dump(obj, f, sort_keys=False, default_flow_style=False)


def _read_json(path: str) -> Optional[Any]:
    try:
        return json.load(open(path, encoding="utf-8"))
    except Exception:
        return None


def write_readme(run_dir: str):
    """Human-readable manifest. Reports configuration and MEASURED numbers only; no conclusions are drawn."""
    cfg = yaml.safe_load(open(os.path.join(run_dir, "config.yaml"), encoding="utf-8"))
    meta = _read_json(os.path.join(run_dir, "metadata.json")) or {}
    exp = cfg.get("experiment", {})
    env = cfg.get("environment", {})
    tr = cfg.get("training", {})
    L = []
    L.append(f"# {meta.get('run_id', exp.get('name'))}\n")
    L.append("## Experiment")
    L.append(f"- name: `{exp.get('name')}`  tags: {exp.get('tags')}")
    L.append(f"- description: {exp.get('description') or '(none)'}")
    L.append("\n## Hypothesis")
    L.append(exp.get("hypothesis") or "_(none provided in the config)_")
    L.append("\n## Configuration (resolved)")
    L.append(f"- initialization: `{env.get('initialization')}` (gap {env.get('start_gap')}), reward: `{env.get('reward')}`"
             + (f" (coefficient {env.get('shaping_coefficient')})" if env.get('reward') == 'connection_shaped' else ""))
    L.append(f"- decision interval: {env.get('decision_interval_seconds')} s, horizon: {env.get('episode_horizon_decisions')} decisions")
    pol = cfg.get("policy", {})
    L.append(f"- policy: `{pol.get('algorithm')}` network `{pol.get('network')}`, hyperparameters {pol.get('hyperparameters')} (fixed: {pol.get('fixed_hyperparameters')})")
    L.append(f"- training: {tr.get('total_timesteps')} timesteps, seeds {tr.get('seeds')}, checkpoint every {tr.get('checkpoint_every')}, n_envs {tr.get('n_envs')}")
    L.append(f"- evaluation: seeds {cfg.get('evaluation', {}).get('seeds')}, episodes/seed {cfg.get('evaluation', {}).get('episodes_per_seed')}, deterministic {cfg.get('evaluation', {}).get('deterministic')}")
    L.append("\n## Checkpoint")
    idx_files = [os.path.join(run_dir, "checkpoints", "index.json")] + sorted(
        os.path.join(run_dir, "seeds", d, "checkpoints", "index.json") for d in (os.listdir(os.path.join(run_dir, "seeds")) if os.path.isdir(os.path.join(run_dir, "seeds")) else []))
    for p in idx_files:
        idx = _read_json(p)
        if idx:
            fin = [e for e in idx if e["file"] == "final.zip"]
            L.append(f"- {os.path.relpath(os.path.dirname(p), run_dir)}: {len(idx)} files; final: "
                     + (f"`final.zip` at timestep {fin[0]['training_timestep']}, sha256 `{fin[0]['sha256']}`" if fin else "(no final.zip)"))
    L.append("\n## Training duration")
    L.append(f"- start: {meta.get('start_time')}  end: {meta.get('end_time')}  wall-clock: {meta.get('wall_clock_seconds')} s  mean SPS: {meta.get('mean_sps')}")
    L.append("\n## Evaluation status")
    res = _read_json(os.path.join(run_dir, "evaluation", "results.json"))
    sub = [d for d in sorted(os.listdir(os.path.join(run_dir, "evaluation"))) if os.path.isdir(os.path.join(run_dir, "evaluation", d))]
    L.append(f"- primary evaluation: {'done' if res else 'not run'}; additional evaluation folders: {sub or 'none'}")
    L.append("\n## Headline metrics (measured)")
    if res:
        L.append(f"- policy `{res.get('policy')}` on {res.get('num_episodes')} held-out episodes: native score mean {res.get('mean_native_score')} "
                 f"(95% CI {res.get('native_score_ci95')}), median {res.get('median_native_score')}, std {res.get('std_native_score')}")
        L.append(f"- return mean {res.get('mean_return')}, terminal rate {res.get('terminal_rate')}, mean episode length {res.get('mean_episode_length')}")
        L.append(f"- action fractions: {res.get('action_fractions')}")
    trn = os.path.join(run_dir, "metrics", "training.csv")
    if os.path.exists(trn):
        rows = list(csv.DictReader(open(trn, encoding="utf-8")))
        if rows:
            last = rows[-1]
            L.append(f"- last training log row: timestep {last['timestep']}, episodes completed {last['episodes_completed']}, "
                     f"ep_return {last['episode_return']}, native score (last-100 mean) {last['episode_native_score']}, episode length {last['episode_length']}")
    L.append("\n## Notes")
    L.append("- Plots are derived from `metrics/*.csv`; the CSV/JSON files are authoritative.")
    L.append("- git: " + (f"commit `{meta.get('git_commit')}` dirty={meta.get('git_dirty')}" if meta.get("git_commit") else "no git repository; see `metadata.json` source_files / `logs/source_snapshot.zip`"))
    open(os.path.join(run_dir, "README.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
