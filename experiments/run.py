"""python -m experiments.run --config experiments/configs/baseline_native.yaml

config -> immutable run dir -> metadata -> train (existing train_ppo, unchanged) -> checkpoints/metrics ->
held-out evaluation -> plots -> README."""
import argparse
import csv
import datetime
import json
import os
import subprocess
import sys
import time

from . import paths
from . import config as cfgmod
from . import metadata as md
from . import runlayout as rl
from . import postprocess as pp
from . import plots
from . import gamectl


def log(msg: str):
    print(f"[experiments.run] {msg}", flush=True)


def _checkpoint_index(seed_dir: str):
    try:
        return json.load(open(os.path.join(seed_dir, "checkpoints", "index.json"), encoding="utf-8"))
    except Exception:
        return []


def run_training_seed(run_dir: str, run_id: str, resolved: dict, seed: int, n_seed_dir: str, tag: str, reproduces=None):
    """Launch the worker subprocess for one seed; always clean up the game instances it started."""
    ipc_dir = os.path.join(paths.LOGS_DIR, f"ipc_{tag}")
    game_log = os.path.join(paths.LOGS_DIR, f"instrumentation_{tag}.log")
    cfg_path = os.path.join(run_dir, "config.yaml")
    stdout_path = os.path.join(n_seed_dir, "logs", "train_stdout.log")
    cmd = [sys.executable, "-u", "-m", "experiments.train_worker", "--seed-dir", n_seed_dir, "--config", cfg_path,
           "--seed", str(seed), "--run-id", run_id, "--ipc-dir", ipc_dir, "--game-log", game_log, "--tag", f"{tag}_"]
    env = dict(os.environ)
    env["PYTHONPATH"] = paths.ROOT + os.pathsep + paths.PYTHON_DIR + os.pathsep + env.get("PYTHONPATH", "")
    log(f"training seed {seed}: {stdout_path}")
    t0 = time.time()
    try:
        with open(stdout_path, "w", encoding="utf-8") as out:
            rc = subprocess.call(cmd, cwd=paths.ROOT, env=env, stdout=out, stderr=subprocess.STDOUT)
    finally:
        # stop exactly the game processes this run started (found through their own logs)
        import train_ppo
        n_envs = resolved["training"]["n_envs"]
        gamectl.kill_by_log(ipc_dir, game_log)
        for i in range(1, n_envs):
            gamectl.kill_by_log(os.path.join(paths.LOGS_DIR, f"ipc_{tag}_{i}"), os.path.join(paths.LOGS_DIR, f"instrumentation_{tag}_{i}.log"))
    return rc, time.time() - t0


def evaluate_final(run_dir: str, run_id: str, resolved: dict, n_seed_dir: str, tag: str, seeds_multi: bool):
    """Primary held-out evaluation of final.zip (+ every checkpoint on the eval_every cadence)."""
    from . import evaluation as ev
    idx = _checkpoint_index(n_seed_dir)
    if not idx:
        log("no checkpoints; skipping evaluation")
        return None
    eval_cfg = {"evaluation": resolved["evaluation"], "logging": resolved["logging"]}
    env_cfg = resolved["environment"]
    every = resolved["training"]["eval_every"]
    chosen, threshold = [], every          # first checkpoint at/after each multiple of eval_every
    for e in sorted((x for x in idx if x["file"] != "final.zip"), key=lambda x: x["training_timestep"]):
        if e["training_timestep"] >= threshold:
            chosen.append(e)
            threshold = (e["training_timestep"] // every + 1) * every
    fin = [e for e in idx if e["file"] == "final.zip"]
    todo = chosen + fin
    rows = []
    primary = None
    with gamectl.GameInstance(f"{tag}_eval") as g:
        for e in todo:
            path = os.path.join(n_seed_dir, "checkpoints", e["file"])
            out = os.path.join(n_seed_dir, "evaluation") if e["file"] == "final.zip" else os.path.join(n_seed_dir, "evaluation", "checkpoints", os.path.splitext(e["file"])[0])
            os.makedirs(out, exist_ok=True)
            log(f"evaluating {e['file']} -> {os.path.relpath(out, run_dir)}")
            res = ev.evaluate_to_dir("ppo", path, env_cfg, eval_cfg, out, g.ipc_dir, run_id, e, progress=lambda m: None)
            rows.append({"file": e["file"], "training_timestep": e["training_timestep"], "sha256": e["sha256"],
                         "mean_native_score": res["mean_native_score"], "native_score_ci95_low": res["native_score_ci95"][0],
                         "native_score_ci95_high": res["native_score_ci95"][1], "mean_return": res["mean_return"],
                         "terminal_rate": res["terminal_rate"], "mean_episode_length": res["mean_episode_length"],
                         "num_episodes": res["num_episodes"]})
            if e["file"] == "final.zip":
                primary = res
    with open(os.path.join(n_seed_dir, "evaluation", "checkpoints.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); [w.writerow(r) for r in rows]
    return primary


def evaluate_gap1_validation(run_dir: str, run_id: str, resolved: dict, n_seed_dir: str, tag: str):
    """Gap-1: evaluate every checkpoint on the VALIDATION manifest only (no continuation, for speed), freeze the
    checkpoint selection (evaluation/selection.json). TEST is evaluated later by experiments.gap1.final_test."""
    import csv as _csv
    from .gap1 import evaluation as E
    from .gap1 import select as S
    g = resolved["environment"]["gap1"]
    idx = _checkpoint_index(n_seed_dir)
    if not idx:
        return None
    rows = []
    with gamectl.GameInstance(f"{tag}_eval") as gi:
        for e in sorted(idx, key=lambda x: (x["training_timestep"], x["file"] == "final.zip")):
            path = os.path.join(n_seed_dir, "checkpoints", e["file"])
            out = os.path.join(n_seed_dir, "evaluation", "checkpoints", os.path.splitext(e["file"])[0])
            log(f"gap1 validation: {e['file']}")
            res = E.evaluate_to_dir("ppo", path, g["validation_split"], g["manifest_dir"], gi.ipc_dir, out,
                                    resolved["environment"]["decision_interval_seconds"], g["construction_window"],
                                    g["continuation_decisions"], True, 0, continue_after_success=False,
                                    extra={"checkpoint_run_id": run_id, "checkpoint_training_timestep": e["training_timestep"]},
                                    progress=lambda m: None, instance=gi)
            rows.append({"file": e["file"], "training_timestep": e["training_timestep"], "sha256": e["sha256"],
                         "gap_completion_rate": res["gap_completion_rate"]["rate"],
                         "ci95_low": res["gap_completion_rate"]["ci95_wilson"][0], "ci95_high": res["gap_completion_rate"]["ci95_wilson"][1],
                         "first_action_accuracy": res["first_action_accuracy"]["rate"],
                         "median_steps_to_success": res["median_steps_to_success"], "mean_steps_to_success": res["mean_steps_to_success"],
                         "destructive_failure_rate": res["destructive_failure_rate"]["rate"], "timeout_rate": res["timeout_rate"]["rate"],
                         "num_configurations": res["num_configurations"]})
    with open(os.path.join(n_seed_dir, "evaluation", "checkpoints.csv"), "w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); [w.writerow(r) for r in rows]
    sel = S.write_selection(n_seed_dir, rows)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4.5))
        xs = [r["training_timestep"] for r in rows]; ys = [r["gap_completion_rate"] for r in rows]
        ax.errorbar(xs, ys, yerr=[[y - r["ci95_low"] for y, r in zip(ys, rows)], [r["ci95_high"] - y for y, r in zip(ys, rows)]], marker="o", capsize=3, label="validation gap completion (95% Wilson CI)")
        ax.plot(xs, [r["first_action_accuracy"] for r in rows], "s--", label="validation first-action accuracy")
        ax.set_ylim(0, 1); ax.set_xlabel("training timesteps"); ax.set_title("Gap-1 validation vs steps"); ax.grid(alpha=.3); ax.legend(fontsize=8)
        os.makedirs(os.path.join(n_seed_dir, "plots"), exist_ok=True)
        fig.tight_layout(); fig.savefig(os.path.join(n_seed_dir, "plots", "gap1_validation_vs_steps.png"), dpi=110); plt.close(fig)
    except Exception as ex:
        log(f"gap1 plot failed: {ex!r}")
    return sel


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m experiments.run")
    ap.add_argument("--config", required=True)
    ap.add_argument("--runs-root", default=None)
    ap.add_argument("--dry-run", action="store_true", help="validate and print the resolved config, create nothing")
    ap.add_argument("--no-eval", action="store_true", help="skip the post-training evaluation")
    ap.add_argument("--reproduces", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)

    try:
        resolved = cfgmod.load_training_config(a.config)
    except cfgmod.ConfigError as e:
        print(str(e), file=sys.stderr)
        return 2
    if a.dry_run:
        import yaml
        print(yaml.safe_dump(resolved, sort_keys=False))
        return 0

    started = md.utc_now()
    t_start = time.time()
    run_id, run_dir = rl.new_run_dir(resolved["experiment"]["name"], a.runs_root)
    log(f"run {run_id}")
    rl.write_yaml(os.path.join(run_dir, "config.yaml"), resolved)
    import shutil
    shutil.copyfile(a.config, os.path.join(run_dir, "logs", "config_as_given.yaml"))
    git = md.git_info(save_patch_to=os.path.join(run_dir, "logs", "git_diff.patch"))
    md.save_source_snapshot(os.path.join(run_dir, "logs", "source_snapshot.zip"))
    snap = md.environment_snapshot()
    meta = {"run_id": run_id, "kind": "training", "start_time": started, "end_time": None, "status": "running",
            **git, **snap, "reproduces": a.reproduces,
            "training_seeds": resolved["training"]["seeds"], "evaluation_seeds": resolved["evaluation"]["seeds"],
            "resolved_environment": resolved["environment"], "resolved_ppo_hyperparameters":
                {**resolved["policy"]["hyperparameters"], **resolved["policy"]["fixed_hyperparameters"], "network": resolved["policy"]["network"]},
            "config_sha256": md.sha256_file(os.path.join(run_dir, "config.yaml")),
            **md.observation_action_shape(), "command": list(getattr(sys, "orig_argv", sys.argv)), "python_executable": sys.executable}
    md.write_json(os.path.join(run_dir, "metadata.json"), meta)

    seeds = resolved["training"]["seeds"]
    interval = resolved["environment"]["decision_interval_seconds"]
    overall_ok = True
    sps_list = []
    seed_summaries = {}
    short = run_id.replace("-", "").replace("_", "")[2:14]
    for seed in seeds:
        sd = rl.seed_dir(run_dir, seeds, seed)
        tag = f"exp{short}s{seed}"
        rc, wall = run_training_seed(run_dir, run_id, resolved, seed, sd, tag)
        ok = rc == 0 and os.path.exists(os.path.join(sd, "checkpoints", "final.zip"))
        overall_ok &= ok
        log(f"seed {seed}: training {'finished' if ok else 'FAILED (rc=%s)' % rc} in {wall:.0f}s")
        eps = pp.build_episodes_csv(sd, resolved["training"]["n_envs"])
        pp.build_training_csv(sd, interval, eps)
        status = {}
        try:
            status = json.load(open(os.path.join(sd, "train_status.json"), encoding="utf-8"))
        except Exception:
            pass
        ts = status.get("final_timestep") or 0
        sps = (ts / wall) if wall > 0 and ts else None
        sps_list.append(sps)
        res = None
        if ok and not a.no_eval and resolved["environment"].get("task") == "gap1":
            try:
                sel = evaluate_gap1_validation(run_dir, run_id, resolved, sd, tag)
                res = {"mean_native_score": None, "mean_return": None, "terminal_rate": None, "num_episodes": None,
                       "gap1_selected": sel["selected_file"], "gap1_validation_completion": sel["validation_gap_completion_rate"]}
            except Exception as e:
                log(f"gap1 validation failed: {e!r}")
                overall_ok = False
        elif ok and not a.no_eval:
            try:
                res = evaluate_final(run_dir, run_id, resolved, sd, tag, len(seeds) > 1)
            except Exception as e:                      # evaluation failure must not hide the training result
                log(f"evaluation failed: {e!r}")
                overall_ok = False
        try:
            plots.make_training_plots(sd, name=f"({resolved['experiment']['name']} seed {seed})")
        except Exception as e:
            log(f"plotting failed: {e!r}")
        seed_summaries[str(seed)] = {"ok": ok, "wall_seconds": round(wall, 1), "final_timestep": ts, "sps": sps,
                                     "evaluation": ({k: res[k] for k in res if k in ("mean_native_score", "mean_return", "terminal_rate", "num_episodes", "observation_shape", "action_space_size", "gap1_selected", "gap1_validation_completion")} if res else None)}

    meta["end_time"] = md.utc_now()
    meta["wall_clock_seconds"] = round(time.time() - t_start, 1)
    valid = [s for s in sps_list if s]
    meta["mean_sps"] = round(sum(valid) / len(valid), 2) if valid else None
    meta["mean_sps_note"] = "final timestep / training wall-clock seconds, including game start-up; SB3 per-iteration fps is in metrics/training.csv"
    meta["status"] = "completed" if overall_ok else "failed"
    for k in ("observation_shape", "action_space_size"):
        for sm in seed_summaries.values():
            if sm.get("evaluation") and sm["evaluation"].get(k):
                meta[k + "_observed"] = sm["evaluation"][k]
    meta["seed_summaries"] = seed_summaries
    if resolved["training"]["n_envs"] > 1:
        meta["note_n_envs"] = "training_timestep in episodes.csv is approximate with n_envs>1"
    md.write_json(os.path.join(run_dir, "metadata.json"), meta)
    rl.write_readme(run_dir)
    log(f"done: {run_dir} ({meta['status']})")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
