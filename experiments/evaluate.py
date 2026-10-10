"""Evaluation CLI (never trains).
  python -m experiments.evaluate --run runs/<id> --config experiments/configs/eval_heldout.yaml
  python -m experiments.evaluate --policy wait   --config experiments/configs/eval_heldout.yaml
  python -m experiments.evaluate --policy random --config experiments/configs/eval_heldout.yaml
  python -m experiments.evaluate --checkpoint runs/<id>/checkpoints/final.zip --config experiments/configs/eval_heldout.yaml
Without --run a new immutable run directory (kind 'evaluation') is created. With --run the results are added to
that run under evaluation/<eval_id>/ (existing files are never overwritten)."""
import argparse
import datetime
import json
import os
import sys

import yaml

from . import paths
from . import config as cfgmod
from . import metadata as md
from . import runlayout as rl
from . import gamectl


def _find_run_root(path: str):
    d = os.path.dirname(os.path.abspath(path))
    for _ in range(5):
        if os.path.exists(os.path.join(d, "config.yaml")) and os.path.exists(os.path.join(d, "metadata.json")):
            return d
        d = os.path.dirname(d)
    return None


def _checkpoint_entry(ckpt: str):
    idx_path = os.path.join(os.path.dirname(os.path.abspath(ckpt)), "index.json")
    try:
        for e in json.load(open(idx_path, encoding="utf-8")):
            if e["file"] == os.path.basename(ckpt):
                return e
    except Exception:
        pass
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m experiments.evaluate")
    ap.add_argument("--config", required=True, help="evaluation YAML")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--run", help="existing run directory (evaluates its checkpoint)")
    src.add_argument("--policy", choices=["wait", "random"])
    src.add_argument("--checkpoint")
    ap.add_argument("--which", default="final", help="with --run: 'final', 'all', or a checkpoint file name (e.g. step_010240.zip)")
    ap.add_argument("--runs-root", default=None)
    ap.add_argument("--ipc-dir", default=None, help="attach to an already-running game instance instead of launching one")
    ap.add_argument("--name", default=None)
    a = ap.parse_args(argv)

    base_env, base_run, targets = None, None, []
    if a.run:
        base_run = os.path.abspath(a.run)
        base_cfg = yaml.safe_load(open(os.path.join(base_run, "config.yaml"), encoding="utf-8"))
        base_env = base_cfg["environment"]
        ckdir = os.path.join(base_run, "checkpoints")
        idx = json.load(open(os.path.join(ckdir, "index.json"), encoding="utf-8"))
        if a.which == "all":
            names = [e["file"] for e in idx]
        else:
            names = [a.which if a.which.endswith(".zip") else a.which + ".zip"]
        targets = [("ppo", os.path.join(ckdir, n)) for n in names]
    elif a.checkpoint:
        root = _find_run_root(a.checkpoint)
        if root:
            base_env = yaml.safe_load(open(os.path.join(root, "config.yaml"), encoding="utf-8"))["environment"]
        targets = [("ppo", os.path.abspath(a.checkpoint))]
    else:
        targets = [(a.policy, None)]

    try:
        eval_cfg = cfgmod.load_eval_config(a.config, base_env)
    except cfgmod.ConfigError as e:
        print(str(e), file=sys.stderr)
        return 2

    from . import evaluation as ev
    env_cfg = eval_cfg["environment"]
    cfg_stem = os.path.splitext(os.path.basename(a.config))[0]

    if base_run:
        run_dir, run_id = base_run, os.path.basename(base_run)
    else:
        label = a.policy if a.policy else os.path.splitext(os.path.basename(a.checkpoint))[0]
        run_id, run_dir = rl.new_run_dir(f"{a.name or cfg_stem}_{label}", a.runs_root, kind="eval_")
        rl.write_yaml(os.path.join(run_dir, "config.yaml"), eval_cfg)
        git = md.git_info(save_patch_to=os.path.join(run_dir, "logs", "git_diff.patch"))
        md.save_source_snapshot(os.path.join(run_dir, "logs", "source_snapshot.zip"))
        md.write_json(os.path.join(run_dir, "metadata.json"), {
            "run_id": run_id, "kind": "evaluation", "start_time": md.utc_now(), "end_time": None, "status": "running",
            **git, **md.environment_snapshot(), "evaluation_seeds": eval_cfg["evaluation"]["seeds"],
            "resolved_environment": env_cfg, "command": list(getattr(sys, "orig_argv", sys.argv)), "training_seeds": None})

    results = []
    inst = None
    try:
        if a.ipc_dir:
            ipc = a.ipc_dir
        else:
            inst = gamectl.GameInstance(f"{run_id[-24:]}_evalcli".replace("-", "").replace(":", "")).start()
            ipc = inst.ipc_dir
        for policy, path in targets:
            entry = _checkpoint_entry(path) if path else None
            eid = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + cfg_stem + "_" + (os.path.splitext(os.path.basename(path))[0] if path else policy)
            if base_run:
                out = os.path.join(run_dir, "evaluation", eid)
            else:
                out = os.path.join(run_dir, "evaluation")
            if os.path.exists(os.path.join(out, "results.json")):
                raise FileExistsError(f"refusing to overwrite {out}/results.json")
            print(f"[experiments.evaluate] {policy} {path or ''} -> {os.path.relpath(out, paths.ROOT)}", flush=True)
            res = ev.evaluate_to_dir(policy, path, env_cfg, eval_cfg, out, ipc, run_id, entry)
            results.append(res)
            print(json.dumps({k: res[k] for k in ("policy", "num_episodes", "mean_native_score", "median_native_score",
                                                  "std_native_score", "mean_return", "terminal_rate", "mean_episode_length")}, indent=2), flush=True)
    finally:
        if inst:
            inst.stop()
        if not base_run:
            meta = json.load(open(os.path.join(run_dir, "metadata.json"), encoding="utf-8"))
            meta["end_time"] = md.utc_now()
            meta["status"] = "completed" if results else "failed"
            md.write_json(os.path.join(run_dir, "metadata.json"), meta)
            try:
                rl.write_readme(run_dir)
            except Exception as e:
                print("[experiments.evaluate] README skipped:", e)
    return 0 if results else 1


if __name__ == "__main__":
    sys.exit(main())
