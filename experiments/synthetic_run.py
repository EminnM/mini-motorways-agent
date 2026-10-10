"""python -m experiments.synthetic_run --config experiments/configs/synthetic_s1_mlp.yaml

Synthetic-lab experiment runner built on the experiment harness: immutable run directory, resolved config, metadata with
source/library hashes, checkpoint index (SHA-256), per-episode / training CSVs, validation per checkpoint, checkpoint
selection on VALIDATION only, a single TEST evaluation of the selected checkpoint, plots. Trains the same MaskablePPO
(unchanged hyperparameters) used on the native game; only the environment is the synthetic sandbox."""
import argparse
import csv
import json
import os
import sys
import time

import yaml

from . import paths
from . import metadata as md
from . import runlayout as rl
from . import postprocess as pp
from . import plots
from .recording import RecorderEnv
from .train_worker import CheckpointRegistry

MAN_DIR = os.path.join(paths.ROOT, "research", "synthetic_lab", "manifests")
HP_ALLOWED = {"n_steps", "batch_size", "learning_rate", "n_epochs", "target_kl", "ent_coef", "gamma"}
HP_DEFAULT = {"n_steps": 512, "batch_size": 256, "learning_rate": 3e-4, "n_epochs": 4, "target_kl": 0.03, "ent_coef": 0.01, "gamma": 0.99}   # = frozen native Baseline-0
SCHEMA = {
    "experiment": {"name": str, "description": str, "hypothesis": str, "tags": list},
    "task": {"n_gaps": int, "reward": str, "construction_window": int, "delay": int, "period": int, "deliveries": int, "manifest_dir": str},
    "policy": {"network": str, "hyperparameters": dict},
    "training": {"total_timesteps": int, "seeds": list, "checkpoint_every": int, "torch_threads": int},
    "evaluation": {"deterministic_primary": bool, "also_stochastic": bool},
}


def resolve(cfg):
    problems = []
    for sec in cfg:
        if sec not in SCHEMA:
            problems.append(f"unknown section {sec}")
            continue
        spec = SCHEMA[sec]
        for k, v in (cfg[sec] or {}).items():
            if k not in spec:
                problems.append(f"unknown key {sec}.{k}")
            elif not isinstance(v, spec[k]):
                problems.append(f"{sec}.{k} must be {spec[k].__name__}")
    for req in ("experiment", "task", "policy", "training"):
        if req not in cfg:
            problems.append(f"missing section {req}")
    if problems:
        raise SystemExit("invalid config:\n  - " + "\n  - ".join(problems))
    r = json.loads(json.dumps(cfg))
    r["task"].setdefault("n_gaps", 1)
    r["task"].setdefault("reward", "completion")
    r["task"].setdefault("construction_window", 20)
    r["task"].setdefault("delay", 12)
    r["task"].setdefault("period", 13)
    r["task"].setdefault("deliveries", 3)
    r["task"].setdefault("manifest_dir", MAN_DIR)
    r["policy"].setdefault("network", "mlp")
    if r["policy"]["network"] not in ("mlp", "spatial_cnn"):
        raise SystemExit("policy.network must be mlp or spatial_cnn")
    hp = r["policy"].setdefault("hyperparameters", {})
    for k in hp:
        if k not in HP_ALLOWED:
            raise SystemExit(f"unknown hyperparameter {k}")
    for k, v in HP_DEFAULT.items():
        hp.setdefault(k, v)
    r["training"].setdefault("checkpoint_every", 25000)
    r["training"].setdefault("torch_threads", 2)
    r.setdefault("evaluation", {})
    r["evaluation"].setdefault("deterministic_primary", True)
    r["evaluation"].setdefault("also_stochastic", True)
    r["experiment"].setdefault("description", "")
    r["experiment"].setdefault("hypothesis", "")
    r["experiment"].setdefault("tags", [])
    return r


def train_seed(r, seed, sd, run_id, log):
    import torch
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import CSVOutputFormat
    torch.set_num_threads(r["training"]["torch_threads"])
    sys.path.insert(0, paths.PYTHON_DIR)
    from synthetic_rl.env import SyntheticGapEnv, load_tasks
    from synthetic_rl.models import SpatialMaskablePolicy
    t = r["task"]
    n = t["n_gaps"]
    tasks = load_tasks(os.path.join(t["manifest_dir"], f"gap{n}_train.jsonl"))
    raw = os.path.join(sd, "metrics", "raw")
    os.makedirs(raw, exist_ok=True)
    meta = {"initialization": "synthetic", "start_gap": n, "reward_mode": t["reward"], "decision_interval_seconds": 1.0}
    env = RecorderEnv(SyntheticGapEnv(tasks, reward_mode=t["reward"], construction_window=t["construction_window"], delay=t["delay"], period=t["period"],
                                      deliveries=t["deliveries"]), os.path.join(raw, f"episodes_{os.getpid()}.jsonl"), meta)
    hp = r["policy"]["hyperparameters"]
    policy = "MlpPolicy" if r["policy"]["network"] == "mlp" else SpatialMaskablePolicy
    model = MaskablePPO(policy, env, learning_rate=hp["learning_rate"], n_steps=hp["n_steps"], batch_size=hp["batch_size"], n_epochs=hp["n_epochs"],
                        target_kl=hp["target_kl"], gamma=hp["gamma"], gae_lambda=0.95, clip_range=0.2, ent_coef=hp["ent_coef"], verbose=0, seed=seed, device="cpu")
    ckdir = os.path.join(sd, "checkpoints")
    registry = CheckpointRegistry(ckdir, run_id, seed)
    every = r["training"]["checkpoint_every"]
    n_params = sum(p.numel() for p in model.policy.parameters())

    class CB(BaseCallback):
        def __init__(self):
            super().__init__()
            self.done = 0
            self.t0 = time.time()

        def _save(self, ts):
            tmp = os.path.join(ckdir, "_tmp.zip")
            self.model.save(tmp)
            registry.register(tmp, ts, f"step_{ts:06d}.zip")

        def _on_training_start(self):
            self.model.logger.output_formats.append(CSVOutputFormat(os.path.join(sd, "metrics", "sb3_progress.csv")))
            self._save(0)

        def _on_step(self):
            if self.num_timesteps - self.done * every >= every:
                self.done += 1
                self._save(self.num_timesteps)
            return True

    t0 = time.time()
    cb = CB()
    model.learn(total_timesteps=r["training"]["total_timesteps"], callback=cb)
    wall = time.time() - t0
    tmp = os.path.join(ckdir, "_tmp.zip")
    model.save(tmp)
    registry.register(tmp, int(model.num_timesteps), "final.zip")
    return {"wall_seconds": wall, "final_timestep": int(model.num_timesteps), "sps": model.num_timesteps / wall, "parameters": n_params}


def evaluate_seed(r, sd, log):
    from sb3_contrib import MaskablePPO
    from synthetic_rl import evaluation as E
    from synthetic_rl.env import load_tasks
    from synthetic_rl.models import SpatialMaskablePolicy
    t = r["task"]
    n = t["n_gaps"]
    val = load_tasks(os.path.join(t["manifest_dir"], f"gap{n}_validation.jsonl"))
    test = load_tasks(os.path.join(t["manifest_dir"], f"gap{n}_test.jsonl"))
    train_idx = {i for x in load_tasks(os.path.join(t["manifest_dir"], f"gap{n}_train.jsonl")) for i in x["solution_action_indices"]}
    idx = json.load(open(os.path.join(sd, "checkpoints", "index.json")))
    rows = []
    co = {"SpatialMaskablePolicy": SpatialMaskablePolicy}
    for e in sorted(idx, key=lambda x: (x["training_timestep"], x["file"] == "final.zip")):
        if e["file"] == "final.zip" and any(x["training_timestep"] == e["training_timestep"] and x["file"] != "final.zip" for x in idx):
            continue
        model = MaskablePPO.load(os.path.join(sd, "checkpoints", e["file"]), device="cpu")
        for mode, det in (("deterministic", True), ("stochastic", False)):
            if mode == "stochastic" and not r["evaluation"]["also_stochastic"]:
                continue
            res = E.summarize(E.evaluate("ppo", val, model, det, t["reward"], t["construction_window"]), train_idx)
            rows.append({"file": e["file"], "training_timestep": e["training_timestep"], "sha256": e["sha256"], "mode": mode, "split": "validation",
                         "completion": res["completion"]["rate"], "ci95_low": res["completion"]["ci95_wilson"][0], "ci95_high": res["completion"]["ci95_wilson"][1],
                         "first_action_accuracy": res["first_action_accuracy"]["rate"], "median_steps_to_success": res["median_steps_to_success"],
                         "destructive_failure": res["destructive_failure"]["rate"], "timeout": res["timeout"]["rate"],
                         "wait": res["action_fractions"]["WAIT"], "add": res["action_fractions"]["ADD"], "remove": res["action_fractions"]["REMOVE"],
                         "solution_prob_first": res["solution_prob_first_decision_mean"], "solution_rank_first_median": res["solution_rank_first_decision_median"],
                         "solution_top5": res["solution_in_top5_first_decision"]})
    os.makedirs(os.path.join(sd, "evaluation"), exist_ok=True)
    with open(os.path.join(sd, "evaluation", "validation_by_checkpoint.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        [w.writerow(x) for x in rows]
    # frozen selection rule (validation only): max completion (deterministic), tie -> lower median steps, tie -> earlier
    det = [x for x in rows if x["mode"] == "deterministic"]
    best = sorted(det, key=lambda x: (-x["completion"], float("inf") if x["median_steps_to_success"] is None else x["median_steps_to_success"], x["training_timestep"]))[0]
    sel = {"rule": "max validation completion (deterministic); tie -> lower median steps; tie -> earlier checkpoint", "selected_file": best["file"],
           "training_timestep": best["training_timestep"], "sha256": best["sha256"], "validation_completion": best["completion"]}
    json.dump(sel, open(os.path.join(sd, "evaluation", "selection.json"), "x"), indent=2)
    out = {"selection": sel}
    for name, ck in (("selected", best["file"]), ("final", "final.zip")):
        model = MaskablePPO.load(os.path.join(sd, "checkpoints", ck), device="cpu")
        for mode, det_flag in (("deterministic", True), ("stochastic", False)):
            rws = E.evaluate("ppo", test, model, det_flag, t["reward"], t["construction_window"])
            res = E.summarize(rws, train_idx)
            res.update({"checkpoint": ck, "checkpoint_sha256": next(x["sha256"] for x in idx if x["file"] == ck), "split": "test", "mode": mode})
            d = os.path.join(sd, "evaluation", f"test_{name}_{mode}")
            os.makedirs(d, exist_ok=True)
            json.dump(res, open(os.path.join(d, "results.json"), "w"), indent=2)
            with open(os.path.join(d, "episodes.csv"), "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rws[0].keys()))
                w.writeheader()
                [w.writerow(x) for x in rws]
            out[f"{name}_{mode}"] = res
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m experiments.synthetic_run")
    ap.add_argument("--config", required=True)
    ap.add_argument("--runs-root", default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    r = resolve(yaml.safe_load(open(a.config, encoding="utf-8")))
    if a.dry_run:
        print(yaml.safe_dump(r, sort_keys=False))
        return 0
    started = md.utc_now()
    t_start = time.time()
    run_id, run_dir = rl.new_run_dir(r["experiment"]["name"], a.runs_root)
    print(f"[synthetic_run] {run_id}", flush=True)
    rl.write_yaml(os.path.join(run_dir, "config.yaml"), r)
    import shutil
    shutil.copyfile(a.config, os.path.join(run_dir, "logs", "config_as_given.yaml"))
    git = md.git_info(save_patch_to=os.path.join(run_dir, "logs", "git_diff.patch"))
    md.save_source_snapshot(os.path.join(run_dir, "logs", "source_snapshot.zip"))
    snap = md.environment_snapshot()
    snap["source_files"].update({f"python/synthetic_rl/{x}": md.sha256_file(os.path.join(paths.PYTHON_DIR, "synthetic_rl", x)) for x in os.listdir(os.path.join(paths.PYTHON_DIR, "synthetic_rl")) if x.endswith(".py")})
    manifest_hashes = {x: md.sha256_file(os.path.join(r["task"]["manifest_dir"], x)) for x in sorted(os.listdir(r["task"]["manifest_dir"])) if x.startswith(f"gap{r['task']['n_gaps']}_") and x.endswith(".jsonl")}
    meta = {"run_id": run_id, "kind": "synthetic_training", "start_time": started, "status": "running", **git, **snap, "task_manifest_sha256": manifest_hashes,
            "training_seeds": r["training"]["seeds"], "resolved_task": r["task"],
            "resolved_ppo_hyperparameters": {**r["policy"]["hyperparameters"], "gae_lambda": 0.95, "clip_range": 0.2, "network": r["policy"]["network"]},
            "observation_shape": [11666], "action_space_size": 3111, "command": list(getattr(sys, "orig_argv", sys.argv)),
            "note": "synthetic sandbox: no game process, DLL hashes recorded only for traceability of the surrounding code base"}
    md.write_json(os.path.join(run_dir, "metadata.json"), meta)
    seeds = r["training"]["seeds"]
    summaries = {}
    for seed in seeds:
        sd = rl.seed_dir(run_dir, seeds, seed)
        print(f"[synthetic_run] training seed {seed} ({r['policy']['network']}, {r['training']['total_timesteps']} steps)", flush=True)
        info = train_seed(r, seed, sd, run_id, print)
        eps = pp.build_episodes_csv(sd, 1)
        pp.build_training_csv(sd, 1.0, eps)
        ev = evaluate_seed(r, sd, print)
        try:
            plots.make_training_plots(sd, name=f"({r['experiment']['name']} seed {seed})")
        except Exception as e:
            print("plot failed", e)
        summaries[str(seed)] = {**info, "selected": ev["selection"]["selected_file"],
                                "test_selected_deterministic_completion": ev["selected_deterministic"]["completion"]["rate"],
                                "test_final_deterministic_completion": ev["final_deterministic"]["completion"]["rate"]}
        print(f"[synthetic_run] seed {seed}: {summaries[str(seed)]}", flush=True)
    meta.update({"end_time": md.utc_now(), "wall_clock_seconds": round(time.time() - t_start, 1), "status": "completed", "seed_summaries": summaries})
    md.write_json(os.path.join(run_dir, "metadata.json"), meta)
    print(f"[synthetic_run] done {run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
