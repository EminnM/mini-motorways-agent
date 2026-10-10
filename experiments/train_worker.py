"""Runs ONE training seed by calling the existing python/train_ppo.py main() UNCHANGED, with passive
instrumentation attached from the outside (monkey-patching names, never editing the file):
  * train_ppo.MiniMotorwaysEnv / mini_motorways_env.make_monitored_env -> same env + RecorderEnv
  * train_ppo.CheckpointCallback -> subclass that additionally registers checkpoints (SHA-256, index) and
    attaches a CSV output to SB3's own logger (same values SB3 prints)
Invoked as a subprocess by experiments.run:  python -m experiments.train_worker --seed-dir D --config C --seed S ..."""
import argparse
import datetime
import hashlib
import json
import os
import shutil
import sys
import time

from . import paths  # noqa: F401
from . import config as cfgmod
from . import recording


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


class CheckpointRegistry:
    def __init__(self, ckpt_dir: str, run_id: str, seed: int):
        self.dir, self.run_id, self.seed = ckpt_dir, run_id, seed
        self.entries = []
        os.makedirs(ckpt_dir, exist_ok=True)

    def register(self, src: str, timestep: int, name: str) -> str:
        dst = os.path.join(self.dir, name)
        if os.path.exists(dst):
            raise FileExistsError(dst)           # never overwrite a checkpoint
        shutil.move(src, dst)
        entry = {"file": name, "training_timestep": int(timestep), "sha256": sha256_file(dst),
                 "created_utc": utc_now(), "run_id": self.run_id, "training_seed": self.seed}
        self.entries.append(entry)
        with open(os.path.join(self.dir, "index.json"), "w", encoding="utf-8") as f:
            json.dump(self.entries, f, indent=2)
        with open(os.path.join(self.dir, "index.csv"), "w", encoding="utf-8") as f:
            f.write("file,training_timestep,sha256,created_utc,run_id,training_seed\n")
            for e in self.entries:
                f.write(f"{e['file']},{e['training_timestep']},{e['sha256']},{e['created_utc']},{e['run_id']},{e['training_seed']}\n")
        return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-dir", required=True, help="directory holding checkpoints/ metrics/ logs/ for this seed")
    ap.add_argument("--config", required=True, help="resolved config.yaml of the run")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--ipc-dir", required=True)
    ap.add_argument("--game-log", required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()

    r = cfgmod.load_yaml(a.config)
    env_cfg = r["environment"]
    ckpt_dir = os.path.join(a.seed_dir, "checkpoints")
    raw_ckpt_dir = os.path.join(ckpt_dir, "_sb3_raw")
    metrics_dir = os.path.join(a.seed_dir, "metrics")
    raw_dir = os.path.join(metrics_dir, "raw")
    for d in (raw_ckpt_dir, raw_dir):
        os.makedirs(d, exist_ok=True)

    os.environ[recording.RAW_DIR_ENV] = raw_dir
    os.environ[recording.META_ENV] = json.dumps({
        "initialization": env_cfg["initialization"], "start_gap": env_cfg["start_gap"],
        "reward_mode": env_cfg["reward"], "decision_interval_seconds": env_cfg["decision_interval_seconds"],
        "diagnostics": bool((r.get("logging") or {}).get("diagnostics", False))})

    import train_ppo
    import mini_motorways_env

    registry = CheckpointRegistry(ckpt_dir, a.run_id, a.seed)
    callbacks = []

    class HarnessCheckpointCallback(train_ppo.CheckpointCallback):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._registered = 0
            self._csv_attached = False
            callbacks.append(self)

        def _on_training_start(self):
            if self.model.num_timesteps == 0 and not registry.entries:
                # step-0 checkpoint = the untrained policy (same seed/initialisation), for learning-curve baselines
                import tempfile
                tmp = os.path.join(raw_ckpt_dir, "initial_model.zip")
                self.model.save(tmp)
                registry.register(tmp, 0, "step_000000.zip")
            if not self._csv_attached:
                from stable_baselines3.common.logger import CSVOutputFormat
                self.model.logger.output_formats.append(
                    CSVOutputFormat(os.path.join(metrics_dir, "sb3_progress.csv")))
                self._csv_attached = True

        def _on_step(self) -> bool:
            cont = super()._on_step()
            while self._registered < len(self.saved_paths):
                ts, path = self.saved_paths[self._registered]
                new = registry.register(path, ts, f"step_{self.num_timesteps:06d}.zip")
                self.saved_paths[self._registered] = (ts, new)
                self._registered += 1
            return cont

    # --- passive instrumentation hooks (the training code itself is not edited)
    wrap = None
    if env_cfg.get("task") == "gap1":
        from experiments.gap1 import gap1_env as G
        g = env_cfg["gap1"]
        tasks = G.load_manifest(os.path.join(g["manifest_dir"], f"{g['train_split']}.jsonl"))
        variant = {"native_score": "native", "completion": "completion", "native_plus_completion": "native_plus_completion",
                   "connection_shaped": "connection_shaped"}[env_cfg["reward"]]
        wrap = lambda e: G.Gap1Env(e, tasks, g["construction_window"], g["continuation_decisions"], reward_variant=variant,
                                   beta=float(g.get("beta", 5.0)), sampling="random", sample_seed=a.seed,
                                   success_terminates=bool(g.get("success_terminates", False)))
    train_ppo.MiniMotorwaysEnv = recording.make_recorded_env_factory(train_ppo.MiniMotorwaysEnv, wrap)
    mini_motorways_env.make_monitored_env = recording.make_recorded_monitored_env
    train_ppo.CheckpointCallback = HarnessCheckpointCallback

    sys.argv = ["train_ppo"] + cfgmod.trainer_argv(r, a.seed, raw_ckpt_dir, a.ipc_dir, a.game_log, a.tag)
    t0 = time.time()
    status = {"seed": a.seed, "started_utc": utc_now(), "argv": sys.argv[1:], "ok": False}
    try:
        train_ppo.main()
        status["ok"] = True
    finally:
        status["ended_utc"] = utc_now()
        status["wall_seconds"] = round(time.time() - t0, 2)
        final_ts = None
        if callbacks:
            final_ts = int(callbacks[-1].model.num_timesteps)
        status["final_timestep"] = final_ts
        raw_final = os.path.join(raw_ckpt_dir, "final_model.zip")
        if os.path.exists(raw_final):
            registry.register(raw_final, final_ts if final_ts is not None else -1, "final.zip")
        with open(os.path.join(a.seed_dir, "train_status.json"), "w", encoding="utf-8") as f:
            json.dump(status, f, indent=2)


if __name__ == "__main__":
    main()
