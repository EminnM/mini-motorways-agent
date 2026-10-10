"""S34 replication trainer = S31 train.py with only the run directory, instance-name prefix and PPO seed parametrised (architecture, hyper-parameters, reward, seeds, budget identical).
Pre-flight: the deployed plugin DLL must be postfix_v8 and the observation must export globals.playableAreaAtTime (environment-version consistency), else the run does not start.
   python research/s34_replication/code/train_rep.py --name rep1_seed131 --seed 131"""
import argparse
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(ROOT, "research", "s31_hybrid_native_rl", "code"))
from hybrid_env import HybridNativeEnv, train_seed_list  # noqa: E402

N_ENVS, N_STEPS, TOTAL = 4, 512, 100352
CKPTS = [0, 24576, 49152, 73728, 100352]
DLL = r"C:\Program Files (x86)\Steam\steamapps\common\Mini Motorways\Bootstrap\MiniMotorwaysBootstrap.dll"
V8 = "2aa42367cbb14babce31a6063038f50be9b3535bd7bc23d8dbbdbb65533aa00a"


def make_env(rank, run, name):
    def _f():
        seeds = train_seed_list(40)[rank::N_ENVS]
        path = os.path.join(run, "episodes", f"worker{rank}.jsonl")
        os.makedirs(os.path.dirname(path), exist_ok=True)

        def log(ep):
            with open(path, "a") as f:
                f.write(json.dumps(ep) + "\n")
        return HybridNativeEnv(seeds, f"s34_{name}_w{rank}", max_decisions=600, log_fn=log)
    return _f


def preflight(name):
    h = hashlib.sha256(open(DLL, "rb").read()).hexdigest()
    assert h == V8, f"deployed DLL is not postfix_v8: {h}"
    e = HybridNativeEnv([700000], f"s34_{name}_preflight", max_decisions=5, log_fn=None)
    e.reset()
    area = e.raw["globals"].get("playableAreaAtTime")
    e.close()
    assert area is not None and len(area) == 4, "playableAreaAtTime missing: environment is not postfix_v8"
    return {"dll_sha256": h, "env_version": "postfix_v8", "playableAreaAtTime_at_reset_700000": area}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--seed", type=int, required=True)
    a = ap.parse_args()
    run = os.path.join(ROOT, "research", "s34_replication", "runs", a.name)
    assert not os.path.exists(os.path.join(run, "train_log.json")), "run directory already used; refusing to overwrite"
    import torch
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import configure
    from hybrid_policy import HybridMaskablePolicy, init_from_migrated_v2
    torch.set_num_threads(4)
    os.makedirs(os.path.join(run, "checkpoints"), exist_ok=True)
    pf = preflight(a.name)
    print("preflight", pf, flush=True)
    venv = SubprocVecEnv([make_env(i, run, a.name) for i in range(N_ENVS)], start_method="spawn")
    model = MaskablePPO(HybridMaskablePolicy, venv, device="cpu", n_steps=N_STEPS, batch_size=256, n_epochs=4, target_kl=0.03, learning_rate=3e-4, ent_coef=0.01, gamma=0.99, gae_lambda=0.95, seed=a.seed)
    rep = init_from_migrated_v2(model, os.path.join(ROOT, "research", "s14_group_observation", "s8c_migrated_v2_step0.zip"))
    model.set_logger(configure(run, ["csv"]))
    model.save(os.path.join(run, "checkpoints", "step_0000000.zip"))
    meta = {"phase": "S34", "name": a.name, "ppo_seed": a.seed, "preflight": pf, "init": rep, "n_envs": N_ENVS, "n_steps": N_STEPS, "total_cap": TOTAL, "hyper": {"batch": 256, "epochs": 4, "target_kl": 0.03, "lr": 3e-4, "ent": 0.01, "gamma": 0.99, "lambda": 0.95}, "train_seeds": train_seed_list(40), "eval_seeds_disjoint": True, "updates": [], "t_start": time.time()}

    class Mon(BaseCallback):
        def __init__(self):
            super().__init__()
            self.next_ck = 1

        def _on_rollout_end(self):
            lg = self.model.logger.name_to_value
            meta["updates"].append({"decisions": int(self.model.num_timesteps), "wall_s": round(time.time() - meta["t_start"], 1), **{k.split("/")[-1]: float(v) for k, v in lg.items() if k.startswith(("train/", "rollout/")) and isinstance(v, (int, float))}})
            while self.next_ck < len(CKPTS) and self.model.num_timesteps >= CKPTS[self.next_ck]:
                self.model.save(os.path.join(run, "checkpoints", f"step_{CKPTS[self.next_ck]:07d}.zip"))
                self.next_ck += 1
            json.dump(meta, open(os.path.join(run, "train_log.json"), "w"), indent=1)
            print(f"[{a.name}] decisions={self.model.num_timesteps} wall={time.time() - meta['t_start']:.0f}s kl={lg.get('train/approx_kl')}", flush=True)

        def _on_step(self):
            return True
    try:
        model.learn(total_timesteps=TOTAL, callback=Mon(), progress_bar=False)
    finally:
        model.save(os.path.join(run, "checkpoints", f"step_{int(model.num_timesteps):07d}.zip"))
        meta["t_end"] = time.time()
        meta["decisions_done"] = int(model.num_timesteps)
        json.dump(meta, open(os.path.join(run, "train_log.json"), "w"), indent=1)
        venv.close()
    print("DONE", a.name, model.num_timesteps, flush=True)


if __name__ == "__main__":
    main()
