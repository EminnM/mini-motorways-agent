"""S31 trainer: MaskablePPO (S1/S8 hyper-parameters) on the hybrid native env, 4 headless game instances (SubprocVecEnv), hard cap 100,352 native decisions, checkpoints at 0/25k/50k/75k/100k.
Training seeds 700000 + 7919*i (i < 40), disjoint from the 12 evaluation seeds.   python research/s31_hybrid_native_rl/code/train.py"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from hybrid_env import HybridNativeEnv, train_seed_list, ROOT  # noqa: E402

OUT = os.path.join(ROOT, "research", "s31_hybrid_native_rl")
RUN = os.path.join(OUT, "runs", "hybrid_s31")
N_ENVS, N_STEPS, TOTAL = 4, 512, 100352
CKPTS = [0, 24576, 49152, 73728, 100352]


def make_env(rank):
    def _f():
        seeds = train_seed_list(40)[rank::N_ENVS]
        path = os.path.join(RUN, "episodes", f"worker{rank}.jsonl")
        os.makedirs(os.path.dirname(path), exist_ok=True)

        def log(ep):
            with open(path, "a") as f:
                f.write(json.dumps(ep) + "\n")
        return HybridNativeEnv(seeds, f"s31_w{rank}", max_decisions=600, log_fn=log)
    return _f


def main():
    import torch
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import configure
    from hybrid_policy import HybridMaskablePolicy, init_from_migrated_v2
    torch.set_num_threads(4)
    os.makedirs(os.path.join(RUN, "checkpoints"), exist_ok=True)
    venv = SubprocVecEnv([make_env(i) for i in range(N_ENVS)], start_method="spawn")
    model = MaskablePPO(HybridMaskablePolicy, venv, device="cpu", n_steps=N_STEPS, batch_size=256, n_epochs=4, target_kl=0.03, learning_rate=3e-4, ent_coef=0.01, gamma=0.99, gae_lambda=0.95, seed=31)
    rep = init_from_migrated_v2(model, os.path.join(ROOT, "research", "s14_group_observation", "s8c_migrated_v2_step0.zip"))
    model.set_logger(configure(RUN, ["csv"]))
    model.save(os.path.join(RUN, "checkpoints", "step_0000000.zip"))
    meta = {"init": rep, "n_envs": N_ENVS, "n_steps": N_STEPS, "total_cap": TOTAL, "hyper": {"batch": 256, "epochs": 4, "target_kl": 0.03, "lr": 3e-4, "ent": 0.01, "gamma": 0.99, "lambda": 0.95}, "train_seeds": train_seed_list(40), "eval_seeds_disjoint": True, "updates": [], "t_start": time.time()}

    class Mon(BaseCallback):
        def __init__(self):
            super().__init__()
            self.next_ck = 1
            self.t_last = time.time()

        def _on_rollout_end(self):
            lg = self.model.logger.name_to_value
            meta["updates"].append({"decisions": int(self.model.num_timesteps), "wall_s": round(time.time() - meta["t_start"], 1), **{k.split("/")[-1]: float(v) for k, v in lg.items() if k.startswith(("train/", "rollout/")) and isinstance(v, (int, float))}})
            while self.next_ck < len(CKPTS) and self.model.num_timesteps >= CKPTS[self.next_ck]:
                self.model.save(os.path.join(RUN, "checkpoints", f"step_{CKPTS[self.next_ck]:07d}.zip"))
                self.next_ck += 1
            json.dump(meta, open(os.path.join(RUN, "train_log.json"), "w"), indent=1)
            print(f"[update] decisions={self.model.num_timesteps} wall={time.time() - meta['t_start']:.0f}s kl={lg.get('train/approx_kl')}", flush=True)

        def _on_step(self):
            return True
    try:
        model.learn(total_timesteps=TOTAL, callback=Mon(), progress_bar=False)
    finally:
        model.save(os.path.join(RUN, "checkpoints", f"step_{int(model.num_timesteps):07d}.zip"))
        meta["t_end"] = time.time()
        meta["decisions_done"] = int(model.num_timesteps)
        json.dump(meta, open(os.path.join(RUN, "train_log.json"), "w"), indent=1)
        venv.close()
    print("DONE", model.num_timesteps, flush=True)


if __name__ == "__main__":
    main()
