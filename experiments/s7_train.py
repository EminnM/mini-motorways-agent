"""S7 training arms A / B / C on the S7 long-route family (PREPARED, NOT RUN - requires separate approval).
   python experiments/s7_train.py --config research/s7_long_route/configs/arm_A.yaml --seed 11 [--dry-run]
Arms (identical network, reward, hyperparameters, decision budget, evaluation tasks):
  A  direct        : random-init SpatialMaskablePolicy, trained only on the target distribution T = uniform over required lengths {7, 10}
  B  curriculum    : random-init, stages [2] -> [3] -> [5] -> [7] -> T with adaptive advancement (preregistered rule in the config)
  C  pretrained+curriculum : B, initialised from the frozen S1 seed-13 checkpoint (policy+value copied via the strict loader; fresh Adam, fresh timestep counter)
Reward: +1 on full house-carpark connectivity, 0 otherwise. No intermediate reward, no demonstrations, no privileged observation.
Evaluation: validation manifest at fixed decision intervals (deterministic policy); TEST manifest only once, on the FINAL checkpoint (--final-test)."""
import argparse
import collections
import hashlib
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

OUT_ROOT = os.path.join(ROOT, "runs")


class CurriculumEnv:
    """thin helper bound to a LongRouteEnv instance: stage-dependent task sampling + episode statistics (does not change observations, mask or reward)"""

    @staticmethod
    def build(manifest, stages, seed):
        from synthetic_rl import s7

        class _Env(s7.LongRouteEnv):
            def __init__(self, tasks_by_k, stages, seed):
                self.by_k = tasks_by_k
                self.stages = stages
                self.stage = 0
                self.stage_decisions = 0
                self.recent = collections.deque(maxlen=100)
                self.total_decisions = 0
                self.stage_log = []
                allt = [t for k in sorted(tasks_by_k) for t in tasks_by_k[k]]
                super().__init__(allt, sampling="random", seed=seed)
                self._pool_rng = np.random.default_rng(seed + 12345)

            def current_lengths(self):
                return self.stages[self.stage]

            def reset(self, *, seed=None, options=None):
                ks = self.current_lengths()
                k = int(self._pool_rng.choice(ks))
                pool = self.by_k[k]
                idx = int(self._pool_rng.integers(len(pool)))
                t = pool[idx]
                self.tasks = [t]
                return super().reset(seed=seed, options={"task_index": 0})

            def step(self, action):
                o, r, te, tr, info = super().step(action)
                self.stage_decisions += 1
                self.total_decisions += 1
                if te or tr:
                    self.recent.append(1.0 if te else 0.0)
                    info["episode_k"] = self.k
                    info["episode_success"] = bool(te)
                return o, r, te, tr, info

        by_k = {}
        for t in manifest:
            by_k.setdefault(t["k_min"], []).append(t)
        return _Env(by_k, stages, seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--final-test", action="store_true")
    ap.add_argument("--torch-threads", type=int, default=1)
    a = ap.parse_args()
    torch.set_num_threads(a.torch_threads)
    cfg = yaml.safe_load(open(a.config))
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.callbacks import BaseCallback
    from synthetic_rl import s7, s7_eval as E
    from synthetic_rl.models import SpatialMaskablePolicy
    man = os.path.join(ROOT, "research", "s7_long_route", "manifests")
    train = [json.loads(l) for l in open(os.path.join(man, "s7_train.jsonl"))]
    val = [json.loads(l) for l in open(os.path.join(man, "s7_validation.jsonl"))]
    test = [json.loads(l) for l in open(os.path.join(man, "s7_test.jsonl"))]
    stages = cfg["training"]["stages"]
    env = CurriculumEnv.build(train, stages, a.seed)
    hp = cfg["ppo"]
    model = MaskablePPO(SpatialMaskablePolicy, env, learning_rate=hp["learning_rate"], n_steps=hp["n_steps"], batch_size=hp["batch_size"], n_epochs=hp["n_epochs"],
                        target_kl=hp["target_kl"], gamma=hp["gamma"], gae_lambda=0.95, clip_range=0.2, ent_coef=hp["ent_coef"], verbose=0, seed=a.seed, device="cpu")
    init = cfg["training"].get("init_checkpoint")
    init_report = None
    if init:
        import train_ppo                                   # strict loader (class/space/name/shape checks; fresh optimizer/schedule/timesteps)
        init_report = train_ppo.init_from_checkpoint(model, os.path.join(ROOT, init["path"]), init.get("sha256"))
    run_id = time.strftime("%Y-%m-%d_%H%M%S") + f"_s7_{cfg['name']}_s{a.seed}"
    run_dir = os.path.join(OUT_ROOT, run_id)
    total = cfg["training"]["total_decisions"]
    plan = {"run_id": run_id, "arm": cfg["name"], "seed": a.seed, "total_decisions": total, "stages": stages, "advance": cfg["training"].get("advance"), "init": init, "n_params": int(sum(p.numel() for p in model.policy.parameters())),
            "eval_every": cfg["training"]["eval_every"], "init_report": {k: init_report[k] for k in ("sha256", "n_parameters_loaded")} if init_report else None}
    print(json.dumps(plan, indent=1))
    if a.dry_run:
        obs, _ = env.reset()
        with torch.no_grad():
            model.predict(obs, action_masks=env.action_masks(), deterministic=True)
        print("dry-run OK: env reset, model forward, no learn() call")
        return
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    json.dump({"config": cfg, "plan": plan}, open(os.path.join(run_dir, "plan.json"), "w"), indent=1)
    adv = cfg["training"].get("advance")

    def run_eval(tag, decisions):
        t0 = time.time()
        rows = E.evaluate("model", val, model=model, deterministic=True)
        summ = E.summarize(rows)
        rec = {"decisions": decisions, "tag": tag, "summary": {str(k): v for k, v in summ.items()}, "eval_seconds": round(time.time() - t0, 1), "stage": env.stage, "stage_decisions": env.stage_decisions}
        with open(os.path.join(run_dir, "eval_validation.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
        p = os.path.join(run_dir, "checkpoints", f"step_{decisions:07d}.zip")
        model.save(p)
        with open(os.path.join(run_dir, "checkpoints", "SHA256.txt"), "a") as f:
            f.write(f"{hashlib.sha256(open(p, 'rb').read()).hexdigest()} *{os.path.basename(p)}\n")
        return rec

    class CB(BaseCallback):
        def __init__(self):
            super().__init__()
            self.next_eval = cfg["training"]["eval_every"]
            self.t0 = time.time()

        def _on_step(self):
            return True

        def _on_rollout_end(self):
            nt = self.model.num_timesteps
            if adv and env.stage < len(stages) - 1:
                s = env.stage_decisions
                if (len(env.recent) >= adv["window"] and np.mean(env.recent) >= adv["success_threshold"] and s >= adv["min_stage_decisions"]) or s >= adv["max_stage_decisions"]:
                    with open(os.path.join(run_dir, "stages.jsonl"), "a") as f:
                        f.write(json.dumps({"left_stage": env.stage, "lengths": stages[env.stage], "stage_decisions": s, "recent_success": float(np.mean(env.recent)) if env.recent else None, "at_decisions": nt}) + "\n")
                    env.stage += 1
                    env.stage_decisions = 0
                    env.recent.clear()
            if nt >= self.next_eval:
                run_eval("periodic", nt)
                self.next_eval += cfg["training"]["eval_every"]

    run_eval("step0", 0)
    t0 = time.time()
    model.learn(total_timesteps=total, callback=CB(), reset_num_timesteps=True)
    run_eval("final", model.num_timesteps)
    json.dump({"wall_s": round(time.time() - t0, 1), "decisions": int(model.num_timesteps), "stage_end": env.stage, "stage_decisions_final": env.stage_decisions,
               "sps": round(model.num_timesteps / (time.time() - t0), 1)}, open(os.path.join(run_dir, "train_status.json"), "w"), indent=1)
    if a.final_test:
        rows = E.evaluate("model", test, model=model, deterministic=True)
        json.dump({"summary": {str(k): v for k, v in E.summarize(rows).items()}, "rows": rows}, open(os.path.join(run_dir, "final_test.json"), "w"))
    print("done", run_dir)


if __name__ == "__main__":
    main()
