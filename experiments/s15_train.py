"""S15 arm runner (synthetic only).   python experiments/s15_train.py --arm A|B --seed 15 --decisions 30720 --deadline-min 33
Arm A: v2 observation with group channels. Arm B: identical architecture and init, group channels zeroed in training and evaluation.
Init = research/s14_group_observation/s8c_migrated_v2_step0.zip (migrated S8-C, BEFORE any S14 PPO update). Checkpoints/evaluations at 0, 5,120, 10,240, 20,480, 30,720 decisions (multiples of 512)."""
import argparse
import hashlib
import json
import math
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sb3_contrib import MaskablePPO  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402
from stable_baselines3.common.logger import configure  # noqa: E402
from synthetic_rl import s15_eval as EV  # noqa: E402
from synthetic_rl.s13_env import load_scenarios  # noqa: E402
from synthetic_rl.s15_env import MixedEnv  # noqa: E402
from synthetic_rl.models_v2 import SpatialMaskablePolicyV2  # noqa: E402,F401

OUT = os.path.join(ROOT, "research", "s15_mixed_training")
CK_MIG = os.path.join(ROOT, "research", "s14_group_observation", "s8c_migrated_v2_step0.zip")
CK_S8C = os.path.join(ROOT, "research", "s8_long_route_training", "runs", "arm_C_s11", "checkpoints", "step_0099840.zip")
TRAIN_SEEDS = [500000, 523757, 531676, 547514, 555433, 571271]
CKPTS = [5120, 10240, 20480, 30720]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


class Mon(BaseCallback):
    def __init__(self):
        super().__init__()
        self.rows = []

    def _on_rollout_start(self):
        self.t0 = time.time()

    def _on_rollout_end(self):
        self.roll = time.time() - self.t0
        self.t1 = time.time()

    def _on_step(self):
        return True

    def after(self, env):
        lg = self.model.logger.name_to_value
        row = {"decisions": int(self.model.num_timesteps), "rollout_s": round(self.roll, 2), "update_s": round(time.time() - self.t1, 2), **{k.split("/")[-1]: float(v) for k, v in lg.items() if k.startswith("train/") and isinstance(v, (int, float, np.floating))},
               "family_decisions": dict(env.decisions)}
        self.rows.append(row)
        print("[update]", json.dumps(row), flush=True)
        return all(math.isfinite(v) for v in row.values() if isinstance(v, float))


def evaluate(model, blind, tag, fixed_obs, fixed_masks):
    pol = EV.Policy(model, blind=blind)
    t0 = time.time()
    s7_sum, s7_rows = EV.eval_s7(pol)
    s13_sum, s13_rows = EV.eval_s13(pol)
    w = model.policy.mlp_extractor.trunk[0].weight.detach()
    rec = {"tag": tag, "s7": s7_sum, "s13": s13_sum, "wait_prob_fixed_s7_obs": pol.wait_prob(fixed_obs, fixed_masks),
           "group_channel_weights": {"abs_mean_v1": float(w[:, :15].abs().mean()), "abs_mean_group": float(w[:, 15:].abs().mean()), "abs_max_group": float(w[:, 15:].abs().max())}, "eval_s": round(time.time() - t0, 1)}
    return rec, {"s7": s7_rows, "s13": s13_rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["A", "B"])
    ap.add_argument("--seed", type=int, default=15)
    ap.add_argument("--decisions", type=int, default=30720)
    ap.add_argument("--deadline-min", type=float, default=33.0)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    assert a.decisions <= 30720
    torch.set_num_threads(a.threads)
    t_start = time.time()
    deadline = t_start + 60 * a.deadline_min
    run = os.path.join(OUT, "runs", f"arm_{a.arm}_s{a.seed}")
    os.makedirs(os.path.join(run, "checkpoints"), exist_ok=True)
    blind = a.arm == "B"
    sets = EV.s7_sets()
    train_sc = [s for s in load_scenarios() if s["game_seed"] in TRAIN_SEEDS]
    env = MixedEnv(sets["train"], train_sc, seed=a.seed, group_blind=blind)
    model = MaskablePPO.load(CK_MIG, env=env, device="cpu", seed=a.seed)
    model.set_logger(configure(run, ["csv"]))
    # init verification: parameters bitwise equal to the migrated checkpoint (which equals S8-C on the v1 channels, gate C)
    src = MaskablePPO.load(CK_MIG, device="cpu")
    same = all(torch.equal(model.policy.state_dict()[k], src.policy.state_dict()[k]) for k in src.policy.state_dict())
    log = {"arm": a.arm, "group_blind": blind, "seed": a.seed, "init": {"checkpoint": CK_MIG, "sha256": sha(CK_MIG), "source_S8C_sha256": sha(CK_S8C), "parameters_bitwise_equal_to_init": same, "n_params": sum(p.numel() for p in model.policy.parameters())},
           "hyperparameters": {"n_steps": model.n_steps, "batch_size": model.batch_size, "n_epochs": model.n_epochs, "target_kl": model.target_kl, "learning_rate": float(model.learning_rate), "ent_coef": model.ent_coef, "gamma": model.gamma, "gae_lambda": model.gae_lambda, "clip_range": 0.2, "threads": a.threads},
           "mixture": {"rule": "at reset choose the family with fewer cumulative decisions (target 50/50 decisions); S7 pools Gap-7 0.4 / Gap-10 0.4 / short(k=2,3,5) 0.2 from TRAIN manifests; S13 = 18 training scenarios of 6 game seeds"},
           "train_scenarios": [s["id"] for s in train_sc], "checkpoints": [], "evaluations": [], "updates": [], "stopped": None}
    fixed_obs, fixed_masks = EV.fixed_s7_observations()
    rows_dir = os.path.join(run, "rows")
    os.makedirs(rows_dir, exist_ok=True)
    rec, rows = evaluate(model, blind, "step_0", fixed_obs, fixed_masks)
    log["evaluations"].append({"decisions": 0, **rec})
    json.dump(rows, open(os.path.join(rows_dir, "step_0.json"), "w"))
    print(f"[eval 0] {json.dumps(rec)} ({time.time() - t_start:.0f}s)", flush=True)
    mon = Mon()
    done = 0
    for target in CKPTS:
        if target > a.decisions:
            break
        while done < target:
            if time.time() > deadline:
                log["stopped"] = f"wall-clock cap at {done} decisions"
                break
            torch_state = torch.get_rng_state()
            model.learn(total_timesteps=model.n_steps, callback=mon, reset_num_timesteps=False, progress_bar=False)
            done = int(model.num_timesteps)
            if not mon.after(env):
                log["stopped"] = f"non-finite statistic at {done}"
                break
        ck = os.path.join(run, "checkpoints", f"step_{done:07d}.zip")
        model.save(ck)
        log["checkpoints"].append({"decisions": done, "path": ck, "sha256": sha(ck), "mixture_stats": env.stats()})
        rs = torch.get_rng_state()
        rec, rows = evaluate(model, blind, f"step_{done}", fixed_obs, fixed_masks)
        torch.set_rng_state(rs)
        log["evaluations"].append({"decisions": done, **rec})
        json.dump(rows, open(os.path.join(rows_dir, f"step_{done}.json"), "w"))
        print(f"[eval {done}] {json.dumps(rec)} ({time.time() - t_start:.0f}s)", flush=True)
        if log["stopped"]:
            break
    log["updates"] = mon.rows
    log["mixture_stats_final"] = env.stats()
    if mon.rows:
        roll = sum(r["rollout_s"] for r in mon.rows); up = sum(r["update_s"] for r in mon.rows)
        log["throughput"] = {"updates": len(mon.rows), "decisions": done, "rollout_decisions_per_s": round(done / roll, 1), "update_s_per_512": round(up / len(mon.rows), 2), "train_decisions_per_s": round(done / (roll + up), 1), "eval_s_total": round(sum(e["eval_s"] for e in log["evaluations"]), 1)}
    log["wall_s"] = round(time.time() - t_start, 1)
    json.dump(log, open(os.path.join(run, "run.json"), "w"), indent=1, default=str)
    print("DONE", json.dumps({k: log[k] for k in ("throughput", "stopped", "wall_s", "mixture_stats_final") if k in log}), flush=True)


if __name__ == "__main__":
    main()
