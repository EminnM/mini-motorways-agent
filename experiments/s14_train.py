"""S14 Gate D: capped PPO feasibility continuation of the MIGRATED S8-C (v2 observation) on the synthetic multi-house environment. Synthetic only, one seed, <= 10,240 decisions.
   python experiments/s14_train.py --decisions 10240 --seed 14
Evaluations (fixed validation scenarios = game seeds 507919 / 579190, 6 scenarios x 5 env seeds, stochastic policy, torch seed fixed): frozen S8-C on the v1 env, migrated checkpoint at step 0 on
the v2 env (representation effect only; proven identical to S8-C by gate C), and the trained checkpoints; plus S7 validation Gap-7 / Gap-10 retention (stochastic and deterministic, seed 0)."""
import argparse
import csv
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
from synthetic_rl import obs_v2 as V  # noqa: E402
from synthetic_rl import s13_env as E13  # noqa: E402
from synthetic_rl import s7, s7_eval  # noqa: E402
from synthetic_rl.s13_baselines import episode, summarize  # noqa: E402
from synthetic_rl.s14_env import MultiHouseEnvV2  # noqa: E402
from synthetic_rl.models_v2 import SpatialMaskablePolicyV2  # noqa: E402,F401  (class must be importable for MaskablePPO.load)

OUT = os.path.join(ROOT, "research", "s14_group_observation")
CK_S8C = os.path.join(ROOT, "research", "s8_long_route_training", "runs", "arm_C_s11", "checkpoints", "step_0099840.zip")
CK_MIG = os.path.join(OUT, "s8c_migrated_v2_step0.zip")
TRAIN_SEEDS = [500000, 523757, 531676, 547514, 555433, 571271]
VAL_SEEDS = [507919, 579190]
EVAL_ENV_SEEDS = 5
EVAL_TORCH_SEED = 8008


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def eval_multi_house(model, v2: bool, scen_all, label):
    """stochastic policy on the fixed validation scenarios; same env seeds and torch seed for every model."""
    val_idx = [i for i, s in enumerate(scen_all) if s["game_seed"] in VAL_SEEDS]
    rows = []
    for idx in val_idx:
        for k in range(EVAL_ENV_SEEDS):
            env = (MultiHouseEnvV2 if v2 else E13.MultiHouseEnv)(scen_all, seed=0)
            torch.manual_seed(EVAL_TORCH_SEED + 100 * idx + k)
            pol = (lambda e: int(model.predict(e._obs_v2, action_masks=e.action_masks(), deterministic=False)[0])) if v2 else \
                  (lambda e: int(model.predict(e._obs, action_masks=e.action_masks(), deterministic=False)[0]))
            rows.append(episode(pol, idx, 7000 + 100 * idx + k, env=env))
    s = summarize(rows)
    s["label"] = label
    return s, rows


def eval_s7(model_for_v1, label):
    sets, _ = s7.build_task_sets()
    out = {"label": label}
    for k in (7, 10):
        for det in (False, True):
            r = s7_eval.evaluate("model", sets["validation"][k], model=model_for_v1, deterministic=det, seed=0)
            out[f"k{k}_{'det' if det else 'sto'}"] = s7_eval.summarize(r)[k]["success"]
    return out


class Monitor(BaseCallback):
    def __init__(self, abort_kl=0.2, abort_consecutive=3):
        super().__init__()
        self.t_roll = None
        self.rows = []
        self.bad = 0
        self.abort_kl, self.abort_consecutive = abort_kl, abort_consecutive
        self.abort_reason = None

    def _on_rollout_start(self):
        self.t_roll = time.time()

    def _on_rollout_end(self):
        self.roll_s = time.time() - self.t_roll
        self.t_upd = time.time()

    def _on_step(self):
        return True

    def _on_training_end(self):
        pass

    def after_update(self):
        lg = self.model.logger.name_to_value
        kl = lg.get("train/approx_kl")
        row = {"decisions": int(self.model.num_timesteps), "rollout_s": round(self.roll_s, 2), "update_s": round(time.time() - self.t_upd, 2),
               **{k.split("/")[-1]: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in lg.items() if k.startswith("train/")}}
        self.rows.append(row)
        finite = all(math.isfinite(v) for v in row.values() if isinstance(v, float))
        if not finite:
            self.abort_reason = "non-finite training statistic"
        if kl is not None and kl > self.abort_kl:
            self.bad += 1
            if self.bad >= self.abort_consecutive:
                self.abort_reason = f"approx_kl > {self.abort_kl} for {self.abort_consecutive} consecutive updates"
        else:
            self.bad = 0
        print(f"[update] {row}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decisions", type=int, default=10240)
    ap.add_argument("--seed", type=int, default=14)
    ap.add_argument("--eval-every", type=int, default=5120)
    a = ap.parse_args()
    assert a.decisions <= 10240, "S14 feasibility cap"
    torch.set_num_threads(2)
    run_dir = os.path.join(OUT, "runs", f"feasibility_s{a.seed}")
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    scen_all = E13.load_scenarios()
    train = [s for s in scen_all if s["game_seed"] in TRAIN_SEEDS]
    env = MultiHouseEnvV2(train, seed=a.seed, sampling="random")
    model = MaskablePPO.load(CK_MIG, env=env, device="cpu", seed=a.seed)
    model.set_logger(configure(run_dir, ["csv"]))
    hp = {"n_steps": model.n_steps, "batch_size": model.batch_size, "n_epochs": model.n_epochs, "target_kl": model.target_kl, "learning_rate": float(model.learning_rate), "ent_coef": model.ent_coef, "gamma": model.gamma, "gae_lambda": model.gae_lambda, "clip_range": 0.2}
    log = {"init": {"migrated_checkpoint": CK_MIG, "sha256": sha(CK_MIG), "source_S8C_sha256": sha(CK_S8C)}, "hyperparameters": hp, "seed": a.seed, "train_scenarios": [s["id"] for s in train], "validation_game_seeds": VAL_SEEDS,
           "eval_env_seeds": EVAL_ENV_SEEDS, "evaluations": [], "s7_retention": [], "updates": [], "abort": None}
    t0 = time.time()
    # --- baselines before training
    s8c = MaskablePPO.load(CK_S8C, device="cpu")
    ev, rows = eval_multi_house(s8c, False, scen_all, "frozen_S8C_v1_env")
    log["evaluations"].append({"decisions": 0, **ev})
    json.dump(rows, open(os.path.join(run_dir, "rows_frozen_S8C.json"), "w"))
    ev, rows = eval_multi_house(model, True, scen_all, "migrated_step0_v2_env")
    log["evaluations"].append({"decisions": 0, **ev})
    json.dump(rows, open(os.path.join(run_dir, "rows_step0.json"), "w"))
    log["s7_retention"].append({"decisions": 0, **eval_s7(s8c, "frozen_S8C")})
    log["s7_retention"].append({"decisions": 0, **eval_s7(V.V2ModelOnV1Env(model), "migrated_step0")})
    print("[eval 0]", json.dumps(log["evaluations"]), json.dumps(log["s7_retention"]), f"{time.time() - t0:.0f}s", flush=True)
    # --- training in eval_every chunks
    mon = Monitor()
    done_dec = 0
    while done_dec < a.decisions and not mon.abort_reason:
        chunk = min(a.eval_every, a.decisions - done_dec)
        torch_state = torch.get_rng_state()
        n_before = len(mon.rows)
        # SB3 has no post-update hook: learn() one rollout at a time so update stats can be captured per update
        for _ in range(chunk // model.n_steps):
            model.learn(total_timesteps=model.n_steps, callback=mon, reset_num_timesteps=False, progress_bar=False)
            mon.after_update()
            if mon.abort_reason:
                break
        done_dec = int(model.num_timesteps)
        ck = os.path.join(run_dir, "checkpoints", f"step_{done_dec:07d}.zip")
        model.save(ck)
        torch_state_train = torch.get_rng_state()
        ev, rows = eval_multi_house(model, True, scen_all, f"trained_{done_dec}")
        log["evaluations"].append({"decisions": done_dec, "checkpoint": ck, "sha256": sha(ck), **ev})
        json.dump(rows, open(os.path.join(run_dir, f"rows_step{done_dec}.json"), "w"))
        log["s7_retention"].append({"decisions": done_dec, **eval_s7(V.V2ModelOnV1Env(model), f"trained_{done_dec}")})
        torch.set_rng_state(torch_state_train)
        print(f"[eval {done_dec}]", json.dumps(log["evaluations"][-1]), json.dumps(log["s7_retention"][-1]), f"{time.time() - t0:.0f}s", flush=True)
    log["updates"] = mon.rows
    log["abort"] = mon.abort_reason
    upd = [r for r in mon.rows]
    if upd:
        roll = sum(r["rollout_s"] for r in upd)
        up = sum(r["update_s"] for r in upd)
        log["throughput"] = {"updates": len(upd), "decisions": done_dec, "rollout_s_total": round(roll, 1), "update_s_total": round(up, 1), "rollout_decisions_per_s": round(done_dec / roll, 1), "update_s_per_512": round(up / len(upd), 2),
                             "training_decisions_per_s_incl_updates": round(done_dec / (roll + up), 1), "projected_s_per_100k": round(100000 / (done_dec / (roll + up)), 0), "threads": 2}
    log["wall_s"] = round(time.time() - t0, 1)
    # first-layer weights on the new channels: did the network start using them?
    w = model.policy.mlp_extractor.trunk[0].weight.detach()
    log["new_channel_weights"] = {"abs_mean_v1_channels": float(w[:, :15].abs().mean()), "abs_mean_group_channels": float(w[:, 15:].abs().mean()), "abs_max_group_channels": float(w[:, 15:].abs().max())}
    json.dump(log, open(os.path.join(run_dir, "feasibility.json"), "w"), indent=1, default=str)
    print("DONE", json.dumps({k: log[k] for k in ("throughput", "abort", "wall_s", "new_channel_weights") if k in log}), flush=True)


if __name__ == "__main__":
    main()
