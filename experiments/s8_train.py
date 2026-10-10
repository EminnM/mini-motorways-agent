"""S8 controlled long-route curriculum pilot (synthetic only). One run = one arm, one seed.
   python experiments/s8_train.py --arm A|B|C --seed 11 --smoke                      # 512-decision smoke test (not part of the budget)
   python experiments/s8_train.py --arm A|B|C --seed 11 --deadline-min 75 --reserve-min 7    # main run
Differences from the prepared S7 script (S7 files are untouched): evaluation schedule 0/10k/25k/50k/75k/100k on VALIDATION (deterministic AND stochastic);
global torch RNG saved/restored around every evaluation (evaluation cannot change the training stream); SB3 training metrics logged to CSV; wall-clock deadline;
preregistered checkpoint selection on validation only; one TEST evaluation of the selected checkpoint; native-legality diagnostic via s8_eval."""
import argparse
import collections
import gzip
import hashlib
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

OUT = os.path.join(ROOT, "research", "s8_long_route_training")
CFG = {a: os.path.join(ROOT, "research", "s7_long_route", "configs", f"arm_{a}.yaml") for a in "ABC"}
EVAL_AT = [0, 10240, 25088, 50176, 75264, 99840]      # multiples of the 512-decision rollout, closest to 0/10k/25k/50k/75k/100k; 99,840 = 195 rollouts <= the 100k cap


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def param_hash(module):
    h = hashlib.sha256()
    for k, v in module.state_dict().items():
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def opt_hash(opt):
    h = hashlib.sha256()
    for st in opt.state.values():
        for k in sorted(st):
            v = st[k]
            h.update(k.encode())
            h.update((v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v)).tobytes())
    return h.hexdigest()


def select_checkpoint(val_records):
    """PREREGISTERED rule (validation only): maximise mean(success@k7, success@k10) of the STOCHASTIC policy; tie -> higher pooled construction efficiency on
    successes at k in {7,10}; tie -> lower mean churn at k in {7,10}; tie -> earlier checkpoint."""
    best = None
    for r in val_records:
        s = r["stochastic"]
        succ = (s["7"]["success"] + s["10"]["success"]) / 2
        eff = [x for x in (s["7"]["mean_efficiency_on_success"], s["10"]["mean_efficiency_on_success"]) if x is not None]
        eff = sum(eff) / len(eff) if eff else 0.0
        churn = (s["7"]["mean_churn"] + s["10"]["mean_churn"]) / 2
        key = (-round(succ, 9), -round(eff, 9), round(churn, 9), r["decisions"])
        if best is None or key < best[0]:
            best = (key, r)
    return best[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list("ABC"))
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--deadline-min", type=float, default=75.0)
    ap.add_argument("--reserve-min", type=float, default=7.0, help="minutes kept for the final validation + TEST evaluation inside the deadline")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--run-root", default=None)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    t_start = time.time()
    cfg = yaml.safe_load(open(CFG[a.arm]))
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import configure
    from synthetic_rl import s8_eval as E
    from synthetic_rl.models import SpatialMaskablePolicy
    from s7_train import CurriculumEnv
    man = os.path.join(ROOT, "research", "s7_long_route", "manifests")
    load = lambda sp: [json.loads(l) for l in open(os.path.join(man, f"s7_{sp}.jsonl"))]
    train, val, test = load("train"), load("validation"), load("test")
    smoke = a.smoke
    run_root = a.run_root or os.path.join(OUT, "smoke" if smoke else "runs")
    run_dir = os.path.join(run_root, f"arm_{a.arm}_s{a.seed}")
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    stages = cfg["training"]["stages"]
    total = 512 if smoke else (cfg["training"]["total_decisions"] // 512) * 512 - (0 if cfg["training"]["total_decisions"] % 512 == 0 else 0)
    eval_at = [0, 512] if smoke else EVAL_AT
    val_eval_tasks = val[::9] if smoke else val                 # smoke: a small slice only
    env = CurriculumEnv.build(train, stages, a.seed)
    hp = cfg["ppo"]
    model = MaskablePPO(SpatialMaskablePolicy, env, learning_rate=hp["learning_rate"], n_steps=hp["n_steps"], batch_size=hp["batch_size"], n_epochs=hp["n_epochs"],
                        target_kl=hp["target_kl"], gamma=hp["gamma"], gae_lambda=0.95, clip_range=0.2, ent_coef=hp["ent_coef"], verbose=0, seed=a.seed, device="cpu")
    model.set_logger(configure(os.path.join(run_dir, "sb3"), ["csv"]))
    init = cfg["training"].get("init_checkpoint")
    init_report = None
    if init:
        import train_ppo
        init_report = train_ppo.init_from_checkpoint(model, os.path.join(ROOT, init["path"]), init.get("sha256"), os.path.join(run_dir, "init_report.json"))
    adv = cfg["training"].get("advance")
    plan = {"arm": a.arm, "seed": a.seed, "smoke": smoke, "total_decisions": total, "stages": stages, "advance": adv, "init_checkpoint": init, "eval_at": eval_at,
            "n_params": int(sum(p.numel() for p in model.policy.parameters())), "ppo": hp, "threads": a.threads, "deadline_min": a.deadline_min, "reserve_min": a.reserve_min,
            "manifest_sha256": open(os.path.join(man, "SHA256.txt")).read(), "config_sha256": sha(CFG[a.arm]),
            "source_sha256": {f: sha(os.path.join(ROOT, f)) for f in ("experiments/s8_train.py", "experiments/s7_train.py", "python/synthetic_rl/s7.py", "python/synthetic_rl/s8_eval.py", "python/synthetic_rl/models.py")},
            "validation_eval": "deterministic + stochastic (seed 0), all validation tasks", "selection_rule": "see PREREGISTERED_SELECTION.json"}
    json.dump(plan, open(os.path.join(run_dir, "plan.json"), "w"), indent=1)
    print(json.dumps({k: plan[k] for k in ("arm", "seed", "smoke", "total_decisions", "n_params")}), flush=True)

    # ---- instrumentation (read-only wrappers around the training env)
    ep_log = {"rejected": 0, "bad_reward": 0, "bad_boundary": 0, "episodes": 0, "successes": 0, "steps": 0, "len_over_horizon": 0}
    ep_len = {"n": 0}
    orig_step = env.step

    def logged_step(action):
        o, r, te, tr, info = orig_step(action)
        ep_log["steps"] += 1
        ep_len["n"] += 1
        if not info["accepted"]:
            ep_log["rejected"] += 1
        if r not in (0.0, 1.0) or (r == 1.0) != bool(te):
            ep_log["bad_reward"] += 1
        if te and tr:
            ep_log["bad_boundary"] += 1
        if te or tr:
            ep_log["episodes"] += 1
            ep_log["successes"] += int(te)
            if ep_len["n"] > env.horizon:
                ep_log["len_over_horizon"] += 1
            ep_len["n"] = 0
        return o, r, te, tr, info
    env.step = logged_step

    val_records = []
    ck_hashes = {}

    def save_ckpt(decisions):
        p = os.path.join(run_dir, "checkpoints", f"step_{decisions:07d}.zip")
        model.save(p)
        ck_hashes[os.path.basename(p)] = sha(p)
        with open(os.path.join(run_dir, "checkpoints", "SHA256.txt"), "a") as f:
            f.write(f"{ck_hashes[os.path.basename(p)]} *{os.path.basename(p)}\n")
        return p

    def run_eval(tag, decisions):
        """validation evaluation; training state (params, optimizer, global torch RNG, timesteps) is verified untouched"""
        t0 = time.time()
        before = (param_hash(model.policy), opt_hash(model.policy.optimizer), model.num_timesteps)
        rng_state = torch.get_rng_state()
        res, rows_all = {}, {}
        for mode, det in (("deterministic", True), ("stochastic", False)):
            rows = E.evaluate("model", val_eval_tasks, model=model, deterministic=det, seed=0)
            res[mode] = E.summarize(rows)
            rows_all[mode] = rows
        torch.set_rng_state(rng_state)
        after = (param_hash(model.policy), opt_hash(model.policy.optimizer), model.num_timesteps)
        rec = {"tag": tag, "decisions": int(decisions), "stage": env.stage, "stage_decisions": env.stage_decisions, "eval_seconds": round(time.time() - t0, 1), "wall_s": round(time.time() - t_start, 1),
               "training_state_unchanged": before == after, **res}
        p = save_ckpt(decisions)
        rec["checkpoint"] = os.path.basename(p)
        rec["checkpoint_sha256"] = ck_hashes[os.path.basename(p)]
        val_records.append(rec)
        with open(os.path.join(run_dir, "eval_validation.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
        with gzip.open(os.path.join(run_dir, "checkpoints", f"val_rows_step_{decisions:07d}.json.gz"), "wt") as f:
            json.dump(rows_all, f)
        st = res["stochastic"]
        print(f"[{a.arm}] eval {tag} @ {decisions}: stochastic success k=2..10 " + " ".join(f"{st[str(k)]['success']*100:.0f}" for k in (2, 3, 5, 7, 10)) +
              f" | det " + " ".join(f"{res['deterministic'][str(k)]['success']*100:.0f}" for k in (2, 3, 5, 7, 10)) + f" ({rec['eval_seconds']}s, wall {rec['wall_s']}s)", flush=True)
        return rec

    stop_at = t_start + 60 * (a.deadline_min - a.reserve_min)
    state = {"next_eval": 1, "stopped_by_deadline": False, "stage_history": []}
    entry = {"stage": 0, "lengths": stages[0], "entered_at_decisions": 0}

    class CB(BaseCallback):
        def _on_step(self):
            if not smoke and time.time() > stop_at:
                state["stopped_by_deadline"] = True
                return False
            return True

        def _on_rollout_start(self):        # called after the previous PPO update: parameters correspond to exactly nt trained decisions
            nt = self.model.num_timesteps
            if adv and env.stage < len(stages) - 1:
                s = env.stage_decisions
                if (len(env.recent) >= adv["window"] and np.mean(env.recent) >= adv["success_threshold"] and s >= adv["min_stage_decisions"]) or s >= adv["max_stage_decisions"]:
                    rec = {"left_stage": env.stage, "lengths": stages[env.stage], "decisions_spent_in_stage": s, "recent_success": float(np.mean(env.recent)) if env.recent else None,
                           "global_decisions": nt, "reason": "threshold" if (len(env.recent) >= adv["window"] and np.mean(env.recent) >= adv["success_threshold"] and s >= adv["min_stage_decisions"]) else "max_stage_decisions"}
                    state["stage_history"].append(rec)
                    with open(os.path.join(run_dir, "stages.jsonl"), "a") as f:
                        f.write(json.dumps(rec) + "\n")
                    env.stage += 1
                    env.stage_decisions = 0
                    env.recent.clear()
            while state["next_eval"] < len(eval_at) and nt >= eval_at[state["next_eval"]]:
                run_eval("periodic", nt)
                state["next_eval"] += 1
            roll = {"decisions": nt, "stage": env.stage, "stage_decisions": env.stage_decisions, "recent_success": float(np.mean(env.recent)) if env.recent else None, "episodes": ep_log["episodes"], "wall_s": round(time.time() - t_start, 1)}
            with open(os.path.join(run_dir, "rollouts.jsonl"), "a") as f:
                f.write(json.dumps(roll) + "\n")

    # ---- step-0 evaluation, then training
    smoke_info = {}
    if smoke:
        p0 = param_hash(model.policy)
        smoke_info["initial_param_hash"] = p0
        if init_report:
            src = MaskablePPO.load(os.path.join(ROOT, init["path"]), device="cpu")
            smoke_info["pretrained_strict_load_bitwise_equal"] = all(torch.equal(v, src.policy.state_dict()[k]) for k, v in model.policy.state_dict().items()) and init_report["n_tensors_loaded"] == len(model.policy.state_dict())
        smoke_info["optimizer_state_empty_before"] = len(model.policy.optimizer.state) == 0
    run_eval("step0", 0)
    if smoke:
        smoke_info["eval_left_training_state_unchanged"] = val_records[0]["training_state_unchanged"]
    t_train0 = time.time()
    model.learn(total_timesteps=total, callback=CB(), reset_num_timesteps=True)
    train_wall = time.time() - t_train0
    final_decisions = int(model.num_timesteps)
    if not val_records or val_records[-1]["decisions"] != final_decisions:
        run_eval("final" if not state["stopped_by_deadline"] else "final_deadline", final_decisions)
    if smoke:
        sb3 = {k: float(v) for k, v in model.logger.name_to_value.items() if isinstance(v, (int, float, np.floating))}     # last train() metrics (the CSV lags one iteration)
        pf = param_hash(model.policy)
        finite = all(torch.isfinite(v).all().item() for v in model.policy.state_dict().values())
        kl_ok = all(np.isfinite(sb3.get(k, 0.0)) for k in ("train/approx_kl", "train/value_loss", "train/policy_gradient_loss", "train/entropy_loss"))
        # checkpoint reload
        pth = os.path.join(run_dir, "checkpoints", f"step_{final_decisions:07d}.zip")
        m2 = MaskablePPO.load(pth, device="cpu")
        o, _ = env.reset()
        same = True
        for _ in range(10):
            mk = env.action_masks()
            a1 = int(model.predict(o, action_masks=mk, deterministic=True)[0])
            a2 = int(m2.predict(o, action_masks=mk, deterministic=True)[0])
            same &= (a1 == a2)
            o, *_ = orig_step(a1)
        smoke_info.update({"decisions_trained": final_decisions, "parameters_changed_by_ppo": pf != p0, "optimizer_state_nonempty_after": len(model.policy.optimizer.state) > 0,
                           "all_parameters_finite": finite, "sb3_last_row": {k: sb3.get(k) for k in ("train/approx_kl", "train/entropy_loss", "train/value_loss", "train/policy_gradient_loss", "train/explained_variance", "train/n_updates")},
                           "losses_finite": kl_ok, "episode_log": ep_log, "rejected_actions_in_rollout": ep_log["rejected"], "reward_and_boundary_violations": ep_log["bad_reward"] + ep_log["bad_boundary"] + ep_log["len_over_horizon"],
                           "eval_left_training_state_unchanged_both": all(r["training_state_unchanged"] for r in val_records), "checkpoint_reload_predictions_equal": same,
                           "reloaded_params_equal": param_hash(m2.policy) == pf, "stage_during_smoke": env.stage, "wall_s": round(time.time() - t_start, 1), "train_seconds": round(train_wall, 1)})
        ok = (smoke_info["parameters_changed_by_ppo"] and smoke_info["optimizer_state_nonempty_after"] and smoke_info["all_parameters_finite"] and smoke_info["losses_finite"] and
              smoke_info["rejected_actions_in_rollout"] == 0 and smoke_info["reward_and_boundary_violations"] == 0 and smoke_info["eval_left_training_state_unchanged_both"] and
              smoke_info["checkpoint_reload_predictions_equal"] and smoke_info["reloaded_params_equal"] and smoke_info.get("pretrained_strict_load_bitwise_equal", True) and smoke_info["optimizer_state_empty_before"])
        smoke_info["SMOKE_PASS"] = bool(ok)
        json.dump(smoke_info, open(os.path.join(run_dir, "smoke_result.json"), "w"), indent=1, default=str)
        print(json.dumps(smoke_info, default=str)[:1800], flush=True)
        sys.exit(0 if ok else 1)

    # ---- selection (validation only) and the single TEST evaluation of the selected checkpoint
    sel = select_checkpoint(val_records)
    json.dump({"selected": sel["checkpoint"], "decisions": sel["decisions"], "rule": "PREREGISTERED_SELECTION.json", "candidates": [{"checkpoint": r["checkpoint"], "decisions": r["decisions"],
               "stochastic_k7": r["stochastic"]["7"]["success"], "stochastic_k10": r["stochastic"]["10"]["success"]} for r in val_records]}, open(os.path.join(run_dir, "selection.json"), "w"), indent=1)
    msel = MaskablePPO.load(os.path.join(run_dir, "checkpoints", sel["checkpoint"]), device="cpu")
    t0 = time.time()
    test_res, test_rows = {}, {}
    for mode, det in (("deterministic", True), ("stochastic", False)):
        rows = E.evaluate("model", test, model=msel, deterministic=det, seed=0)
        test_res[mode] = E.summarize(rows)
        test_rows[mode] = rows
    json.dump({"checkpoint": sel["checkpoint"], "sha256": ck_hashes[sel["checkpoint"]], "decisions": sel["decisions"], "summary": test_res, "eval_seconds": round(time.time() - t0, 1)}, open(os.path.join(run_dir, "final_test.json"), "w"), indent=1)
    with gzip.open(os.path.join(run_dir, "final_test_rows.json.gz"), "wt") as f:
        json.dump(test_rows, f)
    total_wall = time.time() - t_start
    json.dump({"arm": a.arm, "seed": a.seed, "decisions": final_decisions, "stopped_by_deadline": state["stopped_by_deadline"], "stage_end": env.stage, "stage_decisions_in_final_stage": env.stage_decisions,
               "stage_history": state["stage_history"], "train_wall_s": round(train_wall, 1), "total_wall_s": round(total_wall, 1), "train_decisions_per_s": round(final_decisions / max(train_wall, 1e-9), 1),
               "last_ppo_metrics": {k: float(v) for k, v in model.logger.name_to_value.items() if isinstance(v, (int, float, np.floating))}, "episodes": ep_log["episodes"], "successes": ep_log["successes"], "rejected_actions": ep_log["rejected"], "checkpoint_hashes": ck_hashes, "threads": a.threads}, open(os.path.join(run_dir, "train_status.json"), "w"), indent=1)
    print(f"[{a.arm}] DONE decisions={final_decisions} wall={total_wall:.0f}s selected={sel['checkpoint']}", flush=True)


if __name__ == "__main__":
    main()
