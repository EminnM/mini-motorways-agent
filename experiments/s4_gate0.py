"""S4 Gate 0: checkpoint + interface validation for the native spatial-CNN pipeline.
  python experiments/s4_gate0.py offline     # no game: logits/values/masks/probs/actions vs the original S1 inference, MLP path, fwd/bwd
  python experiments/s4_gate0.py native      # one headless game: logging non-interference (paired deterministic native rollouts)
Writes research/native_cnn_gate0/gate0_{offline,native}.json. No optimizer step, no PPO update, no checkpoint is modified."""
import argparse
import gzip
import hashlib
import json
import os
import sys
import time
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from experiments import paths  # noqa: E402,F401  (python/ on sys.path)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import gymnasium as gym  # noqa: E402
from gymnasium import spaces  # noqa: E402

OUT = os.path.join(ROOT, "research", "native_cnn_gate0")
STATES = os.path.join(ROOT, "research", "synthetic_to_native", "native_states_validation_0of1.jsonl.gz")
S1 = {11: "runs/2026-10-08_140406_syn_s1_cnn_s11/checkpoints/step_050000.zip",
      12: "runs/2026-10-08_140406_syn_s1_cnn_s12/checkpoints/step_050000.zip",
      13: "runs/2026-10-08_140406_syn_s1_cnn_s13/checkpoints/final.zip"}
S1_SHA = {11: "c8a7930eb12e9a204be02fa69e98e0dfc6969652555c5e390839bec225186fdc", 12: "980f95f7ce28bfebebdb8c60dbd832c7dc12252c2320304d61fd87dcfa8c603c",
          13: "41c1e085622b8f0d2e70274b2a6f05fedc94266124c1c74f9b39645ade417682"}
MLP_CK = os.path.join(ROOT, "research", "frozen_baselines", "native_reward_connected_49152.zip")
R = {}


def check(name, ok, detail, **extra):
    R[name] = {"passed": bool(ok), "detail": detail, **extra}
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


class DummyEnv(gym.Env):
    """Spaces identical to the native env (Box(-1,1,(11666,)), Discrete(3111)); never touches the game."""
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(11666,), dtype=np.float32)
        self.action_space = spaces.Discrete(3111)

    def reset(self, *, seed=None, options=None):
        return np.zeros(11666, np.float32), {}

    def step(self, a):
        return np.zeros(11666, np.float32), 0.0, False, False, {}

    def action_masks(self):
        return np.ones(3111, bool)


def pipeline_args(net, seed):
    return types.SimpleNamespace(net=net, seed=seed, learning_rate=3e-4, n_steps=512, batch_size=256, n_epochs=4, target_kl=0.03, gamma=0.99, ent_coef=0.01)


def load_states():
    import mini_motorways_env as M
    obs, masks, raws = [], [], []
    for line in gzip.open(STATES, "rt"):
        r = json.loads(line)
        obs.append(M.encode_observation(r["raw"]))
        m = np.zeros(3111, bool)
        m[r["legal_indices"]] = True
        masks.append(m)
        raws.append(r)
    return np.stack(obs), np.stack(masks), raws


def fwd(model, obs, masks):
    pol = model.policy
    with torch.no_grad():
        t = torch.as_tensor(obs, dtype=torch.float32)
        feats = pol.extract_features(t)
        latent_pi, latent_vf = pol.mlp_extractor(feats)
        logits = pol.action_net(latent_pi)
        values = pol.predict_values(t)
        dist = pol.get_distribution(t, action_masks=masks)
        probs = dist.distribution.probs.numpy()
    acts = np.array([int(model.predict(obs[i], action_masks=masks[i], deterministic=True)[0]) for i in range(len(obs))])
    return logits.numpy(), values.numpy().ravel(), probs, acts


def offline():
    import train_ppo as TP
    from sb3_contrib import MaskablePPO
    os.makedirs(OUT, exist_ok=True)
    obs, masks, raws = load_states()
    check("0_saved_native_states", len(obs) == 120 and obs.shape[1] == 11666 and (masks.sum(1) > 0).all(),
          f"{len(obs)} saved raw native states (the S2 interface check covered 156 = these 120 + 36 post-action states whose raw observations were not persisted); obs {obs.shape}, legal counts {masks.sum(1).min()}..{masks.sum(1).max()}")
    # ---- spatial: pipeline vs original S1 inference
    for seed, rel in S1.items():
        p = os.path.join(ROOT, rel)
        ref = MaskablePPO.load(p, device="cpu")                           # original S1 inference implementation
        model = TP.build_model(pipeline_args("spatial", seed), DummyEnv())  # new training pipeline
        rep = TP.init_from_checkpoint(model, p, S1_SHA[seed], os.path.join(OUT, f"init_report_s{seed}.json"))
        a = fwd(ref, obs, masks)
        b = fwd(model, obs, masks)
        d_logit = float(np.abs(a[0] - b[0]).max())
        d_val = float(np.abs(a[1] - b[1]).max())
        d_prob = float(np.abs(a[2] - b[2]).max())
        act_match = int((a[3] == b[3]).sum())
        legal_ok = bool(masks[np.arange(len(obs)), b[3]].all())
        illegal_mass = float((b[2] * (~masks)).sum(1).max())
        opt = model.policy.optimizer
        fresh = (len(opt.state) == 0 and abs(opt.param_groups[0]["lr"] - 3e-4) < 1e-12 and model.num_timesteps == 0 and abs(model.lr_schedule(1.0) - 3e-4) < 1e-12)
        check(f"1_s1_cnn_seed{seed}_pipeline_equals_original_inference",
              d_logit <= 1e-6 and d_val <= 1e-6 and d_prob <= 1e-6 and act_match == len(obs) and legal_ok and illegal_mass == 0.0 and fresh,
              f"max|dlogit|={d_logit:.3g} max|dvalue|={d_val:.3g} max|dprob|={d_prob:.3g}; deterministic actions equal {act_match}/{len(obs)}; chosen actions all legal={legal_ok}; "
              f"max probability mass on illegal actions={illegal_mass:.3g}; {rep['n_tensors_loaded']} tensors / {rep['n_parameters_loaded']} params loaded bitwise-equal; optimizer fresh={fresh}",
              max_abs_diff={"logits": d_logit, "values": d_val, "masked_probs": d_prob}, sha256=rep["sha256"])
    # ---- fail-loud tests
    fails = {}
    bad = TP.build_model(pipeline_args("spatial", 1), DummyEnv())
    for name, fn in (("wrong_sha", lambda: TP.init_from_checkpoint(bad, os.path.join(ROOT, S1[11]), "0" * 64)),
                     ("mlp_checkpoint_into_spatial", lambda: TP.init_from_checkpoint(bad, MLP_CK)),
                     ("net_mlp_with_init", lambda: TP.init_from_checkpoint(TP.build_model(pipeline_args("mlp", 1), DummyEnv()), os.path.join(ROOT, S1[11])))):
        try:
            fn()
            fails[name] = "NO ERROR"
        except Exception as e:
            fails[name] = f"{type(e).__name__}: {str(e)[:90]}"
    before = hashlib.sha256(b"".join(v.numpy().tobytes() for v in bad.policy.state_dict().values())).hexdigest()
    check("2_incompatible_checkpoints_fail_loudly", all(v != "NO ERROR" for v in fails.values()), json.dumps(fails))
    # ---- existing MLP path
    ours = TP.build_model(pipeline_args("mlp", 7), DummyEnv())
    old_mod = types.SimpleNamespace()
    ref_mlp = MaskablePPO("MlpPolicy", DummyEnv(), policy_kwargs=None, learning_rate=3e-4, n_steps=512, batch_size=256, n_epochs=4, target_kl=0.03, gamma=0.99,
                          gae_lambda=0.95, clip_range=0.2, ent_coef=0.01, verbose=1, seed=7)       # the pre-S4 construction, verbatim
    same_init = all(torch.equal(v, ref_mlp.policy.state_dict()[k]) for k, v in ours.policy.state_dict().items()) and set(ours.policy.state_dict()) == set(ref_mlp.policy.state_dict())
    cnn_new = TP.build_model(pipeline_args("cnn", 7), DummyEnv())
    cnn_old = MaskablePPO("MlpPolicy", DummyEnv(), policy_kwargs=dict(features_extractor_class=TP.GridExtractor, features_extractor_kwargs=dict(features_dim=256), net_arch=dict(pi=[256], vf=[256])),
                          learning_rate=3e-4, n_steps=512, batch_size=256, n_epochs=4, target_kl=0.03, gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01, verbose=1, seed=7)
    same_cnn = all(torch.equal(v, cnn_old.policy.state_dict()[k]) for k, v in cnn_new.policy.state_dict().items())
    fm = MaskablePPO.load(MLP_CK, custom_objects={"GridExtractor": TP.GridExtractor})
    fm_out = fwd(fm, obs, masks)
    arch_ok = True
    try:
        ours.policy.load_state_dict(fm.policy.state_dict(), strict=True)   # frozen checkpoint's weights fit the (unchanged) MLP construction
        again = fwd(ours, obs, masks)
        mlp_equal = bool(np.abs(again[0] - fm_out[0]).max() <= 1e-6 and (again[3] == fm_out[3]).all())
    except Exception as e:
        arch_ok, mlp_equal = False, False
    check("3_existing_mlp_path_unchanged", same_init and same_cnn and arch_ok and mlp_equal and bool(masks[np.arange(len(obs)), fm_out[3]].all()),
          f"build_model('mlp'/'cnn', seed 7) initial weights bitwise equal to the pre-S4 construction: mlp={same_init}, cnn={same_cnn}; frozen MLP checkpoint (sha {sha(MLP_CK)[:10]}) loads, "
          f"strict-loads into a freshly built MLP with identical logits={mlp_equal}; its {len(obs)} deterministic actions are all legal")
    # ---- random spatial: forward/backward only
    m = TP.build_model(pipeline_args("spatial", 11), DummyEnv())
    pre = hashlib.sha256(b"".join(v.numpy().tobytes() for v in m.policy.state_dict().values())).hexdigest()
    rng = np.random.default_rng(0)
    idx = rng.choice(len(obs), 32, replace=False)
    acts = torch.as_tensor([int(rng.choice(np.flatnonzero(masks[i]))) for i in idx])
    t0 = time.time()
    values, logp, ent = m.policy.evaluate_actions(torch.as_tensor(obs[idx]), acts, action_masks=masks[idx])
    loss = -(logp).mean() - 0.01 * ent.mean() + 0.5 * (values.squeeze(-1) ** 2).mean()
    loss.backward()
    dt = time.time() - t0
    grads = [p.grad for p in m.policy.parameters()]
    finite = all(g is not None and torch.isfinite(g).all() for g in grads)
    nz = sum(int((g.abs() > 0).any()) for g in grads)
    m.policy.zero_grad()
    post = hashlib.sha256(b"".join(v.numpy().tobytes() for v in m.policy.state_dict().values())).hexdigest()
    check("4_random_spatial_cnn_forward_backward", finite and nz > 0 and pre == post and bool(torch.isfinite(loss)),
          f"random-init spatial CNN on 32 saved native states: loss {float(loss):.4f}, entropy {float(ent.mean()):.3f} nats, all {len(grads)} parameter grads finite, {nz}/{len(grads)} tensors have non-zero grad, "
          f"no optimizer step (weights unchanged), fwd+bwd {1000 * dt:.0f} ms/32 states",
          n_params=int(sum(p.numel() for p in m.policy.parameters())))
    # ---- S1 checkpoints untouched
    ok = all(sha(os.path.join(ROOT, rel)) == S1_SHA[s] for s, rel in S1.items())
    check("5_s1_checkpoints_unmodified", ok, "SHA-256 of the three selected S1 checkpoints equal their recorded values")
    json.dump(R, open(os.path.join(OUT, "gate0_offline.json"), "w"), indent=1)
    sys.exit(0 if all(v["passed"] for v in R.values()) else 1)


def rollout(env_factory, seed, model, n_steps, diag):
    """Deterministic scripted rollout: frozen S1 CNN (argmax) for n_steps decisions. Returns per-step records."""
    env = env_factory(diag)
    obs, info = env.reset(seed=seed)
    rec = {"obs": [hashlib.sha1(obs.tobytes()).hexdigest()], "mask": [], "act": [], "rew": [], "term": [], "applied": []}
    t_diag = 0.0
    for _ in range(n_steps):
        mask = env.action_masks()
        rec["mask"].append(hashlib.sha1(np.packbits(mask).tobytes()).hexdigest())
        a, _ = model.predict(obs, action_masks=mask, deterministic=True)
        a = int(a)
        t0 = time.time()
        obs, r, te, tr, info = env.step(a)
        t_diag += time.time() - t0
        rec["act"].append(a)
        rec["rew"].append(float(r))
        rec["term"].append(bool(te or tr))
        rec["applied"].append(info.get("appliedAction"))
        rec["obs"].append(hashlib.sha1(obs.tobytes()).hexdigest())
        if te or tr:
            break
    rec["step_wall"] = t_diag
    return env, rec


def native():
    from sb3_contrib import MaskablePPO
    from experiments import gamectl, recording
    import mini_motorways_env as M
    os.makedirs(OUT, exist_ok=True)
    model = MaskablePPO.load(os.path.join(ROOT, S1[13]), device="cpu")
    seeds = [182150, 205907, 229664]
    N = 60
    tmp = os.path.join(OUT, "_noninterference_episodes")
    os.makedirs(tmp, exist_ok=True)
    results = []
    with gamectl.GameInstance("s4g0") as g:
        def factory(mode):
            base = M.MiniMotorwaysEnv(max_decisions=600, decision_delta=1.0, request_path=os.path.join(g.ipc_dir, "request.json"),
                                      response_path=os.path.join(g.ipc_dir, "response.json"), start_mode="NORMAL_START")
            if mode == "plain":
                return base
            meta = {"initialization": "normal", "start_gap": 0, "reward_mode": "native_score", "decision_interval_seconds": 1.0, "diagnostics": mode == "diag"}
            return recording.RecorderEnv(base, os.path.join(tmp, f"{mode}.jsonl"), meta)
        for seed in seeds:
            runs = {}
            for mode in ("plain", "diag", "plain2", "recorder_nodiag"):
                env, rec = rollout(factory if mode not in ("plain2",) else (lambda m: factory("plain")), seed, model, N, mode)
                runs[mode] = rec
                if hasattr(env, "close"):
                    try:
                        env.close()
                    except Exception:
                        pass
            ref = runs["plain"]
            cmp = {}
            for mode, rec in runs.items():
                cmp[mode] = {k: rec[k] == ref[k] for k in ("obs", "mask", "act", "rew", "term", "applied")}
            results.append({"seed": seed, "steps": len(ref["act"]), "equal_to_plain": cmp, "per_step_ms": {m: 1000 * r["step_wall"] / max(len(r["act"]), 1) for m, r in runs.items()},
                            "actions_first20": ref["act"][:20], "nonzero_rewards": [(i, r) for i, r in enumerate(ref["rew"]) if r]})
            print("seed", seed, "steps", len(ref["act"]), {m: all(v.values()) for m, v in cmp.items()}, flush=True)
    rows = [json.loads(l) for l in open(os.path.join(tmp, "diag.jsonl"))] if os.path.exists(os.path.join(tmp, "diag.jsonl")) else []
    all_equal = all(all(v.values()) for r in results for v in r["equal_to_plain"].values())
    floor = all(all(r["equal_to_plain"]["plain2"].values()) for r in results)
    diag_err = sum(r.get("diag_errors", 0) for r in rows)
    check("6_logging_does_not_change_native_rollouts", all_equal,
          f"3 seeds x {N} deterministic decisions (frozen S1 CNN argmax), runs: plain env / RecorderEnv+diagnostics / plain again / RecorderEnv without diagnostics: observations, legal masks, actions, rewards, "
          f"termination and applied actions identical across all four = {all_equal}; native determinism floor (plain vs plain again) = {floor}; diag_errors={diag_err}; episode rows written by the diagnostic run: {len(rows)}",
          per_seed=results)
    json.dump(R, open(os.path.join(OUT, "gate0_native.json"), "w"), indent=1)
    sys.exit(0 if all(v["passed"] for v in R.values()) else 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["offline", "native"])
    a = ap.parse_args()
    offline() if a.mode == "offline" else native()
