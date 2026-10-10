"""S8 arrangement benchmark (copy of the S7 one with a thread option); S7 Gate C throughput benchmark WITHOUT training: component timings (no PPO learn() call, throwaway random-init model never evaluated on the task, no task learning).
   python experiments/s7_benchmark.py --tag solo|par1|par2|par3"""
import argparse, json, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
import numpy as np, torch
from synthetic_rl import s7
from synthetic_rl.models import SpatialMaskablePolicy
from sb3_contrib import MaskablePPO
ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="solo"); ap.add_argument("--threads", type=int, default=1); a = ap.parse_args(); torch.set_num_threads(a.threads)
OUT = os.path.join(ROOT, "research", "s7_long_route")
tasks = [json.loads(l) for l in open(os.path.join(OUT, "manifests", "s7_train.jsonl"))]
tasks = [t for t in tasks if t["k_min"] in (7, 10)]
env = s7.LongRouteEnv(tasks, sampling="random", seed=0)
env.reset()
rng = np.random.default_rng(0)
def steps(n, policy):
    t0 = time.time(); obs = env._obs.copy()
    for _ in range(n):
        m = env.action_masks()
        if policy is None: a_ = int(rng.choice(np.flatnonzero(m)))
        else: a_, _ = policy.predict(obs, action_masks=m, deterministic=False); a_ = int(a_)
        obs, r, te, tr, _ = env.step(a_)
        if te or tr: obs, _ = env.reset()
    return n / (time.time() - t0)
res = {"tag": a.tag}
res["env_random_steps_per_s"] = steps(4000, None)
model = MaskablePPO(SpatialMaskablePolicy, env, n_steps=512, batch_size=256, n_epochs=4, seed=0, device="cpu", verbose=0)   # throwaway
res["env_plus_cnn_inference_steps_per_s"] = steps(1500, model)
# one PPO minibatch update cost on real observations (fwd + loss + bwd + Adam), throwaway weights
obs_l, mask_l, act_l = [], [], []
env.reset(); obs = env._obs.copy()
for _ in range(256):
    m = env.action_masks().copy(); a_ = int(rng.choice(np.flatnonzero(m)))
    obs_l.append(obs.copy()); mask_l.append(m); act_l.append(a_)
    obs, r, te, tr, _ = env.step(a_)
    if te or tr: obs, _ = env.reset()
O = torch.as_tensor(np.stack(obs_l)); Mk = np.stack(mask_l); A = torch.as_tensor(act_l)
pol = model.policy; opt = pol.optimizer
ts = []
for i in range(12):
    t0 = time.time()
    v, lp, ent = pol.evaluate_actions(O, A, action_masks=Mk)
    loss = -(lp).mean() - 0.01 * ent.mean() + 0.5 * (v.squeeze(-1) ** 2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    ts.append(time.time() - t0)
res["ppo_minibatch_256_update_ms"] = 1000 * float(np.median(ts[2:]))
t_env_inf = 1.0 / res["env_plus_cnn_inference_steps_per_s"]
rollout = 512 * t_env_inf; upd = 4 * 2 * (res["ppo_minibatch_256_update_ms"] / 1000)
res["predicted_ppo_training_steps_per_s"] = 512 / (rollout + upd)
res["predicted_s_per_100k_decisions"] = 1e5 / res["predicted_ppo_training_steps_per_s"]
json.dump(res, open(os.path.join(OUT, f"../s8_long_route_training/bench_{a.tag}.json"), "w"), indent=1)
print(json.dumps(res))
