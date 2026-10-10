"""python -m synthetic_rl.bench : throughput of the sandbox on an otherwise idle CPU (single process)."""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from synthetic_rl.env import SyntheticGapEnv, load_tasks  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
tasks = load_tasks(os.path.join(ROOT, "research", "synthetic_lab", "manifests", "gap1_train.jsonl"))
out = {}


def run(label, act_fn, seconds=4.0):
    env = SyntheticGapEnv(tasks)
    obs, _ = env.reset(seed=0)
    n = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        obs, r, te, tr, _ = env.step(act_fn(env, obs))
        n += 1
        if te or tr:
            obs, _ = env.reset()
    out[label] = n / (time.perf_counter() - t0)
    print(f"{label:48s} {out[label]:9.0f} steps/s", flush=True)


rng = np.random.default_rng(0)
run("env only, random legal action", lambda e, o: int(rng.choice(np.flatnonzero(e.action_masks()))))
run("env only, oracle action", lambda e, o: e.oracle_action())
from sb3_contrib import MaskablePPO  # noqa: E402
from synthetic_rl.models import SpatialMaskablePolicy  # noqa: E402
import torch  # noqa: E402
torch.set_num_threads(2)
for label, pol in (("env + flat MLP policy inference (batch 1)", "MlpPolicy"), ("env + spatial CNN policy inference (batch 1)", SpatialMaskablePolicy)):
    env0 = SyntheticGapEnv(tasks)
    model = MaskablePPO(pol, env0, n_steps=512, batch_size=256, verbose=0, device="cpu", seed=0)
    run(label, lambda e, o, m=model: int(m.predict(o, action_masks=e.action_masks(), deterministic=False)[0]), 5.0)
# full PPO training throughput (rollout + update), one process, idle CPU
for label, pol in (("PPO training, flat MLP (512x4 epochs)", "MlpPolicy"), ("PPO training, spatial CNN (512x4 epochs)", SpatialMaskablePolicy)):
    env0 = SyntheticGapEnv(tasks)
    model = MaskablePPO(pol, env0, n_steps=512, batch_size=256, n_epochs=4, verbose=0, device="cpu", seed=0)
    t0 = time.perf_counter()
    model.learn(4096)
    out[label] = model.num_timesteps / (time.perf_counter() - t0)
    print(f"{label:48s} {out[label]:9.0f} steps/s", flush=True)
json.dump(out, open(os.path.join(ROOT, "research", "synthetic_lab", "throughput_benchmark.json"), "w"), indent=2)
