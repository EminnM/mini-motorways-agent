"""S31 evaluation on the 12 held-out seeds through the SAME hybrid env wrapper: policies = planner B (always PLAN), random strategy (uniform over PLAN / WAIT / random legal EDIT), and frozen
checkpoints (deterministic and stochastic). One instance per process; split seeds with --idx for parallelism.   python research/s31_hybrid_native_rl/code/evaluate.py --policy ckpt:<path> --mode sto --idx 0,1,2"""
import argparse
import gzip
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
from hybrid_env import HybridNativeEnv, PLAN, EVAL_SEEDS, ROOT  # noqa: E402
from synthetic_rl import layout as L  # noqa: E402

OUT = os.path.join(ROOT, "research", "s31_hybrid_native_rl", "eval")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)          # plannerB | random | ckpt:<path>
    ap.add_argument("--mode", default="sto")            # det | sto (checkpoints only)
    ap.add_argument("--idx", default=",".join(str(i) for i in range(12)))
    ap.add_argument("--tag", default=None)
    ap.add_argument("--torch-seed", type=int, default=8008)
    ap.add_argument("--visible", action="store_true", help="open a game window and pace decisions in real time (viewing only; results written with tag prefix vis_)")
    a = ap.parse_args()
    import torch
    torch.set_num_threads(1)
    model = None
    if a.policy.startswith("ckpt:"):
        from sb3_contrib import MaskablePPO
        from hybrid_policy import HybridMaskablePolicy  # noqa: F401
        model = MaskablePPO.load(a.policy[5:], device="cpu")
    tag = a.tag or (os.path.basename(a.policy.replace("ckpt:", "").replace("\\", "/")).replace(".zip", "") + f"_{a.mode}" if model else a.policy)
    if a.visible:
        tag = "vis_" + tag
    os.makedirs(OUT, exist_ok=True)
    logs = []
    env = HybridNativeEnv(EVAL_SEEDS, f"s31_eval_{tag}_{a.idx.replace(',', '')}", max_decisions=600, log_fn=logs.append, visible=a.visible, realtime=a.visible)
    t0 = time.time()
    for k in [int(x) for x in a.idx.split(",")]:
        seed = EVAL_SEEDS[k]
        rng = random.Random(seed)
        torch.manual_seed(a.torch_seed + k)
        obs, _ = env.reset(options={"seed": seed})
        done = False
        trace = []
        while not done:
            m = env.action_masks()
            if a.policy == "plannerB":
                act = PLAN if m[PLAN] else int(np.flatnonzero(m[:L.N_ACTIONS])[0])
            elif a.policy == "random":
                if not m[PLAN]:
                    act = int(np.flatnonzero(m[:L.N_ACTIONS])[0])
                else:
                    c = rng.choice(["PLAN", "WAIT", "EDIT"])
                    legal_edit = [i for i in np.flatnonzero(m[1:L.UPGRADE_BASE]) + 1]
                    act = PLAN if c == "PLAN" else (0 if c == "WAIT" or not legal_edit else int(rng.choice(legal_edit)))
            else:
                act, _ = model.predict(obs, action_masks=m, deterministic=(a.mode == "det"))
                act = int(act)
            obs, r, te, tr, info = env.step(act)
            trace.append(info["kind"][0])
            done = te or tr
        ep = dict(logs[-1], policy=tag, trace="".join(trace))
        with gzip.open(os.path.join(OUT, f"{tag}_{seed}.json.gz"), "wt") as f:
            json.dump(ep, f)
        print(f"[{tag}] seed {seed}: t {ep['t']} score {ep['score']} terminated {ep['terminated']} plan/edit/wait {ep['plan']}/{ep['edit']}/{ep['wait']} destroyed {ep['destroyed']} illegal {ep['illegal']} ({time.time() - t0:.0f}s)", flush=True)
    env.close()


if __name__ == "__main__":
    main()
