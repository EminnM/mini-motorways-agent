"""S34 evaluation (postfix_v8). Two benchmarks, both through the unchanged HybridNativeEnv wrapper:
  --bench dev   : the 12 designated seeds, 600-decision cap, det (1 rollout) / sto (torch seed 8008+i)      -> eval_dev/<tag>_<seed>.json.gz
  --bench fresh : 8 fresh seeds 900000+7919*i, 1,500-decision horizon, det / sto rollout r (torch seed 20000+1000r+i) -> eval_fresh/<tag>_r<r>_<seed>.json.gz
Tag = <run>__<checkpoint>_<mode> (run = parent directory of the checkpoints folder: hybrid_s31 / rep1_seed131 / rep2_seed231) or plannerB.
Stop conditions checked per episode: illegal action, reward-accounting mismatch (sum rewards != score + penalty), missing playableAreaAtTime export.
   python research/s34_replication/code/eval_rep.py --bench fresh --policy ckpt:<path> --mode sto --rollout 0 --idx 0,1,2,3"""
import argparse
import gzip
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(ROOT, "research", "s31_hybrid_native_rl", "code"))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from hybrid_env import HybridNativeEnv, PLAN, EVAL_SEEDS  # noqa: E402
from synthetic_rl import layout as L  # noqa: E402

FRESH_SEEDS = [900000 + 7919 * i for i in range(8)]
HORIZON = {"dev": 600, "fresh": 1500}
DLL = r"C:\Program Files (x86)\Steam\steamapps\common\Mini Motorways\Bootstrap\MiniMotorwaysBootstrap.dll"
V8 = "2aa42367cbb14babce31a6063038f50be9b3535bd7bc23d8dbbdbb65533aa00a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, choices=["dev", "fresh"])
    ap.add_argument("--policy", required=True)
    ap.add_argument("--mode", default="sto")
    ap.add_argument("--rollout", type=int, default=0)
    ap.add_argument("--idx", default=None)
    a = ap.parse_args()
    assert hashlib.sha256(open(DLL, "rb").read()).hexdigest() == V8, "deployed DLL is not postfix_v8"
    torch.set_num_threads(1)
    seeds = EVAL_SEEDS if a.bench == "dev" else FRESH_SEEDS
    idx = [int(x) for x in a.idx.split(",")] if a.idx else list(range(len(seeds)))
    model = None
    if a.policy.startswith("ckpt:"):
        from sb3_contrib import MaskablePPO
        from hybrid_policy import HybridMaskablePolicy  # noqa: F401
        p = os.path.normpath(a.policy[5:])
        model = MaskablePPO.load(p, device="cpu")
        run = os.path.basename(os.path.dirname(os.path.dirname(p)))
        tag = f"{run}__{os.path.basename(p).replace('.zip', '')}_{a.mode}"
    else:
        tag = a.policy
    out = os.path.join(ROOT, "research", "s34_replication", "eval_" + a.bench)
    os.makedirs(out, exist_ok=True)
    logs = []
    env = HybridNativeEnv(seeds, f"s34e_{a.bench}_{tag}_r{a.rollout}_{''.join(map(str, idx))}"[:60], max_decisions=HORIZON[a.bench], log_fn=logs.append)
    t0 = time.time()
    for k in idx:
        seed = seeds[k]
        tseed = (8008 + k) if a.bench == "dev" else (20000 + 1000 * a.rollout + k)
        torch.manual_seed(tseed)
        obs, _ = env.reset(options={"seed": seed})
        assert env.raw["globals"].get("playableAreaAtTime") is not None, "environment-version mismatch: playableAreaAtTime missing"
        done = False
        rsum = 0.0
        trace = []
        while not done:
            m = env.action_masks()
            if model is None:
                act = PLAN if m[PLAN] else int(np.flatnonzero(m[:L.N_ACTIONS])[0])
            else:
                act, _ = model.predict(obs, action_masks=m, deterministic=(a.mode == "det"))
                act = int(act)
            obs, r, te, tr, info = env.step(act)
            rsum += float(r)
            trace.append(info["kind"][0])
            done = te or tr
        ep = dict(logs[-1], policy=tag, bench=a.bench, rollout=a.rollout, torch_seed=tseed, horizon=HORIZON[a.bench], env_version="postfix_v8", trace="".join(trace))
        assert ep["illegal"] == 0, f"STOP: illegal action on seed {seed}"
        assert abs(rsum - (ep["score"] + ep["penalty"])) < 1e-6, f"STOP: reward accounting mismatch on seed {seed}: {rsum} vs {ep['score']} + {ep['penalty']}"
        name = f"{tag}_{seed}.json.gz" if a.bench == "dev" else f"{tag}_r{a.rollout}_{seed}.json.gz"
        with gzip.open(os.path.join(out, name), "wt") as f:
            json.dump(ep, f)
        print(f"[{a.bench} {tag} r{a.rollout}] seed {seed}: t {ep['t']} score {ep['score']} term {ep['terminated']} plan/edit/wait {ep['plan']}/{ep['edit']}/{ep['wait']} destroyed {ep['destroyed']} ({time.time() - t0:.0f}s)", flush=True)
    env.close()


if __name__ == "__main__":
    main()
