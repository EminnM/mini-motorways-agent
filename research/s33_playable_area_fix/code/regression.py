"""S33 corrected-environment regression (run after deployment): planner B and a frozen hybrid checkpoint on the 12 designated seeds through the hybrid env, with per-action logging of
(a) boundary VIOLATIONS = executed ADD whose far endpoint is outside globals.playableAreaAtTime (must be 0), (b) ATTEMPTED crossings = the policy's probability mass (and whether its unmasked
argmax) on crossing-candidate edges (origin on the area's east column / south row, far endpoint outside) — the actions the pre-fix mask would have offered.
   python research/s33_playable_area_fix/code/regression.py --policy plannerB|ckpt:<path> --mode sto|det --idx 0,...,11"""
import argparse
import gzip
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

OUT = os.path.join(ROOT, "research", "s33_playable_area_fix", "eval_postfix")


def far(x, y, d):
    return (x + 1, y) if d == "E" else (x, y - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--mode", default="sto")
    ap.add_argument("--idx", default=",".join(str(i) for i in range(12)))
    ap.add_argument("--torch-seed", type=int, default=8008)
    a = ap.parse_args()
    torch.set_num_threads(1)
    model = None
    if a.policy.startswith("ckpt:"):
        from sb3_contrib import MaskablePPO
        from hybrid_policy import HybridMaskablePolicy  # noqa: F401
        model = MaskablePPO.load(a.policy[5:], device="cpu")
    tag = ("postfix_" + os.path.basename(a.policy[5:]).replace(".zip", "") + f"_{a.mode}") if model else "postfix_" + a.policy
    os.makedirs(OUT, exist_ok=True)
    logs = []
    env = HybridNativeEnv(EVAL_SEEDS, f"s33_{tag}_{a.idx.replace(',', '')}", max_decisions=600, log_fn=logs.append)
    t0 = time.time()
    for k in [int(x) for x in a.idx.split(",")]:
        seed = EVAL_SEEDS[k]
        torch.manual_seed(a.torch_seed + k)
        obs, _ = env.reset(options={"seed": seed})
        done = False
        viol = 0
        attempted_mass = []
        attempted_mass_south = []    # south crossings: far = yMin-1, outside the stock inclusive rect (genuine violations pre-fix)
        attempted_mass_east = []     # east edges from the export's last column: far = xMax(export)+1, which the stock inclusive rule still allows
        argmax_crossing = 0
        n_dec = 0
        while not done:
            m = env.action_masks()
            area = env.raw["globals"].get("playableAreaAtTime")
            if model is None:
                act = PLAN if m[PLAN] else int(np.flatnonzero(m[:L.N_ACTIONS])[0])
            else:
                act, _ = model.predict(obs, action_masks=m, deterministic=(a.mode == "det"))
                act = int(act)
                # attempted crossing diagnostic: crossing-candidate edges = origins on the east column / south row with far endpoint outside the area
                if area is not None:
                    c_east = [L.action_index("ADD", area[2], y, "E") for y in range(area[1], area[3] + 1)]
                    c_south = [L.action_index("ADD", x, area[1], "S") for x in range(area[0], area[2] + 1)]
                    cands = c_east + c_south
                    with torch.no_grad():
                        logits = model.policy.get_distribution(torch.as_tensor(obs[None])).distribution.logits[0]
                    pre_mask = m.copy()
                    pre_mask[cands] = True
                    lg = logits.masked_fill(~torch.as_tensor(pre_mask), -1e9)
                    p = torch.softmax(lg, dim=0)
                    attempted_mass.append(float(p[cands].sum()))
                    attempted_mass_south.append(float(p[c_south].sum()))
                    attempted_mass_east.append(float(p[c_east].sum()))
                    argmax_crossing += int(int(torch.argmax(lg)) in set(cands))
            obs, r, te, tr, info = env.step(act)
            na = info["native_action"]
            if 0 < na < L.UPGRADE_BASE and area is not None:
                kd = L.decode_action(na)
                if kd[0] == "ADD":
                    fx, fy = far(kd[1], kd[2], kd[3])
                    if not (area[0] <= kd[1] <= area[2] and area[1] <= kd[2] <= area[3] and area[0] <= fx <= area[2] and area[1] <= fy <= area[3]):
                        viol += 1
            n_dec += 1
            done = te or tr
        mean = lambda xs: (sum(xs) / len(xs)) if xs else None  # noqa: E731
        ep = dict(logs[-1], policy=tag, env_version="postfix_v8", violations=viol, attempted_crossing_mass_mean=mean(attempted_mass), attempted_south_mass_mean=mean(attempted_mass_south), attempted_east_mass_mean=mean(attempted_mass_east), argmax_would_cross=argmax_crossing, decisions=n_dec)
        with gzip.open(os.path.join(OUT, f"{tag}_{seed}.json.gz"), "wt") as f:
            json.dump(ep, f)
        print(f"[{tag}] seed {seed}: t {ep['t']} score {ep['score']} term {ep['terminated']} plan/edit/wait {ep['plan']}/{ep['edit']}/{ep['wait']} destroyed {ep['destroyed']} illegal {ep['illegal']} violations {viol} attempted_mass {ep['attempted_crossing_mass_mean']} argmax_cross {argmax_crossing} ({time.time() - t0:.0f}s)", flush=True)
    env.close()


if __name__ == "__main__":
    main()
