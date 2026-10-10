"""S34 matched-rate random EDIT control (postfix_v8, diagnostic). Conditions, all through the unchanged HybridNativeEnv on the 8 fresh seeds, 1,500-decision horizon, 3 stochastic rollouts:
  A  original hybrid        : = the replication's fresh-benchmark stochastic rollouts (research/s34_replication/eval_fresh, same seeds / torch seeds) -- not re-run.
  B  randomized EDIT        : the trained policy is sampled exactly as in A (same torch seed per seed x rollout => common random numbers until divergence); whenever its sampled action is an
                              EDIT (ADD/REMOVE), it is replaced by a uniform draw (dedicated RNG) over the OTHER legal EDIT actions in the current native mask; kept if none exists.
                              PLAN, WAIT and forced upgrade choices are untouched.
  C  planner B + random EDIT: each decision's kind is drawn (dedicated RNG) from the matched run's observed fresh-benchmark A shares of PLAN/EDIT/WAIT (preregistered, see MATCHED_SHARES);
                              EDIT = uniform legal EDIT, PLAN = planner B, WAIT = 0. Forced upgrades as in the hybrid env (the lowest legal index). No neural policy involved.
Per decision the script records the kind, ADD/REMOVE, whether the EDIT was replaced, the destroyed-connection delta and the executed action's boundary status (both endpoints inside
globals.playableAreaAtTime; must always hold). Output: eval/<cond>_<run>_<ckpt>_r<rollout>_<seed>.json.gz
   python research/s34_matched_edit_ablation/code/ablate.py --cond B --ckpt <path> --rollout 0 [--idx 0,1,...]
   python research/s34_matched_edit_ablation/code/ablate.py --cond C --match hybrid_s31 --rollout 0"""
import argparse
import gzip
import hashlib
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(ROOT, "research", "s31_hybrid_native_rl", "code"))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from hybrid_env import HybridNativeEnv, PLAN  # noqa: E402
from synthetic_rl import layout as L  # noqa: E402

FRESH_SEEDS = [900000 + 7919 * i for i in range(8)]
HORIZON = 1500
DLL = r"C:\Program Files (x86)\Steam\steamapps\common\Mini Motorways\Bootstrap\MiniMotorwaysBootstrap.dll"
V8 = "2aa42367cbb14babce31a6063038f50be9b3535bd7bc23d8dbbdbb65533aa00a"
# Preregistered condition-C kind shares = each run's 100k stochastic PLAN/EDIT/WAIT shares observed in the replication's fresh benchmark (condition A), renormalised without UPG.
MATCHED_SHARES = {"hybrid_s31": (0.580, 0.382, 0.031), "rep1_seed131": (0.503, 0.473, 0.018), "rep2_seed231": (0.254, 0.696, 0.044)}
OUT = os.path.join(ROOT, "research", "s34_matched_edit_ablation", "eval")


def far(x, y, d):
    return (x + 1, y) if d == "E" else (x, y - 1)


def boundary_ok(native, area):
    if not (0 < native < L.UPGRADE_BASE) or area is None:
        return True
    k = L.decode_action(native)
    fx, fy = far(k[1], k[2], k[3])
    ins = lambda p: area[0] <= p[0] <= area[2] and area[1] <= p[1] <= area[3]  # noqa: E731
    return ins((k[1], k[2])) and (k[0] == "REMOVE" or ins((fx, fy)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", required=True, choices=["A", "B", "C"])   # A = re-run of the original hybrid with per-decision instrumentation (expected to reproduce the replication's A rollouts)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--match", default=None)
    ap.add_argument("--rollout", type=int, required=True)
    ap.add_argument("--idx", default=None)
    a = ap.parse_args()
    assert hashlib.sha256(open(DLL, "rb").read()).hexdigest() == V8, "deployed DLL is not postfix_v8"
    torch.set_num_threads(1)
    idx = [int(x) for x in a.idx.split(",")] if a.idx else list(range(8))
    model = None
    if a.cond in ("A", "B"):
        from sb3_contrib import MaskablePPO
        from hybrid_policy import HybridMaskablePolicy  # noqa: F401
        p = os.path.normpath(a.ckpt)
        model = MaskablePPO.load(p, device="cpu")
        run = os.path.basename(os.path.dirname(os.path.dirname(p)))
        ck = os.path.basename(p).replace(".zip", "")
        tag = f"{a.cond}_{run}_{ck}"
    else:
        run, ck = a.match, "matched"
        shares = np.array(MATCHED_SHARES[run])
        shares = shares / shares.sum()
        tag = f"C_{run}_matched"
    os.makedirs(OUT, exist_ok=True)
    logs = []
    env = HybridNativeEnv(FRESH_SEEDS, f"s34abl_{tag}_r{a.rollout}_{''.join(map(str, idx))}"[:60], max_decisions=HORIZON, log_fn=logs.append)
    t0 = time.time()
    for k in idx:
        seed = FRESH_SEEDS[k]
        tseed = 20000 + 1000 * a.rollout + k                      # same policy-sampling stream as condition A (common random numbers)
        torch.manual_seed(tseed)
        rng = random.Random(770000 + 1000 * a.rollout + k)        # dedicated, independently controlled RNG for replacements / kind draws
        obs, _ = env.reset(options={"seed": seed})
        assert env.raw["globals"].get("playableAreaAtTime") is not None, "environment-version mismatch"
        done = False
        rsum = 0.0
        per = []   # per-decision records
        n_replaced = n_edit = 0
        while not done:
            m = env.action_masks()
            area = env.raw["globals"].get("playableAreaAtTime")
            legal_edits = [int(i) for i in np.flatnonzero(m[1:L.UPGRADE_BASE]) + 1]
            replaced = False
            orig = None
            if a.cond in ("A", "B"):
                act, _ = model.predict(obs, action_masks=m, deterministic=False)
                act = int(act)
                if a.cond == "B" and 0 < act < L.UPGRADE_BASE:
                    n_edit += 1
                    orig = act
                    alts = [i for i in legal_edits if i != act]
                    if alts:
                        act = rng.choice(alts)
                        replaced = True
                        n_replaced += 1
            else:
                if not m[PLAN]:                                    # forced upgrade interrupt: same handling as the hybrid env / planner-B baseline
                    act = int(np.flatnonzero(m[:L.N_ACTIONS])[0])
                else:
                    u = rng.random()
                    if u < shares[0] or (u < shares[0] + shares[1] and not legal_edits):
                        act = PLAN
                    elif u < shares[0] + shares[1]:
                        act = rng.choice(legal_edits)
                    else:
                        act = 0
            d_before = env.ep["destroyed"]
            obs, r, te, tr, info = env.step(act)
            rsum += float(r)
            na = info["native_action"]
            kind = info["kind"]
            rec = {"k": kind[0], "d": env.ep["destroyed"] - d_before}
            if kind == "EDIT":
                rec["ar"] = L.decode_action(na)[0][0]             # 'A' or 'R'
                rec["rep"] = int(replaced)
                if orig is not None:
                    rec["oar"] = L.decode_action(orig)[0][0]
            if not boundary_ok(na, area):
                rec["boundary_violation"] = 1
            per.append(rec)
            done = te or tr
        ep = dict(logs[-1], policy=tag, cond=a.cond, run=run, ckpt=ck, rollout=a.rollout, torch_seed=tseed, horizon=HORIZON, env_version="postfix_v8",
                  edits_sampled=n_edit, edits_replaced=n_replaced, boundary_violations=sum(x.get("boundary_violation", 0) for x in per),
                  add=sum(1 for x in per if x.get("ar") == "A"), remove=sum(1 for x in per if x.get("ar") == "R"),
                  destroyed_by_edit=sum(x["d"] for x in per if x["k"] == "E"), destroyed_by_plan=sum(x["d"] for x in per if x["k"] == "P"), destroyed_by_other=sum(x["d"] for x in per if x["k"] not in "EP"),
                  per_decision=per)
        assert ep["illegal"] == 0, f"STOP: illegal action on seed {seed}"
        assert ep["boundary_violations"] == 0, f"STOP: boundary violation on seed {seed}"
        assert abs(rsum - (ep["score"] + ep["penalty"])) < 1e-6, f"STOP: reward accounting mismatch on seed {seed}"
        with gzip.open(os.path.join(OUT, f"{tag}_r{a.rollout}_{seed}.json.gz"), "wt") as f:
            json.dump(ep, f)
        print(f"[{tag} r{a.rollout}] seed {seed}: t {ep['t']} score {ep['score']} term {ep['terminated']} P/E/W {ep['plan']}/{ep['edit']}/{ep['wait']} add/rem {ep['add']}/{ep['remove']} replaced {n_replaced}/{n_edit} destroyed {ep['destroyed']} (edit {ep['destroyed_by_edit']}, plan {ep['destroyed_by_plan']}) ({time.time() - t0:.0f}s)", flush=True)
    env.close()


if __name__ == "__main__":
    main()
