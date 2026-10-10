"""S23 runner (approved): per seed, planner-B main line to decision 60 via normal STEP, then ONE DIAG_FORK with branches base/base2/parallel/dense (S22 edit rules, unchanged), horizon 60,
observations at checkpoints. Raw fork responses preserved in raw/fork_<seed>.json.gz.   python research/s23_fork_topology/code/run_fork_topology.py --idx 0,1,...,11"""
import argparse
import gzip
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "s22_throughput", "code"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "s21_native_check", "code"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "spawn_harness", "code"))
from native_adapter import NativePlannerPolicy, ROOT  # noqa: E402
import throughput_probe as TP  # noqa: E402
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import numpy as np  # noqa: E402
import s11_guard as G  # noqa: E402
from synthetic_rl import layout as L  # noqa: E402
from executor import make_policy  # noqa: E402
from planner import Weights  # noqa: E402

OUT = os.path.join(ROOT, "research", "s23_fork_topology")
SEEDS = json.load(open(os.path.join(ROOT, "research", "s9_native_transfer", "seeds.json")))["seeds"]
ANCHOR, HORIZON, CPS = 60, 60, [1, 5, 10, 20, 30, 45, 60]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idx", default=",".join(str(i) for i in range(12)))
    a = ap.parse_args()
    import mini_motorways_env as M
    from experiments import gamectl
    os.makedirs(os.path.join(OUT, "raw"), exist_ok=True)
    t0 = time.time()
    with gamectl.GameInstance("s23_fork") as g:
        env = M.MiniMotorwaysEnv(max_decisions=100000, decision_delta=1.0, request_path=os.path.join(g.ipc_dir, "request.json"), response_path=os.path.join(g.ipc_dir, "response.json"), start_mode="NORMAL_START")
        c = env.client
        for k in [int(x) for x in a.idx.split(",")]:
            seed = SEEDS[k]
            pol = NativePlannerPolicy(make_policy(Weights()))
            env.reset(seed=seed)
            pol.reset()
            raw = env.last_raw_observation
            hashes = []
            for i in range(ANCHOR):
                mask = env.action_masks().copy()
                hashes.append(TP.obs_hash(raw))
                env.step(pol.act(raw, mask))
                raw = env.last_raw_observation
            mask = env.action_masks().copy()
            par = [L.action_index("ADD", *e) for e in TP.edits_parallel(raw, mask)]
            den = [L.action_index("ADD", *e) for e in TP.edits_dense(raw, mask)]
            ctx = G.Ctx(raw)
            func = G.edges_from_raw(raw)
            anchor_state = {"hash": TP.obs_hash(raw), "connected": [list(h) for h in sorted(G.connected_house_set(ctx, func))], "houses": len(ctx.houses), "n_func": len(func), "concrete": raw["globals"]["concreteAvailable"], "score": raw["globals"]["score"],
                            "dests": TP.dest_info(raw), "vehicles": sum(t.get("visibleVehicleCount", 0) for t in raw["tiles"]), "legal_par": [bool(mask[x]) for x in par], "legal_den": [bool(mask[x]) for x in den]}
            branches = [{"name": "base", "actions": []}, {"name": "base2", "actions": []}, {"name": "parallel", "actions": par}, {"name": "dense", "actions": den}]
            rec = c.send("DIAG_FORK", n_pre=0, horizon=HORIZON, checkpoints=CPS, intervention_seconds=0.3, include_obs=True, branches=branches)
            out = {"seed": seed, "idx": k, "anchor": ANCHOR, "anchor_state": anchor_state, "anchor_raw": raw, "mainline_hashes": hashes, "branch_specs": {b["name"]: b["actions"] for b in branches}, "fork": rec}
            with gzip.open(os.path.join(OUT, "raw", f"fork_{seed}.json.gz"), "wt") as f:
                json.dump(out, f)
            if "error" in rec:
                print(f"seed {seed}: ERROR {rec['error'][:200]}", flush=True)
                continue
            sc = {n: rec["branches"][n]["checkpoints"][-1]["score"] for n in rec["branches"]}
            print(f"seed {seed}: anchor conn {len(anchor_state['connected'])}/{anchor_state['houses']} edits par {rec['branches']['parallel']['editsAccepted']}/{len(par)} den {rec['branches']['dense']['editsAccepted']}/{len(den)} score@60 {sc} spawns {{{', '.join(n + ':' + str(len(rec['branches'][n]['spawns'])) for n in rec['branches'])}}} ({time.time() - t0:.0f}s)", flush=True)
        env.close()


if __name__ == "__main__":
    main()
