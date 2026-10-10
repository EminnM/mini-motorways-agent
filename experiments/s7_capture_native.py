"""S7 grounding capture (native, READ-ONLY apart from two tiny ADD/REMOVE cost probes): normal-start states at reset and at the moment the first
house appears, with the full raw observation and native legal-action set. No training, no policy.
   python experiments/s7_capture_native.py --n 24"""
import argparse
import gzip
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from experiments import paths  # noqa: E402,F401
import numpy as np  # noqa: E402

OUT = os.path.join(ROOT, "research", "s7_long_route")
VAL_SEEDS = [182150, 205907, 229664, 253421, 261340, 269259]


def slim(raw):
    return {"simulationTime": raw.get("simulationTime"), "bounds": raw["bounds"], "globals": raw["globals"],
            "tiles": [{"coord": t["coord"], "buildable": t["buildable"], "overWater": t["overWater"], "underMountain": t["underMountain"], "overRail": t["overRail"],
                       "road": t["road"], "contentType": t["contentType"], "groupIndex": t.get("groupIndex"), "hasTrafficLight": t["hasTrafficLight"],
                       "hasRoundabout": t["hasRoundabout"]} for t in raw["tiles"]]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    a = ap.parse_args()
    import mini_motorways_env as M
    from experiments import gamectl
    os.makedirs(OUT, exist_ok=True)
    seeds = (VAL_SEEDS + [311000 + 7919 * i for i in range(max(a.n - 6, 0))])[: a.n]
    t0 = time.time()
    recs = []
    with gamectl.GameInstance("s7cap") as g:
        env = M.MiniMotorwaysEnv(max_decisions=600, decision_delta=1.0, request_path=os.path.join(g.ipc_dir, "request.json"),
                                 response_path=os.path.join(g.ipc_dir, "response.json"), start_mode="NORMAL_START")
        for si, seed in enumerate(seeds):
            obs, info = env.reset(seed=seed)
            r0 = slim(env.last_raw_observation)
            m0 = [int(i) for i in np.flatnonzero(env.action_masks())]
            rec = {"seed": seed, "reset": {"raw": r0, "legal": m0}, "first_house": None, "cost_probe": None}
            step = 0
            while step < 40:
                raw = env.last_raw_observation
                if any(t["contentType"] == "House" for t in raw["tiles"]):
                    break
                mask = env.action_masks()
                obs, r, te, tr, inf = env.step(0 if mask[0] else int(np.flatnonzero(mask)[-1]))
                step += 1
                if te or tr:
                    break
            raw = env.last_raw_observation
            if any(t["contentType"] == "House" for t in raw["tiles"]):
                rec["first_house"] = {"step": step, "raw": slim(raw), "legal": [int(i) for i in np.flatnonzero(env.action_masks())]}
                if si < 3:      # concrete cost / refund probe on a far, isolated legal ADD
                    from synthetic_rl import layout as L
                    legal = [i for i in np.flatnonzero(env.action_masks()) if 0 < i < L.UPGRADE_BASE and L.decode_action(int(i))[0] == "ADD"]
                    c0 = env.last_raw_observation["globals"]["concreteAvailable"]
                    probes = []
                    for a_ in legal[:1] + legal[len(legal) // 2:len(legal) // 2 + 1]:
                        obs, r, te, tr, inf = env.step(int(a_))
                        c1 = env.last_raw_observation["globals"]["concreteAvailable"]
                        dec = L.decode_action(int(a_))
                        rm = L.action_index("REMOVE", dec[1], dec[2], dec[3])
                        legal_rm = bool(env.action_masks()[rm])
                        obs, r, te, tr, inf = env.step(int(rm)) if legal_rm else (obs, 0, False, False, {"appliedAction": "none"})
                        c2 = env.last_raw_observation["globals"]["concreteAvailable"]
                        probes.append({"add": list(dec), "concrete_before": c0, "after_add": c1, "after_remove": c2, "remove_legal": legal_rm, "remove_applied": inf.get("appliedAction")})
                        c0 = c2
                    rec["cost_probe"] = {"total": env.last_raw_observation["globals"]["concreteTotal"], "probes": probes}
            recs.append(rec)
            print(f"seed {seed}: house at step {rec['first_house']['step'] if rec['first_house'] else None} ({time.time() - t0:.0f}s)", flush=True)
        env.close()
    with gzip.open(os.path.join(OUT, "native_normal_start_captures.jsonl.gz"), "wt") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    print("native seconds", round(time.time() - t0, 1))


if __name__ == "__main__":
    main()
