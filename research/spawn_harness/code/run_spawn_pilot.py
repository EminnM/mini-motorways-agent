"""Phase-1 spawn-steering pilot runner (APPROVED). 10 fresh seeds x anchors 60/150 (planner-B main line, replayed per anchor), one DIAG_FORK per anchor with base/base2/block_far/block_near,
90-decision WAIT continuation, observations at checkpoints 10/30/60/90. Raw fork responses + anchor observation/legal set + plans saved to raw/pilot_<seed>_<anchor>.json.gz.
   python research/spawn_harness/code/run_spawn_pilot.py"""
import gzip
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "s21_native_check", "code"))
import spawn_pilot as P  # noqa: E402
from native_adapter import NativePlannerPolicy, ROOT  # noqa: E402
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import numpy as np  # noqa: E402
from executor import make_policy  # noqa: E402
from planner import Weights  # noqa: E402

OUT = os.path.join(ROOT, "research", "spawn_harness", "pilot")


def main():
    import mini_motorways_env as M
    from experiments import gamectl
    os.makedirs(os.path.join(OUT, "raw"), exist_ok=True)
    t0 = time.time()
    with gamectl.GameInstance("spawn_pilot") as g:
        env = M.MiniMotorwaysEnv(max_decisions=100000, decision_delta=1.0, request_path=os.path.join(g.ipc_dir, "request.json"), response_path=os.path.join(g.ipc_dir, "response.json"), start_mode="NORMAL_START")
        c = env.client
        for seed in P.SEEDS:
            for anchor in P.ANCHORS:
                pol = NativePlannerPolicy(make_policy(Weights()))
                env.reset(seed=seed)
                pol.reset()
                raw = env.last_raw_observation
                ended = False
                for i in range(anchor):
                    mask = env.action_masks().copy()
                    _, _, te, tr, _ = env.step(pol.act(raw, mask))
                    raw = env.last_raw_observation
                    if te or tr:
                        ended = True
                        break
                rec = {"seed": seed, "anchor": anchor, "ended_before_anchor": ended, "decisions_played": i + 1 if ended else anchor}
                if ended:
                    rec["skipped"] = "episode ended before the anchor"
                else:
                    legal = [int(a) for a in np.flatnonzero(env.action_masks())]
                    plan = P.plan_spawn_branches(raw, legal)
                    rec["anchor_raw"], rec["legal"], rec["plan"] = raw, legal, {k: v for k, v in plan.items()}
                    if "branches" in plan:
                        req = P.pilot_request(plan["branches"])
                        op = req.pop("op")
                        rec["fork"] = c.send(op, **req)
                    else:
                        rec["skipped"] = plan.get("skipped")
                with gzip.open(os.path.join(OUT, "raw", f"pilot_{seed}_{anchor}.json.gz"), "wt") as f:
                    json.dump(rec, f, default=str)
                if "fork" in rec and "error" not in rec["fork"]:
                    br = rec["fork"]["branches"]
                    print(f"seed {seed} anchor {anchor}: far d={rec['plan']['far']['dist_to_dest']} near d={rec['plan']['near']['dist_to_dest']} edits {br['block_far']['editsAccepted']}/{br['block_near']['editsAccepted']} spawns {{{', '.join(n + ':' + str(len(br[n]['spawns'])) for n in br)}}} ({time.time() - t0:.0f}s)", flush=True)
                else:
                    print(f"seed {seed} anchor {anchor}: SKIPPED {rec.get('skipped') or rec.get('fork', {}).get('error', '')[:100]} ({time.time() - t0:.0f}s)", flush=True)
        env.close()


if __name__ == "__main__":
    main()
