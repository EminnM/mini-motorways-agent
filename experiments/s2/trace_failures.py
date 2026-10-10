"""Re-run selected (checkpoint, native task) pairs deterministically and record the action trace + an ASCII rendering of the
final NATIVE state (raw observation), for failure analysis. Native, small (<= a handful of episodes)."""
import argparse
import json
import os
import sys

import numpy as np

from .. import paths
from .. import gamectl
from ..gap1 import gap1_env as G
from ..gap1 import tasks as T

sys.path.insert(0, paths.PYTHON_DIR)
from synthetic_rl import layout as L  # noqa: E402

OUT = os.path.join(paths.ROOT, "research", "synthetic_to_native")


def ascii_state(raw, task, trace_edges):
    tiles = {(t["coord"][0], t["coord"][1]): t for t in raw["tiles"]}
    foot = [tuple(f) for f in task["destination_footprint"]]
    xs = [f[0] for f in foot] + [task["house"][0], task["a_star"]["x"]] + [e[0] for e in trace_edges]
    ys = [f[1] for f in foot] + [task["house"][1], task["a_star"]["y"]] + [e[1] for e in trace_edges]
    x0, x1, y0, y1 = min(xs) - 3, max(xs) + 4, min(ys) - 3, max(ys) + 4
    a = task["a_star"]
    astar = L.edge_key(a["x"], a["y"], a["dir"])
    chosen = set(L.edge_key(*e) for e in trace_edges)
    lines = []
    for y in range(y1, y0 - 1, -1):
        row = ""
        for x in range(x0, x1 + 1):
            t = tiles.get((x, y))
            if t is None:
                ch = "?"
            elif t["contentType"] == "House":
                ch = "H"
            elif t["contentType"] == "Destination":
                ch = "D"
            elif t["contentType"] == "Carpark":
                ch = "C"
            elif t["overWater"]:
                ch = "~"
            elif any(v in T.LIVE for v in t["road"].values()):
                ch = "+"
            elif t["contentType"] == "Tree":
                ch = "t"
            else:
                ch = "."
            row += ch
            e = (x, y, "E")
            if t and t["road"]["E"] in T.LIVE:
                row += "-" if e not in chosen else "="
            elif e == astar:
                row += "*"
            else:
                row += " "
        lines.append(f"y={y:3d} {row}")
        if y > y0:
            row2 = ""
            for x in range(x0, x1 + 1):
                t = tiles.get((x, y))
                e = (x, y, "S")
                if t and t["road"]["S"] in T.LIVE:
                    row2 += "|" if e not in chosen else "H"
                elif e == astar:
                    row2 += "*"
                else:
                    row2 += " "
                row2 += " "
            lines.append("      " + row2)
    lines.append(f"      x={x0}..{x1}   legend: + road, = / H edge added by the policy, * missing correct edge (a*), ~ water, D destination, C carpark, H house")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, help="comma list ckpt_path@task_id")
    ap.add_argument("--manifest", nargs="+", required=True)
    ap.add_argument("--tag", default="s2trace")
    a = ap.parse_args(argv)
    from sb3_contrib import MaskablePPO
    import mini_motorways_env as M
    tasks = {}
    for m in a.manifest:
        for l in open(m):
            if l.strip():
                r = json.loads(l)
                tasks[r["task_id"]] = r
    out = []
    with gamectl.GameInstance(a.tag) as g:
        for pair in a.pairs.split(","):
            ck, tid = pair.split("@")
            t = tasks[tid]
            model = MaskablePPO.load(ck, device="cpu")
            base = M.MiniMotorwaysEnv(max_decisions=100, decision_delta=1.0, request_path=os.path.join(g.ipc_dir, "request.json"),
                                      response_path=os.path.join(g.ipc_dir, "response.json"), start_mode="CONNECTED_START", start_gap=0)
            env = G.Gap1Env(base, [t], 20, 0, "completion", sampling="sequential", continue_after_success=False, success_terminates=True)
            obs, _ = env.reset(options={"task_index": 0})
            trace, edges, done = [], [], False
            while not done:
                mask = env.action_masks()
                probs = None
                import torch
                with torch.no_grad():
                    ot, _ = model.policy.obs_to_tensor(obs)
                    dist = model.policy.get_distribution(ot, action_masks=mask[None, :])
                    probs = dist.distribution.probs[0].cpu().numpy()
                a_, _ = model.predict(obs, action_masks=mask, deterministic=True)
                a_ = int(a_)
                d = L.decode_action(a_)
                top = np.argsort(-probs)[:3]
                trace.append({"action": a_, "decoded": d, "p": round(float(probs[a_]), 3), "p_correct_edge": round(float(probs[t["a_star"]["action_index"]]), 4),
                              "top3": [(int(i), round(float(probs[i]), 3), list(L.decode_action(int(i)))) for i in top]})
                if d[0] == "ADD":
                    edges.append((d[1], d[2], d[3]))
                obs, r, te, tr, info = env.step(a_)
                done = te or tr
            raw = base.last_raw_observation
            out.append({"pair": pair, "task_id": tid, "success": info["gap1"]["success"], "destructive": info["gap1"]["destructive_failure"], "timeout": info["gap1"]["timeout"],
                        "a_star": t["a_star"], "flags": t.get("s2_flags"), "n_actions": len(trace), "trace": trace, "ascii": ascii_state(raw, t, edges)})
            try:
                base.close()
            except Exception:
                pass
    json.dump(out, open(os.path.join(OUT, "failure_traces.json"), "w"), indent=1)
    for o in out:
        print("=" * 80)
        print(o["task_id"], "success", o["success"], "timeout", o["timeout"], "flags", o["flags"], "a*", o["a_star"])
        print(o["ascii"])
        print("first actions:", [(x["decoded"], x["p"]) for x in o["trace"][:6]])
        print("first-step top3:", o["trace"][0]["top3"], "p(correct edge)=", o["trace"][0]["p_correct_edge"])


if __name__ == "__main__":
    main()
