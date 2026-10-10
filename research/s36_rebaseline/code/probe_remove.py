"""Focused probe: does a REMOVE action the native mask offers actually remove the road, and can it disconnect a house?
Headless, seed 1000003: planner B plays 120 decisions (network exists), then for up to 6 legal REMOVE candidates whose predicted removal disconnects a house (world model), and 3 that do not:
send the REMOVE, record accepted / applyInfo, the edge's road state before/after (per tile side), the house's connectivity before/after, and the native mask legality of REMOVE for
(a) plain road edges, (b) house driveway edges, (c) carpark stubs. Also counts how many legal REMOVEs exist and how many of them would disconnect something.
   python research/s36_rebaseline/code/probe_remove.py"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
for p in (HERE, ROOT, os.path.join(ROOT, "python"), os.path.join(ROOT, "experiments"), os.path.join(ROOT, "research", "s21_native_check", "code")):
    sys.path.insert(0, p)
import numpy as np  # noqa: E402
from native_adapter import NativePlannerPolicy, NativeStateView  # noqa: E402
from executor import make_policy  # noqa: E402
from planner import Weights  # noqa: E402
from world_model import WorldModel, ends  # noqa: E402
from synthetic_rl import layout as L  # noqa: E402


def road_state(raw, e):
    tiles = {(t["coord"][0], t["coord"][1]): t for t in raw["tiles"]}
    a, b = ends(e)
    da, db = ("E", "W") if e[2] == "E" else ("S", "N")
    return {"a": (tiles.get(a) or {}).get("road", {}).get(da), "b": (tiles.get(b) or {}).get("road", {}).get(db)}


def main():
    import mini_motorways_env as M
    from experiments import gamectl
    out = {"trials": []}
    with gamectl.GameInstance("s36_probe_remove") as g:
        env = M.MiniMotorwaysEnv(max_decisions=5000, decision_delta=1.0, request_path=os.path.join(g.ipc_dir, "request.json"), response_path=os.path.join(g.ipc_dir, "response.json"), start_mode="NORMAL_START")
        pol = NativePlannerPolicy(make_policy(Weights()))
        env.reset(seed=1000003)
        pol.reset()
        raw = env.last_raw_observation
        for t in range(120):
            mask = env.action_masks().copy()
            act = pol.act(raw, mask)
            if not mask[act]:
                act = int(np.flatnonzero(mask)[0])
            _, r, te, tr, info = env.step(act)
            raw = env.last_raw_observation
        mask = env.action_masks().copy()
        wm = WorldModel(NativeStateView(raw, mask, pol.t, pol.spawn_t))
        legal_removes = [L.decode_action(int(i)) for i in np.flatnonzero(mask) if 0 < i < L.UPGRADE_BASE and L.decode_action(int(i))[0] == "REMOVE"]
        plain = [k for k in legal_removes if not wm.is_house_edge((k[1], k[2], k[3]))]
        house_e = [k for k in legal_removes if wm.is_house_edge((k[1], k[2], k[3]))]
        stub = [k for k in legal_removes if (k[1], k[2], k[3]) in wm.stub]
        disconnecting = [k for k in legal_removes if wm.lost_connections([], [(k[1], k[2], k[3])])]
        out["state"] = {"t": 120, "active_edges": len(wm.active), "houses": len(wm.houses), "connected": len(wm.connected), "legal_removes": len(legal_removes), "plain": len(plain), "house_edges": len(house_e), "stubs": len(stub), "predicted_disconnecting": len(disconnecting), "concrete": wm.concrete}
        # edges that exist but are NOT offered for REMOVE
        not_offered = [e for e in wm.edges if not mask[L.action_index("REMOVE", *e)]]
        out["state"]["existing_edges_not_removable"] = len(not_offered)
        out["state"]["not_removable_examples"] = [{"edge": e, "state": wm.edges[e], "house_edge": wm.is_house_edge(e), "stub": e in wm.stub} for e in not_offered[:6]]
        trials = disconnecting[:6] + [k for k in plain if k not in disconnecting][:3]
        for k in trials:
            e = (k[1], k[2], k[3])
            before = {"road": road_state(raw, e), "lost_pred": sorted(wm.lost_connections([], [e])), "connected": len(wm.connected), "concrete": wm.concrete}
            _, r, te, tr, info = env.step(L.action_index("REMOVE", *e))
            raw = env.last_raw_observation
            mask = env.action_masks().copy()
            wm = WorldModel(NativeStateView(raw, mask, pol.t, pol.spawn_t))
            after = {"road": road_state(raw, e), "edge_in_model": wm.edges.get(e), "connected": len(wm.connected), "concrete": wm.concrete, "lost_actual": sorted(set(h for h in before["lost_pred"] if h in wm.houses and h not in wm.connected))}
            out["trials"].append({"edge": e, "accepted": info.get("accepted"), "applyInfo": (info.get("applyInfo") or "")[:160], "before": before, "after": after})
            print(json.dumps(out["trials"][-1], default=str)[:600], flush=True)
            if te or tr:
                break
    json.dump(out, open(os.path.join(ROOT, "research", "s36_rebaseline", "probe_remove.json"), "w"), indent=1, default=str)
    print("STATE", out["state"])


if __name__ == "__main__":
    main()
