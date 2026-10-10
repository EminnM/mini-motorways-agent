"""S11 semantic probe (native, ~30 s): print the exact directional road STATES of house tiles and their neighbours around house-touching ADDs, to understand why the
single-driveway replacement model mispredicts. Control behaviour only (stochastic S8-C, native mask); no guard."""
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from experiments import paths  # noqa: F401
import numpy as np
import torch

OUT = os.path.join(ROOT, "research", "s11_connection_guard")
CK = os.path.join(ROOT, "research", "s8_long_route_training", "runs", "arm_C_s11", "checkpoints", "step_0099840.zip")
D = {"N": (0, 1), "E": (1, 0), "S": (0, -1), "W": (-1, 0)}


def main():
    torch.set_num_threads(1)
    from sb3_contrib import MaskablePPO
    import mini_motorways_env as M
    from experiments import gamectl
    from synthetic_rl import layout as L
    model = MaskablePPO.load(CK, device="cpu")
    seed, idx = int(sys.argv[1]), int(sys.argv[2])
    lines = []
    with gamectl.GameInstance("s11probe") as g:
        env = M.MiniMotorwaysEnv(max_decisions=600, decision_delta=1.0, request_path=os.path.join(g.ipc_dir, "request.json"), response_path=os.path.join(g.ipc_dir, "response.json"), start_mode="NORMAL_START")
        torch.manual_seed(1000 + idx)
        obs, _ = env.reset(seed=seed)
        for i in range(60):
            mask = env.action_masks().copy()
            raw = env.last_raw_observation
            tm = {tuple(t["coord"]): t for t in raw["tiles"]}
            a = int(model.predict(obs, action_masks=mask, deterministic=False)[0])
            dec = L.decode_action(a)
            snap = {}
            def state_of(tile):
                t = tm.get(tile)
                return None if t is None else {d: t["road"][d] for d in "NESW" if t["road"][d] != "None"}
            if dec[0] == "ADD":
                e = (dec[1], dec[2], dec[3])
                for tile in (e[:2], L.neighbor(*e)):
                    if tm[tile]["contentType"] == "House":
                        snap["house"] = list(tile)
                        snap["house_states_before"] = state_of(tile)
                        snap["nbr_states_before"] = {f"{tile[0] + dx},{tile[1] + dy}": state_of((tile[0] + dx, tile[1] + dy)) for d, (dx, dy) in D.items()}
            obs, r, te, tr, inf = env.step(a)
            if "house" in snap:
                raw2 = env.last_raw_observation
                tm2 = {tuple(t["coord"]): t for t in raw2["tiles"]}
                h = tuple(snap["house"])
                snap["house_states_after"] = {d: tm2[h]["road"][d] for d in "NESW" if tm2[h]["road"][d] != "None"}
                snap["nbr_states_after"] = {f"{h[0] + dx},{h[1] + dy}": {d: tm2[(h[0] + dx, h[1] + dy)]["road"][d] for d in "NESW" if tm2[(h[0] + dx, h[1] + dy)]["road"][d] != "None"} for d, (dx, dy) in D.items() if (h[0] + dx, h[1] + dy) in tm2}
            lines.append({"i": i, "action": list(dec), "applied": inf.get("appliedAction"), "applyInfo": inf.get("applyInfo"), "accepted": inf.get("accepted"), **snap})
            if te or tr:
                break
        env.close()
    json.dump(lines, open(os.path.join(OUT, f"probe_{seed}.json"), "w"), indent=1)
    for l in lines:
        if "house" in l:
            print(l["i"], l["action"], "house", l["house"], "before", l["house_states_before"], "after", l["house_states_after"], "|", l["applyInfo"][:70])


if __name__ == "__main__":
    main()
