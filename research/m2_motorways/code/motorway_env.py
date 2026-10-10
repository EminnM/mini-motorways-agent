"""M2: thin research subclass of the production env adding motorway edits through the plugin's optional STEP field (the Discrete(3111) action path is untouched: a motorway step sends
action = WAIT plus {"motorway": {...}}) and the MOTORWAY_CHECK dry-run op. python/mini_motorways_env.py is not modified."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(ROOT, "python"))
import mini_motorways_env as M  # noqa: E402
from mini_motorways_env import encode_observation  # noqa: E402


class MotorwayEnv(M.MiniMotorwaysEnv):
    def step_motorway(self, motorway: dict, action: int = 0):
        """motorway = {"op":"add","x1","y1","d1","x2","y2","d2"} (d in N/NE/E/SE/S/SW/W/NW) or {"op":"remove","x","y"}; returns the usual tuple; info has motorwayOk/motorwayInfo."""
        resp = self.client.send("STEP", action=int(action), motorway=motorway)
        obs_vec = encode_observation(resp["observation"])
        self.last_raw_observation = resp["observation"]
        self._current_mask = np.zeros(self.action_space_size, dtype=bool)
        self._current_mask[resp["legalActionIndices"]] = True
        info = {"nativeReward": float(resp["reward"]), "rewardComponents": resp["rewardComponents"], "accepted": resp["accepted"], "appliedAction": resp["appliedAction"], "decisionIndex": resp["decisionIndex"],
                "isolationOk": resp["isolationOk"], "applyInfo": resp.get("applyInfo"), "motorwayOk": resp.get("motorwayOk"), "motorwayInfo": resp.get("motorwayInfo")}
        return obs_vec, float(resp["reward"]), bool(resp["terminated"]), bool(resp["truncated"]), info

    def motorway_check(self, motorway: dict) -> dict:
        return self.client.send("MOTORWAY_CHECK", motorway=motorway)
