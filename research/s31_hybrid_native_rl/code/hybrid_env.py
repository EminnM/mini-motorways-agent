"""S31 hybrid native environment: Discrete(3112) = native Discrete(3111) [0 WAIT, 1..3108 tile ADD/REMOVE = EDIT, 3109/3110 ChooseUpgrade] + 3111 PLAN (delegate this decision to planner B).
Observation = v2 (16,328 floats, native encoder + group slots). Reward = the native env reward (score delta per decision + terminal penalty), unchanged. Mask = native legal mask ∪ {PLAN legal iff WAIT legal}.
Each worker owns one headless game instance. Training seeds are cycled from a list disjoint from the 12 designated evaluation seeds."""
import os
import sys

import gymnasium as gym
import numpy as np
from gymnasium import spaces

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
sys.path.insert(0, os.path.join(ROOT, "research", "s21_native_check", "code"))
from native_adapter import NativePlannerPolicy  # noqa: E402
from synthetic_rl import layout as L  # noqa: E402
from synthetic_rl import obs_v2 as V  # noqa: E402
import s11_guard as G  # noqa: E402
from executor import make_policy  # noqa: E402
from planner import Weights  # noqa: E402

PLAN = L.N_ACTIONS                 # 3111
N_ACTIONS = L.N_ACTIONS + 1        # 3112
EVAL_SEEDS = [500000 + 7919 * i for i in range(12)]


def train_seed_list(n=40):
    return [700000 + 7919 * i for i in range(n)]


class HybridNativeEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, seeds, instance_name, max_decisions=600, log_fn=None, seed_mode="cycle", visible=False, realtime=False):
        super().__init__()
        import mini_motorways_env as M
        from experiments import gamectl
        self.M, self.gamectl = M, gamectl
        self.seeds, self.seed_mode, self._cursor = list(seeds), seed_mode, 0
        self.inst = gamectl.GameInstance(instance_name, headless=not visible).start()
        self.env = M.MiniMotorwaysEnv(max_decisions=max_decisions, decision_delta=1.0, request_path=os.path.join(self.inst.ipc_dir, "request.json"), response_path=os.path.join(self.inst.ipc_dir, "response.json"), start_mode="NORMAL_START")
        if realtime:                                             # visible viewing: ask the plugin to pace each STEP in real time (its 'realtime' request flag)
            _send = self.env.client.send
            self.env.client.send = lambda op, **kw: _send(op, **(dict(kw, realtime=True) if op == "STEP" else kw))
        self.planner = NativePlannerPolicy(make_policy(Weights()))
        self.tracker = V.GroupSlotTracker()
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(V.OBS_DIM_V2,), dtype=np.float32)
        self.action_space = spaces.Discrete(N_ACTIONS)
        self._mask = np.zeros(N_ACTIONS, dtype=bool)
        self.log_fn = log_fn
        self.ep = None

    def _refresh(self):
        self.raw = self.env.last_raw_observation
        nm = self.env.action_masks()
        self._mask[:L.N_ACTIONS] = nm
        self._mask[PLAN] = bool(nm[0])
        self._obs = V.encode_native_v2(self.raw, self.tracker)

    def action_masks(self):
        return self._mask

    def reset(self, *, seed=None, options=None):
        if options and "seed" in options:
            s = options["seed"]
        else:
            s = self.seeds[self._cursor % len(self.seeds)]
            self._cursor += 1
        self.env.reset(seed=s)
        self.planner.reset()
        self.tracker.reset()
        self._refresh()
        self.ep = {"seed": s, "t": 0, "reward": 0.0, "score": 0.0, "plan": 0, "edit": 0, "wait": 0, "upg": 0, "illegal": 0, "destroyed": 0, "terminated": False, "penalty": 0.0}
        return self._obs.copy(), {"seed": s}

    def step(self, action):
        a = int(action)
        ctx = G.Ctx(self.raw)
        func = G.edges_from_raw(self.raw)
        conn_b = G.connected_house_set(ctx, func)
        nm = self._mask
        if a == PLAN:
            native = self.planner.act(self.raw, nm[:L.N_ACTIONS].copy())
            self.ep["plan"] += 1
        else:
            native = a
            self.planner.t += 1                                  # keep the adapter's decision clock aligned when the planner is bypassed
            if a == 0:
                self.ep["wait"] += 1
            elif a >= L.UPGRADE_BASE:
                self.ep["upg"] += 1
            else:
                self.ep["edit"] += 1
        if not nm[native]:
            self.ep["illegal"] += 1
            native = int(np.flatnonzero(nm[:L.N_ACTIONS])[0])
        obs_v1, r, te, tr, info = self.env.step(native)
        self._refresh()
        ctx_a = G.Ctx(self.raw)
        conn_a = G.connected_house_set(ctx_a, G.edges_from_raw(self.raw))
        self.ep["destroyed"] += len([h for h in conn_b if h in ctx_a.house_set and h not in conn_a])
        self.ep["t"] += 1
        self.ep["reward"] += float(r)
        rc = info.get("rewardComponents", {})
        self.ep["score"] += float(rc.get("scoreDelta", 0.0))
        self.ep["penalty"] += float(rc.get("terminalPenalty", 0.0))
        self.ep["terminated"] = bool(te)
        if (te or tr) and self.log_fn:
            self.log_fn(dict(self.ep, houses=len(ctx_a.houses), connected=len(conn_a)))
        return self._obs.copy(), float(r), bool(te), bool(tr), {"native_action": int(native), "kind": "PLAN" if a == PLAN else ("WAIT" if a == 0 else ("UPG" if a >= L.UPGRADE_BASE else "EDIT")), "scoreDelta": float(rc.get("scoreDelta", 0.0))}

    def close(self):
        try:
            self.env.close()
        finally:
            self.inst.stop()
