"""S15 mixed-task training environment: S7 long-route tasks (v1 obs -> v2 via slot 0) and S13 multi-house scenarios (v2), both in the 16,328-float v2 observation and Discrete(3111).
Family choice at every reset = FIXED balancing rule: pick the family with fewer cumulative training decisions so far (target 50 % / 50 % DECISIONS, not episodes). No adaptive curriculum:
the rule has no dependence on performance. S7 task mix inside the family: Gap-7 40 %, Gap-10 40 %, short routes (k = 2, 3, 5) 20 % from the frozen S7 TRAIN manifests.
GroupBlind: zeroes the six group channels (Arm B) in both training and evaluation."""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from . import layout as L
from . import obs_v2 as V
from . import s7
from .s14_env import MultiHouseEnvV2

S7_MIX = (("7", 0.4), ("10", 0.4), ("short", 0.2))


def zero_groups(obs):
    o = np.array(obs, dtype=np.float32, copy=True)
    tiles = o[: L.N_TILES * V.NUM_TILE_FEATURES_V2].reshape(L.N_TILES, V.NUM_TILE_FEATURES_V2)
    tiles[:, V.F_GROUP_SLOT0:] = 0.0
    return o


class MixedEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, s7_tasks_by_k, s13_scenarios, seed=0, group_blind=False):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.pools = {"7": list(s7_tasks_by_k[7]), "10": list(s7_tasks_by_k[10]), "short": [t for k in (2, 3, 5) for t in s7_tasks_by_k[k]]}
        self.s7 = s7.LongRouteEnv(self.pools["7"] + self.pools["10"] + self.pools["short"], sampling="random", seed=seed + 1)
        self.s13 = MultiHouseEnvV2(s13_scenarios, seed=seed + 2, sampling="random")
        self.group_blind = group_blind
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(V.OBS_DIM_V2,), dtype=np.float32)
        self.action_space = spaces.Discrete(L.N_ACTIONS)
        self.decisions = {"s7": 0, "s13": 0}
        self.episodes = {"s7": 0, "s13": 0}
        self.actions = {"s7": {"WAIT": 0, "ADD": 0, "REMOVE": 0}, "s13": {"WAIT": 0, "ADD": 0, "REMOVE": 0}}
        self.s7_by_pool = {"7": 0, "10": 0, "short": 0}
        self.family = None
        self.cur = None

    def _wrap(self, obs):
        o = V.s7_v1_to_v2(obs) if self.family == "s7" else obs
        return zero_groups(o) if self.group_blind else o

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.family = "s7" if self.decisions["s7"] <= self.decisions["s13"] else "s13"
        if self.family == "s7":
            r = self.rng.random()
            pool = "7" if r < 0.4 else ("10" if r < 0.8 else "short")
            self.s7_by_pool[pool] += 1
            tasks = self.pools[pool]
            t = tasks[int(self.rng.integers(len(tasks)))]
            idx = self.s7.tasks.index(t)
            obs, info = self.s7.reset(options={"task_index": idx})
            self.cur = self.s7
        else:
            obs, info = self.s13.reset()
            self.cur = self.s13
        self.episodes[self.family] += 1
        info = dict(info, family=self.family)
        return self._wrap(obs), info

    def step(self, action):
        obs, r, te, tr, info = self.cur.step(action)
        self.decisions[self.family] += 1
        self.actions[self.family][L.decode_action(int(action))[0] if int(action) < L.UPGRADE_BASE else "WAIT"] += 1
        info = dict(info, family=self.family)
        return self._wrap(obs), float(r), bool(te), bool(tr), info

    def action_masks(self):
        return self.cur.action_masks()

    def stats(self):
        return {"decisions": dict(self.decisions), "episodes": dict(self.episodes), "actions": {f: dict(v) for f, v in self.actions.items()}, "s7_episodes_by_pool": dict(self.s7_by_pool)}
