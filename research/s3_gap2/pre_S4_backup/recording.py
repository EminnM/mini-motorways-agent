"""Passive episode / action recording. Nothing here influences the environment: the wrapper forwards every
call unchanged and only reads what the environment already returns (reward, info['appliedAction'],
info['rewardComponents'], the action mask the environment itself published)."""
import json
import os
import time
from collections import Counter
from typing import Any, Dict, Optional

import gymnasium as gym

from . import paths  # noqa: F401

CATEGORIES = ("WAIT", "ADD_ROAD", "REMOVE_ROAD", "UPGRADE")
GAP1_FIELDS = ("task_id", "success", "steps_to_success", "destructive_failure", "timeout", "native_terminal",
               "first_action_correct", "a_star_added_at", "construction_decisions", "continuation_decisions",
               "native_score_total", "n_ADD_correct", "n_ADD_incorrect", "n_REMOVE_before_success")


def categorize(applied: str) -> str:
    if applied == "WAIT":
        return "WAIT"
    if applied.startswith("ADD_ROAD"):
        return "ADD_ROAD"
    if applied.startswith("REMOVE_ROAD"):
        return "REMOVE_ROAD"
    return "UPGRADE"


class EpisodeAccumulator:
    """Accumulates one episode; identical for training (via RecorderEnv) and evaluation loops."""

    def __init__(self, meta: Dict[str, Any]):
        self.meta = dict(meta)
        self.reset_episode(None)

    def reset_episode(self, game_seed: Optional[int], init_info: Any = None):
        self.game_seed = game_seed
        self.init_info = init_info
        self.ret = 0.0
        self.native = 0.0
        self.length = 0
        self.cat = Counter()
        self.idx = Counter()
        self.idx_legal = Counter()
        self.terminal = False
        self.truncated = False
        self.gap1 = None
        self.t0 = time.time()

    def add_step(self, action: int, legal: bool, reward: float, info: Dict[str, Any], terminated: bool, truncated: bool):
        applied = info.get("appliedAction", "WAIT")
        self.cat[categorize(applied)] += 1
        self.idx[int(action)] += 1
        if legal:
            self.idx_legal[int(action)] += 1
        self.ret += float(reward)
        self.native += float(info.get("rewardComponents", {}).get("scoreDelta", 0.0))
        self.length += 1
        self.terminal = bool(terminated)
        self.truncated = bool(truncated)
        if "gap1" in info:
            self.gap1 = info["gap1"]

    def row(self, episode: int, training_timestep: Optional[int]) -> Dict[str, Any]:
        n = max(self.length, 1)
        return {
            "episode": episode,
            "training_timestep": training_timestep,
            "game_seed": self.game_seed,
            "initialization": self.meta["initialization"],
            "start_gap": self.meta.get("start_gap", 0),
            "reward_mode": self.meta["reward_mode"],
            "return": round(self.ret, 6),
            "native_score": round(self.native, 6),
            "length_decisions": self.length,
            "length_sim_seconds": round(self.length * self.meta["decision_interval_seconds"], 6),
            "terminal": self.terminal,
            "truncated": self.truncated,
            "wait_count": self.cat["WAIT"],
            "add_count": self.cat["ADD_ROAD"],
            "remove_count": self.cat["REMOVE_ROAD"],
            "upgrade_count": self.cat["UPGRADE"],
            "wait_fraction": round(self.cat["WAIT"] / n, 6),
            "add_fraction": round(self.cat["ADD_ROAD"] / n, 6),
            "remove_fraction": round(self.cat["REMOVE_ROAD"] / n, 6),
            "upgrade_fraction": round(self.cat["UPGRADE"] / n, 6),
            "wall_seconds": round(time.time() - self.t0, 3),
        }

    def gap1_columns(self):
        if not self.gap1:
            return {}
        return {f"gap1_{k}": self.gap1.get(k) for k in GAP1_FIELDS}

    def action_index_rows(self):
        return {str(i): [c, self.idx_legal[i]] for i, c in self.idx.items()}


class RecorderEnv(gym.Wrapper):
    """Transparent recorder placed directly around MiniMotorwaysEnv (inside SB3's Monitor)."""

    def __init__(self, env, out_path: str, meta: Dict[str, Any]):
        super().__init__(env)
        self._out_path = out_path
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        self._acc = EpisodeAccumulator(meta)
        self._episode = 0
        self._steps_total = 0
        self._started = time.time()
        self._open = False

    # SB3/MaskablePPO reach these through get_wrapper_attr / env_method, which walks to the inner env;
    # explicit delegates keep direct attribute access working too.
    def action_masks(self):
        return self.env.get_wrapper_attr("action_masks")()

    def set_overlay(self, overlay):
        return self.env.get_wrapper_attr("set_overlay")(overlay)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._acc.reset_episode(info.get("seed"), self.env.get_wrapper_attr("last_init_info"))
        self._open = True
        return obs, info

    def step(self, action):
        legal = bool(self.env.get_wrapper_attr("action_masks")()[int(action)])
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._steps_total += 1
        self._acc.add_step(int(action), legal, reward, info, terminated, truncated)
        if terminated or truncated:
            self._flush()
        return obs, reward, terminated, truncated, info

    def _flush(self):
        if not self._open:
            return
        self._open = False
        self._episode += 1
        row = self._acc.row(self._episode, self._steps_total)
        row.update(self._acc.gap1_columns())   # worker-local step count; global timestep assigned in post-processing
        row["worker_steps_total"] = self._steps_total
        row["wall_time"] = round(time.time() - self._started, 3)
        row["unix_time"] = round(time.time(), 3)
        row["worker_pid"] = os.getpid()
        row["action_index_counts"] = self._acc.action_index_rows()
        with open(self._out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")


RAW_DIR_ENV = "EXPERIMENT_RAW_DIR"
META_ENV = "EXPERIMENT_RECORD_META"


def recorder_meta_from_env() -> Dict[str, Any]:
    return json.loads(os.environ[META_ENV])


def make_recorded_env_factory(original_cls, wrap=None):
    """Replacement for train_ppo.MiniMotorwaysEnv (single-env path): same constructor, recorder around it.
    `wrap` (optional) is a task wrapper (e.g. Gap-1) placed between the env and the recorder."""
    def factory(*args, **kwargs):
        env = original_cls(*args, **kwargs)
        if wrap is not None:
            env = wrap(env)
        raw = os.environ[RAW_DIR_ENV]
        return RecorderEnv(env, os.path.join(raw, f"episodes_{os.getpid()}.jsonl"), recorder_meta_from_env())
    return factory


def make_recorded_monitored_env(ipc_dir, max_decisions, decision_delta, start_mode, start_gap=0,
                                shaping=0.0, shaping_gamma=0.99, shaping_flip_y=False):
    """Picklable replacement for mini_motorways_env.make_monitored_env (SubprocVecEnv workers): identical
    construction plus the passive recorder; Monitor stays outermost exactly as before."""
    from stable_baselines3.common.monitor import Monitor
    import mini_motorways_env as M
    env = M.MiniMotorwaysEnv(
        max_decisions=max_decisions, decision_delta=decision_delta,
        request_path=os.path.join(ipc_dir, "request.json"),
        response_path=os.path.join(ipc_dir, "response.json"),
        start_mode=start_mode, start_gap=start_gap, shaping=shaping, shaping_gamma=shaping_gamma,
        shaping_flip_y=shaping_flip_y)
    raw = os.environ[RAW_DIR_ENV]
    env = RecorderEnv(env, os.path.join(raw, f"episodes_{os.getpid()}.jsonl"), recorder_meta_from_env())
    return Monitor(env)
