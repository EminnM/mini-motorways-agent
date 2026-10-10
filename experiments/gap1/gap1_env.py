"""Gap-1 task wrapper around the UNCHANGED MiniMotorwaysEnv.

* Every episode is a real native CONNECTED_START game with exactly one path edit left out (start_gap_index).
* The policy sees the normal PolicyObservation (nothing about a*, the house or the destination is added).
* Construction window: up to K policy decisions. Phase ends on
    SUCCESS            target house connected to target destination (native live road graph)
    DESTRUCTIVE FAIL   any pre-existing required edge removed
    TIMEOUT            K decisions without success
    NATIVE TERMINAL    the game itself ends (reported separately)
* After SUCCESS the episode continues for `continuation` decisions in which the only legal action is WAIT (the policy
  need not keep editing); native score earned meanwhile is the measured outcome. Unless native-reward semantics are
  needed (reward variants below) nothing else changes.
Reward variants (training only): native | completion | native_plus_completion | connection_shaped (the env's own
shaping; the env is built with shaping>0 by the harness)."""
import json
from typing import Any, Dict, List, Optional

import gymnasium as gym
import numpy as np

from . import tasks as T

REWARD_VARIANTS = ("native", "completion", "native_plus_completion", "connection_shaped")


def load_manifest(path: str) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


class Gap1Env(gym.Wrapper):
    def __init__(self, env, tasks: List[Dict[str, Any]], construction_window: int = 20, continuation: int = 40,
                 reward_variant: str = "native", beta: float = 5.0, sampling: str = "random", sample_seed: int = 0,
                 continue_after_success: bool = True, success_terminates: bool = False):
        super().__init__(env)
        self.success_terminates = success_terminates
        assert reward_variant in REWARD_VARIANTS
        self.tasks = tasks
        self.K, self.T_cont = construction_window, continuation
        self.reward_variant, self.beta = reward_variant, beta
        self.sampling = sampling
        self._rng = np.random.default_rng(sample_seed)
        self._cursor = 0
        self.continue_after_success = continue_after_success
        inner = env.unwrapped
        inner.start_mode = "CONNECTED_START"
        inner.start_gap = 0
        self.task = None

    # --- helpers the SB3 stack reaches via get_wrapper_attr
    def action_masks(self):
        base = self.env.get_wrapper_attr("action_masks")()
        if self._phase == "continuation":
            if base[0]:
                m = np.zeros_like(base)
                m[0] = True
                return m
        return base

    def set_overlay(self, overlay):
        return self.env.get_wrapper_attr("set_overlay")(overlay)

    def reset(self, *, seed=None, options=None):
        if options and "task_index" in options:
            idx = int(options["task_index"])
        elif self.sampling == "sequential":
            idx = self._cursor % len(self.tasks)
            self._cursor += 1
        else:
            idx = int(self._rng.integers(len(self.tasks)))
        t = self.tasks[idx]
        self.task = t
        inner = self.env.unwrapped
        inner.extra_reset_args = {"start_gap_index": t["start_gap_index"]}
        obs, info = self.env.reset(seed=t["game_seed"])
        init = inner.last_init_info or {}
        a = t["a_star"]
        if init.get("omittedEdits", "").strip() != f"ADD_ROAD({a['x']},{a['y']},{a['dir']})":
            raise RuntimeError(f"task {t.get('task_id')} did not reproduce: omitted={init.get('omittedEdits')!r}")
        self._house = tuple(t["house"])
        self._dest = [tuple(x) for x in t["destination_footprint"]]
        full = T.parse_edits(t["full_path_edits"])
        self._built = [e for e in full if e != (a["x"], a["y"], a["dir"])]
        self._a_star_action = a["action_index"]
        self._a_star_applied = f"ADD_ROAD({a['x']},{a['y']},{a['dir']})"
        self._phase = "construction"
        self._k = 0
        self._cont = 0
        self._success = False
        self._steps_to_success = None
        self._destructive = False
        self._timeout = False
        self._native_terminal = False
        self._a_star_added_at = None
        self._first_action_correct = None
        self._cat = {"WAIT": 0, "ADD": 0, "REMOVE": 0, "UPGRADE": 0, "ADD_correct": 0, "ADD_incorrect": 0, "REMOVE_before_success": 0}
        self._native_score = 0.0
        info = dict(info)
        info.update(task_id=t.get("task_id"), task_index=idx)
        return obs, info

    def step(self, action):
        action = int(action)
        if self._k == 0 and self._phase == "construction":
            self._first_action_correct = (action == self._a_star_action)
        obs, reward, terminated, truncated, info = self.env.step(action)
        applied = info.get("appliedAction", "WAIT")
        self._native_score += float(info.get("rewardComponents", {}).get("scoreDelta", 0.0))
        completion = 0.0
        if applied == "WAIT":
            self._cat["WAIT"] += 1
        elif applied.startswith("ADD_ROAD"):
            self._cat["ADD"] += 1
            if applied == self._a_star_applied and info.get("accepted"):
                self._cat["ADD_correct"] += 1
                if self._a_star_added_at is None:
                    self._a_star_added_at = self._k + 1
            else:
                self._cat["ADD_incorrect"] += 1
        elif applied.startswith("REMOVE_ROAD"):
            self._cat["REMOVE"] += 1
            if not self._success:
                self._cat["REMOVE_before_success"] += 1
        else:
            self._cat["UPGRADE"] += 1
        raw = self.env.unwrapped.last_raw_observation
        if terminated:
            self._native_terminal = True
        if self._phase == "construction":
            self._k += 1
            connected = T.house_connected(raw, self._house, self._dest)
            if connected and not terminated:
                self._success = True
                self._steps_to_success = self._k
                completion = 1.0
                if self.success_terminates:
                    terminated = True          # success is a true terminal of the task (reward 1, no future value)
                    self._phase = "done"
                elif self.continue_after_success and self.T_cont > 0:
                    self._phase = "continuation"
                else:
                    truncated = True
                    self._phase = "done"
            elif not T.required_edges_intact(raw, self._built):
                self._destructive = True
                terminated = True
                self._phase = "done"
            elif self._k >= self.K and not terminated:
                self._timeout = True
                truncated = True
                self._phase = "done"
        elif self._phase == "continuation":
            self._cont += 1
            if self._cont >= self.T_cont and not terminated:
                truncated = True
                self._phase = "done"
        if terminated or truncated:
            self._phase = "done"
        # reward variants
        if self.reward_variant == "native" or self.reward_variant == "connection_shaped":
            r = reward
        elif self.reward_variant == "completion":
            r = completion
        else:
            r = reward + self.beta * completion
        info = dict(info)
        info["gap1"] = {
            "task_id": self.task.get("task_id"), "success": self._success, "steps_to_success": self._steps_to_success,
            "destructive_failure": self._destructive, "timeout": self._timeout, "native_terminal": self._native_terminal,
            "first_action_correct": self._first_action_correct, "a_star_added_at": self._a_star_added_at,
            "construction_decisions": self._k, "continuation_decisions": self._cont,
            "native_score_total": self._native_score, "phase": self._phase,
            **{f"n_{k}": v for k, v in self._cat.items()},
        }
        info["completion"] = completion
        return obs, float(r), bool(terminated), bool(truncated), info
