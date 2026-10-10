"""SyntheticGapEnv: a Gymnasium environment with the native Mini Motorways observation/action interface.

Rules (everything the simulator models; nothing else exists -- no traffic, spawning, time pressure):
  * tiles on the native 37x21 grid; a house (with a driveway edge), a matching 2x2 destination (with a carpark stub edge)
    and road edges. Edges are undirected, stored canonically as (x, y, 'E'|'S'); 'S' connects (x,y) and (x,y-1).
  * ADD(x,y,d) legal iff both endpoint tiles are inside the reach window and free (not house / destination / carpark),
    the edge does not exist and concrete remains.  REMOVE(x,y,d) legal iff the edge exists, both endpoints are free
    tiles (stub edges can never be removed) and tile (x,y) has exactly ONE road edge (the native rule: ClearTileExplicit
    clears the whole tile, so a single-edge removal is only offered where it is exactly that).  WAIT is always legal
    (except while a delayed reward is pending).  Upgrade slots are always illegal.
  * connectivity: the house is connected when its start tile reaches the goal tile through road edges.
  * episode: construction window K decisions; SUCCESS when connected; DESTRUCTIVE FAILURE when an edge of the original
    network is removed; TIMEOUT (truncated) after K decisions without success.
  * reward modes: 'completion' (+1 on the connecting transition, terminates), 'delayed' (+1 `delay` WAIT-only decisions
    after connecting, then terminates), 'native_like' (after connecting, +1 every `period` decisions, `deliveries` times,
    then truncates: a sparse score stream).
Observation: float32[11666] in the native layout (see layout.py); features the simulator does not model carry the neutral
value 0 (documented in research/SYNTHETIC_RL_LAB.md); `buildable` is 1 on every tile."""
import json
import os
from typing import Any, Dict, List, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from . import layout as L


def load_tasks(path: str) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


class SyntheticGapEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, tasks: List[Dict[str, Any]], reward_mode: str = "completion", construction_window: int = 20,
                 delay: int = 12, period: int = 13, deliveries: int = 3, sampling: str = "random", concrete_total: int = 25,
                 decision_seconds: float = 1.0):
        super().__init__()
        assert reward_mode in ("completion", "delayed", "native_like")
        self.tasks = tasks
        self.reward_mode, self.K = reward_mode, construction_window
        self.delay, self.period, self.deliveries = delay, period, deliveries
        self.sampling = sampling
        self._cursor = 0
        self.concrete_total = concrete_total
        self.dt = decision_seconds
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(L.OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Discrete(L.N_ACTIONS)
        self.task: Optional[Dict[str, Any]] = None
        self.last_init_info: Optional[Dict[str, Any]] = None
        self._obs = np.zeros(L.OBS_DIM, dtype=np.float32)
        self._mask = np.zeros(L.N_ACTIONS, dtype=bool)
        self._gbase = L.N_TILES * L.NUM_TILE_FEATURES

    # ------------------------------------------------------------------ helpers
    def _tile_off(self, x, y):
        return L.tile_index(x, y) * L.NUM_TILE_FEATURES

    def _set_edge_feature(self, e, value):
        x, y, d = e
        nx, ny = L.neighbor(x, y, d)
        self._obs[self._tile_off(x, y) + L.F_ROAD[d]] = value
        self._obs[self._tile_off(nx, ny) + L.F_ROAD[L.OPP[d]]] = value

    def _write_globals(self):
        g = self._gbase
        hours = self.t * 1.2                      # native clock: 24 h per 20 s (hour = 1.2 * seconds)
        self._obs[g + L.G_HOUR] = (int(hours) % 24) / 24.0
        self._obs[g + L.G_DAY] = min(int(hours) // 24, 365) / 365.0
        self._obs[g + L.G_WEEK] = 0.0
        self._obs[g + L.G_SCORE] = self.score / 1000.0
        self._obs[g + L.G_CONCRETE] = self.concrete / self.concrete_total

    def _build_state(self, task):
        self.task = task
        self.t = 0.0
        self.score = 0.0
        self.k = 0
        self.phase = "construction"
        self.success = False
        self.destructive = False
        self.timeout = False
        self.steps_after_connect = 0
        self.deliveries_done = 0
        self.edges = set()
        self.deg = np.zeros((L.H, L.W), dtype=np.int8)
        self.edge_e = np.zeros((L.H, L.W), dtype=bool)
        self.edge_s = np.zeros((L.H, L.W), dtype=bool)
        self.original = {tuple(e) for e in task["built_edges"]}
        self.stubs = {tuple(e) for e in task["stub_edges"]}
        self.solution = [tuple(e) for e in task["solution_edges"]]
        wx0, wy0, wx1, wy1 = task["window"]
        self.free = np.zeros((L.H, L.W), dtype=bool)
        self.free[wy0 - L.YMIN: wy1 - L.YMIN + 1, wx0 - L.XMIN: wx1 - L.XMIN + 1] = True
        occ = [tuple(task["house"]), tuple(task["carpark"])] + [tuple(f) for f in task["footprint"]]
        for (x, y) in occ:
            self.free[y - L.YMIN, x - L.XMIN] = False
        self.concrete = self.concrete_total - len(self.original)
        # observation
        o = self._obs
        o[:] = 0.0
        tiles = o[: self._gbase].reshape(L.N_TILES, L.NUM_TILE_FEATURES)
        tiles[:, L.F_BUILDABLE] = 1.0                       # neutral: no terrain in the sandbox
        hx, hy = task["house"]
        tiles[L.tile_index(hx, hy), L.F_CONTENT] = L.CONTENT_HOUSE
        tiles[L.tile_index(hx, hy), L.F_GROUP] = 1.0
        for (fx, fy) in task["footprint"]:
            tiles[L.tile_index(fx, fy), L.F_CONTENT] = L.CONTENT_DEST
            tiles[L.tile_index(fx, fy), L.F_GROUP] = 1.0
        for e in list(self.original) + list(self.stubs):
            self._add_edge(e, count_obs=True)
        self._write_globals()

    def _add_edge(self, e, count_obs=True):
        x, y, d = e
        nx, ny = L.neighbor(x, y, d)
        self.edges.add(e)
        (self.edge_e if d == "E" else self.edge_s)[y - L.YMIN, x - L.XMIN] = True
        self.deg[y - L.YMIN, x - L.XMIN] += 1
        self.deg[ny - L.YMIN, nx - L.XMIN] += 1
        if count_obs:
            self._set_edge_feature(e, L.ROAD_ACTIVE)

    def _remove_edge(self, e):
        x, y, d = e
        nx, ny = L.neighbor(x, y, d)
        self.edges.discard(e)
        (self.edge_e if d == "E" else self.edge_s)[y - L.YMIN, x - L.XMIN] = False
        self.deg[y - L.YMIN, x - L.XMIN] -= 1
        self.deg[ny - L.YMIN, nx - L.XMIN] -= 1
        self._set_edge_feature(e, 0.0)

    def connected(self) -> bool:
        s, g = tuple(self.task["start"]), tuple(self.task["goal"])
        seen = {s}
        stack = [s]
        while stack:
            u = stack.pop()
            if u == g:
                return True
            for d in "NESW":
                v = L.neighbor(u[0], u[1], d)
                if v in seen:
                    continue
                if L.edge_key(u[0], u[1], d) in self.edges:
                    seen.add(v)
                    stack.append(v)
        return False

    def _compute_mask(self):
        m = self._mask
        m[:] = False
        if self.phase == "continuation":
            m[0] = True
            return m
        m[0] = True
        free = self.free
        addE = np.zeros((L.H, L.W), dtype=bool)
        addS = np.zeros((L.H, L.W), dtype=bool)
        remE = np.zeros((L.H, L.W), dtype=bool)
        remS = np.zeros((L.H, L.W), dtype=bool)
        if self.concrete > 0:
            addE[:, :-1] = free[:, :-1] & free[:, 1:] & ~self.edge_e[:, :-1]
            addS[1:, :] = free[1:, :] & free[:-1, :] & ~self.edge_s[1:, :]
        one = self.deg == 1
        remE[:, :-1] = self.edge_e[:, :-1] & free[:, :-1] & free[:, 1:] & one[:, :-1]
        remS[1:, :] = self.edge_s[1:, :] & free[1:, :] & free[:-1, :] & one[1:, :]
        m[1: 1 + 4 * L.N_TILES] = np.stack([addE, remE, addS, remS], axis=-1).reshape(-1)
        return m

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        if options and "task_index" in options:
            idx = int(options["task_index"])
        elif self.sampling == "sequential":
            idx = self._cursor % len(self.tasks)
            self._cursor += 1
        else:
            idx = int(self.np_random.integers(len(self.tasks)))
        self._build_state(self.tasks[idx])
        self.last_init_info = {"mode": "SYNTHETIC_GAP", "gap": self.task["n_gaps"], "task_id": self.task["task_id"],
                               "omittedEdits": " ".join(f"ADD_ROAD({e[0]},{e[1]},{e[2]})" for e in self.solution),
                               "house": self.task["house"], "destinationFootprint": self.task["footprint"]}
        self._compute_mask()
        return self._obs.copy(), {"seed": int(self.task["signature"], 16) & 0x7FFFFFFF, "task_id": self.task["task_id"], "task_index": idx}

    def action_masks(self) -> np.ndarray:
        return self._mask

    def set_overlay(self, overlay):          # API parity with the native env (no rendering here)
        return None

    def step(self, action: int):
        action = int(action)
        reward = 0.0
        score_delta = 0.0
        terminated = truncated = False
        accepted = False
        kind = L.decode_action(action)
        applied = "WAIT"
        if kind[0] in ("ADD", "REMOVE") and self._mask[action]:
            _, x, y, d = kind
            e = (x, y, d)
            accepted = True
            if kind[0] == "ADD":
                self._add_edge(e)
                self.concrete -= 1
                applied = f"ADD_ROAD({x},{y},{d})"
            else:
                self._remove_edge(e)
                self.concrete += 1
                applied = f"REMOVE_ROAD({x},{y},{d})"
                if e in self.original and self.phase == "construction":
                    self.destructive = True
                    terminated = True
                    self.phase = "done"
        elif kind[0] == "WAIT":
            accepted = True
        self.t += self.dt
        if self.phase == "construction":
            self.k += 1
            if self.connected() and not self.destructive:
                self.success = True
                self.steps_to_success = self.k
                if self.reward_mode == "completion":
                    reward = 1.0
                    terminated = True
                    self.phase = "done"
                else:
                    self.phase = "continuation"
                    self.steps_after_connect = 0
            elif self.k >= self.K and not terminated:
                self.timeout = True
                truncated = True
                self.phase = "done"
        elif self.phase == "continuation":
            self.steps_after_connect += 1
            if self.reward_mode == "delayed" and self.steps_after_connect >= self.delay:
                reward = 1.0
                terminated = True
                self.phase = "done"
            elif self.reward_mode == "native_like" and self.steps_after_connect % self.period == 0:
                reward = 1.0
                self.deliveries_done += 1
                if self.deliveries_done >= self.deliveries:
                    truncated = True
                    self.phase = "done"
        score_delta = reward
        self.score += reward
        self._write_globals()
        self._compute_mask()
        info = {"appliedAction": applied, "accepted": accepted, "decisionIndex": self.k,
                "rewardComponents": {"scoreDelta": score_delta, "terminalPenalty": 0.0},
                "synthetic": {"success": self.success, "destructive_failure": self.destructive, "timeout": self.timeout,
                              "steps_to_success": getattr(self, "steps_to_success", None) if self.success else None,
                              "phase": self.phase}}
        return self._obs.copy(), float(reward), bool(terminated), bool(truncated), info

    # ------------------------------------------------------------------ oracle / baselines / inspection
    def oracle_action(self) -> int:
        for e in self.solution:
            if e not in self.edges:
                return L.action_index("ADD", e[0], e[1], e[2])
        return 0

    def solution_action_set(self) -> set:
        return set(self.task["solution_action_indices"])

    def render_ascii(self) -> str:
        t = self.task
        wx0, wy0, wx1, wy1 = t["window"]
        grid = {}
        for e in self.edges:
            for p in L.edge_endpoints(e):
                grid.setdefault(p, "+")
        for e in self.stubs:
            for p in L.edge_endpoints(e):
                grid.setdefault(p, "+")
        grid[tuple(t["house"])] = "H"
        for f in t["footprint"]:
            grid[tuple(f)] = "D"
        grid[tuple(t["carpark"])] = "C"
        lines = []
        for y in range(wy1, wy0 - 1, -1):
            row = ""
            for x in range(wx0, wx1 + 1):
                ch = grid.get((x, y), ".")
                row += ch
                if x < wx1:
                    row += "-" if L.edge_key(x, y, "E") in self.edges else " "
            lines.append(f"y={y:3d} {row}")
            if y > wy0:
                row2 = ""
                for x in range(wx0, wx1 + 1):
                    row2 += "|" if L.edge_key(x, y, "S") in self.edges else " "
                    if x < wx1:
                        row2 += " "
                lines.append("      " + row2)
        lines.append("      x=" + str(wx0) + ".." + str(wx1))
        return "\n".join(lines)
