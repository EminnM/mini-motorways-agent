"""
Phase 10: Gymnasium wrapper around the C# <-> Python file-based IPC bridge.

This wrapper is deliberately THIN. The C# side (game_plugin/DoorstopBootstrap/
Entrypoint.cs, Phase9ARunner's "Phase 10: C# <-> Python IPC bridge" region)
remains authoritative for: reset, native simulation state, action legality,
action execution (TileEditor.AddRoad / ClearTileExplicit), simulation stepping
(ISimulation.Step()), and terminal state (MotorwaysGame.HasGameEnded). This
file does exactly two things C# does not do for us:
  1. Speaks the IPC protocol (write request.json, poll for response.json).
  2. Converts the native PolicyObservation JSON into a fixed-shape numpy
     array, and the native legalActionIndices list into a boolean mask --
     pure data reshaping, not game-rule logic.

IPC transport: plain request/response FILES (not TCP sockets). An earlier TCP
version of the C# side reproducibly stalled the whole game process for several
minutes before it even reached its first engine scene-load event -- confirmed
(via a clean PowerShell-native launch of the identical build) to be specific
to builds referencing System.Net.Sockets, consistent with antivirus/real-time-
protection heuristic scanning of a freshly-built, unsigned DLL that opens a
network listener. Switching the transport to plain file read/write (the same
System.IO the bootstrap already used for its own log file) removed the stall
entirely with no Windows Defender exclusion needed. See
research/PHASE_10_FIRST_LEARNING_AGENT.md for the full narrative.
"""

import json
import os
import time
from typing import Any, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

IPC_DIR = r"C:\Users\TKA\Desktop\mini moto\logs\ipc"
REQUEST_PATH = os.path.join(IPC_DIR, "request.json")
RESPONSE_PATH = os.path.join(IPC_DIR, "response.json")

# ---------------------------------------------------------------------------
# Observation encoding -- EXACTLY what the policy receives. Documented here
# (and restated in research/PHASE_10_FIRST_LEARNING_AGENT.md section 3) rather
# than left implicit, per the phase brief's explicit requirement.
#
# Per tile (fixed order, 15 floats), for every tile in the native
# PolicyObservation's own `bounds` (the CityDefinition's PlayableArea -- NOT
# the (smaller) action bounds; the agent can OBSERVE the whole playable map
# but can only ACT within the action-space tiles, exactly like a human who can
# see more of the city than the area they're currently allowed to edit in
# this phase's RectInt(-5,-5,16,16) action window):
#   0 buildable            (0/1)
#   1 overWater            (0/1)
#   2 underMountain        (0/1)
#   3 overRail             (0/1)
#   4 road_N               (RoadState ordinal / 3)
#   5 road_E               (RoadState ordinal / 3)
#   6 road_S               (RoadState ordinal / 3)
#   7 road_W               (RoadState ordinal / 3)
#   8 hasTrafficLight      (0/1)
#   9 hasRoundabout        (0/1)
#  10 contentType          (ordinal / 2: None=0, House=1, Destination=2)
#  11 hasGroupIndex        (0/1 -- whether this tile belongs to a tracked group)
#  12 destTotalDemandNorm  (destination.totalDemand / 50, clipped [0,1], else 0)
#  13 destOvercrowding     (destination.isOvercrowding, 0/1, else 0)
#  14 visibleVehicleCount  (clip(count, 0, 10) / 10)
#
# Explicitly EXCLUDED (privileged / non-observable / future information, per
# the Phase 8 PolicyObservation contract this reuses unchanged): contributed
# supply, demand multiplier, the unassigned-vs-waiting demand split, vehicle
# route/pathfinding internals, lane-graph object identity, RNG state, and
# anything from the separate, privileged SimulationSnapshot. None of that is
# in the native PolicyObservation JSON in the first place, so there is
# nothing to filter here -- this file only reshapes what Phase 8 already
# decided a human-observable policy may see.
#
# Global features (10 floats, appended after all tile features):
#   0 hour/24, 1 day/365 (clipped), 2 week/52 (clipped), 3 score/1000,
#   4 concreteAvailable/concreteTotal, 5 bridgesAvailable/bridgesTotal,
#   6 tunnelsAvailable/tunnelsTotal, 7 motorwaysAvailable/motorwaysTotal,
#   8 trafficLightsAvailable/trafficLightsTotal,
#   9 roundaboutsAvailable/roundaboutsTotal
# (each ratio is 0.0 if its Total is 0, to avoid division by zero).
NUM_TILE_FEATURES = 15
NUM_GLOBAL_FEATURES = 11

_ROAD_STATE_ORDINAL = {"None": 0, "Pending": 1, "Active": 2, "Mothballed": 3, "VisiblyActive": 2}
_CONTENT_TYPE_ORDINAL = {"None": 0, "House": 1, "Destination": 2}


def _ratio(available: float, total: float) -> float:
    return float(available) / float(total) if total else 0.0


def encode_observation(obs: dict) -> np.ndarray:
    """Convert one native PolicyObservation JSON object into a flat float32
    vector. Pure reshaping -- see module docstring for the exact layout."""
    tiles = obs["tiles"]
    out = np.zeros(len(tiles) * NUM_TILE_FEATURES + NUM_GLOBAL_FEATURES, dtype=np.float32)

    for ti, tile in enumerate(tiles):
        base = ti * NUM_TILE_FEATURES
        out[base + 0] = 1.0 if tile["buildable"] else 0.0
        out[base + 1] = 1.0 if tile["overWater"] else 0.0
        out[base + 2] = 1.0 if tile["underMountain"] else 0.0
        out[base + 3] = 1.0 if tile["overRail"] else 0.0
        road = tile["road"]
        out[base + 4] = _ROAD_STATE_ORDINAL.get(road["N"], 0) / 3.0
        out[base + 5] = _ROAD_STATE_ORDINAL.get(road["E"], 0) / 3.0
        out[base + 6] = _ROAD_STATE_ORDINAL.get(road["S"], 0) / 3.0
        out[base + 7] = _ROAD_STATE_ORDINAL.get(road["W"], 0) / 3.0
        out[base + 8] = 1.0 if tile["hasTrafficLight"] else 0.0
        out[base + 9] = 1.0 if tile["hasRoundabout"] else 0.0
        out[base + 10] = _CONTENT_TYPE_ORDINAL.get(tile["contentType"], 0) / 2.0
        out[base + 11] = 1.0 if tile.get("groupIndex") is not None else 0.0
        dest = tile.get("destination")
        if dest:
            out[base + 12] = min(max(dest.get("totalDemand", 0) / 50.0, 0.0), 1.0)
            out[base + 13] = 1.0 if dest.get("isOvercrowding") else 0.0
        vvc = tile.get("visibleVehicleCount", 0)
        out[base + 14] = min(max(vvc, 0), 10) / 10.0

    g = obs["globals"]
    gbase = len(tiles) * NUM_TILE_FEATURES
    out[gbase + 0] = g["hour"] / 24.0
    out[gbase + 1] = min(g["day"], 365) / 365.0
    out[gbase + 2] = min(g["week"], 52) / 52.0
    out[gbase + 3] = g["score"] / 1000.0
    out[gbase + 4] = _ratio(g["concreteAvailable"], g["concreteTotal"])
    out[gbase + 5] = _ratio(g["bridgesAvailable"], g["bridgesTotal"])
    out[gbase + 6] = _ratio(g["tunnelsAvailable"], g["tunnelsTotal"])
    out[gbase + 7] = _ratio(g["motorwaysAvailable"], g["motorwaysTotal"])
    out[gbase + 8] = _ratio(g["trafficLightsAvailable"], g["trafficLightsTotal"])
    out[gbase + 9] = _ratio(g["roundaboutsAvailable"], g["roundaboutsTotal"])
    # Weekly upgrade choice pending (0..~3, normalized): without this the policy
    # could TAKE the CHOOSE_UPGRADE action (mask allows it) but could not SEE
    # the state that makes it valuable.
    out[gbase + 10] = min(g.get("pendingUpgradeChoices", 0), 3) / 3.0
    return out


# ---------------------------------------------------------------- shaping (EXPERIMENT)
# Potential-based shaping, OFF by default (user-approved separate experiment only).
# Phi(s) = number of houses whose road-connected component also contains a
# destination of the SAME colour group, computed purely from the observation
# (tile coords, per-edge road state, content type, groupIndex). The shaped reward is
#   r' = r_native + coef * (gamma * Phi(s') - Phi(s))        (Ng, Harada & Russell 1999)
# which provably leaves the optimal policy unchanged. Phi(terminal) = 0.
_DIR_STEP = {"N": (0, 1), "E": (1, 0), "S": (0, -1), "W": (-1, 0)}


def connectivity_potential(obs: dict, flip_y: bool = False) -> int:
    nodes = {}
    for t in obs["tiles"]:
        c = t["coord"]
        nodes[(c[0], c[1])] = t
    # union-find over road-connected tiles
    parent = {}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for k, t in nodes.items():
        if any(v != "None" for v in t["road"].values()) or t["contentType"] != "None":
            parent[k] = k
    for k, t in nodes.items():
        if k not in parent:
            continue
        for d, (dx, dy) in _DIR_STEP.items():
            if t["road"][d] == "None":
                continue
            n = (k[0] + dx, k[1] + (-dy if flip_y else dy))
            if n in parent:
                union(k, n)
    # A destination's footprint tiles carry no road edges themselves: roads reach it
    # through an adjacent Carpark tile (contentType "Carpark"), so join destination
    # tiles with orthogonally adjacent Carpark / Destination tiles.
    for k, t in nodes.items():
        if k not in parent or t["contentType"] not in ("Destination", "Carpark"):
            continue
        for dx, dy in _DIR_STEP.values():
            n = (k[0] + dx, k[1] + dy)
            if n in parent and nodes[n]["contentType"] in ("Destination", "Carpark") and (
                    t["contentType"] == "Destination" or nodes[n]["contentType"] == "Destination"):
                union(k, n)
    dest_groups = {}   # root -> set of groups
    for k, t in nodes.items():
        if k in parent and t["contentType"] == "Destination" and t.get("groupIndex") is not None:
            dest_groups.setdefault(find(k), set()).add(t["groupIndex"])
    count = 0
    for k, t in nodes.items():
        if k in parent and t["contentType"] == "House" and t.get("groupIndex") is not None:
            if t["groupIndex"] in dest_groups.get(find(k), ()):
                count += 1
    return count


class IpcClient:
    """Speaks the file-based request/response protocol described in
    Entrypoint.cs's Phase 10 IPC bridge comment. One JSON object per request,
    an incrementing integer id, atomic write-then-rename on both ends."""

    def __init__(self, request_path: str = REQUEST_PATH, response_path: str = RESPONSE_PATH, timeout: float = 120.0):
        self.request_path = request_path
        self.response_path = response_path
        self.timeout = timeout
        # Seed from wall-clock time, not 1: the C# side remembers the last
        # request id it processed for as long as the GAME PROCESS stays
        # alive, which can span multiple separate Python process runs. If a
        # fresh Python process restarted its counter at 1, the C# side would
        # see id=1 <= its already-higher last-processed id and silently
        # ignore every request forever (confirmed live). Milliseconds since
        # epoch is monotonically higher than any id a previous run could
        # have reached, with enormous margin.
        self._next_id = int(time.time() * 1000)
        os.makedirs(os.path.dirname(request_path), exist_ok=True)

    def send(self, op: str, **kwargs: Any) -> dict:
        rid = self._next_id
        self._next_id += 1
        req = {"id": rid, "op": op}
        req.update(kwargs)

        if os.path.exists(self.response_path):
            try:
                os.remove(self.response_path)
            except OSError:
                pass

        tmp = self.request_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(req, f)
        # os.replace can transiently fail with WinError 5 (access denied) if
        # the C# side's File.ReadAllText happens to have the destination
        # briefly open for reading at the exact moment of the rename --
        # confirmed live. The read is near-instant, so a short retry loop
        # (rather than any locking protocol) is sufficient.
        for attempt in range(50):
            try:
                os.replace(tmp, self.request_path)
                break
            except PermissionError:
                time.sleep(0.01)
        else:
            os.replace(tmp, self.request_path)

        deadline = time.time() + self.timeout
        while time.time() < deadline:
            if os.path.exists(self.response_path):
                try:
                    with open(self.response_path, "r") as f:
                        data = json.load(f)
                except (json.JSONDecodeError, OSError):
                    time.sleep(0.001)
                    continue
                if data.get("id") == rid:
                    if "error" in data:
                        raise RuntimeError(f"IPC op={op} id={rid} returned error: {data['error']}")
                    return data
            time.sleep(0.001)
        raise TimeoutError(f"No IPC response for id={rid} op={op} within {self.timeout}s")


class MiniMotorwaysEnv(gym.Env):
    """Thin Gymnasium wrapper. reset()/step() semantics are standard
    Gymnasium; C# remains authoritative for everything game-related (see
    module docstring). Exposes action_masks() for sb3-contrib's MaskablePPO,
    since standard Stable-Baselines3 PPO cannot consume a dynamic per-step
    legal-action mask over a ~1000-action Discrete space without choosing
    illegal actions constantly."""

    metadata = {"render_modes": []}

    def __init__(self, max_decisions: int = 40, request_path: str = REQUEST_PATH, response_path: str = RESPONSE_PATH, decision_delta: float = 0.3, start_mode: str = 'NORMAL_START', start_gap: int = 0, shaping: float = 0.0, shaping_gamma: float = 0.99, shaping_flip_y: bool = False):
        super().__init__()
        self.max_decisions = max_decisions
        # Simulated seconds advanced per agent decision (multiple of the native
        # 0.1s timestep). Larger = more game time per IPC round-trip = faster
        # training over a fixed sim horizon, coarser control.
        self.decision_delta = decision_delta
        self.start_mode = start_mode  # NORMAL_START | CONNECTED_START (Phase 11A)
        self.start_gap = start_gap    # native road edits left UNBUILT at the destination end
        self.shaping, self.shaping_gamma, self.shaping_flip_y = shaping, shaping_gamma, shaping_flip_y
        self.extra_reset_args = {}        # optional extra RESET fields (e.g. start_gap_index for Gap-1 tasks); empty = unchanged behaviour
        self.last_raw_observation = None  # read-only copy of the last native observation dict (for evaluators/wrappers)
        self._phi = 0.0
        self.last_init_info = None
        self.client = IpcClient(request_path, response_path)

        hello = self.client.send("HELLO")
        self.action_space_size = hello["actionSpaceSize"]
        self.action_space = spaces.Discrete(self.action_space_size)

        # One throwaway reset to learn the fixed observation shape for this
        # map (always the same CityDefinition -- "LosAngeles", the first
        # entry MapDatabase resolves -- for the lifetime of one C# process;
        # see GetCityDefinitionForTesting in Entrypoint.cs). Does not count
        # against the caller's own first reset() -- Gym environments are
        # expected to be reset() before use regardless.
        probe = self.client.send("RESET", seed=1, max_decisions=1)
        probe_vec = encode_observation(probe["observation"])
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=probe_vec.shape, dtype=np.float32)

        self._current_mask = np.zeros(self.action_space_size, dtype=bool)
        self._current_mask[probe["legalActionIndices"]] = True
        self._overlay = None

    def set_overlay(self, overlay: dict):
        """Trainer-provided lines for the in-game training overlay; piggybacked
        on the next STEP request (zero extra IPC round-trips)."""
        self._overlay = overlay

    def action_masks(self) -> np.ndarray:
        """sb3-contrib MaskablePPO calls this to restrict sampling to
        currently-legal actions (the native mask computed by
        EnumerateLegalActions over TileEditor.AddRoad/ClearTileExplicit)."""
        return self._current_mask

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        env_seed = seed if seed is not None else int(self.np_random.integers(1, 2**31 - 1))
        resp = self.client.send("RESET", seed=env_seed, max_decisions=self.max_decisions, decision_delta=self.decision_delta, start_mode=self.start_mode, start_gap=self.start_gap, **self.extra_reset_args)
        self.last_init_info = resp.get("initInfo")
        self.last_raw_observation = resp["observation"]
        obs_vec = encode_observation(resp["observation"])
        if self.shaping:
            self._phi = float(connectivity_potential(resp["observation"], self.shaping_flip_y))
        self._current_mask = np.zeros(self.action_space_size, dtype=bool)
        self._current_mask[resp["legalActionIndices"]] = True
        info = {"seed": resp["seed"], "decisionIndex": resp["decisionIndex"]}
        return obs_vec, info

    def step(self, action: int):
        if self._overlay is not None:
            resp = self.client.send("STEP", action=int(action), overlay=self._overlay)
        else:
            resp = self.client.send("STEP", action=int(action))
        obs_vec = encode_observation(resp["observation"])
        self.last_raw_observation = resp["observation"]
        self._current_mask = np.zeros(self.action_space_size, dtype=bool)
        self._current_mask[resp["legalActionIndices"]] = True
        reward = float(resp["reward"])
        terminated = bool(resp["terminated"])
        truncated = bool(resp["truncated"])
        native_reward = reward
        if self.shaping:
            phi_next = 0.0 if terminated else float(connectivity_potential(resp["observation"], self.shaping_flip_y))
            reward = native_reward + self.shaping * (self.shaping_gamma * phi_next - self._phi)
            self._phi = phi_next
        info = {
            "nativeReward": native_reward,
            "rewardComponents": resp["rewardComponents"],
            "accepted": resp["accepted"],
            "appliedAction": resp["appliedAction"],
            "decisionIndex": resp["decisionIndex"],
            "isolationOk": resp["isolationOk"],
            "applyInfo": resp.get("applyInfo"),     # additive, read-only (S6): lets diagnostics distinguish terminal-on-entry steps
        }
        return obs_vec, reward, terminated, truncated, info

    def close(self):
        try:
            self.client.send("CLOSE")
        except Exception:
            pass


def make_monitored_env(ipc_dir, max_decisions, decision_delta, start_mode, start_gap=0, shaping=0.0, shaping_gamma=0.99, shaping_flip_y=False):
    """Picklable factory for SubprocVecEnv workers: one env bound to ONE game
    instance's IPC directory, wrapped in Monitor so ep_rew_mean/ep_len_mean exist."""
    import os
    from stable_baselines3.common.monitor import Monitor
    env = MiniMotorwaysEnv(
        max_decisions=max_decisions, decision_delta=decision_delta,
        request_path=os.path.join(ipc_dir, "request.json"),
        response_path=os.path.join(ipc_dir, "response.json"),
        start_mode=start_mode, start_gap=start_gap, shaping=shaping, shaping_gamma=shaping_gamma, shaping_flip_y=shaping_flip_y)
    return Monitor(env)
