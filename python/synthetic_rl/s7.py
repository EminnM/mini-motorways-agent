"""S7 native-grounded long-route task family (separate from the frozen S1/S3 generators and env).

Grounding (all from live native normal-start captures, research/s7_long_route/native_normal_start_captures.jsonl.gz, 24 seeds):
  * terrain is IDENTICAL across seeds (127 water tiles = 16.3%, no mountain/rail); trees (8) do not block (68/94 tree-touching edges legal; the
    policy observation encodes trees as 0) and are ignored;
  * a normal start has ONE destination (2x2) with a carpark pair; one carpark tile carries a road STUB edge to a road tile (the goal tile); the
    house spawns with a DRIVEWAY edge (house tile - neighbouring tile). Both are pre-existing road edges that cost no concrete;
  * ADD legality: both endpoints inside the scene's native build WINDOW (the tile set reached by native legal ADDs in that scene), not water, not
    destination/carpark. Edges touching the house tile are legal and REPLACE the current house edge (S5 swap behaviour; 245/245 swaps touch a house);
  * REMOVE(x,y,d) legal iff the edge exists, tile (x,y) has exactly one road edge, the edge does not touch the house (house edges are never removable:
    24/24 native); on the carpark stub it is legal only when the action tile is the carpark tile and has NO effect (native quirk, 19/19);
  * road budget: concrete 25, each ADD costs 1, a REMOVE refunds 1 (native probe 25 -> 24 -> 25); bridges 1/1 in the globals;
  * observation = the native 37x21x15 + 11 layout (layout.py); no solution-path features; reward +1 exactly on house-carpark connectivity, else 0.
Documented simplifications: no traffic/spawns/overcrowding/time pressure; water is impassable (native allows ~50% of one-water-endpoint edges as a bridge: 4.5% of
native legal ADDs are not reproduced); the window is a per-scene tile set copied from native, so house positions outside it do not occur; the synthetic
mask is a SUPERSET of the native mask on boundary edges (see S7 report for measured precision/recall)."""
import hashlib
import heapq
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from . import layout as L

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCENE_FILE = os.path.join(ROOT, "research", "s7_long_route", "scenes.json")
DIRS4 = ("N", "E", "S", "W")
LENGTHS = (2, 3, 5, 7, 10)


def horizon_for(k: int) -> int:
    """PREREGISTERED decision horizon: 10 decisions per required construction edge (k=2 -> 20, the S3 Gap-2 window). Fixed before any S7 result."""
    return 10 * int(k)


def canon(a: Tuple[int, int], b: Tuple[int, int]) -> Tuple[int, int, str]:
    """canonical undirected edge between adjacent tiles: (x,y,'E') = (x,y)-(x+1,y); (x,y,'S') = (x,y)-(x,y-1)"""
    (ax, ay), (bx, by) = a, b
    if ay == by:
        return (min(ax, bx), ay, "E")
    return (ax, max(ay, by), "S")


def ends(e: Tuple[int, int, str]):
    x, y, d = e
    return (x, y), L.neighbor(x, y, d)


class World:
    """Scene data (terrain + scenes), loaded once."""
    _inst = None

    def __init__(self):
        d = json.load(open(SCENE_FILE))
        self.water = {tuple(w) for w in d["water_tiles"]}
        self.concrete_total = d["concrete_total"]
        self.scenes = d["scenes"]
        for s in self.scenes:
            s["dest"] = {tuple(t) for t in s["destination"]}
            s["carp"] = {tuple(t) for t in s["carpark"]}
            s["stub"] = tuple(s["stub_edge"])
            s["goal"] = tuple(s["goal_tile"])
            s["ctile"] = tuple(s["carpark_tile"])
            s["win"] = {tuple(t) for t in s["window_tiles"]}
            # v2 (after the S7 native-trajectory replay): tiles of the native window INCLUDING water tiles; an edge is legal iff both endpoints are in the window,
            # neither is destination/carpark and they are not both water (reproduces 100% of native-legal ADDs on the 24 captures, precision 91.5%)
            s["land"] = {t for t in s["win"] if t not in s["dest"] and t not in s["carp"]}
            cand = set()
            for (x, y) in s["land"]:
                for d_ in ("E", "S"):
                    n = L.neighbor(x, y, d_)
                    if n in s["land"] and not (((x, y) in self.water) and (n in self.water)):
                        cand.add((x, y, d_))
            s["cand"] = sorted(cand)

    @classmethod
    def get(cls):
        if cls._inst is None:
            cls._inst = World()
        return cls._inst


def house_edges(h):
    return [canon(h, L.neighbor(h[0], h[1], d)) for d in DIRS4]


# ------------------------------------------------------------------------------------------- diagnostics: remaining distance (privileged, never an input)
def remaining_distance(world: World, scene: Dict[str, Any], house, edges: set) -> int:
    """Minimum number of additional ADD actions that connects the house to the carpark from the CURRENT road set (0 = connected).
    Dijkstra over tiles: existing edges cost 0, buildable edges cost 1; the house is a single node whose only edge is its driveway: using a house edge
    that is not the current driveway costs 1 (replacement). Path may not pass through the house tile, destination or carpark except the final stub."""
    h = tuple(house)
    land = scene["land"]
    cur_house = [e for e in house_edges(h) if e in edges]
    INF = 10 ** 9
    dist = {}
    pq = []
    # start: step out of the house through one of its (up to 4) edges
    for d in DIRS4:
        v = L.neighbor(h[0], h[1], d)
        if v not in land or v == h or (v in world.water and h in world.water):
            continue
        e = canon(h, v)
        c = 0 if e in cur_house else 1
        if c < dist.get(v, INF):
            dist[v] = c
            heapq.heappush(pq, (c, v))
    goal = scene["goal"]
    stub_cost = 0 if scene["stub"] in edges else 1
    best = INF
    while pq:
        c, u = heapq.heappop(pq)
        if c > dist.get(u, INF):
            continue
        if u == goal:
            best = min(best, c + stub_cost)
            break
        for d in DIRS4:
            v = L.neighbor(u[0], u[1], d)
            if v == h or v not in land or (u in world.water and v in world.water):
                continue
            e = canon(u, v)
            nc = c + (0 if e in edges else 1)
            if nc < dist.get(v, INF):
                dist[v] = nc
                heapq.heappush(pq, (nc, v))
    return best


# ------------------------------------------------------------------------------------------- tasks
def task_signature(t) -> str:
    return hashlib.sha1(json.dumps([t["scene_id"], t["house"], t["driveway_dir"]]).encode()).hexdigest()[:12]


def split_of(scene_id: int, house) -> str:
    """deterministic split by (scene, house tile): the three splits never share a (scene, house) pair"""
    h = int(hashlib.sha1(f"{scene_id}:{house[0]}:{house[1]}".encode()).hexdigest()[:8], 16) % 100
    return "train" if h < 70 else ("validation" if h < 85 else "test")


def enumerate_all_tasks(world: World) -> Dict[int, List[Dict[str, Any]]]:
    """every (scene, house tile, driveway direction) grouped by its exact minimum number of required ADDs; only lengths in LENGTHS are kept"""
    out = {k: [] for k in LENGTHS}
    for s in world.scenes:
        for h in sorted(s["land"]):
            if h in world.water:
                continue
            for d in DIRS4:
                v = L.neighbor(h[0], h[1], d)
                if v not in s["land"] or v in world.water:
                    continue
                edges = {s["stub"], canon(h, v)}
                r = remaining_distance(world, s, h, edges)
                if r in out:
                    t = {"scene_id": s["scene_id"], "house": list(h), "driveway_dir": d, "k_min": r}
                    t["task_id"] = f"s7_k{r}_{task_signature(t)}"
                    t["split"] = split_of(t["scene_id"], t["house"])
                    out[r].append(t)
    return out


def build_task_sets(seed: int = 7, per_len=(("train", 600), ("validation", 60), ("test", 100))):
    """frozen S7 manifests: for every required length, up to the stated number of tasks per split (sampled without replacement, seeded)"""
    world = World.get()
    rng = np.random.default_rng(seed)
    allk = enumerate_all_tasks(world)
    sets = {s: {} for s, _ in per_len}
    stats = {}
    for k in LENGTHS:
        stats[k] = {"enumerated": len(allk[k])}
        for split, n in per_len:
            pool = [t for t in allk[k] if t["split"] == split]
            idx = rng.permutation(len(pool))[:n]
            sets[split][k] = [pool[i] for i in sorted(idx)]
            stats[k][split] = {"available": len(pool), "used": len(sets[split][k])}
    return sets, stats


# ------------------------------------------------------------------------------------------- environment
class LongRouteEnv(gym.Env):
    """Sparse-reward long-route construction in the native observation/action interface. Reward: +1 exactly when the house is connected to the
    carpark (episode terminates), else 0. Horizon (truncation) = preregistered 10 * k decisions."""
    metadata = {"render_modes": []}

    def __init__(self, tasks: List[Dict[str, Any]], sampling: str = "random", horizon_fn=horizon_for, seed: int = 0):
        super().__init__()
        self.world = World.get()
        self.tasks = tasks
        self.sampling = sampling
        self.horizon_fn = horizon_fn
        self._cursor = 0
        self._rng = np.random.default_rng(seed)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(L.OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Discrete(L.N_ACTIONS)
        self._obs = np.zeros(L.OBS_DIM, dtype=np.float32)
        self._mask = np.zeros(L.N_ACTIONS, dtype=bool)
        self._gbase = L.N_TILES * L.NUM_TILE_FEATURES
        self.task = None
        self._static_cache: Dict[Tuple[int, Tuple[int, int]], np.ndarray] = {}

    # --- helpers
    def _off(self, x, y):
        return L.tile_index(x, y) * L.NUM_TILE_FEATURES

    def _static(self, scene_id, house):
        key = (scene_id, tuple(house))
        if key not in self._static_cache:
            s = self.world.scenes[scene_id]
            o = np.zeros(L.OBS_DIM, dtype=np.float32)
            tiles = o[: self._gbase].reshape(L.N_TILES, L.NUM_TILE_FEATURES)
            tiles[:, L.F_BUILDABLE] = 1.0
            for (x, y) in self.world.water:
                tiles[L.tile_index(x, y), L.F_WATER] = 1.0
            for (x, y) in s["dest"]:
                tiles[L.tile_index(x, y), L.F_CONTENT] = L.CONTENT_DEST
                tiles[L.tile_index(x, y), L.F_GROUP] = 1.0
            tiles[L.tile_index(*house), L.F_CONTENT] = L.CONTENT_HOUSE
            tiles[L.tile_index(*house), L.F_GROUP] = 1.0
            self._static_cache[key] = o
        return self._static_cache[key]

    def _set_edge(self, e, val):
        (ax, ay), (bx, by) = ends(e)
        d = e[2]
        self._obs[self._off(ax, ay) + L.F_ROAD[d]] = val
        self._obs[self._off(bx, by) + L.F_ROAD[L.OPP[d]]] = val

    def _add(self, e):
        self.edges.add(e)
        a, b = ends(e)
        self.deg[a] = self.deg.get(a, 0) + 1
        self.deg[b] = self.deg.get(b, 0) + 1
        self._set_edge(e, L.ROAD_ACTIVE)

    def _del(self, e):
        self.edges.discard(e)
        a, b = ends(e)
        self.deg[a] -= 1
        self.deg[b] -= 1
        self._set_edge(e, 0.0)

    def connected(self) -> bool:
        h, target = self.house, self.scene["ctile"]
        adj: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        for e in self.edges:
            a, b = ends(e)
            adj.setdefault(a, []).append(b)
            adj.setdefault(b, []).append(a)
        seen, stack = {h}, [h]
        while stack:
            u = stack.pop()
            if u == target:
                return True
            for v in adj.get(u, ()):
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        return False

    def remaining(self) -> int:
        return remaining_distance(self.world, self.scene, self.house, self.edges)

    def _globals(self):
        g = self._gbase
        hours = self.t * 1.2
        self._obs[g + L.G_HOUR] = (int(hours) % 24) / 24.0
        self._obs[g + L.G_DAY] = min(int(hours) // 24, 365) / 365.0
        self._obs[g + L.G_CONCRETE] = self.concrete / self.world.concrete_total
        self._obs[g + 5] = 1.0                      # bridges 1/1 (native global feature 5)

    def _compute_mask(self):
        m = self._mask
        m[:] = False
        m[0] = True
        h = self.house
        hset = set(house_edges(h))
        for e in self.scene["cand"]:
            if e in self.edges:
                continue
            if self.concrete <= 0 and e not in hset:
                continue
            m[L.action_index("ADD", *e)] = True
        for e in self.edges:
            if e in hset:
                continue                                       # house edges are never removable
            (x, y), _ = ends(e)
            if self.deg.get((x, y), 0) != 1:
                continue
            if e == self.scene["stub"]:
                if (x, y) == self.scene["ctile"]:                  # native: offered only from the carpark tile (and then has no effect)
                    m[L.action_index("REMOVE", *e)] = True
            elif (x, y) in self.scene["land"]:
                m[L.action_index("REMOVE", *e)] = True

    def action_masks(self) -> np.ndarray:
        return self._mask

    def snapshot(self):
        return (set(self.edges), dict(self.deg), self.concrete, self.n, self.t, self._obs.copy(), self._mask.copy(), self.success)

    def restore(self, snap):
        self.edges, self.deg, self.concrete, self.n, self.t = set(snap[0]), dict(snap[1]), snap[2], snap[3], snap[4]
        self._obs[:] = snap[5]
        self._mask[:] = snap[6]
        self.success = snap[7]

    # --- gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options and "task_index" in options:
            idx = options["task_index"]
        elif self.sampling == "sequential":
            idx = self._cursor % len(self.tasks)
            self._cursor += 1
        else:
            idx = int(self._rng.integers(len(self.tasks)))
        t = self.tasks[idx]
        self.task, self.scene = t, self.world.scenes[t["scene_id"]]
        self.house = tuple(t["house"])
        self.k = t["k_min"]
        self.horizon = self.horizon_fn(self.k)
        self.t = 1.0
        self.n = 0
        self.edges, self.deg = set(), {}
        self.concrete = self.world.concrete_total
        self._obs[:] = self._static(t["scene_id"], self.house)
        self._add(self.scene["stub"])
        self.drive = canon(self.house, L.neighbor(self.house[0], self.house[1], t["driveway_dir"]))
        self._add(self.drive)
        self.success = False
        self._globals()
        self._compute_mask()
        return self._obs.copy(), {"task_id": t["task_id"], "k": self.k, "horizon": self.horizon}

    def step(self, action: int):
        a = int(action)
        dec = L.decode_action(a)
        accepted = bool(self._mask[a])
        applied = "REJECTED" if not accepted else dec[0]
        if accepted and dec[0] == "ADD":
            e = (dec[1], dec[2], dec[3])
            a_, b_ = ends(e)
            if self.house in (a_, b_):                          # house edge replacement
                old = [x for x in house_edges(self.house) if x in self.edges]
                for o in old:
                    self._del(o)
                    self.concrete += 1
            self._add(e)
            self.concrete -= 1
        elif accepted and dec[0] == "REMOVE":
            e = (dec[1], dec[2], dec[3])
            if e != self.scene["stub"]:                         # native quirk: REMOVE on the carpark stub is accepted but has no effect
                self._del(e)
                self.concrete += 1
        self.n += 1
        self.t += 1.0
        self._globals()
        success = self.connected()
        self.success = success
        truncated = (not success) and self.n >= self.horizon
        self._compute_mask()
        reward = 1.0 if success else 0.0
        info = {"accepted": accepted, "appliedAction": applied, "success": success, "decisionIndex": self.n, "k": self.k, "task_id": self.task["task_id"],
                "rewardComponents": {"scoreDelta": reward, "terminalPenalty": 0.0}}
        return self._obs.copy(), reward, success, truncated, info
