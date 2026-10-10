"""S20 shared utilities: paths, the goal-conditioned wrapper around the S18-H executor, the goal-conditioned observation (v2 + 2 goal channels),
the extended legal mask (Discrete(3111) + DONE), env state save/restore, and the S16-style valid-set oracle adapted to the executor's state.
Nothing here modifies any existing file; the S18-H planner modules are imported from research/hierarchical_planner/code and used through their
real interfaces (WorldModel, connect_candidates, score_all, Executor)."""
import os
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
S20 = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(S20))
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, os.path.join(ROOT, "research", "hierarchical_planner", "code"))

from synthetic_rl import layout as L  # noqa: E402
from synthetic_rl import obs_v2 as V  # noqa: E402
from synthetic_rl.s13_env import DIRS, canon, ends  # noqa: E402
from synthetic_rl.s14_env import MultiHouseEnvV2  # noqa: E402
from world_model import WorldModel  # noqa: E402
from candidates import connect_candidates, wait_candidate  # noqa: E402
from planner import Weights, plan as full_plan, score_all  # noqa: E402
from executor import Executor  # noqa: E402

N_GOAL_CH = 2
N_TILE_CH = V.NUM_TILE_FEATURES_V2 + N_GOAL_CH            # 23
OBS_DIM_GOAL = L.N_TILES * N_TILE_CH + L.NUM_GLOBAL_FEATURES   # 17,882
A_DONE = L.N_ACTIONS                                       # 3111
N_ACT = L.N_ACTIONS + 1                                    # 3112
TILE_SCALE = 6                                             # every v2 tile value * 6 is an integer in [0, 6] (checked in checks.py)
START_CONCRETE = 80                                        # resources never bind in the construction tasks (documented)

SEEDS = {"train": [(20001, 20900), (24001, 24900)], "val": [(21001, 21150)], "test": [(22001, 22400)]}
S13_NATIVE_TEST_SEEDS = {515838, 539595, 563352, 587109}


def seed_range(split):
    out = []
    for a, b in SEEDS[split]:
        out += list(range(a, b + 1))
    return out


# ------------------------------------------------------------------------------------------------ goal-conditioned executor
class GoalExecutor(Executor):
    """The S18-H executor restricted to ONE objective: CONNECT the given house (its colour group's destination component).
    Candidate generation = the real `connect_candidates` with houses=[goal] (same Dijkstra, same alternatives, same functional-connection veto);
    scoring = the real `score_all`; no REMOVE / other-house candidates are ever generated, so no strategic choice is imitated.
    With goal=None it reproduces the full planner exactly (equivalence is checked in checks.py)."""

    def __init__(self, goal: Optional[Tuple[int, int]] = None, weights: Optional[Weights] = None, log: bool = False):
        super().__init__(weights, log)
        self.goal = goal
        self.planned_targets: List[Any] = []

    def _plan(self, wm: WorldModel, banned, protected):
        if self.goal is None:
            return full_plan(wm, self.w, banned, protected)
        cands = [wait_candidate()] + connect_candidates(wm, self.w.k_alternatives, banned, houses=[self.goal])
        score_all(wm, cands, self.w)
        order = {"CONNECT": 0, "REMOVE": 1, "WAIT": 2, "MOTORWAY": 3}
        return max(cands, key=lambda c: (c.score, -order[c.kind], tuple(-x for x in (c.concrete_cost, c.route_len)), tuple(c.adds)))

    def act(self, env) -> int:            # the base Executor.act with the planning call routed through _plan (logic otherwise identical)
        import time
        t0 = time.perf_counter()
        wm = WorldModel(env)
        if wm.t < self.last_t or self.last_t < 0:
            if self.last_t >= 0:
                self.reset_episode()
        self.last_t = wm.t
        self.stats["decisions"] += 1
        banned, protected = self._banned_and_protected(wm.t)
        if self.plan is not None:
            if self.plan.kind == "CONNECT" and self.plan.targets["house"] in wm.connected:
                self._complete()
            elif self.plan.kind == "REMOVE" and self.plan.removes[0] not in wm.edges:
                self._complete()
            else:
                r = self._invalidated(wm)
                if r is not None:
                    self._abandon(r)
                elif self.plan.kind == "CONNECT" and all(e in self.done_adds for e in self.plan.adds):
                    self._abandon("all_edges_built_but_not_connected")
        if self.plan is None:
            self.plan = self._plan(wm, banned, protected)
            self.plan_t0 = wm.t
            self.done_adds = set()
            if self.plan.kind in ("CONNECT", "REMOVE"):
                self.stats["plans"] += 1
                self.planned_targets.append(self.plan.targets.get("house"))
                if self.log_enabled:
                    self.log.append(f"t={wm.t} PLAN {self.plan.describe()} score={self.plan.score:.2f}")
            if self.plan.would_destroy_functional_connection:
                self.stats["destroy_vetoes"] += 1
                self._abandon("would_destroy")
                self.stats["compute_s"] += time.perf_counter() - t0
                return 0
        a = 0
        p = self.plan
        if p.kind == "CONNECT":
            for e in p.adds:
                if e in self.done_adds:
                    continue
                if wm.edges.get(e) == "A":
                    self.done_adds.add(e)
                    continue
                idx = L.action_index("ADD", *e)
                if wm.legal_now("ADD", e):
                    if wm.lost_connections([e], []):
                        self.stats["destroy_vetoes"] += 1
                        self._abandon("would_destroy")
                        a = 0
                    else:
                        a = idx
                        self.done_adds.add(e)
                        self.recent_added[e] = wm.t
                elif wm.structurally_buildable(e) and wm.concrete == 0:
                    self.stats["waits_for_concrete"] += 1
                    a = 0
                else:
                    self.stats["illegal_suppressed"] += 1
                    self._abandon("mask_changed")
                    a = 0
                break
        elif p.kind == "REMOVE":
            e = p.removes[0]
            if wm.legal_now("REMOVE", e) and not wm.lost_connections([], [e]):
                a = L.action_index("REMOVE", *e)
                self.recent_removed[e] = wm.t
            else:
                self._abandon("remove_invalid")
        if p.kind == "WAIT":
            self.plan = None
        if not wm.mask[a]:
            self.stats["illegal_suppressed"] += 1
            a = 0
        self.stats["compute_s"] += time.perf_counter() - t0
        return int(a)


def goal_route(env, goal) -> Optional[Dict[str, Any]]:
    """the planner's chosen route for `goal` from the current state (alt 0 of the real candidate generator), or None."""
    wm = WorldModel(env)
    if goal in wm.connected:
        return None
    cs = connect_candidates(wm, 0, frozenset(), houses=[goal])
    if not cs:
        return None
    c = cs[0]
    return {"adds": list(c.adds), "route_len": int(c.route_len), "n_adds": len(c.adds), "reused": int(c.reused_edges), "cost": int(c.concrete_cost), "path": list(c.path)}


# ------------------------------------------------------------------------------------------------ observation / mask
def goal_tiles(env, goal) -> Tuple[np.ndarray, np.ndarray]:
    """(house channel, destination channel) as (N_TILES,) float32: ONLY the goal house tile and the destination tiles of its colour group."""
    g = env.houses[goal]["group"]
    h = np.zeros(L.N_TILES, dtype=np.float32)
    d = np.zeros(L.N_TILES, dtype=np.float32)
    h[L.tile_index(*goal)] = 1.0
    for dd in env.dests:
        if dd["group"] == g:
            for p in dd["tiles"]:
                d[L.tile_index(*p)] = 1.0
    return h, d


def goal_obs_u8(env, goal) -> Tuple[np.ndarray, np.ndarray]:
    """compact storage: tiles (N_TILES, 23) uint8 (= value * TILE_SCALE, exact), globals (11,) float32."""
    o = env._obs_v2
    tiles = o[: L.N_TILES * V.NUM_TILE_FEATURES_V2].reshape(L.N_TILES, V.NUM_TILE_FEATURES_V2)
    h, d = goal_tiles(env, goal)
    full = np.concatenate([tiles, h[:, None], d[:, None]], axis=1)
    scaled = np.rint(full * TILE_SCALE)
    assert np.allclose(scaled / TILE_SCALE, full, atol=1e-6) and scaled.min() >= 0 and scaled.max() <= 255
    return scaled.astype(np.uint8), o[L.N_TILES * V.NUM_TILE_FEATURES_V2:].astype(np.float32).copy()


def legal_mask_ext(env) -> np.ndarray:
    m = np.zeros(N_ACT, dtype=bool)
    m[: L.N_ACTIONS] = env.action_masks()
    m[A_DONE] = True
    return m


# ------------------------------------------------------------------------------------------------ env state save / restore
def save_state(env) -> Dict[str, Any]:
    return {"scenario": env.scenario, "t": env.t, "concrete": env.concrete, "total": env.total,
            "edges": [[e[0], e[1], e[2], s] for e, s in env.edges.items()], "free": [list(e) for e in env.free],
            "content": [[p[0], p[1], c] for p, c in env.content.items()], "group": [[p[0], p[1], g] for p, g in env.group.items()],
            "houses": [[h[0], h[1], v["group"], v["t"]] for h, v in env.houses.items()],
            "dests": [{"tiles": [list(p) for p in d["tiles"]], "group": d["group"], "carparks": [list(p) for p in d["carparks"]], "demand": d["demand"], "t": d["t"]} for d in env.dests],
            "slots": [[g, s] for g, s in env.tracker.slots.items()]}


def restore_env(state: Dict[str, Any]) -> MultiHouseEnvV2:
    env = MultiHouseEnvV2([state["scenario"]], seed=0, lam=0.0, mu=0.0, horizon=10 ** 6)
    env.reset()
    env.pending = []
    env.t, env.concrete, env.total = state["t"], state["concrete"], state["total"]
    env.edges = {(e[0], e[1], e[2]): s for e in state["edges"] for s in [e[3]]}
    env.free = {tuple(e) for e in state["free"]}
    env.content = {(p[0], p[1]): c for p, c in [(x[:2], x[2]) for x in state["content"]]}
    env.group = {(p[0], p[1]): g for p, g in [(x[:2], x[2]) for x in state["group"]]}
    env.houses = {(h[0], h[1]): {"group": h[2], "t": h[3]} for h in state["houses"]}
    env.dests = [{"tiles": [tuple(p) for p in d["tiles"]], "group": d["group"], "carparks": [tuple(p) for p in d["carparks"]], "demand": d["demand"], "t": d["t"]} for d in state["dests"]]
    env.tracker.slots = {g: s for g, s in state["slots"]}
    env._compute_mask()
    env._build_obs()
    return env


def water_in_window(env) -> int:
    x0, x1, y0, y1 = env._window()
    return sum(1 for (x, y) in env.water if x0 <= x <= x1 and y0 <= y <= y1)


# ------------------------------------------------------------------------------------------------ valid-set oracle (S16 idea, executor's state)
def oracle_valid(env, goal) -> Tuple[Set[int], int]:
    """S16-style oracle adapted to the S18-H executor's goal definition: goals = every non-structure tile of an Active component containing a
    destination of the goal house's colour (WorldModel.goal_tiles_for_group); costs = concrete only (Active edge 0, house edge 0, new plain edge 1,
    never through other houses / structures, structurally buildable edges only).  Valid = every legal, not-yet-Active ADD leaving the house's current
    Active component (frontier) that lies on SOME minimum-concrete route.  Returns (set of action indices, best cost); ({A_DONE}, 0) if the goal is
    connected; (set(), -1) if unreachable."""
    import heapq
    wm = WorldModel(env)
    if goal in wm.connected:
        return {A_DONE}, 0
    goals = wm.goal_tiles_for_group(wm.houses[goal]["group"])
    if not goals:
        return set(), -1

    def nbrs(u):
        for v in [(u[0] + dx, u[1] + dy) for dx, dy in DIRS.values()]:
            if v in wm.house_set and v != goal:
                continue
            if v in wm.blocked:
                continue
            e = canon(u, v)
            if wm.edges.get(e) == "A":
                yield v, 0, e
            elif wm.structurally_buildable(e):
                yield v, (0 if (u == goal or v == goal) else 1), e

    def dijkstra(src):
        dist = {s: 0 for s in src}
        pq = [(0, s) for s in src]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            if u != goal and u in wm.house_set:
                continue
            for v, c, _ in nbrs(u):
                if d + c < dist.get(v, 1e9):
                    dist[v] = d + c
                    heapq.heappush(pq, (d + c, v))
        return dist
    dh = dijkstra({goal})
    dg = dijkstra(goals)
    best = min((dh.get(g, 10 ** 9) for g in goals), default=10 ** 9)
    if best >= 10 ** 9:
        return set(), -1
    # frontier = the house's current Active component (Active edges only)
    comp = wm.component(goal)
    frontier = {p for p in wm.uf.p if wm.uf.find(p) == comp} | {goal}
    valid = set()
    for u in frontier:
        if u != goal and u in wm.house_set:
            continue
        for v, c, e in nbrs(u):
            if wm.edges.get(e) == "A":
                continue
            if not wm.legal_now("ADD", e):
                continue
            if dh.get(u, 10 ** 9) + c + dg.get(v, 10 ** 9) == best:
                valid.add(L.action_index("ADD", *e))
    return valid, int(best)


def decode(a: int) -> str:
    if a == A_DONE:
        return "DONE"
    d = L.decode_action(a)
    return d[0] if len(d) == 1 else f"{d[0]}({d[1]},{d[2]},{d[3]})"
