"""S13b: minimal MULTI-HOUSE synthetic environment (separate from, and not modifying, S7 / S8). Native observation (37x21x15 + 11 = 11,666 floats) and Discrete(3111) action space.

Mechanics (each grounded in native S9-S13 logs; see research/s13_multi_house for the evidence and the fidelity matrix):
  * scenario = a native game seed's initial structures + one REALISED native spawn timeline (houses / destinations with groups, free driveway / stub edges)
  * concrete: 25 at start; a plain (non-house) ADD costs 1; house-touching ADD / swap costs 0; REMOVE refunds 1; weekly top-ups +20 / +30 / +10 / +20 at decisions 141 / 281 / 421 / 561
  * road states: Active (functional) and Mothballed (inactive). A house has one Active driveway. ADD of a different edge at a house: new edge Active, the old Active edge becomes Mothballed (a previous Mothballed one is deleted);
    ADD of the house's Mothballed edge reactivates it and deletes the displaced Active edge (native probe, S11). Connectivity uses ACTIVE edges only
  * build window = fitted rectangle (max(14, w+3) x max(8, h+3)) anchored 1 tile left of and level with the top of the structure bounding box; ADD legal iff both endpoints in the window, not destination / carpark, not both water,
    not two houses, concrete available (house-touching ADDs are free)
  * demand: each destination gains a pin w.p. UP (0.22 from zero) per decision and loses one w.p. DOWN[n][bucket] (n = connected same-group houses; floor 0) - empirical native table; every functionally connected house
    delivers (+1 reward) w.p. MU=0.0436 per decision, independent of pins (native deliveries are linear in connected houses). A destination with demand >= 7 is overcrowded; overcrowding for 50 consecutive decisions ends the game
  * reward: +1 per delivery, -10 once at game over (continuing delivery reward; no per-edge, distance, road-count or progress terms). Horizon 600 decisions
Simplifications (documented, not hidden): no traffic / vehicles (vehicle channel 0), no weekly upgrade choice, house / destination colours are group indices, spawn positions relocate by a small shift when the agent has built on them,
REMOVE always refunds 1 (native: 82 %), the build window is an approximation of the native one."""
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from . import layout as L

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCEN_FILE = os.path.join(ROOT, "research", "s13_multi_house", "scenarios.json")
SCENE_FILE = os.path.join(ROOT, "research", "s7_long_route", "scenes.json")
DIRS = {"N": (0, 1), "E": (1, 0), "S": (0, -1), "W": (-1, 0)}
OPP = {"N": "S", "S": "N", "E": "W", "W": "E"}
TOPUP_AT = [141, 281, 421, 561]
TOPUP_AMT = [20, 30, 10, 20]
# empirical per-destination demand transitions (native S13a replicate logs, 12 episodes, ~14k destination-steps): P(+1) = UP0 if demand == 0 else UP; P(-1 | n connected same-group houses, demand bucket 1 / >=2)
UP0, UP = 0.22, 0.055
DOWN = {0: (0.005, 0.005), 1: (0.022, 0.074), 2: (0.034, 0.100), 3: (0.043, 0.092), 4: (0.051, 0.104)}
LAM, MU, OVER_THR, OVER_TIMER, TERM_PENALTY, HORIZON = UP, 0.0436, 7, 50, -10.0, 600
WIN_MIN_W, WIN_MIN_H = 14, 8           # build window = structure bbox (w,h) -> (max(14, w+3), max(8, h+3)), left margin 1, top aligned with the top structure row (fitted to native legal-ADD extents)
Edge = Tuple[int, int, str]


def canon(a, b) -> Edge:
    (ax, ay), (bx, by) = a, b
    return (min(ax, bx), ay, "E") if ay == by else (ax, max(ay, by), "S")


def ends(e: Edge):
    x, y, d = e
    return (x, y), ((x + 1, y) if d == "E" else (x, y - 1))


def load_scenarios(split: Optional[str] = None) -> List[Dict[str, Any]]:
    d = json.load(open(SCEN_FILE))
    sc = d["scenarios"]
    return [s for s in sc if split is None or s["split"] == split]


class MultiHouseEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, scenarios: List[Dict[str, Any]], seed: int = 0, sampling: str = "random", horizon: int = HORIZON, lam: float = LAM, mu: float = MU):
        super().__init__()
        self.scenarios = scenarios
        self.sampling = sampling
        self.horizon, self.lam, self.mu = horizon, lam, mu
        self._rng = np.random.default_rng(seed)
        self._cursor = 0
        self.water = {tuple(w) for w in json.load(open(SCENE_FILE))["water_tiles"]}
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(L.OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Discrete(L.N_ACTIONS)
        self._mask = np.zeros(L.N_ACTIONS, dtype=bool)
        self._obs = np.zeros(L.OBS_DIM, dtype=np.float32)
        self.scenario = None

    # ------------------------------------------------------------------ state helpers
    def _reset_state(self, sc):
        self.t = 0
        self.score = 0
        self.concrete, self.total = 25, 25
        self.edges: Dict[Edge, str] = {}
        self.free: set = set()               # driveway / stub edges that were created for free
        self.content: Dict[Tuple[int, int], str] = {}
        self.group: Dict[Tuple[int, int], int] = {}
        self.houses: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self.dests: List[Dict[str, Any]] = []
        self.over_timer = 0
        self.game_over = False
        self.skipped_events = 0
        self.deliveries_total = 0
        self.event_log: List[Dict[str, Any]] = []
        self.pending = sorted(sc["events"], key=lambda e: e["t"])
        self._apply_struct(sc["structures0"], sc["free_edges0"], 0)
        # the very first house arrives with the first logged event (native: after the first decision)

    def _apply_struct(self, struct, free_edges, t) -> bool:
        tiles = [(s[0], s[1]) for s in struct]
        dx = dy = 0
        found = False
        cand = [(0, 0)] + [(a, b) for r in (1, 2, 3) for a in range(-r, r + 1) for b in range(-r, r + 1) if max(abs(a), abs(b)) == r]
        for (ox, oy) in cand:
            ok = True
            for (x, y) in tiles:
                p = (x + ox, y + oy)
                if not L.in_bounds(*p) or p in self.water or p in self.content or any(p in ends(e) for e in self.edges if e not in free_edges_set(free_edges, ox, oy)):
                    ok = False
                    break
            if ok:
                dx, dy, found = ox, oy, True
                break
        if not found:
            self.skipped_events += 1
            return False
        grp_dest = {}
        for (x, y, c, g) in struct:
            p = (x + dx, y + dy)
            self.content[p] = c
            if g is not None:
                self.group[p] = g
            if c == "House":
                self.houses[p] = {"group": g, "t": t}
        dts = [(x + dx, y + dy, g) for (x, y, c, g) in struct if c == "Destination"]
        if dts:
            cps = [(x + dx, y + dy) for (x, y, c, g) in struct if c == "Carpark"]
            self.dests.append({"tiles": [(a, b) for a, b, _ in dts], "group": dts[0][2], "carparks": cps, "demand": 0, "t": t})
        for e in free_edges:
            e2 = (e[0] + dx, e[1] + dy, e[2])
            self.edges[e2] = "A"
            self.free.add(e2)
        self.event_log.append({"t": t, "tiles": len(tiles), "shift": (dx, dy)})
        return True

    def _window(self):
        pts = [p for p, c in self.content.items() if c in ("House", "Destination", "Carpark")]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        bw, bh = max(xs) - min(xs) + 1, max(ys) - min(ys) + 1
        ww, wh = max(WIN_MIN_W, bw + 3), max(WIN_MIN_H, bh + 3)
        x0, y1 = min(xs) - 1, max(ys)
        x1, y0 = x0 + ww - 1, y1 - wh + 1
        return max(x0, L.XMIN), min(x1, L.XMIN + L.W - 1), max(y0, L.YMIN), min(y1, L.YMIN + L.H - 1)

    def functional_edges(self):
        return {e for e, s in self.edges.items() if s == "A"}

    def connected_houses(self) -> set:
        parent: Dict[Tuple[int, int], Tuple[int, int]] = {}

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for p, c in self.content.items():
            parent[p] = p
        for e in self.functional_edges():
            a, b = ends(e)
            parent.setdefault(a, a)
            parent.setdefault(b, b)
            union(a, b)
        for p, c in self.content.items():
            if c in ("Destination", "Carpark"):
                for dx, dy in DIRS.values():
                    n = (p[0] + dx, p[1] + dy)
                    if self.content.get(n) in ("Destination", "Carpark") and (c == "Destination" or self.content[n] == "Destination"):
                        union(p, n)
        dgroups: Dict[Any, set] = {}
        for p, c in self.content.items():
            if c == "Destination":
                dgroups.setdefault(find(p), set()).add(self.group.get(p))
        return {h for h, v in self.houses.items() if v["group"] in dgroups.get(find(h), ())}

    # ------------------------------------------------------------------ masks
    def _house_edges(self, h) -> List[Edge]:
        return [canon(h, (h[0] + dx, h[1] + dy)) for dx, dy in DIRS.values()]

    def _compute_mask(self):
        m = self._mask
        m[:] = False
        m[0] = True
        x0, x1, y0, y1 = self._window()
        blocked = {p for p, c in self.content.items() if c in ("Destination", "Carpark")}
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                for d in ("E", "S"):
                    e = (x, y, d)
                    a, b = ends(e)
                    if not (x0 <= b[0] <= x1 and y0 <= b[1] <= y1 and L.in_bounds(*b)):
                        continue
                    if a in blocked or b in blocked or (a in self.water and b in self.water):
                        continue
                    ha, hb = self.content.get(a) == "House", self.content.get(b) == "House"
                    if ha and hb:
                        continue
                    if ha or hb:
                        if self.edges.get(e) != "A":
                            m[L.action_index("ADD", *e)] = True
                    else:
                        if e not in self.edges and self.concrete > 0:
                            m[L.action_index("ADD", *e)] = True
        deg: Dict[Tuple[int, int], int] = {}
        for e in self.edges:
            for p in ends(e):
                deg[p] = deg.get(p, 0) + 1
        for e, s in self.edges.items():
            a, _ = ends(e)
            if self.content.get(a) == "House" or self.content.get(ends(e)[1]) == "House":
                continue
            if e in self.free:                                    # stub: native offers REMOVE only from the carpark tile and it has no effect
                if self.content.get(a) == "Carpark":
                    m[L.action_index("REMOVE", *e)] = True
                continue
            if deg.get(a, 0) == 1:
                m[L.action_index("REMOVE", *e)] = True

    def action_masks(self) -> np.ndarray:
        return self._mask

    # ------------------------------------------------------------------ observation
    def _off(self, x, y):
        return L.tile_index(x, y) * L.NUM_TILE_FEATURES

    def _build_obs(self):
        o = self._obs
        o[:] = 0.0
        g0 = L.N_TILES * L.NUM_TILE_FEATURES
        tiles = o[:g0].reshape(L.N_TILES, L.NUM_TILE_FEATURES)
        tiles[:, L.F_BUILDABLE] = 1.0
        for (x, y) in self.water:
            tiles[L.tile_index(x, y), L.F_WATER] = 1.0
        for p, c in self.content.items():
            i = L.tile_index(*p)
            if c == "House":
                tiles[i, L.F_CONTENT] = L.CONTENT_HOUSE
                tiles[i, L.F_GROUP] = 1.0
            elif c == "Destination":
                tiles[i, L.F_CONTENT] = L.CONTENT_DEST
                tiles[i, L.F_GROUP] = 1.0
            # Carpark: the native encoder maps it to 0 (and it has no group index)
        for d in self.dests:
            for p in d["tiles"]:
                i = L.tile_index(*p)
                tiles[i, L.F_DEMAND] = min(d["demand"] / 50.0, 1.0)
                tiles[i, L.F_OVERCROWD] = 1.0 if d["demand"] >= OVER_THR else 0.0
        for e, s in self.edges.items():
            val = (2.0 if s == "A" else 3.0) / 3.0
            (ax, ay), (bx, by) = ends(e)
            dd = e[2]
            tiles[L.tile_index(ax, ay), L.F_ROAD[dd]] = val
            tiles[L.tile_index(bx, by), L.F_ROAD[OPP[dd]]] = val
        hour = int(1.2 * self.t) + 1
        day = hour // 24
        o[g0 + L.G_HOUR] = hour / 24.0
        o[g0 + L.G_DAY] = min(day, 365) / 365.0
        o[g0 + L.G_WEEK] = min(day // 7, 52) / 52.0
        o[g0 + L.G_SCORE] = self.score / 1000.0
        o[g0 + L.G_CONCRETE] = self.concrete / self.total
        o[g0 + 5] = 1.0

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        if options and "scenario_index" in options:
            idx = options["scenario_index"]
        elif self.sampling == "sequential":
            idx = self._cursor % len(self.scenarios)
            self._cursor += 1
        else:
            idx = int(self._rng.integers(len(self.scenarios)))
        self.scenario = self.scenarios[idx]
        self._reset_state(self.scenario)
        self._compute_mask()
        self._build_obs()
        return self._obs.copy(), {"scenario": self.scenario["id"]}

    def _apply_action(self, a: int):
        dec = L.decode_action(a)
        if dec[0] == "ADD":
            e = (dec[1], dec[2], dec[3])
            pa, pb = ends(e)
            hs = [p for p in (pa, pb) if self.content.get(p) == "House"]
            if not hs:
                self.edges[e] = "A"
                self.concrete -= 1
            else:
                h = hs[0]
                cur = [x for x in self._house_edges(h) if self.edges.get(x) == "A"]
                moth = [x for x in self._house_edges(h) if self.edges.get(x) == "M"]
                if self.edges.get(e) == "M":
                    for x in cur:
                        self.edges.pop(x, None)
                        self.free.discard(x)
                    self.edges[e] = "A"
                else:
                    for x in moth:
                        self.edges.pop(x, None)
                    for x in cur:
                        self.edges[x] = "M"
                    self.edges[e] = "A"
        elif dec[0] == "REMOVE":
            e = (dec[1], dec[2], dec[3])
            if e in self.free:
                return                                           # native quirk: accepted, no effect
            if e in self.edges:
                self.edges.pop(e)
                self.concrete += 1

    def step(self, action: int):
        a = int(action)
        accepted = bool(self._mask[a])
        if accepted:
            self._apply_action(a)
        self.t += 1
        reward = 0.0
        # spawns scheduled for this decision
        while self.pending and self.pending[0]["t"] <= self.t:
            ev = self.pending.pop(0)
            self._apply_struct(ev["struct"], [tuple(e) for e in ev["free_edges"]], self.t)
        # weekly concrete
        if self.t in TOPUP_AT:
            amt = TOPUP_AMT[TOPUP_AT.index(self.t)]
            self.concrete += amt
            self.total += amt
        elif self.t > TOPUP_AT[-1] and (self.t - 1) % 140 == 0:
            self.concrete += 20
            self.total += 20
        # demand (empirical per-destination transition table) and deliveries (house-generated, independent of pins: native deliveries are linear in connected houses)
        conn = self.connected_houses()
        for d in self.dests:
            n = sum(1 for h in conn if self.houses[h]["group"] == d["group"])
            if d["group"] is None:
                continue                                          # native destinations without a colour group never generate tracked demand
            u = self._rng.random()
            dm = d["demand"]
            if u < (UP0 * self.lam / UP if dm == 0 else self.lam):
                d["demand"] += 1
            elif dm > 0 and self._rng.random() < DOWN[min(n, 4)][0 if dm == 1 else 1]:
                d["demand"] -= 1
        deliveries = int(sum(1 for _ in conn if self._rng.random() < self.mu))
        self.score += deliveries
        self.deliveries_total += deliveries
        reward += deliveries
        if any(d["demand"] >= OVER_THR for d in self.dests):
            self.over_timer += 1
        else:
            self.over_timer = 0
        terminated = self.over_timer >= OVER_TIMER
        if terminated:
            self.game_over = True
            reward += TERM_PENALTY
        truncated = (not terminated) and self.t >= self.horizon
        self._compute_mask()
        self._build_obs()
        info = {"accepted": accepted, "deliveries": deliveries, "connected": len(conn), "houses": len(self.houses), "concrete": self.concrete, "demand_max": max([d["demand"] for d in self.dests] or [0]),
                "over_timer": self.over_timer, "score": self.score, "game_over": terminated, "t": self.t}
        return self._obs.copy(), float(reward), bool(terminated), bool(truncated), info


def free_edges_set(free_edges, ox, oy):
    return {(e[0] + ox, e[1] + oy, e[2]) for e in free_edges}
