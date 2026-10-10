"""S18-H world model: structured state built from VISIBLE information only.

Reads from a MultiHouseEnv (python/synthetic_rl/s13_env.py) exactly what a human player sees on screen:
  tiles (water), structures (houses / destinations / carparks and their COLOUR group), roads with their state (Active / Mothballed),
  the stub/driveway edges, the concrete counter, the decision counter (clock) and the legal-action mask (the game's own placement feedback).
It NEVER reads env.pending (future spawns), the scenario event list or the RNG.

Two layers:
  * construction grid   - per tile: water flag, content, in-build-window flag, degree, plus legality of ADD/REMOVE from the env mask
  * transportation graph- nodes = tiles; typed edges ('road' Active, 'road' Mothballed tracked separately, 'stub'); components over ACTIVE
                          edges only (functional connectivity, same rule as experiments/s11_guard.connected_house_set); houses /
                          destinations with groups; route length and an estimated travel cost = path length (a PROXY: no traffic model).
A 'motorway' edge type is defined but unsupported (no executable action exists in the Discrete(3111) space, see MECHANICS.md)."""
import os
import sys
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(ROOT, "python"))
from synthetic_rl import layout as L  # noqa: E402

Tile = Tuple[int, int]
Edge = Tuple[int, int, str]
DIRS = {"N": (0, 1), "E": (1, 0), "S": (0, -1), "W": (-1, 0)}
EDGE_TYPES = ("road", "stub", "motorway")          # 'motorway': typed, feasibility = unsupported
TOPUP_AT = [141, 281, 421, 561]                     # time-driven, known to the player (weekly)
TOPUP_AMT = [20, 30, 10, 20]
WIN_MIN_W, WIN_MIN_H = 14, 8


def canon(a: Tile, b: Tile) -> Edge:
    (ax, ay), (bx, by) = a, b
    return (min(ax, bx), ay, "E") if ay == by else (ax, max(ay, by), "S")


def ends(e: Edge) -> Tuple[Tile, Tile]:
    x, y, d = e
    return (x, y), ((x + 1, y) if d == "E" else (x, y - 1))


def neighbours(p: Tile):
    for dx, dy in DIRS.values():
        yield (p[0] + dx, p[1] + dy)


def house_edges(h: Tile) -> List[Edge]:
    return [canon(h, n) for n in neighbours(h)]


def next_topup(t: int) -> Tuple[Optional[int], int]:
    """(decisions until the next known top-up, amount); after decision 561 the rule is unknown (None, 0)."""
    for at, amt in zip(TOPUP_AT, TOPUP_AMT):
        if at > t:
            return at - t, amt
    return None, 0


class UnionFind:
    def __init__(self):
        self.p: Dict[Any, Any] = {}

    def add(self, a):
        self.p.setdefault(a, a)

    def find(self, a):
        p = self.p
        while p[a] != a:
            p[a] = p[p[a]]
            a = p[a]
        return a

    def union(self, a, b):
        self.add(a)
        self.add(b)
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            p = self.p
            p[ra] = rb


def functional_components(content: Dict[Tile, str], active: Set[Edge]) -> UnionFind:
    """union-find over ACTIVE edges + destination/carpark adjacency (the S11 / s13_env rule)."""
    uf = UnionFind()
    for p in content:
        uf.add(p)
    for e in active:
        a, b = ends(e)
        uf.union(a, b)
    for p, c in content.items():
        if c in ("Destination", "Carpark"):
            for n in neighbours(p):
                cn = content.get(n)
                if cn in ("Destination", "Carpark") and (c == "Destination" or cn == "Destination"):
                    uf.union(p, n)
    return uf


def connected_house_set(content: Dict[Tile, str], group: Dict[Tile, Any], houses, active: Set[Edge]) -> Set[Tile]:
    uf = functional_components(content, active)
    dg: Dict[Any, Set[Any]] = {}
    for p, c in content.items():
        if c == "Destination" and group.get(p) is not None:
            dg.setdefault(uf.find(p), set()).add(group[p])
    return {h for h in houses if group.get(h) is not None and group[h] in dg.get(uf.find(h), ())}


class WorldModel:
    """Snapshot of the visible state at one decision."""

    def __init__(self, env):
        self.t: int = env.t
        self.concrete: int = env.concrete
        self.total: int = env.total
        self.water: Set[Tile] = env.water
        self.content: Dict[Tile, str] = dict(env.content)
        self.group: Dict[Tile, Any] = {p: v["group"] for p, v in env.houses.items()}
        for d in env.dests:
            for p in d["tiles"]:
                self.group[p] = d["group"]
        self.houses: Dict[Tile, Dict[str, Any]] = {h: {"group": v["group"], "spawn_t": v["t"]} for h, v in env.houses.items()}
        self.dests: List[Dict[str, Any]] = [{"tiles": list(d["tiles"]), "group": d["group"], "carparks": list(d["carparks"]), "demand": d["demand"]} for d in env.dests]
        self.edges: Dict[Edge, str] = dict(env.edges)                 # 'A' | 'M'
        self.stub: Set[Edge] = set(env.free)
        self.active: Set[Edge] = {e for e, s in self.edges.items() if s == "A"}
        self.mothballed: Set[Edge] = {e for e, s in self.edges.items() if s == "M"}
        self.mask: np.ndarray = np.array(env.action_masks(), dtype=bool, copy=True)
        self.blocked: Set[Tile] = {p for p, c in self.content.items() if c in ("Destination", "Carpark")}
        self.house_set: Set[Tile] = set(self.houses)
        self.window = self._window()
        self.uf = functional_components(self.content, self.active)
        self.connected: Set[Tile] = connected_house_set(self.content, self.group, self.houses, self.active)
        self.degree: Dict[Tile, int] = {}
        for e in self.edges:
            for p in ends(e):
                self.degree[p] = self.degree.get(p, 0) + 1
        # destination component -> groups served; used by candidate goals
        self.dest_comp_groups: Dict[Any, Set[Any]] = {}
        for p, c in self.content.items():
            if c == "Destination" and self.group.get(p) is not None:
                self.dest_comp_groups.setdefault(self.uf.find(p), set()).add(self.group[p])

    # -------------------------------------------------------------- construction grid
    def _window(self):
        """the visible build boundary (the fitted native window rule, see s13_env docstring); structures are on screen."""
        pts = [p for p, c in self.content.items() if c in ("House", "Destination", "Carpark")]
        if not pts:
            return L.XMIN, L.XMIN + L.W - 1, L.YMIN, L.YMIN + L.H - 1
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        bw, bh = max(xs) - min(xs) + 1, max(ys) - min(ys) + 1
        ww, wh = max(WIN_MIN_W, bw + 3), max(WIN_MIN_H, bh + 3)
        x0, y1 = min(xs) - 1, max(ys)
        x1, y0 = x0 + ww - 1, y1 - wh + 1
        return max(x0, L.XMIN), min(x1, L.XMIN + L.W - 1), max(y0, L.YMIN), min(y1, L.YMIN + L.H - 1)

    def in_window(self, p: Tile) -> bool:
        x0, x1, y0, y1 = self.window
        return x0 <= p[0] <= x1 and y0 <= p[1] <= y1

    def legal_now(self, kind: str, e: Edge) -> bool:
        """legality according to the game's own mask (authoritative)."""
        return bool(self.mask[L.action_index(kind, *e)])

    def structurally_buildable(self, e: Edge) -> bool:
        """placement rule independent of the concrete counter (so routes can be planned while waiting for a top-up)."""
        a, b = ends(e)
        if not (self.in_window(a) and self.in_window(b) and L.in_bounds(*a) and L.in_bounds(*b)):
            return False
        if a in self.blocked or b in self.blocked or (a in self.water and b in self.water):
            return False
        ha, hb = a in self.house_set, b in self.house_set
        if ha and hb:
            return False
        if ha or hb:
            return self.edges.get(e) != "A"
        return e not in self.edges

    def tile_info(self, p: Tile) -> Dict[str, Any]:
        return {"water": p in self.water, "content": self.content.get(p, "None"), "group": self.group.get(p), "in_window": self.in_window(p), "degree": self.degree.get(p, 0),
                "roads": {d: self.edges.get(canon(p, (p[0] + dx, p[1] + dy))) for d, (dx, dy) in DIRS.items()}}

    # -------------------------------------------------------------- transportation graph
    def is_house_edge(self, e: Edge) -> bool:
        a, b = ends(e)
        return a in self.house_set or b in self.house_set

    def edge_type(self, e: Edge) -> str:
        return "stub" if e in self.stub else "road"

    def component(self, p: Tile):
        return self.uf.find(p) if p in self.uf.p else p

    def goal_tiles_for_group(self, g) -> Set[Tile]:
        """tiles of the Active components that contain a destination of colour group g (excluding structure tiles and houses):
        reaching any of them connects the house (reuse of existing roads is implicit)."""
        comps = {c for c, gs in self.dest_comp_groups.items() if g in gs}
        return {p for p in self.uf.p if self.uf.find(p) in comps and p not in self.blocked and p not in self.house_set}

    def unreached_dests(self) -> List[Dict[str, Any]]:
        """S35 fix: destinations that NO house of their own colour can reach (their component holds no such house). Their pins are never served and
        accumulate until game-over; the house-centric candidate generator never targets them because every house is already 'connected' elsewhere."""
        out = []
        for d in self.dests:
            if d["group"] is None:
                continue
            comp = self.component(d["tiles"][0])
            if not any(v["group"] == d["group"] and self.component(h) == comp for h, v in self.houses.items()):
                out.append(d)
        return out

    def tiles_of_component(self, comp) -> Set[Tile]:
        return {p for p in self.uf.p if self.uf.find(p) == comp and p not in self.blocked and p not in self.house_set}

    def predicted_active_after(self, adds: List[Edge], removes: List[Edge]) -> Set[Edge]:
        """functional edge set after applying ADDs (with the native house-edge replacement rule) and REMOVEs, in order."""
        act = set(self.active)
        for e in adds:
            a, b = ends(e)
            hs = [p for p in (a, b) if p in self.house_set]
            if hs:
                h = hs[0]
                for x in house_edges(h):
                    if x != e:
                        act.discard(x)             # old Active driveway becomes Mothballed (or is deleted on reactivation)
            act.add(e)
        for e in removes:
            act.discard(e)
        return act

    def connected_after(self, adds: List[Edge], removes: List[Edge]) -> Set[Tile]:
        return connected_house_set(self.content, self.group, self.houses, self.predicted_active_after(adds, removes))

    def lost_connections(self, adds: List[Edge], removes: List[Edge]) -> Set[Tile]:
        return self.connected - self.connected_after(adds, removes)

    def demand_of_group(self, g) -> Tuple[int, bool]:
        """(total pins at destinations of group g, any overcrowded)"""
        tot, over = 0, False
        for d in self.dests:
            if d["group"] == g:
                tot += d["demand"]
                over = over or d["demand"] >= 7
        return tot, over

    def summary(self) -> Dict[str, Any]:
        return {"t": self.t, "concrete": self.concrete, "houses": len(self.houses), "connected": len(self.connected), "active_edges": len(self.active), "mothballed_edges": len(self.mothballed),
                "window": self.window, "next_topup": next_topup(self.t)}
