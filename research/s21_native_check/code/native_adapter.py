"""S21: minimal raw-native-observation adapter for the S18-H planner/executor. NO game is launched here.
NativeStateView exposes exactly the attributes WorldModel(env) reads (t, concrete, total, water, content, houses, dests, edges, free, action_masks()) built from one native PolicyObservation
JSON (the dict the plugin returns on RESET/STEP, available as MiniMotorwaysEnv.last_raw_observation) plus the native legal mask. Only visible information is used: tile content/group,
per-edge road state strings, destination demand/overcrowding, globals. Road states: Active/VisiblyActive -> 'A' (functional), Mothballed -> 'M', Pending -> 'P' (present, not functional,
not re-addable); 'P' never occurred in 3,867 logged decisions (S14 audit) but is handled so the planner never treats it as empty land."""
import os
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
sys.path.insert(0, os.path.join(ROOT, "research", "hierarchical_planner", "code"))
from synthetic_rl import layout as L  # noqa: E402

STATE = {"Active": "A", "VisiblyActive": "A", "Mothballed": "M", "Pending": "P"}
DIRS = {"N": (0, 1), "E": (1, 0), "S": (0, -1), "W": (-1, 0)}


def _canon(x, y, d):
    if d == "E":
        return (x, y, "E")
    if d == "S":
        return (x, y, "S")
    if d == "W":
        return (x - 1, y, "E")
    return (x, y + 1, "S")                                   # N: neighbour (x, y+1) looks South to us


# ---- native legality patch (adapter-scoped, reversible): the S18-H WorldModel plans routes with its own geometric window rule (fitted to native in S13, median error 1 tile, p90 4 tiles).
# On real seeds that rule can admit edges the game refuses -> the executor abandons the plan ("mask_changed") and re-plans the same route (seed 507919 live probe: 107 abandons / 120 decisions).
# When the view is native and concrete > 0 the native legal mask is the authority for plain edges; at concrete == 0 the geometric rule is kept so plans can wait for a known top-up.
import world_model as _WM

_orig_sb = _WM.WorldModel.structurally_buildable
_orig_init = _WM.WorldModel.__init__


def _init(self, env):
    _orig_init(self, env)
    self.is_native = bool(getattr(env, "is_native", False))


def _sb(self, e):
    if getattr(self, "is_native", False):
        a, b = _WM.ends(e)
        if a in self.house_set or b in self.house_set:
            return bool(self.mask[L.action_index("ADD", *e)])   # house-edge swaps are free, so the native mask is always authoritative (e.g. a driveway onto water is refused natively, allowed by the S13 rule)
        if self.concrete > 0:
            return bool(self.mask[L.action_index("ADD", *e)])
    return _orig_sb(self, e)


_WM.WorldModel.__init__ = _init
_WM.WorldModel.structurally_buildable = _sb


class NativeStateView:
    """duck-types the subset of MultiHouseEnv the S18-H WorldModel reads."""

    def __init__(self, raw: Dict[str, Any], mask: np.ndarray, t: int, spawn_t: Dict[Tuple[int, int], int], total_override: Optional[int] = None):
        tiles = {(tt["coord"][0], tt["coord"][1]): tt for tt in raw["tiles"]}
        self.t = t
        g = raw["globals"]
        self.concrete = int(g["concreteAvailable"])
        self.total = int(g["concreteTotal"]) if total_override is None else total_override
        self.water: Set[Tuple[int, int]] = {p for p, tt in tiles.items() if tt.get("overWater")}
        self.content: Dict[Tuple[int, int], str] = {}
        self.group: Dict[Tuple[int, int], Any] = {}
        self.houses: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self.edges: Dict[Tuple[int, int, str], str] = {}
        self.pending_edges: Set[Tuple[int, int, str]] = set()
        for p, tt in tiles.items():
            c = tt.get("contentType", "None")
            if c in ("House", "Destination", "Carpark"):
                self.content[p] = c
                if tt.get("groupIndex") is not None:
                    self.group[p] = tt["groupIndex"]
            if c == "House":
                self.houses[p] = {"group": tt.get("groupIndex"), "t": spawn_t.setdefault(p, t)}
            for d, s in tt.get("road", {}).items():
                code = STATE.get(s)
                if code is None:
                    continue
                e = _canon(p[0], p[1], d)
                prev = self.edges.get(e)
                # both endpoint tiles report the same edge; functional wins over inactive if they ever disagree (never seen natively)
                if prev is None or (prev != "A" and code == "A"):
                    self.edges[e] = code
                if code == "P":
                    self.pending_edges.add(e)
        # destinations: connected components of Destination tiles of the SAME groupIndex (4-adjacency); carparks = adjacent Carpark tiles; demand from the tile's destination info
        self.dests: List[Dict[str, Any]] = []
        seen = set()
        for p, c in self.content.items():
            if c != "Destination" or p in seen:
                continue
            comp, st = [], [p]
            seen.add(p)
            while st:
                u = st.pop()
                comp.append(u)
                for dx, dy in DIRS.values():
                    v = (u[0] + dx, u[1] + dy)
                    if self.content.get(v) == "Destination" and v not in seen and self.group.get(v) == self.group.get(p):   # one block = same colour group; adjacent blocks of other colours are separate native models
                        seen.add(v)
                        st.append(v)
            cps = sorted({(u[0] + dx, u[1] + dy) for u in comp for dx, dy in DIRS.values() if self.content.get((u[0] + dx, u[1] + dy)) == "Carpark"})
            infos = [tiles[u].get("destination") or {} for u in comp]
            groups = [self.group[u] for u in comp if u in self.group]
            self.dests.append({"tiles": sorted(comp), "group": groups[0] if groups else None, "carparks": cps, "demand": max([i.get("totalDemand", 0) for i in infos] or [0]),
                               "overcrowding": any(i.get("isOvercrowding") for i in infos)})
        # 'free' edges = stubs at carparks and house driveways (edges touching a Carpark or House tile) -- the S13 env's notion used by the REMOVE candidate filter
        self.free: Set[Tuple[int, int, str]] = set()
        for e in self.edges:
            a, b = (e[0], e[1]), ((e[0] + 1, e[1]) if e[2] == "E" else (e[0], e[1] - 1))
            if self.content.get(a) in ("Carpark", "House") or self.content.get(b) in ("Carpark", "House"):
                self.free.add(e)
        self._mask = np.asarray(mask, dtype=bool)
        self.is_native = True

    def action_masks(self) -> np.ndarray:
        return self._mask

    def connected_houses(self) -> Set[Tuple[int, int]]:
        from world_model import connected_house_set
        grp = dict(self.group)
        return connected_house_set(self.content, grp, self.houses, {e for e, s in self.edges.items() if s == "A"})


class NativePlannerPolicy:
    """wraps an S18-H Executor so it can be driven by a native env: act(raw, mask) -> action. Keeps the per-episode decision counter and house spawn times (visible history only)."""

    def __init__(self, executor):
        self.ex = executor
        self.reset()

    def reset(self):
        self.t = 0
        self.spawn_t: Dict[Tuple[int, int], int] = {}
        self.ex.reset_episode()
        self.last_view: Optional[NativeStateView] = None
        self.forced_actions: List[Tuple[int, int]] = []       # (decision, action) taken because WAIT was illegal (upgrade interrupt) or the planner's action was masked

    def act(self, raw: Dict[str, Any], mask: np.ndarray) -> int:
        view = NativeStateView(raw, mask, self.t, self.spawn_t)
        self.last_view = view
        self.t += 1
        mask = np.asarray(mask, dtype=bool)
        if not mask[0]:
            # native 'ChooseUpgrade' interrupt: while an upgrade choice is pending only the ChooseUpgrade(i) actions are legal (Entrypoint.cs:2537-2545) and the offered
            # options are not observable. Policy: take the lowest legal action index (slot 0), log it, and do not touch the planner's current plan.
            legal = np.flatnonzero(mask)
            a = int(legal[0]) if len(legal) else 0
            self.forced_actions.append((self.t - 1, a))
            return a
        a = int(self.ex.act(view))
        if not mask[a]:
            self.forced_actions.append((self.t - 1, 0))
            a = 0
        return a
