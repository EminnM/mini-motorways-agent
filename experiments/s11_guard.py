"""S11 evaluation-only connection-preservation guard (pure functions on the RAW native observation; never given to the policy).

Purpose: causal probe. For each native-legal ADD, predict the road graph that results from applying it (including the house-edge replacement rule) and VETO the action iff a
house that is connected to a destination of its colour group before the action would lose that connection afterwards. Nothing else is changed: the mask, the network, the
reward and the game are untouched; WAIT / REMOVE / UPGRADE actions are never vetoed.

Counterfactual model of one ADD (edge e between tiles a and b, canonical undirected edge (x,y,'E'|'S'), South = y-1):
  * no endpoint is a House tile              : edge e is added, nothing is removed                       (known)
  * exactly one endpoint is a House tile h   : if e is already an edge of h -> no change (known);
        if h has exactly one edge  -> that edge is removed and e is added (the native house single-driveway replacement)   (known)
        if h has no edge           -> e is added                                                                              (known)
        if h has >= 2 edges        -> unknown (not vetoed)
  * both endpoints are House tiles, or an endpoint is a Destination/Carpark tile       -> unknown (not vetoed)
Adding edges can never disconnect anything, so only candidates whose predicted result REMOVES an edge are evaluated. Water endpoints get no special treatment: native
legality (the mask) is authoritative and a water edge not touching a house is a plain extension.
Connectivity = the same union-find as mini_motorways_env.connectivity_potential (houses whose road component contains a destination of the same groupIndex), but returning
the house set. Candidates predicted 'unknown' are logged and never vetoed."""
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

DIRS = {"N": (0, 1), "E": (1, 0), "S": (0, -1), "W": (-1, 0)}
Edge = Tuple[int, int, str]


def canon(a, b) -> Edge:
    (ax, ay), (bx, by) = a, b
    if ay == by:
        return (min(ax, bx), ay, "E")
    return (ax, max(ay, by), "S")


def ends(e: Edge):
    x, y, d = e
    return (x, y), ((x + 1, y) if d == "E" else (x, y - 1))


FUNCTIONAL = ("Active", "VisiblyActive")


def edges_from_raw(raw: Dict[str, Any], states=FUNCTIONAL) -> Set[Edge]:
    """undirected edges whose road state (on EITHER side of the shared border) is in `states`. Default = FUNCTIONAL (Active / VisiblyActive).
    S11 finding: a replaced house edge may be left 'Mothballed' (inactive but not None); the legacy `!= None` reading counts it as a road."""
    out = set()
    for t in raw["tiles"]:
        x, y = t["coord"][0], t["coord"][1]
        for d, st in t["road"].items():
            if st in states:
                dx, dy = DIRS[d]
                out.add(canon((x, y), (x + dx, y + dy)))
    return out


def edges_present(raw: Dict[str, Any]) -> Set[Edge]:
    """legacy reading: any state other than 'None' (Active, VisiblyActive, Mothballed, Pending)"""
    out = set()
    for t in raw["tiles"]:
        x, y = t["coord"][0], t["coord"][1]
        for d, st in t["road"].items():
            if st != "None":
                dx, dy = DIRS[d]
                out.add(canon((x, y), (x + dx, y + dy)))
    return out


class Ctx:
    """per-step static information read from the raw observation"""

    def __init__(self, raw):
        self.content: Dict[Tuple[int, int], str] = {}
        self.group: Dict[Tuple[int, int], Any] = {}
        self.water: Set[Tuple[int, int]] = set()
        for t in raw["tiles"]:
            k = (t["coord"][0], t["coord"][1])
            self.content[k] = t["contentType"]
            self.group[k] = t.get("groupIndex")
            if t.get("overWater"):
                self.water.add(k)
        self.houses = [k for k, c in self.content.items() if c == "House"]
        self.house_set = set(self.houses)
        self.struct = [k for k, c in self.content.items() if c in ("Destination", "Carpark")]


def connected_house_set(ctx: Ctx, edges: Set[Edge]) -> Set[Tuple[int, int]]:
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

    for k, c in ctx.content.items():
        if c != "None":
            parent[k] = k
    for e in edges:
        a, b = ends(e)
        parent.setdefault(a, a)
        parent.setdefault(b, b)
        union(a, b)
    for k in ctx.struct:
        for dx, dy in DIRS.values():
            n = (k[0] + dx, k[1] + dy)
            if n in parent and ctx.content.get(n) in ("Destination", "Carpark") and (ctx.content[k] == "Destination" or ctx.content[n] == "Destination"):
                union(k, n)
    dest_groups: Dict[Any, Set[Any]] = {}
    for k, c in ctx.content.items():
        if c == "Destination" and ctx.group.get(k) is not None and k in parent:
            dest_groups.setdefault(find(k), set()).add(ctx.group[k])
    return {h for h in ctx.houses if h in parent and ctx.group.get(h) is not None and ctx.group[h] in dest_groups.get(find(h), ())}


def edges_at(edges: Set[Edge], tile) -> Set[Edge]:
    return {e for e in edges if tile in ends(e)}


def predict_add(ctx: Ctx, edges: Set[Edge], e: Edge) -> Dict[str, Any]:
    """edges = FUNCTIONAL edges. Prediction is about the functional (Active) edge set only; whether the displaced driveway is deleted or left Mothballed is irrelevant to it."""
    a, b = ends(e)
    if ctx.content.get(a) in ("Destination", "Carpark") or ctx.content.get(b) in ("Destination", "Carpark"):
        return {"known": False, "reason": "destination_or_carpark_endpoint", "removed": set(), "added": set()}
    hs = [t for t in (a, b) if t in ctx.house_set]
    if not hs:
        return {"known": True, "reason": "plain_extension", "removed": set(), "added": ({e} - edges)}
    if len(hs) == 2:
        return {"known": False, "reason": "both_endpoints_are_houses", "removed": set(), "added": set()}
    h = hs[0]
    cur = edges_at(edges, h)
    if e in cur and len(cur) == 1:
        return {"known": True, "reason": "same_house_edge_noop", "removed": set(), "added": set()}
    if len(cur) == 0:
        return {"known": True, "reason": "house_without_active_edge", "removed": set(), "added": {e}}
    if len(cur) == 1:
        return {"known": True, "reason": "house_edge_replacement", "removed": set(cur), "added": {e}, "house": h}
    return {"known": False, "reason": "house_with_multiple_active_edges", "removed": set(), "added": set()}


def check_candidate(ctx: Ctx, edges: Set[Edge], base_connected: Set, e: Edge) -> Dict[str, Any]:
    pred = predict_add(ctx, edges, e)
    if not pred["known"]:
        return {"verdict": "unknown", "reason": pred["reason"], "lost": []}
    if not pred["removed"] or not base_connected:
        return {"verdict": "allow", "reason": pred["reason"], "lost": []}
    after = (edges - pred["removed"]) | pred["added"]
    lost = sorted(base_connected - connected_house_set(ctx, after))
    if lost:
        return {"verdict": "veto", "reason": "would_disconnect_" + str(len(lost)) + "_connected_house(s)", "lost": lost, "predicted_removed": sorted(pred["removed"])}
    return {"verdict": "allow", "reason": pred["reason"] + "_preserves_connectivity", "lost": []}


def guard_permitted(raw: Dict[str, Any], mask: np.ndarray, decode_action, upgrade_base: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    """returns (permitted mask = native mask minus vetoed ADDs, report). The native mask itself is not modified."""
    ctx = Ctx(raw)
    edges = edges_from_raw(raw)
    base = connected_house_set(ctx, edges)
    permitted = mask.copy()
    vetoed, unknown = [], []
    if base:
        for a in np.flatnonzero(mask):
            if not (0 < a < upgrade_base):
                continue
            k = decode_action(int(a))
            if k[0] != "ADD":
                continue
            r = check_candidate(ctx, edges, base, (k[1], k[2], k[3]))
            if r["verdict"] == "veto":
                permitted[a] = False
                vetoed.append((int(a), r["reason"], r["lost"]))
            elif r["verdict"] == "unknown":
                unknown.append((int(a), r["reason"]))
    return permitted, {"n_connected": len(base), "vetoed": vetoed, "unknown": unknown}
