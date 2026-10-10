"""S18-H candidate generator. Produces typed Candidate objects from a WorldModel (visible state only).

Candidate kinds:
  CONNECT  - route a disconnected house to a component containing a destination of its colour group. Dijkstra over tiles with
             lexicographic cost (concrete, length): existing Active road edge 0 concrete, house edge 0 concrete (free), new plain edge
             1 concrete. k alternatives by penalising the previous route's new edges and re-searching (edge-penalty re-search).
  REMOVE   - remove a redundant dead-end plain road edge (refund 1) that no functional connection relies on.
  WAIT     - do nothing.
  MOTORWAY - typed candidate only; feasibility='unsupported' (no executable action in Discrete(3111); see MECHANICS.md).
Every candidate carries: targets, ordered edge list (ADDs in build order), concrete cost, connectivity delta, would_destroy_functional_connection
(computed with the S11-style functional model in world_model), prerequisites, confidence."""
import heapq
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from world_model import Edge, Tile, WorldModel, canon, ends, house_edges, neighbours


@dataclass
class Candidate:
    kind: str                                  # CONNECT | REMOVE | WAIT | MOTORWAY
    targets: Dict[str, Any] = field(default_factory=dict)
    adds: List[Edge] = field(default_factory=list)       # build order (house end first)
    removes: List[Edge] = field(default_factory=list)
    path: List[Tile] = field(default_factory=list)
    concrete_cost: int = 0                     # net concrete (positive = spend, negative = refund)
    route_len: int = 0                         # tiles on the path minus one (travel-cost PROXY: no traffic model exists)
    reused_edges: int = 0
    connectivity_delta: int = 0                # houses connected after - before (predicted)
    would_destroy_functional_connection: bool = False
    lost: List[Tile] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    feasibility: str = "feasible"              # feasible | needs_concrete | unsupported
    confidence: float = 1.0
    alt_index: int = 0
    score: float = 0.0
    score_terms: Dict[str, float] = field(default_factory=dict)

    def key(self):
        return (self.kind, tuple(self.targets.get("house", ())), tuple(self.adds), tuple(self.removes))

    def describe(self) -> str:
        if self.kind == "CONNECT":
            return f"CONNECT house={self.targets['house']} group={self.targets['group']} cost={self.concrete_cost} len={self.route_len} reuse={self.reused_edges} alt={self.alt_index} feas={self.feasibility}"
        if self.kind == "REMOVE":
            return f"REMOVE edge={self.removes[0]} refund=1"
        return self.kind


ROUTE_LAMBDA: Optional[float] = None   # S35 geometry option: None = lexicographic (concrete, length) as in S18-H; a number = scalar cost length + LAMBDA * new concrete
                                        # (LAMBDA is in tiles of driving per road tile; 4.0 matches the scorer's own exchange rate W_COST / W_LEN = 0.6 / 0.15)


def dijkstra_route(wm: WorldModel, house: Tile, goals: Set[Tile], penalty: Dict[Edge, float], banned: Set[Edge]) -> Optional[List[Tile]]:
    """lexicographic (concrete, length + penalties) shortest path from `house` to any goal tile; never passes through another house or a structure.
    With ROUTE_LAMBDA set, the primary key becomes length + LAMBDA * concrete (secondary: concrete), so a shortcut costing a few tiles can beat a long free detour."""
    lam = ROUTE_LAMBDA
    dist: Dict[Tile, Tuple[float, float]] = {house: (0.0, 0.0)}
    prev: Dict[Tile, Tile] = {}
    pq = [((0.0, 0.0), house)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, (1e9, 1e9)):
            continue
        if u in goals and u != house:
            path = [u]
            while path[-1] != house:
                path.append(prev[path[-1]])
            return path[::-1]
        if u != house and u in wm.house_set:
            continue
        for v in neighbours(u):
            if v in wm.house_set or v in wm.blocked:
                continue
            e = canon(u, v)
            if e in banned:
                continue
            if wm.edges.get(e) == "A":
                c = (0.0, 1.0)
            elif u == house and wm.structurally_buildable(e):
                c = (0.0, 1.0 + (0.5 if wm.edges.get(e) != "M" and wm.edges.get(e) is None and any(wm.edges.get(x) == "A" for x in house_edges(house)) else 0.0))
            elif wm.structurally_buildable(e):
                c = (1.0, 1.0 + penalty.get(e, 0.0))
            else:
                continue
            if lam is not None:                        # scalar objective: primary = length + lam * concrete (penalties count as length), secondary = concrete
                c = (c[1] + lam * c[0], c[0])
            nd = (d[0] + c[0], d[1] + c[1])
            if nd < dist.get(v, (1e9, 1e9)):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    return None


def route_to_candidate(wm: WorldModel, house: Tile, path: List[Tile], alt: int) -> Candidate:
    adds, reused = [], 0
    for u, v in zip(path[:-1], path[1:]):
        e = canon(u, v)
        if wm.edges.get(e) == "A":
            reused += 1
        else:
            adds.append(e)
    cost = sum(0 if wm.is_house_edge(e) else 1 for e in adds)
    lost = sorted(wm.lost_connections(adds, []))
    conn_after = wm.connected_after(adds, [])
    g = wm.houses[house]["group"]
    c = Candidate(kind="CONNECT", targets={"house": house, "group": g, "goal": path[-1]}, adds=adds, path=path, concrete_cost=cost, route_len=len(path) - 1, reused_edges=reused,
                  connectivity_delta=len(conn_after) - len(wm.connected), would_destroy_functional_connection=bool(lost), lost=lost, alt_index=alt,
                  feasibility="feasible" if cost <= wm.concrete else "needs_concrete", confidence=1.0 if cost <= wm.concrete else 0.8)
    if cost > wm.concrete:
        c.prerequisites.append(f"concrete {cost - wm.concrete} more (next top-up known by clock)")
    if house not in conn_after:
        c.confidence = 0.0
        c.prerequisites.append("predicted route does not connect the house (model mismatch)")
    return c


def connect_candidates(wm: WorldModel, k_alternatives: int = 2, banned: Set[Edge] = frozenset(), houses: Optional[List[Tile]] = None) -> List[Candidate]:
    out: List[Candidate] = []
    for h in (houses if houses is not None else sorted(wm.houses, key=lambda p: (wm.houses[p]["spawn_t"], p))):
        if h in wm.connected:
            continue
        goals = wm.goal_tiles_for_group(wm.houses[h]["group"])
        if not goals:
            continue
        penalty: Dict[Edge, float] = {}
        seen = set()
        for alt in range(1 + k_alternatives):
            path = dijkstra_route(wm, h, goals, penalty, set(banned))
            if path is None:
                break
            c = route_to_candidate(wm, h, path, alt)
            if tuple(c.adds) in seen:
                break
            seen.add(tuple(c.adds))
            out.append(c)
            for e in c.adds:
                penalty[e] = penalty.get(e, 0.0) + 3.0
    return out


def dest_candidates(wm: WorldModel, k_alternatives: int = 2, banned: Set[Edge] = frozenset()) -> List[Candidate]:
    """S35 fix (opt-in via Weights.serve_unreached_dests): CONNECT candidates that bring a house of colour g to a destination of colour g that no house of g can
    currently reach (wm.unreached_dests). Route = Dijkstra from the house to the road tiles of the destination's own component (its spawn stub), same cost model
    and alternatives as connect_candidates. targets carries 'dest' so the executor completes the plan when house and destination share a component, not when the
    house is 'connected' (it already is). connectivity_delta = 1 per unreached destination served (the houses of that colour all gain access to it)."""
    out: List[Candidate] = []
    for d in wm.unreached_dests():
        g = d["group"]
        goals = wm.tiles_of_component(wm.component(d["tiles"][0]))
        if not goals:
            continue
        for h in sorted(wm.houses, key=lambda p: (wm.houses[p]["spawn_t"], p)):
            if wm.houses[h]["group"] != g:
                continue
            penalty: Dict[Edge, float] = {}
            seen = set()
            for alt in range(1 + k_alternatives):
                path = dijkstra_route(wm, h, goals, penalty, set(banned))
                if path is None:
                    break
                c = route_to_candidate(wm, h, path, alt)
                if tuple(c.adds) in seen:
                    break
                seen.add(tuple(c.adds))
                c.targets["dest"] = d["tiles"][0]
                c.connectivity_delta = 1
                c.confidence = 1.0 if c.concrete_cost <= wm.concrete else 0.8
                c.prerequisites = [p for p in c.prerequisites if "does not connect" not in p]
                if wm.lost_connections(c.adds, []):
                    c.would_destroy_functional_connection = True
                out.append(c)
                for e in c.adds:
                    penalty[e] = penalty.get(e, 0.0) + 3.0
    return out


def _bfs_drive(wm: WorldModel, start: Tile, goals: Set[Tile]) -> Optional[int]:
    """driving distance over ACTIVE edges from a house to any goal tile (static shortest-path proxy for the native router, S25)."""
    from collections import deque
    seen = {start: 0}
    q = deque([start])
    while q:
        u = q.popleft()
        if u in goals:
            return seen[u]
        for v in neighbours(u):
            if wm.edges.get(canon(u, v)) == "A" and v not in seen and v not in wm.house_set:
                seen[v] = seen[u] + 1
                q.append(v)
    return None


def bypass_candidates(wm: WorldModel, ratio: float = 2.0, max_cost: int = 10, min_demand: int = 3) -> List[Candidate]:
    """Traffic-aware gate: a colour is considered only when its destinations currently hold >= min_demand pins in total or one of them is overcrowded (visible demand = cars are not
    keeping up); colours under no pressure get no bypass, however long their detour (tiles, junctions and spawn perturbation are not spent for nothing)."""
    """S37 (opt-in via Weights.bypass): shortcut candidates for CONNECTED houses whose driving distance to the nearest road tile adjacent to a destination of their colour is
    >= ratio x the shortest LEGAL buildable path, when that path costs <= max_cost new tiles, is affordable now and destroys no functional connection.
    targets carries 'bypass': True, 'drive' (current), 'shortest'; connectivity_delta = 0; the executor completes the plan when every edge of the route is Active."""
    out: List[Candidate] = []
    global ROUTE_LAMBDA
    saved = ROUTE_LAMBDA
    for h in sorted(wm.houses, key=lambda p: (wm.houses[p]["spawn_t"], p)):
        if h not in wm.connected:
            continue
        g = wm.houses[h]["group"]
        pins, over = wm.demand_of_group(g)
        if pins < min_demand and not over:
            continue
        goals: Set[Tile] = set()
        for d in wm.dests:
            if d["group"] != g:
                continue
            for p in d["tiles"] + d["carparks"]:
                for n in neighbours(p):
                    if n not in wm.blocked and n not in wm.house_set and n in wm.uf.p:
                        goals.add(n)
        if not goals:
            continue
        drive = _bfs_drive(wm, h, goals)
        if drive is None:
            continue
        ROUTE_LAMBDA = 0.0                                    # pure length: the geometric shortest legal path
        path = dijkstra_route(wm, h, goals, {}, set())
        ROUTE_LAMBDA = saved
        if path is None:
            continue
        short = len(path) - 1
        if short == 0 or drive < ratio * short:
            continue
        c = route_to_candidate(wm, h, path, alt=7)
        if c.concrete_cost == 0 or c.concrete_cost > max_cost or c.concrete_cost > wm.concrete or c.would_destroy_functional_connection:
            continue
        c.targets.update({"bypass": True, "drive": drive, "shortest": short})
        c.connectivity_delta = 0
        c.confidence = 1.0
        c.prerequisites = [p for p in c.prerequisites if "does not connect" not in p]
        out.append(c)
    ROUTE_LAMBDA = saved
    return out


def remove_candidates(wm: WorldModel, protected: Set[Edge] = frozenset()) -> List[Candidate]:
    """dead-end plain edges (REMOVE legal under the mask) whose removal keeps every currently connected house connected."""
    out = []
    for e in sorted(wm.edges):
        if e in wm.stub or wm.is_house_edge(e) or e in protected:
            continue
        if not wm.legal_now("REMOVE", e):
            continue
        lost = sorted(wm.lost_connections([], [e]))
        c = Candidate(kind="REMOVE", targets={"edge": e}, removes=[e], concrete_cost=-1, connectivity_delta=-len(lost), would_destroy_functional_connection=bool(lost), lost=lost)
        if not lost:
            out.append(c)
    return out


def motorway_candidates(wm: WorldModel) -> List[Candidate]:
    """typed only: a straight motorway between the two most distant same-group house/destination pairs would be the natural target; there is NO
    executable action for it in the integration (verified: ActionKind = {Wait, Add, Remove, ChooseUpgrade}), so feasibility is 'unsupported'."""
    out = []
    for h in wm.houses:
        if h in wm.connected:
            continue
        g = wm.houses[h]["group"]
        for d in wm.dests:
            if d["group"] == g:
                c = Candidate(kind="MOTORWAY", targets={"house": h, "dest": d["tiles"][0], "group": g}, feasibility="unsupported", confidence=0.0,
                              prerequisites=["motorway upgrade available (globals.motorwaysAvailable)", "an executable ADD_MOTORWAY action (not exposed by the Python integration)"])
                out.append(c)
                break
    return out


def wait_candidate() -> Candidate:
    return Candidate(kind="WAIT", confidence=1.0)


def generate(wm: WorldModel, k_alternatives: int = 2, banned: Set[Edge] = frozenset(), protected: Set[Edge] = frozenset(), include_motorway: bool = True, serve_unreached_dests: bool = False,
             bypass: bool = False, bypass_ratio: float = 2.0, bypass_max_cost: int = 10, bypass_min_demand: int = 3) -> List[Candidate]:
    cands = [wait_candidate()]
    cands += connect_candidates(wm, k_alternatives, banned)
    if serve_unreached_dests:
        cands += dest_candidates(wm, k_alternatives, banned)
    if bypass:
        cands += bypass_candidates(wm, bypass_ratio, bypass_max_cost, bypass_min_demand)
    cands += remove_candidates(wm, protected)
    if include_motorway:
        cands += motorway_candidates(wm)
    return cands
