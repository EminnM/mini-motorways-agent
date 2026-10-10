"""Procedural Gap-N task generation (pure Python, milliseconds per task).

A task is a tiny city on the native 37x21 grid: one house (with a driveway edge to its start tile), one matching 2x2
destination (with a carpark tile whose stub edge reaches a goal tile), and a road path start -> goal of which `n_gaps`
edges are LEFT OUT. Solving the task = adding exactly the left-out edges (any order).

Uniqueness: the minimum number of new edges needed to connect house and destination equals n_gaps AND there is exactly
one shortest edge set (counted on the contracted component graph), so no alternative / accidental solution exists."""
import hashlib
import json
import random
from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple

from . import layout as L

Tile = Tuple[int, int]


class TaskGenError(RuntimeError):
    pass


def _canon(a: Tile, b: Tile):
    (ax, ay), (bx, by) = a, b
    if bx == ax + 1 and by == ay:
        return (ax, ay, "E")
    if bx == ax - 1 and by == ay:
        return (bx, by, "E")
    if bx == ax and by == ay - 1:
        return (ax, ay, "S")
    if bx == ax and by == ay + 1:
        return (bx, by, "S")
    raise ValueError("tiles not adjacent")


def _neighbors(t: Tile):
    for d in "NESW":
        yield L.neighbor(t[0], t[1], d)


class _UF:
    def __init__(self):
        self.p = {}

    def find(self, a):
        self.p.setdefault(a, a)
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def min_completion(existing_edges: Set[Tuple[int, int, str]], occupied: Set[Tile], window, house: Tile, carpark: Tile, footprint: List[Tile]):
    """(min number of new edges to connect house to destination, number of distinct shortest edge sequences)."""
    wx0, wy0, wx1, wy1 = window
    uf = _UF()
    for (x, y, d) in existing_edges:
        uf.union((x, y), L.neighbor(x, y, d))
    for f in footprint:
        uf.union(f, carpark)
    src, dst = uf.find(house), uf.find(carpark)
    if src == dst:
        return 0, 1
    free = lambda t: wx0 <= t[0] <= wx1 and wy0 <= t[1] <= wy1 and t not in occupied
    adj: Dict[Any, List[Any]] = {}
    for x in range(wx0, wx1 + 1):
        for y in range(wy0, wy1 + 1):
            u = (x, y)
            if not free(u):
                continue
            for d in ("E", "S"):
                v = L.neighbor(x, y, d)
                if not free(v) or (x, y, d) in existing_edges:
                    continue
                cu, cv = uf.find(u), uf.find(v)
                if cu == cv:
                    continue
                adj.setdefault(cu, []).append(cv)
                adj.setdefault(cv, []).append(cu)
    dist = {src: 0}
    cnt = {src: 1}
    q = deque([src])
    while q:
        u = q.popleft()
        for v in adj.get(u, []):
            if v not in dist:
                dist[v] = dist[u] + 1
                cnt[v] = cnt[u]
                q.append(v)
            elif dist[v] == dist[u] + 1:
                cnt[v] += cnt[u]
    return dist.get(dst, None), cnt.get(dst, 0)


def _walk(rng: random.Random, start: Tile, goal: Tile, blocked: Set[Tile], window, min_len: int, max_len: int):
    wx0, wy0, wx1, wy1 = window
    nodes = 0
    path = [start]
    seen = {start}

    def dfs(cur):
        nonlocal nodes
        nodes += 1
        if nodes > 4000:
            return False
        if cur == goal:
            return len(path) - 1 >= min_len
        if len(path) - 1 >= max_len:
            return False
        cand = []
        for nb in _neighbors(cur):
            if nb in seen or nb in blocked:
                continue
            if not (wx0 <= nb[0] <= wx1 and wy0 <= nb[1] <= wy1):
                continue
            cand.append((abs(nb[0] - goal[0]) + abs(nb[1] - goal[1]) + rng.uniform(0, 2.2), nb))
        cand.sort()
        for _, nb in cand:
            seen.add(nb)
            path.append(nb)
            if dfs(nb):
                return True
            path.pop()
            seen.discard(nb)
        return False

    return list(path) if dfs(start) else None


def _try(rng: random.Random, n_gaps: int, clear_footprint: bool = True) -> Optional[Dict[str, Any]]:
    ww, wh = rng.randint(11, 15), rng.randint(8, 11)
    wx0 = rng.randint(L.XMIN, L.XMIN + L.W - ww)
    wy0 = rng.randint(L.YMIN, L.YMIN + L.H - wh)
    window = (wx0, wy0, wx0 + ww - 1, wy0 + wh - 1)
    wx1, wy1 = window[2], window[3]
    dx = rng.randint(wx0 + 2, wx1 - 3)
    dy = rng.randint(wy0 + 2, wy1 - 3)
    footprint = [(dx, dy), (dx + 1, dy), (dx, dy + 1), (dx + 1, dy + 1)]
    side = rng.choice("NESW")
    r = rng.randint(0, 1)
    carpark = {"W": (dx - 1, dy + r), "E": (dx + 2, dy + r), "S": (dx + r, dy - 1), "N": (dx + r, dy + 2)}[side]
    stub_dir = rng.choice([d for d in "NESW" if d != {"W": "E", "E": "W", "S": "N", "N": "S"}[side]])
    goal = L.neighbor(carpark[0], carpark[1], stub_dir)
    house = (rng.randint(wx0, wx1), rng.randint(wy0, wy1))
    drive = rng.choice("NESW")
    start = L.neighbor(house[0], house[1], drive)
    occupied = set(footprint) | {carpark, house}
    for t in (goal, start):
        if t in occupied or not (wx0 <= t[0] <= wx1 and wy0 <= t[1] <= wy1):
            return None
    if len(occupied) != 6 or start == goal:
        return None
    blocked = set(occupied)
    if clear_footprint:
        # The native observation encodes the carpark tile as content 0 (Carpark is not in _CONTENT_TYPE_ORDINAL), so a road
        # that merely passes beside the destination is indistinguishable from the real entry. Keep every other road tile
        # off the destination's neighbourhood so the carpark is the ONLY road tile touching the footprint.
        near = {nb for f in footprint for nb in _neighbors(f)} - set(footprint) - {carpark}
        if goal in near or start in near:
            return None
        blocked |= near
    dist = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
    min_len = max(n_gaps + 2, dist)
    path = _walk(rng, start, goal, blocked, window, min_len, min(14, min_len + 5))
    if path is None:
        return None
    n_edges = len(path) - 1
    if n_edges < n_gaps + 2:
        return None
    gap_idx = sorted(rng.sample(range(n_edges), n_gaps))
    path_edges = [_canon(path[i], path[i + 1]) for i in range(n_edges)]
    solution = [path_edges[i] for i in gap_idx]
    stubs = {_canon(house, start), _canon(carpark, goal)}
    built = [e for i, e in enumerate(path_edges) if i not in gap_idx]
    existing = set(built) | stubs
    dmin, count = min_completion(existing, occupied, window, house, carpark, footprint)
    if dmin != n_gaps or count != 1:
        return None
    base = {"window": list(window), "house": list(house), "driveway": drive, "start": list(start), "footprint": [list(f) for f in footprint],
            "carpark": list(carpark), "stub_dir": stub_dir, "goal": list(goal), "path": [list(p) for p in path]}
    base_sig = hashlib.sha256(json.dumps(base, sort_keys=True).encode()).hexdigest()[:16]
    task = dict(base)
    task.update({"n_gaps": n_gaps, "gap_indices": gap_idx, "built_edges": [list(e) for e in built], "stub_edges": [list(e) for e in sorted(stubs)],
                 "solution_edges": [list(e) for e in solution], "path_length_edges": n_edges, "base_signature": base_sig,
                 "signature": hashlib.sha256((base_sig + str(gap_idx)).encode()).hexdigest()[:16],
                 "solution_action_indices": [L.action_index("ADD", e[0], e[1], e[2]) for e in solution]})
    return task


def make_task(rng: random.Random, n_gaps: int = 1, attempts: int = 400, clear_footprint: bool = True) -> Dict[str, Any]:
    for _ in range(attempts):
        t = _try(rng, n_gaps, clear_footprint)
        if t is not None:
            return t
    raise TaskGenError(f"no valid Gap-{n_gaps} task after {attempts} attempts")


def make_split_sets(n_gaps: int, n_train: int, n_val: int, n_test: int, seed: int = 0, clear_footprint: bool = True):
    """Disjoint splits: no base layout (house, destination, path) appears in two splits; within a split a base layout
    may recur with different gap choices (as in the native Gap-1 manifests)."""
    rng = random.Random(seed * 1000 + n_gaps)
    sets = {"test": [], "validation": [], "train": []}
    need = {"test": n_test, "validation": n_val, "train": n_train}
    owner: Dict[str, str] = {}
    sigs = set()
    guard = 0
    while any(len(sets[s]) < need[s] for s in sets):
        guard += 1
        if guard > 200 * (n_train + n_val + n_test):
            raise TaskGenError("could not fill splits")
        t = make_task(rng, n_gaps, clear_footprint=clear_footprint)
        if t["signature"] in sigs:
            continue
        b = t["base_signature"]
        if b in owner:
            s = owner[b]
        else:
            open_splits = [s for s in ("test", "validation", "train") if len(sets[s]) < need[s]]
            # fill held-out splits first, then train
            s = open_splits[0]
            owner[b] = s
        if len(sets[s]) >= need[s]:
            continue
        sigs.add(t["signature"])
        t["split"] = s
        t["task_id"] = f"syn_gap{n_gaps}_{s}_{len(sets[s]) + 1:06d}"
        sets[s].append(t)
    return sets
