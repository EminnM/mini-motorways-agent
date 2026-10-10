"""S13b scripted reference controller. PRIVILEGED (reads env internals: house groups, destination groups, stub edges) - used ONLY to demonstrate solvability and in tests, never as a learned policy or an inference guard.
Policy: pick the oldest unconnected house that has a same-group destination; Dijkstra over tiles from the house to the outer end of a stub edge of a same-group destination (existing Active edges cost 0, new plain edges cost 1,
house-edge swaps cost 0, bridging a river tile allowed by the native rule 'not both endpoints water'); then build the first missing edge on the path (starting at the house). Waits when concrete is insufficient."""
import heapq
from typing import Optional

from . import layout as L
from .s13_env import DIRS, canon, ends


def _plan(env, house):
    g = env.houses[house]["group"]
    goals = set()
    for d in env.dests:
        if d["group"] == g:
            for e in env.free:
                a, b = ends(e)
                for cp in d["carparks"]:
                    if cp in (a, b):
                        goals.add(b if a == cp else a)
    if not goals:
        return None
    x0, x1, y0, y1 = env._window()
    blocked = {p for p, c in env.content.items() if c in ("Destination", "Carpark")}
    dist = {house: 0}
    prev = {}
    pq = [(0, house)]
    while pq:
        dd, u = heapq.heappop(pq)
        if dd > dist.get(u, 1e9):
            continue
        if u in goals and u != house:
            path = [u]
            while path[-1] != house:
                path.append(prev[path[-1]])
            return path[::-1]
        if u != house and env.content.get(u) == "House":
            continue
        for dx, dy in DIRS.values():
            v = (u[0] + dx, u[1] + dy)
            if not (x0 <= v[0] <= x1 and y0 <= v[1] <= y1) or v in blocked or (v in env.water and u in env.water):
                continue
            if env.content.get(v) == "House":
                continue
            e = canon(u, v)
            cost = 0 if (env.edges.get(e) == "A" or u == house) else 1
            nd = dd + cost
            if nd < dist.get(v, 1e9):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    return None


def act(env) -> int:
    conn = env.connected_houses()
    todo = sorted((h for h in env.houses if h not in conn), key=lambda h: env.houses[h]["t"])
    for h in todo:
        path = _plan(env, h)
        if path is None:
            continue
        for u, v in zip(path[:-1], path[1:]):
            e = canon(u, v)
            if env.edges.get(e) == "A":
                continue
            a = L.action_index("ADD", *e)
            if env.action_masks()[a]:
                return a
            return 0                                         # cannot afford: wait for a top-up
        # whole path already active but house still unconnected (should not happen): fall through
    return 0
