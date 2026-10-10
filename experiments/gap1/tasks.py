"""Gap-1 task helpers. Everything here is EVALUATION / TASK-DEFINITION metadata: none of it is ever given to the
policy (the policy sees only the unchanged PolicyObservation vector)."""
import re
from typing import Any, Dict, List, Optional, Set, Tuple

W, H, XMIN, YMIN = 37, 21, -18, -10          # existing action-space bounds (python/mini_motorways_env.py)
LIVE = ("Pending", "Active", "VisiblyActive")  # road states that count as a built road (not None / Mothballed)
DIRS = {"N": (0, 1), "E": (1, 0), "S": (0, -1), "W": (-1, 0)}
OPP = {"N": "S", "S": "N", "E": "W", "W": "E"}
EDIT_RE = re.compile(r"ADD_ROAD\((-?\d+),(-?\d+),([NESW])\)")


def parse_edits(s: str) -> List[Tuple[int, int, str]]:
    return [(int(m.group(1)), int(m.group(2)), m.group(3)) for m in EDIT_RE.finditer(s or "")]


def action_index_add(x: int, y: int, d: str) -> int:
    """Existing action indexing: 1 + ((y-yMin)*37 + (x-xMin))*4 + dir*2 + kind (East=0/South=1, Add=0)."""
    return 1 + ((y - YMIN) * W + (x - XMIN)) * 4 + (0 if d == "E" else 1) * 2 + 0


def tiles_of(obs: Dict[str, Any]) -> Dict[Tuple[int, int], Dict[str, Any]]:
    return {(t["coord"][0], t["coord"][1]): t for t in obs["tiles"]}


def edge_live(tiles, x: int, y: int, d: str) -> bool:
    t = tiles.get((x, y))
    return bool(t) and t["road"][d] in LIVE


def components(obs: Dict[str, Any]):
    """Union-find over live road edges; destination footprints join through adjacent Carpark tiles."""
    tiles = tiles_of(obs)
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

    for k, t in tiles.items():
        if any(v in LIVE for v in t["road"].values()) or t["contentType"] != "None":
            parent[k] = k
    for k, t in tiles.items():
        if k not in parent:
            continue
        for d, (dx, dy) in DIRS.items():
            if t["road"][d] in LIVE:
                n = (k[0] + dx, k[1] + dy)
                if n in parent:
                    union(k, n)
    for k, t in tiles.items():
        if k not in parent or t["contentType"] not in ("Destination", "Carpark"):
            continue
        for dx, dy in DIRS.values():
            n = (k[0] + dx, k[1] + dy)
            if n in parent and tiles[n]["contentType"] in ("Destination", "Carpark") and (
                    t["contentType"] == "Destination" or tiles[n]["contentType"] == "Destination"):
                union(k, n)
    return tiles, parent, find


def house_connected(obs: Dict[str, Any], house: Tuple[int, int], dest_footprint: List[Tuple[int, int]]) -> bool:
    """True iff the target house's road component contains the target destination (any footprint tile)."""
    tiles, parent, find = components(obs)
    if tuple(house) not in parent:
        return False
    r = find(tuple(house))
    return any(tuple(d) in parent and find(tuple(d)) == r for d in dest_footprint)


def required_edges_intact(obs: Dict[str, Any], built_edits: List[Tuple[int, int, str]]) -> bool:
    tiles = tiles_of(obs)
    return all(edge_live(tiles, x, y, d) for x, y, d in built_edits)


def layout_signature(info: Dict[str, Any], include_gap: bool) -> str:
    base = (tuple(info["house"]), tuple(map(tuple, info["destinationFootprint"])), info["fullPathEdits"])
    return repr(base + ((info["omittedEdits"],) if include_gap else ()))
