"""INDEPENDENT S7 oracles. They share no code with the generator's `remaining_distance`:
  * `nx_min_edits`      : networkx shortest path on a graph built ONLY from the environment's live legal-ADD mask + the existing road edges decoded from the
                          policy observation (cost 1 per legal ADD edge, 0 per existing edge). Scalable to any length.
  * `obs_connected`     : connectivity decided from the observation vector alone (road channels) + the scene's carpark tile.
  * `nx_components_connected` : same decision through networkx connected components.
  * `brute_force_depth` : exhaustive breadth-first search over ACTION SEQUENCES through the real env.step (small tasks only): the true minimum number of
                          actions to reach reward +1."""
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from . import layout as L


def road_edges_from_obs(obs: np.ndarray) -> List[Tuple[int, int, str]]:
    tiles = obs[: L.N_TILES * L.NUM_TILE_FEATURES].reshape(L.H, L.W, L.NUM_TILE_FEATURES)
    out = []
    for yi in range(L.H):
        for xi in range(L.W):
            for d in ("E", "S"):
                if tiles[yi, xi, L.F_ROAD[d]] > 0:
                    out.append((L.XMIN + xi, L.YMIN + yi, d))
    return out


def house_from_obs(obs: np.ndarray) -> Tuple[int, int]:
    tiles = obs[: L.N_TILES * L.NUM_TILE_FEATURES].reshape(L.H, L.W, L.NUM_TILE_FEATURES)
    ys, xs = np.where(np.isclose(tiles[:, :, L.F_CONTENT], L.CONTENT_HOUSE))
    assert len(ys) == 1, "expected exactly one house tile in the observation"
    return (L.XMIN + int(xs[0]), L.YMIN + int(ys[0]))


def _graph(obs, mask):
    G = nx.Graph()
    for a in np.flatnonzero(mask):
        if 0 < a < L.UPGRADE_BASE:
            k, x, y, d = L.decode_action(int(a))
            if k == "ADD":
                G.add_edge((x, y), L.neighbor(x, y, d), w=1)
    for (x, y, d) in road_edges_from_obs(obs):
        G.add_edge((x, y), L.neighbor(x, y, d), w=0)
    return G


def nx_min_edits(obs: np.ndarray, mask: np.ndarray, carpark_tile) -> Optional[int]:
    """minimum number of further ADD actions (house tile -> carpark tile), or None if unreachable; uses only obs + mask + the carpark tile."""
    G = _graph(obs, mask)
    h, c = house_from_obs(obs), tuple(carpark_tile)
    if h not in G or c not in G:
        return None
    try:
        return int(nx.shortest_path_length(G, h, c, weight="w"))
    except nx.NetworkXNoPath:
        return None


def nx_route(obs, mask, carpark_tile, forbid_edges=()):
    """one shortest route as a list of (x,y,d) edges that still need an ADD, optionally avoiding some edges (for alternative-route tests)"""
    G = _graph(obs, mask)
    for (x, y, d) in forbid_edges:
        a, b = (x, y), L.neighbor(x, y, d)
        if G.has_edge(a, b) and G[a][b]["w"] == 1:
            G.remove_edge(a, b)
    h, c = house_from_obs(obs), tuple(carpark_tile)
    path = nx.shortest_path(G, h, c, weight="w")
    need = []
    for a, b in zip(path[:-1], path[1:]):
        if G[a][b]["w"] == 1:
            (x1, y1), (x2, y2) = a, b
            need.append((min(x1, x2), y1, "E") if y1 == y2 else (x1, max(y1, y2), "S"))
    return need


def count_shortest_routes(obs, mask, carpark_tile, cap=200) -> int:
    G = _graph(obs, mask)
    h, c = house_from_obs(obs), tuple(carpark_tile)
    best = nx.shortest_path_length(G, h, c, weight="w")
    n = 0
    for p in nx.all_shortest_paths(G, h, c, weight="w"):
        n += 1
        if n >= cap:
            break
    return n


def obs_connected(obs: np.ndarray, carpark_tile) -> bool:
    adj: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    for (x, y, d) in road_edges_from_obs(obs):
        a, b = (x, y), L.neighbor(x, y, d)
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    h, c = house_from_obs(obs), tuple(carpark_tile)
    seen, st = {h}, [h]
    while st:
        u = st.pop()
        if u == c:
            return True
        for v in adj.get(u, ()):
            if v not in seen:
                seen.add(v)
                st.append(v)
    return False


def nx_components_connected(obs: np.ndarray, carpark_tile) -> bool:
    G = nx.Graph()
    for (x, y, d) in road_edges_from_obs(obs):
        G.add_edge((x, y), L.neighbor(x, y, d))
    h, c = house_from_obs(obs), tuple(carpark_tile)
    return h in G and c in G and nx.has_path(G, h, c)


def exists_win_within(env, depth: int, counter: Optional[List[int]] = None) -> bool:
    """exhaustive depth-limited search through the real env.step (all legal actions, WAIT/REMOVE included): True iff some action sequence of length
    <= depth reaches reward +1 from the CURRENT env state."""
    snap = env.snapshot()
    legal = [int(a) for a in np.flatnonzero(env.action_masks())]
    for a in legal:
        env.restore(snap)
        _, r, te, tr, _ = env.step(a)
        if counter is not None:
            counter[0] += 1
        if r > 0:
            env.restore(snap)
            return True
        if depth > 1 and not (te or tr) and exists_win_within(env, depth - 1, counter):
            env.restore(snap)
            return True
    env.restore(snap)
    return False


def brute_force_min_actions(env, task_index: int, max_depth: int):
    """true minimum number of actions to +1 by iterative deepening; (d or None, step calls)"""
    env.reset(options={"task_index": task_index})
    cnt = [0]
    for d in range(1, max_depth + 1):
        if exists_win_within(env, d, cnt):
            return d, cnt[0]
    return None, cnt[0]
