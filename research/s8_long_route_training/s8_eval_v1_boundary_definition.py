"""S8 evaluation on the S7 long-route family: S7 metrics plus (a) ADD budgets, (b) churn, (c) the synthetic-only NATIVE-LEGALITY diagnostic.

Native-legality classes (evidence: 24 native start-state captures, research/s8_long_route_training/legality_evidence.json; S7-legal ADDs that were native-legal):
  interior_dry     : both endpoints dry and not on the build-window boundary ring   -> 2192/2192 native-legal  ("supported legal")
  window_boundary  : an endpoint on the window ring (a window tile with a 4-neighbour outside it) -> 83.1% native-legal ("unknown")
  water_endpoint   : an endpoint over water                                         -> 71.0% native-legal ("unknown")
Evidence covers start states only; legality of arbitrary later synthetic states is NOT established, so anything outside interior_dry is labelled unknown, never 'illegal'.
Evaluation is read-only for training: global torch RNG is saved/restored by the caller."""
import collections
import math
import statistics
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np

from . import layout as L
from . import s7


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0.0, c - h), min(1.0, c + h)]


def edge_class(world: s7.World, scene: Dict[str, Any], e: Tuple[int, int, str]) -> str:
    a, b = s7.ends(e)
    if a in world.water or b in world.water:
        return "water_endpoint"
    for t in (a, b):
        if any(L.neighbor(t[0], t[1], d) not in scene["win"] for d in s7.DIRS4):
            return "window_boundary"
    return "interior_dry"


def _route(env) -> List[Tuple[int, int, str]]:
    """edges of one house -> carpark path in the final road graph (BFS), for the legality diagnostic"""
    adj = {}
    for e in env.edges:
        a, b = s7.ends(e)
        adj.setdefault(a, []).append((b, e))
        adj.setdefault(b, []).append((a, e))
    start, goal = env.house, env.scene["ctile"]
    prev = {start: None}
    q = collections.deque([start])
    while q:
        u = q.popleft()
        if u == goal:
            break
        for v, e in adj.get(u, ()):
            if v not in prev:
                prev[v] = (u, e)
                q.append(v)
    path = []
    u = goal
    while u in prev and prev[u] is not None:
        u, e = prev[u]
        path.append(e)
    return path


def evaluate(policy: str, tasks: List[Dict[str, Any]], model=None, deterministic: bool = True, seed: int = 0) -> List[Dict[str, Any]]:
    import torch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    env = s7.LongRouteEnv(tasks, sampling="sequential")
    world = env.world
    rows = []
    for i, t in enumerate(tasks):
        obs, _ = env.reset(options={"task_index": i})
        k = t["k_min"]
        scene = env.scene
        initial = set(env.edges)
        r_prev = env.remaining()
        max_prog = 0.0
        n_add = n_rm = n_wait = n_irrel = n_repl = 0
        first_add_useful = None
        first_useful_step = None
        steps = 0
        success = False
        hset = set(s7.house_edges(env.house))
        cls = collections.Counter()
        done = False
        while not done:
            mask = env.action_masks()
            if policy == "wait":
                a = 0
            elif policy == "random":
                a = int(rng.choice(np.flatnonzero(mask)))
            else:
                a, _ = model.predict(obs, action_masks=mask, deterministic=deterministic)
                a = int(a)
            dec = L.decode_action(a)
            obs, r, te, tr, info = env.step(a)
            steps += 1
            done = te or tr
            r_now = env.remaining()
            if dec[0] == "ADD":
                n_add += 1
                e = (dec[1], dec[2], dec[3])
                cls[edge_class(world, scene, e)] += 1
                if e in hset:
                    n_repl += 1
                useful = r_now < r_prev
                if first_add_useful is None:
                    first_add_useful = bool(useful)
                if useful and first_useful_step is None:
                    first_useful_step = steps
                if not useful:
                    n_irrel += 1
            elif dec[0] == "REMOVE":
                n_rm += 1
            elif dec[0] == "WAIT":
                n_wait += 1
            max_prog = max(max_prog, (k - r_now) / k)
            r_prev = r_now
            success = bool(te)
        route_classes = None
        if success:
            route = [e for e in _route(env) if e not in initial]
            route_classes = collections.Counter(edge_class(world, scene, e) for e in route)
        rows.append({"task_id": t["task_id"], "k": k, "success": success, "steps": steps, "n_add": n_add, "n_remove": n_rm, "n_wait": n_wait, "irrelevant_adds": n_irrel,
                     "house_edge_replacements": n_repl, "churn": n_rm + n_repl, "first_add_useful": first_add_useful, "first_useful_step": first_useful_step,
                     "max_progress": min(max_prog, 1.0), "efficiency": (k / n_add) if (success and n_add) else None,
                     "add_classes": dict(cls), "route_classes": dict(route_classes) if route_classes is not None else None,
                     "route_all_supported_legal": (bool(route_classes) and set(route_classes) <= {"interior_dry"}) if success else None, "scene_id": t["scene_id"]})
    return rows


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by = defaultdict(list)
    for r in rows:
        by[r["k"]].append(r)
    out = {}
    for k, rs in sorted(by.items()):
        n = len(rs)
        s = sum(r["success"] for r in rs)
        succ = [r for r in rs if r["success"]]
        adds = sum(r["n_add"] for r in rs)
        steps = max(sum(r["steps"] for r in rs), 1)
        fa = [r["first_add_useful"] for r in rs if r["first_add_useful"] is not None]
        eff = [r["efficiency"] for r in succ if r["efficiency"] is not None]
        cls = collections.Counter()
        for r in rs:
            cls.update(r["add_classes"])
        tot_cls = max(sum(cls.values()), 1)
        out[str(k)] = {
            "n": n, "success": s / n, "success_ci95": wilson(s, n),
            "success_within_k_adds": sum(1 for r in succ if r["n_add"] <= k) / n, "success_within_k_plus_2_adds": sum(1 for r in succ if r["n_add"] <= k + 2) / n,
            "success_within_2k_adds": sum(1 for r in succ if r["n_add"] <= 2 * k) / n,
            "first_add_useful": (sum(fa) / len(fa)) if fa else None, "mean_max_progress": statistics.fmean(r["max_progress"] for r in rs),
            "mean_efficiency_on_success": statistics.fmean(eff) if eff else None,
            "irrelevant_adds_per_success": statistics.fmean(r["irrelevant_adds"] for r in succ) if succ else None,
            "mean_irrelevant_adds_all": statistics.fmean(r["irrelevant_adds"] for r in rs), "mean_adds_all": adds / n,
            "mean_churn": statistics.fmean(r["churn"] for r in rs), "mean_removes": statistics.fmean(r["n_remove"] for r in rs),
            "mean_house_edge_replacements": statistics.fmean(r["house_edge_replacements"] for r in rs),
            "action_share": {"WAIT": sum(r["n_wait"] for r in rs) / steps, "ADD": adds / steps, "REMOVE": sum(r["n_remove"] for r in rs) / steps},
            "add_native_legality_share": {c: cls[c] / tot_cls for c in ("interior_dry", "window_boundary", "water_endpoint")},
            "successful_routes_all_supported_legal": (sum(1 for r in succ if r["route_all_supported_legal"]) / len(succ)) if succ else None,
        }
    tot = len(rows)
    out["all"] = {"n": tot, "success": sum(r["success"] for r in rows) / tot}
    return out
