"""S7 evaluation: per-episode rich metrics on LongRouteEnv (privileged diagnostics only; the policy never sees them)."""
import math
import statistics
from collections import defaultdict
from typing import Any, Dict, List

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


def evaluate(policy: str, tasks: List[Dict[str, Any]], model=None, deterministic: bool = True, seed: int = 0) -> List[Dict[str, Any]]:
    import torch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    env = s7.LongRouteEnv(tasks, sampling="sequential")
    rows = []
    for i, t in enumerate(tasks):
        obs, _ = env.reset(options={"task_index": i})
        k = t["k_min"]
        r_prev = env.remaining()
        assert r_prev == k
        max_prog = 0.0
        n_add = n_rm = n_wait = n_irrel = n_repl = 0
        first_add_useful = None
        first_useful_step = None
        steps = 0
        success = False
        hset = set(s7.house_edges(env.house))
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
        rows.append({"task_id": t["task_id"], "k": k, "success": success, "steps": steps, "n_add": n_add, "n_remove": n_rm, "n_wait": n_wait, "irrelevant_adds": n_irrel,
                     "house_edge_replacements": n_repl, "first_add_useful": first_add_useful, "first_useful_step": first_useful_step, "max_progress": min(max_prog, 1.0),
                     "efficiency": (k / n_add) if (success and n_add) else None, "exact_k_adds": bool(success and n_add == k), "scene_id": t["scene_id"]})
    return rows


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by = defaultdict(list)
    for r in rows:
        by[r["k"]].append(r)
    out = {}
    for k, rs in sorted(by.items()):
        n = len(rs)
        s = sum(r["success"] for r in rs)
        adds = sum(r["n_add"] for r in rs)
        eff = [r["efficiency"] for r in rs if r["efficiency"] is not None]
        fa = [r["first_add_useful"] for r in rs if r["first_add_useful"] is not None]
        out[k] = {"n": n, "success": s / n, "success_ci95": wilson(s, n), "exact_k_adds_success": sum(r["exact_k_adds"] for r in rs) / n,
                  "first_add_useful": (sum(fa) / len(fa)) if fa else None, "episodes_with_any_useful_add": sum(r["first_useful_step"] is not None for r in rs) / n,
                  "mean_max_progress": statistics.fmean(r["max_progress"] for r in rs), "frac_reaching_half_route": sum(r["max_progress"] >= 0.5 for r in rs) / n,
                  "mean_irrelevant_adds": statistics.fmean(r["irrelevant_adds"] for r in rs), "irrelevant_share_of_adds": (sum(r["irrelevant_adds"] for r in rs) / adds) if adds else None,
                  "mean_removes": statistics.fmean(r["n_remove"] for r in rs), "mean_house_edge_replacements": statistics.fmean(r["house_edge_replacements"] for r in rs),
                  "mean_efficiency_on_success": statistics.fmean(eff) if eff else None, "median_steps_on_success": statistics.median([r["steps"] for r in rs if r["success"]]) if s else None,
                  "action_share": {"WAIT": sum(r["n_wait"] for r in rs) / max(sum(r["steps"] for r in rs), 1), "ADD": adds / max(sum(r["steps"] for r in rs), 1),
                                   "REMOVE": sum(r["n_remove"] for r in rs) / max(sum(r["steps"] for r in rs), 1)}}
    tot = len(rows)
    out["all"] = {"n": tot, "success": sum(r["success"] for r in rows) / tot}
    return out
