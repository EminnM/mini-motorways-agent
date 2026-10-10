"""Evaluation of policies on frozen synthetic task sets: wait / random / oracle / MaskablePPO model. Never trains."""
import math
import statistics
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

import numpy as np

from . import layout as L
from .env import SyntheticGapEnv


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0.0, c - h), min(1.0, c + h)]


def _solution_stats(model, obs, mask, sol_actions):
    import torch
    with torch.no_grad():
        obs_t, _ = model.policy.obs_to_tensor(obs)
        dist = model.policy.get_distribution(obs_t, action_masks=mask[None, :])
        probs = dist.distribution.probs[0].cpu().numpy()
    best = max(probs[a] for a in sol_actions)
    return float(sum(probs[a] for a in sol_actions)), int((probs > best).sum()) + 1


def evaluate(policy: str, tasks: List[Dict[str, Any]], model=None, deterministic: bool = True, reward_mode: str = "completion",
             construction_window: int = 20, seed: int = 0) -> List[Dict[str, Any]]:
    env = SyntheticGapEnv(tasks, reward_mode=reward_mode, construction_window=construction_window, sampling="sequential")
    rng = np.random.default_rng(seed)
    rows = []
    for i, t in enumerate(tasks):
        obs, _ = env.reset(options={"task_index": i})
        sol = set(t["solution_action_indices"])
        done = False
        cat = Counter()
        first_correct = None
        p_first = rank_first = None
        n = 0
        added_correct = added_wrong = 0
        info = {}
        while not done:
            mask = env.action_masks()
            if n == 0 and model is not None:
                p_first, rank_first = _solution_stats(model, obs, mask, sol)
            if policy == "wait":
                a = 0
            elif policy == "random":
                a = int(rng.choice(np.flatnonzero(mask)))
            elif policy == "oracle":
                a = env.oracle_action()
            else:
                a, _ = model.predict(obs, action_masks=mask, deterministic=deterministic)
                a = int(a)
            if first_correct is None:
                first_correct = a in sol
            kind = L.decode_action(a)[0]
            cat[kind] += 1
            if kind == "ADD":
                if a in sol:
                    added_correct += 1
                else:
                    added_wrong += 1
            obs, r, te, tr, info = env.step(a)
            n += 1
            done = te or tr
        s = info["synthetic"]
        sx = t["solution_edges"][0]
        rows.append({"task_id": t["task_id"], "n_gaps": t["n_gaps"], "success": bool(s["success"]), "steps_to_success": s["steps_to_success"],
                     "first_action_correct": bool(first_correct), "destructive_failure": bool(s["destructive_failure"]), "timeout": bool(s["timeout"]),
                     "decisions": n, "n_wait": cat["WAIT"], "n_add": cat["ADD"], "n_remove": cat["REMOVE"], "n_add_correct": added_correct, "n_add_wrong": added_wrong,
                     "solution_prob_first_decision": p_first, "solution_rank_first_decision": rank_first,
                     "gap_x": sx[0], "gap_y": sx[1], "orientation": "horizontal" if sx[2] == "E" else "vertical", "path_length_edges": t["path_length_edges"],
                     "solution_index": t["solution_action_indices"][0]})
    return rows


def summarize(rows: List[Dict[str, Any]], train_solution_indices: Optional[set] = None) -> Dict[str, Any]:
    n = len(rows)

    def rate(key):
        k = sum(1 for r in rows if r[key])
        return {"rate": k / n, "count": k, "n": n, "ci95_wilson": wilson(k, n)}

    steps = [r["steps_to_success"] for r in rows if r["success"]]
    tot = max(sum(r["n_wait"] + r["n_add"] + r["n_remove"] for r in rows), 1)
    pr = [r["solution_prob_first_decision"] for r in rows if r["solution_prob_first_decision"] is not None]
    rk = [r["solution_rank_first_decision"] for r in rows if r["solution_rank_first_decision"] is not None]
    cnt = Counter(r["solution_index"] for r in rows)
    top_idx, top_n = cnt.most_common(1)[0]
    breakdown = {}
    for name, keyf in (("orientation", lambda r: r["orientation"]), ("gap_x_bin", lambda r: f"x{(r['gap_x'] + 18) // 8 * 8 - 18}..{(r['gap_x'] + 18) // 8 * 8 - 11}"),
                       ("path_length", lambda r: (r["path_length_edges"] // 3) * 3)):
        g = defaultdict(list)
        for r in rows:
            g[keyf(r)].append(r)
        breakdown[name] = {str(k): {"n": len(v), "completion": sum(1 for r in v if r["success"]) / len(v)} for k, v in sorted(g.items(), key=lambda kv: str(kv[0]))}
    out = {"num_tasks": n, "completion": rate("success"), "first_action_accuracy": rate("first_action_correct"),
           "median_steps_to_success": statistics.median(steps) if steps else None, "mean_steps_to_success": statistics.fmean(steps) if steps else None,
           "destructive_failure": rate("destructive_failure"), "timeout": rate("timeout"),
           "action_fractions": {"WAIT": sum(r["n_wait"] for r in rows) / tot, "ADD": sum(r["n_add"] for r in rows) / tot, "REMOVE": sum(r["n_remove"] for r in rows) / tot},
           "correct_add_fraction": sum(r["n_add_correct"] for r in rows) / tot, "wrong_add_fraction": sum(r["n_add_wrong"] for r in rows) / tot,
           "solution_prob_first_decision_mean": statistics.fmean(pr) if pr else None,
           "solution_rank_first_decision_median": statistics.median(rk) if rk else None,
           "solution_in_top5_first_decision": (sum(1 for x in rk if x <= 5) / len(rk)) if rk else None,
           "most_common_solution_index_share": top_n / n, "breakdown": breakdown}
    if train_solution_indices is not None:
        unseen = [r for r in rows if r["solution_index"] not in train_solution_indices]
        out["tasks_with_solution_index_unseen_in_train"] = {"n": len(unseen), "completion": (sum(1 for r in unseen if r["success"]) / len(unseen)) if unseen else None}
    return out
