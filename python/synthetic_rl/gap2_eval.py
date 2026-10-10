"""Rich evaluation for multi-step (Gap-N) synthetic tasks: separates 'completed' from HOW it was completed (partial progress,
first useful action, the second edit after the first, irrelevant / destructive edits), so a policy that guesses roads until it
happens to connect is distinguishable from one that places exactly the two required edges."""
import math
import statistics
from collections import Counter, defaultdict
from typing import Any, Dict, List

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


def _probs(model, obs, mask):
    import torch
    with torch.no_grad():
        obs_t, _ = model.policy.obs_to_tensor(obs)
        dist = model.policy.get_distribution(obs_t, action_masks=mask[None, :])
        return dist.distribution.probs[0].cpu().numpy()


def _rank_prob(probs, actions):
    if not actions:
        return None, None
    best = max(probs[a] for a in actions)
    return int((probs > best).sum()) + 1, float(sum(probs[a] for a in actions))


def evaluate_rich(policy: str, tasks: List[Dict[str, Any]], model=None, deterministic: bool = True, seed: int = 0, window: int = 20,
                  reward_mode: str = "completion") -> List[Dict[str, Any]]:
    import torch
    torch.manual_seed(seed)
    env = SyntheticGapEnv(tasks, reward_mode=reward_mode, construction_window=window, sampling="sequential")
    rng = np.random.default_rng(seed)
    rows = []
    for i, t in enumerate(tasks):
        obs, _ = env.reset(options={"task_index": i})
        sol = list(t["solution_action_indices"])
        sol_set = set(sol)
        original = {tuple(e) for e in t["built_edges"]}
        added_correct = set()
        irrelevant = removes = 0
        destructive = False
        actions = []
        rank0 = prob0 = rank1 = prob1 = None
        second_after_first_correct = None
        first_correct_at = None
        done = False
        info = {}
        n = 0
        while not done:
            mask = env.action_masks()
            probs = None
            need_probs = model is not None and (n == 0 or (len(added_correct) == 1 and rank1 is None))
            if need_probs:
                probs = _probs(model, obs, mask)
                remaining = [a for a in sol if a not in added_correct]
                r, p = _rank_prob(probs, remaining)
                if n == 0:
                    rank0, prob0 = r, p
                else:
                    rank1, prob1 = r, p
            if policy == "wait":
                a = 0
            elif policy == "random":
                a = int(rng.choice(np.flatnonzero(mask)))
            elif policy == "oracle":
                a = env.oracle_action()
            else:
                a, _ = model.predict(obs, action_masks=mask, deterministic=deterministic)
                a = int(a)
            kind = L.decode_action(a)
            actions.append(a)
            if len(added_correct) == 1 and second_after_first_correct is None and first_correct_at is not None and n == first_correct_at + 1:
                second_after_first_correct = a in sol_set and a not in added_correct
            if kind[0] == "ADD":
                if a in sol_set:
                    if a not in added_correct and first_correct_at is None:
                        first_correct_at = n
                    added_correct.add(a)
                else:
                    irrelevant += 1
            elif kind[0] == "REMOVE":
                removes += 1
            obs, r_, te, tr, info = env.step(a)
            n += 1
            done = te or tr
        s = info["synthetic"]
        d = s["destructive_failure"]
        sx = [tuple(e) for e in t["solution_edges"]]
        orient = "".join(sorted("H" if e[2] == "E" else "V" for e in sx))
        rows.append({
            "task_id": t["task_id"], "n_gaps": t["n_gaps"], "success": bool(s["success"]), "steps_to_success": s["steps_to_success"], "decisions": n,
            "first_action_useful": bool(actions[0] in sol_set), "n_correct_edges_added": len(added_correct), "partial_progress": len(added_correct) >= 1,
            "partial_only": len(added_correct) == 1 and not s["success"], "second_action_correct_after_first_edit": second_after_first_correct,
            "irrelevant_adds": irrelevant, "removes": removes, "destructive_failure": bool(d), "timeout": bool(s["timeout"]),
            "success_without_irrelevant_edits": bool(s["success"] and irrelevant == 0), "success_with_le2_irrelevant": bool(s["success"] and irrelevant <= 2),
            "n_wait": sum(1 for a in actions if a == 0), "n_add": sum(1 for a in actions if L.decode_action(a)[0] == "ADD"), "n_remove": removes,
            "rank_best_remaining_decision1": rank0, "prob_remaining_decision1": prob0, "rank_remaining_after_first_edit": rank1, "prob_remaining_after_first_edit": prob1,
            "orientation_pair": orient, "path_length_edges": t["path_length_edges"], "window_x": t["window"][0], "min_solution_x": min(e[0] for e in sx)})
    return rows


def summarize_rich(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)

    def rate(key, sub=None):
        sub = rows if sub is None else sub
        k = sum(1 for r in sub if r[key])
        return {"rate": k / len(sub) if sub else None, "count": k, "n": len(sub), "ci95_wilson": wilson(k, len(sub))}

    steps = [r["steps_to_success"] for r in rows if r["success"]]
    tot = max(sum(r["n_wait"] + r["n_add"] + r["n_remove"] for r in rows), 1)
    after = [r for r in rows if r["second_action_correct_after_first_edit"] is not None]
    r0 = [r["rank_best_remaining_decision1"] for r in rows if r["rank_best_remaining_decision1"] is not None]
    r1 = [r["rank_remaining_after_first_edit"] for r in rows if r["rank_remaining_after_first_edit"] is not None]
    p0 = [r["prob_remaining_decision1"] for r in rows if r["prob_remaining_decision1"] is not None]
    p1 = [r["prob_remaining_after_first_edit"] for r in rows if r["prob_remaining_after_first_edit"] is not None]
    only1 = [r for r in rows if r["first_action_useful"]]
    out = {"num_tasks": n, "completion": rate("success"), "first_useful_action_accuracy": rate("first_action_useful"), "partial_progress_at_least_one_edge": rate("partial_progress"),
           "partial_only_not_completed": rate("partial_only"), "destructive_failure": rate("destructive_failure"), "timeout": rate("timeout"),
           "median_steps_to_success": statistics.median(steps) if steps else None, "mean_steps_to_success": statistics.fmean(steps) if steps else None,
           "success_with_zero_irrelevant_edits": rate("success_without_irrelevant_edits"), "success_with_at_most_2_irrelevant_edits": rate("success_with_le2_irrelevant"),
           "mean_irrelevant_adds_per_episode": statistics.fmean(r["irrelevant_adds"] for r in rows), "mean_removes_per_episode": statistics.fmean(r["removes"] for r in rows),
           "second_edit_correct_given_first_correct_edit_was_followed": ({"rate": sum(1 for r in after if r["second_action_correct_after_first_edit"]) / len(after), "n": len(after)} if after else None),
           "completion_given_first_action_useful": rate("success", only1) if only1 else None,
           "action_fractions": {"WAIT": sum(r["n_wait"] for r in rows) / tot, "ADD": sum(r["n_add"] for r in rows) / tot, "REMOVE": sum(r["n_remove"] for r in rows) / tot},
           "rank_of_best_remaining_edge_decision1_median": statistics.median(r0) if r0 else None, "prob_on_remaining_edges_decision1_mean": statistics.fmean(p0) if p0 else None,
           "rank_of_remaining_edge_after_first_edit_median": statistics.median(r1) if r1 else None, "prob_on_remaining_edge_after_first_edit_mean": statistics.fmean(p1) if p1 else None,
           "n_with_probs_after_first_edit": len(r1)}
    bd = {}
    for name, keyf in (("orientation_pair", lambda r: r["orientation_pair"]), ("path_length", lambda r: (r["path_length_edges"] // 3) * 3),
                       ("map_x_of_solution", lambda r: f"x{(r['min_solution_x'] + 18) // 9 * 9 - 18}..{(r['min_solution_x'] + 18) // 9 * 9 - 10}")):
        g = defaultdict(list)
        for r in rows:
            g[keyf(r)].append(r)
        bd[name] = {str(k): {"n": len(v), "completion": sum(1 for r in v if r["success"]) / len(v)} for k, v in sorted(g.items(), key=lambda kv: str(kv[0]))}
    out["breakdown"] = bd
    return out
