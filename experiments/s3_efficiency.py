"""S3-D Part A: re-score recorded S3-A rows (no new simulation)."""
import json, os, numpy as np
R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "research", "s3_gap2")
raw = json.load(open(os.path.join(R, "s3a_zero_shot_rows.json")))
rng = np.random.default_rng(0)
def wil(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0, c - h), 3), round(min(1, c + h), 3)]
def vec(rows, f): return np.array([f(r) for r in rows], float)
def budget(rows, k): return vec(rows, lambda r: r["success"] and r["n_add"] <= k)
out = {}
for key, rows in raw.items():
    s = [r for r in rows if r["success"]]
    eff = [2 / r["n_add"] for r in s]
    out[key] = {"n": len(rows), "completion_20_decisions": float(vec(rows, lambda r: r["success"]).mean()),
        "exactly_2_ADDs": float(budget(rows, 2).mean()), "le3_ADDs": float(budget(rows, 3).mean()), "le5_ADDs": float(budget(rows, 5).mean()),
        "le10_ADDs": float(budget(rows, 10).mean()),
        "mean_irrelevant_adds_per_success": float(np.mean([r["irrelevant_adds"] for r in s])) if s else None,
        "median_irrelevant_adds_per_success": float(np.median([r["irrelevant_adds"] for r in s])) if s else None,
        "mean_total_adds_per_success": float(np.mean([r["n_add"] for r in s])) if s else None,
        "mean_total_decisions_per_success": float(np.mean([r["decisions"] for r in s])) if s else None,
        "second_correct_after_first_correct": (lambda a: [float(np.mean(a)), len(a)])([r["second_action_correct_after_first_edit"] for r in rows if r["second_action_correct_after_first_edit"] is not None]),
        "first_action_useful": float(vec(rows, lambda r: r["first_action_useful"]).mean()),
        "efficiency_success_mean": float(np.mean(eff)) if eff else None,
        "efficiency_hist_success": {b: int(sum(1 for e in eff if lo < e <= hi)) for b, (lo, hi) in {"1.0 (2 ADDs)": (0.99, 1.0), "0.5-0.99 (3-4 ADDs)": (0.49, 0.99), "0.25-0.5 (5-8 ADDs)": (0.24, 0.49), "<=0.25 (9+ ADDs)": (0, 0.24)}.items()},
        "efficiency_all_episodes_mean_failure_as_0": float(np.mean([2 / r["n_add"] if r["success"] else 0 for r in rows])),
        "n_wait_total": int(sum(r["n_wait"] for r in rows)), "n_add_total": int(sum(r["n_add"] for r in rows)), "n_remove_total": int(sum(r["n_remove"] for r in rows)),
        "decisions_mean_all": float(np.mean([r["decisions"] for r in rows]))}
# pooled-stochastic per ckpt (3 seeds x 200) and paired differences on deterministic
def paired(a, b, k):
    x, y = budget(raw[a], k), budget(raw[b], k); d = x - y
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)]
    return {"diff": float(d.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "a_only": int(((x == 1) & (y == 0)).sum()), "b_only": int(((x == 0) & (y == 1)).sum())}
pairs = {}
for a, b in (("s13_det", "s12_det"), ("s13_det", "s11_det"), ("s12_det", "s11_det")):
    pairs[f"{a}-{b}"] = {f"le{k}": paired(a, b, k) for k in (2, 3, 5, 20)}
# correlation completion vs efficiency across checkpoints (det) and per-task: first action useful vs completion
corr = {k: {"completion": out[k]["completion_20_decisions"], "mean_irrelevant_adds_all_episodes": float(np.mean([r["irrelevant_adds"] for r in raw[k]])),
            "first_action_useful": out[k]["first_action_useful"], "le3": out[k]["le3_ADDs"]} for k in ("s11_det", "s12_det", "s13_det")}
# tasks where all three ckpts: how many solved by every ckpt within <=3
t3 = np.array([budget(raw[f"s{s}_det"], 3) for s in (11, 12, 13)]); 
extra = {"tasks_solved_le3_by_all_three": int(t3.all(0).sum()), "tasks_solved_le3_by_any": int(t3.any(0).sum())}
# window sensitivity: success among those with <= k decisions (decisions incl. WAIT)
dec = {k: {f"decisions_le{m}": float(vec(raw[k], lambda r, m=m: r["success"] and r["steps_to_success"] is not None and r["steps_to_success"] <= m).mean()) for m in (2, 3, 5, 10, 20)} for k in ("s11_det", "s12_det", "s13_det")}
json.dump({"per_run": out, "paired_det": pairs, "corr": corr, "extra": extra, "decision_budgets_det": dec}, open(os.path.join(R, "s3_efficiency.json"), "w"), indent=1)
for k in sorted(out): 
    o = out[k]; print(k, *(f"{o[f]:.3f}" for f in ("completion_20_decisions", "exactly_2_ADDs", "le3_ADDs", "le5_ADDs", "le10_ADDs")), "irrel/succ", round(o["mean_irrelevant_adds_per_success"], 2), "med", o["median_irrelevant_adds_per_success"], "eff", round(o["efficiency_success_mean"], 3), "effall", round(o["efficiency_all_episodes_mean_failure_as_0"], 3), "2nd", [round(o["second_correct_after_first_correct"][0], 3), o["second_correct_after_first_correct"][1]], "1st", o["first_action_useful"])
print(json.dumps(pairs, indent=0)); print(extra); print(dec); print(out["s13_det"]["efficiency_hist_success"], out["s11_det"]["efficiency_hist_success"], out["s12_det"]["efficiency_hist_success"])
