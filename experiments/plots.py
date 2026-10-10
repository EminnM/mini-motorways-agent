"""Plots derived ONLY from the stored CSV/JSON files (raw data is authoritative). Where a smoothed line is shown
the raw series is drawn with it."""
import csv
import json
import os
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _read(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _num(rows, key, xkey="training_timestep"):
    xs, ys = [], []
    for r in rows:
        if r.get(key, "") not in ("", None) and r.get(xkey, "") not in ("", None):
            xs.append(float(r[xkey])); ys.append(float(r[key]))
    return xs, ys


def _rolling(ys, window):
    out = []
    for i in range(len(ys)):
        lo = max(0, i - window + 1)
        out.append(sum(ys[lo:i + 1]) / (i + 1 - lo))
    return out


def _series_plot(path, title, ylabel, raw_x, raw_y, window, extra=None, raw_label="per-episode (raw)", smooth_label=None):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if raw_x:
        ax.plot(raw_x, raw_y, ".", ms=3, alpha=.35, label=raw_label)
        w = min(window, max(1, len(raw_y)))
        ax.plot(raw_x, _rolling(raw_y, w), lw=1.6, label=smooth_label or f"rolling mean (window={w}, display only)")
    for xs, ys, label in (extra or []):
        if xs:
            ax.plot(xs, ys, lw=1.2, label=label)
    ax.set_xlabel("training timesteps"); ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(alpha=.3)
    if raw_x or extra:
        ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def make_training_plots(seed_dir: str, name: str = "", window: int = 25):
    metrics = os.path.join(seed_dir, "metrics")
    plots = os.path.join(seed_dir, "plots")
    os.makedirs(plots, exist_ok=True)
    eps = _read(os.path.join(metrics, "episodes.csv"))
    trn = _read(os.path.join(metrics, "training.csv"))
    made = []
    x, y = _num(eps, "return")
    tx, ty = _num(trn, "episode_return", "timestep")
    _series_plot(os.path.join(plots, "return_vs_steps.png"), f"Episode return {name}", "return (reward as seen by the agent)", x, y, window,
                 extra=[(tx, ty, "SB3 ep_rew_mean (training.csv)")])
    made.append("return_vs_steps.png")
    x, y = _num(eps, "native_score")
    tx, ty = _num(trn, "episode_native_score", "timestep")
    _series_plot(os.path.join(plots, "native_score_vs_steps.png"), f"Native score per episode {name}", "native score (sum of scoreDelta)", x, y, window,
                 extra=[(tx, ty, "mean of last 100 episodes (training.csv)")])
    made.append("native_score_vs_steps.png")
    x, y = _num(eps, "length_decisions")
    tx, ty = _num(trn, "episode_length", "timestep")
    _series_plot(os.path.join(plots, "episode_length_vs_steps.png"), f"Episode length {name}", "decisions", x, y, window,
                 extra=[(tx, ty, "SB3 ep_len_mean (training.csv)")])
    made.append("episode_length_vs_steps.png")
    # action distribution: stacked fractions over episode bins + overall bar from actions.csv
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [3, 1.2]})
    if eps:
        nb = min(20, len(eps))
        size = max(1, len(eps) // nb)
        bins = [eps[i:i + size] for i in range(0, len(eps), size)]
        labels = [float(b[-1]["training_timestep"]) for b in bins]
        bottoms = [0.0] * len(bins)
        for cat, col in (("WAIT", "wait_fraction"), ("ADD_ROAD", "add_fraction"), ("REMOVE_ROAD", "remove_fraction"), ("UPGRADE", "upgrade_fraction")):
            vals = [sum(float(e[col]) for e in b) / len(b) for b in bins]
            a1.bar(range(len(bins)), vals, bottom=bottoms, label=cat)
            bottoms = [bt + v for bt, v in zip(bottoms, vals)]
        a1.set_xticks(range(len(bins))); a1.set_xticklabels([f"{int(l)}" for l in labels], rotation=60, fontsize=7)
        a1.set_xlabel("training timestep at end of episode bin"); a1.set_ylabel("share of decisions")
        a1.set_title("Action mix during training (mean over episodes in bin)"); a1.legend(fontsize=7)
    acts = _read(os.path.join(metrics, "actions.csv"))
    if acts:
        a2.bar([a["category"] for a in acts], [float(a["fraction"]) for a in acts])
        a2.set_title("Overall (actions.csv)"); a2.tick_params(axis="x", rotation=30, labelsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(plots, "action_distribution.png"), dpi=110); plt.close(fig)
    made.append("action_distribution.png")
    # evaluation-vs-steps if checkpoint evaluations exist
    ev = _read(os.path.join(seed_dir, "evaluation", "checkpoints.csv"))
    if ev and "mean_native_score" in ev[0]:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        xs = [float(r["training_timestep"]) for r in ev]
        ys = [float(r["mean_native_score"]) for r in ev]
        lo = [float(r["native_score_ci95_low"]) for r in ev]
        hi = [float(r["native_score_ci95_high"]) for r in ev]
        ax.errorbar(xs, ys, yerr=[[y - l for y, l in zip(ys, lo)], [h - y for y, h in zip(ys, hi)]], marker="o", capsize=3, label="held-out mean native score (95% CI)")
        ax.set_xlabel("training timesteps"); ax.set_ylabel("native score"); ax.set_title(f"Held-out evaluation vs steps {name}")
        ax.grid(alpha=.3); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(plots, "eval_score_vs_steps.png"), dpi=110); plt.close(fig)
        made.append("eval_score_vs_steps.png")
    return made
