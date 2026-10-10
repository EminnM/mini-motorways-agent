"""python -m experiments.compare runs/run_A runs/run_B [...]  -> comparison/{summary.csv,comparison.png}
Compares native score, return, episode length, terminal rate and action distribution from each run's stored
evaluation/results.json (raw data authoritative). No statistical testing."""
import argparse
import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import paths
from .recording import CATEGORIES


def load(run_dir: str, eval_rel: str):
    p = os.path.join(run_dir, eval_rel, "results.json")
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p} not found (run `python -m experiments.evaluate` first, or pass --eval <subfolder>)")
    return json.load(open(p, encoding="utf-8"))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m experiments.compare")
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--eval", default="evaluation", help="evaluation folder inside each run (default: evaluation)")
    ap.add_argument("--out", default=None, help="output folder (default: comparison/ in the current directory)")
    a = ap.parse_args(argv)
    out = a.out or os.path.join(os.getcwd(), "comparison")
    os.makedirs(out, exist_ok=True)
    rows = []
    for r in a.runs:
        res = load(r, a.eval)
        meta = {}
        try:
            meta = json.load(open(os.path.join(r, "metadata.json"), encoding="utf-8"))
        except Exception:
            pass
        row = {"run": os.path.basename(os.path.normpath(r)), "policy": res.get("policy"), "checkpoint_sha256": res.get("checkpoint_sha256"),
               "num_episodes": res["num_episodes"], "mean_native_score": res["mean_native_score"],
               "native_score_ci95_low": res["native_score_ci95"][0], "native_score_ci95_high": res["native_score_ci95"][1],
               "median_native_score": res["median_native_score"], "mean_return": res["mean_return"],
               "mean_episode_length": res["mean_episode_length"], "terminal_rate": res["terminal_rate"],
               "initialization": res["environment"]["initialization"], "reward_mode_eval": res["environment"].get("reward")}
        for c in CATEGORIES:
            row[f"frac_{c}"] = res["action_fractions"][c]
        rows.append(row)
    with open(os.path.join(out, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); [w.writerow(r) for r in rows]
    names = [r["run"][-28:] for r in rows]
    fig, ax = plt.subplots(1, 5, figsize=(20, 4.4))
    x = range(len(rows))
    ms = [r["mean_native_score"] for r in rows]
    ax[0].bar(x, ms, yerr=[[m - r["native_score_ci95_low"] for m, r in zip(ms, rows)], [r["native_score_ci95_high"] - m for m, r in zip(ms, rows)]], capsize=4)
    ax[0].set_title("mean native score (95% CI)")
    ax[1].bar(x, [r["mean_return"] for r in rows]); ax[1].set_title("mean return")
    ax[2].bar(x, [r["mean_episode_length"] for r in rows]); ax[2].set_title("mean episode length (decisions)")
    ax[3].bar(x, [r["terminal_rate"] for r in rows]); ax[3].set_title("terminal rate"); ax[3].set_ylim(0, 1)
    bottoms = [0.0] * len(rows)
    for c in CATEGORIES:
        vals = [r[f"frac_{c}"] for r in rows]
        ax[4].bar(x, vals, bottom=bottoms, label=c)
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    ax[4].set_title("action distribution"); ax[4].legend(fontsize=7)
    for a_ in ax:
        a_.set_xticks(list(x)); a_.set_xticklabels(names, rotation=35, ha="right", fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(out, "comparison.png"), dpi=110); plt.close(fig)
    print(f"wrote {os.path.join(out, 'summary.csv')} and comparison.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
