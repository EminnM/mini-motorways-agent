"""S8: aggregate the three runs into tables / curves (reads research/s8_long_route_training/runs/*, writes summary.json + figures).   python experiments/s8_report.py"""
import csv
import gzip
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "research", "s8_long_route_training", "runs")
OUT = os.path.join(ROOT, "research", "s8_long_route_training")
KS = ("2", "3", "5", "7", "10")


def load_run(arm):
    d = os.path.join(RUNS, f"arm_{arm}_s11")
    r = {"dir": d}
    r["val"] = [json.loads(l) for l in open(os.path.join(d, "eval_validation.jsonl"))]
    r["status"] = json.load(open(os.path.join(d, "train_status.json"))) if os.path.exists(os.path.join(d, "train_status.json")) else None
    r["test"] = json.load(open(os.path.join(d, "final_test.json"))) if os.path.exists(os.path.join(d, "final_test.json")) else None
    r["sel"] = json.load(open(os.path.join(d, "selection.json"))) if os.path.exists(os.path.join(d, "selection.json")) else None
    r["stages"] = [json.loads(l) for l in open(os.path.join(d, "stages.jsonl"))] if os.path.exists(os.path.join(d, "stages.jsonl")) else []
    p = os.path.join(d, "sb3", "progress.csv")
    r["sb3"] = list(csv.DictReader(open(p))) if os.path.exists(p) else []
    return r


def main():
    runs = {a: load_run(a) for a in "ABC" if os.path.exists(os.path.join(RUNS, f"arm_{a}_s11", "eval_validation.jsonl"))}
    summ = {"arms": {}}
    for a, r in runs.items():
        curve = []
        for v in r["val"]:
            row = {"decisions": v["decisions"], "stage": v["stage"], "tag": v["tag"], "wall_s": v["wall_s"]}
            for mode in ("stochastic", "deterministic"):
                row[mode] = {k: round(v[mode][k]["success"], 4) for k in KS}
                row[mode + "_eff"] = {k: (round(v[mode][k]["mean_efficiency_on_success"], 3) if v[mode][k]["mean_efficiency_on_success"] is not None else None) for k in KS}
                row[mode + "_irrel_all"] = {k: round(v[mode][k]["mean_irrelevant_adds_all"], 1) for k in KS}
            curve.append(row)
        sb = [{k: float(x[k]) for k in ("time/total_timesteps", "train/entropy_loss", "train/approx_kl", "train/explained_variance", "train/value_loss", "train/policy_gradient_loss", "train/clip_fraction", "time/fps", "rollout/ep_rew_mean", "rollout/ep_len_mean") if x.get(k) not in ("", None)} for x in r["sb3"]]
        summ["arms"][a] = {"validation_curve": curve, "stages": r["stages"], "selection": r["sel"], "status": r["status"], "ppo_metrics_every_iteration": sb,
                           "test": r["test"]["summary"] if r["test"] else None, "test_checkpoint": r["test"]["checkpoint"] if r["test"] else None}
    json.dump(summ, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 5, figsize=(22, 7), sharey=True)
        cols = {"A": "tab:red", "B": "tab:blue", "C": "tab:green"}
        for mi, mode in enumerate(("stochastic", "deterministic")):
            for ki, k in enumerate(KS):
                ax = axes[mi][ki]
                for a, r in runs.items():
                    xs = [v["decisions"] for v in r["val"]]
                    ys = [100 * v[mode][k]["success"] for v in r["val"]]
                    ax.plot(xs, ys, "-o", color=cols[a], label=f"arm {a}")
                ax.set_title(f"{mode}, k={k}")
                ax.set_xlabel("training decisions")
                if ki == 0:
                    ax.set_ylabel("validation success %")
                ax.grid(alpha=0.3)
        axes[0][0].legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUT, "validation_curves.png"), dpi=110)
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        for a, r in runs.items():
            xs = [float(x["time/total_timesteps"]) for x in r["sb3"] if x.get("train/entropy_loss") not in ("", None)]
            for ax, key, ttl in zip(axes, ("train/entropy_loss", "train/approx_kl", "train/explained_variance"), ("entropy_loss (=-entropy, nats)", "approx_kl", "explained_variance")):
                ys = [float(x[key]) for x in r["sb3"] if x.get("train/entropy_loss") not in ("", None)]
                ax.plot(xs, ys, color=cols[a], label=f"arm {a}")
                ax.set_title(ttl)
                ax.set_xlabel("decisions")
                ax.grid(alpha=0.3)
        axes[0].legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUT, "ppo_diagnostics.png"), dpi=110)
    except Exception as e:
        print("plot failed:", e)
    # paired TEST comparison (same tasks)
    paired = {}
    rows = {}
    for a, r in runs.items():
        p = os.path.join(r["dir"], "final_test_rows.json.gz")
        if os.path.exists(p):
            rows[a] = json.load(gzip.open(p, "rt"))
    for mode in ("stochastic", "deterministic"):
        for x, y in (("A", "B"), ("A", "C"), ("B", "C")):
            if x in rows and y in rows:
                for k in (5, 7, 10):
                    rx = {t["task_id"]: t["success"] for t in rows[x][mode] if t["k"] == k}
                    ry = {t["task_id"]: t["success"] for t in rows[y][mode] if t["k"] == k}
                    ids = sorted(rx)
                    paired[f"{mode}_{x}_vs_{y}_k{k}"] = {"n": len(ids), f"{x}_only": sum(1 for i in ids if rx[i] and not ry[i]), f"{y}_only": sum(1 for i in ids if ry[i] and not rx[i]), "both": sum(1 for i in ids if rx[i] and ry[i])}
    json.dump(paired, open(os.path.join(OUT, "paired_test_comparison.json"), "w"), indent=1)
    print("wrote summary.json, figures, paired comparison for", list(runs))


if __name__ == "__main__":
    main()
