"""Frozen-baseline report: training curves + held-out evaluation + action distributions.
Reads logs/mlp_train.out, logs/shaped_train.out and research/frozen_baselines/eval_*.jsonl."""
import glob
import json
import math
import os
import re
import statistics as st

ROOT = r"C:\Users\TKA\Desktop\mini moto"
OUT = os.path.join(ROOT, "research", "frozen_baselines")


def parse_log(path):
    """Rows (timesteps, ep_rew_mean, ep_len_mean, approx_kl, entropy_loss, explained_variance) per PPO iteration."""
    rows, cur = [], {}
    pat = re.compile(r"\|\s+([A-Za-z_/]+)\s+\|\s+([-+0-9.eE]+)\s+\|")
    for line in open(path, encoding="utf-8", errors="replace"):
        m = pat.search(line)
        if not m:
            continue
        k, v = m.group(1), float(m.group(2))
        cur[k] = v
        if k == "approx_kl" and "total_timesteps" in cur:   # last field of an iteration block we need
            rows.append(dict(cur))
            cur = {}
    return rows


def curves():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    series = {"native reward (connected start)": parse_log(os.path.join(ROOT, "logs", "mlp_train.out")),
              "connection-shaped (normal start)": parse_log(os.path.join(ROOT, "logs", "shaped_train.out"))}
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fields = [("ep_rew_mean", "episode reward (training return)"), ("ep_len_mean", "episode length (decisions)"),
              ("approx_kl", "approx KL"), ("entropy_loss", "entropy loss"), ("explained_variance", "explained variance")]
    for ax, (f, title) in zip(axes.flat, fields):
        for name, rows in series.items():
            xs = [r["total_timesteps"] for r in rows if f in r]
            ys = [r[f] for r in rows if f in r]
            ax.plot(xs, ys, label=name, linewidth=1.2)
        ax.set_title(title); ax.set_xlabel("timesteps"); ax.grid(alpha=.3)
    axes.flat[0].legend(fontsize=8)
    axes.flat[5].axis("off")
    axes.flat[5].text(0, .5, "shaped-run reward includes the shaping bonus and\nthe -10 terminal penalty, so the two reward\ncurves are not directly comparable.\nSee the held-out table for native scores.", fontsize=9)
    plt.tight_layout()
    p = os.path.join(OUT, "training_curves.png")
    plt.savefig(p, dpi=110)
    return p, {k: (rows[0], rows[-1], len(rows)) for k, rows in series.items()}


def load_eval(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip().startswith("{") and '"seed"' in l]


def se(xs):
    return st.pstdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else 0.0


def table():
    files = sorted(glob.glob(os.path.join(OUT, "eval_*.jsonl")))
    runs = {os.path.basename(f)[5:-6]: load_eval(f) for f in files}
    wait = {}
    for name, eps in runs.items():
        if name.startswith("wait_"):
            wait[name[5:]] = {e["seed"]: e for e in eps}
    lines = []
    hdr = f"{'run':38s} {'n':>2s} {'return':>13s} {'finalScore':>11s} {'len':>6s} {'term%':>6s} {'1st rew':>8s} {'WAIT/ADD/REM/UPG per ep':>26s} {'conn.destroyed':>14s} {'vs WAIT (W/L/T)':>16s}"
    lines.append(hdr)
    for name, eps in runs.items():
        if not eps:
            continue
        mode = "CONNECTED_START" if "CONNECTED" in name.split("_")[-2] + name.split("_")[-1] or name.endswith("CONNECTED_START") or "on_CONNECTED" in name else "NORMAL_START"
        if name.startswith(("wait_", "random_")):
            mode = name.split("_", 1)[1]
        ret = [e["return"] for e in eps]
        fs = [e["finalScore"] for e in eps]
        ln = [e["length"] for e in eps]
        term = sum(1 for e in eps if e["terminated"]) / len(eps)
        fr = [e["firstRewardDecision"] for e in eps if e.get("firstRewardDecision") is not None]
        ac = {k: st.mean(e["actionCounts"].get(k, 0) for e in eps) for k in ("WAIT", "ADD", "REMOVE", "UPGRADE")}
        cd = sum(1 for e in eps if e.get("connectionDestroyedAtDecision") is not None)
        wl = ""
        base = wait.get(mode)
        if base and not name.startswith("wait_"):
            w = l = t = 0
            for e in eps:
                b = base.get(e["seed"])
                if b is None:
                    continue
                d = e["return"] - b["return"]
                w += d > 0; l += d < 0; t += d == 0
            wl = f"{w}/{l}/{t}"
        lines.append(f"{name:38s} {len(eps):2d} {st.mean(ret):6.2f}±{se(ret):4.2f} {st.mean(fs):11.2f} {st.mean(ln):6.1f} {term:6.0%} {(st.mean(fr) if fr else float('nan')):8.1f} "
                     f"{ac['WAIT']:6.1f}/{ac['ADD']:5.1f}/{ac['REMOVE']:4.1f}/{ac['UPGRADE']:3.1f} {cd:>10d}/{len(eps)} {wl:>16s}")
    return "\n".join(lines), runs


if __name__ == "__main__":
    p, summ = curves()
    print("training curves ->", p)
    for k, (first, last, n) in summ.items():
        print(f"  {k}: {n} iterations; first {first.get('total_timesteps')} steps rew={first.get('ep_rew_mean')} len={first.get('ep_len_mean')}; "
              f"last {last.get('total_timesteps')} steps rew={last.get('ep_rew_mean')} len={last.get('ep_len_mean')}")
    tbl, _ = table()
    print()
    print(tbl)
    open(os.path.join(OUT, "eval_table.txt"), "w", encoding="utf-8").write(tbl + "\n")
