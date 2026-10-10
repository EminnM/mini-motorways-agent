"""Aggregates the S1 runs (MLP vs spatial CNN) into tables, plots and research/SYNTHETIC_RL_LAB.md data sections."""
import csv
import glob
import json
import os
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = r"C:\Users\TKA\Desktop\mini moto"
sys.path.insert(0, os.path.join(ROOT, "python"))
LAB = os.path.join(ROOT, "research", "synthetic_lab")
MAN = os.path.join(LAB, "manifests")


def runs(arch):
    return sorted(glob.glob(os.path.join(ROOT, "runs", f"*syn_s1_{arch}_s*")))


def mean_sd(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None
    return statistics.fmean(xs), (statistics.stdev(xs) if len(xs) > 1 else 0.0)


def fmt(m, s, pct=True):
    if m is None:
        return "n/a"
    return f"{100 * m:.1f}% ± {100 * s:.1f}" if pct else f"{m:.2f} ± {s:.2f}"


def load_val(rd):
    p = os.path.join(rd, "evaluation", "validation_by_checkpoint.csv")
    return list(csv.DictReader(open(p, encoding="utf-8")))


def load_res(rd, name):
    return json.load(open(os.path.join(rd, "evaluation", name, "results.json")))


def build():
    from synthetic_rl import evaluation as E
    from synthetic_rl.env import load_tasks
    test = load_tasks(os.path.join(MAN, "gap1_test.jsonl"))
    val = load_tasks(os.path.join(MAN, "gap1_validation.jsonl"))
    train_idx = {i for x in load_tasks(os.path.join(MAN, "gap1_train.jsonl")) for i in x["solution_action_indices"]}
    base = {}
    for name in ("wait", "random", "oracle"):
        reps = [1, 2, 3, 4, 5] if name == "random" else [0]
        res = [E.summarize(E.evaluate(name, test, None, True, seed=s), train_idx) for s in reps]
        base[name] = {"completion": statistics.fmean(r["completion"]["rate"] for r in res), "first_action": statistics.fmean(r["first_action_accuracy"]["rate"] for r in res),
                      "median_steps": res[0]["median_steps_to_success"], "destructive": statistics.fmean(r["destructive_failure"]["rate"] for r in res),
                      "timeout": statistics.fmean(r["timeout"]["rate"] for r in res), "actions": res[0]["action_fractions"], "dominant_index_share": res[0]["most_common_solution_index_share"]}
    base_val = {n: statistics.fmean(E.summarize(E.evaluate(n, val, None, True, seed=s))["completion"]["rate"] for s in ([1, 2, 3, 4, 5] if n == "random" else [0])) for n in ("wait", "random", "oracle")}
    data = {}
    for arch in ("mlp", "cnn"):
        d = {"runs": runs(arch), "val": [], "meta": []}
        for rd in d["runs"]:
            d["val"].append(load_val(rd))
            d["meta"].append(json.load(open(os.path.join(rd, "metadata.json"))))
        data[arch] = d
    return base, base_val, data


def main():
    base, base_val, data = build()
    L = []
    steps = sorted({int(r["training_timestep"]) for r in data["mlp"]["val"][0]})
    # ---------------------------------------------------------------- figures
    fig, ax = plt.subplots(2, 3, figsize=(17, 9))
    colors = {"mlp": "tab:red", "cnn": "tab:blue"}
    for arch in ("mlp", "cnn"):
        for mode, ls in (("deterministic", "-"), ("stochastic", "--")):
            xs, ms, ss = [], [], []
            for st in steps:
                vals = [float(r["completion"]) for v in data[arch]["val"] for r in v if int(r["training_timestep"]) == st and r["mode"] == mode]
                if vals:
                    xs.append(st); ms.append(statistics.fmean(vals)); ss.append(statistics.stdev(vals) if len(vals) > 1 else 0)
            ax[0, 0].errorbar(xs, ms, yerr=ss, fmt=ls, color=colors[arch], marker="o", capsize=3, label=f"{arch} ({mode})")
    for n, c in (("random", "gray"), ("oracle", "green")):
        ax[0, 0].axhline(base_val[n], ls=":", color=c, label=n)
    ax[0, 0].set_title("validation completion vs training decisions (mean ± sd over 3 seeds)"); ax[0, 0].set_ylim(0, 1.02); ax[0, 0].legend(fontsize=7); ax[0, 0].grid(alpha=.3)
    for k, (key, title) in enumerate((("first_action_accuracy", "validation first-action accuracy"), ("solution_rank_first_median", "median rank of a correct action at decision 1"))):
        a = ax[0, k + 1]
        for arch in ("mlp", "cnn"):
            xs, ms = [], []
            for st in steps:
                vals = [float(r[key]) for v in data[arch]["val"] for r in v if int(r["training_timestep"]) == st and r["mode"] == "deterministic" and r[key] not in ("", "None")]
                if vals:
                    xs.append(st); ms.append(statistics.fmean(vals))
            a.plot(xs, ms, "o-", color=colors[arch], label=arch)
        a.set_title(title); a.legend(fontsize=7); a.grid(alpha=.3)
    # training success curve
    for arch in ("mlp", "cnn"):
        allb = {}
        for rd in data[arch]["runs"]:
            eps = list(csv.DictReader(open(os.path.join(rd, "metrics", "episodes.csv"), encoding="utf-8")))
            for e in eps:
                b = int(float(e["training_timestep"])) // 2500
                allb.setdefault(b, []).append(float(e["return"]))
        xs = sorted(allb)
        ax[1, 0].plot([x * 2500 + 1250 for x in xs], [statistics.fmean(allb[x]) for x in xs], "o-", color=colors[arch], label=arch)
    ax[1, 0].set_title("training episodes: mean return (= success rate) per 2,500 decisions"); ax[1, 0].legend(); ax[1, 0].grid(alpha=.3)
    # test breakdowns for selected, deterministic
    for arch in ("mlp", "cnn"):
        rs = [load_res(rd, "test_selected_deterministic") for rd in data[arch]["runs"]]
        for k, (grp, a) in enumerate((("orientation", ax[1, 1]), ("gap_x_bin", ax[1, 2]))):
            keys = sorted(rs[0]["breakdown"][grp])
            vals = [[r["breakdown"][grp].get(kk, {"completion": 0})["completion"] for r in rs] for kk in keys]
            xs = np.arange(len(keys)) + (0.2 if arch == "cnn" else -0.2)
            a.bar(xs, [statistics.fmean(v) for v in vals], width=0.4, color=colors[arch], label=arch, yerr=[statistics.stdev(v) if len(v) > 1 else 0 for v in vals], capsize=2)
            a.set_xticks(np.arange(len(keys))); a.set_xticklabels(keys, rotation=30, fontsize=7); a.set_ylim(0, 1.02)
            a.set_title(f"TEST completion by {grp.replace('_', ' ')} (selected ckpt, deterministic)")
    ax[1, 1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(LAB, "s1_mlp_vs_cnn.png"), dpi=110)
    # ---------------------------------------------------------------- tables
    L.append("### S1 results (3 training seeds per architecture; synthetic Gap-1, immediate completion reward, native-compatible 3,111-action space)\n")
    L.append("Held-out TEST (400 unseen layouts), checkpoint SELECTED on VALIDATION only (deterministic policy unless stated); mean ± sd across seeds.\n")
    L.append("| policy | test completion | first-action accuracy | median steps | destructive | timeout | WAIT / ADD / REMOVE | P(correct) at decision 1 | median rank of correct action |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for n, b in base.items():
        L.append(f"| {n.upper() if n != 'oracle' else 'ORACLE'} | {100 * b['completion']:.1f}% | {100 * b['first_action']:.1f}% | {b['median_steps']} | {100 * b['destructive']:.1f}% | {100 * b['timeout']:.1f}% | "
                 f"{100 * b['actions']['WAIT']:.0f} / {100 * b['actions']['ADD']:.0f} / {100 * b['actions']['REMOVE']:.0f} % | - | - |")
    summary = {}
    for arch in ("mlp", "cnn"):
        for which, label in (("selected_deterministic", "selected ckpt, deterministic"), ("selected_stochastic", "selected ckpt, stochastic"), ("final_deterministic", "final ckpt, deterministic")):
            rs = [load_res(rd, f"test_{which}") for rd in data[arch]["runs"]]
            cm, cs = mean_sd([r["completion"]["rate"] for r in rs])
            fm, fs = mean_sd([r["first_action_accuracy"]["rate"] for r in rs])
            dm, ds = mean_sd([r["destructive_failure"]["rate"] for r in rs])
            tm, ts = mean_sd([r["timeout"]["rate"] for r in rs])
            md_ = [r["median_steps_to_success"] for r in rs]
            af = {k: statistics.fmean(r["action_fractions"][k] for r in rs) for k in ("WAIT", "ADD", "REMOVE")}
            pm, ps = mean_sd([r["solution_prob_first_decision_mean"] for r in rs])
            rm, _ = mean_sd([r["solution_rank_first_decision_median"] for r in rs])
            L.append(f"| {'flat MLP' if arch == 'mlp' else 'spatial CNN'} ({label}) | {fmt(cm, cs)} | {fmt(fm, fs)} | {md_} | {fmt(dm, ds)} | {fmt(tm, ts)} | "
                     f"{100 * af['WAIT']:.0f} / {100 * af['ADD']:.0f} / {100 * af['REMOVE']:.0f} % | {pm:.4f} | {rm:.1f} |")
            summary[(arch, which)] = {"completion": (cm, cs), "per_seed": [r["completion"]["rate"] for r in rs]}
    L.append(f"\nPer-seed TEST completion (selected checkpoint, deterministic): MLP {[round(x, 3) for x in summary[('mlp', 'selected_deterministic')]['per_seed']]}, "
             f"CNN {[round(x, 3) for x in summary[('cnn', 'selected_deterministic')]['per_seed']]}.")
    L.append("\n### Validation completion by training step (200 validation layouts; mean ± sd over seeds)\n")
    L.append("| decisions | MLP det. | MLP stoch. | CNN det. | CNN stoch. |\n|---|---|---|---|---|")
    for st in steps:
        cells = []
        for arch in ("mlp", "cnn"):
            for mode in ("deterministic", "stochastic"):
                vals = [float(r["completion"]) for v in data[arch]["val"] for r in v if int(r["training_timestep"]) == st and r["mode"] == mode]
                cells.append(fmt(*mean_sd(vals)) if vals else "n/a")
        L.append(f"| {st} | " + " | ".join(cells) + " |")
    L.append(f"\nValidation baselines: WAIT {100 * base_val['wait']:.1f}%, RANDOM LEGAL {100 * base_val['random']:.1f}% (5 seeds), ORACLE {100 * base_val['oracle']:.0f}%.")
    L.append("\n### Cost\n| architecture | parameters | training SPS (per process, 6 processes in parallel) | wall-clock for 50k decisions | native game reference |\n|---|---|---|---|---|")
    for arch in ("mlp", "cnn"):
        ms_ = [m["seed_summaries"][list(m["seed_summaries"])[0]] for m in data[arch]["meta"]]
        L.append(f"| {arch} | {ms_[0]['parameters']:,} | {statistics.fmean(x['sps'] for x in ms_):.0f} | {statistics.fmean(x['wall_seconds'] for x in ms_) / 60:.1f} min | native ~4-5 SPS (about 3 h per 50k) |")
    # memorisation checks
    rs = [load_res(rd, "test_selected_deterministic") for rd in data["cnn"]["runs"]] + [load_res(rd, "test_selected_deterministic") for rd in data["mlp"]["runs"]]
    L.append("\n### Memorisation checks (TEST)\n")
    L.append(f"- Most common correct-action index covers {100 * base['oracle']['dominant_index_share']:.1f}% of test tasks, so repeating one index cannot exceed that.")
    for arch in ("mlp", "cnn"):
        us = [load_res(rd, "test_selected_deterministic").get("tasks_with_solution_index_unseen_in_train") for rd in data[arch]["runs"]]
        L.append(f"- {arch}: tasks whose correct action index never occurs in TRAIN: n={us[0]['n'] if us and us[0] else 0}; completion on them per seed {[None if u is None or u['completion'] is None else round(u['completion'], 3) for u in us]}")
    open(os.path.join(LAB, "s1_results_section.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
