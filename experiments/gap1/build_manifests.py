"""Build the frozen TRAIN / VALIDATION / TEST manifests from verified candidates and report the task distribution.
Splitting is by BASE LAYOUT (house, destination, full path): layouts never cross splits."""
import argparse
import glob
import hashlib
import json
import os
import random
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments import paths  # noqa: E402

OUT = os.path.join(paths.ROOT, "research", "gap1", "manifests")


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=600)
    ap.add_argument("--val", type=int, default=120)
    ap.add_argument("--test", type=int, default=240)
    ap.add_argument("--rng-seed", type=int, default=77)
    a = ap.parse_args()
    rows, reasons = [], Counter()
    for p in sorted(glob.glob(os.path.join(OUT, "_candidates_raw_*.jsonl"))):
        for line in open(p, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                reasons[r["reason"]] += 1
                if r["ok"]:
                    rows.append(r)
    # dedupe on the exact configuration, then group by base layout
    seen, uniq = set(), []
    for r in rows:
        if r["layout_signature"] in seen:
            continue
        seen.add(r["layout_signature"])
        uniq.append(r)
    groups = defaultdict(list)
    for r in uniq:
        groups[r["base_layout_signature"]].append(r)
    keys = sorted(groups)
    random.Random(a.rng_seed).shuffle(keys)
    need = {"test": a.test, "validation": a.val, "train": a.train}
    split_rows = {"test": [], "validation": [], "train": []}
    for k in keys:
        for s in ("test", "validation", "train"):          # fill held-out splits first
            if len(split_rows[s]) < need[s]:
                split_rows[s].extend(groups[k])
                break
    for s in split_rows:
        split_rows[s] = split_rows[s][:need[s]] if len(split_rows[s]) > need[s] else split_rows[s]
    report = [f"candidates: {sum(reasons.values())}  verified ok: {len(rows)}  unique configurations: {len(uniq)}  base layouts: {len(groups)}",
              f"rejection reasons: {dict(reasons)}"]
    # assign ids + write
    for s, lst in split_rows.items():
        out = []
        for i, r in enumerate(lst, 1):
            rec = {k: v for k, v in r.items() if k not in ("ok", "reason", "permille")}
            rec["task_id"] = f"gap1_{s}_{i:06d}"
            rec["split"] = s
            out.append(rec)
        path = os.path.join(OUT, f"{s}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for rec in out:
                f.write(json.dumps(rec) + "\n")
        split_rows[s] = out
        report.append(f"{s}: {len(out)} configurations -> {os.path.relpath(path, paths.ROOT)}")
    # disjointness checks
    sigs = {s: {r["layout_signature"] for r in lst} for s, lst in split_rows.items()}
    bases = {s: {r["base_layout_signature"] for r in lst} for s, lst in split_rows.items()}
    seeds = {s: {r["game_seed"] for r in lst} for s, lst in split_rows.items()}
    for x, y in (("train", "validation"), ("train", "test"), ("validation", "test")):
        report.append(f"overlap {x}/{y}: configurations {len(sigs[x] & sigs[y])}, base layouts {len(bases[x] & bases[y])}, game seeds {len(seeds[x] & seeds[y])}")
    # distributions
    def hist(vals, bins=None):
        c = Counter(vals)
        return dict(sorted(c.items()))
    for s, lst in split_rows.items():
        idx = [r["a_star"]["action_index"] for r in lst]
        top = Counter(idx).most_common(5)
        report.append(f"\n[{s}] n={len(lst)}")
        report.append(f"  distinct a* action indices: {len(set(idx))}; top-5 shares: " + ", ".join(f"{i}:{n/len(lst):.1%}" for i, n in top))
        report.append(f"  orientation: {hist(r['gap_orientation'] for r in lst)}")
        xs = [r["a_star"]["x"] for r in lst]; ys = [r["a_star"]["y"] for r in lst]
        report.append(f"  gap x range {min(xs)}..{max(xs)} (distinct {len(set(xs))}); gap y range {min(ys)}..{max(ys)} (distinct {len(set(ys))})")
        report.append(f"  gap x histogram: {hist(xs)}")
        report.append(f"  gap y histogram: {hist(ys)}")
        report.append(f"  path length (edits): {hist(r['path_length_edits'] for r in lst)}")
        report.append(f"  gap position along path (edit index): {hist(r['gap_position_in_path'] for r in lst)}")
        report.append(f"  house x range {min(r['house'][0] for r in lst)}..{max(r['house'][0] for r in lst)}, house y range {min(r['house'][1] for r in lst)}..{max(r['house'][1] for r in lst)}")
    open(os.path.join(OUT, "distribution_report.txt"), "w", encoding="utf-8").write("\n".join(report) + "\n")
    with open(os.path.join(OUT, "SHA256.txt"), "w", encoding="utf-8") as f:
        for s in split_rows:
            f.write(f"{sha(os.path.join(OUT, s + '.jsonl'))}  {s}.jsonl\n")
    # figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    cols = {"train": "tab:blue", "validation": "tab:orange", "test": "tab:green"}
    for s, lst in split_rows.items():
        ax[0, 0].hist([r["a_star"]["action_index"] for r in lst], bins=40, alpha=.5, label=s, color=cols[s])
        ax[0, 1].hist([r["a_star"]["x"] for r in lst], bins=range(-19, 19), alpha=.5, label=s, color=cols[s])
        ax[0, 2].hist([r["a_star"]["y"] for r in lst], bins=range(-11, 11), alpha=.5, label=s, color=cols[s])
        ax[1, 1].hist([r["path_length_edits"] for r in lst], bins=range(4, 20), alpha=.5, label=s, color=cols[s])
        ax[1, 2].hist([r["gap_position_in_path"] for r in lst], bins=range(0, 20), alpha=.5, label=s, color=cols[s])
    for s, lst in split_rows.items():
        h = sum(1 for r in lst if r["gap_orientation"] == "horizontal") / max(len(lst), 1)
        ax[1, 0].bar([s], [h], color=cols[s])
    for a_, t in zip(ax.flat, ["correct action index a*", "gap x", "gap y", "share horizontal gaps", "path length (edits)", "gap position along path"]):
        a_.set_title(t)
    ax[0, 0].legend(); ax[1, 0].set_ylim(0, 1)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "distributions.png"), dpi=110)
    print("\n".join(report))


if __name__ == "__main__":
    main()
