"""python -m synthetic_rl.make_tasks --n-gaps 1 [--train 2000 --val 200 --test 400 --seed 0]
Writes frozen, hashed task manifests to research/synthetic_lab/manifests/."""
import argparse
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from synthetic_rl import generator as G  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "research", "synthetic_lab", "manifests")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-gaps", type=int, default=1)
    ap.add_argument("--train", type=int, default=2000)
    ap.add_argument("--val", type=int, default=200)
    ap.add_argument("--test", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    sets = G.make_split_sets(a.n_gaps, a.train, a.val, a.test, a.seed)
    lines = []
    for s, tasks in sets.items():
        p = os.path.join(OUT, f"gap{a.n_gaps}_{s}.jsonl")
        with open(p, "w", encoding="utf-8") as f:
            for t in tasks:
                f.write(json.dumps(t) + "\n")
        h = hashlib.sha256(open(p, "rb").read()).hexdigest()
        lines.append(f"{h}  gap{a.n_gaps}_{s}.jsonl  ({len(tasks)} tasks, {len({t['base_signature'] for t in tasks})} base layouts)")
    with open(os.path.join(OUT, f"gap{a.n_gaps}_SHA256.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines), f"\n{time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
