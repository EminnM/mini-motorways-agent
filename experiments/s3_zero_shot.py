"""S3-A: zero-shot evaluation of the three frozen S1 Gap-1 CNN checkpoints on the frozen synthetic Gap-2 TEST set. No gradient updates.
Run from repo root:  python experiments/s3_zero_shot.py"""
import hashlib
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, ROOT)
from sb3_contrib import MaskablePPO  # noqa: E402
from synthetic_rl.env import load_tasks  # noqa: E402
from synthetic_rl.gap2_eval import evaluate_rich, summarize_rich  # noqa: E402
from synthetic_rl.models import SpatialMaskablePolicy  # noqa: E402,F401

OUT = os.path.join(ROOT, "research", "s3_gap2")
CK = {11: "2026-10-08_140406_syn_s1_cnn_s11/checkpoints/step_050000.zip",
      12: "2026-10-08_140406_syn_s1_cnn_s12/checkpoints/step_050000.zip",
      13: "2026-10-08_140406_syn_s1_cnn_s13/checkpoints/final.zip"}
SEEDS = [0, 1, 2]  # paired eval seeds for stochastic / random


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    os.makedirs(OUT, exist_ok=True)
    tasks = load_tasks(os.path.join(ROOT, "research", "synthetic_lab", "manifests", "gap2_test.jsonl"))
    res = {"task_set": "gap2_test (200 tasks)", "window_decisions": 20, "baselines": {}, "checkpoints": {}}
    raw = {}
    t0 = time.time()
    for name in ("wait", "oracle"):
        rows = evaluate_rich(name, tasks)
        res["baselines"][name] = summarize_rich(rows)
    rr = [summarize_rich(evaluate_rich("random", tasks, seed=s)) for s in SEEDS]
    res["baselines"]["random_legal"] = {"per_seed_completion": [r["completion"]["rate"] for r in rr], "seed0_summary": rr[0],
                                        "mean_completion": sum(r["completion"]["rate"] for r in rr) / len(rr)}
    for seed, rel in CK.items():
        p = os.path.join(ROOT, "runs", rel)
        model = MaskablePPO.load(p, device="cpu")
        e = {"path": rel, "sha256": sha(p)}
        rows = evaluate_rich("model", tasks, model=model, deterministic=True, seed=0)
        e["deterministic"] = summarize_rich(rows)
        raw[f"s{seed}_det"] = rows
        sto = []
        for s in SEEDS:
            r = evaluate_rich("model", tasks, model=model, deterministic=False, seed=s)
            sto.append(summarize_rich(r))
            raw[f"s{seed}_sto{s}"] = r
        e["stochastic_seed_summaries"] = sto
        e["stochastic_completion_mean"] = sum(x["completion"]["rate"] for x in sto) / len(sto)
        res["checkpoints"][f"cnn_s{seed}"] = e
        print(f"cnn_s{seed}: det completion {e['deterministic']['completion']['rate']:.3f}, first-useful {e['deterministic']['first_useful_action_accuracy']['rate']:.3f}, "
              f"partial {e['deterministic']['partial_progress_at_least_one_edge']['rate']:.3f}, stoch {e['stochastic_completion_mean']:.3f}", flush=True)
    res["wall_clock_s"] = time.time() - t0
    json.dump(res, open(os.path.join(OUT, "s3a_zero_shot.json"), "w"), indent=1)
    json.dump(raw, open(os.path.join(OUT, "s3a_zero_shot_rows.json"), "w"))
    print("wait", res["baselines"]["wait"]["completion"]["rate"], "oracle", res["baselines"]["oracle"]["completion"]["rate"], "random", res["baselines"]["random_legal"]["mean_completion"])


if __name__ == "__main__":
    main()
