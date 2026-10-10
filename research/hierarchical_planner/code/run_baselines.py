"""S18-H baselines on the synthetic MultiHouseEnv (no native game, no PPO). Reuses s13_baselines.episode()/summarize() so the metrics are the
same as research/s13_multi_house/baselines.json. 36 scenarios x 3 env seeds (seeds 7000 + 100*idx + s, identical to S13).
  WAIT                      : action 0 always
  A_greedy_nearest          : Executor with greedy weights (cheapest-route house first, no resource / demand reasoning)
  B_resource_aware_planner  : Executor with the full scoring function
  C_motorway_planner        : UNTESTED - no executable motorway action exists (see MECHANICS.md); typed candidates only
Weights were NOT tuned on the test seeds (515838/539595/563352/587109); the only tuning was the greedy-sign fix found on two train/validation episodes.
python research/hierarchical_planner/code/run_baselines.py -> research/hierarchical_planner/baselines.json"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from world_model import WorldModel  # noqa: E402  (sets sys.path for synthetic_rl)
from executor import Executor  # noqa: E402
from planner import Weights  # noqa: E402
from synthetic_rl import s13_baselines as B  # noqa: E402
from synthetic_rl import s13_env as E  # noqa: E402

SC = B.SC
OUT = os.path.join(os.path.dirname(HERE), "baselines.json")
VAL_SEEDS = {507919, 579190}
TEST_SEEDS = {515838, 539595, 563352, 587109}


def split_of(row):
    gs = int(row["scenario"].split("_")[0])
    return "test" if gs in TEST_SEEDS else ("validation" if gs in VAL_SEEDS else "train")


def run(name, factory, idxs):
    rows, stats = [], []
    t0 = time.time()
    for idx in idxs:
        for s in range(B.ENV_SEEDS):
            ex = factory()
            pol = ex.act if ex is not None else (lambda env: 0)
            r = B.episode(pol, idx, 7000 + 100 * idx + s)
            r["split3"] = split_of(r)
            if ex is not None:
                r["abandoned_plans"] = ex.stats["abandoned"]
                r["plans"] = ex.stats["plans"]
                r["compute_ms_per_decision"] = round(1000 * ex.stats["compute_s"] / max(1, ex.stats["decisions"]), 2)
                r["abandon_reasons"] = ex.stats["abandon_reasons"]
                r["waits_for_concrete"] = ex.stats["waits_for_concrete"]
            rows.append(r)
    out = {"all": B.summarize(rows), "test_split": B.summarize([r for r in rows if r["split3"] == "test"]), "validation_split": B.summarize([r for r in rows if r["split3"] == "validation"]),
           "train_split": B.summarize([r for r in rows if r["split3"] == "train"]), "wall_s": round(time.time() - t0, 1)}
    if "abandoned_plans" in rows[0]:
        for k in ("all", "test_split", "validation_split", "train_split"):
            sub = [r for r in rows if k == "all" or r["split3"] == k.replace("_split", "")]
            out[k]["abandoned_plans_total"] = sum(r["abandoned_plans"] for r in sub)
            out[k]["plans_total"] = sum(r["plans"] for r in sub)
            out[k]["compute_ms_per_decision_median"] = round(float(np.median([r["compute_ms_per_decision"] for r in sub])), 2)
            out[k]["compute_ms_per_decision_max"] = round(float(max(r["compute_ms_per_decision"] for r in sub)), 2)
            out[k]["waits_for_concrete_total"] = sum(r["waits_for_concrete"] for r in sub)
            reasons = {}
            for r in sub:
                for kk, vv in r["abandon_reasons"].items():
                    reasons[kk] = reasons.get(kk, 0) + vv
            out[k]["abandon_reasons"] = reasons
        out["all"]["deliveries_total"] = sum(r["deliveries"] for r in rows)
    out["episodes"] = rows
    print(name, json.dumps({k: v for k, v in out["all"].items()}), flush=True)
    return out


def main():
    idxs = list(range(len(SC)))
    res = {}
    res["WAIT"] = run("WAIT", lambda: None, idxs)
    res["A_greedy_nearest"] = run("A_greedy_nearest", lambda: Executor(Weights(greedy=True)), idxs)
    res["B_resource_aware_planner"] = run("B_resource_aware_planner", lambda: Executor(Weights()), idxs)
    res["C_motorway_planner"] = {"status": "UNTESTED", "reason": "no executable motorway action in Discrete(3111) (ActionKind = Wait/Add/Remove/ChooseUpgrade, Entrypoint.cs:2469); candidates are typed with feasibility='unsupported'"}
    # reference numbers from S13 (same episode() metric code, same seeds)
    ref = json.load(open(os.path.join(os.path.dirname(os.path.dirname(HERE)), "s13_multi_house", "baselines.json")))
    res["reference_S13"] = {k: {"all": v["all"], "test_split": v["test_split"], "train_split": v["train_split"]} for k, v in ref.items()}
    json.dump({k: ({kk: vv for kk, vv in v.items() if kk != "episodes"} if isinstance(v, dict) else v) for k, v in res.items()}, open(OUT, "w"), indent=1)
    json.dump({k: v["episodes"] for k, v in res.items() if isinstance(v, dict) and "episodes" in v}, open(OUT.replace(".json", "_episodes.json"), "w"))


if __name__ == "__main__":
    main()
