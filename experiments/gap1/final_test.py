"""Evaluate the checkpoint FROZEN by evaluation/selection.json (chosen on VALIDATION only) on the TEST manifest.
  python -m experiments.gap1.final_test --run runs/<run_id> [--stochastic]
Refuses to run if the selected checkpoint's SHA-256 no longer matches, or if test results already exist."""
import argparse
import json
import os
import sys

import yaml

from .. import paths
from .. import gamectl
from . import evaluation as E


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m experiments.gap1.final_test")
    ap.add_argument("--run", required=True, help="run directory (or seeds/seed_x sub-run)")
    ap.add_argument("--ipc-dir", default=None)
    ap.add_argument("--stochastic", action="store_true", help="additionally sample actions (training-time behaviour)")
    a = ap.parse_args(argv)
    sd = os.path.abspath(a.run)
    cfg_dir = sd
    while not os.path.exists(os.path.join(cfg_dir, "config.yaml")):
        cfg_dir = os.path.dirname(cfg_dir)
    cfg = yaml.safe_load(open(os.path.join(cfg_dir, "config.yaml"), encoding="utf-8"))
    g = cfg["environment"]["gap1"]
    sel = json.load(open(os.path.join(sd, "evaluation", "selection.json"), encoding="utf-8"))
    ck = os.path.join(sd, "checkpoints", sel["selected_file"])
    from ..train_worker import sha256_file
    if sha256_file(ck) != sel["sha256"]:
        raise SystemExit("selected checkpoint hash does not match selection.json")
    run_id = os.path.basename(cfg_dir)
    results = {}
    inst = None
    try:
        if a.ipc_dir:
            ipc = a.ipc_dir
        else:
            inst = gamectl.GameInstance(f"{run_id[-20:]}_gap1test".replace("-", "").replace("_", "")).start()
            ipc = inst.ipc_dir
        for det in ([True] + ([False] if a.stochastic else [])):
            out = os.path.join(sd, "evaluation", "test_selected" + ("" if det else "_stochastic"))
            if os.path.exists(os.path.join(out, "results.json")):
                raise SystemExit(f"{out} already exists; TEST is evaluated once per frozen selection")
            res = E.evaluate_to_dir("ppo", ck, g["test_split"] if "test_split" in g else "test", g["manifest_dir"], ipc, out,
                                    cfg["environment"]["decision_interval_seconds"], g["construction_window"],
                                    g["continuation_decisions"], det, 0, continue_after_success=True,
                                    extra={"selection": sel, "run_id": run_id}, instance=inst)
            results["deterministic" if det else "stochastic"] = res
            print(json.dumps({k: res[k] for k in ("gap_completion_rate", "first_action_accuracy", "median_steps_to_success",
                                                  "destructive_failure_rate", "timeout_rate")}, indent=1, default=str))
    finally:
        if inst:
            inst.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
