"""Gap-1 evaluation CLI (never trains) on a frozen manifest split.
  python -m experiments.gap1.evaluate_cli --policy wait|random --split test
  python -m experiments.gap1.evaluate_cli --checkpoint <zip> --split test [--stochastic]
Creates an immutable run directory (kind evaluation) with config, metadata (hashes, versions, manifest hash),
episodes.csv (one row per configuration) and results.json."""
import argparse
import json
import os
import sys

import yaml

from .. import paths
from .. import metadata as md
from .. import runlayout as rl
from .. import gamectl
from . import evaluation as E


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m experiments.gap1.evaluate_cli")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--policy", choices=["wait", "random"])
    src.add_argument("--checkpoint")
    ap.add_argument("--split", default="test", help="manifest file stem in --manifest-dir (train | validation | test | custom e.g. s2_pilot)")
    ap.add_argument("--manifest-dir", default=os.path.join(paths.ROOT, "research", "gap1", "manifests"))
    ap.add_argument("--construction-window", type=int, default=20)
    ap.add_argument("--continuation", type=int, default=40)
    ap.add_argument("--delta", type=float, default=1.0)
    ap.add_argument("--stochastic", action="store_true")
    ap.add_argument("--random-seed", type=int, default=0)
    ap.add_argument("--no-continuation", action="store_true")
    ap.add_argument("--name", default=None)
    ap.add_argument("--runs-root", default=None)
    ap.add_argument("--ipc-dir", default=None)
    ap.add_argument("--effective-dll", default=None, help="path of the bootstrap DLL actually loaded by the attached instance (recorded in metadata)")
    a = ap.parse_args(argv)
    label = a.policy or os.path.splitext(os.path.basename(a.checkpoint))[0]
    run_id, run_dir = rl.new_run_dir(f"gap1_{a.name or label}_{a.split}", a.runs_root, kind="eval_")
    cfg = {"experiment": {"name": f"gap1_{label}_{a.split}", "description": "Gap-1 held-out evaluation", "hypothesis": "", "tags": ["gap1", "evaluation"]},
           "task": "gap1", "split": a.split, "manifest_dir": a.manifest_dir, "construction_window": a.construction_window,
           "continuation_decisions": 0 if a.no_continuation else a.continuation, "decision_interval_seconds": a.delta,
           "policy": a.policy or "ppo", "checkpoint": os.path.abspath(a.checkpoint) if a.checkpoint else None,
           "deterministic": not a.stochastic, "random_seed": a.random_seed}
    rl.write_yaml(os.path.join(run_dir, "config.yaml"), cfg)
    git = md.git_info(save_patch_to=os.path.join(run_dir, "logs", "git_diff.patch"))
    md.save_source_snapshot(os.path.join(run_dir, "logs", "source_snapshot.zip"))
    meta = {"run_id": run_id, "kind": "evaluation", "start_time": md.utc_now(), "status": "running", **git,
            **md.environment_snapshot(), "command": list(getattr(sys, "orig_argv", sys.argv))}
    if a.effective_dll:
        meta["bootstrap_dll_effective_path"] = a.effective_dll
        meta["bootstrap_dll_effective_sha256"] = md.sha256_file(a.effective_dll)
        meta["note_effective_dll"] = "evaluated on an attached instance that loads this DLL via the Doorstop target override, not the deployed one"
    md.write_json(os.path.join(run_dir, "metadata.json"), meta)
    inst = None
    try:
        if a.ipc_dir:
            ipc = a.ipc_dir
        else:
            inst = gamectl.GameInstance(f"g1e{run_id[-18:]}".replace("-", "").replace("_", "")).start()
            ipc = inst.ipc_dir
        res = E.evaluate_to_dir(a.policy or "ppo", a.checkpoint, a.split, a.manifest_dir, ipc, os.path.join(run_dir, "evaluation"),
                                a.delta, a.construction_window, a.continuation, not a.stochastic, a.random_seed,
                                continue_after_success=not a.no_continuation, extra={"evaluated_in_run_id": run_id}, instance=inst)
        meta["status"] = "completed"
        print(json.dumps({k: res[k] for k in ("policy", "split", "num_configurations", "gap_completion_rate", "first_action_accuracy",
                                              "median_steps_to_success", "destructive_failure_rate", "timeout_rate")}, indent=1, default=str))
    except Exception:
        meta["status"] = "failed"
        raise
    finally:
        if inst:
            inst.stop()
        meta["end_time"] = md.utc_now()
        md.write_json(os.path.join(run_dir, "metadata.json"), meta)
    print("run dir:", run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
