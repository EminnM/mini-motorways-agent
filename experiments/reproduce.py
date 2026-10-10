"""python -m experiments.reproduce runs/<run_id> [--check-only]
Loads the run's resolved config, compares the CURRENT code/DLL/game build/library versions with the recorded
metadata and WARNS about every difference, then launches a new run (it never claims exactness)."""
import argparse
import json
import os
import sys

import yaml

from . import paths
from . import metadata as md


def compare_environment(old: dict, new: dict):
    diffs = []
    for key in ("bootstrap_dll_deployed_sha256", "bootstrap_dll_built_sha256", "entrypoint_cs_sha256", "source_tree_sha256"):
        if old.get(key) != new.get(key):
            diffs.append((key, old.get(key), new.get(key)))
    og, ng = old.get("game") or {}, new.get("game") or {}
    for key in ("game_build_id", "unity_version", "game_exe_sha256"):
        if og.get(key) != ng.get(key):
            diffs.append((f"game.{key}", og.get(key), ng.get(key)))
    for k, v in (old.get("libraries") or {}).items():
        if (new.get("libraries") or {}).get(k) != v:
            diffs.append((f"libraries.{k}", v, (new.get("libraries") or {}).get(k)))
    for key in ("os", "cpu", "gpu_used_by_torch"):
        if old.get(key) != new.get(key):
            diffs.append((key, old.get(key), new.get(key)))
    oldf, newf = old.get("source_files") or {}, new.get("source_files") or {}
    changed = sorted(f for f in set(oldf) | set(newf) if oldf.get(f) != newf.get(f))
    return diffs, changed


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m experiments.reproduce")
    ap.add_argument("run")
    ap.add_argument("--check-only", action="store_true", help="only report differences, do not launch")
    ap.add_argument("--runs-root", default=None)
    a = ap.parse_args(argv)
    run = os.path.abspath(a.run)
    cfg_path = os.path.join(run, "config.yaml")
    meta = json.load(open(os.path.join(run, "metadata.json"), encoding="utf-8"))
    if meta.get("kind") == "evaluation":
        print("this is an evaluation-only run; re-run it with python -m experiments.evaluate", file=sys.stderr)
        return 2
    new = md.environment_snapshot()
    diffs, changed = compare_environment(meta, new)
    print(f"reproducing {meta['run_id']} (recorded {meta.get('start_time')})")
    if not diffs and not changed:
        print("OK: DLL, game build, library versions and all source files match the recorded run.")
    else:
        print("WARNING: this will NOT be an exact reproduction. Differences from the recorded run:")
        for k, o, n in diffs:
            print(f"  - {k}: recorded={o!r} now={n!r}")
        if changed:
            print(f"  - {len(changed)} source/config file(s) differ from the recorded snapshot:")
            for f in changed[:40]:
                print(f"      {f}")
        snap = os.path.join(run, "logs", "source_snapshot.zip")
        if os.path.exists(snap):
            print(f"  (the exact original sources are in {snap})")
    if meta.get("git_dirty"):
        print("  note: the original run had uncommitted changes (logs/git_diff.patch)")
    if a.check_only:
        return 0 if not diffs and not changed else 3
    from . import run as runmod
    args = ["--config", cfg_path, "--reproduces", meta["run_id"]]
    if a.runs_root:
        args += ["--runs-root", a.runs_root]
    return runmod.main(args)


if __name__ == "__main__":
    sys.exit(main())
