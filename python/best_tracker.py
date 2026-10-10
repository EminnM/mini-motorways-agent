"""
Best-checkpoint tracker. Watches one or more checkpoint directories; each NEW
checkpoint is evaluated (deterministic policy) on the VALIDATION seed set on a
dedicated headless game instance, the held-out validation curve is appended to
a jsonl, and the best checkpoint overall (by mean return, then mean score) is
published atomically to best_models/overall_best.{json,zip}. The stream
showcase plays THAT file -- the best model, not the latest.

Selection uses VAL_SEEDS only; the final report (phase11a_runner --seeds
heldout) uses the disjoint TEST seeds, so there is no selection bias.
"""

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phase11a_runner as pr  # noqa: E402
import train_ppo  # noqa: E402  (registers GridExtractor for CNN checkpoints)
from eval_seeds import VAL_SEEDS  # noqa: E402
from mini_motorways_env import IpcClient  # noqa: E402
from sb3_contrib import MaskablePPO  # noqa: E402

ROOT = r"C:\Users\TKA\Desktop\mini moto"
IPC = os.path.join(ROOT, "logs", "ipc_best")
LOG = os.path.join(ROOT, "logs", "instrumentation_best.log")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_models")


def launch_game(timeout=150):
    os.makedirs(IPC, exist_ok=True)
    for f in ("request.json", "response.json"):
        try:
            os.remove(os.path.join(IPC, f))
        except OSError:
            pass
    open(LOG, "w").close()
    env = os.environ.copy()
    env["MINI_MOTORWAYS_IPC_DIR"] = IPC
    env["MINI_MOTORWAYS_LOG_PATH"] = LOG
    subprocess.Popen([train_ppo.GAME_EXE, "-batchmode", "-nographics"], cwd=train_ppo.GAME_DIR, env=env)
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if "File-based IPC ready" in open(LOG, encoding="utf-8", errors="replace").read():
                return True
        except OSError:
            pass
        time.sleep(2)
    return False


def steps_of(path):
    m = re.search(r"checkpoint_(\d+)\.zip$", path)
    return int(m.group(1)) if m else -1


def atomic_copy(src, dst):
    tmp = dst + ".tmp"
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


def atomic_json(obj, dst):
    tmp = dst + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, dst)


def evaluate(client, model, args):
    pr.PPO_MODEL = model
    pr.STOCHASTIC = False
    rng = np.random.default_rng(0)
    eps = [pr.run_episode(client, sd, "ppo", args.start_mode, args.max_decisions, args.decision_delta, rng)
           for sd in VAL_SEEDS]
    return {
        "valMeanReturn": float(np.mean([e["return"] for e in eps])),
        "valMeanScore": float(np.mean([e["finalScore"] or 0 for e in eps])),
        "valMeanLength": float(np.mean([e["length"] for e in eps])),
        "valTerminalRate": float(np.mean([e["terminated"] for e in eps])),
        "valFractionPositive": float(np.mean([e["nScoreEvents"] > 0 for e in eps])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True, help="label=checkpoint_dir pairs, e.g. mlp=ppo_ckpt_connected_mlp")
    ap.add_argument("--start-mode", default="CONNECTED_START")
    ap.add_argument("--max-decisions", type=int, default=600)
    ap.add_argument("--decision-delta", type=float, default=1.0)
    ap.add_argument("--poll", type=float, default=30)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    dirs = dict(d.split("=", 1) for d in args.dirs)
    print(f"[best] launching evaluation instance; watching {dirs}", flush=True)
    if not launch_game():
        raise SystemExit("evaluation instance did not come up")
    client = IpcClient(request_path=os.path.join(IPC, "request.json"),
                       response_path=os.path.join(IPC, "response.json"), timeout=120)

    done = set()
    curve_path = {lab: os.path.join(OUT, f"{lab}_val_curve.jsonl") for lab in dirs}
    best = None
    best_path = os.path.join(OUT, "overall_best.json")
    if os.path.exists(best_path):
        best = json.load(open(best_path))
    # resume: never re-evaluate what the curve files already hold
    for lab, cp in curve_path.items():
        if os.path.exists(cp):
            for line in open(cp):
                try:
                    done.add((lab, json.loads(line)["checkpointSteps"]))
                except Exception:
                    pass

    while True:
        new = []
        for lab, d in dirs.items():
            for z in glob.glob(os.path.join(d, "checkpoint_*.zip")):
                if (lab, steps_of(z)) not in done:
                    new.append((steps_of(z), lab, z))
        new.sort()
        if not new:
            time.sleep(args.poll)
            continue
        steps, lab, z = new[0]
        time.sleep(2)  # let the writer finish the file
        try:
            model = MaskablePPO.load(z, custom_objects={"GridExtractor": train_ppo.GridExtractor})
            t0 = time.time()
            res = evaluate(client, model, args)
        except Exception as e:  # game died mid-eval: relaunch and retry this checkpoint
            print(f"[best] eval of {lab}@{steps} failed ({e}); relaunching evaluation instance", flush=True)
            if not launch_game():
                raise
            client = IpcClient(request_path=os.path.join(IPC, "request.json"),
                               response_path=os.path.join(IPC, "response.json"), timeout=120)
            continue
        done.add((lab, steps))
        rec = {"label": lab, "checkpointSteps": steps, "checkpoint": z, "evaluatedAt": time.time(),
               "evalSeconds": round(time.time() - t0, 1), **res}
        with open(curve_path[lab], "a") as f:
            f.write(json.dumps(rec) + "\n")
        better = best is None or (res["valMeanReturn"], res["valMeanScore"]) > (best["valMeanReturn"], best["valMeanScore"])
        print(f"[best] {lab}@{steps}: valReturn={res['valMeanReturn']:+.2f} valScore={res['valMeanScore']:.2f} "
              f"len={res['valMeanLength']:.0f} term={res['valTerminalRate']:.2f}  {'** NEW BEST **' if better else ''}", flush=True)
        if better:
            dst = os.path.join(OUT, "overall_best.zip")
            atomic_copy(z, dst)
            best = {**rec, "zip": dst}
            atomic_json(best, best_path)


if __name__ == "__main__":
    main()
