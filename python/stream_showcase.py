"""
Streaming showcase: continuously plays the BEST checkpoint (as ranked on the
held-out validation seeds by best_tracker.py -- not simply the latest) in its
OWN game instance (separate process + IPC channel via the MINI_MOTORWAYS_*
environment variables), realtime-paced so motion is smooth and watchable,
while the trainers keep training and the tracker keeps evaluating.

Same environment setup as training (shared code path, not a re-implementation):
  * same start mode (CONNECTED_START by default) -- the city the stream shows is
    set up exactly the way training episodes are, and the overlay says so;
  * same decision interval, episode horizon and observation encoding;
  * loads MLP *and* CNN checkpoints (GridExtractor registered via train_ppo);
  * deterministic (argmax) play by default, i.e. the policy exactly as scored.

Loop: read best_models/overall_best.json -> load that checkpoint -> play one
episode -> at episode end, if a different checkpoint is now best, hot-swap to it.
The window is pinned always-on-top; the game instance self-heals.
"""

import argparse
import ctypes
import ctypes.wintypes
import json
import os
import re
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

STREAM_DIR = r"C:\Users\TKA\Desktop\mini moto\logs\ipc_stream"
STREAM_LOG = r"C:\Users\TKA\Desktop\mini moto\logs\instrumentation_stream.log"
GAME_EXE = r"C:\Program Files (x86)\Steam\steamapps\common\Mini Motorways\Mini Motorways.exe"
GAME_DIR = r"C:\Program Files (x86)\Steam\steamapps\common\Mini Motorways"
BEST_JSON = os.path.join(HERE, "best_models", "overall_best.json")

os.makedirs(STREAM_DIR, exist_ok=True)

import train_ppo  # noqa: E402  (GridExtractor for CNN checkpoints)
from mini_motorways_env import IpcClient, encode_observation  # noqa: E402
from sb3_contrib import MaskablePPO  # noqa: E402


def pin_stream_window(pid, x=940, y=40, w=960, h=640):
    """Fixed position, ALWAYS ON TOP: it is the live stream source."""
    user32 = ctypes.windll.user32
    found = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)
    def cb(hwnd, _):
        wp = ctypes.wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(wp))
        if wp.value == pid and user32.IsWindowVisible(hwnd):
            found.append(hwnd)
        return True

    for _ in range(30):
        found.clear()
        user32.EnumWindows(cb, 0)
        if found:
            user32.SetWindowPos(found[0], -1, x, y, w, h, 0x0040)  # HWND_TOPMOST
            return True
        time.sleep(1)
    return False


def kill_previous_stream_game():
    """Kill only a PREVIOUS stream game, identified by the PID in its own log."""
    try:
        m = re.search(r"Process ID: (\d+)", open(STREAM_LOG, encoding="utf-8", errors="replace").read())
        if m:
            subprocess.run(["taskkill", "/PID", m.group(1), "/F"], capture_output=True)
            time.sleep(2)
    except OSError:
        pass


def launch_stream_game(timeout=150):
    kill_previous_stream_game()
    try:
        open(STREAM_LOG, "w").close()
    except OSError:
        pass
    for f in ("request.json", "response.json"):
        try:
            os.remove(os.path.join(STREAM_DIR, f))
        except OSError:
            pass
    env = os.environ.copy()
    env["MINI_MOTORWAYS_IPC_DIR"] = STREAM_DIR
    env["MINI_MOTORWAYS_LOG_PATH"] = STREAM_LOG
    proc = subprocess.Popen([GAME_EXE], cwd=GAME_DIR, env=env)
    start = time.time()
    while time.time() - start < timeout:
        try:
            if "File-based IPC ready" in open(STREAM_LOG, encoding="utf-8", errors="replace").read():
                pin_stream_window(proc.pid)
                return True
        except OSError:
            pass
        time.sleep(2)
    return False


def read_best():
    try:
        return json.load(open(BEST_JSON))
    except (OSError, ValueError):
        return None


def load_model(best):
    return MaskablePPO.load(best["zip"], custom_objects={"GridExtractor": train_ppo.GridExtractor})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-mode", choices=["NORMAL_START", "CONNECTED_START"], default="CONNECTED_START")
    ap.add_argument("--max-decisions", type=int, default=600)
    ap.add_argument("--decision-delta", type=float, default=1.0)
    ap.add_argument("--stochastic", action="store_true", help="sample the policy instead of argmax")
    args = ap.parse_args()

    print("[stream] waiting for the first evaluated (best) checkpoint...", flush=True)
    best = read_best()
    while best is None:
        time.sleep(15)
        best = read_best()

    print("[stream] launching dedicated game instance...", flush=True)
    if not launch_stream_game():
        raise SystemExit("stream game instance did not come up")
    client = IpcClient(request_path=os.path.join(STREAM_DIR, "request.json"),
                       response_path=os.path.join(STREAM_DIR, "response.json"), timeout=120)

    loaded_key, model = None, None
    rng = np.random.default_rng()
    episode = 0
    while True:
        best = read_best() or best
        key = (best["label"], best["checkpointSteps"])
        if key != loaded_key:
            model = load_model(best)
            loaded_key = key
            print(f"[stream] now playing BEST: {best['label']}@{best['checkpointSteps']} "
                  f"(val return {best['valMeanReturn']:+.2f}, val score {best['valMeanScore']:.1f})", flush=True)

        episode += 1
        seed = int(rng.integers(1, 2**31 - 1))
        try:
            resp = client.send("RESET", seed=seed, max_decisions=args.max_decisions,
                               decision_delta=args.decision_delta, start_mode=args.start_mode)
            if "error" in resp:
                raise RuntimeError(resp["error"])
            n = resp["actionSpaceSize"]
            ep_return, steps = 0.0, 0
            while True:
                obs_vec = encode_observation(resp["observation"])
                mask = np.zeros(n, dtype=bool)
                mask[resp["legalActionIndices"]] = True
                action, _ = model.predict(obs_vec, action_masks=mask, deterministic=not args.stochastic)
                overlay = {
                    f"BEST MODEL  {best['label'].upper()} @ {best['checkpointSteps']} steps":
                        f"val return {best['valMeanReturn']:+.1f}  val score {best['valMeanScore']:.1f}",
                    "start mode": args.start_mode,
                    "episode": f"{episode}   step {steps}   return {ep_return:+.1f}",
                }
                resp = client.send("STEP", action=int(action), realtime=True, overlay=overlay)
                if "error" in resp:
                    raise RuntimeError(resp["error"])
                ep_return += float(resp["reward"])
                steps += 1
                if resp.get("terminated") or resp.get("truncated"):
                    g = resp["observation"]["globals"]
                    print(f"[stream] ep{episode} ({best['label']}@{best['checkpointSteps']}): "
                          f"{'GAME OVER' if resp.get('terminated') else 'survived'} at step {steps}, "
                          f"day {g['day']}, score {g['score']}, return {ep_return:+.1f}", flush=True)
                    break
        except (RuntimeError, TimeoutError, OSError) as e:
            print(f"[stream] episode failed ({e}); relaunching stream game...", flush=True)
            if not launch_stream_game():
                raise


if __name__ == "__main__":
    main()
