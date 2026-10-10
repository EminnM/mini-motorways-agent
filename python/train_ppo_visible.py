"""
Phase 10B: visible PPO training via restart-between-episodes.

Repeated in-process frontend re-attachment (pop to menu, re-attach
GameContainerScreen) was tested live and found to crash with a DIFFERENT
NullReferenceException on each attempt (ScreenStack.GetTransitionDetailsFrom,
ThemedComponent.ReleaseTheme, UnityEngine.Object.get_name) -- a genuinely
fragile, not-reliably-fixable-quickly area of GameContainerScreen's pooled-
reuse lifecycle. The first attach in a fresh process is proven reliable
(Phase 9A, and every episode-0 attempt here). So: one game process per
episode. Slower (game boot + stock-menu-wait overhead per episode, ~15-25s)
but has not failed once.

PPO policy weights persist ACROSS process restarts via save/load
(MaskablePPO.save/.load) and model.set_env(new_env) + learn(...,
reset_num_timesteps=False) -- a supported Stable-Baselines3 pattern for
continuing training with a different environment instance.
"""

import argparse
import os
import subprocess
import time

from sb3_contrib import MaskablePPO

from mini_motorways_env import MiniMotorwaysEnv

GAME_EXE = r"C:\Program Files (x86)\Steam\steamapps\common\Mini Motorways\Mini Motorways.exe"
GAME_DIR = r"C:\Program Files (x86)\Steam\steamapps\common\Mini Motorways"
LOG_PATH = r"C:\Users\TKA\Desktop\mini moto\logs\instrumentation.log"


def launch_game(timeout=40):
    if os.path.exists(LOG_PATH):
        try:
            os.remove(LOG_PATH)
        except OSError:
            pass
    proc = subprocess.Popen([GAME_EXE], cwd=GAME_DIR)
    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.exists(LOG_PATH):
            try:
                with open(LOG_PATH, "r", errors="ignore") as f:
                    content = f.read()
            except OSError:
                content = ""
            if "File-based IPC ready" in content:
                return proc
            if "EXCEPTION" in content:
                raise RuntimeError("Game startup threw an exception -- see instrumentation.log")
        time.sleep(1)
    raise TimeoutError(f"Game did not reach IPC-ready within {timeout}s")


def kill_game():
    subprocess.run(["taskkill", "/IM", "Mini Motorways.exe", "/F"], capture_output=True)
    time.sleep(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, required=True)
    ap.add_argument("--max-decisions", type=int, default=600)
    ap.add_argument("--checkpoint-dir", default="ppo_checkpoints_visible")
    ap.add_argument("--resume-from", default=None, help="Path to an existing checkpoint to continue training from")
    args = ap.parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    model = None
    if args.resume_from:
        print(f"Will resume from {args.resume_from} on the first episode.")

    for ep in range(args.episodes):
        print(f"\n=== Episode {ep + 1}/{args.episodes}: launching game ===", flush=True)
        try:
            launch_game()
        except Exception as e:
            print(f"Episode {ep + 1}: game launch failed: {e}")
            kill_game()
            continue

        try:
            env = MiniMotorwaysEnv(max_decisions=args.max_decisions)
            if model is None:
                if args.resume_from:
                    model = MaskablePPO.load(args.resume_from, env=env)
                else:
                    model = MaskablePPO(
                        "MlpPolicy", env,
                        n_steps=args.max_decisions,
                        batch_size=min(64, args.max_decisions),
                        n_epochs=10, gamma=0.99, gae_lambda=0.95,
                        clip_range=0.2, ent_coef=0.01, verbose=1, seed=7,
                    )
            else:
                model.set_env(env)

            model.learn(total_timesteps=args.max_decisions, reset_num_timesteps=False)
            ckpt = os.path.join(args.checkpoint_dir, f"checkpoint_ep{ep + 1}.zip")
            model.save(ckpt)
            print(f"Saved {ckpt}", flush=True)
            env.close()
        except Exception as e:
            print(f"Episode {ep + 1} FAILED: {e}", flush=True)
        finally:
            kill_game()

    if model is not None:
        final_path = os.path.join(args.checkpoint_dir, "final_model.zip")
        model.save(final_path)
        print(f"\nSaved final model to {final_path}")


if __name__ == "__main__":
    main()
