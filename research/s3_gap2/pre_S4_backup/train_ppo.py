"""
Phase 10: train one simple MaskablePPO agent against the native Mini
Motorways environment over the file-based IPC bridge.

No hyperparameter search -- these are reasonable, documented, off-the-shelf
PPO settings for a small discrete-action-masked MLP policy on a single CPU
environment. ONE environment only (no vectorization/parallelism -- out of
scope for this phase, and also a hard requirement of the IPC transport: the
C# bootstrap exposes exactly one shared request.json/response.json pair, so
two concurrent MiniMotorwaysEnv clients would race on the same files --
confirmed live when an earlier version of this script tried a second
"eval_env" instance alongside the training env). Checkpoints periodically
during training; deterministic evaluation against the fixed, held-out
EVAL_SEEDS is done as a SEPARATE pass over the saved checkpoints AFTER
training finishes, reusing the same single env instance (training has
stopped touching it by then), for exactly the same single-environment
reason.
"""

import argparse
import json
import os
import re
import subprocess
import time

import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn

from mini_motorways_env import MiniMotorwaysEnv

TRAIN_SEED = 7
EVAL_SEEDS = [9101, 9102, 9103, 9104, 9105]

GAME_EXE = r"C:\Program Files (x86)\Steam\steamapps\common\Mini Motorways\Mini Motorways.exe"
GAME_DIR = r"C:\Program Files (x86)\Steam\steamapps\common\Mini Motorways"
DEFAULT_IPC_DIR = r"C:\Users\TKA\Desktop\mini moto\logs\ipc"
DEFAULT_LOG = r"C:\Users\TKA\Desktop\mini moto\logs\instrumentation.log"


class Instance:
    """The game instance THIS trainer owns (several trainers can run side by
    side, each with its own IPC dir + log; none may touch the others')."""
    ipc_dir = DEFAULT_IPC_DIR
    log = DEFAULT_LOG
    headless = False
    proc = None


class Inst:
    """An additional (headless) game instance owned by this trainer."""
    def __init__(self, ipc_dir, log):
        self.ipc_dir, self.log, self.headless, self.proc = ipc_dir, log, True, None


INSTS = [Instance]  # instance 0 = the (possibly windowed) default/primary instance


def kill_instance(inst):
    """Kill ONLY this instance, identified by the PID its own bootstrap wrote
    into its own log (never a blanket taskkill /IM)."""
    try:
        txt = open(inst.log, encoding="utf-8", errors="replace").read()
        m = re.search(r"Process ID: (\d+)", txt)
        if m:
            subprocess.run(["taskkill", "/PID", m.group(1), "/F"], capture_output=True)
    except OSError:
        pass
    if inst.proc is not None:
        try:
            inst.proc.kill()
        except Exception:
            pass


def launch_instances(insts, timeout=240):
    """(Re)launch the given instances IN PARALLEL and wait until every one's IPC
    bridge is ready. The bootstrap keeps a pending request.json across restarts
    and only answers it after init, so a relaunch mid-training is recoverable."""
    import shutil
    for inst in insts:
        kill_instance(inst)
    time.sleep(3)
    for inst in insts:
        try:  # rotate (never clear): the previous log is the evidence for WHY
            shutil.copyfile(inst.log, inst.log + ".prev")
        except OSError:
            pass
        try:
            open(inst.log, "w").close()
        except OSError:
            pass
        os.makedirs(inst.ipc_dir, exist_ok=True)
        for f in ("response.json",):
            try:
                os.remove(os.path.join(inst.ipc_dir, f))
            except OSError:
                pass
        env = os.environ.copy()
        env["MINI_MOTORWAYS_IPC_DIR"] = inst.ipc_dir
        env["MINI_MOTORWAYS_LOG_PATH"] = inst.log
        cmd = [GAME_EXE] + (["-batchmode", "-nographics"] if inst.headless else [])
        inst.proc = subprocess.Popen(cmd, cwd=GAME_DIR, env=env)
    start = time.time()
    pending = list(insts)
    while pending and time.time() - start < timeout:
        still = []
        for inst in pending:
            try:
                with open(inst.log, encoding="utf-8", errors="replace") as f:
                    if "File-based IPC ready" in f.read():
                        continue
            except OSError:
                pass
            still.append(inst)
        pending = still
        if pending:
            time.sleep(2)
    return not pending


def relaunch_game(timeout=240):
    """(Re)launch every game instance this trainer owns."""
    return launch_instances(INSTS, timeout)


def make_env(args):
    return MiniMotorwaysEnv(
        max_decisions=args.max_decisions, decision_delta=args.decision_delta,
        request_path=os.path.join(Instance.ipc_dir, "request.json"),
        response_path=os.path.join(Instance.ipc_dir, "response.json"),
        start_mode=args.start_mode, start_gap=args.start_gap,
        shaping=args.shaping, shaping_gamma=args.gamma, shaping_flip_y=args.shaping_flip_y)


def make_training_env(args):
    if args.n_envs <= 1:
        return make_env(args)
    from functools import partial
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from mini_motorways_env import make_monitored_env
    fns = [partial(make_monitored_env, inst.ipc_dir, args.max_decisions, args.decision_delta, args.start_mode, args.start_gap,
                  args.shaping, args.gamma, args.shaping_flip_y)
           for inst in INSTS]
    return SubprocVecEnv(fns, start_method="spawn")


class GridExtractor(BaseFeaturesExtractor):
    """CNN over the tile grid: the flat observation is 777 tiles x 15 features
    (row-major, y outer / x inner, 21 x 37) followed by the 11 global scalars.
    Reshape to (15, 21, 37), convolve (so 'is this road connected to that
    building' is a local, translation-invariant pattern), concatenate the
    globals, project to features_dim. Policy/value heads are unchanged."""

    def __init__(self, observation_space, features_dim=256, h=21, w=37, c=15):
        super().__init__(observation_space, features_dim)
        self.h, self.w, self.c = h, w, c
        self.cnn = nn.Sequential(
            nn.Conv2d(c, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten())
        with torch.no_grad():
            n = self.cnn(torch.zeros(1, c, h, w)).shape[1]
        n_glob = observation_space.shape[0] - h * w * c
        self.fc = nn.Sequential(nn.Linear(n + n_glob, features_dim), nn.ReLU())

    def forward(self, obs):
        b = obs.shape[0]
        tiles = obs[:, : self.h * self.w * self.c].reshape(b, self.h, self.w, self.c).permute(0, 3, 1, 2)
        glob = obs[:, self.h * self.w * self.c:]
        return self.fc(torch.cat([self.cnn(tiles), glob], dim=1))


def run_eval_episodes(env: MiniMotorwaysEnv, model, seeds):
    returns, score_deltas, lengths = [], [], []
    wait_c = add_c = remove_c = 0
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        terminated = truncated = False
        ep_return = 0.0
        ep_score = 0.0
        steps = 0
        while not (terminated or truncated):
            mask = env.action_masks()
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            ep_return += reward
            ep_score += info["rewardComponents"]["scoreDelta"]
            applied = info["appliedAction"]
            if applied == "WAIT":
                wait_c += 1
            elif applied.startswith("ADD_ROAD"):
                add_c += 1
            elif applied.startswith("REMOVE_ROAD"):
                remove_c += 1
            steps += 1
        returns.append(ep_return)
        score_deltas.append(ep_score)
        lengths.append(steps)
    return {
        "meanReturn": float(np.mean(returns)),
        "meanScoreDelta": float(np.mean(score_deltas)),
        "meanLength": float(np.mean(lengths)),
        "wait": wait_c, "add": add_c, "remove": remove_c,
    }


class CheckpointCallback(BaseCallback):
    def __init__(self, checkpoint_dir: str, checkpoint_every_steps: int, verbose: int = 1):
        super().__init__(verbose)
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_every_steps = checkpoint_every_steps
        self._checkpoints_done = 0
        self.saved_paths = []
        self._sps_t0 = time.time()
        self._sps_n0 = 0

    def _push_overlay(self):
        # What the model "thinks" right now: its value estimate for the state it
        # just acted in, the log-prob of the chosen action, plus run telemetry.
        try:
            v = self.locals.get("values")
            lp = self.locals.get("log_probs")
            now = time.time()
            sps = (self.num_timesteps - self._sps_n0) / max(now - self._sps_t0, 1e-6)
            if now - self._sps_t0 > 30:
                self._sps_t0, self._sps_n0 = now, self.num_timesteps
            self.training_env.env_method("set_overlay", {
                "PPO value V(s)": f"{float(v[0]):+.3f}" if v is not None else "?",
                "action logprob": f"{float(lp[0]):+.2f}" if lp is not None else "?",
                "timesteps": f"{self.num_timesteps}/{self.model._total_timesteps}",
                "train sps": f"{sps:.1f}  ({self.training_env.num_envs} envs)",
            }, indices=[0])
        except Exception:
            pass

    def _on_step(self) -> bool:
        self._push_overlay()
        if self.num_timesteps - self._checkpoints_done * self.checkpoint_every_steps >= self.checkpoint_every_steps:
            self._checkpoints_done += 1
            path = os.path.join(self.checkpoint_dir, f"checkpoint_{self.num_timesteps}.zip")
            self.model.save(path)
            self.saved_paths.append((self.num_timesteps, path))
            print(f"[checkpoint] saved {path}")
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total-timesteps", type=int, default=8000)
    ap.add_argument("--max-decisions", type=int, default=30)
    ap.add_argument("--decision-delta", type=float, default=0.3, help="Simulated seconds per agent decision (multiple of 0.1)")
    ap.add_argument("--n-steps", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--checkpoint-dir", default="ppo_checkpoints")
    ap.add_argument("--checkpoint-every-steps", type=int, default=2000)
    ap.add_argument("--skip-post-eval", action="store_true", help="Skip the post-training per-checkpoint evaluation pass")
    ap.add_argument("--torch-threads", type=int, default=2)
    ap.add_argument("--n-epochs", type=int, default=10)
    ap.add_argument("--target-kl", type=float, default=None, help="early-stop a PPO update when approx KL exceeds this")
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--start-mode", choices=["NORMAL_START", "CONNECTED_START"], default="NORMAL_START")
    ap.add_argument("--net", choices=["mlp", "cnn"], default="mlp")
    ap.add_argument("--start-gap", type=int, default=0, help="CONNECTED_START with the last N native road edits left unbuilt (gap curriculum)")
    ap.add_argument("--seed", type=int, default=TRAIN_SEED)
    ap.add_argument("--ipc-dir", default=None, help="own IPC dir (multi-instance); default = the shared default instance")
    ap.add_argument("--game-log", default=None)
    ap.add_argument("--headless", action="store_true", help="launch this trainer's game with -batchmode -nographics")
    ap.add_argument("--launch-game", action="store_true", help="launch (and own) a game instance before training")
    ap.add_argument("--shaping", type=float, default=0.0, help="EXPERIMENT: coefficient of potential-based connectivity shaping (0 = pure native reward)")
    ap.add_argument("--shaping-flip-y", action="store_true")
    ap.add_argument("--n-envs", type=int, default=1, help="parallel game instances feeding ONE PPO (instance 0 uses --ipc-dir/--game-log, the rest are headless)")
    ap.add_argument("--tag", default="p", help="name prefix for the extra instances' ipc dirs/logs")
    args = ap.parse_args()
    # PyTorch's default is one intra-op thread per core, and OpenMP workers
    # spin-wait between parallel regions: two trainers + several game instances
    # on one 12-core box measured ~250% CPU each for a tiny network. Cap it.
    torch.set_num_threads(args.torch_threads)

    if args.ipc_dir:
        Instance.ipc_dir = args.ipc_dir
    if args.game_log:
        Instance.log = args.game_log
    Instance.headless = args.headless
    for i in range(1, args.n_envs):
        INSTS.append(Inst(rf"C:\Users\TKA\Desktop\mini moto\logs\ipc_{args.tag}{i}",
                          rf"C:\Users\TKA\Desktop\mini moto\logs\instrumentation_{args.tag}{i}.log"))
    if args.launch_game:
        print(f"[train] launching {len(INSTS)} game instance(s) in parallel (primary ipc={Instance.ipc_dir}, headless={Instance.headless})", flush=True)
        if not relaunch_game():
            raise SystemExit("game instance(s) did not come up")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    log_path = os.path.join(args.checkpoint_dir, "train_eval_log.jsonl")

    # Initial construction is as failure-prone as any episode boundary (the
    # probe RESET attaches a real episode); retry with a game relaunch, same
    # policy as the in-training supervisor below.
    env = None
    for attempt in range(4):
        try:
            env = make_training_env(args)
            break
        except Exception as e:
            print(f"[supervisor] env construction failed (attempt {attempt + 1}): {e}", flush=True)
            if attempt == 3:
                raise
            if not relaunch_game():
                raise

    policy_kwargs = None
    if args.net == "cnn":
        policy_kwargs = dict(
            features_extractor_class=GridExtractor,
            features_extractor_kwargs=dict(features_dim=256),
            net_arch=dict(pi=[256], vf=[256]))
    model = MaskablePPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        target_kl=args.target_kl,
        gamma=args.gamma,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=args.ent_coef,
        verbose=1,
        seed=args.seed,
    )

    checkpoint_cb = CheckpointCallback(args.checkpoint_dir, args.checkpoint_every_steps)

    # Supervisor: the game is a live Unity process whose stock screen stack can
    # hard-die in ways outside the bridge's control (confirmed live: a process
    # death mid-training). On ANY env failure: relaunch the game, rebuild the
    # env on the fresh process, and continue the SAME model from where it was
    # (reset_num_timesteps=False). Checkpoints bound the worst-case loss.
    max_restarts = 20
    restarts = 0
    while True:
        try:
            # After a supervisor restart reset_num_timesteps=False makes SB3 ADD total_timesteps to the steps already
            # done, silently overshooting the budget (observed: a 122,880-step run trained past 144k). Ask only for the
            # REMAINING steps. First launch (restarts == 0) is unchanged.
            model.learn(total_timesteps=(args.total_timesteps if restarts == 0 else max(args.total_timesteps - model.num_timesteps, 1)),
                        callback=checkpoint_cb, reset_num_timesteps=(restarts == 0))
            break
        except (RuntimeError, TimeoutError, OSError, EOFError) as e:
            restarts += 1
            print(f"[supervisor] env failure #{restarts}: {e}", flush=True)
            if restarts > max_restarts:
                print("[supervisor] too many restarts, giving up", flush=True)
                raise
            if model.num_timesteps >= args.total_timesteps:
                break
            try:
                env.close()
            except Exception:
                pass
            # The game may have auto-restarted on its own -- try attaching to it
            # first; only force a relaunch if that fails too.
            env = None
            try:
                env = make_training_env(args)
                print("[supervisor] re-attached to the running game process", flush=True)
            except Exception as attach_err:
                print(f"[supervisor] re-attach failed ({attach_err}); relaunching game process...", flush=True)
                if not relaunch_game():
                    print("[supervisor] game did not come back up, giving up", flush=True)
                    raise
                env = make_training_env(args)
            model.set_env(env)
            print(f"[supervisor] resumed at {model.num_timesteps}/{args.total_timesteps} timesteps", flush=True)

    final_path = os.path.join(args.checkpoint_dir, "final_model.zip")
    model.save(final_path)
    checkpoint_cb.saved_paths.append((args.total_timesteps, final_path))
    print(f"Saved final model to {final_path}")

    if not args.skip_post_eval and args.n_envs > 1:
        print('[train] post-training eval needs a single env; skipped (use best_tracker / phase11a_runner)')
    elif not args.skip_post_eval:
        print("\n--- Post-training deterministic evaluation on fixed EVAL_SEEDS, per checkpoint ---")
        with open(log_path, "w") as f:
            for timesteps, path in checkpoint_cb.saved_paths:
                m = MaskablePPO.load(path)
                result = run_eval_episodes(env, m, EVAL_SEEDS)
                record = {"timesteps": timesteps, "checkpoint": path, **result}
                print(f"[eval] timesteps={timesteps} meanReturn={result['meanReturn']:.3f} "
                      f"meanScoreDelta={result['meanScoreDelta']:.3f} meanLength={result['meanLength']:.2f} "
                      f"wait={result['wait']} add={result['add']} remove={result['remove']}")
                f.write(json.dumps(record) + "\n")
        print(f"\nWrote per-checkpoint evaluation curve to {log_path}")

    env.close()


if __name__ == "__main__":
    main()
