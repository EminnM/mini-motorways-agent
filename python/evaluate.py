"""
Phase 10: evaluate RandomPolicy and/or a trained MaskablePPO checkpoint
against the SAME native environment, same seeds, same decision interval and
episode limit -- so the two are directly comparable, per the phase brief.

Usage:
  python evaluate.py --policy random --seeds 2001,2002,2003,2004,2005,2006,2007,2008,2009,2010 --out ../research/phase10_random_baseline.jsonl
  python evaluate.py --policy ppo --model ppo_checkpoints/final_model.zip --seeds 2001,...,2010 --out ../research/phase10_ppo_evaluation.jsonl
  python evaluate.py --policy ppo --model ppo_checkpoints/final_model.zip --seeds 9001 --max-decisions 10 --dump-actions ../logs/phase10_replay_actions.json
"""

import argparse
import json
import statistics
import sys

import numpy as np

from mini_motorways_env import MiniMotorwaysEnv


def random_policy_fn(rng):
    def fn(obs, mask):
        legal = np.where(mask)[0]
        return int(rng.choice(legal))
    return fn


def ppo_policy_fn(model):
    def fn(obs, mask):
        action, _ = model.predict(obs, action_masks=mask, deterministic=True)
        return int(action)
    return fn


def run_episode(env: MiniMotorwaysEnv, seed: int, policy_fn, max_decisions: int):
    obs, info = env.reset(seed=seed)
    total_reward = 0.0
    total_score_delta = 0.0
    total_terminal_penalty = 0.0
    wait_count = add_count = remove_count = rejected_count = 0
    actions_taken = []
    terminated = truncated = False
    decision_index = 0

    while not (terminated or truncated):
        mask = env.action_masks()
        action = policy_fn(obs, mask)
        obs, reward, terminated, truncated, step_info = env.step(action)
        total_reward += reward
        rc = step_info["rewardComponents"]
        total_score_delta += rc["scoreDelta"]
        total_terminal_penalty += rc["terminalPenalty"]
        applied = step_info["appliedAction"]
        if applied == "WAIT":
            wait_count += 1
        elif applied.startswith("ADD_ROAD"):
            add_count += 1
        elif applied.startswith("REMOVE_ROAD"):
            remove_count += 1
        if not step_info["accepted"]:
            rejected_count += 1
        actions_taken.append(action)
        decision_index = step_info["decisionIndex"]

    return {
        "seed": seed,
        "return": total_reward,
        "scoreDeltaSum": total_score_delta,
        "terminalPenaltySum": total_terminal_penalty,
        "episodeLength": decision_index,
        "terminated": terminated,
        "truncated": truncated,
        "waitCount": wait_count,
        "addCount": add_count,
        "removeCount": remove_count,
        "rejectedCount": rejected_count,
        "actions": actions_taken,
    }


def summarize(episodes):
    returns = [e["return"] for e in episodes]
    scores = [e["scoreDeltaSum"] for e in episodes]
    lengths = [e["episodeLength"] for e in episodes]
    terminal_rate = sum(1 for e in episodes if e["terminated"]) / len(episodes)
    return {
        "numEpisodes": len(episodes),
        "meanReturn": statistics.mean(returns),
        "medianReturn": statistics.median(returns),
        "stdReturn": statistics.pstdev(returns) if len(returns) > 1 else 0.0,
        "meanScoreDeltaSum": statistics.mean(scores),
        "medianScoreDeltaSum": statistics.median(scores),
        "meanEpisodeLength": statistics.mean(lengths),
        "terminalRate": terminal_rate,
        "meanWaitCount": statistics.mean([e["waitCount"] for e in episodes]),
        "meanAddCount": statistics.mean([e["addCount"] for e in episodes]),
        "meanRemoveCount": statistics.mean([e["removeCount"] for e in episodes]),
        "meanRejectedCount": statistics.mean([e["rejectedCount"] for e in episodes]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=["random", "ppo"], required=True)
    ap.add_argument("--model", default=None, help="Path to a MaskablePPO .zip checkpoint (required for --policy ppo)")
    ap.add_argument("--seeds", required=True, help="Comma-separated list of evaluation seeds")
    ap.add_argument("--max-decisions", type=int, default=40)
    ap.add_argument("--out", default=None, help="Write per-episode JSONL results here")
    ap.add_argument("--dump-actions", default=None, help="For a single-seed run: write the taken action-index sequence as a JSON array (for the Phase 9A visible replay)")
    ap.add_argument("--random-seed", type=int, default=12345, help="Seed for the RandomPolicy's OWN rng (never the env's seed)")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]

    env = MiniMotorwaysEnv(max_decisions=args.max_decisions)

    if args.policy == "random":
        rng = np.random.default_rng(args.random_seed)
        policy_fn = random_policy_fn(rng)
    else:
        if not args.model:
            print("--model is required for --policy ppo", file=sys.stderr)
            sys.exit(1)
        from sb3_contrib import MaskablePPO
        model = MaskablePPO.load(args.model)
        policy_fn = ppo_policy_fn(model)

    episodes = []
    for seed in seeds:
        ep = run_episode(env, seed, policy_fn, args.max_decisions)
        print(f"seed={seed} return={ep['return']:.2f} scoreDeltaSum={ep['scoreDeltaSum']:.2f} "
              f"len={ep['episodeLength']} terminated={ep['terminated']} truncated={ep['truncated']} "
              f"W/A/R={ep['waitCount']}/{ep['addCount']}/{ep['removeCount']} rejected={ep['rejectedCount']}")
        episodes.append(ep)

    env.close()

    summary = summarize(episodes)
    print("\n--- summary ---")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    if args.out:
        with open(args.out, "w") as f:
            for ep in episodes:
                f.write(json.dumps({"policy": args.policy, **ep}) + "\n")
            f.write(json.dumps({"policy": args.policy, "summary": summary}) + "\n")
        print(f"\nWrote {len(episodes)} episodes + summary to {args.out}")

    if args.dump_actions:
        if len(episodes) != 1:
            print("--dump-actions requires exactly one seed", file=sys.stderr)
        else:
            with open(args.dump_actions, "w") as f:
                json.dump(episodes[0]["actions"], f)
            print(f"Wrote {len(episodes[0]['actions'])} action indices to {args.dump_actions}")


if __name__ == "__main__":
    main()
