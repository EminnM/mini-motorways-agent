"""
Phase 10B step 2: characterize the native environment's time-to-reward under
RandomPolicy, across multiple seeds and a configurable horizon, BEFORE
choosing a training episode length. Does not touch reward/observation/action
semantics -- this is pure measurement.

Usage:
  python characterize_time_to_reward.py --decisions 600 --seeds 3001,3002,...,3010 --out ../research/phase10b_time_to_reward_600.jsonl
"""

import argparse
import json
import statistics

import numpy as np

from mini_motorways_env import MiniMotorwaysEnv

DECISION_DELTA = 0.3  # must match Entrypoint.cs's IpcDecisionDelta


def run_episode(env: MiniMotorwaysEnv, seed: int, rng, max_decisions: int):
    obs, info = env.reset(seed=seed)
    first_score_decision = None
    first_score_time = None
    total_score = 0.0
    positive_events = 0
    terminated = truncated = False
    decision_index = 0
    wait_c = add_c = remove_c = 0
    legal_counts = []

    while not (terminated or truncated):
        mask = env.action_masks()
        legal_counts.append(int(mask.sum()))
        legal = np.where(mask)[0]
        action = int(rng.choice(legal))
        obs, reward, terminated, truncated, step_info = env.step(action)
        decision_index = step_info["decisionIndex"]
        rc = step_info["rewardComponents"]
        if rc["scoreDelta"] > 0:
            positive_events += 1
            total_score += rc["scoreDelta"]
            if first_score_decision is None:
                first_score_decision = decision_index
                first_score_time = decision_index * DECISION_DELTA
        applied = step_info["appliedAction"]
        if applied == "WAIT":
            wait_c += 1
        elif applied.startswith("ADD_ROAD"):
            add_c += 1
        elif applied.startswith("REMOVE_ROAD"):
            remove_c += 1

    return {
        "seed": seed,
        "maxDecisions": max_decisions,
        "firstScoreDecision": first_score_decision,
        "firstScoreTime": first_score_time,
        "finalScore": total_score,
        "positiveEvents": positive_events,
        "terminated": terminated,
        "truncated": truncated,
        "finalDecisionIndex": decision_index,
        "finalSimTime": decision_index * DECISION_DELTA,
        "waitCount": wait_c,
        "addCount": add_c,
        "removeCount": remove_c,
        "legalCountMean": statistics.mean(legal_counts) if legal_counts else 0,
        "legalCountMin": min(legal_counts) if legal_counts else 0,
        "legalCountMax": max(legal_counts) if legal_counts else 0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decisions", type=int, required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--random-seed", type=int, default=54321)
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    env = MiniMotorwaysEnv(max_decisions=args.decisions)
    rng = np.random.default_rng(args.random_seed)

    episodes = []
    for seed in seeds:
        ep = run_episode(env, seed, rng, args.decisions)
        print(f"seed={seed} firstScoreTime={ep['firstScoreTime']} finalScore={ep['finalScore']} "
              f"positiveEvents={ep['positiveEvents']} terminated={ep['terminated']} "
              f"finalSimTime={ep['finalSimTime']:.1f}s legalMean={ep['legalCountMean']:.1f}")
        episodes.append(ep)

    env.close()

    n_with_score = sum(1 for e in episodes if e["firstScoreDecision"] is not None)
    summary = {
        "decisions": args.decisions,
        "simSecondsPerEpisode": args.decisions * DECISION_DELTA,
        "numEpisodes": len(episodes),
        "numEpisodesWithAnyScore": n_with_score,
        "fractionWithAnyScore": n_with_score / len(episodes),
        "meanFinalScore": statistics.mean(e["finalScore"] for e in episodes),
        "medianFinalScore": statistics.median(e["finalScore"] for e in episodes),
        "meanFirstScoreTime": statistics.mean(e["firstScoreTime"] for e in episodes if e["firstScoreTime"] is not None) if n_with_score else None,
        "terminalRate": sum(1 for e in episodes if e["terminated"]) / len(episodes),
    }
    print("\n--- summary ---")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    with open(args.out, "w") as f:
        for ep in episodes:
            f.write(json.dumps(ep) + "\n")
        f.write(json.dumps({"summary": summary}) + "\n")
    print(f"\nWrote {len(episodes)} episodes + summary to {args.out}")


if __name__ == "__main__":
    main()
