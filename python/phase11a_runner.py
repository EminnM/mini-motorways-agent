"""
Phase 11A characterization runner: runs a simple policy (wait | random) for a
set of seeds under NORMAL_START or CONNECTED_START and records, per episode:
score events, time to first reward, return, length, terminal, action mix,
whether the initialised connection was destroyed, and whether a vehicle was
ever seen on the initialised road. Identical measurement code for every
condition so the comparison is clean.

WAIT-only policy: WAIT every decision, except that a pending weekly upgrade
choice (which makes WAIT illegal under the forced-choice rule) is claimed with
option 0 -- the only non-destructive thing legal at that moment.
"""

import argparse
import json
import re
import time

import numpy as np

import os
from mini_motorways_env import IpcClient, encode_observation

W, H = 37, 21
UPGRADE_BASE = 1 + W * H * 4
ADD_RE = re.compile(r"ADD_ROAD\((-?\d+),(-?\d+),([ES])\)")
APPLIED_RE = re.compile(r"(ADD_ROAD|REMOVE_ROAD)\((-?\d+),(-?\d+),([ES])\)")


def path_tiles_from_init(init_info):
    """Tiles touched by the initialisation's road edits (origin + the adjacent
    tile; South is y-1 on this map). Used only to MEASURE destruction/use."""
    tiles = set()
    if not init_info or not init_info.get("actions"):
        return tiles
    for m in ADD_RE.finditer(init_info["actions"]):
        x, y, d = int(m.group(1)), int(m.group(2)), m.group(3)
        tiles.add((x, y))
        tiles.add((x + 1, y) if d == "E" else (x, y - 1))
    return tiles


def vehicles_on(obs, tiles):
    n = 0
    for t in obs["tiles"]:
        if tuple(t["coord"]) in tiles:
            n += t.get("visibleVehicleCount", 0)
    return n


PPO_MODEL = None
STOCHASTIC = False


START_GAP = 0
ORACLE_QUEUE = []


def run_episode(client, seed, policy, start_mode, max_decisions, delta, rng):
    r = client.send("RESET", seed=seed, max_decisions=max_decisions,
                    decision_delta=delta, start_mode=start_mode, start_gap=START_GAP)
    init = r.get("initInfo")
    path = path_tiles_from_init(init)
    ORACLE_QUEUE.clear()
    if init and init.get("omittedEdits"):
        for m in ADD_RE.finditer(init["omittedEdits"]):
            ORACLE_QUEUE.append((int(m.group(1)), int(m.group(2)), m.group(3)))
    ep = {"seed": seed, "startMode": start_mode, "policy": policy,
          "init": {k: init.get(k) for k in ("house", "gap", "pathEdges", "applied", "rejected",
                                             "concreteBefore", "concreteAfter", "concreteSpent",
                                             "initSimTime", "error")} if init else None}
    counts = {"WAIT": 0, "ADD": 0, "REMOVE": 0, "UPGRADE": 0}
    score_events = []          # (decision, simTime, delta)
    ret = 0.0
    destroyed_at = None
    vehicle_on_path = False
    first_reward_decision = None
    first_reward_simtime = None
    sim_t0 = r["observation"]["simulationTime"]
    steps = 0
    terminated = truncated = False
    if path and vehicles_on(r["observation"], path):
        vehicle_on_path = True

    while not (terminated or truncated):
        legal = r["legalActionIndices"]
        if policy == "wait":
            ups = [a for a in legal if a >= UPGRADE_BASE]
            action = 0 if 0 in legal else (ups[0] if ups else legal[0])
        elif policy == "oracle":
            # CEILING/solvability check only (never a training signal): applies the
            # edits the gap initializer left unbuilt, in order, then WAITs.
            if ORACLE_QUEUE:
                x, y, d = ORACLE_QUEUE.pop(0)
                tile = (y - (-10)) * W + (x - (-18))
                cand = 1 + tile * 4 + (0 if d == "E" else 1) * 2
                action = cand if cand in legal else (0 if 0 in legal else legal[0])
            else:
                ups = [a for a in legal if a >= UPGRADE_BASE]
                action = 0 if 0 in legal else (ups[0] if ups else legal[0])
        elif policy == "ppo":
            mask = np.zeros(r["actionSpaceSize"] if "actionSpaceSize" in r else W * H * 4 + 3, dtype=bool)
            mask[legal] = True
            act, _ = PPO_MODEL.predict(encode_observation(r["observation"]), action_masks=mask,
                                       deterministic=not STOCHASTIC)
            action = int(act)
        else:
            action = int(rng.choice(legal))
        r = client.send("STEP", action=action)
        if "error" in r:
            ep["error"] = r["error"]
            break
        steps += 1
        applied = r.get("appliedAction", "WAIT")
        if applied == "WAIT":
            counts["WAIT"] += 1
        elif applied.startswith("ADD_ROAD"):
            counts["ADD"] += 1
        elif applied.startswith("REMOVE_ROAD"):
            counts["REMOVE"] += 1
            m = APPLIED_RE.match(applied)
            if m and r.get("accepted") and destroyed_at is None and (int(m.group(2)), int(m.group(3))) in path:
                destroyed_at = steps
        else:
            counts["UPGRADE"] += 1
        d = r["rewardComponents"]["scoreDelta"]
        ret += r["reward"]
        if d > 0:
            score_events.append((steps, round(r["observation"]["simulationTime"] - sim_t0, 1), d))
            if first_reward_decision is None:
                first_reward_decision = steps
                first_reward_simtime = round(r["observation"]["simulationTime"] - sim_t0, 1)
        if path and not vehicle_on_path and vehicles_on(r["observation"], path):
            vehicle_on_path = True
        terminated = bool(r.get("terminated"))
        truncated = bool(r.get("truncated"))

    g = r["observation"]["globals"] if "observation" in r else {}
    ep.update({
        "length": steps, "terminated": terminated, "truncated": truncated,
        "finalScore": g.get("score"), "return": round(ret, 2),
        "scoreEvents": score_events, "nScoreEvents": len(score_events),
        "firstRewardDecision": first_reward_decision, "firstRewardSimTime": first_reward_simtime,
        "actionCounts": counts, "connectionDestroyedAtDecision": destroyed_at,
        "vehicleSeenOnInitRoad": vehicle_on_path, "finalDay": g.get("day"),
    })
    return ep


def summarize(eps):
    n = len(eps)
    pos = [e for e in eps if e.get("nScoreEvents", 0) > 0]
    tot = {k: sum(e["actionCounts"][k] for e in eps) for k in ("WAIT", "ADD", "REMOVE", "UPGRADE")}
    steps = max(sum(tot.values()), 1)
    firsts = [e["firstRewardDecision"] for e in pos]
    return {
        "episodes": n,
        "fractionWithPositiveReward": len(pos) / n,
        "meanFinalScore": float(np.mean([e["finalScore"] or 0 for e in eps])),
        "meanReturn": float(np.mean([e["return"] for e in eps])),
        "meanLength": float(np.mean([e["length"] for e in eps])),
        "terminalRate": float(np.mean([e["terminated"] for e in eps])),
        "medianFirstRewardDecision": float(np.median(firsts)) if firsts else None,
        "actionMix": {k: round(v / steps, 3) for k, v in tot.items()},
        "fractionConnectionDestroyed": float(np.mean([e["connectionDestroyedAtDecision"] is not None for e in eps])),
        "fractionVehicleSeenOnInitRoad": float(np.mean([e["vehicleSeenOnInitRoad"] for e in eps])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=["wait", "random", "ppo", "oracle"], required=True)
    ap.add_argument("--model", default=None, help="checkpoint .zip for --policy ppo")
    ap.add_argument("--stochastic", action="store_true", help="sample from the PPO policy instead of argmax")
    ap.add_argument("--ipc-dir", default=None)
    ap.add_argument("--start-gap", type=int, default=0)
    ap.add_argument("--start-mode", choices=["NORMAL_START", "CONNECTED_START"], required=True)
    ap.add_argument("--seeds", default="heldout", help="heldout | a-b | comma list")
    ap.add_argument("--max-decisions", type=int, default=600)
    ap.add_argument("--decision-delta", type=float, default=1.0)
    ap.add_argument("--random-seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.seeds == "heldout":
        from eval_seeds import EVAL_SEEDS
        seeds = list(EVAL_SEEDS)
    elif args.seeds == "val":
        from eval_seeds import VAL_SEEDS
        seeds = list(VAL_SEEDS)
    elif "," in args.seeds:
        seeds = [int(x) for x in args.seeds.split(",")]
    elif "-" in args.seeds:
        a, b = args.seeds.split("-")
        seeds = list(range(int(a), int(b) + 1))
    else:
        seeds = [int(args.seeds)]
    global PPO_MODEL, STOCHASTIC, START_GAP
    START_GAP = args.start_gap
    if args.policy == "ppo":
        from sb3_contrib import MaskablePPO
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import train_ppo  # registers GridExtractor for CNN checkpoints
        PPO_MODEL = MaskablePPO.load(args.model, custom_objects={"GridExtractor": train_ppo.GridExtractor})
        STOCHASTIC = args.stochastic
    if args.ipc_dir:
        client = IpcClient(request_path=os.path.join(args.ipc_dir, "request.json"),
                           response_path=os.path.join(args.ipc_dir, "response.json"), timeout=120)
    else:
        client = IpcClient(timeout=120)
    rng = np.random.default_rng(args.random_seed)
    eps = []
    t0 = time.time()
    with open(args.out, "w") as f:
        for sd in seeds:
            ep = run_episode(client, sd, args.policy, args.start_mode, args.max_decisions,
                             args.decision_delta, rng)
            eps.append(ep)
            f.write(json.dumps(ep) + "\n")
            f.flush()
            print(f"[{args.policy}/{args.start_mode}] seed {sd}: len={ep['length']} "
                  f"term={ep['terminated']} score={ep['finalScore']} events={ep['nScoreEvents']} "
                  f"firstReward={ep['firstRewardDecision']} vehOnRoad={ep['vehicleSeenOnInitRoad']} "
                  f"destroyed={ep['connectionDestroyedAtDecision']}", flush=True)
        summ = summarize(eps)
        f.write(json.dumps({"SUMMARY": summ}) + "\n")
    print("SUMMARY", json.dumps(summ, indent=1))
    print(f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
