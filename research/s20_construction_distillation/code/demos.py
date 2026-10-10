"""S20 demonstrations: for every scene the explicit goal = (goal house tile, its colour group's destinations); the S18-H GoalExecutor is rolled
for that single objective and every decision is recorded: goal-conditioned observation (v2 tiles + 2 goal channels, uint8 x6; globals float32),
extended legal mask (3111 env actions + DONE), planner action, remaining concrete, done flag, oracle valid set.  The terminal state (goal
functionally connected) gets the DONE label (index 3111).  Also used by train.py for DAgger relabelling (label_state)."""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
from common import A_DONE, GoalExecutor, goal_obs_u8, legal_mask_ext, oracle_valid, restore_env  # noqa: E402

DATA = os.path.join(C.S20, "data")
MAX_STEPS = 60


def label_state(env, goal):
    """planner label for an arbitrary state: DONE if the goal is connected, else the first action of a fresh GoalExecutor plan (deterministic).
    Returns (action, planned_house)."""
    if goal in env.connected_houses():
        return A_DONE, None
    ex = GoalExecutor(goal)
    a = ex.act(env)
    return int(a), (ex.planned_targets[-1] if ex.planned_targets else None)


def record_scene(sc, with_valid=True):
    goal = tuple(sc["meta"]["goal"])
    env = restore_env(sc["state"])
    ex = GoalExecutor(goal)
    rows = []
    completed = False
    for step in range(MAX_STEPS + 1):
        tiles, glob = goal_obs_u8(env, goal)
        mask = legal_mask_ext(env)
        done = goal in env.connected_houses()
        if done:
            a = A_DONE
        else:
            a = ex.act(env)
        valid, _ = oracle_valid(env, goal) if with_valid else (set(), 0)
        rows.append({"tiles": tiles, "glob": glob, "mask": mask, "action": a, "concrete": env.concrete, "done": done, "valid": sorted(valid), "step": step,
                     "planned_house": ex.planned_targets[-1] if ex.planned_targets else None})
        if done:
            completed = True
            break
        if a == 0:
            break                                                 # the planner waits (abandoned plan): recorded as a WAIT label, trajectory ends
        env.step(a)
    return rows, completed, ex.stats


def pack(rows_all):
    """list of per-transition dicts -> arrays (valid sets flattened with offsets)."""
    n = len(rows_all)
    tiles = np.stack([r["tiles"] for r in rows_all])
    glob = np.stack([r["glob"] for r in rows_all])
    mask = np.packbits(np.stack([r["mask"] for r in rows_all]), axis=1)
    action = np.array([r["action"] for r in rows_all], dtype=np.int32)
    concrete = np.array([r["concrete"] for r in rows_all], dtype=np.int16)
    done = np.array([r["done"] for r in rows_all], dtype=bool)
    scene = np.array([r["scene"] for r in rows_all], dtype=np.int32)
    step = np.array([r["step"] for r in rows_all], dtype=np.int16)
    vflat = np.array([a for r in rows_all for a in r["valid"]], dtype=np.int32)
    voff = np.zeros(n + 1, dtype=np.int64)
    voff[1:] = np.cumsum([len(r["valid"]) for r in rows_all])
    return dict(tiles=tiles, glob=glob, mask=mask, action=action, concrete=concrete, done=done, scene=scene, step=step, vflat=vflat, voff=voff)


def build_split(split):
    scenes = json.load(open(os.path.join(DATA, f"scenes_{split}.json")))
    rows_all, traj = [], []
    for i, sc in enumerate(scenes):
        rows, completed, st = record_scene(sc)
        for r in rows:
            r["scene"] = i
        rows_all += rows
        traj.append({"scene": i, "seed": sc["meta"]["seed"], "steps": len(rows) - 1, "completed": completed, "n_adds_planned": sc["meta"]["n_adds"], "abandoned": st["abandoned"],
                     "planned_houses": sorted({tuple(r["planned_house"]) for r in rows if r["planned_house"] is not None}), "waits": sum(1 for r in rows if r["action"] == 0)})
    return pack(rows_all), traj, scenes


if __name__ == "__main__":
    t0 = time.time()
    stats = {}
    for split in ("train", "val", "test"):
        arrs, traj, scenes = build_split(split)
        np.savez_compressed(os.path.join(DATA, f"demos_{split}.npz"), **arrs)
        json.dump(traj, open(os.path.join(DATA, f"traj_{split}.json"), "w"))
        acts = arrs["action"]
        n = len(acts)
        stats[split] = {"trajectories": len(traj), "transitions": int(n), "completed": int(sum(t["completed"] for t in traj)),
                        "done_labels": int((acts == A_DONE).sum()), "wait_labels": int((acts == 0).sum()),
                        "remove_labels": int(sum(1 for a in acts if 0 < a < A_DONE and C.L.decode_action(int(a))[0] == "REMOVE")),
                        "add_labels": int(sum(1 for a in acts if 0 < a < A_DONE and C.L.decode_action(int(a))[0] == "ADD")),
                        "steps_mean": float(np.mean([t["steps"] for t in traj])), "steps_max": int(max(t["steps"] for t in traj)),
                        "steps_eq_planned_adds": int(sum(t["steps"] == t["n_adds_planned"] for t in traj)),
                        "mean_valid_size_nonterminal": float(np.mean(np.diff(arrs["voff"])[~arrs["done"]])),
                        "planned_only_goal": int(sum(t["planned_houses"] == [tuple(scenes[t["scene"]]["meta"]["goal"])] for t in traj)),
                        "route_len_hist": {str(k): int(v) for k, v in zip(*np.unique([s["meta"]["route_len"] for s in scenes], return_counts=True))},
                        "existing_edges_hist": {str(k): int(v) for k, v in zip(*np.unique([min(s["meta"]["n_edges_existing"] // 10 * 10, 40) for s in scenes], return_counts=True))},
                        "multi_route_frac": float(np.mean([s["meta"]["multi_route"] for s in scenes]))}
        print(split, json.dumps(stats[split]), f"{time.time() - t0:.0f}s", flush=True)
    stats["demo_s"] = round(time.time() - t0, 1)
    json.dump(stats, open(os.path.join(DATA, "demo_stats.json"), "w"), indent=1)
