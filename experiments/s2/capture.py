"""Capture LIVE native states for S2 (no learned policy involved).

For every frozen native Gap-1 task of a split: reset the real game into the task, store the raw native observation,
the native legal-action indices, the init info and a compact carpark-ambiguity summary. For the first K tasks also run
action-mapping tests on the live game (WAIT / ADD / REMOVE) and store the post-action raw observations.
  python -m experiments.s2.capture --split validation --full --action-tests 12
  python -m experiments.s2.capture --split test --shard 0/1            # summaries only (ambiguity statistics)"""
import argparse
import gzip
import json
import os
import sys
import time

import numpy as np

from .. import paths
from .. import gamectl
from ..gap1 import tasks as T

OUT = os.path.join(paths.ROOT, "research", "synthetic_to_native")
MAN = os.path.join(paths.ROOT, "research", "gap1", "manifests")
W = 37


def tiles_of(raw):
    return {(t["coord"][0], t["coord"][1]): t for t in raw["tiles"]}


def deg(t):
    return sum(1 for v in t["road"].values() if v in T.LIVE)


def ambiguity_summary(raw):
    """Carpark ambiguity of one native state, computed from the RAW native observation (which contains contentType
    'Carpark') and from what survives the python encoder (carpark -> content 0)."""
    tiles = tiles_of(raw)
    foot = {k for k, t in tiles.items() if t["contentType"] == "Destination"}
    carp = {k for k, t in tiles.items() if t["contentType"] == "Carpark"}
    entry = {k for k in carp if deg(tiles[k]) > 0 and any((k[0] + dx, k[1] + dy) in foot for dx, dy in T.DIRS.values())}
    adj_road = set()
    for f in foot:
        for dx, dy in T.DIRS.values():
            c = (f[0] + dx, f[1] + dy)
            if c in foot or c not in tiles:
                continue
            if deg(tiles[c]) > 0:
                adj_road.add(c)
    leaves = {c for c in adj_road if deg(tiles[c]) == 1}
    n_adj = len(adj_road)
    if n_adj <= 1:
        cat = "unique"                 # exactly one road tile touches the footprint -> it is the entry
    elif len(leaves) == 1 and leaves <= entry:
        cat = "resolvable_by_leaf_rule"
    else:
        cat = "unresolvable_by_leaf_rule"
    return {"n_footprint_tiles": len(foot), "n_carpark_tiles_in_raw": len(carp), "entry_tiles_true": sorted(entry), "n_adjacent_road_tiles": n_adj,
            "adjacent_road_tiles": sorted(adj_road), "adjacent_leaf_tiles": sorted(leaves), "category": cat,
            "carpark_visible_in_encoded_observation": False,
            "carpark_visible_in_raw_native_json": bool(carp)}


def edge_live(raw_tiles, x, y, d):
    t = raw_tiles.get((x, y))
    nx, ny = x + T.DIRS[d][0], y + T.DIRS[d][1]
    n = raw_tiles.get((nx, ny))
    return (t is not None and t["road"][d] in T.LIVE), (n is not None and n["road"][T.OPP[d]] in T.LIVE)


def action_tests(env, task, rng):
    """Live tests of index mapping: WAIT, one legal ADD, one legal REMOVE (if any), then a*. Returns a list of results."""
    res = []
    raws = []
    mask = env.action_masks()
    # WAIT
    before = {k for k, t in tiles_of(env.last_raw_observation).items() for d, v in t["road"].items() if v in T.LIVE for _ in [0]}
    obs, r, te, tr, info = env.step(0)
    after = {k for k, t in tiles_of(env.last_raw_observation).items() for d, v in t["road"].items() if v in T.LIVE for _ in [0]}
    raws.append(("after_WAIT", env.last_raw_observation))
    res.append({"test": "WAIT_index0", "ok": info["appliedAction"] == "WAIT" and before == after, "applied": info["appliedAction"]})
    mask = env.action_masks()
    legal = np.flatnonzero(mask)
    a_star = task["a_star"]["action_index"]
    adds = [int(a) for a in legal if 1 <= a < 1 + W * 21 * 4 and (a - 1) % 2 == 0 and a != a_star]
    rems = [int(a) for a in legal if 1 <= a < 1 + W * 21 * 4 and (a - 1) % 2 == 1]
    if adds:
        a = int(rng.choice(adds))
        r0 = (a - 1) // 4
        x, y, d = -18 + r0 % W, -10 + r0 // W, "E" if ((a - 1) % 4) // 2 == 0 else "S"
        obs, r, te, tr, info = env.step(a)
        e1, e2 = edge_live(tiles_of(env.last_raw_observation), x, y, d)
        raws.append(("after_ADD", env.last_raw_observation))
        res.append({"test": f"ADD({x},{y},{d})_index{a}", "ok": bool(info["accepted"]) and e1 and e2 and info["appliedAction"] == f"ADD_ROAD({x},{y},{d})",
                    "applied": info["appliedAction"], "edge_endpoint_flags": [e1, e2]})
    mask = env.action_masks()
    rems = [int(a) for a in np.flatnonzero(mask) if 1 <= a < 1 + W * 21 * 4 and (a - 1) % 2 == 1]
    if rems:
        a = int(rng.choice(rems))
        r0 = (a - 1) // 4
        x, y, d = -18 + r0 % W, -10 + r0 // W, "E" if ((a - 1) % 4) // 2 == 0 else "S"
        obs, r, te, tr, info = env.step(a)
        e1, e2 = edge_live(tiles_of(env.last_raw_observation), x, y, d)
        raws.append(("after_REMOVE", env.last_raw_observation))
        res.append({"test": f"REMOVE({x},{y},{d})_index{a}", "ok": bool(info["accepted"]) and not e1 and info["appliedAction"] == f"REMOVE_ROAD({x},{y},{d})",
                    "applied": info["appliedAction"], "edge_endpoint_flags": [e1, e2]})
    return res, raws


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="validation")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--full", action="store_true", help="store the full raw observation + legal indices for every task")
    ap.add_argument("--action-tests", type=int, default=0)
    ap.add_argument("--tag", default="s2cap")
    ap.add_argument("--restart-every", type=int, default=60)
    a = ap.parse_args(argv)
    import mini_motorways_env as M
    si, sn = (int(x) for x in a.shard.split("/"))
    tasks = [json.loads(l) for l in open(os.path.join(MAN, f"{a.split}.jsonl")) if l.strip()]
    tasks = [t for j, t in enumerate(tasks) if j % sn == si]
    os.makedirs(OUT, exist_ok=True)
    out_summary = os.path.join(OUT, f"capture_{a.split}_{si}of{sn}.jsonl")
    out_full = gzip.open(os.path.join(OUT, f"native_states_{a.split}_{si}of{sn}.jsonl.gz"), "wt", encoding="utf-8") if a.full else None
    rng = np.random.default_rng(0)
    t0 = time.time()
    inst = gamectl.GameInstance(f"{a.tag}{a.split[:2]}{si}").start()

    def make_env():
        return M.MiniMotorwaysEnv(max_decisions=200, decision_delta=1.0, request_path=os.path.join(inst.ipc_dir, "request.json"),
                                  response_path=os.path.join(inst.ipc_dir, "response.json"), start_mode="CONNECTED_START", start_gap=0)

    env = make_env()
    n_done = 0
    try:
        with open(out_summary, "w", encoding="utf-8") as f:
            for i, t in enumerate(tasks):
                for attempt in range(3):
                    try:
                        env.extra_reset_args = {"start_gap_index": t["start_gap_index"]}
                        obs, info = env.reset(seed=t["game_seed"])
                        break
                    except (TimeoutError, RuntimeError, OSError):
                        try:
                            env.close()
                        except Exception:
                            pass
                        inst.start()
                        env = make_env()
                else:
                    raise RuntimeError("reset failed repeatedly")
                raw = env.last_raw_observation
                mask_idx = np.flatnonzero(env.action_masks()).tolist()
                amb = ambiguity_summary(raw)
                init = env.last_init_info or {}
                rec = {"task_id": t["task_id"], "a_star_index": t["a_star"]["action_index"], "a_star_legal_native": t["a_star"]["action_index"] in set(mask_idx),
                       "house_connected_at_reset": T.house_connected(raw, tuple(t["house"]), [tuple(x) for x in t["destination_footprint"]]),
                       "native_legal_count": len(mask_idx), "obs_len": int(obs.shape[0]), "bounds": raw["bounds"], "ambiguity": amb,
                       "orientation": t["gap_orientation"], "gap_position_in_path": t["gap_position_in_path"], "path_length_edits": t["path_length_edits"]}
                if a.action_tests and i < a.action_tests:
                    rec["action_tests"] = action_tests(env, t, rng)
                f.write(json.dumps(rec) + "\n")
                f.flush()
                if out_full is not None:
                    if not (a.action_tests and i < a.action_tests):
                        pass
                    out_full.write(json.dumps({"task_id": t["task_id"], "raw": raw, "legal_indices": mask_idx, "obs_nonzero": int(np.count_nonzero(obs)),
                                               "obs_head_checksum": float(obs.sum())}) + "\n")
                n_done += 1
                if n_done % 50 == 0:
                    print(f"{a.split} shard {si}/{sn}: {n_done}/{len(tasks)} {time.time() - t0:.0f}s", flush=True)
                if n_done % a.restart_every == 0 and i + 1 < len(tasks):
                    try:
                        env.close()
                    except Exception:
                        pass
                    inst.start()
                    env = make_env()
    finally:
        if out_full is not None:
            out_full.close()
        try:
            env.close()
        except Exception:
            pass
        inst.stop()
    print(f"done {a.split} {n_done} tasks in {time.time() - t0:.0f}s -> {out_summary}")


if __name__ == "__main__":
    main()
