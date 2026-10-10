"""Generate and verify Gap-1 task manifests with the REAL native simulation.

For each candidate (game seed, per-mille gap position) the environment is RESET into CONNECTED_START with exactly one
path edit left out. A candidate is kept only if
  * the omitted edit a* is in the native legal-action mask,
  * the target house is NOT connected at the start,
  * applying a* (one real STEP) connects the target house to the target destination (native state, via the
    observation's live road graph), and
  * WAITing afterwards for the continuation window yields native score (recorded as metadata).
Writes research/gap1/manifests/{train,validation,test}.jsonl + stats. Splits are by BASE LAYOUT (house, destination,
full path): every configuration of one base layout lives in exactly one split, so TEST layouts are never seen in TRAIN."""
import argparse
import hashlib
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments import paths  # noqa: F401,E402
from mini_motorways_env import IpcClient  # noqa: E402
from experiments.gap1 import tasks as T  # noqa: E402

OUT = os.path.join(paths.ROOT, "research", "gap1", "manifests")


def verify_candidate(c: IpcClient, seed: int, permille: int, cont: int, delta: float):
    r = c.send("RESET", seed=seed, max_decisions=1000, decision_delta=delta, start_mode="CONNECTED_START",
               start_gap=0, start_gap_index=1000 + permille)
    info = r.get("initInfo") or {}
    if info.get("error") or "omittedEdits" not in info or not info.get("omittedEdits") or info.get("gap") != 1:
        return None, "init_failed"
    omitted = T.parse_edits(info["omittedEdits"])
    if len(omitted) != 1 or omitted[0][2] not in ("E", "S"):
        return None, "omitted_dir"
    gx, gy, gd = omitted[0]
    a_star = T.action_index_add(gx, gy, gd)
    house, dest = tuple(info["house"]), [tuple(x) for x in info["destinationFootprint"]]
    if a_star not in set(r["legalActionIndices"]):
        return None, "a_star_not_legal"
    if T.house_connected(r["observation"], house, dest):
        return None, "already_connected"
    s = c.send("STEP", action=a_star)
    if "error" in s:
        return None, "step_error"
    if not T.house_connected(s["observation"], house, dest):
        return None, "a_star_does_not_connect"
    score = s["rewardComponents"]["scoreDelta"]
    first = None
    for i in range(cont):
        s = c.send("STEP", action=0)
        if "error" in s:
            break
        d = s["rewardComponents"]["scoreDelta"]
        score += d
        if d > 0 and first is None:
            first = i + 1
        if s.get("terminated"):
            break
    full = T.parse_edits(info["fullPathEdits"])
    built = [e for e in full if e != omitted[0]]
    rec = {
        "game_seed": seed, "start_mode": "CONNECTED_START", "start_gap_index": info["gapIndex"],
        "decision_interval_seconds": delta,
        "a_star": {"x": gx, "y": gy, "dir": gd, "action_index": a_star},
        "gap_orientation": "horizontal" if gd == "E" else "vertical",
        "gap_position_in_path": info["gapIndex"], "path_length_edits": info["pathLengthEdits"],
        "house": list(house), "house_group": info.get("houseGroup"),
        "destination_footprint": [list(x) for x in dest], "destination_group": info.get("destinationGroup"),
        "full_path_edits": info["fullPathEdits"], "built_edits_count": len(built),
        "legal_action_count": len(r["legalActionIndices"]),
        "oracle_native_score_after_a_star_continuation": score, "oracle_first_score_delay_decisions": first,
        "oracle_continuation_decisions": cont,
        "base_layout_signature": hashlib.sha256(T.layout_signature(info, False).encode()).hexdigest()[:16],
        "layout_signature": hashlib.sha256(T.layout_signature(info, True).encode()).hexdigest()[:16],
    }
    return rec, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ipc-dir", default=os.path.join(paths.LOGS_DIR, "ipc_snap"))
    ap.add_argument("--candidates", type=int, default=1400)
    ap.add_argument("--rng-seed", type=int, default=20261008)
    ap.add_argument("--delta", type=float, default=1.0)
    ap.add_argument("--continuation", type=int, default=30)
    ap.add_argument("--raw-out", default=os.path.join(OUT, "_candidates_raw.jsonl"))
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    c = IpcClient(os.path.join(a.ipc_dir, "request.json"), os.path.join(a.ipc_dir, "response.json"), timeout=300)
    c.send("HELLO")
    rng = random.Random(a.rng_seed)
    done = {}
    if os.path.exists(a.raw_out):      # resumable
        for line in open(a.raw_out, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                done[(r["game_seed"], r["permille"])] = r
    reasons = Counter()
    t0 = time.time()
    with open(a.raw_out, "a", encoding="utf-8") as f:
        for i in range(a.candidates):
            seed = rng.randint(1000, 2_000_000_000)
            permille = rng.randint(0, 999)
            if (seed, permille) in done:
                continue
            try:
                rec, why = verify_candidate(c, seed, permille, a.continuation, a.delta)
            except Exception as e:
                rec, why = None, f"exception:{type(e).__name__}"
            reasons[why] += 1
            row = {"game_seed": seed, "permille": permille, "ok": rec is not None, "reason": why}
            if rec:
                row.update(rec)
            f.write(json.dumps(row) + "\n"); f.flush()
            if (i + 1) % 25 == 0:
                print(f"{i+1}/{a.candidates} kept={reasons['ok']} reasons={dict(reasons)} {time.time()-t0:.0f}s", flush=True)
    print("candidate generation done", dict(reasons))


if __name__ == "__main__":
    main()
