"""Spawn-steering pilot (PREPARED FOR REVIEW, NOT EXECUTED): branch planning rules, strategic placement metric and classification of spawn changes, all predefined.
Design per anchor (one DIAG_FORK, all branches from the same snapshot, WAIT continuation):
  base, base2            no edits (restore-fidelity / negative control)
  block_far              3-edge disconnected E-segment on the empty in-window tiles FARTHEST from every destination block (hypothesis: removes 'bad' far spawn candidates -> nearer spawns)
  block_near             same-size segment on empty in-window tiles NEAREST to a destination block (opposite direction: removes 'good' candidates -> farther spawns)
Both interventions perturb the placer's PRNG stream equally in kind (one draw per feasible candidate), so a DIRECTIONAL effect (far-block -> nearer, near-block -> farther) isolates geometric blocking
from stream perturbation, which is direction-neutral. Classification per branch and spawn: BLOCKED (base's spawn tile or its driveway tile is occupied by the intervention), SHIFTED (base's tile still
free but a different tile won: stream re-alignment / feasibility), UNCHANGED. Strategic metric per new house: Manhattan distance to the nearest same-colour destination tile (lower = cheaper to connect);
per new destination: distance to the nearest house. Opportunity cost: intervention concrete (3) and the change in the planner's connection cost for houses present at the anchor (Dijkstra on the
branch's final road graph vs base). No example is selected after seeing outcomes: seeds, anchors and rules are fixed here."""
import os
import sys
from typing import Any, Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import harness as H  # noqa: E402
from synthetic_rl import layout as L  # noqa: E402

SEEDS = [600000 + 7919 * i for i in range(10)]        # fresh seeds: not the 12 evaluation seeds, not the S13 test seeds
ANCHORS = (60, 150)
HORIZON, CPS, SEG = 90, (10, 30, 60, 90), 3


def destination_tiles(raw):
    return [(t["coord"][0], t["coord"][1], t.get("groupIndex")) for t in raw["tiles"] if t["contentType"] == "Destination"]


def house_tiles(raw):
    return [(t["coord"][0], t["coord"][1], t.get("groupIndex")) for t in raw["tiles"] if t["contentType"] == "House"]


def _dist_to_dests(x, y, dests):
    return min((abs(x - dx) + abs(y - dy) for dx, dy, _ in dests), default=99)


def plan_spawn_branches(raw, legal_actions: List[int]) -> Dict[str, Any]:
    segs = H.horizontal_segments(raw, legal_actions, seg_len=SEG, touch_forbidden=True)
    dests = destination_tiles(raw)
    for s in segs:
        pts = [(s["x"] + e, s["y"]) for e in range(SEG + 1)]
        s["dist_to_dest"] = min(_dist_to_dests(px, py, dests) for px, py in pts)
    if len(segs) < 2:
        return {"skipped": "fewer than 2 legal segments", "n_segments": len(segs)}
    far = max(segs, key=lambda s: (s["dist_to_dest"], -s["x"], -s["y"]))      # deterministic tie-break
    near = min(segs, key=lambda s: (s["dist_to_dest"], s["x"], s["y"]))
    if far["dist_to_dest"] - near["dist_to_dest"] < 3:
        return {"skipped": "far and near segments not separated by >= 3 tiles", "far": far["dist_to_dest"], "near": near["dist_to_dest"]}
    return {"branches": [{"name": "base", "actions": []}, {"name": "base2", "actions": []}, {"name": "block_far", "actions": far["actions"]}, {"name": "block_near", "actions": near["actions"]}],
            "far": far, "near": near, "n_segments": len(segs)}


def occupied_tiles(actions: List[int]):
    out = set()
    for a in actions:
        k = L.decode_action(int(a))
        if k[0] == "ADD":
            x, y, d = k[1], k[2], k[3]
            out |= {(x, y), (x + 1, y) if d == "E" else (x, y - 1)}
    return out


def classify_spawns(base_spawns, branch_spawns, intervention_actions):
    """per index-matched spawn: BLOCKED / SHIFTED / UNCHANGED (driveway tile = east neighbour of a house tile, the placer's default driveway side is not exported -> any 4-neighbour counts)"""
    occ = occupied_tiles(intervention_actions)
    out = []
    for b, o in zip(base_spawns, branch_spawns):
        same = b[1:] == o[1:]
        bt = (b[3], b[4])
        touched = bt in occ or any((bt[0] + dx, bt[1] + dy) in occ for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)))
        out.append("UNCHANGED" if same else ("BLOCKED" if touched else "SHIFTED"))
    return out


def strategic_metric(spawns, raw_at_anchor):
    """mean Manhattan distance of new houses to the nearest same-colour destination tile at the anchor (and of new destinations to the nearest anchor house); lower is better."""
    dests = destination_tiles(raw_at_anchor)
    houses = house_tiles(raw_at_anchor)
    vals = []
    for (_, typ, g, x, y) in spawns:
        if typ == "H":
            same = [d for d in dests if d[2] == g]
            vals.append(_dist_to_dests(x, y, same) if same else _dist_to_dests(x, y, dests))
        else:
            vals.append(min((abs(x - hx) + abs(y - hy) for hx, hy, _ in houses), default=99))
    return (sum(vals) / len(vals)) if vals else None


def pilot_request(branches):
    return H.fork_request(branches, horizon=HORIZON, checkpoints=CPS, intervention_seconds=0.3) | {"include_obs": True}
