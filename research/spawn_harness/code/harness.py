"""Spawn-steering causal harness (Python side). Works on fork records produced by the plugin's DIAG_FORK op (proposed, research/spawn_harness/plugin_patch/DIAG_FORK.cs) and on the
existing DIAG_SPAWN records (research/spawn_counterfactual/pilot.jsonl), which share the per-branch schema {editsAccepted, terminalStep, spawns:[[t,type,group,x,y]], checkpoints:[{t,houses,dests,prng,score}]}.
No game is launched by this module; `ForkSession` only builds requests / parses responses for a caller that owns an IpcClient (not used tonight).

Branch roles: 'base' and 'base2' (duplicate restores: equivalence check), 'block' (road segment on candidate spawn tiles), 'sham' (same-size road segment intended NOT to touch any spawn candidate:
validity is established post hoc by PRNG equality with base at the first checkpoint/spawn, never assumed), optional 'cost' bookkeeping from concrete fields.
Confounding rule: an intervention branch whose PRNG state differs from base at the first checkpoint that precedes any spawn difference is RNG-confounded (the placer's stream shifted);
spawn differences in such a branch cannot be attributed to geometry."""
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(ROOT, "python"))
from synthetic_rl import layout as L  # noqa: E402


# ------------------------------------------------------------------------------------------------ parsing
def branch_record(b: Dict[str, Any]) -> Dict[str, Any]:
    sp = b.get("spawns", [])
    if isinstance(sp, str):
        sp = json.loads(sp.replace("'", '"'))
    cps = b.get("checkpoints", [])
    if isinstance(cps, str):
        cps = json.loads(cps.replace("'", '"'))
    return {"edits_accepted": int(b.get("editsAccepted", 0)), "terminal_step": int(b.get("terminalStep", -1)), "spawns": [(float(s[0]), str(s[1]), int(s[2]), int(s[3]), int(s[4])) for s in sp],
            "checkpoints": [{"t": int(c["t"]), "houses": int(c.get("houses", -1)), "dests": int(c.get("dests", -1)), "prng": str(c.get("prng", "?")), "score": float(c.get("score", 0.0)), "concrete": int(c.get("concrete", -1))} for c in cps],
            "prng_after_edits": b.get("prngAfterEdits"), "prng_at_restore": b.get("prngAtRestore"), "concrete_at_restore": b.get("concreteAtRestore"), "concrete_after_edits": b.get("concreteAfterEdits"), "edits": b.get("edits")}


def parse_fork(rec: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {name: branch_record(b) for name, b in rec["branches"].items()}


# ------------------------------------------------------------------------------------------------ checks
def equivalent_initial_state(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """two restores of the same snapshot with identical (empty) edits must produce identical spawn sequences, PRNG states and scores at every checkpoint."""
    same_spawns = a["spawns"] == b["spawns"]
    same_prng = len(a["checkpoints"]) == len(b["checkpoints"]) and all(x["prng"] == y["prng"] for x, y in zip(a["checkpoints"], b["checkpoints"]))
    same_score = len(a["checkpoints"]) == len(b["checkpoints"]) and all(x["score"] == y["score"] for x, y in zip(a["checkpoints"], b["checkpoints"]))
    return {"equivalent": same_spawns and same_prng and same_score, "same_spawns": same_spawns, "same_prng": same_prng, "same_score": same_score, "same_terminal": a["terminal_step"] == b["terminal_step"]}


def first_spawn_difference(base: Dict[str, Any], other: Dict[str, Any]) -> Optional[float]:
    for s, o in zip(base["spawns"], other["spawns"]):
        if s != o:
            return min(s[0], o[0])
    if len(base["spawns"]) != len(other["spawns"]):
        n = min(len(base["spawns"]), len(other["spawns"]))
        rest = (base["spawns"] + other["spawns"])[n:]
        return min((x[0] for x in rest), default=None) if rest else None
    return None


def rng_confounded(base: Dict[str, Any], other: Dict[str, Any]) -> Dict[str, Any]:
    """RNG-confounded iff the PRNG state differs from base either right after the edits (DIAG_FORK provides prngAfterEdits) or at the first checkpoint that lies at or before the
    first spawn difference. If the first PRNG difference appears only AFTER spawns already differed, the stream shift is a consequence of the spawn difference, not its cause (not confounded)."""
    fsd = first_spawn_difference(base, other)
    if other.get("prng_after_edits") and base.get("prng_after_edits") and other["prng_after_edits"] != base["prng_after_edits"]:
        return {"confounded": True, "evidence": "prng differs immediately after the edits", "first_spawn_difference_t": fsd}
    for x, y in zip(base["checkpoints"], other["checkpoints"]):
        if x["prng"] != y["prng"]:
            if fsd is None or x["t"] <= fsd:
                return {"confounded": True, "evidence": f"prng differs at checkpoint t={x['t']} (first spawn difference at {fsd})", "first_spawn_difference_t": fsd}
            return {"confounded": False, "evidence": f"prng first differs at t={x['t']} after the first spawn difference at {fsd} (downstream shift)", "first_spawn_difference_t": fsd}
    return {"confounded": False, "evidence": "prng identical at every checkpoint", "first_spawn_difference_t": fsd}


# ------------------------------------------------------------------------------------------------ metrics
def spawn_metrics(base: Dict[str, Any], other: Dict[str, Any], horizon_t: Optional[float] = None) -> Dict[str, Any]:
    bs = [s for s in base["spawns"] if horizon_t is None or s[0] <= horizon_t]
    os_ = [s for s in other["spawns"] if horizon_t is None or s[0] <= horizon_t]
    bset, oset = {s[1:] for s in bs}, {s[1:] for s in os_}
    idx_disp = [abs(a[3] - b[3]) + abs(a[4] - b[4]) for a, b in zip(bs, os_) if a[1] == b[1]]
    return {"n_spawns_base": len(bs), "n_spawns_other": len(os_), "set_overlap": len(bset & oset), "set_only_base": len(bset - oset), "set_only_other": len(oset - bset), "index_matched_manhattan_displacement": idx_disp,
            "mean_index_displacement": (sum(idx_disp) / len(idx_disp)) if idx_disp else None, "first_spawn_difference_t": first_spawn_difference(base, other)}


def score_deltas(base: Dict[str, Any], other: Dict[str, Any]) -> Dict[int, float]:
    return {y["t"]: y["score"] - x["score"] for x, y in zip(base["checkpoints"], other["checkpoints"])}


def intervention_cost(b: Dict[str, Any]) -> Optional[int]:
    if b.get("concrete_at_restore") is not None and b.get("concrete_after_edits") is not None and b["concrete_at_restore"] >= 0 and b["concrete_after_edits"] >= 0:
        return int(b["concrete_at_restore"]) - int(b["concrete_after_edits"])
    return None


def analyse(rec: Dict[str, Any], base_name="base", base2_name="base2", interventions=("block", "sham")) -> Dict[str, Any]:
    br = parse_fork(rec)
    base = br[base_name]
    out = {"equivalence": equivalent_initial_state(base, br[base2_name]) if base2_name in br else None, "branches": {}}
    for name in interventions:
        if name not in br:
            continue
        b = br[name]
        out["branches"][name] = {"edits_accepted": b["edits_accepted"], "cost": intervention_cost(b), "confound": rng_confounded(base, b), "spawns": spawn_metrics(base, b), "score_delta": score_deltas(base, b),
                                 "terminal_step_delta": (b["terminal_step"] - base["terminal_step"]) if (b["terminal_step"] > 0 and base["terminal_step"] > 0) else None}
    return out


# ------------------------------------------------------------------------------------------------ branch planning from a raw observation (no game needed)
def legal_add_tiles(legal_actions: List[int]) -> Dict[Tuple[int, int], set]:
    d: Dict[Tuple[int, int], set] = {}
    for a in legal_actions:
        k = L.decode_action(int(a))
        if k[0] == "ADD":
            d.setdefault((k[1], k[2]), set()).add(k[3])
    return d


def horizontal_segments(raw: Dict[str, Any], legal_actions: List[int], seg_len: int = 3, near=(2, 5), touch_forbidden=True):
    """candidate straight E-segments (seg_len edges) with every edge legal, not touching (4-adjacent) any road/building/carpark tile; returns dicts with Manhattan distance to the nearest building."""
    net, bld = set(), []
    for t in raw["tiles"]:
        x, y = t["coord"]
        any_road = any(v != "None" for v in t["road"].values())
        if t["contentType"] in ("House", "Destination"):
            bld.append((x, y))
        if any_road or t["contentType"] in ("House", "Destination", "Carpark"):
            net.add((x, y))
    tiles = legal_add_tiles(legal_actions)
    segs = []
    for (x, y), dirs in tiles.items():
        if "E" not in dirs:
            continue
        if not all("E" in tiles.get((x + e, y), set()) for e in range(seg_len)):
            continue
        pts = [(x + e, y) for e in range(seg_len + 1)]
        if touch_forbidden and any((p in net) or ((p[0] + dx, p[1] + dy) in net) for p in pts for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))):
            continue
        dmin = min((abs(bx - px) + abs(by - py) for bx, by in bld for px, py in pts), default=999)
        segs.append({"x": x, "y": y, "dist_to_building": dmin, "actions": [L.action_index("ADD", x + e, y, "E") for e in range(seg_len)]})
    return segs


def plan_branches(raw: Dict[str, Any], legal_actions: List[int], seg_len: int = 3, near=(2, 5), far_min: int = 10, rng=None) -> Dict[str, Any]:
    """block = segment near buildings (likely spawn candidates); sham = farthest available segment (its RNG-neutrality is verified post hoc, never assumed)."""
    import random
    rng = rng or random.Random(0)
    segs = horizontal_segments(raw, legal_actions, seg_len)
    nears = [s for s in segs if near[0] <= s["dist_to_building"] <= near[1]]
    fars = sorted([s for s in segs if s["dist_to_building"] >= far_min], key=lambda s: -s["dist_to_building"])
    if not nears:
        return {"skipped": "no_block_segment", "n_segments": len(segs)}
    block = rng.choice(nears)
    sham = rng.choice(fars) if fars else (max(segs, key=lambda s: s["dist_to_building"]) if segs else None)
    branches = [{"name": "base", "actions": []}, {"name": "base2", "actions": []}, {"name": "block", "actions": block["actions"]}]
    if sham:
        branches.append({"name": "sham", "actions": sham["actions"]})
    return {"branches": branches, "block": block, "sham": sham, "sham_is_far": bool(fars), "n_segments": len(segs), "n_near": len(nears), "n_far": len(fars)}


def fork_request(branches: List[Dict[str, Any]], horizon: int = 120, checkpoints=(30, 60, 90, 120), intervention_seconds: float = 0.3, n_pre: int = 0) -> Dict[str, Any]:
    return {"op": "DIAG_FORK", "n_pre": n_pre, "horizon": horizon, "checkpoints": list(checkpoints), "intervention_seconds": intervention_seconds, "branches": branches}


class ForkSession:
    """thin driver for a caller that owns a live IpcClient: play the main line with any policy via normal STEP calls, then fork. NOT exercised tonight (no DIAG_FORK in the deployed DLL)."""

    def __init__(self, client):
        self.client = client

    def fork(self, branches, **kw):
        req = fork_request(branches, **kw)
        op = req.pop("op")
        return self.client.send(op, **req)
