"""Compare the synthetic adapter's assumptions with LIVE native observations/masks captured by experiments.s2.capture.
Machine-readable report: research/synthetic_to_native/interface_report.json. Separates ENCODING mismatches (an index /
channel / convention differs) from genuine STATE differences (the native game simply has things the sandbox lacks)."""
import glob
import gzip
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

from .. import paths
from ..gap1 import tasks as T

sys.path.insert(0, paths.PYTHON_DIR)
from synthetic_rl import layout as L  # noqa: E402
import mini_motorways_env as native  # noqa: E402

OUT = os.path.join(paths.ROOT, "research", "synthetic_to_native")
W, H = 37, 21
ROAD_ORD = {"None": 0, "Pending": 1, "Active": 2, "Mothballed": 3, "VisiblyActive": 2}


def independent_tile_features(t):
    """Re-derive the 15 tile features from a raw native tile using the DOCUMENTED meanings (the sandbox's assumptions)."""
    road = t["road"]
    content = {"House": 1, "Destination": 2}.get(t["contentType"], 0) / 2.0
    dest = t.get("destination") or {}
    return [1.0 if t["buildable"] else 0.0, 1.0 if t["overWater"] else 0.0, 1.0 if t["underMountain"] else 0.0, 1.0 if t["overRail"] else 0.0,
            ROAD_ORD[road["N"]] / 3.0, ROAD_ORD[road["E"]] / 3.0, ROAD_ORD[road["S"]] / 3.0, ROAD_ORD[road["W"]] / 3.0,
            1.0 if t["hasTrafficLight"] else 0.0, 1.0 if t["hasRoundabout"] else 0.0, content,
            1.0 if t.get("groupIndex") is not None else 0.0, min(max(dest.get("totalDemand", 0) / 50.0, 0.0), 1.0) if dest else 0.0,
            1.0 if dest.get("isOvercrowding") else 0.0, min(max(t.get("visibleVehicleCount", 0), 0), 10) / 10.0]


def load_states():
    states = []
    for p in sorted(glob.glob(os.path.join(OUT, "native_states_validation_*.jsonl.gz"))):
        for line in gzip.open(p, "rt", encoding="utf-8"):
            if line.strip():
                states.append(json.loads(line))
    return states


def synthetic_masks_from_native(raw, legal_indices):
    tiles = {(t["coord"][0], t["coord"][1]): t for t in raw["tiles"]}
    nat_add, nat_rem = set(), set()
    pts = []
    for a in legal_indices:
        d = L.decode_action(a)
        if d[0] in ("ADD", "REMOVE"):
            _, x, y, dd = d
            (nat_add if d[0] == "ADD" else nat_rem).add((x, y, dd))
            pts += [(x, y), L.neighbor(x, y, dd)]
    if not pts:
        return None
    wx0, wx1 = min(p[0] for p in pts), max(p[0] for p in pts)
    wy0, wy1 = min(p[1] for p in pts), max(p[1] for p in pts)
    live = lambda t, d: t["road"][d] in T.LIVE
    free = lambda xy: xy in tiles and wx0 <= xy[0] <= wx1 and wy0 <= xy[1] <= wy1 and tiles[xy]["contentType"] not in ("House", "Destination", "Carpark")
    syn_add, syn_rem = set(), set()
    for (x, y), t in tiles.items():
        for d in ("E", "S"):
            n = L.neighbor(x, y, d)
            if free((x, y)) and free(n):
                if not live(t, d):
                    syn_add.add((x, y, d))
                else:
                    if sum(1 for v in t["road"].values() if v in T.LIVE) == 1:
                        syn_rem.add((x, y, d))
    reasons = Counter()
    ex = {}
    for e in syn_add - nat_add:                      # synthetic says legal, native says no
        x, y, d = e
        why = []
        for xy in ((x, y), L.neighbor(x, y, d)):
            t = tiles[xy]
            if t["overWater"]:
                why.append("endpoint_over_water")
            if t["underMountain"]:
                why.append("endpoint_under_mountain")
            if t["overRail"]:
                why.append("endpoint_over_rail")
            if not t["buildable"]:
                why.append("endpoint_not_buildable")
            if t["contentType"] not in ("None",):
                why.append("endpoint_content_" + t["contentType"])
            if any(v in ("Mothballed",) for v in t["road"].values()):
                why.append("endpoint_has_mothballed_road")
        if not why:
            why = ["unexplained_by_terrain_content"]
        for w in sorted(set(why)):
            reasons[w] += 1
        ex.setdefault(tuple(sorted(set(why))), e)
    return {"window": [wx0, wy0, wx1, wy1], "native_add": len(nat_add), "synthetic_add": len(syn_add), "add_both": len(nat_add & syn_add),
            "native_only_add": len(nat_add - syn_add), "synthetic_only_add": len(syn_add - nat_add), "synthetic_only_reasons": dict(reasons),
            "native_rem": len(nat_rem), "synthetic_rem": len(syn_rem), "rem_both": len(nat_rem & syn_rem),
            "native_only_rem": len(nat_rem - syn_rem), "synthetic_only_rem": len(syn_rem - nat_rem)}


def main():
    states = load_states()
    caps = []
    for p in sorted(glob.glob(os.path.join(OUT, "capture_validation_*.jsonl"))):
        caps += [json.loads(l) for l in open(p) if l.strip()]
    for c in caps:                       # normalise: action_tests is [results, [(name, raw), ...]] in the stored records
        at = c.get("action_tests")
        if at and isinstance(at, list) and len(at) == 2 and isinstance(at[0], list) and (not at[0] or isinstance(at[0][0], dict)):
            for name, raw in at[1]:
                states.append({"task_id": c["task_id"], "kind": name, "raw": raw, "legal_indices": []})
            c["action_tests"] = at[0]
    rep = {"n_native_states": len(states), "n_tasks_captured": len(caps), "checks": {}, "encoding_mismatches": [], "state_differences": {}}
    enc_mis = rep["encoding_mismatches"]
    # ---- 1,2,3: sizes
    lens = {len(native.encode_observation(s["raw"])) for s in states}
    rep["checks"]["1_observation_length_11666"] = {"passed": lens == {11666} and all(c["obs_len"] == 11666 for c in caps), "observed_lengths": sorted(lens)}
    max_idx = max(max(s["legal_indices"]) for s in states if s["legal_indices"])
    rep["checks"]["2_action_space_3111"] = {"passed": L.N_ACTIONS == 3111 and max_idx < 3111, "max_native_legal_index": int(max_idx),
                                            "note": "Discrete(3111) is read from HELLO by MiniMotorwaysEnv (actionSpaceSize); every native legal index < 3111"}
    bounds = {json.dumps(s["raw"]["bounds"], sort_keys=True) for s in states}
    rep["checks"]["3_grid_37x21_bounds"] = {"passed": bounds == {json.dumps({"xMin": -18, "yMin": -10, "width": 37, "height": 21}, sort_keys=True)}, "observed": sorted(bounds)}
    # ---- 4,6: tile order + 15 channels vs independent derivation
    tile_order_bad = feat_bad = 0
    per_feature_bad = Counter()
    nz_native = np.zeros(15)
    n_tiles = 0
    for s in states:
        vec = native.encode_observation(s["raw"])
        grid = vec[: L.N_TILES * 15].reshape(H, W, 15)                         # the CNN's reshape convention
        for i, t in enumerate(s["raw"]["tiles"]):
            x, y = t["coord"]
            if (x, y) != (L.XMIN + i % W, L.YMIN + i // W):
                tile_order_bad += 1
            ind = independent_tile_features(t)
            got = grid[y - L.YMIN, x - L.XMIN]
            for j in range(15):
                if abs(float(got[j]) - ind[j]) > 1e-6:
                    feat_bad += 1
                    per_feature_bad[j] += 1
            nz_native += (got != 0)
            n_tiles += 1
    rep["checks"]["4_all_15_tile_features_mapped_to_correct_tile_and_channel"] = {"passed": feat_bad == 0 and tile_order_bad == 0, "tiles_checked": n_tiles,
                                                                                     "channel_mismatches": feat_bad, "per_channel": dict(per_feature_bad)}
    rep["checks"]["6_tile_coordinate_convention_row_major_y_outer_x_inner"] = {"passed": tile_order_bad == 0, "mismatching_tiles": tile_order_bad}
    if feat_bad or tile_order_bad:
        enc_mis.append({"tile_order": tile_order_bad, "feature_channels": dict(per_feature_bad)})
    # ---- 5: globals
    gbad = 0
    gstats = defaultdict(list)
    for s in states:
        vec = native.encode_observation(s["raw"])
        g = s["raw"]["globals"]
        gv = vec[L.N_TILES * 15:]
        exp = [g["hour"] / 24.0, min(g["day"], 365) / 365.0, min(g["week"], 52) / 52.0, g["score"] / 1000.0,
               native._ratio(g["concreteAvailable"], g["concreteTotal"]), native._ratio(g["bridgesAvailable"], g["bridgesTotal"]),
               native._ratio(g["tunnelsAvailable"], g["tunnelsTotal"]), native._ratio(g["motorwaysAvailable"], g["motorwaysTotal"]),
               native._ratio(g["trafficLightsAvailable"], g["trafficLightsTotal"]), native._ratio(g["roundaboutsAvailable"], g["roundaboutsTotal"]),
               min(g.get("pendingUpgradeChoices", 0), 3) / 3.0]
        for j in range(11):
            gstats[j].append(float(gv[j]))
            if abs(float(gv[j]) - exp[j]) > 1e-6:
                gbad += 1
    rep["checks"]["5_all_11_global_features_consistent"] = {"passed": gbad == 0, "mismatches": gbad,
                                                            "native_value_ranges": {j: [round(min(v), 3), round(max(v), 3)] for j, v in gstats.items()}}
    # ---- 7: South = y-1 (symmetry of road states)
    bad_s_minus, bad_s_plus, edges = 0, 0, 0
    for s in states:
        tiles = {(t["coord"][0], t["coord"][1]): t for t in s["raw"]["tiles"]}
        for (x, y), t in tiles.items():
            if t["road"]["S"] in T.LIVE:
                edges += 1
                n_minus, n_plus = tiles.get((x, y - 1)), tiles.get((x, y + 1))
                if not (n_minus and n_minus["road"]["N"] in T.LIVE):
                    bad_s_minus += 1
                if not (n_plus and n_plus["road"]["N"] in T.LIVE):
                    bad_s_plus += 1
    rep["checks"]["7_south_is_y_minus_1"] = {"passed": bad_s_minus == 0 and bad_s_plus > bad_s_minus, "south_edges_checked": edges,
                                             "mismatches_if_south_is_y_minus_1": bad_s_minus, "mismatches_if_south_is_y_plus_1": bad_s_plus}
    # ---- 8,9: live action mapping tests
    after_remove = {st["task_id"]: st["raw"] for st in states if st.get("kind") == "after_REMOVE"}
    tests = []
    for c in caps:
        for x in c.get("action_tests", []):
            x = dict(x)
            x["task_id"] = c["task_id"]
            tests.append(x)
    byk = defaultdict(list)
    for x in tests:
        byk[x["test"].split("(")[0].split("_")[0]].append(x["ok"])
    carpark_removes, other_fail = [], []
    for x in tests:
        if x["test"].startswith("REMOVE") and not x["ok"]:
            xs, ys, _ = x["test"].split("(")[1].split(")")[0].split(",")
            raw = after_remove.get(x["task_id"])
            ct = None
            if raw:
                ct = next(t["contentType"] for t in raw["tiles"] if tuple(t["coord"]) == (int(xs), int(ys)))
            (carpark_removes if ct == "Carpark" else other_fail).append({"test": x["test"], "task_id": x["task_id"], "tile_content": ct, "applied": x["applied"]})
    rem_total = len(byk.get("REMOVE", []))
    rep["checks"]["8_ADD_REMOVE_index_mapping_live"] = {
        "passed": bool(byk.get("ADD")) and all(byk["ADD"]) and not other_fail,
        "ADD_tests_ok": [sum(byk.get("ADD", [])), len(byk.get("ADD", []))],
        "REMOVE_tests_ok": [sum(byk.get("REMOVE", [])), rem_total],
        "REMOVE_failures_on_carpark_tiles": len(carpark_removes), "REMOVE_failures_other": len(other_fail),
        "interpretation": "index -> (tile, direction, kind) is correct in every case where the action acts; the failing REMOVE tests all target a CARPARK tile: the native mask lists REMOVE there, the game reports it applied, and the carpark stub road stays Active (legal-but-ineffective).",
        "carpark_remove_examples": carpark_removes[:5], "other_failures": other_fail[:5]}
    rep["state_differences"]["native_remove_on_carpark_is_ineffective"] = {"count": len(carpark_removes), "of_remove_tests": rem_total,
                                                                          "sandbox_rule": "stub edges can never be removed and REMOVE is never offered there"}
    rep["checks"]["9_WAIT_index0_live"] = {"passed": bool(byk.get("WAIT")) and all(byk["WAIT"]), "tests": [sum(byk.get("WAIT", [])), len(byk.get("WAIT", []))]}
    # ---- 10: mask semantics vs synthetic reconstruction
    agg = Counter()
    reasons = Counter()
    for s in states:
        m = synthetic_masks_from_native(s["raw"], s["legal_indices"])
        if m is None:
            continue
        for k in ("native_add", "synthetic_add", "add_both", "native_only_add", "synthetic_only_add", "native_rem", "synthetic_rem", "rem_both", "native_only_rem", "synthetic_only_rem"):
            agg[k] += m[k]
        for k, v in m["synthetic_only_reasons"].items():
            reasons[k] += v
        agg["states"] += 1
    wait_ok = all(0 in s["legal_indices"] for s in states if s.get("kind", "reset") == "reset")
    idx_ok = all(all(0 <= a < 3111 for a in s["legal_indices"]) for s in states)
    rep["checks"]["10_legal_mask_meaning_and_ordering"] = {
        "passed": wait_ok and idx_ok and byk.get("ADD") is not None, "wait_legal_in_all_reset_states": wait_ok,
        "note": "index -> (kind, tile, dir) mapping is validated live by checks 8/9; the set-level comparison below measures RULE differences between the sandbox and the native game, not ordering/meaning",
        "totals_over_states": dict(agg), "add_precision_synth_vs_native": agg["add_both"] / max(agg["synthetic_add"], 1), "add_recall_synth_vs_native": agg["add_both"] / max(agg["native_add"], 1),
        "remove_precision": agg["rem_both"] / max(agg["synthetic_rem"], 1), "remove_recall": agg["rem_both"] / max(agg["native_rem"], 1),
        "synthetic_only_add_reasons_endpoint_counts": dict(reasons)}
    # ---- state differences: channel population natively vs the sandbox
    from synthetic_rl.env import SyntheticGapEnv, load_tasks
    syn_tasks = load_tasks(os.path.join(paths.ROOT, "research", "synthetic_lab", "manifests", "gap1_train.jsonl"))[:200]
    senv = SyntheticGapEnv(syn_tasks, sampling="sequential")
    nz_syn = np.zeros(15)
    for i in range(len(syn_tasks)):
        o, _ = senv.reset(options={"task_index": i})
        nz_syn += (o[: L.N_TILES * 15].reshape(L.N_TILES, 15) != 0).sum(axis=0)
    names = ["buildable", "overWater", "underMountain", "overRail", "roadN", "roadE", "roadS", "roadW", "trafficLight", "roundabout", "content", "groupFlag", "destDemand", "overcrowding", "vehicles"]
    rep["state_differences"]["tile_channel_nonzero_fraction"] = {n: {"native": round(float(nz_native[j] / n_tiles), 4), "sandbox": round(float(nz_syn[j] / (len(syn_tasks) * L.N_TILES)), 4)} for j, n in enumerate(names)}
    rep["state_differences"]["global_features"] = {"native_ranges": rep["checks"]["5_all_11_global_features_consistent"]["native_value_ranges"],
                                                   "note": "sandbox: hour/day from t, week 0, concrete ratio, other resource ratios 0, pending upgrades 0"}
    rep["state_differences"]["legal_action_count"] = {"native_mean_at_reset": float(np.mean([len(s["legal_indices"]) for s in states if s.get("kind", "reset") == "reset"])), "sandbox_mean": 202}
    rep["verdict"] = {"encoding_mismatches": len(enc_mis), "all_checks_passed": all(v["passed"] for v in rep["checks"].values())}
    json.dump(rep, open(os.path.join(OUT, "interface_report.json"), "w"), indent=2)
    for k, v in rep["checks"].items():
        print(f"[{'PASS' if v['passed'] else 'FAIL'}] {k}: " + json.dumps({a: b for a, b in v.items() if a != 'passed'})[:240])
    print("state differences:", json.dumps(rep["state_differences"]["tile_channel_nonzero_fraction"]))
    print("legal count native vs sandbox:", rep["state_differences"]["legal_action_count"])


if __name__ == "__main__":
    main()
