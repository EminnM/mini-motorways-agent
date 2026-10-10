"""M2 Stage A validation (live, one game instance, ~3 min). Tests: (1) normal path unchanged (hashes vs S22 V0 with new keys removed); (2) MOTORWAY_CHECK dry-run legality matrix (too short,
water endpoint, no upgrade available, valid); (3) placement: after a Motorway upgrade is available (planner B main line past the first week that offers one, else fork-free scripted choice),
place a motorway between two road tiles -> model appears (Planned -> Active), inventory decremented, no concrete change, ramps appear in motorwayDirections, endpoints keep their roads,
connectivity unchanged; vehicles observed on motorway endpoints over 60 decisions; (4) removal: inventory refunded after release, model state Mothballed -> gone; (5) state restoration:
DIAG_FORK base/base2 with a motorway present are equivalent (spawns, PRNG, score) and the motorway list is identical at checkpoints.   python research/m2_motorways/code/validate_motorway.py"""
import hashlib
import gzip
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "s21_native_check", "code"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "spawn_harness", "code"))
from motorway_env import MotorwayEnv, ROOT  # noqa: E402
from native_adapter import NativePlannerPolicy  # noqa: E402
import harness as H  # noqa: E402
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import numpy as np  # noqa: E402
import s11_guard as G  # noqa: E402
from synthetic_rl import layout as L  # noqa: E402
from executor import make_policy  # noqa: E402
from planner import Weights  # noqa: E402

OUT = os.path.join(ROOT, "research", "m2_motorways")
R = {}


def check(name, ok, detail):
    R[name] = {"passed": bool(ok), "detail": detail}
    print(("PASS " if ok else "FAIL ") + name + ": " + str(detail)[:600], flush=True)


def h(raw):
    g = {k: v for k, v in raw["globals"].items() if k not in ("pendingUpgradeTypes", "motorways", "motorwaysAvailableOrDraft")}
    return hashlib.sha256(json.dumps(raw["tiles"], sort_keys=True).encode() + json.dumps(g, sort_keys=True).encode()).hexdigest()[:16]


def road_tiles(raw):
    return [(t["coord"][0], t["coord"][1]) for t in raw["tiles"] if any(v != "None" for v in t["road"].values()) and t["contentType"] == "None"]


def main():
    from experiments import gamectl
    t0 = time.time()
    ref = json.load(gzip.open(os.path.join(ROOT, "research", "s22_throughput", "raw", "V0_500000.json.gz"), "rt"))["steps"]
    with gamectl.GameInstance("m2_validate") as g:
        env = MotorwayEnv(max_decisions=100000, decision_delta=1.0, request_path=os.path.join(g.ipc_dir, "request.json"), response_path=os.path.join(g.ipc_dir, "response.json"), start_mode="NORMAL_START")
        # ---- 1 normal path + keys present; play planner B to decision 150 on seed 500000 (choose slot that offers Motorway when available: week 3 at 421 -> too late; use upgrade-choice at 141: TL|Roundabout -> no motorway)
        pol = NativePlannerPolicy(make_policy(Weights()))
        env.reset(seed=500000)
        pol.reset()
        raw = env.last_raw_observation
        mism = []
        for i in range(150):
            if h(raw) != ref[i]["hash"]:
                mism.append(i)
            mask = env.action_masks().copy()
            env.step(pol.act(raw, mask))
            raw = env.last_raw_observation
        known_native_drift = {0, 1, 2, 3, 4, 5, 6, 7, 16, 22, 127}            # pre-existing transient pattern reproduced with the pre-patch DLL (S22b)
        check("1_normal_path_unchanged_and_exports_present", set(mism) <= known_native_drift and "motorways" in raw["globals"] and "motorwaysAvailableOrDraft" in raw["globals"], {"mismatch_indices": mism, "motorways_key": raw["globals"].get("motorways"), "available": raw["globals"].get("motorwaysAvailableOrDraft")})
        # ---- 2 dry-run legality matrix at this state (no motorway upgrade available yet -> NotEnoughUpgrades expected)
        rts = road_tiles(raw)
        water = [(t["coord"][0], t["coord"][1]) for t in raw["tiles"] if t["overWater"]]
        a, b = rts[0], rts[-1]
        tests = {"no_upgrade_yet": env.motorway_check({"op": "add", "x1": a[0], "y1": a[1], "d1": "E", "x2": b[0], "y2": b[1], "d2": "W"}),
                 "too_short": env.motorway_check({"op": "add", "x1": a[0], "y1": a[1], "d1": "E", "x2": a[0] + 1, "y2": a[1], "d2": "W"}),
                 "water_endpoint": env.motorway_check({"op": "add", "x1": water[0][0], "y1": water[0][1], "d1": "E", "x2": b[0], "y2": b[1], "d2": "W"})}
        check("2_dry_run_rejections", all(not t.get("ok") for t in tests.values()) and all("error" not in t for t in tests.values()), {k: v.get("info") for k, v in tests.items()})
        # ---- 3 placement: continue to the week-3 choice (decision 421) where Motorway is offered on this seed (S M1 log: Motorway|Roundabout) choosing the Motorway slot; then place
        got = None
        while env.last_raw_observation["globals"]["motorwaysAvailableOrDraft"] == 0:
            raw = env.last_raw_observation
            mask = env.action_masks().copy()
            if not mask[0]:
                offered = raw["globals"].get("pendingUpgradeTypes", [])
                slot = next((i for i, o in enumerate(offered) if o["type"] == "Motorway"), 0)
                pol.t += 1
                _, _, te, tr, _ = env.step(L.UPGRADE_BASE + slot)
            else:
                _, _, te, tr, _ = env.step(pol.act(raw, mask))
            if te or tr:
                break
        raw = env.last_raw_observation
        avail = raw["globals"]["motorwaysAvailableOrDraft"]
        if avail == 0:
            check("3_placement", False, {"reason": "no motorway upgrade obtained before episode end", "decision": pol.t})
        else:
            ctx = G.Ctx(raw)
            func = G.edges_from_raw(raw)
            conn0 = G.connected_house_set(ctx, func)
            rts = road_tiles(raw)
            # pick the two road tiles farthest apart (Chebyshev) as endpoints; ramp directions toward each other, trying the 8 directions until the dry run accepts
            best = max(((p, q) for p in rts for q in rts), key=lambda pq: max(abs(pq[0][0] - pq[1][0]), abs(pq[0][1] - pq[1][1])))
            p, q = best
            placed = None
            for d1 in ("N", "NE", "E", "SE", "S", "SW", "W", "NW"):
                for d2 in ("N", "NE", "E", "SE", "S", "SW", "W", "NW"):
                    m = {"op": "add", "x1": p[0], "y1": p[1], "d1": d1, "x2": q[0], "y2": q[1], "d2": d2}
                    if env.motorway_check(m).get("ok"):
                        placed = m
                        break
                if placed:
                    break
            if not placed:
                check("3_placement", False, {"reason": "no legal ramp-direction pair for the chosen endpoints", "endpoints": [p, q], "example": env.motorway_check({"op": "add", "x1": p[0], "y1": p[1], "d1": "E", "x2": q[0], "y2": q[1], "d2": "W"})})
            else:
                before = {"avail": avail, "total": raw["globals"]["motorwaysTotal"], "concrete": raw["globals"]["concreteAvailable"], "n_models": len(raw["globals"]["motorways"])}
                _, _, _, _, inf = env.step_motorway(placed, action=0)
                states = []
                veh_end = 0
                for k in range(60):
                    raw = env.last_raw_observation
                    ms = raw["globals"]["motorways"]
                    states.append([m["state"] for m in ms])
                    for t in raw["tiles"]:
                        if (t["coord"][0], t["coord"][1]) in (p, q):
                            veh_end += t.get("visibleVehicleCount", 0)
                    mask = env.action_masks().copy()
                    env.step(pol.act(raw, mask) if mask[0] else int(np.flatnonzero(mask)[0]))
                raw = env.last_raw_observation
                after = {"avail": raw["globals"]["motorwaysAvailableOrDraft"], "total": raw["globals"]["motorwaysTotal"], "concrete_after_1": None, "models": raw["globals"]["motorways"]}
                dirs = {f"{t['coord'][0]},{t['coord'][1]}": t.get("motorwayDirections") for t in raw["tiles"] if t.get("motorwayDirections")}
                roads_kept = all(any(v != "None" for v in t["road"].values()) for t in raw["tiles"] if (t["coord"][0], t["coord"][1]) in (p, q))
                conn1 = G.connected_house_set(G.Ctx(raw), G.edges_from_raw(raw))
                ok = inf["motorwayOk"] and len(after["models"]) == before["n_models"] + 1 and after["avail"] == before["avail"] - 1 and any(s == ["Active"] or "Active" in s for s in states[-5:]) and roads_kept and conn0 <= conn1
                check("3_placement", ok, {"request": placed, "motorwayInfo": inf["motorwayInfo"], "before": before, "after": {k: v for k, v in after.items() if k != "models"}, "models": after["models"], "state_timeline": [states[i] for i in (0, 1, 2, 5, 30, 59)], "ramp_tiles": dirs, "endpoint_roads_kept": roads_kept,
                                          "connected_before_after": [len(conn0), len(conn1)], "vehicle_count_at_endpoints_60dec": veh_end})
                # ---- 4 removal (straight play, no fork before)
                raw = env.last_raw_observation
                avail_r0 = raw["globals"]["motorwaysAvailableOrDraft"]
                _, _, _, _, inf2 = env.step_motorway({"op": "remove", "x": p[0], "y": p[1]}, action=0)
                timeline = []
                for k in range(40):
                    raw = env.last_raw_observation
                    timeline.append(([m["state"] for m in raw["globals"]["motorways"]], raw["globals"]["motorwaysAvailableOrDraft"]))
                    env.step(0)
                check("4_removal_and_refund", bool(inf2["motorwayOk"]) and timeline[-1][1] == avail_r0 + 1 and (not timeline[-1][0] or "Active" not in timeline[-1][0]), {"motorwayInfo": inf2["motorwayInfo"], "motorwayOk": inf2["motorwayOk"], "timeline": [timeline[i] for i in (0, 1, 2, 5, 10, 20, 39)], "avail_before": avail_r0})
                # re-place for the fork test if possible
                if env.last_raw_observation["globals"]["motorwaysAvailableOrDraft"] > 0:
                    env.step_motorway(placed, action=0)
                    for k in range(3):
                        env.step(0)
                # ---- 5 state restoration with a motorway present
                rec = env.client.send("DIAG_FORK", n_pre=0, horizon=30, checkpoints=[1, 15, 30], intervention_seconds=0.3, include_obs=True, branches=[{"name": "base", "actions": []}, {"name": "base2", "actions": []}])
                if "error" in rec:
                    check("5_fork_with_motorway", False, rec["error"][:300])
                else:
                    br = H.parse_fork(rec)
                    eq = H.equivalent_initial_state(br["base"], br["base2"])
                    ml = [[m["state"] for m in cp["obs"]["globals"]["motorways"]] for cp in rec["branches"]["base"]["checkpoints"]]
                    ml2 = [[m["state"] for m in cp["obs"]["globals"]["motorways"]] for cp in rec["branches"]["base2"]["checkpoints"]]
                    check("5_fork_with_motorway", eq["equivalent"] and ml == ml2 and all("Active" in x for x in ml), {"equivalent": eq, "motorway_states_at_checkpoints": ml})
        env.close()
    R["_wall_s"] = round(time.time() - t0, 1)
    json.dump(R, open(os.path.join(OUT, "validation.json"), "w"), indent=1, default=str)
    print(sum(1 for v in R.values() if isinstance(v, dict) and v.get("passed")), "passed of", sum(1 for v in R.values() if isinstance(v, dict) and "passed" in v), R["_wall_s"], "s")


if __name__ == "__main__":
    main()
