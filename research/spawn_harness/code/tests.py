"""Offline tests for the spawn harness (no game). Fixtures: the 19 recorded DIAG_SPAWN forks of the stopped pilot (research/spawn_counterfactual/pilot.jsonl: branches A, A2 = duplicate WAIT
restores, B = near disconnected segment, C = 'far' segment) plus synthetic fork records.   python research/spawn_harness/code/tests.py -> research/spawn_harness/tests.json"""
import copy
import gzip
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import harness as H  # noqa: E402
from synthetic_rl import layout as L  # noqa: E402

OUT = os.path.join(H.ROOT, "research", "spawn_harness")
PILOT = os.path.join(H.ROOT, "research", "spawn_counterfactual", "pilot.jsonl")
R = {}


def check(name, ok, detail):
    R[name] = {"passed": bool(ok), "detail": detail}
    print(("PASS " if ok else "FAIL ") + name + ": " + str(detail)[:400], flush=True)


def load_pilot():
    recs = [json.loads(l) for l in open(PILOT)]
    return [r for r in recs if "branches" in r]


def t1_equivalence_on_pilot():
    recs = load_pilot()
    eq = [H.equivalent_initial_state(*(lambda br: (br["A"], br["A2"]))(H.parse_fork(r))) for r in recs]
    n_ok = sum(e["equivalent"] for e in eq)
    check("T1_duplicate_restores_are_equivalent_on_pilot_records", n_ok == len(eq), {"records": len(eq), "equivalent": n_ok, "failures": [e for e in eq if not e["equivalent"]][:3]})


def t2_confound_detection_on_pilot():
    recs = load_pilot()
    rows = []
    for r in recs:
        br = H.parse_fork(r)
        for name in ("B", "C"):
            c = H.rng_confounded(br["A"], br[name])
            rows.append({"seed": r.get("seed"), "anchor": r.get("anchor"), "branch": name, **c})
    nB = sum(1 for x in rows if x["branch"] == "B" and x["confounded"])
    nC = sum(1 for x in rows if x["branch"] == "C" and x["confounded"])
    # the pilot notes recorded PRNG differences in 12/19 C branches (any checkpoint); our stricter rule only counts differences at/before the first spawn difference
    any_prng_diff_C = sum(1 for r in recs if any(x["prng"] != y["prng"] for x, y in zip(H.parse_fork(r)["A"]["checkpoints"], H.parse_fork(r)["C"]["checkpoints"])))
    check("T2_rng_confounding_detected_on_pilot", any_prng_diff_C >= 1 and nC >= 1, {"records": len(recs), "B_confounded": nB, "C_confounded": nC, "C_any_prng_difference": any_prng_diff_C, "pilot_notes_claim": "12/19 C branches"})
    R["T2_rows"] = rows


def t3_synthetic_fixtures():
    base = {"editsAccepted": 0, "terminalStep": -1, "spawns": [[10, "H", 0, 1, 1], [40, "D", 1, 5, 5]], "checkpoints": [{"t": 30, "houses": 1, "dests": 1, "prng": "P1", "score": 1}, {"t": 60, "houses": 2, "dests": 2, "prng": "P2", "score": 3}]}
    rec = {"branches": {"base": base, "base2": copy.deepcopy(base)}}
    # clean geometric effect: spawn differs at t=40, prng identical at t=30, differs only at t=60 (downstream)
    clean = copy.deepcopy(base); clean["spawns"][1] = [40, "D", 1, 7, 5]; clean["checkpoints"][1]["prng"] = "P2x"; clean["editsAccepted"] = 3
    clean["concreteAtRestore"] = 25; clean["concreteAfterEdits"] = 22
    # confounded: prng already differs at t=30 (< first spawn difference 40)
    conf = copy.deepcopy(clean); conf["checkpoints"][0]["prng"] = "P1x"
    # confounded via prngAfterEdits
    conf2 = copy.deepcopy(clean); conf2["prngAfterEdits"] = "Q"; base2 = copy.deepcopy(base); base2["prngAfterEdits"] = "P0"
    rec["branches"].update({"block": clean, "sham": conf})
    rec["branches"]["base"]["prngAfterEdits"] = "P0"
    a = H.analyse(rec)
    ok = a["equivalence"]["equivalent"] and not a["branches"]["block"]["confound"]["confounded"] and a["branches"]["sham"]["confound"]["confounded"] and a["branches"]["block"]["cost"] == 3 \
        and a["branches"]["block"]["spawns"]["first_spawn_difference_t"] == 40 and a["branches"]["block"]["spawns"]["mean_index_displacement"] == 1.0 and a["branches"]["block"]["score_delta"] == {30: 0.0, 60: 0.0}
    c2 = H.rng_confounded(H.branch_record(rec["branches"]["base"]), H.branch_record(conf2))
    # non-equivalent duplicate restore must be flagged
    bad = copy.deepcopy(base); bad["spawns"][0] = [10, "H", 0, 2, 1]
    ne = H.equivalent_initial_state(H.branch_record(base), H.branch_record(bad))
    check("T3_synthetic_fixtures", ok and c2["confounded"] and not ne["equivalent"], {"analysis": {k: (v if k != "branches" else {n: {kk: vv for kk, vv in b.items() if kk != "spawns"} for n, b in v.items()}) for k, v in a.items()}, "prng_after_edits_case": c2, "non_equivalent_detected": not ne["equivalent"]})


def t4_branch_planning_on_native_state():
    caps = [json.loads(l) for l in gzip.open(os.path.join(H.ROOT, "research", "s7_long_route", "native_normal_start_captures.jsonl.gz"), "rt")]
    plans = []
    for c in caps:
        st = c.get("first_house") or c["reset"]
        p = H.plan_branches(st["raw"], st["legal"], seg_len=3, near=(2, 5), far_min=8)
        plans.append(p)
    ok_plans = [p for p in plans if "branches" in p]
    legal_ok = all(all(a in set(st["legal"]) for b in p["branches"] for a in b["actions"]) for p, c in zip(plans, caps) for st in [c.get("first_house") or c["reset"]] if "branches" in p)
    req = H.fork_request(ok_plans[0]["branches"]) if ok_plans else None
    check("T4_branch_plans_use_only_legal_actions_and_build_valid_requests", len(ok_plans) > 0 and legal_ok and req is not None and req["op"] == "DIAG_FORK" and len(req["branches"]) >= 3,
          {"native_states": len(caps), "plannable": len(ok_plans), "skipped": [p.get("skipped") for p in plans if "skipped" in p][:5], "example_request": req and {k: (v if k != "branches" else [(b["name"], len(b["actions"])) for b in v]) for k, v in req.items()},
           "sham_far_available": sum(1 for p in ok_plans if p["sham_is_far"]), "note": "sham RNG-neutrality is verified post hoc by PRNG equality, never assumed"})


def main():
    for f in (t1_equivalence_on_pilot, t2_confound_detection_on_pilot, t3_synthetic_fixtures, t4_branch_planning_on_native_state):
        try:
            f()
        except Exception as ex:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            check(f.__name__, False, repr(ex))
    json.dump(R, open(os.path.join(OUT, "tests.json"), "w"), indent=1, default=str)
    print(sum(1 for v in R.values() if isinstance(v, dict) and v.get("passed")), "passed of", sum(1 for v in R.values() if isinstance(v, dict) and "passed" in v))


if __name__ == "__main__":
    main()
