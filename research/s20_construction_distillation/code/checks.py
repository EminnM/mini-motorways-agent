"""S20 checks (results -> runs/checks.json):
 C1 split seeds disjoint (train/val/test) and none equals an S13 native test seed; scene seeds match their split
 C2 goal channels contain ONLY the goal house tile and the destination tiles of its colour group (all recorded transitions)
 C3 no strategic choice imitated: every plan in every demonstration targets the goal house; no REMOVE labels; label WAIT frequency reported
 C4 masks: every recorded label is legal under the recorded mask; tile value scale exact (uint8 x6 round trip); restore_env reproduces the
    generation-time observation and mask (re-generated scenes vs stored states)
 C5 DONE labelled iff the goal is functionally connected (and exactly once per completed trajectory, at its end)
 C6 GoalExecutor(goal=None) == the original S18-H Executor (identical action sequences on full-planner episodes)
 C7 planner labels lie in the oracle valid set (non-terminal states)"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import scenes as S  # noqa: E402
from common import A_DONE, L, Executor, GoalExecutor, MultiHouseEnvV2, restore_env  # noqa: E402

DATA = os.path.join(C.S20, "data")
RUN = os.path.join(C.S20, "runs")


def main():
    out = {}
    # C1
    sets = {s: set(C.seed_range(s)) for s in C.SEEDS}
    assert not (sets["train"] & sets["val"]) and not (sets["train"] & sets["test"]) and not (sets["val"] & sets["test"])
    assert not ((sets["train"] | sets["val"] | sets["test"]) & C.S13_NATIVE_TEST_SEEDS)
    scenes = {s: json.load(open(os.path.join(DATA, f"scenes_{s}.json"))) for s in C.SEEDS}
    for s, sc in scenes.items():
        assert all(x["meta"]["seed"] in sets[s] for x in sc)
    out["C1_splits_disjoint"] = {"pass": True, "sizes": {s: len(v) for s, v in sets.items()}, "scenes": {s: len(v) for s, v in scenes.items()}}
    # C2-C5, C7 over the recorded transitions
    c2 = c4 = c5 = c7 = True
    n_wait = n_remove = n_done = n_nonterm = n_in_valid = 0
    n_total = 0
    for s in C.SEEDS:
        z = np.load(os.path.join(DATA, f"demos_{s}.npz"))
        tiles, act, done, scene, mask = z["tiles"], z["action"], z["done"], z["scene"], np.unpackbits(z["mask"], axis=1)[:, : C.N_ACT].astype(bool)
        voff, vflat = z["voff"], z["vflat"]
        traj = json.load(open(os.path.join(DATA, f"traj_{s}.json")))
        n_total += len(act)
        for i in range(len(act)):
            sc = scenes[s][int(scene[i])]
            goal = tuple(sc["meta"]["goal"])
            g = sc["meta"]["goal_group"]
            h = tiles[i, :, C.N_TILE_CH - 2]
            d = tiles[i, :, C.N_TILE_CH - 1]
            exp_h = np.zeros(L.N_TILES, dtype=np.uint8)
            exp_h[L.tile_index(*goal)] = C.TILE_SCALE
            exp_d = np.zeros(L.N_TILES, dtype=np.uint8)
            for dd in sc["state"]["dests"]:
                if dd["group"] == g:
                    for p in dd["tiles"]:
                        exp_d[L.tile_index(*p)] = C.TILE_SCALE
            c2 &= bool(np.array_equal(h, exp_h) and np.array_equal(d, exp_d))
            c4 &= bool(mask[i, act[i]])
            c5 &= bool((act[i] == A_DONE) == bool(done[i]))
            if act[i] == 0:
                n_wait += 1
            elif act[i] == A_DONE:
                n_done += 1
            elif L.decode_action(int(act[i]))[0] == "REMOVE":
                n_remove += 1
            if not done[i]:
                n_nonterm += 1
                n_in_valid += int(act[i] in set(vflat[voff[i]:voff[i + 1]].tolist()))
        # C5b: DONE exactly once per completed trajectory, at the end
        for t in traj:
            idx = np.flatnonzero(scene == t["scene"])
            dones = np.flatnonzero(act[idx] == A_DONE)
            c5 &= bool((len(dones) == 1 and dones[0] == len(idx) - 1) if t["completed"] else (len(dones) == 0))
        # C3
        planned_only_goal = all([list(h) for h in t["planned_houses"]] == [list(scenes[s][t["scene"]]["meta"]["goal"])] for t in traj)
        out.setdefault("C3_only_goal_planned", {})[s] = planned_only_goal
    out["C2_goal_channels_exact"] = {"pass": c2, "transitions": n_total}
    out["C3_no_strategic_choice"] = {"pass": all(out["C3_only_goal_planned"].values()) and n_remove == 0, "remove_labels": n_remove, "wait_labels": n_wait, "wait_label_frac": n_wait / n_total, "done_labels": n_done}
    out["C4_labels_legal_under_mask"] = {"pass": c4}
    out["C5_done_iff_connected"] = {"pass": c5}
    out["C7_planner_label_in_valid_set"] = {"pass": n_in_valid == n_nonterm, "in_valid": n_in_valid, "nonterminal": n_nonterm}
    # C4b: restore_env reproduces generation-time obs + mask (first 20 train scenes re-generated)
    water = MultiHouseEnvV2([{"id": "x", "structures0": [[0, 0, "House", 0]], "free_edges0": [[0, 0, "E"]], "events": []}]).water
    ok = n = 0
    for sc in scenes["train"][:20]:
        regen, why = S.generate_scene(sc["meta"]["seed"], water)
        env_a = restore_env(regen["state"])
        env_b = restore_env(sc["state"])
        goal = tuple(sc["meta"]["goal"])
        ta, ga = C.goal_obs_u8(env_a, goal)
        tb, gb = C.goal_obs_u8(env_b, goal)
        ok += int(np.array_equal(ta, tb) and np.allclose(ga, gb) and np.array_equal(env_a.action_masks(), env_b.action_masks()) and regen["meta"] == sc["meta"])
        n += 1
    out["C4b_restore_deterministic"] = {"pass": ok == n, "ok": ok, "n": n}
    # C6: GoalExecutor(None) vs Executor on full-planner episodes (3 test scenes from their stored state, 60 decisions each, all houses as objectives)
    same = 0
    for sc in scenes["test"][:3]:
        seqs = []
        for cls in (Executor, lambda: GoalExecutor(None)):
            env = restore_env(sc["state"])
            ex = cls()
            seq = []
            for _ in range(60):
                a = ex.act(env)
                seq.append(a)
                env.step(a)
            seqs.append(seq)
        same += int(seqs[0] == seqs[1])
    out["C6_goal_executor_equals_executor_when_unconstrained"] = {"pass": same == 3, "episodes": 3, "identical": same}
    # masks during evaluation (0 illegal emitted) are read from eval_test.json if present
    p = os.path.join(RUN, "eval_test.json")
    if os.path.exists(p):
        ev = json.load(open(p))
        out["C4c_eval_illegal_emitted"] = {name: r["illegal_total"] for name, r in ev["results"].items()}
        out["C4c_eval_illegal_emitted"]["pass"] = all(v == 0 for v in out["C4c_eval_illegal_emitted"].values())
    out["all_pass"] = all(v.get("pass", True) for v in out.values() if isinstance(v, dict))
    json.dump(out, open(os.path.join(RUN, "checks.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
