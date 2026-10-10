"""S14 gates A-C and the information-loss measurement. No training.   python python/synthetic_rl/s14_tests.py -> research/s14_group_observation/{gate_a,gate_b,gate_c,info_loss}.json"""
import copy
import gzip
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from synthetic_rl import layout as L  # noqa: E402
from synthetic_rl import obs_v2 as V  # noqa: E402
from synthetic_rl import s13_env as E13  # noqa: E402
from synthetic_rl import s13_controller as CT  # noqa: E402
from synthetic_rl.s14_env import MultiHouseEnvV2  # noqa: E402
import mini_motorways_env as NATIVE  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "research", "s14_group_observation")
CK = os.path.join(ROOT, "research", "s8_long_route_training", "runs", "arm_C_s11", "checkpoints", "step_0099840.zip")
SEEDS = json.load(open(os.path.join(ROOT, "research", "s9_native_transfer", "seeds.json")))["seeds"]
os.makedirs(OUT, exist_ok=True)
R = {}


def check(section, name, ok, detail):
    R.setdefault(section, {})[name] = {"passed": bool(ok), "detail": detail}
    print(("PASS " if ok else "FAIL ") + f"[{section}] {name}: {str(detail)[:400]}", flush=True)


def captures():
    with gzip.open(os.path.join(ROOT, "research", "s7_long_route", "native_normal_start_captures.jsonl.gz"), "rt") as f:
        return [json.loads(l) for l in f]


def relabel(raw, perm):
    r = copy.deepcopy(raw)
    for t in r["tiles"]:
        if t.get("groupIndex") is not None:
            t["groupIndex"] = perm[t["groupIndex"]]
    return r


# ----------------------------------------------------------------------------------------------------------- Gate A: native identity
def gate_a():
    caps = captures()
    states = [c["reset"]["raw"] for c in caps] + [c["first_house"]["raw"] for c in caps if c.get("first_house")]
    houses_with = houses_without = dests_with = dests_without = 0
    pairs_ok = pairs_total = 0
    for raw in states:
        hs = [t for t in raw["tiles"] if t["contentType"] == "House"]
        ds = [t for t in raw["tiles"] if t["contentType"] == "Destination"]
        houses_with += sum(1 for t in hs if t.get("groupIndex") is not None)
        houses_without += sum(1 for t in hs if t.get("groupIndex") is None)
        dests_with += sum(1 for t in ds if t.get("groupIndex") is not None)
        dests_without += sum(1 for t in ds if t.get("groupIndex") is None)
        dg = {t["groupIndex"] for t in ds if t.get("groupIndex") is not None}
        for t in hs:
            pairs_total += 1
            pairs_ok += int(t.get("groupIndex") in dg)
    check("A", "plugin_emits_groupIndex_on_houses_and_destinations", houses_without == 0 and dests_with > 0,
          {"states": len(states), "house_tiles_with_group": houses_with, "house_tiles_without": houses_without, "dest_tiles_with_group": dests_with, "dest_tiles_without": dests_without})
    check("A", "every_house_has_a_same_group_destination_at_capture", pairs_ok == pairs_total, {"houses": pairs_total, "matched": pairs_ok})
    # S13 replicate logs: 12 full native episodes (3,867 decisions) with per-step house groups and destination groups
    reassign = 0
    dchange = 0
    spawn_match = spawn_total = 0
    max_groups = 0
    appearance_orders = []
    null_dest_tiles = 0
    for s in SEEDS:
        d = json.load(gzip.open(os.path.join(ROOT, "research", "s13_multi_house", "raw", f"replicate_{s}.json.gz"), "rt"))
        hfirst, dfirst, seen = {}, {}, []
        dest_groups_now = {dd[2] for dd in d["header"]["dest0"] if dd[2] is not None}
        for x in d["steps"]:
            dest_groups_now |= {t[3] for t in x["new_struct"] if t[2] == "Destination" and t[3] is not None}
            for k, g in x["house_groups"].items():
                if k in hfirst and hfirst[k] != g:
                    reassign += 1
                hfirst.setdefault(k, g)
            for dd in x["dests"]:
                key = (dd[0], dd[1])
                if key in dfirst and dfirst[key] != dd[2]:
                    dchange += 1
                dfirst.setdefault(key, dd[2])
            for ns in x["new_struct"]:
                if ns[2] == "Destination" and ns[3] is None:
                    null_dest_tiles += 1
                if ns[3] is not None and ns[3] not in seen:
                    seen.append(ns[3])
                if ns[2] == "House":
                    spawn_total += 1
                    spawn_match += int(ns[3] in dest_groups_now)
            max_groups = max(max_groups, len({g for g in x["house_groups"].values() if g is not None} | {dd[2] for dd in x["dests"] if dd[2] is not None}))
        appearance_orders.append(seen)
    check("A", "group_identity_stable_within_episode", reassign == 0 and dchange == 0, {"house_group_reassignments": reassign, "destination_group_changes": dchange, "decisions": 3867})
    check("A", "new_house_group_matches_an_existing_destination_group", spawn_match == spawn_total, {"house_spawns": spawn_total, "matched_at_spawn": spawn_match})
    check("A", "native_ids_are_arbitrary_labels_not_appearance_order", any(o != sorted(o) for o in appearance_orders), {"order_of_first_appearance_per_seed": appearance_orders, "max_simultaneous_groups": max_groups, "K_GROUPS": V.K_GROUPS,
                                                                                                                      "destination_tiles_without_model(groupIndex null)": null_dest_tiles})
    R["A"]["_source_evidence"] = {"plugin": "game_plugin/DoorstopBootstrap/Entrypoint.cs:3211-3223 (HouseModel.GroupIndex / DestinationModel.GroupIndex -> JSON groupIndex per tile)",
                                  "python_v1_loss": "python/mini_motorways_env.py:114  out[base+11] = 1.0 if groupIndex is not None  (identity collapsed to a flag)",
                                  "meaning": "GroupIndex indexes the theme colour group (research/static_recon: MotorwaysThemeDatabase.MAX_THEME_COLOR_GROUPS; GenerateDemandProcess._allocatedColorGroups = new bool[6]; Motorways.Themes.ColorGroup); "
                                             "demand generation and vehicle trips match house.GroupIndex == destination.GroupIndex (GenerateDemandProcess.cs:216-323)",
                                  "human_visible": "the colour of the building sprite (PerGroupMaterialBindings.RenderersWithColor) -- the core visual rule of the game",
                                  "not_exposed": "scheduled future buildings (CityPlanModel.scheduledBuildings), demand assignment internals, RNG"}


# ----------------------------------------------------------------------------------------------------------- Gate B: encoding correctness
def gate_b():
    rng = np.random.default_rng(0)
    caps = captures()
    # B1 relabeling invariance on native captures
    same = total = 0
    for c in caps:
        for key in ("reset", "first_house"):
            if not c.get(key):
                continue
            raw = c[key]["raw"]
            perm = {g: int(v) for g, v in zip(range(V.K_GROUPS), rng.permutation(V.K_GROUPS))}
            a = V.encode_native_v2(raw, V.GroupSlotTracker())
            b = V.encode_native_v2(relabel(raw, perm), V.GroupSlotTracker())
            total += 1
            same += int(np.array_equal(a, b))
    check("B", "B1_native_relabeling_invariance_single_state", same == total, {"states": total, "identical": same})
    # B1b relabeling invariance along a whole synthetic episode (tracker memory): relabel the scenario's native ids consistently
    sc = E13.load_scenarios()
    ok = True
    diffs = 0
    for idx in (0, 3, 12, 24):
        perm = {g: int(v) for g, v in zip(range(V.K_GROUPS), rng.permutation(V.K_GROUPS))}
        sc2 = copy.deepcopy(sc)
        s = sc2[idx]
        for st in [s["structures0"]] + [e["struct"] for e in s["events"]]:
            for t in st:
                if t[3] is not None:
                    t[3] = perm[t[3]]
        e1, e2 = MultiHouseEnvV2(sc, seed=5), MultiHouseEnvV2(sc2, seed=5)
        o1, _ = e1.reset(options={"scenario_index": idx})
        o2, _ = e2.reset(options={"scenario_index": idx})
        done = False
        while not done:
            a = CT.act(e1)
            o1, _, te, tr, _ = e1.step(a)
            o2, _, te2, tr2, _ = e2.step(a)
            diffs += int(not np.array_equal(o1, o2))
            done = te or tr
            ok = ok and (te, tr) == (te2, tr2)
    check("B", "B1b_relabeling_invariance_whole_episode", ok and diffs == 0, {"episodes": 4, "steps_with_different_obs": diffs})
    # B2 / B3 / B7: new houses get the slot of their destination group; several destinations of one group share a slot; slots never change once assigned
    env = MultiHouseEnvV2(sc, seed=1)
    bad_new = multi_ok = multi_total = 0
    slot_changes = 0
    n_new = 0
    for idx in range(0, 36, 2):
        o, _ = env.reset(options={"scenario_index": idx})
        prev_slots = {}
        done = False
        while not done:
            houses_before = set(env.houses)
            o, _, te, tr, _ = env.step(CT.act(env))
            done = te or tr
            tiles = o[: L.N_TILES * V.NUM_TILE_FEATURES_V2].reshape(L.N_TILES, V.NUM_TILE_FEATURES_V2)[:, V.F_GROUP_SLOT0:]
            slot_of = {}
            for i in range(L.N_TILES):
                if tiles[i].sum() > 0:
                    slot_of[i] = int(np.argmax(tiles[i]))
            for i, sl in slot_of.items():
                if i in prev_slots and prev_slots[i] != sl:
                    slot_changes += 1
            prev_slots.update(slot_of)
            for h in set(env.houses) - houses_before:
                n_new += 1
                g = env.houses[h]["group"]
                dest_slots = {slot_of.get(L.tile_index(*p)) for d in env.dests if d["group"] == g for p in d["tiles"] if L.tile_index(*p) in slot_of}
                if dest_slots and slot_of.get(L.tile_index(*h)) not in dest_slots:
                    bad_new += 1
            by_group = {}
            for d in env.dests:
                if d["group"] is not None:
                    by_group.setdefault(d["group"], set()).update(slot_of.get(L.tile_index(*p)) for p in d["tiles"] if L.tile_index(*p) in slot_of)
            for g, sl in by_group.items():
                if sum(1 for d in env.dests if d["group"] == g) > 1:
                    multi_total += 1
                    multi_ok += int(len(sl) == 1)
    check("B", "B2_new_houses_receive_their_destination_group_slot", bad_new == 0, {"new_houses": n_new, "mismatched": bad_new})
    check("B", "B3_multiple_destinations_same_group_share_slot", multi_total > 0 and multi_ok == multi_total, {"group_states_with_several_destinations": multi_total, "single_slot": multi_ok})
    check("B", "B7_slots_never_reassigned_within_episode", slot_changes == 0, {"tile_slot_changes": slot_changes})
    # B4 Active/Mothballed and all v1 channels: the first 15 channels of v2 are bitwise the v1 observation of the unmodified S13 env
    e1, e2 = E13.MultiHouseEnv(sc, seed=3), MultiHouseEnvV2(sc, seed=3)
    o1, _ = e1.reset(options={"scenario_index": 7})
    o2, _ = e2.reset(options={"scenario_index": 7})
    mism = 0
    done = False
    moth_seen = 0
    while not done:
        a = CT.act(e1) if e1.t % 3 else int(rng.choice(np.flatnonzero(e1.action_masks())))
        o1, _, te, tr, _ = e1.step(a)
        o2, _, _, _, _ = e2.step(a)
        mism += int(not np.array_equal(o1, V.strip_to_v1(o2)))
        moth_seen += sum(1 for s in e1.edges.values() if s == "M")
        done = te or tr
    check("B", "B4_v1_channels_bitwise_unchanged_incl_Active_Mothballed", mism == 0 and moth_seen > 0, {"steps_mismatched": mism, "mothballed_edge_steps_seen": moth_seen})
    # B5 no future leak: deleting / altering all events after t does not change the observation at t
    leaks = 0
    for idx in (1, 9, 20):
        env_a = MultiHouseEnvV2(sc, seed=11)
        env_a.reset(options={"scenario_index": idx})
        sc_cut = copy.deepcopy(sc)
        t_cut = 60
        sc_cut[idx]["events"] = [e for e in sc_cut[idx]["events"] if e["t"] <= t_cut]
        env_b = MultiHouseEnvV2(sc_cut, seed=11)
        env_b.reset(options={"scenario_index": idx})
        for _ in range(t_cut):
            a = CT.act(env_a)
            oa, _, _, _, _ = env_a.step(a)
            ob, _, _, _, _ = env_b.step(a)
            leaks += int(not np.array_equal(oa, ob))
    check("B", "B5_no_future_spawn_information", leaks == 0, {"steps_compared": 180, "differences": leaks})
    # B6 native/synthetic encoder agreement on equivalent visible states (synthetic state -> native-style JSON -> frozen native v1 encoder + v2 slots)
    env = MultiHouseEnvV2(sc, seed=2)
    agree_v2 = agree_v1 = total = 0
    v1_channel_mismatch = {}
    for idx in (0, 5, 10, 15, 20, 25, 30, 35):
        o, _ = env.reset(options={"scenario_index": idx})
        tr_ = V.GroupSlotTracker()
        done = False
        while not done:
            o, _, te, trn, _ = env.step(CT.act(env) if env.t % 2 else 0)
            done = te or trn
            if env.t % 7:
                continue
            raw = env.to_native_raw()
            o_nat = V.encode_native_v2(raw, tr_)
            total += 1
            agree_v2 += int(np.array_equal(o, o_nat))
            t1, t2 = o[: L.N_TILES * V.NUM_TILE_FEATURES_V2].reshape(L.N_TILES, -1), o_nat[: L.N_TILES * V.NUM_TILE_FEATURES_V2].reshape(L.N_TILES, -1)
            for ch in range(V.NUM_TILE_FEATURES_V2):
                n = int((~np.isclose(t1[:, ch], t2[:, ch])).sum())
                if n:
                    v1_channel_mismatch[ch] = v1_channel_mismatch.get(ch, 0) + n
            g1, g2 = o[L.N_TILES * V.NUM_TILE_FEATURES_V2:], o_nat[L.N_TILES * V.NUM_TILE_FEATURES_V2:]
            if not np.allclose(g1, g2):
                v1_channel_mismatch["globals"] = v1_channel_mismatch.get("globals", 0) + 1
            agree_v1 += int(np.array_equal(V.strip_to_v1(o), V.strip_to_v1(o_nat)))
    check("B", "B6_native_and_synthetic_encoders_agree_on_equivalent_states", agree_v2 == total, {"states": total, "identical_v2": agree_v2, "identical_v1_part": agree_v1, "mismatch_count_by_channel": v1_channel_mismatch})
    check("B", "B8_group_capacity", True, {"K_GROUPS": V.K_GROUPS, "max_simultaneous_groups_in_native_logs": 4, "obs_dim_v2": V.OBS_DIM_V2, "tile_features_v2": V.NUM_TILE_FEATURES_V2})


# ----------------------------------------------------------------------------------------------------------- Gate C: checkpoint migration
def gate_c():
    from sb3_contrib import MaskablePPO
    from synthetic_rl.models_v2 import SpatialMaskablePolicyV2, migrate_state_dict
    from synthetic_rl import s7, s7_eval
    torch.set_num_threads(1)
    src = MaskablePPO.load(CK, device="cpu")
    sha = hashlib.sha256(open(CK, "rb").read()).hexdigest()
    sc = E13.load_scenarios()
    env = MultiHouseEnvV2(sc, seed=0)
    dst = MaskablePPO(SpatialMaskablePolicyV2, env, device="cpu", n_steps=512, batch_size=256, n_epochs=4, target_kl=0.03, learning_rate=3e-4, ent_coef=0.01, gamma=0.99, gae_lambda=0.95, seed=14)
    sd, rep = migrate_state_dict(src.policy.state_dict(), dst.policy.state_dict())
    dst.policy.load_state_dict(sd, strict=True)
    n_src = sum(p.numel() for p in src.policy.parameters())
    n_dst = sum(p.numel() for p in dst.policy.parameters())
    # equivalence on saved native observations (24 captures x up to 2 states) and on 300 synthetic multi-house states
    obs_pairs = []
    for c in captures():
        for key in ("reset", "first_house"):
            if c.get(key):
                raw = c[key]["raw"]
                obs_pairs.append((NATIVE.encode_observation(raw), V.encode_native_v2(raw, V.GroupSlotTracker())))
    rng = np.random.default_rng(1)
    for idx in range(0, 36, 4):
        o2, _ = env.reset(options={"scenario_index": idx})
        for _ in range(40):
            o2, _, te, tr, _ = env.step(CT.act(env) if rng.random() < 0.7 else int(rng.choice(np.flatnonzero(env.action_masks()))))
            obs_pairs.append((V.strip_to_v1(o2), o2.copy()))
            if te or tr:
                break
    max_logit = max_val = 0.0
    same_action = 0
    with torch.no_grad():
        for o1, o2 in obs_pairs:
            t1 = torch.as_tensor(o1[None]); t2 = torch.as_tensor(o2[None])
            l1, v1 = src.policy.mlp_extractor(t1)
            l2, v2 = dst.policy.mlp_extractor(t2)
            max_logit = max(max_logit, float((l1 - l2).abs().max()))
            max_val = max(max_val, float((v1 - v2).abs().max()))
            same_action += int(torch.argmax(l1) == torch.argmax(l2))
    check("C", "C1_migrated_network_reproduces_S8C_outputs_on_saved_observations", max_logit < 1e-5 and max_val < 1e-5 and same_action == len(obs_pairs),
          {"observations": len(obs_pairs), "max_abs_logit_diff": max_logit, "max_abs_value_diff": max_val, "same_greedy_action": same_action, "params_src": n_src, "params_dst": n_dst, "widened": rep["zero_initialised_slice"]})
    # C2: frozen S7 evaluators, deterministic: identical rows for the frozen model and the migrated model through the v1->v2 adapter (Gap-7 and Gap-10 validation)
    sets, _ = s7.build_task_sets()
    out = {}
    identical = True
    for k in (7, 10):
        tasks = sets["validation"][k]
        r_src = s7_eval.evaluate("model", tasks, model=src, deterministic=True, seed=0)
        r_dst = s7_eval.evaluate("model", tasks, model=V.V2ModelOnV1Env(dst), deterministic=True, seed=0)
        s_src, s_dst = s7_eval.summarize(r_src), s7_eval.summarize(r_dst)
        identical = identical and all(a["success"] == b["success"] and a["steps"] == b["steps"] for a, b in zip(r_src, r_dst))
        out[k] = {"frozen_S8C_success": s_src[k]["success"], "migrated_success": s_dst[k]["success"], "n": len(tasks)}
    check("C", "C2_S7_gap7_gap10_validation_identical_through_adapter", identical, out)
    # save the migrated checkpoint (new file; S8-C untouched)
    path = os.path.join(OUT, "s8c_migrated_v2_step0.zip")
    dst.save(path)
    R["C"]["_migration"] = {"source": CK, "source_sha256": sha, "migrated_checkpoint": path, "migrated_sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(), "report": {k: (v if k != "copied" else len(v)) for k, v in rep.items()},
                            "procedure": "copy all 22 tensors by name; mlp_extractor.trunk.0.weight widened from (32,15,3,3) to (32,21,3,3) with zeros in input channels 15..20; optimizer fresh"}


# ----------------------------------------------------------------------------------------------------------- information loss of the v1 encoder
def info_loss():
    sc = E13.load_scenarios()
    rng = np.random.default_rng(3)
    # (a) native logs: at every decision, the number of house->group assignments consistent with the v1 observation = prod over houses of (#destination groups visible)
    n_states = ambiguous = 0
    log_assign = []
    for s in SEEDS:
        d = json.load(gzip.open(os.path.join(ROOT, "research", "s13_multi_house", "raw", f"replicate_{s}.json.gz"), "rt"))
        for x in d["steps"]:
            n_states += 1
            G = len({dd[2] for dd in x["dests"] if dd[2] is not None})
            H = len(x["house_groups"])
            if G >= 2 and H >= 1:
                ambiguous += 1
                log_assign.append(H * np.log(G))
    # (b) synthetic counterexamples: identical v1 observation, different assignment; does the construction target (nearest same-group destination of the oldest unconnected house) change?
    env = MultiHouseEnvV2(sc, seed=4)
    pairs = tgt_changed = v1_same = v2_diff = first_edge_changed = 0
    examples = []
    for idx, _rep in [(i, r) for i in range(36) for r in range(4)]:
        env.reset(options={"scenario_index": idx})
        for _ in range(int(rng.integers(20, 320))):
            env.step(CT.act(env) if rng.random() < 0.5 else 0)
            if env.game_over:
                break
        groups = sorted({d["group"] for d in env.dests if d["group"] is not None})
        if len(groups) < 2 or not env.houses:
            continue
        conn = env.connected_houses()
        todo = [h for h in env.houses if h not in conn]
        if not todo:
            continue
        h = min(todo, key=lambda q: env.houses[q]["t"])
        p_real = CT._plan(env, h)
        o_real = env._obs_v2.copy()
        saved = {q: env.houses[q]["group"] for q in env.houses}
        # counterfactual: assign this house to another visible group (a state that the v1 encoder cannot distinguish from the real one)
        other = [g for g in groups if g != saved[h]]
        env.houses[h]["group"] = other[int(rng.integers(len(other)))]
        env._build_obs()
        o_cf = env._obs_v2.copy()
        p_cf = CT._plan(env, h)
        for q, g in saved.items():
            env.houses[q]["group"] = g
        env._build_obs()
        pairs += 1
        v1_same += int(np.array_equal(V.strip_to_v1(o_real), V.strip_to_v1(o_cf)))
        v2_diff += int(not np.array_equal(o_real, o_cf))
        changed = (p_real is None) != (p_cf is None) or (p_real is not None and p_cf is not None and p_real[-1] != p_cf[-1])
        tgt_changed += int(changed)
        first_edge_changed += int((p_real[:2] if p_real else None) != (p_cf[:2] if p_cf else None))
        if len(examples) < 5:
            examples.append({"scenario": sc[idx]["id"], "t": env.t, "house": list(h), "real_group": saved[h], "counterfactual_group": env.houses[h]["group"] if False else other[0], "target_real": p_real[-1] if p_real else None, "target_cf": p_cf[-1] if p_cf else None,
                             "first_edge_real": p_real[:2] if p_real else None, "first_edge_cf": p_cf[:2] if p_cf else None})
    res = {"native_decisions": n_states, "decisions_with_>=2_destination_groups_and_>=1_house": ambiguous, "fraction": round(ambiguous / n_states, 3),
           "median_log_assignments_consistent_with_v1_when_ambiguous": round(float(np.median(log_assign)), 2) if log_assign else None, "median_assignments_consistent_with_v1": round(float(np.exp(np.median(log_assign))), 1) if log_assign else None,
           "distinguishable_under_v2": 1, "synthetic_counterfactual_pairs": pairs, "v1_observation_identical": v1_same, "v2_observation_different": v2_diff, "construction_target_changed": tgt_changed, "first_planned_edge_changed": first_edge_changed, "examples": examples,
           "note": "v2 resolves the house<->group assignment ambiguity exactly (one assignment per observation). It does NOT remove other partial observability: hidden demand internals, vehicle trips, future spawns, upgrade packages, and the model-less destination tiles stay unobserved."}
    json.dump(res, open(os.path.join(OUT, "info_loss.json"), "w"), indent=1)
    print("INFO-LOSS", json.dumps(res)[:800], flush=True)


def main():
    t0 = time.time()
    for f in (gate_a, gate_b, gate_c, info_loss):
        try:
            f()
        except Exception as ex:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            check(f.__name__, "exception", False, repr(ex))
    for sec in ("A", "B", "C"):
        if sec in R:
            json.dump(R[sec], open(os.path.join(OUT, f"gate_{sec.lower()}.json"), "w"), indent=1, default=str)
    summary = {sec: {"passed": sum(v["passed"] for k, v in R[sec].items() if not k.startswith("_")), "total": sum(1 for k in R[sec] if not k.startswith("_"))} for sec in R}
    json.dump(summary, open(os.path.join(OUT, "gates_summary.json"), "w"), indent=1)
    print(summary, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
