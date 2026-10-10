"""S11 offline validation of the connection-preservation guard (no game). python experiments/s11_offline_tests.py"""
import copy
import gzip
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import numpy as np
import s11_guard as G
from synthetic_rl import layout as L

OUT = os.path.join(ROOT, "research", "s11_connection_guard")
RES = []


def test(fn):
    try:
        info = fn()
        RES.append((fn.__name__, True, info or ""))
    except AssertionError as e:
        RES.append((fn.__name__, False, "ASSERTION: " + str(e)))
    except Exception as e:
        import traceback
        RES.append((fn.__name__, False, f"{type(e).__name__}: {e} {traceback.format_exc()[-300:]}"))
    print(("PASS " if RES[-1][1] else "FAIL ") + RES[-1][0] + ": " + RES[-1][2][:330], flush=True)
    return fn


def mk_raw(houses, edges, water=(), moth=(), dest=((8, 3), (9, 3), (8, 4), (9, 4)), carpark=((7, 3), (7, 4)), group=0, W=12, H=8, one_sided=False):
    """houses: {(x,y): group}; edges: set of canonical edges; one tile grid with the destination footprint, carparks, houses"""
    roads = {}
    for e in edges:
        a, b = G.ends(e)
        for t, o in ((a, b), (b, a)):
            d = [k for k, (dx, dy) in G.DIRS.items() if (t[0] + dx, t[1] + dy) == o][0]
            roads.setdefault(t, {})[d] = "Mothballed" if e in moth else "Active"
    tiles = []
    for y in range(H):
        for x in range(W):
            r = {d: "None" for d in "NESW"}
            r.update(roads.get((x, y), {}))
            c, g = "None", None
            if (x, y) in dest:
                c, g = "Destination", group
            elif (x, y) in carpark:
                c = "Carpark"
            elif (x, y) in houses:
                c, g = "House", houses[(x, y)]
            tiles.append({"coord": [x, y], "buildable": True, "overWater": (x, y) in water, "underMountain": False, "overRail": False, "road": r, "contentType": c, "groupIndex": g})
    return {"tiles": tiles, "globals": {}}


STUB = G.canon((7, 3), (6, 3))
BASE = {G.canon((2, 3), (3, 3)), G.canon((3, 3), (4, 3)), G.canon((4, 3), (5, 3)), G.canon((5, 3), (6, 3)), STUB}   # house (2,3) -> ... -> carpark: connected


def verdict(raw, e):
    ctx = G.Ctx(raw)
    edges = G.edges_from_raw(raw)
    return G.check_candidate(ctx, edges, G.connected_house_set(ctx, edges), e)


@test
def test_1_replacement_that_destroys_a_connection_is_vetoed():
    raw = mk_raw({(2, 3): 0}, BASE)
    ctx = G.Ctx(raw)
    assert G.connected_house_set(ctx, G.edges_from_raw(raw)) == {(2, 3)}
    v = verdict(raw, G.canon((2, 3), (2, 2)))
    assert v["verdict"] == "veto" and v["lost"] == [(2, 3)], v
    return str(v["reason"])


@test
def test_2_replacement_that_preserves_connectivity_is_allowed():
    edges = (BASE - {G.canon((2, 3), (3, 3))}) | {G.canon((2, 3), (2, 4)), G.canon((2, 4), (3, 4)), G.canon((3, 4), (3, 3))}   # house reaches the network through a detour
    raw = mk_raw({(2, 3): 0}, edges)
    assert G.connected_house_set(G.Ctx(raw), G.edges_from_raw(raw)) == {(2, 3)}
    v = verdict(raw, G.canon((2, 3), (3, 3)))          # swap to the direct edge: still connected
    assert v["verdict"] == "allow" and "preserves" in v["reason"], v
    return v["reason"]


@test
def test_3_extension_of_an_existing_route_is_allowed():
    raw = mk_raw({(2, 3): 0}, BASE)
    v = verdict(raw, G.canon((4, 3), (4, 2)))
    assert v["verdict"] == "allow" and v["reason"] == "plain_extension", v
    return v["reason"]


@test
def test_4_adding_the_edge_that_connects_a_new_house_is_allowed_and_connects_it():
    edges = BASE | {G.canon((2, 6), (3, 6)), G.canon((3, 6), (3, 5))}
    raw = mk_raw({(2, 3): 0, (2, 6): 0}, edges)
    ctx = G.Ctx(raw)
    e0 = G.edges_from_raw(raw)
    assert G.connected_house_set(ctx, e0) == {(2, 3)}
    e_new = G.canon((3, 5), (3, 4))
    v = verdict(raw, e_new)
    # (3,4) must also reach the network: add the last link and test the final ADD
    raw2 = mk_raw({(2, 3): 0, (2, 6): 0}, edges | {G.canon((3, 4), (3, 3))})
    ctx2 = G.Ctx(raw2)
    e2 = G.edges_from_raw(raw2)
    assert v["verdict"] == "allow"
    assert G.connected_house_set(ctx2, e2 | {e_new}) == {(2, 3), (2, 6)}
    return "ADD that links the second house is allowed and the post-ADD graph connects both houses"


@test
def test_5_an_add_that_would_disconnect_several_houses_protects_all():
    # a transit house (two edges) joins two connected houses to the network; model predicts removal of the shared edge -> both lost; guard logic must report both
    shared = G.canon((3, 3), (4, 3))
    raw = mk_raw({(2, 3): 0, (3, 4): 0}, BASE | {G.canon((3, 4), (3, 3))})
    ctx = G.Ctx(raw)
    edges = G.edges_from_raw(raw)
    base = G.connected_house_set(ctx, edges)
    assert base == {(2, 3), (3, 4)}
    orig = G.predict_add
    G.predict_add = lambda c, ed, e: {"known": True, "reason": "forced_for_test", "removed": {shared}, "added": set()}
    try:
        r = G.check_candidate(ctx, edges, base, G.canon((6, 5), (6, 6)))
    finally:
        G.predict_add = orig
    assert r["verdict"] == "veto" and r["lost"] == [(2, 3), (3, 4)], r
    return "veto lists both houses " + str(r["lost"])


@test
def test_6_water_terrain_follows_native_legality_and_adds_no_extra_rule():
    raw = mk_raw({(2, 3): 0}, BASE, water={(4, 2)})
    v = verdict(raw, G.canon((4, 3), (4, 2)))           # edge onto a water tile, not touching a house
    assert v["verdict"] == "allow" and v["reason"] == "plain_extension"
    return "water-endpoint ADD not touching a house is a plain extension; legality is left to the native mask"


@test
def test_7_unknown_counterfactuals_are_not_vetoed():
    raw = mk_raw({(2, 3): 0}, BASE | {G.canon((2, 3), (2, 4))})        # house with two edges
    v = verdict(raw, G.canon((2, 3), (2, 2)))
    assert v["verdict"] == "unknown" and v["reason"] == "house_with_multiple_active_edges", v
    raw2 = mk_raw({(2, 3): 0, (2, 4): 0}, BASE)                         # edge between two houses
    v2 = verdict(raw2, G.canon((2, 3), (2, 4)))
    assert v2["verdict"] == "unknown" and v2["reason"] == "both_endpoints_are_houses", v2
    v3 = verdict(mk_raw({(2, 3): 0}, BASE), G.canon((6, 3), (7, 3)))     # carpark endpoint
    assert v3["verdict"] == "unknown", v3
    return "multi-edge house, house-house edge and carpark endpoint -> 'unknown' (not vetoed)"


@test
def test_8_no_connected_houses_means_no_restriction():
    raw = mk_raw({(2, 3): 0}, {G.canon((2, 3), (2, 2))})
    mask = np.zeros(L.N_ACTIONS, bool)
    mask[0] = True
    for e in [G.canon((2, 3), (3, 3)), G.canon((2, 3), (2, 4)), G.canon((4, 4), (4, 5))]:
        mask[L.action_index("ADD", *e)] = True
    perm, rep = G.guard_permitted(raw, mask, L.decode_action, L.UPGRADE_BASE)
    assert np.array_equal(perm, mask) and rep["n_connected"] == 0 and not rep["vetoed"]
    return "permitted == native mask"


@test
def test_9_wait_remove_and_upgrade_are_never_vetoed():
    raw = mk_raw({(2, 3): 0}, BASE)
    mask = np.zeros(L.N_ACTIONS, bool)
    mask[0] = True
    mask[L.action_index("REMOVE", 3, 3, "E")] = True
    mask[L.UPGRADE_BASE] = True
    bad = L.action_index("ADD", *G.canon((2, 3), (2, 2)))
    mask[bad] = True
    perm, rep = G.guard_permitted(raw, mask, L.decode_action, L.UPGRADE_BASE)
    assert not perm[bad] and perm[0] and perm[L.UPGRADE_BASE] and perm[L.action_index("REMOVE", 3, 3, "E")] and perm.sum() == mask.sum() - 1
    return "only the destroying ADD is removed"


@test
def test_10_real_saved_native_states_have_no_connections_so_no_restriction():
    import mini_motorways_env as M
    n = vetoes = 0
    for line in gzip.open(os.path.join(ROOT, "research", "s7_long_route", "native_normal_start_captures.jsonl.gz"), "rt"):
        r = json.loads(line)["first_house"]
        raw = {"tiles": r["raw"]["tiles"], "globals": r["raw"]["globals"]}
        mask = np.zeros(L.N_ACTIONS, bool)
        mask[r["legal"]] = True
        before = (copy.deepcopy(raw), mask.copy())
        perm, rep = G.guard_permitted(raw, mask, L.decode_action, L.UPGRADE_BASE)
        assert np.array_equal(perm, mask) and raw == before[0] and np.array_equal(mask, before[1])
        n += 1
        vetoes += len(rep["vetoed"])
    return f"{n} real native states: permitted == native mask, vetoes {vetoes}; raw observation and mask untouched"


@test
def test_11_stochastic_conditional_sampling_matches_the_renormalised_policy():
    import torch
    from sb3_contrib import MaskablePPO
    import mini_motorways_env as M
    m = MaskablePPO.load(os.path.join(ROOT, "research", "s8_long_route_training", "runs", "arm_C_s11", "checkpoints", "step_0099840.zip"), device="cpu")
    line = next(iter(gzip.open(os.path.join(ROOT, "research", "s7_long_route", "native_normal_start_captures.jsonl.gz"), "rt")))
    r = json.loads(line)["first_house"]
    obs = M.encode_observation({"tiles": r["raw"]["tiles"], "globals": r["raw"]["globals"]})
    mask = np.zeros(L.N_ACTIONS, bool)
    mask[r["legal"]] = True
    with torch.no_grad():
        p_native = m.policy.get_distribution(torch.as_tensor(obs[None]), action_masks=mask[None]).distribution.probs[0].numpy()
    top = np.argsort(-p_native)[:12]
    veto = top[:5]                                        # veto the 5 most likely actions
    perm = mask.copy()
    perm[veto] = False
    with torch.no_grad():
        p_perm = m.policy.get_distribution(torch.as_tensor(obs[None]), action_masks=perm[None]).distribution.probs[0].numpy()
    cond = p_native * perm
    cond = cond / cond.sum()
    assert np.abs(p_perm - cond).max() < 1e-6
    torch.manual_seed(5)
    acts = np.array([int(m.predict(obs, action_masks=perm, deterministic=False)[0]) for _ in range(3000)])
    emp = np.bincount(acts, minlength=L.N_ACTIONS) / len(acts)
    tv = 0.5 * np.abs(emp - cond).sum()
    assert not np.isin(acts, veto).any() and tv < 0.05, tv
    # pairing property: with an empty veto set the guarded sampler consumes the RNG exactly like the control sampler
    torch.manual_seed(9)
    a1 = [int(m.predict(obs, action_masks=mask, deterministic=False)[0]) for _ in range(50)]
    torch.manual_seed(9)
    a2 = [int(m.predict(obs, action_masks=mask.copy(), deterministic=False)[0]) for _ in range(50)]
    assert a1 == a2
    return f"mass removed by vetoing the top-5 actions {1 - float((p_native * perm).sum()):.3f}; sampled distribution vs analytic conditional: total variation {tv:.4f} (3000 draws); no vetoed action ever drawn; empty-veto sampling identical to control sampling"


@test
def test_12_mothballed_edges_are_not_functional_connections():
    # house (2,3): Active edge to an isolated tile, plus a MOTHBALLED edge toward the network -> not connected (legacy reading would call it connected)
    iso = G.canon((2, 3), (2, 2))
    link = G.canon((2, 3), (3, 3))
    raw = mk_raw({(2, 3): 0}, (BASE - {link}) | {iso, link}, moth={link})
    ctx = G.Ctx(raw)
    assert G.connected_house_set(ctx, G.edges_from_raw(raw)) == set()
    assert G.connected_house_set(ctx, G.edges_present(raw)) == {(2, 3)}
    # re-adding the mothballed link reactivates it: allowed, and the house becomes connected
    v = G.check_candidate(ctx, G.edges_from_raw(raw), set(), link)
    assert v["verdict"] == "allow"
    # connected house whose Active edge is replaced by a route that is only mothballed -> the Active-edge model still vetoes the dangerous replacement
    raw2 = mk_raw({(2, 3): 0}, BASE | {G.canon((2, 3), (2, 2))}, moth={G.canon((2, 3), (2, 2))})
    ctx2 = G.Ctx(raw2)
    e2 = G.edges_from_raw(raw2)
    assert G.connected_house_set(ctx2, e2) == {(2, 3)}
    v2 = G.check_candidate(ctx2, e2, {(2, 3)}, G.canon((2, 3), (2, 4)))
    assert v2["verdict"] == "veto"
    return "mothballed edges excluded from functional connectivity; reactivation allowed; replacement toward a dead end vetoed"


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    json.dump([{"test": n, "passed": p, "detail": d} for n, p, d in RES], open(os.path.join(OUT, "offline_test_results.json"), "w"), indent=1)
    print(f"\n{sum(r[1] for r in RES)}/{len(RES)} passed")
    sys.exit(0 if all(r[1] for r in RES) else 1)
