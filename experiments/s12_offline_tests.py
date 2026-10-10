"""S12 offline regression tests of the concrete-conservation guard (no game). python experiments/s12_offline_tests.py"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import numpy as np
import s11_guard as G
import s12_guard as C
from synthetic_rl import layout as L

OUT = os.path.join(ROOT, "research", "s12_concrete_conservation")
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
    print(("PASS " if RES[-1][1] else "FAIL ") + RES[-1][0] + ": " + RES[-1][2][:300], flush=True)
    return fn


def mk_raw(houses, edges, water=(), moth=(), dest=((8, 3), (9, 3), (8, 4), (9, 4)), carpark=((7, 3), (7, 4)), group=0, W=12, H=8):
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


def cls(raw, e):
    ctx = G.Ctx(raw)
    func = G.edges_from_raw(raw)
    present = G.edges_present(raw)
    return C.classify_add(ctx, func, present, C.components(func), e)


STUB = G.canon((7, 3), (6, 3))
PART = {G.canon((2, 3), (3, 3)), G.canon((3, 3), (4, 3)), G.canon((4, 3), (5, 3)), STUB}      # house -> ... -> (5,3); carpark stub is a separate component


@test
def t01_necessary_first_edge_of_a_long_route_is_allowed():
    r = cls(mk_raw({(2, 3): 0}, {G.canon((2, 3), (3, 3)), STUB}), G.canon((3, 3), (4, 3)))
    assert r["verdict"] == "allow" and r["cls"] == "extension", r
    r2 = cls(mk_raw({(2, 3): 0}, {G.canon((2, 3), (2, 2)), STUB}), G.canon((4, 5), (4, 6)))
    assert r2["verdict"] == "allow" and r2["cls"] == "new_segment", r2
    return "extension of the first segment and a fresh segment are allowed"


@test
def t02_necessary_intermediate_edge_is_allowed():
    r = cls(mk_raw({(2, 3): 0}, PART), G.canon((5, 3), (5, 4)))
    r2 = cls(mk_raw({(2, 3): 0}, PART), G.canon((4, 3), (4, 4)))
    assert r["verdict"] == "allow" and r2["verdict"] == "allow"
    return "tip and mid-route extensions allowed (" + r["cls"] + ", " + r2["cls"] + ")"


@test
def t03_final_connecting_edge_is_allowed():
    r = cls(mk_raw({(2, 3): 0}, PART), G.canon((5, 3), (6, 3)))
    assert r["verdict"] == "allow" and r["cls"] == "joins_components", r
    raw2 = mk_raw({(2, 3): 0}, PART | {G.canon((5, 3), (6, 3))})
    assert G.connected_house_set(G.Ctx(raw2), G.edges_from_raw(raw2)) == {(2, 3)}
    return "joins the house component to the carpark component and connects the house"


@test
def t04_harmless_preparatory_edge_is_not_vetoed():
    r = cls(mk_raw({(2, 3): 0}, PART), G.canon((4, 3), (4, 2)))
    r2 = cls(mk_raw({(2, 3): 0}, PART), G.canon((1, 6), (2, 6)))
    assert r["verdict"] == "allow" and r2["verdict"] == "allow"
    return "spur and isolated preparatory segment: " + r["cls"] + " / " + r2["cls"]


@test
def t05_noop_add_is_vetoed_only_with_a_resource_effect():
    r = cls(mk_raw({(2, 3): 0}, PART), G.canon((3, 3), (4, 3)))              # edge already active: consumes nothing (observed dC = 0)
    assert r["verdict"] == "allow" and r["cls"].startswith("noop"), r
    rc = cls(mk_raw({(2, 3): 0}, PART | {G.canon((3, 3), (3, 4)), G.canon((3, 4), (4, 4))}), G.canon((4, 4), (4, 3)))   # closes a cycle: consumes 1, adds no reach
    assert rc["verdict"] == "veto" and rc["cls"] == "cycle_closing", rc
    return "existing active edge -> allow (no resource effect); new cycle-closing edge -> veto"


@test
def t06_house_edge_replacement_with_mothballed_effects_is_classified_not_vetoed():
    base = {G.canon((2, 3), (3, 3)), G.canon((3, 3), (4, 3)), STUB}
    raw = mk_raw({(2, 3): 0}, base | {G.canon((2, 3), (2, 4))}, moth={G.canon((2, 3), (2, 4))})
    r = cls(raw, G.canon((2, 3), (2, 4)))
    r2 = cls(mk_raw({(2, 3): 0}, base), G.canon((2, 3), (2, 2)))
    assert r["verdict"] == "allow" and r["cls"].startswith("house_"), r
    assert r2["verdict"] == "allow" and r2["cls"] == "house_house_edge_replacement", r2
    return "reactivation / replacement: " + r["cls"] + ", " + r2["cls"]


@test
def t07_history_and_inventory_do_not_create_redundancy():
    assert cls(mk_raw({(2, 3): 0}, PART), G.canon((5, 3), (6, 3)))["verdict"] == "allow"
    cyc = PART | {G.canon((3, 3), (3, 4)), G.canon((3, 4), (4, 4))}
    e = G.canon((4, 4), (4, 3))
    r2 = cls(mk_raw({(2, 3): 0}, cyc | {e}, moth={e}), e)               # non-functional present edge: re-activation, never judged redundant
    assert r2["verdict"] == "unknown" and "reactivation" in r2["cls"], r2
    return "verdicts use only the current functional graph; re-activation of a non-functional edge -> unknown"


@test
def t08_stale_connectivity_is_never_reused():
    e = G.canon((4, 3), (4, 4))
    assert cls(mk_raw({(2, 3): 0}, PART), e)["verdict"] == "allow"
    s2 = mk_raw({(2, 3): 0, (4, 6): 0}, PART | {G.canon((4, 4), (3, 4)), G.canon((3, 4), (3, 3))})
    r = cls(s2, e)
    assert r["verdict"] == "veto" and r["cls"] == "cycle_closing", r
    return "same candidate edge: allow before the network changed, veto after (components rebuilt from the current observation on every call)"


@test
def t09_zero_concrete_state_never_adds_actions():
    raw = mk_raw({(2, 3): 0}, PART)
    rng = np.random.default_rng(0)
    for _ in range(50):
        mask = np.zeros(L.N_ACTIONS, bool)
        mask[0] = True
        for e in [G.canon((2, 3), (2, 2)), G.canon((2, 3), (2, 4))]:
            if rng.random() < 0.7:
                mask[L.action_index("ADD", *e)] = True
        perm, rep = C.conservation_permitted(raw, mask, L.decode_action, L.UPGRADE_BASE)
        assert np.array_equal(perm, mask) and not rep["vetoed"]
    raw3 = mk_raw({(2, 3): 0}, PART | {G.canon((3, 3), (3, 4)), G.canon((3, 4), (4, 4)), G.canon((4, 4), (4, 3))})
    for _ in range(200):
        mask = np.zeros(L.N_ACTIONS, bool)
        mask[0] = True
        mask[L.UPGRADE_BASE] = True
        for e in [G.canon((3, 3), (4, 3)), G.canon((4, 3), (4, 2)), G.canon((6, 5), (6, 6))]:
            mask[L.action_index("ADD", *e)] = bool(rng.random() < .5)
        perm, rep = C.conservation_permitted(raw3, mask, L.decode_action, L.UPGRADE_BASE)
        assert (perm & ~mask).sum() == 0 and perm[0] and perm[L.UPGRADE_BASE]
    return "permitted is always a subset of the native mask; WAIT / upgrade untouched; at zero concrete (house-touching ADDs only) nothing changes"


@test
def t10_unknown_transitions_are_allowed():
    raw = mk_raw({(2, 3): 0, (2, 4): 0}, PART, water={(5, 5)})
    for e, want in ((G.canon((2, 3), (2, 4)), "unknown_both_endpoints_are_houses"), (G.canon((6, 3), (7, 3)), "unknown_destination_or_carpark_endpoint"), (G.canon((5, 4), (5, 5)), "unknown_water_endpoint")):
        r = cls(raw, e)
        assert r["verdict"] == "unknown" and r["cls"] == want, (e, r)
    return "house-house, carpark and water edges -> unknown (not vetoed)"


@test
def t11_redundancy_means_identical_connectivity_with_or_without_the_cycle_edge():
    base = PART | {G.canon((5, 3), (6, 3)), G.canon((3, 3), (3, 4)), G.canon((3, 4), (4, 4))}
    cyc_edge = G.canon((4, 4), (4, 3))
    raw_a, raw_b = mk_raw({(2, 3): 0}, base), mk_raw({(2, 3): 0}, base | {cyc_edge})
    assert cls(raw_a, cyc_edge)["verdict"] == "veto"
    assert G.connected_house_set(G.Ctx(raw_a), G.edges_from_raw(raw_a)) == G.connected_house_set(G.Ctx(raw_b), G.edges_from_raw(raw_b)) == {(2, 3)}
    return "connected set identical with and without the vetoed edge"


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    json.dump([{"test": n, "passed": p, "detail": d} for n, p, d in RES], open(os.path.join(OUT, "offline_test_results.json"), "w"), indent=1)
    print(f"\n{sum(r[1] for r in RES)}/{len(RES)} passed")
    sys.exit(0 if all(r[1] for r in RES) else 1)
