"""S12 evaluation-only CONCRETE-CONSERVATION guard (pure functions on the raw native observation; never given to the policy). Reuses the corrected S11 transition model
(functional = Active / VisiblyActive; displaced house edges may be left Mothballed) but is a SEPARATE guard: it does not contain the connection-preservation veto.

Vetoes only a CYCLE-CLOSING EXTENSION: a native-legal ADD of a plain (non-house) edge whose two endpoints already lie in the same connected component of the current functional
road graph, whose endpoints are not House / Destination / Carpark / water tiles, and whose edge is not already present in any non-None state. See
research/s12_concrete_conservation/FEASIBILITY_PREREGISTRATION.md for why this class is permanently redundant for connectivity and what is deliberately NOT vetoed.
Everything else is 'allow' (or 'unknown' when the transition model cannot predict the action)."""
from typing import Any, Dict, List, Set, Tuple

import numpy as np

import s11_guard as G

Edge = G.Edge


def components(edges: Set[Edge]) -> Dict[Tuple[int, int], Tuple[int, int]]:
    parent: Dict[Tuple[int, int], Tuple[int, int]] = {}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for e in edges:
        a, b = G.ends(e)
        parent.setdefault(a, a)
        parent.setdefault(b, b)
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    return {t: find(t) for t in parent}


def classify_add(ctx: G.Ctx, func: Set[Edge], present: Set[Edge], comp, e: Edge) -> Dict[str, Any]:
    """class of one ADD: cycle_closing (vetoed) | extension (allow) | joins_components | new_segment | house_replacement / house_other | unknown_* """
    pred = G.predict_add(ctx, func, e)
    if not pred["known"]:
        return {"verdict": "unknown", "cls": "unknown_" + pred["reason"]}
    if pred["reason"] != "plain_extension":
        return {"verdict": "allow", "cls": "house_" + pred["reason"]}
    if not pred["added"]:
        return {"verdict": "allow", "cls": "noop_edge_already_active_no_resource_effect"}      # consumes nothing (observed dC = 0): nothing to conserve
    a, b = G.ends(e)
    if a in ctx.water or b in ctx.water:
        return {"verdict": "unknown", "cls": "unknown_water_endpoint"}
    if ctx.content.get(a) not in ("None",) or ctx.content.get(b) not in ("None",):
        return {"verdict": "unknown", "cls": "unknown_content_endpoint"}        # e.g. Tree tiles
    if e in present and e not in func:
        return {"verdict": "unknown", "cls": "unknown_reactivation_of_nonfunctional_edge"}
    ca, cb = comp.get(a), comp.get(b)
    if ca is not None and ca == cb:
        return {"verdict": "veto", "cls": "cycle_closing"}
    if ca is None and cb is None:
        return {"verdict": "allow", "cls": "new_segment"}
    if ca is not None and cb is not None:
        return {"verdict": "allow", "cls": "joins_components"}
    return {"verdict": "allow", "cls": "extension"}


def conservation_permitted(raw: Dict[str, Any], mask: np.ndarray, decode_action, upgrade_base: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    ctx = G.Ctx(raw)
    func = G.edges_from_raw(raw)
    present = G.edges_present(raw)
    comp = components(func)
    permitted = mask.copy()
    vetoed: List[Tuple[int, str]] = []
    classes: Dict[str, List[int]] = {}
    for a in np.flatnonzero(mask):
        if not (0 < a < upgrade_base):
            continue
        k = decode_action(int(a))
        if k[0] != "ADD":
            continue
        r = classify_add(ctx, func, present, comp, (k[1], k[2], k[3]))
        classes.setdefault(r["cls"], []).append(int(a))
        if r["verdict"] == "veto":
            permitted[a] = False
            vetoed.append((int(a), r["cls"]))
    return permitted, {"vetoed": vetoed, "class_actions": classes}
