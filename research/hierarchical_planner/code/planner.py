"""S18-H deterministic heuristic scorer. Every term is explicit and logged in Candidate.score_terms.

score(CONNECT) = W_CONN * connectivity_delta
               + W_DEMAND * (pins at the target group's destinations) + W_OVER * [that group is overcrowded]
               - W_COST * concrete_cost * scarcity(t, concrete, cost, other_needs)
               + W_REUSE * reused_edges / max(1, route_len)
               - W_LEN * route_len                            (travel-cost PROXY = path length; no traffic model)
               + W_WAIT * min(60, decisions since the house spawned) / 60    (new houses are served within 60 decisions when possible)
               - W_UNAFFORDABLE * [cost > concrete] * (cost - concrete)
               - INF if would_destroy_functional_connection and the candidate is not an explicit reroute
score(REMOVE)  = W_REFUND * [some CONNECT candidate is unaffordable by <= refundable amount] - W_REMOVE_BASE
score(WAIT)    = 0
score(MOTORWAY)= -INF (unsupported)
scarcity: concrete is time-driven: the next top-up (141/281/421/561) is known from the clock. A spend that leaves less concrete than the other
pending needs before a far-away top-up is penalised; when the top-up is close or nothing else is pending, spending everything is fine."""
from dataclasses import dataclass
from typing import List, Optional

from candidates import Candidate, generate
from world_model import WorldModel, next_topup

INF = 1e9


@dataclass
class Weights:
    W_CONN: float = 10.0
    W_DEMAND: float = 0.8
    W_OVER: float = 4.0
    W_COST: float = 0.6
    W_REUSE: float = 1.5
    W_LEN: float = 0.15
    W_WAIT: float = 1.0
    W_UNAFFORDABLE: float = 0.8
    W_REFUND: float = 2.0
    W_REMOVE_BASE: float = 0.5
    k_alternatives: int = 2
    greedy: bool = False          # baseline A: nearest-house shortest route, no resource terms
    serve_unreached_dests: bool = False   # S35 fix: also target destinations no house of their colour can reach (default off: S18-S34 behaviour unchanged)
    route_lambda: Optional[float] = None  # S35 geometry: route search cost = length + lambda * new concrete (None = lexicographic concrete-first, S18-S34 behaviour)
    bypass: bool = False                  # S37: shortcut candidates for connected houses with a long detour (see candidates.bypass_candidates)
    bypass_ratio: float = 2.0             # intervene when driving distance >= ratio x shortest legal path
    bypass_max_cost: int = 10             # never spend more than this many tiles on one bypass
    bypass_min_demand: int = 3            # traffic gate: only colours whose destinations hold >= this many pins (or are overcrowded) get a bypass
    W_BYPASS: float = 1.0                 # score per tile of driving saved (drive - shortest); cost term as for CONNECT


def scarcity(wm: WorldModel, cost: int, other_needs: int) -> float:
    """multiplier >= 1: how precious each unit of concrete is right now."""
    dt, amt = next_topup(wm.t)
    left = wm.concrete - cost
    if dt is None:
        dt, amt = 140, 0
    shortfall = max(0, other_needs - max(0, left))
    return 1.0 + (shortfall / max(1, amt + 1)) * min(1.0, dt / 140.0)


def score_all(wm: WorldModel, cands: List[Candidate], w: Weights) -> None:
    connects = [c for c in cands if c.kind == "CONNECT"]
    # cheapest route per disconnected house other than the candidate's own house = "other pending needs"
    cheapest = {}
    for c in connects:
        h = c.targets["house"]
        cheapest[h] = min(cheapest.get(h, 1e9), c.concrete_cost)
    for c in cands:
        terms = {}
        if c.kind == "WAIT":
            c.score, c.score_terms = 0.0, {"wait": 0.0}
            continue
        if c.kind == "MOTORWAY":
            c.score, c.score_terms = -INF, {"unsupported": -INF}
            continue
        if c.kind == "REMOVE":
            short = [cheapest[h] - wm.concrete for h in cheapest if cheapest[h] > wm.concrete]
            terms["refund"] = w.W_REFUND if (short and min(short) <= 3) else 0.0
            terms["base"] = -w.W_REMOVE_BASE
            c.score = sum(terms.values())
            c.score_terms = terms
            continue
        # CONNECT
        if c.would_destroy_functional_connection or c.confidence == 0.0:
            c.score, c.score_terms = -INF, {"destroys_or_invalid": -INF}
            continue
        h = c.targets["house"]
        if c.targets.get("bypass"):                      # S37 shortcut for an already-connected house: value = driving tiles saved, cost as for any construction
            other = sum(v for k, v in cheapest.items() if k != h)
            terms["bypass_gain"] = w.W_BYPASS * (c.targets["drive"] - c.targets["shortest"])
            terms["cost"] = -w.W_COST * c.concrete_cost * scarcity(wm, c.concrete_cost, other)
            c.score, c.score_terms = sum(terms.values()), terms
            continue
        if w.greedy:                                   # baseline A: nearest house (cheapest route), no demand / scarcity / reuse reasoning
            terms["base"] = 100.0
            terms["neg_cost"] = -float(c.concrete_cost)
            terms["neg_len"] = -0.01 * c.route_len
            terms["unaffordable"] = -100.0 if c.concrete_cost > wm.concrete else 0.0
            c.score, c.score_terms = sum(terms.values()), terms
            continue
        g = c.targets["group"]
        pins, over = wm.demand_of_group(g)
        other = sum(v for k, v in cheapest.items() if k != h)
        terms["conn"] = w.W_CONN * c.connectivity_delta
        terms["demand"] = w.W_DEMAND * pins
        terms["overcrowded"] = w.W_OVER if over else 0.0
        terms["cost"] = -w.W_COST * c.concrete_cost * scarcity(wm, c.concrete_cost, other)
        terms["reuse"] = w.W_REUSE * c.reused_edges / max(1, c.route_len)
        terms["len_proxy"] = -w.W_LEN * c.route_len
        terms["waited"] = w.W_WAIT * min(60, wm.t - wm.houses[h]["spawn_t"]) / 60.0
        terms["unaffordable"] = -w.W_UNAFFORDABLE * max(0, c.concrete_cost - wm.concrete)
        c.score, c.score_terms = sum(terms.values()), terms


def plan(wm: WorldModel, w: Weights, banned=frozenset(), protected=frozenset()) -> Optional[Candidate]:
    """deterministic: candidates generated in a fixed order, best score wins, ties by (kind order, key)."""
    import candidates as _C
    _C.ROUTE_LAMBDA = w.route_lambda
    cands = generate(wm, w.k_alternatives, banned, protected, include_motorway=False, serve_unreached_dests=w.serve_unreached_dests, bypass=w.bypass, bypass_ratio=w.bypass_ratio, bypass_max_cost=w.bypass_max_cost, bypass_min_demand=w.bypass_min_demand)
    score_all(wm, cands, w)
    order = {"CONNECT": 0, "REMOVE": 1, "WAIT": 2, "MOTORWAY": 3}
    best = max(cands, key=lambda c: (c.score, -order[c.kind], tuple(-x for x in (c.concrete_cost, c.route_len)), tuple(c.adds)))
    return best
