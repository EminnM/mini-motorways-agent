"""S35 strategic controller on top of planner B (S18-H executor + S21 native adapter). Isolated: imports the S18-H modules, modifies nothing shared.

Two strategic decision types, exercised ONLY at planner B's own plan-selection points (when the executor has no current plan) and ONLY when a meaningful choice exists:
  D1 PRIORITY : which disconnected house to connect next, among the CONNECT candidates that are feasible now (route found, affordable, no functional connection destroyed).
                options: 'planner' (planner B's score) | 'demand' (house whose colour group has the most pins / is overcrowded; tie -> planner score) | 'cheapest' (least concrete; tie -> planner score)
  D2 ROUTE    : for the chosen house, 'reuse' (planner B's lexicographic cheapest route, which joins existing roads wherever it can) | 'dedicated' (a route that avoids every existing
                road edge of a MIXED component, i.e. one already carrying houses of another colour group; new road all the way, joining the network only at its last tile).
                'dedicated' is offered only if such a route exists, is affordable now, differs from the reuse route, and destroys nothing.
Everything else (execution one primitive per decision, invalidation/abandon logic, destroy vetoes, mask legality, upgrade interrupts, REMOVE/WAIT fallbacks) is planner B's own code:
the controller only substitutes WHICH Candidate becomes the executor's plan. If no feasible CONNECT exists the planner's own choice is used unchanged (safe fallback).
Rules: PlannerRule (= planner B exactly, used to verify the wrapper is transparent), FixedRule(priority, route), RandomRule(rng) (uniform over the live options at each decision point)."""
import os
import random
import sys
import time
from typing import Dict, List, Optional, Set

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(ROOT, "research", "s21_native_check", "code"))
import native_adapter  # noqa: E402,F401  (installs the native-legality patch and sys.path for the S18-H modules)
from synthetic_rl import layout as L  # noqa: E402
from candidates import Candidate, dijkstra_route, generate, route_to_candidate  # noqa: E402
from executor import Executor  # noqa: E402
from planner import Weights, plan, score_all  # noqa: E402
from world_model import WorldModel, ends  # noqa: E402

PRIORITY_OPTIONS = ("planner", "demand", "cheapest")
ROUTE_OPTIONS = ("reuse", "dedicated")


def mixed_component_edges(wm: WorldModel) -> Set:
    """active road edges whose functional component contains houses of >= 2 colour groups."""
    groups: Dict = {}
    for h, v in wm.houses.items():
        groups.setdefault(wm.component(h), set()).add(v["group"])
    mixed = {c for c, gs in groups.items() if len(gs) >= 2}
    return {e for e in wm.active if wm.component(ends(e)[0]) in mixed}


def dedicated_candidate(wm: WorldModel, house, banned: Set) -> Optional[Candidate]:
    goals = wm.goal_tiles_for_group(wm.houses[house]["group"])
    if not goals:
        return None
    path = dijkstra_route(wm, house, goals, {}, set(banned) | mixed_component_edges(wm))
    if path is None:
        return None
    c = route_to_candidate(wm, house, path, alt=9)
    c.targets["style"] = "dedicated"
    return c


def feasible(c: Candidate, wm: WorldModel) -> bool:
    return c.kind == "CONNECT" and c.confidence > 0 and not c.would_destroy_functional_connection and c.concrete_cost <= wm.concrete


class PlannerRule:
    name = "planner"

    def choose(self, point):
        return "planner", "reuse"


class FixedRule:
    def __init__(self, priority: str, route: str):
        assert priority in PRIORITY_OPTIONS and route in ROUTE_OPTIONS
        self.priority, self.route = priority, route
        self.name = f"{priority}+{route}"

    def choose(self, point):
        return self.priority, self.route


class RandomRule:
    name = "random"

    def __init__(self, seed: int):
        self.rng = random.Random(seed)

    def choose(self, point):
        return self.rng.choice(PRIORITY_OPTIONS), self.rng.choice(ROUTE_OPTIONS)


class StrategicExecutor(Executor):
    """Executor.act with the plan-selection step replaced by a strategic choice (the rest of act() is copied verbatim from S18-H so behaviour is otherwise identical)."""

    def __init__(self, rule, weights: Optional[Weights] = None):
        super().__init__(weights, log=False)
        self.rule = rule
        self.points: List[Dict] = []          # decision-point log (per episode; reset with the episode)

    def reset_episode(self):
        super().reset_episode()
        self.points = []

    def _select(self, wm: WorldModel, banned, protected) -> Candidate:
        default = plan(wm, self.w, banned, protected)
        cands = generate(wm, self.w.k_alternatives, banned, protected, include_motorway=False)
        score_all(wm, cands, self.w)
        feas = [c for c in cands if feasible(c, wm)]
        if not feas or default.kind != "CONNECT" or not feasible(default, wm):
            return default                                                 # no meaningful choice: planner B's own decision
        houses = sorted({c.targets["house"] for c in feas})
        pr, rt = self.rule.choose(None)
        # D1 priority
        best_by_house = {}
        pkey = lambda c: (c.score, tuple(-x for x in (c.concrete_cost, c.route_len)), tuple(c.adds))   # noqa: E731  planner B's exact tie-break (planner.plan) so 'planner'+'reuse' == planner B
        for c in feas:                                                     # planner's preferred candidate per house
            h = c.targets["house"]
            if h not in best_by_house or pkey(c) > pkey(best_by_house[h]):
                best_by_house[h] = c
        if pr == "planner" or len(houses) < 2:
            chosen_house = default.targets["house"]
        elif pr == "demand":
            chosen_house = max(houses, key=lambda h: (wm.demand_of_group(wm.houses[h]["group"])[0] + (100 if wm.demand_of_group(wm.houses[h]["group"])[1] else 0), best_by_house[h].score))
        else:
            chosen_house = min(houses, key=lambda h: (best_by_house[h].concrete_cost, -best_by_house[h].score))
        reuse = best_by_house[chosen_house]
        # D2 route
        ded = dedicated_candidate(wm, chosen_house, banned)
        ded_ok = ded is not None and feasible(ded, wm) and tuple(ded.adds) != tuple(reuse.adds)
        pick = ded if (rt == "dedicated" and ded_ok) else reuse
        self.points.append({"t": wm.t, "n_houses": len(houses), "d1_live": len(houses) >= 2, "d2_live": bool(ded_ok), "priority": pr, "route": rt if ded_ok else "reuse",
                            "planner_house": list(default.targets["house"]), "chosen_house": list(chosen_house), "reuse_cost": reuse.concrete_cost, "dedicated_cost": ded.concrete_cost if ded_ok else None,
                            "differs_from_planner": tuple(pick.adds) != tuple(default.adds), "concrete": wm.concrete})
        return pick

    def act(self, env) -> int:
        t0 = time.perf_counter()
        wm = WorldModel(env)
        if wm.t < self.last_t or self.last_t < 0:
            if self.last_t >= 0:
                self.reset_episode()
        self.last_t = wm.t
        self.stats["decisions"] += 1
        banned, protected = self._banned_and_protected(wm.t)
        if self.plan is not None:
            if self.plan.kind == "CONNECT" and self.plan.targets["house"] in wm.connected:
                self._complete()
            elif self.plan.kind == "REMOVE" and self.plan.removes[0] not in wm.edges:
                self._complete()
            else:
                r = self._invalidated(wm)
                if r is not None:
                    self._abandon(r)
                elif self.plan.kind == "CONNECT" and all(e in self.done_adds for e in self.plan.adds):
                    self._abandon("all_edges_built_but_not_connected")
        if self.plan is None:
            self.plan = self._select(wm, banned, protected)                # <-- the only change vs Executor.act
            self.plan_t0 = wm.t
            self.done_adds = set()
            if self.plan.kind in ("CONNECT", "REMOVE"):
                self.stats["plans"] += 1
            if self.plan.would_destroy_functional_connection:
                self.stats["destroy_vetoes"] += 1
                self._abandon("would_destroy")
                self.stats["compute_s"] += time.perf_counter() - t0
                return 0
        a = 0
        p = self.plan
        if p.kind == "CONNECT":
            for e in p.adds:
                if e in self.done_adds:
                    continue
                if wm.edges.get(e) == "A":
                    self.done_adds.add(e)
                    continue
                idx = L.action_index("ADD", *e)
                if wm.legal_now("ADD", e):
                    if wm.lost_connections([e], []):
                        self.stats["destroy_vetoes"] += 1
                        self._abandon("would_destroy")
                        a = 0
                    else:
                        a = idx
                        self.done_adds.add(e)
                        self.recent_added[e] = wm.t
                elif wm.structurally_buildable(e) and wm.concrete == 0:
                    self.stats["waits_for_concrete"] += 1
                    a = 0
                else:
                    self.stats["illegal_suppressed"] += 1
                    self._abandon("mask_changed")
                    a = 0
                break
        elif p.kind == "REMOVE":
            e = p.removes[0]
            if wm.legal_now("REMOVE", e) and not wm.lost_connections([], [e]):
                a = L.action_index("REMOVE", *e)
                self.recent_removed[e] = wm.t
            else:
                self._abandon("remove_invalid")
        if p.kind == "WAIT":
            self.plan = None
        if not wm.mask[a]:
            self.stats["illegal_suppressed"] += 1
            a = 0
        self.stats["compute_s"] += time.perf_counter() - t0
        return int(a)
