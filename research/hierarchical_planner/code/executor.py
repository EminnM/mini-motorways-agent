"""S18-H persistent executor: select candidate -> validate legality -> one primitive action per decision -> track progress ->
detect invalidation (spawn on the route, road-state change, mask change) -> replan with a logged reason.
Guarantees: every emitted action is legal under the env mask at emission time (else WAIT); no ADD of an edge removed within the last
OSC_WINDOW decisions and no REMOVE of an edge added within it (no oscillation); no action whose predicted result destroys a functional
connection is ever emitted (the plan is abandoned with reason 'would_destroy'). The executor reads only the WorldModel (visible state)."""
import time
from typing import Any, Dict, List, Optional, Set

from synthetic_rl import layout as L  # noqa: E402  (sys.path set by world_model)

from candidates import Candidate
from planner import Weights, plan
from world_model import Edge, WorldModel, ends, house_edges

OSC_WINDOW = 30


class Executor:
    def __init__(self, weights: Optional[Weights] = None, log: bool = False):
        self.w = weights or Weights()
        self.log_enabled = log
        self.reset_episode()

    def reset_episode(self):
        self.plan: Optional[Candidate] = None
        self.plan_t0 = 0
        self.done_adds: Set[Edge] = set()
        self.recent_added: Dict[Edge, int] = {}
        self.recent_removed: Dict[Edge, int] = {}
        self.last_t = -1
        self.pending_house_edge: Optional[Edge] = None
        self.stats: Dict[str, Any] = {"plans": 0, "completed": 0, "abandoned": 0, "abandon_reasons": {}, "waits_for_concrete": 0, "compute_s": 0.0, "decisions": 0, "illegal_suppressed": 0, "destroy_vetoes": 0}
        self.log: List[str] = []

    def _abandon(self, reason: str):
        self.stats["abandoned"] += 1
        self.stats["abandon_reasons"][reason] = self.stats["abandon_reasons"].get(reason, 0) + 1
        if self.log_enabled:
            self.log.append(f"t={self.last_t} ABANDON {self.plan.describe() if self.plan else None} reason={reason}")
        self.plan = None

    def _complete(self):
        self.stats["completed"] += 1
        if self.log_enabled:
            self.log.append(f"t={self.last_t} COMPLETE {self.plan.describe()}")
        self.plan = None

    def _invalidated(self, wm: WorldModel) -> Optional[str]:
        p = self.plan
        if p.kind == "CONNECT":
            h = p.targets["house"]
            if h not in wm.houses:
                return "house_gone"
            if "dest" in p.targets:               # S35 fix: destination-targeted plan (house already connected elsewhere)
                d = tuple(p.targets["dest"])
                if wm.content.get(d) != "Destination":
                    return "dest_gone"
                if self._dest_served(wm, p):
                    return None                   # completion handled by caller
            elif h in wm.connected and not p.targets.get("bypass"):
                return None                       # completion handled by caller (bypass plans fall through to the route checks below)
            for tile in p.path[1:]:
                if tile in wm.house_set or tile in wm.blocked:
                    return "spawn_on_route"
            for e in p.adds:
                if e in self.done_adds:
                    if wm.edges.get(e) != "A":
                        return "road_state_changed"
                    continue
                if not wm.structurally_buildable(e):
                    return "edge_no_longer_buildable"
            # predicted-result safety: the remaining ADDs must not destroy any functional connection
            remaining = [e for e in p.adds if e not in self.done_adds]
            if wm.lost_connections(remaining, []):
                return "would_destroy"
        elif p.kind == "REMOVE":
            e = p.removes[0]
            if e not in wm.edges or not wm.legal_now("REMOVE", e) or wm.lost_connections([], [e]):
                return "remove_invalid"
        return None

    @staticmethod
    def _dest_served(wm: WorldModel, p: Candidate) -> bool:
        """a destination-targeted CONNECT is complete when the destination's component contains a house of its colour."""
        d = tuple(p.targets["dest"])
        comp = wm.component(d)
        g = p.targets["group"]
        return any(v["group"] == g and wm.component(h) == comp for h, v in wm.houses.items())

    def _banned_and_protected(self, t: int):
        self.recent_added = {e: t0 for e, t0 in self.recent_added.items() if t - t0 <= OSC_WINDOW}
        self.recent_removed = {e: t0 for e, t0 in self.recent_removed.items() if t - t0 <= OSC_WINDOW}
        return set(self.recent_removed), set(self.recent_added)

    def act(self, env) -> int:
        t0 = time.perf_counter()
        wm = WorldModel(env)
        if wm.t < self.last_t or self.last_t < 0:
            if self.last_t >= 0:
                self.reset_episode()
        self.last_t = wm.t
        self.stats["decisions"] += 1
        banned, protected = self._banned_and_protected(wm.t)

        # progress / invalidation of the current plan
        if self.plan is not None:
            if self.plan.kind == "CONNECT" and self.plan.targets.get("bypass"):      # S37 shortcut: complete when every edge of the route is Active
                if all(wm.edges.get(e) == "A" for e in self.plan.adds):
                    self._complete()
                else:
                    r = self._invalidated(wm)
                    if r is not None:
                        self._abandon(r)
            elif self.plan.kind == "CONNECT" and "dest" in self.plan.targets:
                if self._dest_served(wm, self.plan):
                    self._complete()
                else:
                    r = self._invalidated(wm)
                    if r is not None:
                        self._abandon(r)
                    elif all(e in self.done_adds for e in self.plan.adds):
                        self._abandon("all_edges_built_but_dest_not_served")
            elif self.plan.kind == "CONNECT" and self.plan.targets["house"] in wm.connected:
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
            self.plan = plan(wm, self.w, banned, protected)
            self.plan_t0 = wm.t
            self.done_adds = set()
            if self.plan.kind in ("CONNECT", "REMOVE"):
                self.stats["plans"] += 1
                if self.log_enabled:
                    self.log.append(f"t={wm.t} PLAN {self.plan.describe()} score={self.plan.score:.2f} terms={ {k: round(v, 2) for k, v in self.plan.score_terms.items()} }")
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
                    if wm.lost_connections([e], []):          # never silently destroy a functional connection
                        self.stats["destroy_vetoes"] += 1
                        self._abandon("would_destroy")
                        a = 0
                    else:
                        a = idx
                        self.done_adds.add(e)
                        self.recent_added[e] = wm.t
                elif wm.structurally_buildable(e) and wm.concrete == 0:
                    self.stats["waits_for_concrete"] += 1      # plan kept; the next top-up is known from the clock
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
            self.plan = None                                     # WAIT is re-evaluated every decision
        if not wm.mask[a]:
            self.stats["illegal_suppressed"] += 1
            a = 0
        self.stats["compute_s"] += time.perf_counter() - t0
        return int(a)


def make_policy(weights: Optional[Weights] = None, log: bool = False) -> Executor:
    """returns the Executor; use ex.act as the policy callable env -> action."""
    return Executor(weights, log)
