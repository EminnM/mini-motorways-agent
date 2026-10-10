"""Measurement-only native diagnostics (S4 / P2). READ-ONLY: consumes data the environment already returned (raw native
observation dict, action mask, reward, info) and never writes to the environment, the policy inputs, rewards, masks, stepping or
termination. The useful-connection metric uses the groupIndex field of the RAW native observation (not part of the policy's 11,666-float
vector); it is a diagnostic only. Any exception inside the diagnostics is swallowed and counted (`diag_errors`) so a diagnostics bug
cannot alter or crash an episode.

Definitions
  delivery            = positive native scoreDelta event (the score is incremented per completed delivery; unit size recorded in
                        `diag_score_event_values`, so the 1-point-per-delivery assumption is checkable).
  useful connection   = connectivity_potential(raw obs): number of houses whose road-connected component contains a destination
                        of the same colour group (the same routine as the existing `--shaping` potential, used here WITH NO REWARD EFFECT).
  road edge           = an Active/Pending/Mothballed/VisiblyActive road entry on the E or S side of a tile (each undirected edge once).
  churn               = REMOVE of an edge the agent itself added earlier in the episode, plus re-ADD of an edge the agent removed.
  repeated edit       = the same (kind, edge) applied more than once in an episode."""
import math
import time
from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np

from . import paths  # noqa: F401  (python/ on sys.path)

CAP = 300


def _road_edges(raw: Dict[str, Any]) -> int:
    n = 0
    for t in raw["tiles"]:
        r = t["road"]
        if r["E"] != "None":
            n += 1
        if r["S"] != "None":
            n += 1
    return n


class NativeDiagnostics:
    def __init__(self, connectivity: bool = True):
        self.connectivity = connectivity
        self.reset(None)

    def reset(self, raw0: Optional[Dict[str, Any]]):
        self.errors = 0
        self.step_i = 0
        self.score_events: List[List[float]] = []
        self.nonzero_rewards: List[List[float]] = []
        self.n_nonzero_reward = 0
        self.n_nonzero_nonterminal = 0
        self.first_nonzero_nonterminal = None
        self.interm_sum = 0.0
        self.term_sum = 0.0
        self.deliveries = 0.0
        self.first_delivery = None
        self.phi = 0
        self.phi_max = 0
        self.first_connection = None
        self.n_conn_up = 0
        self.n_conn_down = 0
        self.own_added = set()
        self.removed_by_agent = set()
        self.edit_counts = Counter()
        self.n_remove_own = 0
        self.n_readd = 0
        self.legal_total: List[int] = []
        self.legal_add: List[int] = []
        self.legal_remove: List[int] = []
        self.legal_upgrade = 0
        self.act_idx = Counter()
        self.entropies: List[float] = []
        self.n_accepted_add = 0
        self.n_accepted_remove = 0
        self.edges0 = self.edges_max = self.edges_last = None
        self.final_status = None
        self.terminal_on_entry = None
        self.terminal_penalty_missing = None
        self.sim_prev = None if raw0 is None else raw0.get("simulationTime")
        self.t_prev = time.time()
        try:
            if raw0 is not None:
                self.edges0 = self.edges_max = self.edges_last = _road_edges(raw0)
                if self.connectivity:
                    from mini_motorways_env import connectivity_potential
                    self.phi = self.phi_max = int(connectivity_potential(raw0))
                    if self.phi >= 1:
                        self.first_connection = 0
        except Exception:
            self.errors += 1

    def add_step(self, action: int, accepted: Any, reward: float, info: Dict[str, Any], terminated: bool, truncated: bool,
                 raw_after: Optional[Dict[str, Any]], mask_before: Optional[np.ndarray], policy_entropy: Optional[float] = None):
        i = self.step_i
        self.step_i += 1
        now = time.time()
        wall_interval, self.t_prev = now - self.t_prev, now
        try:
            if terminated or truncated:
                self._final_status(action, accepted, reward, info, terminated, truncated, raw_after, wall_interval)
        except Exception:
            self.errors += 1
        try:
            from synthetic_rl import layout as L
            self.act_idx[int(action)] += 1
            if policy_entropy is not None and not math.isnan(policy_entropy):
                self.entropies.append(float(policy_entropy))
            if mask_before is not None:
                idx = np.flatnonzero(mask_before)
                self.legal_total.append(int(idx.size))
                kinds = [L.decode_action(int(a))[0] for a in idx]
                self.legal_add.append(kinds.count("ADD"))
                self.legal_remove.append(kinds.count("REMOVE"))
                self.legal_upgrade += kinds.count("UPGRADE")
            rc = info.get("rewardComponents", {})
            sd, tp = float(rc.get("scoreDelta", 0.0)), float(rc.get("terminalPenalty", 0.0))
            self.interm_sum += sd
            self.term_sum += tp
            if sd != 0.0:
                if len(self.score_events) < CAP:
                    self.score_events.append([i, sd])
                if sd > 0:
                    self.deliveries += sd
                    if self.first_delivery is None:
                        self.first_delivery = i
            if float(reward) != 0.0:
                self.n_nonzero_reward += 1
                if len(self.nonzero_rewards) < CAP:
                    self.nonzero_rewards.append([i, float(reward)])
                if sd != 0.0:
                    self.n_nonzero_nonterminal += 1
                    if self.first_nonzero_nonterminal is None:
                        self.first_nonzero_nonterminal = i
            dec = L.decode_action(int(action))
            if dec[0] in ("ADD", "REMOVE") and accepted is not False:
                edge = (dec[1], dec[2], dec[3])
                self.edit_counts[(dec[0], edge)] += 1
                if dec[0] == "ADD":
                    self.n_accepted_add += 1
                    if edge in self.removed_by_agent:
                        self.n_readd += 1
                    self.own_added.add(edge)
                else:
                    self.n_accepted_remove += 1
                    if edge in self.own_added:
                        self.n_remove_own += 1
                    self.removed_by_agent.add(edge)
            if raw_after is not None:
                self.sim_prev = raw_after.get("simulationTime", self.sim_prev)
                self.edges_last = _road_edges(raw_after)
                self.edges_max = max(self.edges_max or 0, self.edges_last)
                if self.connectivity:
                    from mini_motorways_env import connectivity_potential
                    p = int(connectivity_potential(raw_after))
                    if p > self.phi:
                        self.n_conn_up += p - self.phi
                    elif p < self.phi:
                        self.n_conn_down += self.phi - p
                    self.phi = p
                    self.phi_max = max(self.phi_max, p)
                    if p >= 1 and self.first_connection is None:
                        self.first_connection = i + 1        # connection present after this decision
        except Exception:
            self.errors += 1

    def _final_status(self, action, accepted, reward, info, terminated, truncated, raw_after, wall_interval):
        """Measurement only (S6): classify how the episode ended. Never feeds back into the environment or the reward.
        terminal_on_entry   = the final STEP started with the game already terminal (applyInfo 'terminal state (not applied)', accepted False)
        terminal_penalty_missing = native game-over (terminated, not truncated) whose terminalPenalty component is 0"""
        rc = info.get("rewardComponents", {})
        ai = info.get("applyInfo")
        on_entry = (ai == "terminal state (not applied)") if ai is not None else (accepted is False)
        pen = float(rc.get("terminalPenalty", 0.0))
        sim_after = raw_after.get("simulationTime") if raw_after is not None else None
        self.terminal_on_entry = bool(terminated and on_entry)
        self.terminal_penalty_missing = bool(terminated and not truncated and pen == 0.0)
        self.final_status = {"reward": float(reward), "terminated": bool(terminated), "truncated": bool(truncated), "accepted": accepted,
                             "appliedAction": info.get("appliedAction"), "applyInfo": ai, "rewardComponents": rc, "decisionIndex": info.get("decisionIndex"),
                             "sim_time_before": self.sim_prev, "sim_time_after": sim_after,
                             "sim_time_advance": (None if sim_after is None or self.sim_prev is None else round(sim_after - self.sim_prev, 6)),
                             "wall_interval_s": round(wall_interval, 4)}

    def row(self) -> Dict[str, Any]:
        n = max(self.step_i, 1)
        tot = sum(self.act_idx.values())
        emp_ent = -sum((c / tot) * math.log(c / tot) for c in self.act_idx.values()) if tot else None
        rep = sum(1 for v in self.edit_counts.values() if v > 1)
        fc, fd = self.first_connection, self.first_delivery
        lm = lambda xs: (float(np.mean(xs)) if xs else None)
        return {
            "diag_deliveries": self.deliveries, "diag_first_delivery_step": fd,
            "diag_score_event_values": sorted({v for _, v in self.score_events}),
            "diag_n_score_events": len(self.score_events),
            "diag_n_nonzero_reward": self.n_nonzero_reward, "diag_n_nonzero_nonterminal_reward": self.n_nonzero_nonterminal,
            "diag_first_nonzero_nonterminal_step": self.first_nonzero_nonterminal,
            "diag_intermediate_reward_sum": self.interm_sum, "diag_terminal_reward_sum": self.term_sum,
            "diag_nonzero_reward_events": self.nonzero_rewards,
            "diag_connections_max": self.phi_max, "diag_connections_final": self.phi,
            "diag_first_connection_step": fc, "diag_connection_to_first_delivery_steps": (fd - fc if fc is not None and fd is not None else None),
            "diag_connection_increments": self.n_conn_up, "diag_connection_decrements": self.n_conn_down,
            "diag_road_edges_initial": self.edges0, "diag_road_edges_max": self.edges_max, "diag_road_edges_final": self.edges_last,
            "diag_n_add_accepted": self.n_accepted_add, "diag_n_remove_accepted": self.n_accepted_remove,
            "diag_distinct_edges_added": len({e for (k, e) in self.edit_counts if k == "ADD"}),
            "diag_repeated_edits": rep, "diag_remove_of_own_add": self.n_remove_own, "diag_readd_after_remove": self.n_readd,
            "diag_churn": self.n_remove_own + self.n_readd,
            "diag_legal_mean": lm(self.legal_total), "diag_legal_min": (min(self.legal_total) if self.legal_total else None),
            "diag_legal_max": (max(self.legal_total) if self.legal_total else None),
            "diag_legal_add_mean": lm(self.legal_add), "diag_legal_remove_mean": lm(self.legal_remove), "diag_legal_upgrade_steps": self.legal_upgrade,
            "diag_empirical_action_entropy": emp_ent, "diag_distinct_actions": len(self.act_idx),
            "diag_policy_entropy_mean": lm(self.entropies),
            "diag_terminal_penalty_missing": self.terminal_penalty_missing, "diag_terminal_on_entry": self.terminal_on_entry,
            "diag_final_status": self.final_status,
            "diag_errors": self.errors,
        }
