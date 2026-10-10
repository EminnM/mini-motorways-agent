"""EVALUATION-ONLY anti-cycle action selector (S6). Not used by training; no permanent change to any training action selector.

Purpose: a causal probe. If a deterministic policy is trapped in a two-state road-network oscillation A -> B -> A -> B, skip the
cycle-continuing action for that one decision and take the next-highest-ranked LEGAL action instead.

What it does NOT do: change the native action mask, the network, the weights, the reward, the observation given to the policy, or the game.
It chooses among actions the native mask already allows, using the policy's own logit ranking.

PRIVILEGED / EVALUATION-ONLY INFORMATION: the filter keeps a per-episode history of road-network signatures (built from the raw native
observation, i.e. outside the policy's 11,666-float input) and of (state, action) -> next-state transitions it has itself observed. This
memory never reaches the policy; it only vetoes an action at selection time. Any result obtained with the filter is an *intervention* result.

Cycle criterion (state-based, not action-index-based). With road-state signatures s_0..s_t (s_t = current decision's state):
    detected  <=>  t >= 3  and  s_t == s_{t-2}  and  s_{t-1} == s_{t-3}  and  s_t != s_{t-1}
i.e. the road network has actually gone A,B,A,B (two full oscillation periods).  Repeated WAITs (A,A) or a single revisit (A,B,A) never trigger.
Intervention (only when detected AND the policy's top-ranked legal action is a *known* cycle-continuing action):
    excluded = {legal a : observed transition (s_t, a) -> s_{t-1}}        (re-creates the other cycle state)
    walk the policy ranking from the top; skip excluded actions; reject a candidate whose observed transition lands in a state already
    recognised as part of a cycle this episode (recorded as a rejection with its reason); choose the first remaining candidate.
    If every legal action is rejected, keep the original action and record 'no_alternative'.
Candidates with unobserved transitions are accepted (their effect is unknown until tried). Longer cycles (period >= 3) are not intervened on;
they are only counted by `revisit_stats`."""
import hashlib
from typing import Any, Dict, List, Optional, Sequence, Tuple


def road_signature(raw: Dict[str, Any]) -> str:
    """Hash of the full road-network configuration (edge, state) from the RAW native observation. Evaluation-only/privileged."""
    items = []
    for t in raw["tiles"]:
        c = t["coord"]
        r = t["road"]
        for d in ("E", "S"):
            if r[d] != "None":
                items.append((c[0], c[1], d, r[d]))
    items.sort()
    return hashlib.sha1(repr(items).encode()).hexdigest()[:16]


class AntiCycleFilter:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.reset()

    def reset(self):
        self.hist: List[str] = []          # signatures of PREVIOUS decisions' states
        self.trans: Dict[Tuple[str, int], str] = {}
        self.cycle_states = set()
        self.interventions: List[Dict[str, Any]] = []
        self.n_detected = 0                # decisions at which the A,B,A,B criterion held
        self.detected_not_continuing = 0   # criterion held but the top action was not a known cycle-continuing action

    def detected(self, s_t: str) -> bool:
        seq = self.hist + [s_t]
        return len(seq) >= 4 and seq[-1] == seq[-3] and seq[-2] == seq[-4] and seq[-1] != seq[-2]

    def choose(self, s_t: str, ranking: Sequence[int], t: int) -> Tuple[int, Optional[Dict[str, Any]]]:
        """ranking = LEGAL actions in descending policy preference (ties already broken deterministically). Returns (action, intervention-or-None)."""
        original = int(ranking[0])
        if not self.enabled or not self.detected(s_t):
            return original, None
        self.n_detected += 1
        other = self.hist[-1]
        self.cycle_states.update((s_t, other))
        excluded = [int(a) for a in ranking if self.trans.get((s_t, int(a))) == other]
        if original not in excluded:
            self.detected_not_continuing += 1
            return original, None
        rejected = [(a, "continues_cycle") for a in excluded]
        chosen = None
        for rank, a in enumerate(ranking):
            a = int(a)
            if a in excluded:
                continue
            nxt = self.trans.get((s_t, a))
            if nxt is not None and nxt in self.cycle_states:
                rejected.append((a, "known_transition_into_cycle_state"))
                continue
            chosen, chosen_rank = a, rank
            break
        iv = {"step": t, "state": s_t, "other_cycle_state": other, "original_action": original, "excluded": excluded, "rejected": rejected}
        if chosen is None:
            iv.update({"chosen_action": original, "chosen_rank": 0, "outcome": "no_alternative"})
            self.interventions.append(iv)
            return original, iv
        iv.update({"chosen_action": chosen, "chosen_rank": chosen_rank, "outcome": "replaced"})
        self.interventions.append(iv)
        return chosen, iv

    def observe(self, s_t: str, action: int, s_next: str):
        """Call after the step: records the transition and appends the state to the history."""
        self.trans[(s_t, int(action))] = s_next
        self.hist.append(s_t)


def ranking_from_logits(logits, mask) -> List[int]:
    """Legal actions sorted by descending logit; ties -> lower index (same rule as argmax over the masked logits)."""
    import numpy as np
    legal = np.flatnonzero(mask)
    order = np.argsort(-np.asarray(logits)[legal], kind="stable")
    return [int(a) for a in legal[order]]


def revisit_stats(sigs: Sequence[str]) -> Dict[str, int]:
    """Post-hoc, policy-independent road-state statistics of a signature sequence s_0..s_T (state at each decision).
    two_state_oscillation_steps : t with s_t == s_{t-2} != s_{t-1}
    two_state_cycles            : maximal runs of >= 4 consecutive states alternating A,B,A,B...
    revisits                    : decisions whose state was already seen earlier in the episode (any period, excluding s_t == s_{t-1})"""
    n = len(sigs)
    osc = sum(1 for t in range(2, n) if sigs[t] == sigs[t - 2] and sigs[t] != sigs[t - 1])
    cycles, t = 0, 0
    while t < n - 3:
        if sigs[t] == sigs[t + 2] and sigs[t + 1] == sigs[t + 3] and sigs[t] != sigs[t + 1]:
            k = t + 4
            while k < n and sigs[k] == sigs[k - 2]:
                k += 1
            cycles += 1
            t = k
        else:
            t += 1
    seen, rev = set(), 0
    for t, s in enumerate(sigs):
        if s in seen and not (t > 0 and sigs[t - 1] == s):
            rev += 1
        seen.add(s)
    return {"two_state_oscillation_steps": osc, "two_state_cycles": cycles, "revisits": rev}
