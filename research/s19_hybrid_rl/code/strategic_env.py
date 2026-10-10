"""S19 strategic (semi-Markov) environment on top of the S13 MultiHouseEnv and the S18-H world model / candidate generator / executor.

At each STRATEGIC decision point the agent sees a list of candidates (CONNECT routes incl. k alternatives, REMOVE-redundant edges) plus WAIT,
ordered by a STABLE GEOMETRIC KEY (never by heuristic score), capped at K slots with a validity mask (slot K = WAIT, always legal).
Selecting a candidate commits it to a persistent executor (S18-H rules: one legal primitive action per decision, invalidation -> abandon);
WAIT advances the env by a FIXED interval of WAIT_INTERVAL primitive decisions.  A strategic step returns the accumulated primitive reward,
the number of primitive decisions elapsed (info['duration']) and the within-step discounted reward (info['reward_disc'] = sum_j gamma^j r_j),
so a trainer can use gamma^duration bootstrapping (SMDP correctness).

Visible-state rule: everything the policy sees comes from WorldModel (env.content/houses/dests/edges/free/concrete/total/t/water/action_masks)
plus env.score; env.pending / scenario events are never read (asserted by inspection: WorldModel.__init__ and global_features are the only env
readers in this file, apart from the gym plumbing in StrategicEnv.reset/step).

Two interfaces share one controller:
  * StrategicEnv (gymnasium)                 - for training: reset(options={'scenario_index', 'env_seed'}), step(slot)
  * StrategicPolicy(chooser).act(env) -> int - a PRIMITIVE policy callable usable with s13_baselines.episode() so that every policy
                                              (S18-H A/B, learned C, random D, WAIT) is scored with the identical metric code."""
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
S18_CODE = os.path.join(ROOT, "research", "hierarchical_planner", "code")
for p in (S18_CODE, os.path.join(ROOT, "python")):
    if p not in sys.path:
        sys.path.insert(0, p)
from world_model import WorldModel, next_topup  # noqa: E402  (read-only import of the S18-H code)
from candidates import Candidate, generate  # noqa: E402
from executor import Executor  # noqa: E402
from planner import Weights, plan as heuristic_plan  # noqa: E402
from synthetic_rl import layout as L  # noqa: E402
from synthetic_rl import s13_env as E  # noqa: E402

SC = E.load_scenarios()
TEST_SEEDS = {515838, 539595, 563352, 587109}
VAL_SEEDS = {507919, 579190}


def game_seed(idx: int) -> int:
    return int(SC[idx]["id"].split("_")[0])


def split3(idx: int) -> str:
    gs = game_seed(idx)
    return "test" if gs in TEST_SEEDS else ("validation" if gs in VAL_SEEDS else "train")


TRAIN_IDX = [i for i in range(len(SC)) if split3(i) == "train"]
VAL_IDX = [i for i in range(len(SC)) if split3(i) == "validation"]
TEST_IDX = [i for i in range(len(SC)) if split3(i) == "test"]
TEST_IDS = {SC[i]["id"] for i in TEST_IDX}


def env_seed(idx: int, s: int) -> int:
    """identical env seeds to S13 / S18-H baselines (7000 + 100*idx + s)."""
    return 7000 + 100 * idx + s


K = 24                    # candidate slots (slot K = WAIT)
WAIT_SLOT = K
WAIT_INTERVAL = 5         # primitive decisions per WAIT macro-action (fixed)
K_ALT = 2                 # Dijkstra alternatives per house (S18-H default)
GLOBAL_DIM = 12
CAND_DIM = 16
KINDS = ("CONNECT", "REMOVE", "WAIT")
GLOBAL_NAMES = ["concrete/25", "concrete/total", "decisions_to_topup/140", "topup_amount/30", "t/600", "houses_connected/10", "houses_unconnected/10",
                "max_demand/7", "mean_demand/7", "overcrowded_flag", "functional_edges/60", "score/300"]
CAND_NAMES = ["is_connect", "is_remove", "concrete_cost/10", "route_len/20", "new_edges/20", "reused_active_edges/20", "newly_served_houses/3", "group_demand/7",
              "group_overcrowded", "disruption_flag", "travel_cost_proxy(path_len)/20", "feasible", "affordability_margin/10", "waited/60", "alt_index/2", "reuse_fraction"]
FEATURE_GROUPS = {"cost": [2, 12], "demand": [7, 8], "disruption": [9]}


def candidate_sort_key(c: Candidate):
    """stable GEOMETRIC ordering key (house x, y, goal x, y, route length, first edge, full edge list); kind first; no score involved."""
    if c.kind == "CONNECT":
        h, g = c.targets["house"], c.targets["goal"]
        first = c.adds[0] if c.adds else (0, 0, "E")
        return (0, h[0], h[1], g[0], g[1], c.route_len, first[0], first[1], first[2], tuple(c.adds))
    e = c.removes[0]
    return (1, e[0], e[1], 0, 0, 0, e[0], e[1], e[2], ())


def cand_key(c: Optional[Candidate]) -> Tuple:
    if c is None or c.kind == "WAIT":
        return ("WAIT",)
    return (c.kind, tuple(c.targets.get("house", ())), tuple(c.adds), tuple(c.removes))


def global_features(wm: WorldModel, score: int) -> np.ndarray:
    dt, amt = next_topup(wm.t)
    dt = 140 if dt is None else dt
    demands = [d["demand"] for d in wm.dests if d["group"] is not None] or [0]
    n_h, n_c = len(wm.houses), len(wm.connected)
    n_func = len([e for e in wm.active if e not in wm.stub])
    return np.array([wm.concrete / 25.0, wm.concrete / max(1, wm.total), dt / 140.0, amt / 30.0, wm.t / 600.0, n_c / 10.0, (n_h - n_c) / 10.0,
                     max(demands) / 7.0, float(np.mean(demands)) / 7.0, float(any(d >= 7 for d in demands)), n_func / 60.0, score / 300.0], dtype=np.float32)


def candidate_features(wm: WorldModel, c: Candidate) -> np.ndarray:
    f = np.zeros(CAND_DIM, dtype=np.float32)
    if c.kind == "CONNECT":
        g = c.targets["group"]
        pins, over = wm.demand_of_group(g)
        h = c.targets["house"]
        f[0] = 1.0
        f[2] = c.concrete_cost / 10.0
        f[3] = c.route_len / 20.0
        f[4] = len(c.adds) / 20.0
        f[5] = c.reused_edges / 20.0
        f[6] = c.connectivity_delta / 3.0
        f[7] = pins / 7.0
        f[8] = float(over)
        f[9] = float(c.would_destroy_functional_connection)
        f[10] = (len(c.path) - 1) / 20.0
        f[11] = float(c.feasibility == "feasible" and c.confidence > 0)
        f[12] = float(np.clip((wm.concrete - c.concrete_cost) / 10.0, -2.0, 2.0))
        f[13] = min(60, wm.t - wm.houses[h]["spawn_t"]) / 60.0
        f[14] = c.alt_index / 2.0
        f[15] = c.reused_edges / max(1, c.route_len)
    elif c.kind == "REMOVE":
        f[1] = 1.0
        f[2] = c.concrete_cost / 10.0          # -0.1 (refund)
        f[4] = 0.0
        f[9] = float(c.would_destroy_functional_connection)
        f[11] = 1.0
        f[12] = float(np.clip((wm.concrete + 1) / 10.0, -2.0, 2.0))
    return f


def candidate_legal(c: Candidate) -> bool:
    if c.kind == "CONNECT":
        return c.feasibility == "feasible" and c.confidence > 0 and not c.would_destroy_functional_connection
    if c.kind == "REMOVE":
        return not c.would_destroy_functional_connection
    return False


class CommittedExecutor(Executor):
    """S18-H executor whose plan is set from OUTSIDE (the strategic policy) instead of by the heuristic planner.
    Keeps all S18-H guarantees: legality re-check against the env mask, oscillation ban/protect windows, would-destroy veto, invalidation reasons."""

    def commit(self, cand: Candidate, t: int):
        self.plan = cand
        self.plan_t0 = t
        self.done_adds = set()
        self.stats["plans"] += 1
        self.last_status = None

    def refresh(self, wm: WorldModel) -> Optional[str]:
        """progress / invalidation of the current plan; returns 'completed' | 'abandoned' | None (still running) | 'idle'."""
        self.last_t = wm.t
        self._banned_and_protected(wm.t)
        if self.plan is None:
            return "idle"
        if self.plan.kind == "CONNECT" and self.plan.targets["house"] in wm.connected:
            self._complete()
            return "completed"
        if self.plan.kind == "REMOVE" and self.plan.removes[0] not in wm.edges:
            self._complete()
            return "completed"
        r = self._invalidated(wm)
        if r is not None:
            self._abandon(r)
            return "abandoned"
        if self.plan.kind == "CONNECT" and all(e in self.done_adds for e in self.plan.adds):
            self._abandon("all_edges_built_but_not_connected")
            return "abandoned"
        return None

    def emit(self, wm: WorldModel) -> int:
        """one primitive action for the committed plan (S18-H act() body); may abandon the plan (then returns WAIT)."""
        self.stats["decisions"] += 1
        p = self.plan
        a = 0
        if p is None:
            return 0
        if p.would_destroy_functional_connection:
            self.stats["destroy_vetoes"] += 1
            self._abandon("would_destroy")
            return 0
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
        if not wm.mask[a]:
            self.stats["illegal_suppressed"] += 1
            a = 0
        return int(a)


class StrategicController:
    """decision boundaries + candidate set + feature construction + executor; shared by StrategicEnv and StrategicPolicy."""

    def __init__(self, K: int = K, wait_interval: int = WAIT_INTERVAL, k_alt: int = K_ALT, gamma: float = 0.99, shuffle_rng: Optional[np.random.Generator] = None,
                 feature_ablation: Optional[List[int]] = None):
        self.K, self.wait_interval, self.k_alt, self.gamma = K, wait_interval, k_alt, gamma
        self.shuffle_rng = shuffle_rng          # diagnostics only: permute candidate slots at evaluation
        self.feature_ablation = feature_ablation
        self.reset()

    def reset(self):
        self.ex = CommittedExecutor()
        self.ex.reset_episode()
        self.wait_left = 0
        self.cur: Optional[Dict[str, Any]] = None
        self.records: List[Dict[str, Any]] = []
        self.open: Optional[Dict[str, Any]] = None
        self.truncated_candidates = 0
        self.decisions = 0
        self.illegal_selected = 0
        self._wm_t = -1
        self._wm_id = None
        self._wm = None

    def wm(self, env) -> WorldModel:
        if self._wm is None or self._wm_t != env.t or self._wm_id != id(env):
            self._wm = WorldModel(env)
            self._wm_t, self._wm_id = env.t, id(env)
        return self._wm

    def _close(self, env, status: str):
        if self.open is not None:
            self.open["duration"] = env.t - self.open["t0"]
            self.open["outcome"] = status
            self.records.append(self.open)
            self.open = None

    def needs_decision(self, env) -> bool:
        if self.wait_left > 0:
            return False
        if self.ex.plan is not None:
            st = self.ex.refresh(self.wm(env))
            if self.ex.plan is not None:
                return False
            self._close(env, st)
        elif self.open is not None:
            self._close(env, "wait_done" if self.open["kind"] == "WAIT" else "abandoned")
        return True

    def finish(self, env):
        """close the record that is open at episode end."""
        self._close(env, "episode_end")

    def observe(self, env) -> Dict[str, np.ndarray]:
        wm = self.wm(env)
        banned, protected = self.ex._banned_and_protected(wm.t)
        cands = [c for c in generate(wm, self.k_alt, banned, protected, include_motorway=False) if c.kind != "WAIT"]
        cands.sort(key=candidate_sort_key)
        if len(cands) > self.K:
            self.truncated_candidates += len(cands) - self.K
            cands = cands[: self.K]
        if self.shuffle_rng is not None and len(cands) > 1:
            perm = self.shuffle_rng.permutation(len(cands))
            cands = [cands[i] for i in perm]
        g = global_features(wm, env.score)
        cf = np.zeros((self.K, CAND_DIM), dtype=np.float32)
        mask = np.zeros(self.K + 1, dtype=bool)
        for i, c in enumerate(cands):
            cf[i] = candidate_features(wm, c)
            mask[i] = candidate_legal(c)
        mask[self.K] = True
        if self.feature_ablation:
            cf[:, self.feature_ablation] = 0.0
        self.cur = {"cands": cands, "wm": wm, "banned": banned, "protected": protected, "t": wm.t}
        return {"global": g, "cand": cf, "mask": mask}

    def commit(self, slot: int, env):
        assert self.cur is not None and self.cur["t"] == env.t, "observe() must precede commit() at the same decision"
        self.decisions += 1
        cands = self.cur["cands"]
        if slot == self.K:
            self.wait_left = self.wait_interval
            kind, key, c = "WAIT", ("WAIT",), None
        else:
            c = cands[slot]
            if not candidate_legal(c):
                self.illegal_selected += 1
            self.ex.commit(c, env.t)
            kind, key = c.kind, cand_key(c)
        self.open = {"t0": env.t, "slot": int(slot), "kind": kind, "key": key, "alt": (c.alt_index if c is not None else 0), "cost": (c.concrete_cost if c is not None else 0),
                     "route_len": (c.route_len if c is not None else 0), "n_legal": int(sum(1 for x in cands if candidate_legal(x))) + 1, "n_cands": len(cands) + 1,
                     "house": (tuple(c.targets["house"]) if (c is not None and c.kind == "CONNECT") else None)}

    def primitive(self, env) -> int:
        if self.wait_left > 0:
            self.wait_left -= 1
            return 0
        return self.ex.emit(self.wm(env))

    # ---- summary of strategic behaviour over the episode
    def summary(self) -> Dict[str, Any]:
        recs = self.records
        n = len(recs)
        kinds = {k: sum(1 for r in recs if r["kind"] == k) for k in KINDS}
        connects = [r for r in recs if r["kind"] == "CONNECT"]
        return {"strategic_actions": n, "wait_frequency": (kinds["WAIT"] / n if n else 0.0), "kind_counts": kinds, "mean_duration": (float(np.mean([r["duration"] for r in recs])) if n else 0.0),
                "mean_duration_connect": (float(np.mean([r["duration"] for r in connects])) if connects else 0.0), "alt_selected": sum(1 for r in connects if r["alt"] > 0),
                "abandoned_plans": self.ex.stats["abandoned"], "plans": self.ex.stats["plans"], "completed_plans": self.ex.stats["completed"], "abandon_reasons": dict(self.ex.stats["abandon_reasons"]),
                "illegal_suppressed": self.ex.stats["illegal_suppressed"], "illegal_selected": self.illegal_selected, "truncated_candidates": self.truncated_candidates,
                "waits_for_concrete": self.ex.stats["waits_for_concrete"], "repeated_keys": _repeated(recs)}


def _repeated(recs) -> int:
    seen, rep = set(), 0
    for r in recs:
        if r["kind"] == "WAIT":
            continue
        if r["key"] in seen:
            rep += 1
        seen.add(r["key"])
    return rep


Chooser = Callable[[Dict[str, np.ndarray], StrategicController], int]


class StrategicPolicy:
    """primitive policy adapter: policy(env) -> action, for s13_baselines.episode(). chooser(obs, ctrl) -> slot."""

    def __init__(self, chooser: Chooser, ctrl: Optional[StrategicController] = None):
        self.chooser = chooser
        self.ctrl = ctrl or StrategicController()
        self.last_t = -1

    def act(self, env) -> int:
        if env.t < self.last_t or self.last_t < 0 or env.t == 0:
            self.ctrl.reset()
        self.last_t = env.t
        if self.ctrl.needs_decision(env):
            obs = self.ctrl.observe(env)
            slot = int(self.chooser(obs, self.ctrl))
            self.ctrl.commit(slot, env)
        return self.ctrl.primitive(env)


class StrategicEnv(gym.Env):
    """gymnasium view: obs = dict(global (GLOBAL_DIM,), cand (K, CAND_DIM), mask (K+1,)), action = Discrete(K+1)."""
    metadata = {"render_modes": []}

    def __init__(self, scenario_indices: List[int], seed: int = 0, gamma: float = 0.99, K: int = K, wait_interval: int = WAIT_INTERVAL, k_alt: int = K_ALT):
        super().__init__()
        self.scenario_indices = list(scenario_indices)
        self.rng = np.random.default_rng(seed)
        self.gamma = gamma
        self.env = E.MultiHouseEnv(SC, seed=seed)
        self.ctrl = StrategicController(K, wait_interval, k_alt, gamma)
        self.observation_space = spaces.Dict({"global": spaces.Box(-5, 5, (GLOBAL_DIM,), np.float32), "cand": spaces.Box(-5, 5, (K, CAND_DIM), np.float32), "mask": spaces.MultiBinary(K + 1)})
        self.action_space = spaces.Discrete(K + 1)
        self.scenario_index = None

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        options = options or {}
        idx = options.get("scenario_index", None)
        if idx is None:
            idx = int(self.scenario_indices[self.rng.integers(len(self.scenario_indices))])
        es = options.get("env_seed", None)
        if es is None:
            es = int(self.rng.integers(1 << 30))
        self.scenario_index = idx
        self.env._rng = np.random.default_rng(es)
        self.env.reset(options={"scenario_index": idx})
        self.ctrl.reset()
        assert self.ctrl.needs_decision(self.env)
        return self.ctrl.observe(self.env), {"scenario": SC[idx]["id"], "scenario_index": idx, "env_seed": es, "split": split3(idx)}

    def step(self, slot: int):
        env, ctrl = self.env, self.ctrl
        ctrl.commit(int(slot), env)
        rec = ctrl.open
        r_sum, r_disc, dur, done, term = 0.0, 0.0, 0, False, False
        deliveries = 0
        while True:
            a = ctrl.primitive(env)
            _, r, te, tr, inf = env.step(a)
            r_sum += r
            r_disc += (self.gamma ** dur) * r
            dur += 1
            deliveries += inf["deliveries"]
            if te or tr:
                done, term = True, te
                ctrl.finish(env)
                break
            if ctrl.needs_decision(env):
                break
        info = {"duration": dur, "reward_disc": r_disc, "kind": rec["kind"], "slot": rec["slot"], "deliveries": deliveries, "t": env.t, "game_over": term,
                "outcome": (ctrl.records[-1]["outcome"] if ctrl.records and ctrl.records[-1] is rec else "running")}
        obs = ctrl.observe(env) if not done else None
        return obs, float(r_sum), bool(term), bool(done and not term), info


# ---- reference choosers (A / B heuristics expressed as choosers over the same candidate set; used for agreement diagnostics)
def heuristic_choice(ctrl: StrategicController, weights: Weights) -> Tuple[int, Tuple]:
    """what the S18-H planner would pick on the current state (full candidate set, same ban/protect windows); returns (slot or WAIT_SLOT or -1 if not in the K list, key)."""
    best = heuristic_plan(ctrl.cur["wm"], weights, ctrl.cur["banned"], ctrl.cur["protected"])
    key = cand_key(best)
    if key == ("WAIT",):
        return ctrl.K, key
    for i, c in enumerate(ctrl.cur["cands"]):
        if cand_key(c) == key:
            return i, key
    return -1, key


def random_chooser(rng: np.random.Generator) -> Chooser:
    def ch(obs, ctrl):
        legal = np.flatnonzero(obs["mask"])
        return int(rng.choice(legal))
    return ch


def wait_chooser(obs, ctrl):
    return ctrl.K
