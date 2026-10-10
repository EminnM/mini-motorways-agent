"""S20 autonomous rollouts (no planner intervention): a policy is a callable (env, goal) -> action index in [0, 3112).
The episode ends when the goal house is functionally connected (success), when the policy emits DONE (premature if not connected), or at the
step budget.  Every emitted action is checked against the env mask (illegal -> counted and replaced by WAIT)."""
import numpy as np
import torch

import common as C
from common import A_DONE, L, ends, goal_obs_u8, legal_mask_ext


class LearnedPolicy:
    """goal-conditioned network policy; mode 'greedy' (masked argmax) or 'sample' (masked softmax sampling, temperature 1)."""

    def __init__(self, model, mode="greedy", seed=0, allow_done=True):
        self.model, self.mode, self.allow_done = model, mode, allow_done   # allow_done=False: diagnostic variant, DONE masked out (termination only by connection / budget)
        self.gen = torch.Generator().manual_seed(seed)
        self.raw_legal = 0
        self.calls = 0

    def logits(self, env, goal):
        tiles, glob = goal_obs_u8(env, goal)
        mask = legal_mask_ext(env)
        if not self.allow_done:
            mask[A_DONE] = False
        with torch.no_grad():
            lo = self.model(torch.from_numpy(tiles)[None], torch.from_numpy(glob)[None])[0]
        return lo, mask

    def __call__(self, env, goal):
        lo, mask = self.logits(env, goal)
        self.calls += 1
        self.raw_legal += int(mask[int(lo.argmax())])
        ml = lo.masked_fill(~torch.from_numpy(mask), -1e9)
        if self.mode == "greedy":
            return int(ml.argmax())
        p = torch.softmax(ml, 0)
        return int(torch.multinomial(p, 1, generator=self.gen))


def planner_policy(env, goal):
    """reference A: the goal-conditioned S18-H executor (a persistent executor per task is created by run_task)."""
    raise RuntimeError("use PlannerPolicy")


class PlannerPolicy:
    def __init__(self):
        self.ex = None
        self.goal = None

    def __call__(self, env, goal):
        if self.ex is None or self.goal != goal:
            self.ex, self.goal = C.GoalExecutor(goal), goal
        if goal in env.connected_houses():
            return A_DONE
        return int(self.ex.act(env))


class RandomLegalPolicy:
    """uniform over legal ADD / REMOVE actions (never WAIT / DONE: a WAIT-including random policy would mostly idle)."""

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def __call__(self, env, goal):
        legal = np.flatnonzero(env.action_masks()[1:L.UPGRADE_BASE]) + 1
        return int(self.rng.choice(legal)) if len(legal) else 0


class S8CPolicy:
    """frozen S8-C (v1 observation, env mask, stochastic as in S13). It receives NO goal: it connects whatever it decides to connect."""

    def __init__(self, model, seed=8008):
        self.model = model
        torch.manual_seed(seed)

    def __call__(self, env, goal):
        return int(self.model.predict(env._obs, action_masks=env.action_masks(), deterministic=False)[0])


def run_task(sc, policy, budget, query_done=False):
    goal = tuple(sc["meta"]["goal"])
    env = C.restore_env(sc["state"])
    others0 = env.connected_houses() - {goal}
    c0 = env.concrete
    steps = illegal = water_adds = removes = house_swaps = 0
    premature_done = False
    while steps < budget:
        if goal in env.connected_houses():
            break
        a = policy(env, goal)
        if a == A_DONE:
            premature_done = True
            break
        if not env.action_masks()[a]:
            illegal += 1
            a = 0
        d = L.decode_action(a)
        if d[0] == "ADD":
            e = (d[1], d[2], d[3])
            pa, pb = ends(e)
            water_adds += int(pa in env.water or pb in env.water)
            house_swaps += int(any(env.content.get(p) == "House" and p != goal for p in (pa, pb)))
        elif d[0] == "REMOVE":
            removes += 1
        env.step(a)
        steps += 1
    success = goal in env.connected_houses()
    done_ok = None
    if success and query_done:
        done_ok = policy(env, goal) == A_DONE
    others1 = env.connected_houses() - {goal}
    return {"success": success, "steps": steps, "concrete_spent": c0 - env.concrete, "destroyed": len(others0 - others1), "illegal": illegal, "premature_done": premature_done,
            "done_at_completion": done_ok, "water_adds": water_adds, "removes": removes, "other_house_swaps": house_swaps, "budget": budget}
