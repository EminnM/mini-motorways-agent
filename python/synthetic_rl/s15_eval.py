"""S15 evaluation: fixed VALIDATION sets only (S7 validation manifests k=7, k=10; S13 validation game seeds 507919 / 579190 -- the four S13 test seeds are never touched).
Identical scenarios, env seeds and torch sampling seeds for every checkpoint of both arms; Arm B and the ablation zero the group channels before prediction."""
import numpy as np
import torch

from . import layout as L
from . import obs_v2 as V
from . import s7, s7_eval
from .s13_env import load_scenarios, ends
from .s14_env import MultiHouseEnvV2
from .s15_env import zero_groups

VAL_SEEDS = [507919, 579190]
EVAL_ENV_SEEDS = 5
TORCH_SEED = 8008
SC_ALL = load_scenarios()
VAL_IDX = [i for i, s in enumerate(SC_ALL) if s["game_seed"] in VAL_SEEDS]
_SETS = None


def s7_sets():
    global _SETS
    if _SETS is None:
        _SETS, _ = s7.build_task_sets()
    return _SETS


class Policy:
    """model + optional group blinding; predict on v2 observations."""

    def __init__(self, model, blind=False):
        self.model, self.blind = model, blind

    def predict(self, obs_v2, action_masks=None, deterministic=True):
        o = zero_groups(obs_v2) if self.blind else obs_v2
        return self.model.predict(o, action_masks=action_masks, deterministic=deterministic)

    def wait_prob(self, obs_v2_batch, masks):
        o = np.stack([zero_groups(x) if self.blind else x for x in obs_v2_batch])
        with torch.no_grad():
            t = torch.as_tensor(o)
            logits = self.model.policy.mlp_extractor.forward_actor(t)
            logits = logits.masked_fill(~torch.as_tensor(np.stack(masks)), -1e9)
            p = torch.softmax(logits, dim=1)[:, 0]
        return float(p.mean())


class V1Adapter:
    def __init__(self, pol: Policy):
        self.pol = pol

    def predict(self, obs_v1, action_masks=None, deterministic=True):
        return self.pol.predict(V.s7_v1_to_v2(np.asarray(obs_v1, dtype=np.float32)), action_masks=action_masks, deterministic=deterministic)


def eval_s7(pol: Policy):
    out = {}
    rows_all = {}
    for k in (7, 10):
        for det in (True, False):
            rows = s7_eval.evaluate("model", s7_sets()["validation"][k], model=V1Adapter(pol), deterministic=det, seed=0)
            s = s7_eval.summarize(rows)[k]
            succ_rows = [r for r in rows if r["success"]]
            tag = f"k{k}_{'det' if det else 'sto'}"
            out[tag] = {"n": len(rows), "success": s["success"], "ci95": s["success_ci95"], "wait_fraction": s["action_share"]["WAIT"], "irrelevant_adds_per_episode": s["mean_irrelevant_adds"],
                        "irrelevant_share_of_adds": s["irrelevant_share_of_adds"], "construction_actions_per_success": (sum(r["n_add"] + r["n_remove"] for r in succ_rows) / len(succ_rows)) if succ_rows else None,
                        "mean_steps": sum(r["steps"] for r in rows) / len(rows)}
            rows_all[tag] = [{"task_id": r["task_id"], "success": r["success"], "steps": r["steps"], "n_add": r["n_add"], "n_wait": r["n_wait"]} for r in rows]
    return out, rows_all


def s13_episode(pol: Policy, idx, env_seed, torch_seed):
    env = MultiHouseEnvV2(SC_ALL, seed=env_seed)
    env._rng = np.random.default_rng(env_seed)
    torch.manual_seed(torch_seed)
    obs, _ = env.reset(options={"scenario_index": idx})
    prev_conn, spawn_t, first_conn = set(), {}, {}
    destroyed = zero_c = wait_disc = disc_steps = plain_adds = 0
    misrouted_steps = 0
    ret, done = 0.0, False
    while not done:
        a = int(pol.predict(obs, action_masks=env.action_masks(), deterministic=False)[0])
        dec = L.decode_action(a)
        legal = bool(env.action_masks()[a])
        unconnected = [h for h in env.houses if h not in prev_conn]
        if unconnected:
            disc_steps += 1
            wait_disc += int(a == 0)
        if legal and dec[0] == "ADD" and not any(env.content.get(p) == "House" for p in ends((dec[1], dec[2], dec[3]))):
            plain_adds += 1
        obs, r, te, tr, info = env.step(a)
        done = te or tr
        ret += r
        conn = env.connected_houses()
        destroyed += len([h for h in prev_conn if h in env.houses and h not in conn])
        for h in env.houses:
            spawn_t.setdefault(h, env.t)
            if h in conn:
                first_conn.setdefault(h, env.t)
        # misrouted: house whose road component reaches a destination but none of its own group
        misrouted_steps += _misrouted(env, conn)
        prev_conn = conn
        zero_c += int(env.concrete == 0)
    n = env.t
    new = [h for h, t0 in spawn_t.items() if t0 > 1 and t0 + 60 < n + 1]
    ok60 = sum(1 for h in new if h in first_conn and first_conn[h] - spawn_t[h] <= 60)
    lat = [first_conn[h] - spawn_t[h] for h in new if h in first_conn]
    return {"scenario": SC_ALL[idx]["id"], "env_seed": env_seed, "return": ret, "deliveries": env.deliveries_total, "decisions": n, "survived": not env.game_over, "houses": len(env.houses), "connected_end": len(prev_conn),
            "new_eligible": len(new), "new_connected_60": ok60, "latency_median": float(np.median(lat)) if lat else None, "frac_zero_concrete": zero_c / n, "destroyed": destroyed, "plain_adds": plain_adds,
            "deliveries_per_plain_add": env.deliveries_total / max(1, plain_adds), "wait_frac_while_disconnected": (wait_disc / disc_steps) if disc_steps else None, "misrouted_house_steps": misrouted_steps}


def _misrouted(env, conn):
    from .s13_env import ends
    parent = {}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for p in env.content:
        parent[p] = p
    for e in env.functional_edges():
        a, b = ends(e)
        parent.setdefault(a, a); parent.setdefault(b, b)
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    for p, c in env.content.items():
        if c in ("Destination", "Carpark"):
            for dx, dy in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                q = (p[0] + dx, p[1] + dy)
                if env.content.get(q) in ("Destination", "Carpark") and (c == "Destination" or env.content[q] == "Destination"):
                    ra, rb = find(p), find(q)
                    if ra != rb:
                        parent[ra] = rb
    comp_dest = {}
    for p, c in env.content.items():
        if c == "Destination":
            comp_dest.setdefault(find(p), set()).add(env.group.get(p))
    n = 0
    for h in env.houses:
        if h in conn:
            continue
        gs = comp_dest.get(find(h))
        if gs and env.houses[h]["group"] not in gs:
            n += 1
    return n


def eval_s13(pol: Policy):
    rows = []
    for idx in VAL_IDX:
        for k in range(EVAL_ENV_SEEDS):
            rows.append(s13_episode(pol, idx, 7000 + 100 * idx + k, TORCH_SEED + 100 * idx + k))
    return summarize_s13(rows), rows


def summarize_s13(rows):
    def med(k):
        v = [r[k] for r in rows if r[k] is not None]
        return round(float(np.median(v)), 3) if v else None
    def mean(k):
        v = [r[k] for r in rows if r[k] is not None]
        return round(float(np.mean(v)), 3) if v else None
    elig = sum(r["new_eligible"] for r in rows)
    return {"episodes": len(rows), "return_mean": mean("return"), "return_median": med("return"), "deliveries_total": sum(r["deliveries"] for r in rows), "deliveries_median": med("deliveries"), "deliveries_mean": mean("deliveries"),
            "decisions_median": med("decisions"), "survived": sum(r["survived"] for r in rows), "new_connected_60": [sum(r["new_connected_60"] for r in rows), elig], "new_rate_60": round(sum(r["new_connected_60"] for r in rows) / max(1, elig), 3),
            "latency_median": med("latency_median"), "destroyed_total": sum(r["destroyed"] for r in rows), "destroyed_median": med("destroyed"), "frac_zero_concrete_median": med("frac_zero_concrete"),
            "wait_frac_while_disconnected_mean": mean("wait_frac_while_disconnected"), "plain_adds_median": med("plain_adds"), "deliveries_per_plain_add_median": med("deliveries_per_plain_add"), "misrouted_house_steps_total": sum(r["misrouted_house_steps"] for r in rows)}


def fixed_s7_observations(n=60):
    """fixed v2 observations (step 0 of S7 validation Gap-7 tasks) and their masks, for the WAIT-probability diagnostic."""
    env = s7.LongRouteEnv(s7_sets()["validation"][7], sampling="sequential")
    obs, masks = [], []
    for i in range(min(n, len(env.tasks))):
        o, _ = env.reset(options={"task_index": i})
        obs.append(V.s7_v1_to_v2(o))
        masks.append(env.action_masks().copy())
    return obs, masks
