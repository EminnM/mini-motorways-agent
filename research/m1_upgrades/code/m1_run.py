"""M1 Stage A/B runner: planner B (unchanged road construction) with an upgrade-choice ARM: slot0 (blind), random (seeded), ruleR (PREREGISTERED_RULE.md). Logs every upgrade choice (offered types,
chosen slot/type, applyInfo type, inventory before/after), per-decision connectivity/deliveries, and episode outcomes.   python research/m1_upgrades/code/m1_run.py --arm ruleR --idx 0,...,11"""
import argparse
import gzip
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "s21_native_check", "code"))
from native_adapter import NativePlannerPolicy, ROOT  # noqa: E402
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import numpy as np  # noqa: E402
import s11_guard as G  # noqa: E402
from synthetic_rl import layout as L  # noqa: E402
from executor import make_policy  # noqa: E402
from planner import Weights  # noqa: E402

OUT = os.path.join(ROOT, "research", "m1_upgrades")
SEEDS = json.load(open(os.path.join(ROOT, "research", "s9_native_transfer", "seeds.json")))["seeds"]
INV = ("concrete", "bridges", "motorways", "trafficLights", "roundabouts", "tunnels")


def rank(t, g):
    if t == "Concrete" and g["concreteAvailable"] < 10:
        return 1
    if t == "Bridge" and g["bridgesAvailable"] == 0:
        return 2
    if t in ("DoubleDestination", "Destination", "House"):
        return 3 + ("DoubleDestination", "Destination", "House").index(t) * 0.1
    if t == "Concrete":
        return 4
    if t == "Bridge":
        return 5
    return 6


def choose(arm, offered, g, rng):
    if arm == "slot0" or not offered:
        return 0
    if arm == "random":
        return rng.randrange(len(offered))
    ranks = [rank(o["type"], g) for o in offered]
    return int(np.argmin(ranks))                              # argmin returns the first minimum -> ties go to slot 0


def inventory(g):
    return {k: [g[f"{k}Available"], g[f"{k}Total"]] for k in INV}


def run_episode(env, arm, seed, deadline):
    pol = NativePlannerPolicy(make_policy(Weights()))
    rng = random.Random(seed)
    env.reset(seed=seed)
    pol.reset()
    raw = env.last_raw_observation
    steps, choices = [], []
    te = tr = False
    i, cum = 0, 0.0
    while not (te or tr):
        if time.time() > deadline:
            break
        mask = env.action_masks().copy()
        g = raw["globals"]
        ctx = G.Ctx(raw)
        func = G.edges_from_raw(raw)
        conn = G.connected_house_set(ctx, func)
        rec = {"i": i, "houses": len(ctx.houses), "connected": len(conn), "concrete": g["concreteAvailable"], "pending": g.get("pendingUpgradeChoices", 0)}
        if not mask[0] and g.get("pendingUpgradeChoices", 0) > 0:
            offered = g.get("pendingUpgradeTypes", [])
            slot = choose(arm, offered, g, rng)
            a = L.UPGRADE_BASE + slot
            if not mask[a]:
                a = int(np.flatnonzero(mask)[0])
            ch = {"i": i, "offered": offered, "slot": slot, "chosen_type_export": offered[slot]["type"] if slot < len(offered) else None, "inventory_before": inventory(g), "concrete_before": g["concreteAvailable"]}
            pol.t += 1                                            # keep the adapter's decision clock aligned (the planner is bypassed on this decision; without this the planner's clock lags one decision per choice)
            pol.forced_actions.append((i, int(a)))
        else:
            a = pol.act(raw, mask)
            ch = None
        _, r, te, tr, inf = env.step(a)
        raw = env.last_raw_observation
        rc = inf.get("rewardComponents", {})
        cum += float(rc.get("scoreDelta", 0.0))
        rec.update({"action": int(a), "kind": L.decode_action(int(a))[0], "legal": bool(mask[a]), "scoreDelta": float(rc.get("scoreDelta", 0.0)), "score_cum": cum, "new_houses": [list(h) for h in G.Ctx(raw).houses if h not in ctx.house_set]})
        if ch:
            info = inf.get("applyInfo", "") or ""
            ch["apply_info"] = info[:160]
            ch["chosen_type_apply"] = info.split("type=")[1].split(")")[0] if "type=" in info else None
            ch["inventory_after"] = inventory(raw["globals"])
            choices.append(ch)
        steps.append(rec)
        i += 1
    return {"arm": arm, "seed": seed, "steps": steps, "choices": choices, "end": {"terminated": bool(te), "truncated": bool(tr), "cut": not (te or tr), "exec_stats": dict(pol.ex.stats)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["slot0", "random", "ruleR"])
    ap.add_argument("--idx", default=",".join(str(i) for i in range(12)))
    ap.add_argument("--deadline-min", type=float, default=12.0)
    a = ap.parse_args()
    import mini_motorways_env as M
    from experiments import gamectl
    os.makedirs(os.path.join(OUT, "raw"), exist_ok=True)
    t0 = time.time()
    with gamectl.GameInstance(f"m1_{a.arm}") as g:
        env = M.MiniMotorwaysEnv(max_decisions=600, decision_delta=1.0, request_path=os.path.join(g.ipc_dir, "request.json"), response_path=os.path.join(g.ipc_dir, "response.json"), start_mode="NORMAL_START")
        for k in [int(x) for x in a.idx.split(",")]:
            seed = SEEDS[k]
            rec = run_episode(env, a.arm, seed, t0 + 60 * a.deadline_min)
            with gzip.open(os.path.join(OUT, "raw", f"{a.arm}_{seed}.json.gz"), "wt") as f:
                json.dump(rec, f)
            st = rec["steps"]
            print(f"[{a.arm}] seed {seed}: steps {len(st)} deliveries {st[-1]['score_cum']} conn {st[-1]['connected']}/{st[-1]['houses']} choices {[(c['i'], [o['type'] for o in c['offered']], c['slot'], c['chosen_type_apply']) for c in rec['choices']]} ({time.time() - t0:.0f}s)", flush=True)
        env.close()


if __name__ == "__main__":
    main()
