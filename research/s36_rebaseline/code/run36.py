"""S36 runner (postfix_v8): planner B (historical), planner B+ (serve_unreached_dests=True) and frozen historical hybrid checkpoints (which call the ORIGINAL planner through PLAN),
on three benchmarks with the same env, stepping (decision_delta 1.0) and upgrade handling (lowest legal slot on forced interrupts):
  --bench dev   : 12 designated seeds 500000+7919*i, horizon 600
  --bench fresh : 8 seeds 900000+7919*i, horizon 1500
  --bench new   : 12 predeclared seeds 1100003+7919*i (unused before S36), horizon 1500
  --policy plannerB | plannerBplus | ckpt:<path>   [--rollout r for stochastic checkpoints: torch seed 36000 + 1000*r + seed_index]
Per episode: deliveries, length, terminated, illegal, boundary violations (every executed ADD vs playableAreaAtTime), destroyed functional connections (house connected before the step,
still present, not connected after), kinds, concrete/bridges at end and every 25 decisions, unreached destinations (count + total demand) every 25 decisions and at the end,
houses/connected every 25 decisions, destination demand state at the end, executor stats (planners).  Output: eval/<bench>/<tag>_r<r>_<seed>.json.gz
   python research/s36_rebaseline/code/run36.py --bench new --policy plannerBplus --idx 0,1,2"""
import argparse
import gzip
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
for p in (HERE, ROOT, os.path.join(ROOT, "python"), os.path.join(ROOT, "experiments"), os.path.join(ROOT, "research", "s21_native_check", "code"), os.path.join(ROOT, "research", "s31_hybrid_native_rl", "code")):
    sys.path.insert(0, p)
import numpy as np  # noqa: E402
from native_adapter import NativePlannerPolicy, NativeStateView  # noqa: E402
from executor import make_policy  # noqa: E402
from planner import Weights  # noqa: E402
from world_model import WorldModel  # noqa: E402
from synthetic_rl import layout as L  # noqa: E402

SEEDS = {"dev": [500000 + 7919 * i for i in range(12)], "fresh": [900000 + 7919 * i for i in range(8)], "new": [1100003 + 7919 * i for i in range(12)]}
HORIZON = {"dev": 600, "fresh": 1500, "new": 1500}
DLL = r"C:\Program Files (x86)\Steam\steamapps\common\Mini Motorways\Bootstrap\MiniMotorwaysBootstrap.dll"
V8 = "2aa42367cbb14babce31a6063038f50be9b3535bd7bc23d8dbbdbb65533aa00a"


def far(k):
    return (k[1] + 1, k[2]) if k[3] == "E" else (k[1], k[2] - 1)


def inside(p, ar):
    return ar[0] <= p[0] <= ar[2] and ar[1] <= p[1] <= ar[3]


def dest_state(raw):
    out = []
    for t in raw["tiles"]:
        if t.get("contentType") == "Destination" and t.get("destination"):
            d = t["destination"]
            out.append({"coord": t["coord"], "group": t.get("groupIndex"), "demand": d.get("totalDemand", 0), "over": bool(d.get("isOvercrowding"))})
    return out


def unreached(wm):
    ds = wm.unreached_dests()
    groups_with_houses = {v["group"] for v in wm.houses.values()}
    return {"n": len(ds), "demand": sum(d["demand"] for d in ds), "tiles": [d["tiles"][0] for d in ds],
            "n_with_house_of_colour": sum(1 for d in ds if d["group"] in groups_with_houses),           # serviceable in principle (a route may still be missing)
            "n_no_house_of_colour": sum(1 for d in ds if d["group"] not in groups_with_houses)}          # nothing any road action can do


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, choices=list(SEEDS))
    ap.add_argument("--policy", required=True)
    ap.add_argument("--rollout", type=int, default=0)
    ap.add_argument("--idx", default=None)
    a = ap.parse_args()
    assert hashlib.sha256(open(DLL, "rb").read()).hexdigest() == V8, "deployed DLL is not postfix_v8"
    import mini_motorways_env as M
    from experiments import gamectl
    seeds = SEEDS[a.bench]
    idx = [int(x) for x in a.idx.split(",")] if a.idx else list(range(len(seeds)))
    out_dir = os.path.join(ROOT, "research", "s36_rebaseline", "eval", a.bench)
    os.makedirs(out_dir, exist_ok=True)
    model = None
    if a.policy.startswith("ckpt:"):
        import torch
        from sb3_contrib import MaskablePPO
        from hybrid_policy import HybridMaskablePolicy  # noqa: F401
        torch.set_num_threads(1)
        p = os.path.normpath(a.policy[5:])
        model = MaskablePPO.load(p, device="cpu")
        run = os.path.basename(os.path.dirname(os.path.dirname(p)))
        tag = f"hyb_{run}__{os.path.basename(p).replace('.zip', '')}_sto"
    else:
        tag = a.policy
    weights = {"plannerB": Weights(), "plannerBplus": Weights(serve_unreached_dests=True), "plannerBbypass": Weights(serve_unreached_dests=True, bypass=True)}.get(a.policy)
    t0 = time.time()
    name = f"s36_{a.bench}_{tag}_r{a.rollout}_{''.join(map(str, idx))}"[:48]
    if model is None:
        g = gamectl.GameInstance(name).start()
        env = M.MiniMotorwaysEnv(max_decisions=HORIZON[a.bench], decision_delta=1.0, request_path=os.path.join(g.ipc_dir, "request.json"), response_path=os.path.join(g.ipc_dir, "response.json"), start_mode="NORMAL_START")
    else:
        from hybrid_env import HybridNativeEnv                     # launches its own instance; same env/adapter/upgrade handling underneath
        henv = HybridNativeEnv(seeds, name, max_decisions=HORIZON[a.bench], log_fn=None)
    try:
        for k in idx:
            seed = seeds[k]
            if model is None:
                pol = NativePlannerPolicy(make_policy(weights))
                bypasses = []                                   # every bypass plan the planner adopts (any colour): house, group, drive -> shortest, cost
                env.reset(seed=seed)
                pol.reset()
                raw = env.last_raw_observation
            else:
                import torch
                torch.manual_seed(36000 + 1000 * a.rollout + k)
                obs, _ = henv.reset(options={"seed": seed})
                raw = henv.raw
            assert raw["globals"].get("playableAreaAtTime") is not None, "environment-version mismatch"
            score, t, illegal, bviol, destroyed = 0.0, 0, 0, 0, 0
            kinds = {"WAIT": 0, "ADD": 0, "REMOVE": 0, "UPG": 0, "PLAN": 0, "EDIT": 0}
            timeline = []
            spawn_t = {}
            te = tr = False
            prev_conn = None
            while not (te or tr):
                if model is None:
                    mask = env.action_masks().copy()
                    act = pol.act(raw, mask)
                    if not mask[act]:
                        illegal += 1
                        act = int(np.flatnonzero(mask)[0])
                    native = act
                    view = pol.last_view
                    pl = pol.ex.plan
                    if pl is not None and pl.kind == "CONNECT" and pl.targets.get("bypass") and (not bypasses or bypasses[-1]["adds"] != [list(e) for e in pl.adds]):
                        bypasses.append({"t": t, "house": list(pl.targets["house"]), "group": pl.targets["group"], "drive": pl.targets["drive"], "shortest": pl.targets["shortest"], "cost": pl.concrete_cost, "adds": [list(e) for e in pl.adds]})
                else:
                    m = henv.action_masks()
                    act, _ = model.predict(obs, action_masks=m, deterministic=False)
                    act = int(act)
                    view = NativeStateView(raw, m[:L.N_ACTIONS], t, spawn_t)
                wm = WorldModel(view)
                conn_before = set(wm.connected)
                kd = None
                if model is None:
                    kd = L.decode_action(native)
                    kinds["UPG" if kd[0] == "UPGRADE" else kd[0]] += 1
                    _, r, te, tr, info = env.step(native)
                    raw = env.last_raw_observation
                    score += float(info["rewardComponents"].get("scoreDelta", 0.0))
                else:
                    obs, r, te, tr, info = henv.step(act)
                    raw = henv.raw
                    native = info["native_action"]
                    kd = L.decode_action(native)
                    kinds[info["kind"]] += 1
                    score += float(info["scoreDelta"])
                    if info["kind"] == "EDIT":
                        kinds[kd[0]] += 1
                if kd[0] == "ADD":
                    ar = wm_area = raw["globals"].get("playableAreaAtTime")
                    if ar and not (inside((kd[1], kd[2]), ar) and inside(far(kd), ar)):
                        bviol += 1
                t += 1
                view_after = pol.last_view if model is None else NativeStateView(raw, (env.action_masks() if model is None else henv.action_masks()[:L.N_ACTIONS]), t, spawn_t)
                if model is None:
                    view_after = NativeStateView(raw, env.action_masks(), pol.t, pol.spawn_t)
                wm_after = WorldModel(view_after)
                destroyed += sum(1 for h in conn_before if h in wm_after.houses and h not in wm_after.connected)
                if t % 25 == 0 or te or tr:
                    gg = raw["globals"]
                    u = unreached(wm_after)
                    dem = {}
                    for d in wm_after.dests:
                        dem[str(d["group"])] = max(dem.get(str(d["group"]), 0), d["demand"])
                    timeline.append({"t": t, "score": score, "concrete": gg["concreteAvailable"], "bridges": gg.get("bridgesAvailable"), "houses": len(wm_after.houses), "connected": len(wm_after.connected), "unreached_dests": u["n"], "unreached_demand": u["demand"], "active_edges": len(wm_after.active), "max_demand_by_colour": dem})
            gg = raw["globals"]
            u = unreached(wm_after)
            ep = {"bench": a.bench, "policy": tag, "rollout": a.rollout, "seed": seed, "env_version": "postfix_v8", "horizon": HORIZON[a.bench], "t": t, "score": score, "terminated": bool(te), "illegal": illegal if model is None else henv.ep["illegal"], "boundary_violations": bviol, "destroyed": destroyed,
                  "kinds": kinds, "concrete_end": gg["concreteAvailable"], "concrete_total_end": gg["concreteTotal"], "bridges_end": gg.get("bridgesAvailable"), "houses_end": len(wm_after.houses), "connected_end": len(wm_after.connected), "unreached_end": u, "dest_state_end": dest_state(raw),
                  "exec_stats": ({k_: v for k_, v in pol.ex.stats.items() if k_ != "compute_s"} if model is None else None), "timeline": timeline, "bypasses": (bypasses if model is None else None)}
            with gzip.open(os.path.join(out_dir, f"{tag}_r{a.rollout}_{seed}.json.gz"), "wt") as f:
                json.dump(ep, f)
            print(f"[{a.bench} {tag} r{a.rollout}] seed {seed}: t {t} score {score:.0f} term {te} illegal {ep['illegal']} bviol {bviol} destroyed {destroyed} conn {ep['connected_end']}/{ep['houses_end']} unreached {u['n']} (demand {u['demand']}) concrete {ep['concrete_end']} ({time.time() - t0:.0f}s)", flush=True)
    finally:
        if model is None:
            try:
                env.close()
            finally:
                g.stop()
        else:
            henv.close()


if __name__ == "__main__":
    main()
