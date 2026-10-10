"""S11 native runner: frozen S8-C checkpoint, STOCHASTIC inference, native mask. Two conditions:
  control : action = predict(obs, native mask)                       (the guard is evaluated and LOGGED, never applied)
  guard   : action = predict(obs, native mask minus guard-vetoed ADDs) (same policy distribution conditioned on permitted actions; same RNG stream)
The CNN sees exactly the same observation in both; the guard uses the raw native observation (privileged, evaluation-only).
   python experiments/s11_runner.py --cond control|guard --seeds s9 --idx 0,1,2,3
   python experiments/s11_runner.py --seeds val --idx 0,1,2,3 --cond control        # counterfactual-model validation seeds (not in the S9 list)
   python experiments/s11_runner.py --twin                                          # side-effect check (guard computed but not applied vs plain)"""
import argparse
import gzip
import hashlib
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from experiments import paths  # noqa: E402,F401
import numpy as np  # noqa: E402
import torch  # noqa: E402
import s11_guard as G  # noqa: E402

OUT = os.path.join(ROOT, "research", "s11_connection_guard")
CK = os.path.join(ROOT, "research", "s8_long_route_training", "runs", "arm_C_s11", "checkpoints", "step_0099840.zip")
S9SEEDS = json.load(open(os.path.join(ROOT, "research", "s9_native_transfer", "seeds.json")))["seeds"]
VALSEEDS = [600000 + 7919 * i for i in range(12)]


def run_episode(env, model, seed, ep_idx, cond, deadline, max_steps=None, hashes=False, apply_guard=True, instrument=True):
    from synthetic_rl import layout as L
    torch.manual_seed(1000 + ep_idx)
    obs, info = env.reset(seed=seed)
    steps, hs = [], []
    te = tr = False
    i = 0
    cum = 0.0
    cut = False
    last_info = {}
    while not (te or tr):
        if time.time() > deadline or (max_steps is not None and i >= max_steps):
            cut = True
            break
        mask = env.action_masks().copy()
        raw_b = env.last_raw_observation
        rec = {"i": i}
        veto_rep = None
        if instrument:
            ctx = G.Ctx(raw_b)
            edges_b = G.edges_from_raw(raw_b)
            conn_b = G.connected_house_set(ctx, edges_b)
            permitted, veto_rep = G.guard_permitted(raw_b, mask, L.decode_action, L.UPGRADE_BASE)
        use_perm = bool(instrument and cond == "guard" and apply_guard and veto_rep["vetoed"])
        if use_perm:
            with torch.no_grad():
                p_nat = model.policy.get_distribution(torch.as_tensor(obs[None], dtype=torch.float32), action_masks=mask[None]).distribution.probs[0].numpy()
            rng_state = torch.get_rng_state()
            a_native = int(model.predict(obs, action_masks=mask, deterministic=False)[0])
            torch.set_rng_state(rng_state)
            a = int(model.predict(obs, action_masks=permitted, deterministic=False)[0])
            rec.update({"vetoed_mass": float(sum(p_nat[v[0]] for v in veto_rep["vetoed"])), "modified": bool(a != a_native), "native_choice": a_native})
        else:
            a = int(model.predict(obs, action_masks=mask, deterministic=False)[0])
            if instrument and veto_rep["vetoed"]:           # control: what the guard WOULD have done
                with torch.no_grad():
                    p_nat = model.policy.get_distribution(torch.as_tensor(obs[None], dtype=torch.float32), action_masks=mask[None]).distribution.probs[0].numpy()
                rec.update({"vetoed_mass": float(sum(p_nat[v[0]] for v in veto_rep["vetoed"])), "modified": False})
        dec = L.decode_action(a)
        obs, r, te, tr, inf = env.step(a)
        raw_a = env.last_raw_observation
        last_info = inf
        if hashes:
            hs.append((a, float(r), hashlib.sha1(obs.tobytes()).hexdigest()))
        if instrument:
            ctx_a = G.Ctx(raw_a)
            edges_a = G.edges_from_raw(raw_a)
            conn_a = G.connected_house_set(ctx_a, edges_a)
            leg_a = G.connected_house_set(ctx_a, G.edges_present(raw_a))
            m2 = env.action_masks()
            kinds = [L.decode_action(int(x))[0] for x in np.flatnonzero(m2) if 0 < x < L.UPGRADE_BASE]
            g = raw_a["globals"]
            rc = inf.get("rewardComponents", {})
            cum += float(rc.get("scoreDelta", 0.0))
            lost = sorted(h for h in conn_b if h in ctx_a.house_set and h not in conn_a)
            new_struct = {k for k, c in ctx_a.content.items() if c in ("House", "Destination", "Carpark") and ctx.content.get(k) != c}
            pred = None
            if dec[0] == "ADD":
                e = (dec[1], dec[2], dec[3])
                p = G.predict_add(ctx, edges_b, e)
                exp_after = (edges_b - p["removed"]) | p["added"]
                diff_obs_add = {x for x in (edges_a - edges_b) if not (set(G.ends(x)) & new_struct)}
                diff_obs_rem = {x for x in (edges_b - edges_a) if not (set(G.ends(x)) & new_struct)}
                pa, pr = set(p["added"]), set(p["removed"])
                ok = (diff_obs_add == pa) and (diff_obs_rem == pr)
                pred = {"known": p["known"], "reason": p["reason"], "match": (ok if p["known"] else None), "pred_added": sorted(pa), "pred_removed": sorted(pr), "obs_added": sorted(diff_obs_add), "obs_removed": sorted(diff_obs_rem),
                        "spawn_in_step": bool(new_struct)}
                chk = G.check_candidate(ctx, edges_b, conn_b, e) if conn_b else {"verdict": "allow", "lost": []}
                rec["executed_verdict"] = chk["verdict"]
                rec["executed_pred_lost"] = chk.get("lost", [])
            rec.update({"kind": dec[0], "action": a, "n_legal": int(mask.sum()), "n_legal_add_next": kinds.count("ADD"), "n_legal_remove_next": kinds.count("REMOVE"), "concrete": g["concreteAvailable"],
                        "concrete_total": g["concreteTotal"], "edges": len(edges_a), "added": len(edges_a - edges_b), "removed": len(edges_b - edges_a), "houses": len(ctx_a.houses), "dests": 0,
                        "connected": len(conn_a), "connected_legacy": len(leg_a), "connected_before": len(conn_b), "lost": [list(h) for h in lost], "n_vetoed": len(veto_rep["vetoed"]), "n_unknown": len(veto_rep["unknown"]),
                        "reward": float(r), "scoreDelta": float(rc.get("scoreDelta", 0.0)), "score_cum": cum, "accepted": inf.get("accepted"), "pred": pred,
                        "unconnected": [list(h) for h in ctx_a.houses if h not in conn_a], "connected_houses": [list(h) for h in sorted(conn_a)] if (conn_a != conn_b) else None,
                        "new_houses": [list(k) for k in new_struct if ctx_a.content.get(k) == "House"], "new_dest_tiles": sum(1 for k in new_struct if ctx_a.content.get(k) == "Destination")})
            steps.append(rec)
        i += 1
    info_end = {"terminated": bool(te), "truncated": bool(tr), "cut": cut, "penalty_missing": bool(te and not tr and float(last_info.get("rewardComponents", {}).get("terminalPenalty", 0.0)) == 0.0),
                "final_applyInfo": last_info.get("applyInfo"), "final_reward": float(last_info.get("nativeReward", 0.0))}
    return steps, hs, info_end


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", default="control", choices=["control", "guard"])
    ap.add_argument("--seeds", default="s9", choices=["s9", "val"])
    ap.add_argument("--idx", default="0")
    ap.add_argument("--twin", action="store_true")
    ap.add_argument("--deadline-min", type=float, default=7.0)
    a = ap.parse_args()
    torch.set_num_threads(1)
    from sb3_contrib import MaskablePPO
    import mini_motorways_env as M
    from experiments import gamectl
    model = MaskablePPO.load(CK, device="cpu")
    seeds = S9SEEDS if a.seeds == "s9" else VALSEEDS
    t0 = time.time()
    tag = "s11twin" if a.twin else f"s11{a.cond}{a.seeds}_{a.idx.replace(',', '')}"
    with gamectl.GameInstance(tag) as g:
        env = M.MiniMotorwaysEnv(max_decisions=600, decision_delta=1.0, request_path=os.path.join(g.ipc_dir, "request.json"), response_path=os.path.join(g.ipc_dir, "response.json"), start_mode="NORMAL_START")
        if a.twin:
            h1 = run_episode(env, model, S9SEEDS[0], 0, "control", time.time() + 300, 60, True, False, False)[1]
            h2 = run_episode(env, model, S9SEEDS[0], 0, "guard", time.time() + 300, 60, True, False, True)[1]      # guard + instrumentation computed, NOT applied
            h3 = run_episode(env, model, S9SEEDS[0], 0, "control", time.time() + 300, 60, True, False, False)[1]
            res = {"steps": len(h1), "plain_vs_instrumented_guard_dry_identical": h1 == h2, "plain_vs_plain_identical": h1 == h3, "wall_s": round(time.time() - t0, 1)}
            json.dump(res, open(os.path.join(OUT, "twin_noninterference.json"), "w"), indent=1)
            print(json.dumps(res))
            env.close()
            return
        os.makedirs(os.path.join(OUT, "raw"), exist_ok=True)
        deadline = t0 + 60 * a.deadline_min
        for k in [int(x) for x in a.idx.split(",")]:
            seed = seeds[k]
            st, _, end = run_episode(env, model, seed, k, a.cond, deadline)
            with gzip.open(os.path.join(OUT, "raw", f"{a.seeds}_{a.cond}_{seed}.json.gz"), "wt") as f:
                json.dump({"seed": seed, "idx": k, "cond": a.cond, "end": end, "steps": st}, f)
            deliv = st[-1]["score_cum"]
            print(f"[{a.cond}/{a.seeds}] seed {seed}: steps {len(st)} deliveries {deliv} connected_end {st[-1]['connected']}/{st[-1]['houses']} lost_events {sum(len(s['lost']) for s in st)} vetoes_total {sum(s['n_vetoed'] for s in st)} "
                  f"penalty_missing {end['penalty_missing']} ({time.time() - t0:.0f}s)", flush=True)
        env.close()


if __name__ == "__main__":
    main()
