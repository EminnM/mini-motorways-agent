"""S12 native runner: frozen S8-C checkpoint, STOCHASTIC inference, native mask. Conditions:
  control  : action = predict(obs, native mask)                                (conservation guard evaluated and LOGGED, never applied)
  conserve : action = predict(obs, native mask minus cycle-closing extension ADDs)   (same policy distribution conditioned on permitted actions, same RNG stream)
No connection-preservation guard in either condition. Logs edge-level changes (functional / non-functional) for offline reconstruction of the road graph.
   python experiments/s12_runner.py --cond control|conserve --idx 0,1,2     (S9 seeds)      python experiments/s12_runner.py --twin"""
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
import s12_guard as C  # noqa: E402

OUT = os.path.join(ROOT, "research", "s12_concrete_conservation")
CK = os.path.join(ROOT, "research", "s8_long_route_training", "runs", "arm_C_s11", "checkpoints", "step_0099840.zip")
S9SEEDS = json.load(open(os.path.join(ROOT, "research", "s9_native_transfer", "seeds.json")))["seeds"]
E2L = lambda es: [list(e) for e in sorted(es)]


def run_episode(env, model, seed, ep_idx, cond, deadline, max_steps=None, hashes=False, apply_guard=True, instrument=True):
    from synthetic_rl import layout as L
    torch.manual_seed(1000 + ep_idx)
    obs, info = env.reset(seed=seed)
    raw0 = env.last_raw_observation
    header = {"F0": E2L(G.edges_from_raw(raw0)), "P0": E2L(G.edges_present(raw0)) if instrument else None,
              "concrete0": raw0["globals"]["concreteAvailable"], "concrete_total0": raw0["globals"]["concreteTotal"]}
    steps, hs = [], []
    te = tr = False
    i = 0
    cum = 0.0
    cut = False
    last_info = {}
    prev_conc, prev_tot = header["concrete0"], header["concrete_total0"]
    while not (te or tr):
        if time.time() > deadline or (max_steps is not None and i >= max_steps):
            cut = True
            break
        mask = env.action_masks().copy()
        raw_b = env.last_raw_observation
        rec = {"i": i}
        if instrument:
            ctx = G.Ctx(raw_b)
            func_b = G.edges_from_raw(raw_b)
            present_b = G.edges_present(raw_b)
            conn_b = G.connected_house_set(ctx, func_b)
            permitted, rep = C.conservation_permitted(raw_b, mask, L.decode_action, L.UPGRADE_BASE)
            with torch.no_grad():
                p_nat = model.policy.get_distribution(torch.as_tensor(obs[None], dtype=torch.float32), action_masks=mask[None]).distribution.probs[0].numpy()
            cls_n = {k: len(v) for k, v in rep["class_actions"].items()}
            cls_mass = {k: round(float(sum(p_nat[a] for a in v)), 4) for k, v in rep["class_actions"].items()}
            vetoed_mass = float(sum(p_nat[v[0]] for v in rep["vetoed"]))
            rec.update({"cls_n": cls_n, "cls_mass": cls_mass, "n_vetoable": len(rep["vetoed"]), "vetoable_mass": round(vetoed_mass, 4)})
            only_wait_after = bool(rep["vetoed"] and int(permitted.sum()) == 1 and permitted[0])
        use_perm = bool(instrument and cond == "conserve" and apply_guard and rep["vetoed"])
        if use_perm:
            rng_state = torch.get_rng_state()
            a_native = int(model.predict(obs, action_masks=mask, deterministic=False)[0])
            torch.set_rng_state(rng_state)
            a = int(model.predict(obs, action_masks=permitted, deterministic=False)[0])
            rec.update({"modified": bool(a != a_native), "native_choice": a_native, "only_wait_remains": only_wait_after})
        else:
            a = int(model.predict(obs, action_masks=mask, deterministic=False)[0])
        dec = L.decode_action(a)
        obs, r, te, tr, inf = env.step(a)
        raw_a = env.last_raw_observation
        last_info = inf
        if hashes:
            hs.append((a, float(r), hashlib.sha1(obs.tobytes()).hexdigest()))
        if instrument:
            ctx_a = G.Ctx(raw_a)
            func_a = G.edges_from_raw(raw_a)
            present_a = G.edges_present(raw_a)
            conn_a = G.connected_house_set(ctx_a, func_a)
            m2 = env.action_masks()
            kinds = [L.decode_action(int(x))[0] for x in np.flatnonzero(m2) if 0 < x < L.UPGRADE_BASE]
            g = raw_a["globals"]
            rc = inf.get("rewardComponents", {})
            cum += float(rc.get("scoreDelta", 0.0))
            nf_b, nf_a = present_b - func_b, present_a - func_a
            new_struct = {k: (c, ctx_a.group.get(k)) for k, c in ctx_a.content.items() if c in ("House", "Destination", "Carpark") and ctx.content.get(k) != c}
            ex = None
            if dec[0] == "ADD":
                ex = C.classify_add(ctx, func_b, present_b, C.components(func_b), (dec[1], dec[2], dec[3]))["cls"]
            lost = sorted(h for h in conn_b if h in ctx_a.house_set and h not in conn_a)
            rec.update({"kind": dec[0], "action": a, "exec_cls": ex, "n_legal": int(mask.sum()), "n_legal_add_next": kinds.count("ADD"), "n_legal_remove_next": kinds.count("REMOVE"),
                        "concrete_prev": prev_conc, "concrete": g["concreteAvailable"], "concrete_total": g["concreteTotal"], "topup": g["concreteTotal"] - prev_tot,
                        "fa": E2L(func_a - func_b), "fr": E2L(func_b - func_a), "nfa": E2L(nf_a - nf_b), "nfr": E2L(nf_b - nf_a), "n_func": len(func_a), "n_nonfunc": len(nf_a),
                        "houses": len(ctx_a.houses), "connected": len(conn_a), "lost": [list(h) for h in lost], "reward": float(r), "scoreDelta": float(rc.get("scoreDelta", 0.0)), "score_cum": cum,
                        "accepted": inf.get("accepted"), "unconnected": [list(h) for h in ctx_a.houses if h not in conn_a], "connected_houses": [list(h) for h in sorted(conn_a)] if conn_a != conn_b else None,
                        "new_struct": [[k[0], k[1], v[0], v[1]] for k, v in new_struct.items()]})
            prev_conc, prev_tot = g["concreteAvailable"], g["concreteTotal"]
            steps.append(rec)
        i += 1
    end = {"terminated": bool(te), "truncated": bool(tr), "cut": cut, "penalty_missing": bool(te and not tr and float(last_info.get("rewardComponents", {}).get("terminalPenalty", 0.0)) == 0.0),
           "final_applyInfo": last_info.get("applyInfo")}
    return header, steps, hs, end


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", default="control", choices=["control", "conserve"])
    ap.add_argument("--idx", default="0")
    ap.add_argument("--twin", action="store_true")
    ap.add_argument("--deadline-min", type=float, default=4.5)
    a = ap.parse_args()
    torch.set_num_threads(1)
    from sb3_contrib import MaskablePPO
    import mini_motorways_env as M
    from experiments import gamectl
    model = MaskablePPO.load(CK, device="cpu")
    t0 = time.time()
    tag = "s12twin" if a.twin else f"s12{a.cond}_{a.idx.replace(',', '')}"
    with gamectl.GameInstance(tag) as g:
        env = M.MiniMotorwaysEnv(max_decisions=600, decision_delta=1.0, request_path=os.path.join(g.ipc_dir, "request.json"), response_path=os.path.join(g.ipc_dir, "response.json"), start_mode="NORMAL_START")
        if a.twin:
            h1 = run_episode(env, model, S9SEEDS[0], 0, "control", time.time() + 300, 60, True, False, False)[2]
            h2 = run_episode(env, model, S9SEEDS[0], 0, "conserve", time.time() + 300, 60, True, False, True)[2]
            h3 = run_episode(env, model, S9SEEDS[0], 0, "control", time.time() + 300, 60, True, False, False)[2]
            first_div = next((j for j, (x, y) in enumerate(zip(h1, h3)) if x != y), None)
            res = {"steps": len(h1), "plain_vs_instrumented_guard_dry_identical": h1 == h2, "plain_vs_plain_identical": h1 == h3, "plain_vs_plain_first_divergence_step": first_div, "wall_s": round(time.time() - t0, 1)}
            json.dump(res, open(os.path.join(OUT, "twin_noninterference.json"), "w"), indent=1)
            print(json.dumps(res))
            env.close()
            return
        os.makedirs(os.path.join(OUT, "raw"), exist_ok=True)
        deadline = t0 + 60 * a.deadline_min
        for k in [int(x) for x in a.idx.split(",")]:
            seed = S9SEEDS[k]
            header, st, _, end = run_episode(env, model, seed, k, a.cond, deadline)
            with gzip.open(os.path.join(OUT, "raw", f"{a.cond}_{seed}.json.gz"), "wt") as f:
                json.dump({"seed": seed, "idx": k, "cond": a.cond, "header": header, "end": end, "steps": st}, f)
            vet = sum(1 for s in st if s["n_vetoable"] > 0)
            print(f"[{a.cond}] seed {seed}: steps {len(st)} deliveries {st[-1]['score_cum']} connected_end {st[-1]['connected']}/{st[-1]['houses']} steps_with_vetoable {vet} "
                  f"exec_cycle_closing {sum(1 for s in st if s['exec_cls'] == 'cycle_closing')} penalty_missing {end['penalty_missing']} ({time.time() - t0:.0f}s)", flush=True)
        env.close()


if __name__ == "__main__":
    main()
