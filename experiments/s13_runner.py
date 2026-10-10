"""S13a native noise-floor replicate: frozen S8-C stochastic policy, native mask, NO guard. Same 12 S9 seeds, but an independently chosen torch sampling seed (6000 + i instead of 1000 + i).
Read-only extra logging for the S13b grounding: initial structures, per-destination demand / overcrowding, legal-ADD window extent, week / day clock, vehicles.
   python experiments/s13_runner.py --idx 0,1"""
import argparse
import gzip
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

OUT = os.path.join(ROOT, "research", "s13_multi_house")
CK = os.path.join(ROOT, "research", "s8_long_route_training", "runs", "arm_C_s11", "checkpoints", "step_0099840.zip")
S9SEEDS = json.load(open(os.path.join(ROOT, "research", "s9_native_transfer", "seeds.json")))["seeds"]
SAMPLING_SEED_BASE = 6000
E2L = lambda es: [list(e) for e in sorted(es)]


def dest_info(raw):
    tiles = {(t["coord"][0], t["coord"][1]): t for t in raw["tiles"] if t["contentType"] == "Destination"}
    seen, out = set(), []
    for k in tiles:
        if k in seen:
            continue
        comp, st = [], [k]
        seen.add(k)
        while st:
            u = st.pop()
            comp.append(u)
            for dx, dy in G.DIRS.values():
                v = (u[0] + dx, u[1] + dy)
                if v in tiles and v not in seen:
                    seen.add(v)
                    st.append(v)
        d = [tiles[c].get("destination") or {} for c in comp]
        out.append([min(comp)[0], min(comp)[1], tiles[comp[0]].get("groupIndex"), max(x.get("totalDemand", 0) for x in d), int(any(x.get("isOvercrowding") for x in d))])
    return out


def run_episode(env, model, seed, sampling_seed, deadline):
    from synthetic_rl import layout as L
    torch.manual_seed(sampling_seed)
    obs, info = env.reset(seed=seed)
    raw0 = env.last_raw_observation
    ctx0 = G.Ctx(raw0)
    header = {"F0": E2L(G.edges_from_raw(raw0)), "structures0": [[k[0], k[1], c, ctx0.group.get(k)] for k, c in ctx0.content.items() if c in ("House", "Destination", "Carpark")],
              "dest0": dest_info(raw0), "concrete0": raw0["globals"]["concreteAvailable"], "concrete_total0": raw0["globals"]["concreteTotal"], "globals0": raw0["globals"], "sampling_seed": sampling_seed, "game_seed": seed}
    steps = []
    te = tr = False
    i = 0
    cum = 0.0
    cut = False
    last_info = {}
    prev_conc, prev_tot = header["concrete0"], header["concrete_total0"]
    while not (te or tr):
        if time.time() > deadline:
            cut = True
            break
        mask = env.action_masks().copy()
        raw_b = env.last_raw_observation
        ctx = G.Ctx(raw_b)
        func_b = G.edges_from_raw(raw_b)
        present_b = G.edges_present(raw_b)
        conn_b = G.connected_house_set(ctx, func_b)
        a = int(model.predict(obs, action_masks=mask, deterministic=False)[0])
        dec = L.decode_action(a)
        # legal non-house ADD window extent (before the action)
        xs, ys, nadd = [], [], 0
        for x in np.flatnonzero(mask):
            if 0 < x < L.UPGRADE_BASE:
                k = L.decode_action(int(x))
                if k[0] == "ADD":
                    nadd += 1
                    for p in (G.ends((k[1], k[2], k[3]))):
                        xs.append(p[0])
                        ys.append(p[1])
        win = [min(xs), max(xs), min(ys), max(ys)] if xs else None
        obs, r, te, tr, inf = env.step(a)
        raw_a = env.last_raw_observation
        last_info = inf
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
        ex = C.classify_add(ctx, func_b, present_b, C.components(func_b), (dec[1], dec[2], dec[3]))["cls"] if dec[0] == "ADD" else None
        lost = sorted(h for h in conn_b if h in ctx_a.house_set and h not in conn_a)
        steps.append({"i": i, "kind": dec[0], "action": a, "exec_cls": ex, "n_legal": int(mask.sum()), "n_legal_add_before": nadd, "window_before": win, "n_legal_add_next": kinds.count("ADD"), "n_legal_remove_next": kinds.count("REMOVE"),
                      "concrete_prev": prev_conc, "concrete": g["concreteAvailable"], "concrete_total": g["concreteTotal"], "topup": g["concreteTotal"] - prev_tot, "hour": g["hour"], "day": g["day"], "week": g["week"],
                      "fa": E2L(func_a - func_b), "fr": E2L(func_b - func_a), "nfa": E2L(nf_a - nf_b), "nfr": E2L(nf_b - nf_a), "n_func": len(func_a), "n_nonfunc": len(nf_a), "houses": len(ctx_a.houses),
                      "connected": len(conn_a), "lost": [list(h) for h in lost], "reward": float(r), "scoreDelta": float(rc.get("scoreDelta", 0.0)), "score_cum": cum, "unconnected": [list(h) for h in ctx_a.houses if h not in conn_a],
                      "connected_houses": [list(h) for h in sorted(conn_a)] if conn_a != conn_b else None, "new_struct": [[k[0], k[1], v[0], v[1]] for k, v in new_struct.items()], "dests": dest_info(raw_a),
                      "vehicles": sum(t.get("visibleVehicleCount", 0) for t in raw_a["tiles"]), "pending_upgrades": g.get("pendingUpgradeChoices", 0), "house_groups": {f"{h[0]},{h[1]}": ctx_a.group.get(h) for h in ctx_a.houses}})
        prev_conc, prev_tot = g["concreteAvailable"], g["concreteTotal"]
        i += 1
    end = {"terminated": bool(te), "truncated": bool(tr), "cut": cut, "penalty_missing": bool(te and not tr and float(last_info.get("rewardComponents", {}).get("terminalPenalty", 0.0)) == 0.0)}
    return header, steps, end


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idx", default="0")
    ap.add_argument("--deadline-min", type=float, default=4.2)
    a = ap.parse_args()
    torch.set_num_threads(1)
    from sb3_contrib import MaskablePPO
    import mini_motorways_env as M
    from experiments import gamectl
    model = MaskablePPO.load(CK, device="cpu")
    t0 = time.time()
    with gamectl.GameInstance(f"s13rep_{a.idx.replace(',', '')}") as g:
        env = M.MiniMotorwaysEnv(max_decisions=600, decision_delta=1.0, request_path=os.path.join(g.ipc_dir, "request.json"), response_path=os.path.join(g.ipc_dir, "response.json"), start_mode="NORMAL_START")
        os.makedirs(os.path.join(OUT, "raw"), exist_ok=True)
        for k in [int(x) for x in a.idx.split(",")]:
            seed = S9SEEDS[k]
            header, st, end = run_episode(env, model, seed, SAMPLING_SEED_BASE + k, t0 + 60 * a.deadline_min)
            with gzip.open(os.path.join(OUT, "raw", f"replicate_{seed}.json.gz"), "wt") as f:
                json.dump({"seed": seed, "idx": k, "header": header, "end": end, "steps": st}, f)
            print(f"[replicate] seed {seed} sampling_seed {SAMPLING_SEED_BASE + k}: steps {len(st)} deliveries {st[-1]['score_cum']} connected_end {st[-1]['connected']}/{st[-1]['houses']} ({time.time() - t0:.0f}s)", flush=True)
        env.close()


if __name__ == "__main__":
    main()
