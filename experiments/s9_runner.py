"""S9 native zero-shot transfer: inference only. One process = one policy condition on the fixed seed list, one game instance.
   python experiments/s9_runner.py --who p1det|p1sto|p2det|p2sto [--deadline-min 25]
   python experiments/s9_runner.py --twin            # diagnostics non-interference check (60 decisions, plain vs instrumented)
No gradient step, no optimizer, no mask override, no action filtering: the action is exactly model.predict(obs, native mask).
Everything beyond that (unmasked logits, road signatures, remaining-distance oracle, edge sets) is READ-ONLY measurement."""
import argparse
import gzip
import hashlib
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from experiments import paths  # noqa: E402,F401
import numpy as np  # noqa: E402
import torch  # noqa: E402

OUT = os.path.join(ROOT, "research", "s9_native_transfer")
SEEDS = json.load(open(os.path.join(OUT, "seeds.json")))["seeds"]
CK = {"p1": os.path.join(ROOT, "runs", "2026-10-08_140406_syn_s1_cnn_s13", "checkpoints", "final.zip"),
      "p2": os.path.join(ROOT, "research", "s8_long_route_training", "runs", "arm_C_s11", "checkpoints", "step_0099840.zip")}


def roads_of(raw):
    return {(t["coord"][0], t["coord"][1], d) for t in raw["tiles"] for d in "ES" if t["road"][d] != "None"}


def houses_of(raw):
    return [(t["coord"][0], t["coord"][1]) for t in raw["tiles"] if t["contentType"] == "House"]


def run_episode(env, model, deterministic, seed, ep_idx, deadline, max_steps=None, keep_hashes=False, instrument=True):
    import networkx as nx
    from synthetic_rl import layout as L
    from synthetic_rl import s7_oracle as O
    from experiments.anticycle import road_signature, revisit_stats
    from experiments.evaluation import policy_entropy
    from experiments.recording import EpisodeAccumulator
    meta = {"initialization": "normal", "start_gap": 0, "reward_mode": "native_score", "decision_interval_seconds": 1.0, "diagnostics": instrument}
    acc = EpisodeAccumulator(meta)
    t0 = time.time()
    torch.manual_seed(1000 + ep_idx)                       # common sampling seed per episode index (both policies)
    obs, info = env.reset(seed=seed)
    acc.reset_episode(info.get("seed", seed), None, env.last_raw_observation if instrument else None)
    steps, sigs, hashes = [], [], []
    st = {"first_house": None, "carpark": None, "k_req": None, "initial_edges": None, "rem_min": None, "conn_done": False, "n_add_before_conn": None, "path_new": None, "irrel_before_conn": None}
    n_add = n_rm = n_swap = n_noop_add = n_water_add = n_dest_add = 0
    p_ill, top_ill = [], []
    house_counts = []
    te = tr = False
    cut = False
    i = 0
    while not (te or tr):
        if time.time() > deadline or (max_steps is not None and i >= max_steps):
            cut = True
            break
        mask = env.action_masks().copy()
        raw_b = env.last_raw_observation
        rb = roads_of(raw_b) if instrument else None
        if instrument:
            hs = houses_of(raw_b)
            house_counts.append(len(hs))
            sig = road_signature(raw_b)
            sigs.append(sig)
            if st["first_house"] is None and hs:
                st["first_house"] = hs[0]
                cps = [(t["coord"][0], t["coord"][1]) for t in raw_b["tiles"] if t["contentType"] == "Carpark"]
                rset = rb
                tmap0 = {(t["coord"][0], t["coord"][1]): t for t in raw_b["tiles"]}
                st["carpark"] = next((c for c in cps if any(v != "None" for v in tmap0[c]["road"].values())), cps[0] if cps else None)      # the carpark tile that carries the road stub
                st["initial_edges"] = set(rb)
            rem = None
            if st["first_house"] is not None and st["carpark"] is not None:
                G = O._graph(obs, mask)
                fh, cp = tuple(st["first_house"]), tuple(st["carpark"])
                try:
                    rem = int(nx.shortest_path_length(G, fh, cp, weight="w")) if (fh in G and cp in G) else None
                except nx.NetworkXNoPath:
                    rem = None
                if st["k_req"] is None and rem is not None:
                    st["k_req"] = rem
                if rem is not None:
                    st["rem_min"] = rem if st["rem_min"] is None else min(st["rem_min"], rem)
            with torch.no_grad():
                pol = model.policy
                lg = pol.action_net(pol.mlp_extractor(pol.extract_features(torch.as_tensor(obs[None], dtype=torch.float32)))[0])[0]
                pr = torch.softmax(lg, dim=0).numpy()
            p_ill.append(float(pr[~mask].sum()))
            top_ill.append(bool(not mask[int(np.argmax(lg.numpy()))]))
            ent = policy_entropy(model, obs, mask)
        a, _ = model.predict(obs, action_masks=mask, deterministic=deterministic)
        a = int(a)
        dec = L.decode_action(a)
        legal_sel = bool(mask[a])
        obs, r, te, tr, inf = env.step(a)
        raw_a = env.last_raw_observation
        if keep_hashes:
            hashes.append((a, float(r), hashlib.sha1(obs.tobytes()).hexdigest()))
        added = removed = 0
        if instrument:
            ra = roads_of(raw_a)
            added, removed = len(ra - rb), len(rb - ra)
            acc.add_step(a, legal_sel, r, inf, te, tr, raw_after=raw_a, mask_before=mask, policy_entropy=ent)
            if dec[0] == "ADD":
                n_add += 1
                tm = {(t["coord"][0], t["coord"][1]): t for t in raw_b["tiles"]}
                e = (dec[1], dec[2], dec[3])
                ends = [(e[0], e[1]), L.neighbor(*e)]
                if any(tm[p]["overWater"] for p in ends if p in tm):
                    n_water_add += 1
                if any(tm[p]["contentType"] in ("Destination", "Carpark") for p in ends if p in tm):
                    n_dest_add += 1
                touches_house = any(tm[p]["contentType"] == "House" for p in ends if p in tm)
                if touches_house and removed >= 1:
                    n_swap += 1
                if added == 0 and removed == 0:
                    n_noop_add += 1
            elif dec[0] == "REMOVE":
                n_rm += 1
            if not st["conn_done"] and acc.diag.phi >= 1:
                st["conn_done"] = True
                st["n_add_before_conn"] = n_add
                try:
                    G = nx.Graph()
                    for (x, y, d) in ra:
                        G.add_edge((x, y), L.neighbor(x, y, d))
                    path = nx.shortest_path(G, tuple(st["first_house"]), tuple(st["carpark"])) if st["first_house"] is not None and tuple(st["first_house"]) in G and tuple(st["carpark"]) in G else None
                    if path:
                        edges = [(min(a_[0], b_[0]), a_[1], "E") if a_[1] == b_[1] else (a_[0], max(a_[1], b_[1]), "S") for a_, b_ in zip(path[:-1], path[1:])]
                        st["path_new"] = sum(1 for e_ in edges if e_ not in st["initial_edges"])
                        st["irrel_before_conn"] = max(n_add - st["path_new"], 0)
                except Exception:
                    pass
            rc = inf.get("rewardComponents", {})
            steps.append([i, a, dec[0], dec[1] if len(dec) > 1 else None, dec[2] if len(dec) > 2 else None, dec[3] if len(dec) > 3 else None, int(mask.sum()), float(r),
                          float(rc.get("scoreDelta", 0.0)), inf.get("accepted"), added, removed, len(houses_of(raw_a)), acc.diag.phi, rem, round(p_ill[-1], 4), top_ill[-1]])
        i += 1
    if instrument:
        sigs.append(road_signature(env.last_raw_observation))
        row = acc.row(ep_idx + 1, None)
        row.pop("action_index_counts", None)
        legal = [s[6] for s in steps]
        row.update({"seed": seed, "ep_idx": ep_idx, "deterministic": deterministic, "cut_by_wallclock_or_steplimit": cut, "n_steps": i, "first_house": st["first_house"], "carpark": st["carpark"],
                    "k_required_native_mask_oracle": st["k_req"], "remaining_min_reached": st["rem_min"], "n_add": n_add, "n_remove": n_rm, "n_house_swaps": n_swap, "n_noop_adds": n_noop_add,
                    "n_water_endpoint_adds": n_water_add, "n_dest_adjacent_adds": n_dest_add, "mean_unmasked_mass_on_illegal": float(np.mean(p_ill)) if p_ill else None,
                    "frac_steps_unmasked_top1_illegal": float(np.mean(top_ill)) if top_ill else None, "n_houses_max": max(house_counts) if house_counts else 0,
                    "house_count_first_increase_step": next((j for j in range(1, len(house_counts)) if house_counts[j] > house_counts[1 if len(house_counts) > 1 else 0]), None),
                    "legal_mean": float(np.mean(legal)) if legal else None, "legal_min": min(legal) if legal else None, "legal_max": max(legal) if legal else None,
                    "legal_series_every10": legal[::10], "road_state_stats": revisit_stats(sigs), "n_add_before_first_connection": st["n_add_before_conn"],
                    "path_new_edges_at_first_connection": st["path_new"], "irrelevant_adds_before_first_connection": st["irrel_before_conn"], "wall_s": round(time.time() - t0, 1)})
        return row, steps, hashes
    return None, steps, hashes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--who")
    ap.add_argument("--twin", action="store_true")
    ap.add_argument("--deadline-min", type=float, default=25.0)
    ap.add_argument("--stop-new-after-min", type=float, default=21.0)
    a = ap.parse_args()
    torch.set_num_threads(1)
    from sb3_contrib import MaskablePPO
    import mini_motorways_env as M
    from experiments import gamectl
    os.makedirs(os.path.join(OUT, "raw"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "rows"), exist_ok=True)
    t_start = time.time()
    tag = "s9twin" if a.twin else f"s9{a.who}"
    who = "p2det" if a.twin else a.who
    model = MaskablePPO.load(CK[who[:2]], device="cpu")
    deterministic = who.endswith("det")
    with gamectl.GameInstance(tag) as g:
        env = M.MiniMotorwaysEnv(max_decisions=600, decision_delta=1.0, request_path=os.path.join(g.ipc_dir, "request.json"),
                                 response_path=os.path.join(g.ipc_dir, "response.json"), start_mode="NORMAL_START")
        if a.twin:
            _, _, h1 = run_episode(env, model, True, SEEDS[0], 0, time.time() + 600, max_steps=60, keep_hashes=True, instrument=False)
            _, _, h2 = run_episode(env, model, True, SEEDS[0], 0, time.time() + 600, max_steps=60, keep_hashes=True, instrument=True)
            _, _, h3 = run_episode(env, model, True, SEEDS[0], 0, time.time() + 600, max_steps=60, keep_hashes=True, instrument=False)
            res = {"steps": len(h1), "plain_vs_instrumented_identical": h1 == h2, "plain_vs_plain_identical (native determinism floor)": h1 == h3, "wall_s": round(time.time() - t_start, 1)}
            json.dump(res, open(os.path.join(OUT, "twin_noninterference.json"), "w"), indent=1)
            print(json.dumps(res), flush=True)
            try:
                env.close()
            except Exception:
                pass
            return
        deadline = t_start + 60 * a.deadline_min
        for idx, seed in enumerate(SEEDS):
            if time.time() > t_start + 60 * a.stop_new_after_min:
                print(f"[{who}] wall-clock guard: not starting seed index {idx}", flush=True)
                break
            row, steps, _ = run_episode(env, model, deterministic, seed, idx, deadline)
            row["policy"] = who
            with open(os.path.join(OUT, "rows", f"{who}.jsonl"), "a") as f:
                f.write(json.dumps(row) + "\n")
            with gzip.open(os.path.join(OUT, "raw", f"{who}_{seed}.json.gz"), "wt") as f:
                json.dump({"columns": ["i", "action", "kind", "x", "y", "dir", "n_legal", "reward", "scoreDelta", "accepted", "edges_added", "edges_removed", "n_houses", "useful_connections", "remaining_to_first_house", "unmasked_mass_on_illegal", "unmasked_top1_illegal"], "steps": steps}, f)
            print(f"[{who}] seed {seed}: steps {row['n_steps']} k_req {row['k_required_native_mask_oracle']} conn_max {row['diag_connections_max']} first_conn {row['diag_first_connection_step']} deliveries {row['diag_deliveries']} "
                  f"first_deliv {row['diag_first_delivery_step']} adds {row['n_add']} swaps {row['n_house_swaps']} houses_max {row['n_houses_max']} pen_missing {row['diag_terminal_penalty_missing']} ({time.time() - t_start:.0f}s)", flush=True)
        try:
            env.close()
        except Exception:
            pass
    print(f"[{who}] DONE wall {time.time() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
