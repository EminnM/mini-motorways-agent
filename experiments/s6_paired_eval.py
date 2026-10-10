"""S6 paired native evaluation: A = unfiltered deterministic S1 CNN (seed 13), B = same checkpoint behind the EVALUATION-ONLY anti-cycle filter.
Inference only. Same checkpoint, validation seeds, horizon, native reward, native legal actions; every episode starts from a fresh RESET(seed).
   python experiments/s6_paired_eval.py --seeds 182150,205907 --cap-min 15
Writes research/native_cnn_anticycle/paired_<seed>_<A|B>.json"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from experiments import paths  # noqa: E402,F401
import numpy as np  # noqa: E402

OUT = os.path.join(ROOT, "research", "native_cnn_anticycle")
CKPT = os.path.join(ROOT, "runs", "2026-10-08_140406_syn_s1_cnn_s13", "checkpoints", "final.zip")


def edges(raw):
    s = set()
    for t in raw["tiles"]:
        for d in ("E", "S"):
            if t["road"][d] != "None":
                s.add((t["coord"][0], t["coord"][1], d))
    return s


def run_episode(env, model, seed, use_filter, deadline):
    import torch
    from sb3_contrib.common.maskable.utils import get_action_masks  # noqa: F401
    from experiments.anticycle import AntiCycleFilter, road_signature, ranking_from_logits, revisit_stats
    from experiments.evaluation import policy_entropy
    from experiments.recording import EpisodeAccumulator
    from synthetic_rl import layout as L
    meta = {"initialization": "normal", "start_gap": 0, "reward_mode": "native_score", "decision_interval_seconds": 1.0, "diagnostics": True}
    acc = EpisodeAccumulator(meta)
    filt = AntiCycleFilter(enabled=use_filter)
    obs, info = env.reset(seed=seed)
    acc.reset_episode(info.get("seed", seed), None, env.last_raw_observation)
    steps, sigs = [], []
    te = tr = False
    i = 0
    cut = False
    while not (te or tr):
        if time.time() > deadline:
            cut = True
            break
        mask = env.action_masks().copy()
        raw_b = env.last_raw_observation
        sig = road_signature(raw_b)               # EVALUATION-ONLY privileged state (raw native observation)
        with torch.no_grad():
            pol = model.policy
            logits = pol.action_net(pol.mlp_extractor(pol.extract_features(torch.as_tensor(obs[None], dtype=torch.float32)))[0])[0].numpy()
        ranking = ranking_from_logits(logits, mask)
        pred = int(model.predict(obs, action_masks=mask, deterministic=True)[0])
        assert ranking[0] == pred, "ranking head != model.predict"
        act, iv = filt.choose(sig, ranking, i)
        ent = policy_entropy(model, obs, mask)
        e0 = edges(raw_b)
        mb = mask.copy()
        obs, r, te, tr, inf = env.step(act)
        raw_a = env.last_raw_observation
        sig_next = road_signature(raw_a)
        filt.observe(sig, act, sig_next)
        acc.add_step(act, bool(mb[act]), r, inf, te, tr, raw_after=raw_a, mask_before=mb, policy_entropy=ent)
        e1 = edges(raw_a)
        sigs.append(sig)
        dec = L.decode_action(act)
        steps.append({"i": i, "action": act, "decoded": list(dec), "policy_top": pred, "intervened": iv is not None, "reward": r, "road_sig": sig, "n_legal": int(mb.sum()),
                      "edges_added": len(e1 - e0), "edges_removed": len(e0 - e1), "terminated": te, "truncated": tr})
        i += 1
    sigs.append(road_signature(env.last_raw_observation))
    row = acc.row(1, None)
    row.pop("action_index_counts", None)
    stats = revisit_stats(sigs)
    adds = [s for s in steps if s["decoded"][0] == "ADD"]
    revisit_adds = sum(1 for k, s in enumerate(steps) if s["decoded"][0] == "ADD" and sigs[k + 1] in set(sigs[:k + 1]))
    swap_adds = sum(1 for s in adds if s["edges_removed"] > 0)
    return {"seed": seed, "arm": "B_filtered" if use_filter else "A_unfiltered", "row": row, "road_state_stats": stats, "revisit_adds": revisit_adds, "swap_adds": swap_adds,
            "n_adds": len(adds), "n_interventions": len(filt.interventions), "interventions": filt.interventions, "n_cycle_criterion_decisions": filt.n_detected,
            "detected_not_continuing": filt.detected_not_continuing, "cut_by_wallclock": cut, "steps": steps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="182150,205907")
    ap.add_argument("--cap-min", type=float, default=15.0)
    a = ap.parse_args()
    from sb3_contrib import MaskablePPO
    import mini_motorways_env as M
    from experiments import gamectl
    os.makedirs(OUT, exist_ok=True)
    model = MaskablePPO.load(CKPT, device="cpu")
    t0 = time.time()
    deadline = t0 + 60 * a.cap_min
    with gamectl.GameInstance("s6pair") as g:
        env = M.MiniMotorwaysEnv(max_decisions=600, decision_delta=1.0, request_path=os.path.join(g.ipc_dir, "request.json"),
                                 response_path=os.path.join(g.ipc_dir, "response.json"), start_mode="NORMAL_START")
        for seed in [int(s) for s in a.seeds.split(",")]:
            for use_filter in (False, True):
                if time.time() > deadline:
                    print("wall-clock cap reached; stopping (partial results)", flush=True)
                    break
                res = run_episode(env, model, seed, use_filter, deadline)
                res["wall_s"] = round(time.time() - t0, 1)
                json.dump(res, open(os.path.join(OUT, f"paired_{seed}_{'B' if use_filter else 'A'}.json"), "w"))
                r = res["row"]
                print(f"seed {seed} {res['arm']}: len {r['length_decisions']} deliveries {r['diag_deliveries']} conn_max {r['diag_connections_max']} first_conn {r['diag_first_connection_step']} "
                      f"first_deliv {r['diag_first_delivery_step']} interventions {res['n_interventions']} cycles {res['road_state_stats']['two_state_cycles']} "
                      f"osc_steps {res['road_state_stats']['two_state_oscillation_steps']} penalty_missing {r['diag_terminal_penalty_missing']} ({res['wall_s']}s)", flush=True)
        try:
            env.close()
        except Exception:
            pass
    print("native wall seconds", round(time.time() - t0, 1))


if __name__ == "__main__":
    main()
