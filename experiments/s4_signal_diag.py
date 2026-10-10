"""S4 Part 5: capped INFERENCE-ONLY native learning-signal diagnostic (no gradient, no optimizer, no checkpoint write).
  python experiments/s4_signal_diag.py run --policies wait,random --tag s4a --deadline-min 25
  python experiments/s4_signal_diag.py run --policies mlp --tag s4b
  python experiments/s4_signal_diag.py run --policies cnn_det --tag s4c      (cnn_stoch likewise)
  python experiments/s4_signal_diag.py summarize
Native NORMAL_START, native reward, 1.0 s decisions, 600-decision horizon (= the proposed pilot / frozen baselines). Seeds are VALIDATION
seeds only (the TEST seeds are not touched). Every policy sees the same seeds."""
import argparse
import json
import os
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from experiments import paths  # noqa: E402,F401
import numpy as np  # noqa: E402

OUT = os.path.join(ROOT, "research", "native_cnn_gate0")
SEEDS = [182150, 205907, 229664, 253421]
CKPT = {"mlp": os.path.join(ROOT, "research", "frozen_baselines", "native_reward_connected_49152.zip"),
        "cnn_det": os.path.join(ROOT, "runs", "2026-10-08_140406_syn_s1_cnn_s13", "checkpoints", "final.zip"),
        "cnn_stoch": os.path.join(ROOT, "runs", "2026-10-08_140406_syn_s1_cnn_s13", "checkpoints", "final.zip")}


def run(a):
    import train_ppo
    from sb3_contrib import MaskablePPO
    import mini_motorways_env as M
    from experiments import gamectl
    from experiments.recording import EpisodeAccumulator
    from experiments.evaluation import policy_entropy, UPGRADE_BASE
    from experiments.train_worker import sha256_file
    os.makedirs(OUT, exist_ok=True)
    t_start = time.time()
    deadline = t_start + 60 * a.deadline_min
    meta = {"initialization": "normal", "start_gap": 0, "reward_mode": "native_score", "decision_interval_seconds": 1.0, "diagnostics": True}
    with gamectl.GameInstance(a.tag) as g:
        env = M.MiniMotorwaysEnv(max_decisions=600, decision_delta=1.0, request_path=os.path.join(g.ipc_dir, "request.json"),
                                 response_path=os.path.join(g.ipc_dir, "response.json"), start_mode="NORMAL_START")
        for pol in a.policies.split(","):
            model = None
            if pol in CKPT:
                model = MaskablePPO.load(CKPT[pol], device="cpu", custom_objects={"GridExtractor": train_ppo.GridExtractor})
            rng = np.random.default_rng(0)
            out_path = os.path.join(OUT, f"diag_{pol}.jsonl")
            done_seeds = set()
            if os.path.exists(out_path):
                done_seeds = {json.loads(l)["requested_seed"] for l in open(out_path) if l.strip()}
            acc = EpisodeAccumulator(meta)
            for seed in SEEDS:
                if seed in done_seeds:
                    continue
                if time.time() > deadline:
                    print(f"[{pol}] wall-clock cap reached before seed {seed}; stopping (not extended)", flush=True)
                    break
                obs, info = env.reset(seed=seed)
                acc.reset_episode(info.get("seed", seed), None, env.last_raw_observation)
                te = tr = False
                cut = False
                while not (te or tr):
                    if time.time() > deadline + 120:
                        cut = True
                        break
                    mask = env.action_masks()
                    legal = np.flatnonzero(mask)
                    ent = None
                    if pol == "wait":
                        ups = [int(x) for x in legal if x >= UPGRADE_BASE]
                        act = 0 if mask[0] else (ups[0] if ups else int(legal[0]))
                    elif pol == "random":
                        act = int(rng.choice(legal))
                    else:
                        act, _ = model.predict(obs, action_masks=mask, deterministic=(pol != "cnn_stoch"))
                        act = int(act)
                        ent = policy_entropy(model, obs, mask)
                    mb = mask.copy()
                    obs, r, te, tr, info = env.step(act)
                    acc.add_step(act, bool(mb[act]), r, info, te, tr, raw_after=env.last_raw_observation, mask_before=mb, policy_entropy=ent)
                row = acc.row(len(done_seeds) + 1, None)
                row.update({"requested_seed": seed, "policy": pol, "cut_by_wallclock": cut, "action_index_counts": acc.action_index_rows(),
                            "checkpoint_sha256": (sha256_file(CKPT[pol]) if pol in CKPT else None)})
                done_seeds.add(seed)
                with open(out_path, "a") as f:
                    f.write(json.dumps(row) + "\n")
                print(f"[{pol}] seed {seed}: len {row['length_decisions']} score {row['native_score']} return {row['return']} deliveries {row['diag_deliveries']} "
                      f"conn_max {row['diag_connections_max']} adds {row['add_count']} removes {row['remove_count']} ({time.time() - t_start:.0f}s)", flush=True)
        try:
            env.close()
        except Exception:
            pass
    print("wall seconds", round(time.time() - t_start), flush=True)


def summarize():
    pols = ["wait", "random", "mlp", "cnn_det", "cnn_stoch"]
    rows = {p: [json.loads(l) for l in open(os.path.join(OUT, f"diag_{p}.jsonl")) if l.strip()] for p in pols if os.path.exists(os.path.join(OUT, f"diag_{p}.jsonl"))}
    S = {}
    for p, rs in rows.items():
        n = len(rs)
        f = lambda k: [r[k] for r in rs if r.get(k) is not None]
        mean = lambda xs: (statistics.fmean(xs) if xs else None)
        tot = max(sum(r["length_decisions"] for r in rs), 1)
        S[p] = {"episodes": n, "seeds": [r["requested_seed"] for r in rs],
                "frac_ge1_delivery": sum(1 for r in rs if r["diag_deliveries"] > 0) / n,
                "frac_nonzero_intermediate_reward": sum(1 for r in rs if r["diag_n_nonzero_nonterminal_reward"] > 0) / n,
                "first_delivery_step_mean": mean(f("diag_first_delivery_step")), "mean_deliveries": mean(f("diag_deliveries")),
                "frac_ge1_connection": sum(1 for r in rs if r["diag_connections_max"] > 0) / n, "connections_max_mean": mean(f("diag_connections_max")),
                "connections_final_mean": mean(f("diag_connections_final")), "first_connection_step_mean": mean(f("diag_first_connection_step")),
                "connection_to_first_delivery_steps": f("diag_connection_to_first_delivery_steps"),
                "survival_mean": mean(f("length_decisions")), "survival_per_seed": f("length_decisions"),
                "return_mean": mean(f("return")), "intermediate_reward_sum_mean": mean(f("diag_intermediate_reward_sum")), "terminal_reward_sum_mean": mean(f("diag_terminal_reward_sum")),
                "native_score_mean": mean(f("native_score")),
                "action_fractions": {k: sum(r[k + "_count"] for r in rs) / tot for k in ("wait", "add", "remove", "upgrade")},
                "adds_accepted_mean": mean(f("diag_n_add_accepted")), "removes_accepted_mean": mean(f("diag_n_remove_accepted")),
                "distinct_edges_added_mean": mean(f("diag_distinct_edges_added")), "churn_mean": mean(f("diag_churn")), "repeated_edits_mean": mean(f("diag_repeated_edits")),
                "road_edges_initial_mean": mean(f("diag_road_edges_initial")), "road_edges_max_mean": mean(f("diag_road_edges_max")),
                "legal_actions_mean": mean(f("diag_legal_mean")), "legal_add_mean": mean(f("diag_legal_add_mean")),
                "policy_entropy_mean": mean(f("diag_policy_entropy_mean")), "empirical_action_entropy_mean": mean(f("diag_empirical_action_entropy")),
                "score_event_values": sorted({v for r in rs for v in r["diag_score_event_values"]}),
                "first_nonzero_reward_steps": f("diag_first_nonzero_nonterminal_step"), "diag_errors": sum(r["diag_errors"] for r in rs),
                "cut_by_wallclock": sum(1 for r in rs if r["cut_by_wallclock"])}
    json.dump(S, open(os.path.join(OUT, "diagnostic_summary.json"), "w"), indent=1)
    for p, s in S.items():
        print(p, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items() if k in ("episodes", "frac_ge1_delivery", "frac_nonzero_intermediate_reward", "first_delivery_step_mean",
              "frac_ge1_connection", "connections_max_mean", "survival_mean", "return_mean", "adds_accepted_mean", "removes_accepted_mean", "churn_mean", "action_fractions")})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "summarize"])
    ap.add_argument("--policies", default="wait,random")
    ap.add_argument("--tag", default="s4diag")
    ap.add_argument("--deadline-min", type=float, default=25.0)
    a = ap.parse_args()
    run(a) if a.mode == "run" else summarize()
