"""S19 trainer: masked PPO-clip with GAE on a SEMI-MARKOV decision process (per-transition discount gamma^duration), learned value baseline,
entropy bonus.  One configuration, fixed seed, ~20 min hard budget, periodic validation on the validation game seeds (507919/579190) with
checkpoint selection by validation deliveries median (never by test).  Rollouts run in 4 worker processes (1 torch thread each).

SMDP handling: a strategic transition (s, a, R, d, s') carries the within-step discounted reward R = sum_{j<d} gamma^j r_j (from StrategicEnv
info['reward_disc']) and the duration d; the critic target / GAE use gamma^d:
    delta_t = R_t + gamma^{d_t} V(s_{t+1}) - V(s_t),      A_t = delta_t + gamma^{d_t} lambda A_{t+1}
(gamma decays per primitive decision, lambda per strategic transition; at lambda = 1 this is exactly the discounted SMDP return)
(unit-tested against a brute-force discounted return in checks.py).  Rewards are scaled by REWARD_SCALE for the optimiser only; deliveries /
survival are tracked separately from the training return.

python research/s19_hybrid_rl/code/train.py -> research/s19_hybrid_rl/runs/s19/{train_log.json, best.pt, last.pt, ckpt_*.pt}"""
import json
import multiprocessing as mp
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import strategic_env as S  # noqa: E402
from policy import CandidatePolicy  # noqa: E402

RUN_DIR = os.path.join(os.path.dirname(HERE), "runs", "s19")
CONFIG: Dict[str, Any] = {"seed": 19, "gamma": 0.99, "lam": 0.95, "lr": 3e-4, "clip": 0.2, "ent_coef": 0.01, "vf_coef": 0.5, "max_grad_norm": 0.5, "epochs": 4, "minibatch": 256,
                          "episodes_per_iter": 16, "workers": 4, "budget_s": 1200.0, "val_every": 5, "reward_scale": 0.1, "hidden": 64, "K": S.K, "wait_interval": S.WAIT_INTERVAL,
                          "k_alt": S.K_ALT, "adv_norm": True, "val_env_seeds": 3, "train_scenarios": [S.SC[i]["id"] for i in S.TRAIN_IDX], "val_scenarios": [S.SC[i]["id"] for i in S.VAL_IDX]}

_WENV = None


def _init_worker():
    global _WENV
    torch.set_num_threads(1)
    _WENV = S.StrategicEnv(S.TRAIN_IDX, seed=0, gamma=CONFIG["gamma"], K=CONFIG["K"], wait_interval=CONFIG["wait_interval"], k_alt=CONFIG["k_alt"])


def _load(pol: CandidatePolicy, sd_np: Dict[str, np.ndarray]):
    pol.load_state_dict({k: torch.as_tensor(v) for k, v in sd_np.items()})


def run_episode(args) -> Dict[str, Any]:
    """worker: one strategic episode; returns the trajectory (numpy) + episode stats."""
    sd_np, idx, env_seed, deterministic, sample_seed, keep_traj = args
    global _WENV
    if _WENV is None:
        _init_worker()
    pol = CandidatePolicy(hidden=CONFIG["hidden"])
    _load(pol, sd_np)
    pol.eval()
    gen = torch.Generator().manual_seed(int(sample_seed))
    env = _WENV
    obs, info = env.reset(options={"scenario_index": int(idx), "env_seed": int(env_seed)})
    G, C, M, A, LP, V, R, D = [], [], [], [], [], [], [], []
    ret_raw, deliveries, term = 0.0, 0, False
    while True:
        a, lp, v = pol.act(obs, deterministic=deterministic, gen=gen)
        nobs, r, te, tr, inf = env.step(a)
        if keep_traj:
            G.append(obs["global"]); C.append(obs["cand"]); M.append(obs["mask"]); A.append(a); LP.append(lp); V.append(v); R.append(inf["reward_disc"]); D.append(inf["duration"])
        ret_raw += r
        deliveries += inf["deliveries"]
        if te or tr:
            term = te
            break
        obs = nobs
    summ = env.ctrl.summary()
    stats = {"scenario": info["scenario"], "split": info["split"], "env_seed": int(env_seed), "return_raw": ret_raw, "deliveries": deliveries, "decisions": env.env.t, "survived_600": bool(not term),
             "wait_frequency": summ["wait_frequency"], "kind_counts": summ["kind_counts"], "mean_duration": summ["mean_duration"], "strategic_actions": summ["strategic_actions"],
             "abandoned_plans": summ["abandoned_plans"], "illegal_selected": summ["illegal_selected"], "illegal_suppressed": summ["illegal_suppressed"], "truncated_candidates": summ["truncated_candidates"],
             "connected_end": len(env.env.connected_houses()), "houses": len(env.env.houses)}
    traj = None
    if keep_traj:
        traj = {"g": np.stack(G), "c": np.stack(C), "m": np.stack(M), "a": np.array(A, np.int64), "lp": np.array(LP, np.float32), "v": np.array(V, np.float32), "r": np.array(R, np.float32), "d": np.array(D, np.int64)}
    return {"stats": stats, "traj": traj}


def gae_smdp(r: np.ndarray, v: np.ndarray, d: np.ndarray, gamma: float, lam: float, v_last: float = 0.0):
    """GAE with per-transition discount gamma^d (episode ends at the last transition: v_last = 0 for terminal / horizon)."""
    T = len(r)
    adv = np.zeros(T, np.float64)
    last, vn = 0.0, v_last
    for t in range(T - 1, -1, -1):
        gd = gamma ** d[t]
        delta = r[t] + gd * vn - v[t]
        last = delta + gd * lam * last
        adv[t] = last
        vn = v[t]
    return adv.astype(np.float32), (adv + v).astype(np.float32)


def sd_numpy(pol: CandidatePolicy) -> Dict[str, np.ndarray]:
    return {k: v.detach().cpu().numpy().copy() for k, v in pol.state_dict().items()}


def validate(pool, sd_np, deterministic=True, seed_base=7000) -> Dict[str, Any]:
    jobs = [(sd_np, idx, S.env_seed(idx, s), deterministic, 100000 + 10 * idx + s, False) for idx in S.VAL_IDX for s in range(CONFIG["val_env_seeds"])]
    res = pool.map(run_episode, jobs)
    st = [r["stats"] for r in res]
    dl = [s["deliveries"] for s in st]
    return {"episodes": len(st), "deliveries_median": float(np.median(dl)), "deliveries_mean": float(np.mean(dl)), "survived": int(sum(s["survived_600"] for s in st)),
            "return_mean": float(np.mean([s["return_raw"] for s in st])), "wait_frequency": float(np.mean([s["wait_frequency"] for s in st])), "scenarios": sorted({s["scenario"] for s in st})}


def main():
    os.makedirs(RUN_DIR, exist_ok=True)
    cfg = CONFIG
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    torch.set_num_threads(cfg["workers"])
    pol = CandidatePolicy(hidden=cfg["hidden"])
    opt = torch.optim.Adam(pol.parameters(), lr=cfg["lr"], eps=1e-5)
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(cfg["workers"], initializer=_init_worker)
    log: Dict[str, Any] = {"config": cfg, "iterations": [], "validation": [], "train_scenarios_seen": {}, "wall_s": 0.0, "stopped_reason": None}
    t_start = time.time()
    best = (-1.0, -1.0, None)
    it, cursor, iter_times = 0, 0, []
    try:
        while True:
            elapsed = time.time() - t_start
            est = float(np.mean(iter_times[-3:])) if iter_times else 0.0
            if elapsed + est > cfg["budget_s"]:
                log["stopped_reason"] = f"budget ({cfg['budget_s']} s) reached at iteration {it}"
                break
            t_it = time.time()
            sd_np = sd_numpy(pol)
            jobs = []
            for _ in range(cfg["episodes_per_iter"]):
                idx = S.TRAIN_IDX[cursor % len(S.TRAIN_IDX)]
                cursor += 1
                jobs.append((sd_np, idx, int(rng.integers(1 << 30)), False, int(rng.integers(1 << 30)), True))
            res = pool.map(run_episode, jobs)
            # ---- GAE per episode (SMDP)
            G, C, M, A, LP, ADV, RET, VV = [], [], [], [], [], [], [], []
            for r in res:
                tj = r["traj"]
                adv, ret = gae_smdp(tj["r"] * cfg["reward_scale"], tj["v"], tj["d"], cfg["gamma"], cfg["lam"])
                G.append(tj["g"]); C.append(tj["c"]); M.append(tj["m"]); A.append(tj["a"]); LP.append(tj["lp"]); ADV.append(adv); RET.append(ret); VV.append(tj["v"])
                s = r["stats"]
                assert s["split"] == "train" and s["scenario"] not in S.TEST_IDS, "test scenario in training!"
                log["train_scenarios_seen"][s["scenario"]] = log["train_scenarios_seen"].get(s["scenario"], 0) + 1
            g = torch.as_tensor(np.concatenate(G)); c = torch.as_tensor(np.concatenate(C)); m = torch.as_tensor(np.concatenate(M)); a = torch.as_tensor(np.concatenate(A))
            lp_old = torch.as_tensor(np.concatenate(LP)); adv = torch.as_tensor(np.concatenate(ADV)); ret = torch.as_tensor(np.concatenate(RET))
            N = len(a)
            if cfg["adv_norm"]:
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            # ---- PPO update
            pol.train()
            stats = {"pg_loss": [], "v_loss": [], "entropy": [], "approx_kl": [], "clipfrac": [], "entropy_decision_states": []}
            for _ in range(cfg["epochs"]):
                perm = torch.randperm(N)
                for i0 in range(0, N, cfg["minibatch"]):
                    b = perm[i0:i0 + cfg["minibatch"]]
                    logits, v = pol(g[b], c[b], m[b])
                    logp_all = torch.log_softmax(logits, -1)
                    logp = logp_all.gather(1, a[b].unsqueeze(1)).squeeze(1)
                    p = logp_all.exp()
                    ent_i = -(p * logp_all.masked_fill(~m[b], 0.0)).sum(-1)
                    ent = ent_i.mean()
                    multi = m[b].sum(-1) > 1                     # states with a real choice (not WAIT-only)
                    ratio = (logp - lp_old[b]).exp()
                    pg = -torch.min(ratio * adv[b], ratio.clamp(1 - cfg["clip"], 1 + cfg["clip"]) * adv[b]).mean()
                    vl = 0.5 * ((v - ret[b]) ** 2).mean()
                    loss = pg + cfg["vf_coef"] * vl - cfg["ent_coef"] * ent
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(pol.parameters(), cfg["max_grad_norm"])
                    opt.step()
                    with torch.no_grad():
                        stats["pg_loss"].append(float(pg)); stats["v_loss"].append(float(vl)); stats["entropy"].append(float(ent))
                        if multi.any():
                            stats["entropy_decision_states"].append(float(ent_i[multi].mean()))
                        stats["approx_kl"].append(float(((ratio - 1) - (logp - lp_old[b])).mean())); stats["clipfrac"].append(float(((ratio - 1).abs() > cfg["clip"]).float().mean()))
            pol.eval()
            st = [r["stats"] for r in res]
            kinds = {k: sum(s["kind_counts"][k] for s in st) for k in S.KINDS}
            tot = max(1, sum(kinds.values()))
            row = {"iter": it, "wall_s": round(time.time() - t_start, 1), "episodes": len(st), "transitions": int(N), "return_raw_mean": float(np.mean([s["return_raw"] for s in st])),
                   "deliveries_mean": float(np.mean([s["deliveries"] for s in st])), "deliveries_median": float(np.median([s["deliveries"] for s in st])), "survival_rate": float(np.mean([s["survived_600"] for s in st])),
                   "entropy": float(np.mean(stats["entropy"])), "entropy_decision_states": float(np.mean(stats["entropy_decision_states"])) if stats["entropy_decision_states"] else 0.0, "value_loss": float(np.mean(stats["v_loss"])), "pg_loss": float(np.mean(stats["pg_loss"])), "approx_kl": float(np.mean(stats["approx_kl"])),
                   "clipfrac": float(np.mean(stats["clipfrac"])), "wait_frequency": kinds["WAIT"] / tot, "kind_frequency": {k: v / tot for k, v in kinds.items()}, "mean_duration": float(np.mean([s["mean_duration"] for s in st])),
                   "abandoned_plans": int(sum(s["abandoned_plans"] for s in st)), "illegal_selected": int(sum(s["illegal_selected"] for s in st)), "truncated_candidates": int(sum(s["truncated_candidates"] for s in st)),
                   "value_mean": float(np.concatenate(VV).mean()), "return_target_mean": float(ret.mean())}
            log["iterations"].append(row)
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items() if k != "kind_frequency"}), flush=True)
            # ---- validation + checkpoint selection (validation seeds only)
            if it % cfg["val_every"] == 0:
                val = validate(pool, sd_numpy(pol))
                val["iter"] = it
                val["wall_s"] = round(time.time() - t_start, 1)
                log["validation"].append(val)
                torch.save({"state_dict": pol.state_dict(), "config": cfg, "iter": it}, os.path.join(RUN_DIR, f"ckpt_{it:04d}.pt"))
                key = (val["deliveries_median"], val["deliveries_mean"])
                if key > best[:2]:
                    best = (key[0], key[1], it)
                    torch.save({"state_dict": pol.state_dict(), "config": cfg, "iter": it, "val": val}, os.path.join(RUN_DIR, "best.pt"))
                print("VAL", json.dumps({k: v for k, v in val.items() if k != "scenarios"}), "best_iter", best[2], flush=True)
            iter_times.append(time.time() - t_it)
            it += 1
    finally:
        # final validation of the last policy (also a selection candidate)
        val = validate(pool, sd_numpy(pol))
        val["iter"] = it
        val["wall_s"] = round(time.time() - t_start, 1)
        log["validation"].append(val)
        torch.save({"state_dict": pol.state_dict(), "config": cfg, "iter": it, "val": val}, os.path.join(RUN_DIR, "last.pt"))
        key = (val["deliveries_median"], val["deliveries_mean"])
        if key > best[:2]:
            best = (key[0], key[1], it)
            torch.save({"state_dict": pol.state_dict(), "config": cfg, "iter": it, "val": val}, os.path.join(RUN_DIR, "best.pt"))
        log["best"] = {"iter": best[2], "val_deliveries_median": best[0], "val_deliveries_mean": best[1]}
        log["wall_s"] = round(time.time() - t_start, 1)
        log["total_iterations"] = it
        log["total_episodes"] = int(sum(r["episodes"] for r in log["iterations"]))
        log["total_transitions"] = int(sum(r["transitions"] for r in log["iterations"]))
        log["total_primitive_decisions"] = int(sum(r["mean_duration"] * r["transitions"] for r in log["iterations"]))
        json.dump(log, open(os.path.join(RUN_DIR, "train_log.json"), "w"), indent=1)
        pool.close()
        pool.join()
        print("DONE", json.dumps({k: v for k, v in log.items() if k in ("best", "wall_s", "total_iterations", "total_episodes", "total_transitions", "stopped_reason")}), flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1:                       # smoke runs: python train.py <budget_s> [run_dir]
        CONFIG["budget_s"] = float(sys.argv[1])
    if len(sys.argv) > 2:
        RUN_DIR = sys.argv[2]
    main()
