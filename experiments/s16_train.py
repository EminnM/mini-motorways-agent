"""S16 supervised diagnostic: train models A (existing 7x7 CNN), B (dilated 31x31 CNN), C (B, group-blind) on identical data; fixed budget; held-out evaluation with paired bootstrap.
   python experiments/s16_train.py"""
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from synthetic_rl import layout as L  # noqa: E402
from synthetic_rl import obs_v2 as V  # noqa: E402
from synthetic_rl import s16_models as M  # noqa: E402

OUT = os.path.join(ROOT, "research", "s16_colour_routing")
EPOCHS, TIME_PER_MODEL_S, LR, BATCH, SEED = int(os.environ.get('S16_EPOCHS', 30)), 210, 1e-3, 64, 16
HOLDOUT_SLOT = 2                                   # query slot 2 never seen in training -> "unseen group permutation" test


def load(split):
    z = np.load(os.path.join(OUT, f"{split}.npz"))
    meta = json.load(open(os.path.join(OUT, f"{split}.json")))
    obs = torch.as_tensor(z["obs"].astype(np.float32))
    add_mask = torch.zeros(len(meta), L.N_TILES * 4, dtype=torch.bool)
    tgt = torch.zeros(len(meta), L.N_TILES * 4, dtype=torch.bool)
    oth = torch.zeros(len(meta), L.N_TILES * 4, dtype=torch.bool)
    for i, (m, mk) in enumerate(zip(meta, z["mask"])):
        add_mask[i] = torch.as_tensor(mk[1:L.UPGRADE_BASE])
        for a in m["valid"]:
            tgt[i, a - 1] = True
        for a in m["other_valid"]:
            oth[i, a - 1] = True
    return meta, obs, add_mask, tgt, oth


def set_loss(logits, add_mask, tgt):
    lg = logits.masked_fill(~add_mask, -1e9)
    lse_all = torch.logsumexp(lg, dim=1)
    lse_valid = torch.logsumexp(lg.masked_fill(~tgt, -1e9), dim=1)
    return (lse_all - lse_valid).mean()


@torch.no_grad()
def predict(model, obs, add_mask, bs=128):
    model.eval()
    out = []
    for i in range(0, len(obs), bs):
        lg = model(obs[i:i + bs]).masked_fill(~add_mask[i:i + bs], -1e9)
        out.append(lg.argmax(dim=1))
    return torch.cat(out)


def metrics(meta, pred, tgt, oth):
    n = len(meta)
    hit = np.array([bool(tgt[i, pred[i]]) for i in range(n)])
    decoy = np.array([bool(oth[i, pred[i]]) and not hit[i] for i in range(n)])
    idx = {k: np.array([i for i, m in enumerate(meta) if cond(m)]) for k, cond in (("all", lambda m: True), ("within_rf", lambda m: m["within_rf"]), ("beyond_rf", lambda m: not m["within_rf"]),
                                                                                  ("decoy_nearer", lambda m: m["decoy_nearer"]), ("colour_dependent", lambda m: m["colour_dependent"]), ("unseen_slot", lambda m: m["query_slot"] == HOLDOUT_SLOT),
                                                                                  ("cf_twins", lambda m: m["cf"]), ("originals_of_cf", lambda m: m["colour_dependent"] and not m["cf"]))}
    res = {k: {"n": int(len(v)), "valid_set_acc": float(hit[v].mean()) if len(v) else None, "decoy_target_rate": float(decoy[v].mean()) if len(v) else None} for k, v in idx.items()}
    # counterfactual PAIR accuracy: both the original and its colour-swapped twin correct
    pairs = {}
    for i, m in enumerate(meta):
        if m["colour_dependent"]:
            pairs.setdefault(m["geom"], []).append(i)
    both = [all(hit[j] for j in v) for v in pairs.values() if len(v) == 2]
    res["cf_pairs_both_correct"] = {"n": len(both), "acc": float(np.mean(both)) if both else None}
    return res, hit


def paired_bootstrap(h1, h2, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    d = h1.astype(float) - h2.astype(float)
    bs = [d[rng.integers(len(d), size=len(d))].mean() for _ in range(n)]
    return {"mean_diff": float(d.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "n": int(len(d))}


def main():
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    t_all = time.time()
    tr_meta, tr_obs, tr_mask, tr_tgt, tr_oth = load("train")
    cd_only = os.environ.get("S16_CD_ONLY") == "1"               # Outcome-C investigation: train only on colour-dependent pairs (geometry alone caps at 50 % there)
    keep = torch.as_tensor([m["query_slot"] != HOLDOUT_SLOT and (m["colour_dependent"] or not cd_only) for m in tr_meta])
    tr_obs, tr_mask, tr_tgt = tr_obs[keep], tr_mask[keep], tr_tgt[keep]
    va = load("val")
    te = load("test")
    # sanity: labels feasible, counterfactual targets change, ambiguity of identical observations
    checks = {"train_samples_used": int(keep.sum()), "train_slot2_dropped": int((~keep).sum())}
    for name, (meta, obs, mask, tgt, oth) in (("val", va), ("test", te)):
        feas = all(bool((tgt[i] & ~mask[i]).sum() == 0) and tgt[i].any() for i in range(len(meta)))
        cf_change = all(not bool((tgt[i] & oth[i]).any()) for i, m in enumerate(meta) if m["colour_dependent"])
        seen, amb = {}, 0
        for i in range(len(meta)):
            h = hash(obs[i].numpy().tobytes())
            if h in seen and not torch.equal(tgt[seen[h]], tgt[i]):
                amb += 1
            seen.setdefault(h, i)
        checks[name] = {"labels_feasible_and_nonempty": feas, "cf_target_sets_disjoint": cf_change, "identical_obs_with_conflicting_targets": amb, "n": len(meta)}
    # v1 identity of counterfactual twins (test)
    meta_t, obs_t = te[0], te[1]
    by_geom = {}
    for i, m in enumerate(meta_t):
        if m["colour_dependent"]:
            by_geom.setdefault(m["geom"], []).append(i)
    v1_same = sum(1 for v in by_geom.values() if len(v) == 2 and np.array_equal(V.strip_to_v1(obs_t[v[0]].numpy()), V.strip_to_v1(obs_t[v[1]].numpy())))
    v2_diff = sum(1 for v in by_geom.values() if len(v) == 2 and not torch.equal(obs_t[v[0]], obs_t[v[1]]))
    checks["test_cf_pairs"] = {"pairs": len(by_geom), "v1_identical": v1_same, "v2_different": v2_diff}
    results = {"checks": checks, "config": {"epochs": EPOCHS, "time_cap_per_model_s": TIME_PER_MODEL_S, "lr": LR, "batch": BATCH, "seed": SEED, "optimizer": "Adam", "loss": "-log sum_{valid} softmax over legal ADD actions"}, "models": {}}
    hits = {}
    for name in "ABC":
        torch.manual_seed(SEED)
        model = M.make(name)
        n_par = sum(p.numel() for p in model.parameters())
        rf_emp = M.effective_rf(M.make(name))
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        curve = []
        t0 = time.time()
        ep = 0
        best = (0.0, None)
        while ep < EPOCHS and time.time() - t0 < TIME_PER_MODEL_S:
            model.train()
            perm = torch.randperm(len(tr_obs))
            tl = 0.0
            for i in range(0, len(perm), BATCH):
                j = perm[i:i + BATCH]
                loss = set_loss(model(tr_obs[j]), tr_mask[j], tr_tgt[j])
                opt.zero_grad()
                loss.backward()
                opt.step()
                tl += float(loss) * len(j)
            ep += 1
            pv = predict(model, va[1], va[2])
            mv, _ = metrics(va[0], pv, va[3], va[4])
            curve.append({"epoch": ep, "train_loss": round(tl / len(perm), 4), "val_acc": round(mv["all"]["valid_set_acc"], 4), "val_cf_pair_acc": mv["cf_pairs_both_correct"]["acc"], "s": round(time.time() - t0, 1)})
            if mv["all"]["valid_set_acc"] >= best[0]:
                best = (mv["all"]["valid_set_acc"], {k: v.clone() for k, v in model.state_dict().items()})
        model.load_state_dict(best[1])
        torch.save(model.state_dict(), os.path.join(OUT, f"model_{name}{'_cd' if cd_only else ''}.pt"))
        pt = predict(model, te[1], te[2])
        mt, hit = metrics(te[0], pt, te[3], te[4])
        hits[name] = hit
        fails = [{"geom": m["geom"], "cf": m["cf"], "query": m["query"], "query_slot": m["query_slot"], "decoy_slot": m["decoy_slot"], "within_rf": m["within_rf"], "cost_correct": m["cost_correct"], "cost_decoy": m["cost_decoy"],
                  "pred": L.decode_action(int(pt[i]) + 1), "valid": [L.decode_action(a) for a in m["valid"]], "chose_decoy_target": bool(te[4][i, pt[i]])} for i, m in enumerate(te[0]) if not hit[i] and m["colour_dependent"]][:6]
        results["models"][name] = {"params": n_par, "theoretical_rf": model.rf, "empirical_rf": rf_emp, "epochs_done": ep, "train_s": round(time.time() - t0, 1), "best_val_acc": best[0], "curve": curve, "test": mt, "failure_examples": fails}
        print(name, json.dumps({k: v for k, v in results["models"][name].items() if k not in ("curve", "failure_examples")}), flush=True)
    meta = te[0]
    sub = {"all": np.arange(len(meta)), "colour_dependent": np.array([i for i, m in enumerate(meta) if m["colour_dependent"]]), "beyond_rf": np.array([i for i, m in enumerate(meta) if not m["within_rf"]]), "within_rf": np.array([i for i, m in enumerate(meta) if m["within_rf"]]),
           "unseen_slot": np.array([i for i, m in enumerate(meta) if m["query_slot"] == HOLDOUT_SLOT])}
    results["paired"] = {f"{a}_minus_{b}": {k: paired_bootstrap(hits[a][v], hits[b][v]) for k, v in sub.items() if len(v)} for a, b in (("B", "A"), ("B", "C"), ("A", "C"))}
    results["compute_s_total"] = round(time.time() - t_all, 1)
    json.dump(results, open(os.path.join(OUT, "results_cd_only.json" if cd_only else "results.json"), "w"), indent=1, default=str)
    print("PAIRED", json.dumps(results["paired"]))
    print("TOTAL", results["compute_s_total"], "s")


if __name__ == "__main__":
    main()
