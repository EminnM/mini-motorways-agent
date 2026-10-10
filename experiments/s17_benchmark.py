"""S17: (1) hybrid construction + migration + equivalence on saved native and S7 observations; (2) attention functional checks; (3) controlled supervised benchmark on the cached S16 dataset
(A existing CNN, B CNN+attention, C attention group-blind; all trunks initialised from the migrated S8-C); (4) inference-only ablations; (5) cost measurements.   python experiments/s17_benchmark.py"""
import gzip
import hashlib
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sb3_contrib import MaskablePPO  # noqa: E402
from synthetic_rl import layout as L  # noqa: E402
from synthetic_rl import obs_v2 as V  # noqa: E402
from synthetic_rl import s7  # noqa: E402
from synthetic_rl.models_v2 import SpatialMaskablePolicyV2  # noqa: E402,F401
from synthetic_rl.models_v3 import SpatialMaskablePolicyV3, SpatialExtractorV3, SupervisedTileNet, migrate_v2_to_v3  # noqa: E402
from synthetic_rl.s14_env import MultiHouseEnvV2  # noqa: E402
from synthetic_rl.s13_env import load_scenarios  # noqa: E402
import mini_motorways_env as NATIVE  # noqa: E402
import s16_train as S16  # noqa: E402

OUT = os.path.join(ROOT, "research", "s17_global_attention")
S16D = os.path.join(ROOT, "research", "s16_colour_routing")
CK_MIG = os.path.join(ROOT, "research", "s14_group_observation", "s8c_migrated_v2_step0.zip")
os.makedirs(OUT, exist_ok=True)
EPOCHS, TCAP, LR, BATCH, SEED = 30, 210, 1e-3, 64, 17
try:
    import psutil
except Exception:  # noqa: BLE001
    psutil = None


def rss_mb():
    return round(psutil.Process().memory_info().rss / 2**20, 1) if psutil else None


def saved_observations():
    obs = []
    with gzip.open(os.path.join(ROOT, "research", "s7_long_route", "native_normal_start_captures.jsonl.gz"), "rt") as f:
        for line in f:
            c = json.loads(line)
            for key in ("reset", "first_house"):
                if c.get(key):
                    obs.append(V.encode_native_v2(c[key]["raw"], V.GroupSlotTracker()))
    sets, _ = s7.build_task_sets()
    env = s7.LongRouteEnv(sets["validation"][7], sampling="sequential")
    for i in range(40):
        o, _ = env.reset(options={"task_index": i})
        obs.append(V.s7_v1_to_v2(o))
        for _ in range(5):
            o, _, te, tr, _ = env.step(int(np.random.default_rng(i).choice(np.flatnonzero(env.action_masks()))))
            obs.append(V.s7_v1_to_v2(o))
            if te or tr:
                break
    e13 = MultiHouseEnvV2(load_scenarios(), seed=3)
    o, _ = e13.reset(options={"scenario_index": 4})
    for _ in range(60):
        o, _, te, tr, _ = e13.step(int(np.random.default_rng(len(obs)).choice(np.flatnonzero(e13.action_masks()))))
        obs.append(o)
        if te or tr:
            break
    return torch.as_tensor(np.stack(obs))


def main():
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    t_all = time.time()
    R = {}
    # ---------------- 1. hybrid + migration + equivalence
    v2 = MaskablePPO.load(CK_MIG, device="cpu")
    env = MultiHouseEnvV2(load_scenarios(), seed=0)
    v3 = MaskablePPO(SpatialMaskablePolicyV3, env, device="cpu", n_steps=512, batch_size=256, n_epochs=4, target_kl=0.03, learning_rate=3e-4, ent_coef=0.01, gamma=0.99, gae_lambda=0.95, seed=SEED)
    sd, rep = migrate_v2_to_v3(v2.policy.state_dict(), v3.policy.state_dict())
    v3.policy.load_state_dict(sd, strict=True)
    obs = saved_observations()
    with torch.no_grad():
        l2, val2 = v2.policy.mlp_extractor(obs)
        l3, val3 = v3.policy.mlp_extractor(obs)
    R["migration"] = {"source": CK_MIG, "source_sha256": hashlib.sha256(open(CK_MIG, "rb").read()).hexdigest(), **rep, "params_v2": sum(p.numel() for p in v2.policy.parameters()), "params_v3": sum(p.numel() for p in v3.policy.parameters()),
                      "equivalence": {"observations": int(obs.shape[0]), "max_abs_logit_diff": float((l2 - l3).abs().max()), "max_abs_value_diff": float((val2 - val3).abs().max()), "same_argmax": int((l2.argmax(1) == l3.argmax(1)).sum())}}
    path = os.path.join(OUT, "s8c_migrated_v3_attention_step0.zip")
    v3.save(path)
    R["migration"]["saved"] = path
    print("MIGRATION", json.dumps(R["migration"]), flush=True)
    # ---------------- 2. attention checks on a randomly initialised (non-zero) attention block
    ex = SpatialExtractorV3()
    ex.load_state_dict(v3.policy.mlp_extractor.state_dict())
    torch.nn.init.normal_(ex.attn.out.weight, std=0.05)
    x = obs[:1].clone().requires_grad_(True)
    fm = ex.features(x)
    lg = ex.tile_logits(fm).reshape(L.H, L.W, 4)
    lg[2, 2].sum().backward()
    g = x.grad[0, : L.N_TILES * V.NUM_TILE_FEATURES_V2].reshape(L.H, L.W, V.NUM_TILE_FEATURES_V2).abs()
    far = float(g[18:, 30:].sum())
    grp = float(g[:, :, L.NUM_TILE_FEATURES:].sum())
    gp = {n: float(p.grad.abs().sum()) if p.grad is not None else 0.0 for n, p in ex.attn.named_parameters()}
    ex.use_attention = False
    with torch.no_grad():
        lg_off = ex.tile_logits(ex.features(obs[:4]))
    ex.use_attention = True
    with torch.no_grad():
        lg_on = ex.tile_logits(ex.features(obs[:4]))
    R["attention_checks"] = {"1_gradient_from_tile(2,2)_logit_to_tiles_rows>=18_cols>=30": far, "2_logits_change_when_residual_enabled(max_abs)": float((lg_on - lg_off).abs().max()), "3_grad_abs_sum_attention_params": gp,
                             "4_gradient_to_group_channels_through_attention": grp, "5_B_and_C_share_code_and_init": "SupervisedTileNet(blind=True) zeroes channels 15-20 before the trunk; same seed, same extractor construction",
                             "7_inputs": "forward() receives only the v2 observation vector; labels/masks are used by the loss and argmax only"}
    print("CHECKS", json.dumps(R["attention_checks"]), flush=True)
    # ---------------- 3. supervised benchmark (cached S16 data)
    tr_meta, tr_obs, tr_mask, tr_tgt, tr_oth = S16.load("train")
    keep = torch.as_tensor([m["query_slot"] != S16.HOLDOUT_SLOT for m in tr_meta])
    tr_obs, tr_mask, tr_tgt = tr_obs[keep], tr_mask[keep], tr_tgt[keep]
    va, te = S16.load("val"), S16.load("test")
    models = {}
    base_sd = v3.policy.mlp_extractor.state_dict()
    for name in "ABC":
        torch.manual_seed(SEED)
        ext = SpatialExtractorV3(use_attention=(name != "A"))
        ext.load_state_dict(base_sd)                             # pretrained trunk + tile head (S8-C), attention out-proj = 0
        net = SupervisedTileNet(ext, blind=(name == "C"))
        n_par = sum(p.numel() for n_, p in net.named_parameters() if name != "A" or not n_.startswith("ex.attn"))
        opt = torch.optim.Adam([p for n_, p in net.named_parameters() if name != "A" or not n_.startswith("ex.attn")], lr=LR)
        curve, ep, best = [], 0, (-1.0, None)
        t0, m0 = time.time(), rss_mb()
        while ep < EPOCHS and time.time() - t0 < TCAP:
            net.train()
            perm = torch.randperm(len(tr_obs))
            tl = 0.0
            for i in range(0, len(perm), BATCH):
                j = perm[i:i + BATCH]
                loss = S16.set_loss(net(tr_obs[j]), tr_mask[j], tr_tgt[j])
                opt.zero_grad(); loss.backward(); opt.step()
                tl += float(loss.detach()) * len(j)
            ep += 1
            mv, _ = S16.metrics(va[0], S16.predict(net, va[1], va[2]), va[3], va[4])
            curve.append({"epoch": ep, "train_loss": round(tl / len(perm), 4), "val_acc": round(mv["all"]["valid_set_acc"], 4), "val_cd_acc": round(mv["colour_dependent"]["valid_set_acc"], 4), "val_cf_pair_acc": mv["cf_pairs_both_correct"]["acc"], "s": round(time.time() - t0, 1)})
            if mv["all"]["valid_set_acc"] >= best[0]:
                best = (mv["all"]["valid_set_acc"], {k: v.clone() for k, v in net.state_dict().items()})
        train_s = time.time() - t0
        net.load_state_dict(best[1])
        torch.save(net.state_dict(), os.path.join(OUT, f"model_{name}.pt"))
        mt, hit = S16.metrics(te[0], S16.predict(net, te[1], te[2]), te[3], te[4])
        models[name] = {"net": net, "hit": hit}
        R.setdefault("models", {})[name] = {"params_trained": n_par, "epochs": ep, "train_s": round(train_s, 1), "train_samples_per_s": round(ep * len(tr_obs) / train_s, 1), "rss_mb_before_after": [m0, rss_mb()], "best_val_acc": best[0], "curve": curve, "test": mt}
        print(name, json.dumps({k: v for k, v in R["models"][name].items() if k != "curve"}), flush=True)
    # ---------------- 4. ablations of the trained attention model (inference only)
    netB = models["B"]["net"]
    abl = {}
    for tag, att, blind in (("normal", True, False), ("attention_residual_disabled", False, False), ("group_channels_zeroed", True, True)):
        netB.ex.use_attention, netB.blind = att, blind
        mt, hit = S16.metrics(te[0], S16.predict(netB, te[1], te[2]), te[3], te[4])
        abl[tag] = {"test": {k: mt[k] for k in ("all", "colour_dependent", "beyond_rf", "unseen_slot", "cf_twins", "cf_pairs_both_correct")}, "hit": hit}
    netB.ex.use_attention, netB.blind = True, False
    # CF swaps change the selected target? (check 6)
    pB = S16.predict(netB, te[1], te[2])
    pairs = {}
    for i, m in enumerate(te[0]):
        if m["colour_dependent"]:
            pairs.setdefault(m["geom"], []).append(i)
    changed = sum(1 for v in pairs.values() if len(v) == 2 and int(pB[v[0]]) != int(pB[v[1]]))
    R["ablations"] = {k: v["test"] for k, v in abl.items()}
    R["attention_checks"]["6_cf_pairs_with_different_prediction_B"] = [changed, len(pairs)]
    meta = te[0]
    sub = {"all": np.arange(len(meta)), "colour_dependent": np.array([i for i, m in enumerate(meta) if m["colour_dependent"]]), "beyond_rf": np.array([i for i, m in enumerate(meta) if not m["within_rf"]]), "unseen_slot": np.array([i for i, m in enumerate(meta) if m["query_slot"] == S16.HOLDOUT_SLOT])}
    H = {k: v["hit"] for k, v in models.items()}
    R["paired"] = {f"{a}_minus_{b}": {k: S16.paired_bootstrap(H[a][v], H[b][v]) for k, v in sub.items()} for a, b in (("B", "A"), ("B", "C"), ("A", "C"))}
    R["paired"]["B_minus_B_no_attention"] = {k: S16.paired_bootstrap(H["B"][v], abl["attention_residual_disabled"]["hit"][v]) for k, v in sub.items()}
    R["paired"]["B_minus_B_groups_zeroed"] = {k: S16.paired_bootstrap(H["B"][v], abl["group_channels_zeroed"]["hit"][v]) for k, v in sub.items()}
    # failure examples (B, colour-dependent)
    R["failure_examples_B"] = [{"geom": m["geom"], "cf": m["cf"], "query": m["query"], "slots": [m["query_slot"], m["decoy_slot"]], "within_rf": m["within_rf"], "costs": [m["cost_correct"], m["cost_decoy"]], "pred": L.decode_action(int(pB[i]) + 1), "valid": [L.decode_action(a) for a in m["valid"]],
                                "chose_decoy": bool(te[4][i, pB[i]])} for i, m in enumerate(meta) if not H["B"][i] and m["colour_dependent"]][:6]
    # ---------------- 5. cost: inference latency (policy forward, batch 1 and 64), peak memory
    with torch.no_grad():
        for tag, pol in (("v2_cnn", v2.policy.mlp_extractor), ("v3_cnn_attention", v3.policy.mlp_extractor)):
            for b in (1, 64):
                xb = obs[:b]
                pol(xb)
                t0 = time.time()
                for _ in range(20):
                    pol(xb)
                R.setdefault("cost", {})[f"{tag}_batch{b}_ms_per_obs"] = round((time.time() - t0) / 20 / b * 1000, 3)
    R["cost"]["rss_mb_end"] = rss_mb()
    R["cost"]["train_throughput_samples_per_s"] = {k: R["models"][k]["train_samples_per_s"] for k in "ABC"}
    R["compute_s_total"] = round(time.time() - t_all, 1)
    json.dump(R, open(os.path.join(OUT, "results.json"), "w"), indent=1, default=str)
    print("ABLATIONS", json.dumps(R["ablations"]))
    print("PAIRED", json.dumps(R["paired"]))
    print("COST", json.dumps(R["cost"]), "TOTAL", R["compute_s_total"])


if __name__ == "__main__":
    main()
