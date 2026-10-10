"""S17: re-verify zero-residual equivalence after the ortho-init fix; re-save the migrated v3 checkpoint; retrain model B from the TRUE zero-residual init (same procedure) as an init-sensitivity check."""
import json, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python")); sys.path.insert(0, os.path.join(ROOT, "experiments"))
import numpy as np, torch
from sb3_contrib import MaskablePPO
from synthetic_rl.models_v2 import SpatialMaskablePolicyV2  # noqa
from synthetic_rl.models_v3 import SpatialMaskablePolicyV3, SpatialExtractorV3, SupervisedTileNet, migrate_v2_to_v3
from synthetic_rl.s14_env import MultiHouseEnvV2
from synthetic_rl.s13_env import load_scenarios
import s16_train as S16, s17_benchmark as B17
torch.set_num_threads(4); torch.manual_seed(17); t_all = time.time()
v2 = MaskablePPO.load(B17.CK_MIG, device="cpu")
v3 = MaskablePPO(SpatialMaskablePolicyV3, MultiHouseEnvV2(load_scenarios(), seed=0), device="cpu", n_steps=512, batch_size=256, n_epochs=4, target_kl=0.03, learning_rate=3e-4, ent_coef=0.01, gamma=0.99, gae_lambda=0.95, seed=17)
sd, rep = migrate_v2_to_v3(v2.policy.state_dict(), v3.policy.state_dict()); v3.policy.load_state_dict(sd, strict=True)
obs = B17.saved_observations()
with torch.no_grad():
    l2, val2 = v2.policy.mlp_extractor(obs); l3, val3 = v3.policy.mlp_extractor(obs)
eq = {"observations": int(obs.shape[0]), "max_abs_logit_diff": float((l2 - l3).abs().max()), "max_abs_value_diff": float((val2 - val3).abs().max()), "same_argmax": int((l2.argmax(1) == l3.argmax(1)).sum()), "out_proj_abs_sum": float(v3.policy.mlp_extractor.attn.out.weight.abs().sum())}
v3.save(os.path.join(B17.OUT, "s8c_migrated_v3_attention_step0.zip")); print("EQUIVALENCE_FIXED", json.dumps(eq), flush=True)
# retrain B from zero residual
tr_meta, tr_obs, tr_mask, tr_tgt, _ = S16.load("train"); keep = torch.as_tensor([m["query_slot"] != S16.HOLDOUT_SLOT for m in tr_meta]); tr_obs, tr_mask, tr_tgt = tr_obs[keep], tr_mask[keep], tr_tgt[keep]
va, te = S16.load("val"), S16.load("test")
torch.manual_seed(17); ext = SpatialExtractorV3(); ext.load_state_dict(v3.policy.mlp_extractor.state_dict()); net = SupervisedTileNet(ext)
opt = torch.optim.Adam(net.parameters(), lr=B17.LR); t0 = time.time(); ep = 0; best = (-1, None); curve = []
while ep < B17.EPOCHS and time.time() - t0 < B17.TCAP:
    net.train(); perm = torch.randperm(len(tr_obs)); tl = 0.0
    for i in range(0, len(perm), B17.BATCH):
        j = perm[i:i + B17.BATCH]; loss = S16.set_loss(net(tr_obs[j]), tr_mask[j], tr_tgt[j]); opt.zero_grad(); loss.backward(); opt.step(); tl += float(loss.detach()) * len(j)
    ep += 1; mv, _ = S16.metrics(va[0], S16.predict(net, va[1], va[2]), va[3], va[4])
    curve.append({"epoch": ep, "train_loss": round(tl / len(perm), 4), "val_acc": round(mv["all"]["valid_set_acc"], 4), "val_cd_acc": round(mv["colour_dependent"]["valid_set_acc"], 4), "s": round(time.time() - t0, 1)})
    if mv["all"]["valid_set_acc"] >= best[0]: best = (mv["all"]["valid_set_acc"], {k: v.clone() for k, v in net.state_dict().items()})
net.load_state_dict(best[1]); torch.save(net.state_dict(), os.path.join(B17.OUT, "model_B_zero_init.pt"))
mt, hit = S16.metrics(te[0], S16.predict(net, te[1], te[2]), te[3], te[4])
R = json.load(open(os.path.join(B17.OUT, "results.json")))
R["migration"]["equivalence_after_ortho_init_fix"] = eq
R["models"]["B_zero_residual_init"] = {"epochs": ep, "train_s": round(time.time() - t0, 1), "best_val_acc": best[0], "curve": curve, "test": mt, "note": "same procedure as B but attention output projection truly zero at init (the first benchmark's B/C started from SB3's orthogonal init of that projection; trunks pretrained in both)"}
R["compute_s_total_incl_fix"] = R["compute_s_total"] + round(time.time() - t_all, 1)
json.dump(R, open(os.path.join(B17.OUT, "results.json"), "w"), indent=1, default=str)
print("B_ZERO", json.dumps({k: v for k, v in R["models"]["B_zero_residual_init"].items() if k != "curve"}), "TOTAL", R["compute_s_total_incl_fix"], flush=True)
