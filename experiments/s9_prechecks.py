"""S9 pre-launch checks (no game): checkpoint hashes, strict load into the S4 spatial architecture, spaces, finite masked probabilities, deterministic inference
reproducibility on saved native observations, no training process running.   python experiments/s9_prechecks.py"""
import gzip
import hashlib
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import numpy as np
import torch
OUT = os.path.join(ROOT, "research", "s9_native_transfer")
P1 = ("runs/2026-10-08_140406_syn_s1_cnn_s13/checkpoints/final.zip", "41c1e085622b8f0d2e70274b2a6f05fedc94266124c1c74f9b39645ade417682")
P2 = ("research/s8_long_route_training/runs/arm_C_s11/checkpoints/step_0099840.zip", None)
R = {}


def check(name, ok, detail):
    R[name] = {"passed": bool(ok), "detail": detail}
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)


def sha(p):
    return hashlib.sha256(open(os.path.join(ROOT, p), "rb").read()).hexdigest()


def main():
    import train_ppo
    import mini_motorways_env as M
    from sb3_contrib import MaskablePPO
    sys.path.insert(0, os.path.join(ROOT, "experiments"))
    from s4_gate0 import DummyEnv, pipeline_args
    ft = json.load(open(os.path.join(ROOT, "research", "s8_long_route_training", "runs", "arm_C_s11", "final_test_v2.json")))
    sel = json.load(open(os.path.join(ROOT, "research", "s8_long_route_training", "runs", "arm_C_s11", "selection.json")))
    p2sha = ft["sha256"]
    h1, h2 = sha(P1[0]), sha(P2[0])
    check("1_checkpoint_hashes", h1 == P1[1] and h2 == p2sha and sel["selected"] == "step_0099840.zip",
          f"P1 S1 seed13 final.zip {h1[:16]} == recorded; P2 S8-C selected-by-validation step_0099840.zip {h2[:16]} == recorded S8 TEST hash; S8 selection.json selected={sel['selected']}")
    models, strict = {}, {}
    for name, (p, _) in (("P1", P1), ("P2", P2)):
        m = train_ppo.build_model(pipeline_args("spatial", 11), DummyEnv())
        rep = train_ppo.init_from_checkpoint(m, os.path.join(ROOT, p))
        strict[name] = (rep["n_tensors_loaded"], rep["n_parameters_loaded"])
        models[name] = MaskablePPO.load(os.path.join(ROOT, p), device="cpu")
    check("2_strict_load_into_S4_spatial_architecture", all(v == (16, 73416) for v in strict.values()), f"both checkpoints load strictly (class, spaces, 16 tensors / 73,416 parameters, bitwise): {strict}")
    sp = {n: (tuple(m.observation_space.shape), int(m.action_space.n), float(m.observation_space.low.min()), float(m.observation_space.high.max())) for n, m in models.items()}
    dummy = DummyEnv()
    check("3_observation_and_action_spaces_match", all(v == ((11666,), 3111, -1.0, 1.0) for v in sp.values()) and models["P2"].observation_space == dummy.observation_space,
          f"{sp}; native encoder length {M.NUM_TILE_FEATURES * 777 + M.NUM_GLOBAL_FEATURES}")
    obs_l, mask_l = [], []
    for line in gzip.open(os.path.join(ROOT, "research", "synthetic_to_native", "native_states_validation_0of1.jsonl.gz"), "rt"):
        r = json.loads(line)
        obs_l.append(M.encode_observation(r["raw"]))
        mk = np.zeros(3111, bool)
        mk[r["legal_indices"]] = True
        mask_l.append(mk)
    for line in gzip.open(os.path.join(OUT, "..", "s7_long_route", "native_normal_start_captures.jsonl.gz"), "rt"):
        r = json.loads(line)["first_house"]
        obs_l.append(M.encode_observation({"tiles": r["raw"]["tiles"], "globals": r["raw"]["globals"]}))
        mk = np.zeros(3111, bool)
        mk[r["legal"]] = True
        mask_l.append(mk)
    O, Mk = np.stack(obs_l), np.stack(mask_l)
    fin, det_ok, illegal = {}, {}, {}
    for n, m in models.items():
        with torch.no_grad():
            dist = m.policy.get_distribution(torch.as_tensor(O), action_masks=Mk)
            pr = dist.distribution.probs.numpy()
        fin[n] = bool(np.isfinite(pr).all() and np.allclose(pr.sum(1), 1.0, atol=1e-5))
        illegal[n] = float((pr * (~Mk)).sum(1).max())
        a1 = [int(m.predict(O[i], action_masks=Mk[i], deterministic=True)[0]) for i in range(len(O))]
        m2 = MaskablePPO.load(os.path.join(ROOT, P1[0] if n == "P1" else P2[0]), device="cpu")
        a2 = [int(m2.predict(O[i], action_masks=Mk[i], deterministic=True)[0]) for i in range(len(O))]
        det_ok[n] = (a1 == a2 and all(Mk[i, a] for i, a in enumerate(a1)))
    check("4_masked_probabilities_finite_and_deterministic_inference_reproducible", all(fin.values()) and all(det_ok.values()) and max(illegal.values()) == 0.0,
          f"{len(O)} saved native states (120 Gap-1 + 24 normal-start): probabilities finite and sum to 1 {fin}; max mass on illegal actions {illegal}; deterministic actions identical across two loads and always legal {det_ok}")
    out = subprocess.run(["tasklist"], capture_output=True, text=True).stdout.lower()
    busy = [l.split()[0] for l in out.splitlines() if l.startswith(("python", "mini motorways"))]
    check("5_no_training_or_game_process_running", len(busy) <= 1, f"running python/game processes (this checker counts as 1): {busy}")
    json.dump(R, open(os.path.join(OUT, "prechecks.json"), "w"), indent=1)
    sys.exit(0 if all(v["passed"] for v in R.values()) else 1)


if __name__ == "__main__":
    main()
