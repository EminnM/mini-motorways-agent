# S4 — Native CNN integration, Gate 0, and learning-signal diagnostic

**Recommendation: INCONCLUSIVE** for the 10,240-decision-per-arm PPO pilot (details in §7). No PPO training, optimizer step, reward change, observation change, carpark change, synthetic retraining or Gap-3 work was done. Native game code and S1/S2/Baseline-0 artifacts are unmodified (S1 checkpoint SHA-256 re-verified).
Raw data: `research/native_cnn_gate0/` (`gate0_offline.json`, `gate0_native.json`, `diag_*.jsonl`, `diagnostic_summary.json`, `init_report_s*.json`, logs).

## 1. P1 / P2 / P3 implementation
**P1 — spatial CNN in the native pipeline** (`python/train_ppo.py`; originals saved in `research/s3_gap2/pre_S4_backup/`)
* `--net spatial` builds `MaskablePPO(SpatialMaskablePolicy, …)` — the exact S1 class (`synthetic_rl/models.py`, 16 tensors, 73,416 parameters), same PPO hyperparameters, `device=cpu`. `--net mlp` / `--net cnn` construction is unchanged (factored into `build_model`).
* `--init-checkpoint PATH [--init-checkpoint-sha256 H]` → `init_from_checkpoint`: strict. Raises on SHA mismatch, wrong policy class, observation/action-space mismatch, parameter-name or shape mismatch, or use with a non-spatial net; verifies bitwise equality after load; writes `init_report.json`.
* **Loaded:** all 16 policy+value tensors (conv trunk, tile head, global branch, wait/upgrade head, value head). **Reset/fresh:** Adam optimizer (no moments, verified empty), learning-rate schedule (constant `--learning-rate`, 3e-4; S1 also used a constant rate, so no schedule difference), `num_timesteps=0`, rollout buffer, logger, RNG state. **Not inherited:** S1 training hyperparameters (the new run's args apply). **Incompatible:** none for S1 checkpoints; anything else fails loudly.
* Harness: `policy.network ∈ {mlp, cnn, spatial}`; `policy.init_checkpoint` / `init_checkpoint_sha256` (string with `{seed}` or per-seed map; validated per training seed) → `trainer_argv`.

**P2 — measurement-only diagnostics** (`experiments/diagnostics.py`, wired through `RecorderEnv`, `EpisodeAccumulator`, `evaluation.run_policy`; off unless `logging.diagnostics: true`)
Per episode: deliveries (positive score events; event sizes recorded), first-delivery step, survival, ADD/REMOVE counts and accepted counts, distinct edges added, repeated edits, REMOVE-of-own-ADD / re-ADD churn, useful connections (max/final, first-connection step, connection→first-delivery latency, increments/decrements; uses the raw observation's `groupIndex` and the existing `--shaping` potential function with **no reward effect**), road-edge counts, legal-action counts (total/ADD/REMOVE), action-index counts and empirical action entropy, masked policy entropy (evaluation loops, read-only forward pass), count/timing of nonzero rewards split into intermediate vs terminal. Exceptions inside diagnostics are swallowed and counted (`diag_errors`).

**P3 — configs** (`experiments/configs/`, validated with `--dry-run`, **not executed**): `native_pilot_a1_mlp.yaml`, `native_pilot_a2_cnn_scratch.yaml`, `native_pilot_a3_cnn_pretrained.yaml`, plus `eval_native_pilot_val.yaml` / `_test.yaml`. Identical except network/init: NORMAL_START, `native_score`, 1.0 s decisions, 600-decision horizon (the values used by the frozen baselines' runner), n_steps 512, batch 256, n_epochs 4, target_kl 0.03, lr 3e-4, ent_coef 0.01, γ 0.99, 10,240 decisions, seeds 11 & 12, checkpoints at 5,120/10,240, validation seeds for curves. A3 maps seed→selected S1 checkpoint with SHA-256.

## 2. Gate 0 — numerical checks (all PASS)
120 raw native states (the S2 interface check covered 156; the other 36 post-action states had no persisted raw observation, so 120 are used), legal counts 121–176.

| Check | Result |
|---|---|
| New pipeline (`build_model`+`init_from_checkpoint`) vs original S1 inference (`MaskablePPO.load`), seeds 11/12/13 | max abs diff **0** for raw logits, values and masked probabilities (tolerance 1e-6); deterministic actions equal **120/120** for each; all chosen actions legal; probability mass on illegal actions 0; 16 tensors/73,416 params bitwise-equal |
| Optimizer / schedule after init | Adam state empty, lr 3e-4, `num_timesteps` 0 |
| Fail loudly | wrong SHA, MLP checkpoint into spatial net, `--init` with `--net mlp`: all raise `ValueError` |
| Existing MLP/CNN paths | `build_model('mlp'/'cnn', seed 7)` initial weights bitwise-equal to the pre-S4 construction; frozen MLP checkpoint loads, strict-loads into a freshly built MLP with identical logits; its 120 actions all legal |
| Random spatial CNN fwd/bwd on 32 saved native states | loss finite, 16/16 grads finite and non-zero, entropy 4.97 nats, weights unchanged (no optimizer step), ~129 ms per 32-state fwd+bwd |
| S1 checkpoints unmodified | SHA-256 equal to recorded values |

(The zero differences are expected — the weights are copied — so this test proves the loading/forward path, not new behaviour.)

## 3. Logging non-interference (native, paired deterministic rollouts) — PASS
3 validation seeds × 60 decisions, frozen S1 CNN argmax policy, four runs per seed (plain env / RecorderEnv + diagnostics / plain again / RecorderEnv without diagnostics): observation hashes, legal-mask hashes, actions, rewards, termination flags and applied actions **identical across all four** (native determinism floor, plain vs plain, also identical). Per-step cost with diagnostics ≈ +3–8 ms on a ~45–60 ms env step. Caveat: none of those 60-step rollouts ended an episode, so that run flushed no episode rows; the diagnostic code's end-of-episode path was exercised by Part 5 (20 episodes, `diag_errors = 0`), where logging could not be compared against a diagnostics-off twin.

## 4. Native inference-only diagnostic (no gradients/optimizer/checkpoint writes)
NORMAL_START, native reward, 1.0 s decisions, 600 horizon, 4 common **validation** seeds (182150, 205907, 229664, 253421), 4 game instances in parallel. Frozen native MLP = `native_reward_connected_49152`; CNN = S1 seed 13 (best S1 seed), deterministic and stochastic. Wall-clock **282 s** total (cap 30 min).

| Policy | ≥1 delivery | nonzero intermediate reward | ≥1 useful connection | survival (decisions) | return (mean) | ADD / REMOVE / WAIT share | distinct edges added | churn | masked policy entropy |
|---|---|---|---|---|---|---|---|---|---|
| WAIT | 0/4 | 0/4 | 0/4 | 143,142,153,144 | −7.5 | 0 / 0 / 99% | 0 | 0 | – |
| RANDOM LEGAL | 0/4 | 0/4 | 0/4 | 146,130,154,137 | −5.0 | 67 / 29 / 3% | 71 | 42 | – |
| frozen native MLP | 0/4 | 0/4 | 0/4 | 146,129,153,144 | −10.0 | 52 / 48 / 0% | 20 | 81 | 4.4 nats |
| S1 CNN, deterministic | 0/4 | 0/4 | 0/4 | 143,142,153,144 | −7.5 | 93 / 0 / 6% | 13 | 0 | 2.7 nats |
| S1 CNN, stochastic | **1/4** | **1/4** | **1/4** | 143,142,153,**329** | +2.75 | 98 / 0.1 / 1% | 51 | 0 | 1.6 nats |

Details
* **The one positive episode** (CNN stochastic, seed 253421): first useful connection after 29 decisions, first delivery at decision 40 (11 decisions later), 41 deliveries (score events of 1 and 2), 38 intermediate nonzero rewards, 8 connection increments / 2 decrements, 81 road edges, survival 329 vs ~145, return +31 (41 − 10). The other three stochastic episodes and every other policy's 16 episodes had no connection and no delivery.
* **Survival is policy-independent unless a delivery happens.** WAIT and the deterministic CNN die at identical decisions on every seed (143/142/153/144): game over is time-driven in these starts; roads placed by RANDOM/MLP shift it by a few decisions only (spawn placement depends on roads).
* **Deterministic CNN gets stuck:** it repeats one ADD up to 67 times per episode (13 distinct edges from ~135 ADDs); the repeated ADD is legal and accepted in the native mask and does not change the state, so argmax repeats forever. The stochastic CNN escapes (51 distinct edges). So deterministic evaluation understates what the policy can do in the real game, and a PPO run (which samples) does not share this failure.
* **Terminal reward quirk (environment/reward implementation issue, observed, not fixed):** 5 of 20 terminated episodes returned 0 instead of −10 for identical outcomes (WAIT seed 182150 and the CNN's same-seed episodes; RANDOM seeds 205907 and 253421). The native STEP handler pays the −10 only when the game turns terminal during that step (`justTerminated`); when the game is already terminal at the start of the next STEP, no penalty is paid. This adds ±10 noise to returns/values that is unrelated to behaviour. Forbidden to change in this phase; recorded as a risk.
* **Speed:** 4 instances in parallel with diagnostics ran at ≈ 5.6 decisions/s (WAIT, no model), 2.4 (MLP), 2.1–2.7 (CNN, including masked-entropy forward pass) per instance.
* `diag_errors = 0` over 20 episodes.

## 5. Which failure is it? (distinguishing A / B / C)
* **A. Policy fails to discover useful construction:** true for 19 of 20 episodes — including a deterministic CNN that builds 7–22 edges and makes no house–destination connection, and a frozen MLP/RANDOM that build 20–71 edges and make none. The only discovery came from the stochastic CNN.
* **B. Useful roads built but reward not received within the horizon:** not observed. Where a connection existed, reward followed within 11 decisions (and then flowed densely: 38 intermediate rewards). The horizon (600) is not the limit; the game ends at ~143 s without a delivery.
* **C. Environment / reward implementation problem:** the missing terminal penalty in 25% of terminations (noise, policy-independent). No evidence that the connection→reward pathway is broken (it paid 41 points when a connection existed). The observation/mask interface was validated in S2/Gate 0.

## 6. Learning-signal assessment
There is a plausible, but thin, policy-dependent signal: a CNN that samples actions discovers a delivering connection in this fresh-map setting (1 of 4 episodes), RANDOM, WAIT, the frozen MLP and the deterministic CNN did not (0 of 16). Once a connection exists, rewards are dense and arrive fast (≈11 decisions, return +31 vs −10), so credit assignment is short. Limits: 4 episodes per policy (Wilson 95% interval for 1/4 is ≈ 5–70%); the historical frozen evaluation found RANDOM earning a reward in 1 of 10 held-out episodes (first reward at decision 105), so the base rate of a lucky delivery is ≈ 10% and 1/4 is not distinguishable from it; 1 positive episode cannot establish that the pretrained prior raises it. Not concluded: whether PPO can learn from this signal; the only statement supported is that the signal is neither absent nor demonstrated to be stronger than chance.

## 7. Recommendation for the proposed 10,240-decision PPO pilot: **INCONCLUSIVE**
* *Not NO-GO:* no interface or pipeline blocker; a reward-bearing episode exists under the pretrained policy; a pilot run would see about 70 episodes per run (512-decision rollouts ≈ 3.5 episodes), so even a 10% positive rate gives several rewarded episodes per run.
* *Not GO:* the evidence that the pretrained CNN produces more reward than RANDOM is one episode in four, indistinguishable from the ≈10% RANDOM base rate seen historically.
* **No additional compute is started.** If you want a decision-relevant, inexpensive measurement before committing ~3–4 h to the pilot, the targeted option is to extend this same inference-only diagnostic to ~16 validation seeds each for RANDOM and the stochastic CNN (≈ 16 × ~200 decisions per policy at ~2.5 decisions/s ≈ 20–30 min with two instances) to estimate the delivery rate per episode with a usable interval. That requires your approval.
* If the pilot is later approved, adjust for what this report found: evaluate **both** deterministic and stochastic policies (deterministic argmax loops on a repeated ADD); treat the missing-terminal-penalty noise as a known confounder; note measured parallel throughput of 2–3 decisions/s per instance (not 4–5), i.e. a 10,240-decision run ≈ 1–1.5 h and the 6-run pilot ≈ 3–4.5 h with 3 lanes; keep the A1/A2/A3 configs unchanged.

## 8. Source hashes after changes (sha256 prefix)
`python/train_ppo.py ac40ffbe`, `experiments/recording.py 9fe4b055`, `config.py c32149ca`, `train_worker.py 2668941d`, `evaluation.py eb4332ff`, `postprocess.py 6750f927`, `diagnostics.py 9b896911`. Pre-S4 originals and hashes: `research/s3_gap2/pre_S4_backup/`, `research/s3_gap2/pre_S4_source_hashes.txt`. Patch record: `experiments/_patch_s4.py`.
Repro: `python experiments/s4_gate0.py offline|native`, `python experiments/s4_signal_diag.py run --policies … --tag …`, `… summarize`.
