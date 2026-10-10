# Native full-game CNN-vs-MLP pilot — PLAN ONLY (nothing here has been run)

Status: design document awaiting approval. No native training/evaluation, game, reward, feature or S1/S2 change was made to produce it.

## 0. Facts this plan rests on
* Synthetic evidence: spatial CNN ≫ flat MLP on Gap-1 (CNN ~90% vs MLP ~7% held-out); zero-shot native Gap-1 26–28/29 (CNN) vs 2/29 (frozen MLP) vs 4/29 RANDOM; Gap-2 zero-shot up to 88.5% but via many attempts (`S3_EFFICIENCY.md`: only 26% within 2 ADDs for the best checkpoint).
* Frozen native references (`research/frozen_baselines/`, 10 held-out EVAL_SEEDS, 1 s decisions, 600-decision horizon, reward = native score delta, terminal −10): WAIT on NORMAL_START −10.0 (final score 0, 143 decisions); RANDOM −9.7 (score 0.3); native-reward MLP (trained on CONNECTED_START, 49,152 decisions) −9.0 on NORMAL; connection-shaped MLP −10.0. In normal starts nothing yet scores, so reward is nearly flat at the bottom: **survival length and any delivery are the only sensitive signals at small budgets.**
* Native throughput measured so far: ~3.6 decisions/s (single-seed Gap-1 pilot incl. one restart), ~4–5 each with three parallel instances. Game-bound; the CNN trainer adds only a few ms/decision. The game hangs when many instances run in parallel → ≤3 at a time.
* Interface (S2): 0 encoding mismatches over 156 live states; obs Box(11,666), Discrete(3,111).

## 1. Hypothesis and measurable outcomes
**H-N1 (primary).** Starting PPO from the synthetic-pretrained spatial CNN yields better native NORMAL_START play after a small, fixed budget than starting from random initialisation (CNN or MLP).
**H-N2 (architecture, compute-matched).** At identical budget/hyperparameters/seeds, a randomly initialised spatial CNN learns faster than a randomly initialised MLP.
**Null results are informative:** at 10k decisions the reward may be indistinguishable from WAIT/RANDOM for all arms; that is a "budget/signal too small" result, not evidence against spatial bias.

Outcomes (all on the 10 held-out EVAL_SEEDS at the end; 6 VAL_SEEDS for the mid-training curve; paired by seed against WAIT and RANDOM):
| Metric | Definition / source |
|---|---|
| Native score / deliveries | episode `native_score_total` (sum of `rewardComponents.scoreDelta`); final game score |
| Return | native score delta + terminal penalty, as in the frozen table |
| Survival | decisions until native game over (cap 600) |
| Useful connections | number of house→destination connections present, computed from the raw observation by the existing connectivity routine (the one used by `--shaping`, run with coefficient 0 as a *measurement only*) — **to be verified that it can be logged without touching reward** |
| Action efficiency | ADDs per delivery; fraction of ADDs that create/extend a connection to a destination; irrelevant-ADD rate |
| Road churn | REMOVE count, REMOVE of own recent ADD (within 20 decisions), connections destroyed (already in `eval_table.txt`) |
| Action mix | WAIT/ADD/REMOVE/UPGRADE fractions |
| Training diagnostics | entropy, approx_kl, clip fraction, explained variance, SPS (harness `metrics/training.csv`) |

## 2. Arms and controls (same env, reward, termination, hyperparameters, budget, seeds)
| Arm | Init | Network | Role |
|---|---|---|---|
| R0 | frozen 49,152-decision native-reward MLP (SHA `96278bb3…`) and shaped MLP (`187099a4…`) | MLP | **Historical reference only** — different budget, CONNECTED_START / shaping, different training conditions. Re-evaluated, not retrained. Not a compute-matched control. |
| A1 | random | MlpPolicy (same as S1 MLP arm) | compute-matched MLP control |
| A2 | random | SpatialMaskablePolicy (73,416 params) | compute-matched spatial CNN, scratch |
| A3 | synthetic-pretrained S1 CNN seed k (selected checkpoints: s11 step_050000, s12 step_050000, s13 final) | SpatialMaskablePolicy | pretraining effect (A3 vs A2 isolates pretraining; A2 vs A1 isolates architecture) |
| W / Rnd | – | – | WAIT and RANDOM LEGAL on the same seeds (existing frozen WAIT/RANDOM evals reusable after verifying the same interval/horizon; otherwise re-run, cheap) |

Seed pairing: run seed k uses S1 checkpoint k for A3; A1/A2 use the same PPO seed k. Pilot seeds: 11 and 12 (2 seeds ×3 arms = 6 runs); expand to 13 only on a go.
Fixed for all arms: NORMAL_START, decision interval 1.0 s, horizon 600 decisions (matching the frozen evals; the stock defaults in `eval_heldout.yaml` are 0.3 s / 30 — **always pass explicit values**), native reward only (`reward: native_score`), PPO n_steps 512, batch 256, n_epochs 4, target_kl 0.03, lr 3e-4 constant, γ 0.99, GAE 0.95, clip 0.2, ent_coef 0.01. No privileged inputs: only PolicyObservation and the native legal mask.
Pretrained handling (documented, deterministic): policy weights copied; optimizer re-initialised (S1 optimizer state not carried); timestep counter restarts at 0; learning-rate is constant so there is no schedule difference; the VecNormalize-free pipeline means no running statistics to restore.

## 3. Compatibility of the existing CNN checkpoint with the native pipeline (investigation only)
| Item | Finding | Action |
|---|---|---|
| Observation/action spaces | Identical (Box(11,666), Discrete(3,111)); S2 loaded all three S1 CNNs into the native env and ran them zero-shot | none |
| Loading for *inference* | Works (S2) | none |
| Loading for *training* | `python/train_ppo.py` only builds `"MlpPolicy"` (`--net mlp`) or `GridExtractor` (`--net cnn`, a *different* strided CNN with a 256-unit FC head, not the S1 `SpatialMaskablePolicy`). The harness (`experiments/config.py`) only allows `policy.network ∈ {mlp, cnn}` and has no init-from-checkpoint option. | **Prerequisite P1 (needs approval):** add `--net spatial` and `--init-checkpoint` to `train_ppo.py`/harness (plain Python training code; no game, reward or observation change). Without it, "CNN" in the existing pipeline would be the wrong architecture. |
| Custom class on load | `SpatialMaskablePolicy` must be importable (`synthetic_rl.models`); add to `python/` path | P1 |
| Carpark/Tree encoded as 0 | Python encoder omits Carpark; present in the pretrained policy's training data identically (synthetic also encodes it as 0), so semantics match. 29% / 40% of native Gap-1 val/test tasks are ambiguous; did not hurt Gap-1 zero-shot. | **Not fixed**, per instruction. Record as a risk. |
| Water 16% of native tiles, bridges global, trees, vehicles/traffic, multiple colours/demand, UPGRADE slots, REMOVE-on-carpark-stub | Never exercised in synthetic training (water 0%, bridges 0, WAIT-only continuation, upgrade logits untrained) | confounder, see §6 |
| Value head | Pretrained on sparse {0,+1} returns; native returns include −10 terminal | confounder; no fix in pilot |
| Start state | Synthetic episodes always start with a nearly-built path with gaps; NORMAL_START has an empty road map | the pretrained prior ("close the gap") has nothing to close at t=0 |
| CNN compute cost | negligible vs the game (~250 ms/decision) | none |

## 4. Protocol
1. **Gate 0 (no training, minutes):** after P1 only: load A3 weights into the native trainer; confirm identical action logits vs the standalone S1 checkpoint on the 156 saved S2 native states (`native_states_validation_0of1.jsonl.gz`, no game needed); confirm A2/A1 build; run the 512-step harness smoke test; verify the connectivity metric logs with reward unchanged (episode reward bit-identical with metric on/off on one seed).
2. **Train** each of A1–A3 for 10,240 decisions (20 PPO rollouts of 512) per seed; checkpoints at 0, 5,120, 10,240; ≤3 game instances concurrently (one seed batch = three arms).
3. **Evaluate** every checkpoint deterministically on the 6 VAL_SEEDS (curve; stochastic only for the final checkpoint). The 10 EVAL_SEEDS (TEST) are evaluated **once**, on the final checkpoint of each run, plus W, Rnd, R0.
4. Report per-seed (no averaging away seed variation), paired differences to WAIT/RANDOM and between arms with bootstrap CIs, action mix, churn; plus all negative/flat outcomes.
5. Preserve raw rollouts, configs, checkpoint SHA-256, source hash; separate output dir `runs/…` and `research/native_cnn_pilot/`.

## 5. Compute budget and wall time (estimates from measured SPS, not promises)
* Training: 6 runs × 10,240 decisions ≈ 61k decisions. At ~4 SPS per instance: ~43 min per run; in 3 parallel lanes ≈ 45–60 min per seed batch → **~1.5–2 h for both seeds**.
* Validation evals: 6 VAL_SEEDS × 3 checkpoints × 6 runs; episodes last ~140 decisions if the agent dies early, up to 600: ~2.5k–11k decisions ≈ 10–45 min of 3-lane time. TEST: 6 runs + R0 ×(10 seeds × 140–600 decisions) ≈ 10–45 min.
* **Expected total ≈ 3–4.5 h wall-clock** (upper bound ~5 h if agents survive long). Smoke/Gate 0: <15 min. Decisions/sec to be reported from the harness `training.csv`.
* The historical ~49k-decision runs would be ~3.4 h each at 4 SPS; the pilot is 1/5 of that on purpose.

## 6. Risks and confounders
* **Reward is flat in NORMAL_START** (WAIT −10, RANDOM −9.7): 10k decisions may be too few to move return for any arm. Mitigation: survival/connection metrics and a pre-declared "inconclusive" outcome.
* Pretraining distribution mismatch (water, traffic, multi-destination, value scale) may make A3 *worse* than A2 (negative transfer) or let PPO erase the prior within a few updates.
* Pretrained policy never learned to start from empty roads (no gaps to fill).
* Fresh optimizer on a pretrained net: early updates with large-magnitude value errors can destabilise features; monitored via approx_kl/entropy gates, not fixed.
* Seeds: 2 seeds cannot support population claims; native episodes are not deterministic across restarts (known restore divergence), so CIs reflect eval-seed paired variance only.
* Historical R0 is not compute-matched; do not report A-arms vs R0 as an architecture comparison.
* Carpark omission is shared by every arm (not a differentiator), but may limit all native ADD precision.
* Process fragility: game hangs under parallel load; supervisor restarts; budget accounting after restart (use remaining-budget fix already in `train_ppo.py`).
* Action-efficiency risk: the Gap-2 evidence shows 3–5 irrelevant ADDs per success; in the real game roads cost resources and cause churn.

## 7. Go / no-go criteria (pre-registered)
**Stop immediately (no-go) if:** NaN/inf; approx_kl above 0.15 for 3 consecutive updates; policy entropy < 10% of its step-0 value; REMOVE+UPGRADE > 60% of actions with no connection gain; ≥2 supervisor restarts in one run; Gate 0 logits mismatch the S1 checkpoint.
**Interpretation after 10,240 decisions (both seeds, TEST seeds, paired):**
* *Go to a larger run (e.g. 3 seeds × 49k, same arms)* if A3 beats A2 and A1 on survival (≥ +20% mean decisions) **or** has >0 deliveries on more seeds than WAIT/RANDOM, in both pilot seeds, with the paired bootstrap CI of the survival difference excluding 0 when pooled over seeds, and no stop condition tripped.
* *Architecture-only go* if A2 beats A1 on the same criterion even though A3 does not.
* *Inconclusive* (do **not** extend automatically) if all arms are within WAIT/RANDOM noise: report; options for the user (longer budget; a gap/connection curriculum in native normal starts; observation fix for carpark) are then a new decision.
* *No-go for pretraining* if A3 is worse than A2 in both seeds.

## 8. Prerequisites that require your approval (none done)
P1 `--net spatial` + `--init-checkpoint` in `train_ppo.py` and harness schema (training code only).
P2 measurement-only logging of connections/churn in `RecorderEnv` (no env/reward change).
P3 configs below.

## 9. Exact commands (proposed; to be created/run only after approval)
Configs (to be added under `experiments/configs/`, differing only in `policy.network` and `policy.init_checkpoint`):
```yaml
# native_pilot_{mlp,cnn_scratch,cnn_pretrained}.yaml (shared fields)
environment: {initialization: normal, reward: native_score, decision_interval_seconds: 1.0, episode_horizon_decisions: 600}
policy: {algorithm: maskable_ppo, network: mlp|spatial, init_checkpoint: null|runs/2026-10-08_140406_syn_s1_cnn_s{seed}/checkpoints/<selected>.zip,
         hyperparameters: {n_steps: 512, batch_size: 256, n_epochs: 4, target_kl: 0.03, learning_rate: 3.0e-4, ent_coef: 0.01, gamma: 0.99}}
training: {total_timesteps: 10240, seeds: [11, 12], checkpoint_every: 5120, eval_every: 5120}
evaluation: {seeds: [182150, 205907, 229664, 253421, 261340, 269259], episodes_per_seed: 1, deterministic: true}   # VAL during training
```
```bash
# Gate 0 (no training)
python -m experiments.run --config experiments/configs/native_pilot_cnn_pretrained.yaml --dry-run
python experiments/s3_gate0_logit_check.py --pretrained runs/2026-10-08_140406_syn_s1_cnn_s11/checkpoints/step_050000.zip
python -m experiments.run --config experiments/configs/smoke_test.yaml
# Training, one seed batch at a time (3 lanes); repeat for seed 12 by changing the seeds list
python -m experiments.run --config experiments/configs/native_pilot_mlp.yaml --no-eval
python -m experiments.run --config experiments/configs/native_pilot_cnn_scratch.yaml --no-eval
python -m experiments.run --config experiments/configs/native_pilot_cnn_pretrained.yaml --no-eval
# Validation curve, then TEST once on final checkpoints
python -m experiments.evaluate --run runs/<run_id> --config experiments/configs/eval_native_pilot_val.yaml --which all
python -m experiments.evaluate --run runs/<run_id> --config experiments/configs/eval_native_pilot_test.yaml --which final
python -m experiments.evaluate --policy wait   --config experiments/configs/eval_native_pilot_test.yaml
python -m experiments.evaluate --policy random --config experiments/configs/eval_native_pilot_test.yaml
python -m experiments.compare runs/<mlp> runs/<cnn_scratch> runs/<cnn_pretrained>
```
(`experiments/s3_gate0_logit_check.py`, the three YAMLs and the two eval YAMLs do not exist yet; they are part of P1–P3. The harness `--no-eval` flag and `--which` options are as documented in `research/EXPERIMENT_HARNESS.md`.)
