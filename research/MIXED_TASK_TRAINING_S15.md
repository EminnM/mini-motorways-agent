# S15 — Mixed-Task PPO: Retention of Long-Route Skill and the Value of Group Identity

Scope kept: synthetic only; no native PPO or evaluation; no reward, architecture, KL-penalty, freezing or curriculum change; two arms, one seed each, 30,720 decisions each (61,440 total ≤ 60k + rounding to 512-multiples); no test-seed use; S1–S14 artifacts untouched. Wall-clock: 952.9 s (Arm A) and 938.5 s (Arm B) **run in parallel**, including all evaluations — total ≈ 16 min of the 35-minute cap. Nothing was extended.
Artifacts: `research/s15_mixed_training/` — `runs/arm_{A,B}_s15/{run.json, progress.csv, checkpoints/step_*.zip, rows/step_*.json}`, `analysis.json`, `ablation_trained_checkpoints.json`, `ablation_rows.json`, `logs/`.
Code (new): `python/synthetic_rl/s15_env.py` (MixedEnv, GroupBlind), `s15_eval.py`, `experiments/s15_train.py`, `s15_analyze.py`.

## 1. Configuration (identical for both arms except the group channels)
* Init: `research/s14_group_observation/s8c_migrated_v2_step0.zip` (migrated S8-C **before** any S14 PPO update); verified bitwise equal at start (`run.json: parameters_bitwise_equal_to_init = true`, 75,144 parameters). The S14 10k checkpoint was not used.
* Observation v2 (21 channels, 16,328 floats). **Arm A**: group channels live. **Arm B**: channels 15–20 zeroed by a wrapper in training *and* evaluation; same architecture, same weights at init.
* PPO: n_steps 512, batch 256, 4 epochs, target_kl 0.03, lr 3e-4, ent_coef 0.01, γ 0.99, λ 0.95, clip 0.2; seed 15 both arms; 2 threads each.
* Mixture rule (fixed, non-adaptive): at every reset pick the family with fewer cumulative decisions so far. S7 pool: Gap-7 40 % / Gap-10 40 % / short k∈{2,3,5} 20 % from the frozen S7 TRAIN manifests. S13: the 18 training scenarios (game seeds 500000/523757/531676/547514/555433/571271). Rewards as in the respective environments (S7: +1 on connection; S13: +1 per delivery, −10 game over) — unchanged.
* **Actual decision ratio**: Arm A 15,340 S7 / 15,380 S13 (49.9 % / 50.1 %; 354 S7 episodes [Gap-7 130, Gap-10 142, short 82] / 54 S13 episodes); Arm B 15,275 / 15,445 (49.7 % / 50.3 %; 345 / 54).
* Checkpoints and evaluations at 0, 5,120, 10,240, 20,480, 30,720 decisions.
* Evaluation (validation only): S7 validation manifests Gap-7 (n=60) and Gap-10 (n=39), deterministic and stochastic (torch seed 0); S13 validation = 6 scenarios of game seeds 507919/579190 × 5 env seeds = 30 episodes, stochastic policy, env seeds 7000+… and torch seeds 8008+… identical for every checkpoint of both arms. The four S13 test seeds were never evaluated.
* Throughput: rollouts 228 decisions/s, PPO update 7.4 s per 512 (two arms sharing the CPU) → 53 training decisions/s; evaluation 350–367 s per arm in total.

## 2. S7 retention (validation; det = deterministic, sto = stochastic; gate: Gap-7 det ≥ 0.75, Gap-10 det ≥ 0.65, no WAIT collapse)

| decisions | A Gap-7 det / sto | A Gap-10 det / sto | A WAIT det / steps | B Gap-7 det / sto | B Gap-10 det / sto | B WAIT det / steps | gate A / B |
|---|---|---|---|---|---|---|---|
| 0 | 0.83 / 0.93 | 0.77 / 0.85 | 0.15 / 22.5 | 0.83 / 0.93 | 0.77 / 0.85 | 0.15 / 22.5 | pass / pass |
| 5,120 | 0.72 / **0.97** | **0.79** / 0.85 | 0.11 / 29.5 | 0.75 / 0.78 | 0.49 / 0.44 | 0.09 / 28.8 | fail (Gap-7 0.72) / fail (Gap-10 0.49) |
| 10,240 | 0.68 / 0.78 | 0.72 / 0.64 | 0.13 / 31.3 | 0.62 / 0.78 | 0.59 / 0.49 | 0.10 / 36.7 | fail / fail |
| 20,480 | 0.32 / 0.75 | 0.15 / 0.51 | 0.00 / 51.5 | 0.68 / 0.53 | 0.21 / 0.28 | 0.00 / 33.8 | fail / fail |
| 30,720 | 0.23 / 0.75 | 0.31 / 0.56 | 0.00 / 56.2 | 0.62 / 0.57 | 0.41 / 0.31 | 0.00 / 37.7 | fail / fail |

Wilson 95 % CI half-widths are ≈ ±0.10 (Gap-7) and ±0.13 (Gap-10); the 5,120 shortfalls (0.72 vs 0.75; 0.49 vs 0.65) are within/at noise for Gap-7 and clear for Gap-10. The preregistered thresholds were applied as written.

**No trained checkpoint of either arm passes the retention gate.** The 50/50 decision mixture slowed but did not prevent deterministic long-route degradation; it did prevent the S14 WAIT collapse (WAIT fraction fell to 0, WAIT probability on 60 fixed Gap-7 start observations stayed 0.008–0.030 in A and ≤ 0.0003 in B).

Failure mode this time (Arm A, deterministic Gap-7): episodes lengthen (22.5 → 56 steps), irrelevant ADDs per episode rise (10.4 → 17.2 at 20k), REMOVE share in S7 training actions grows 0.09 → 0.30 (Arm A) and 0.07 → 0.18 (B); the greedy policy drifts into add/remove loops rather than waiting. The stochastic policy of Arm A keeps 0.75–0.97 on Gap-7 throughout, so the knowledge is not erased — the argmax is. Arm B degrades in both modes (stochastic Gap-7 0.93 → 0.57).

## 3. S13 multi-house validation (30 matched episodes per cell)

| decisions | arm | deliveries total / median | new houses ≤60 (rate) | latency median | destroyed total / median | zero-concrete median | WAIT while disconnected | survived | decisions median | return mean | misrouted house-steps |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | A = B | 1229 / 22 | 41/129 (0.32) | 64 | 1691 / 47.5 | 0.62 | 0.02 | 9 | 291.5 | 34.0 | 1781 |
| 5,120 | A | **1342 / 20** | **49/135 (0.36)** | **48.5** | 1491 / 51 | 0.65 | 0.08 | **10** | 321 | **38.1** | 2175 |
| 5,120 | B | 755 / 8 | 35/112 (0.31) | 48 | 1346 / 23 | 0.66 | 0.06 | 5 | 189 | 16.8 | 1701 |
| 10,240 | A | 697 / 8 | 36/120 (0.30) | 28 | 923 / 22.5 | 0.63 | 0.14 | 3 | 237.5 | 14.2 | 2281 |
| 10,240 | B | 458 / 3.5 | 19/92 (0.21) | 57 | 779 / 9 | 0.65 | 0.01 | 4 | 181.5 | 6.6 | 1820 |
| 20,480 | A | 1082 / 19.5 | 39/137 (0.28) | 60 | 1608 / 47.5 | 0.58 | 0.03 | 7 | 300.5 | 28.4 | 1951 |
| 20,480 | B | 459 / 1 | 22/92 (0.24) | 69 | 796 / 5 | 0.67 | 0.00 | 4 | 179.5 | 6.6 | 1368 |
| 30,720 | A | 833 / 7 | 36/120 (0.30) | 43.5 | 777 / 20.5 | 0.59 | 0.12 | 5 | 217 | 19.4 | 3222 |
| 30,720 | B | 979 / 16.5 | 28/107 (0.26) | 63 | 1220 / 43 | 0.67 | 0.00 | 7 | 219 | 25.0 | 538 |

Paired A − B on the same 30 (scenario, seed) episodes (`analysis.json: A_vs_B_matched`): deliveries median diff +11 (21 higher / 9 lower) at 5k, +4.5 (17/10) at 10k, +9.5 (23/6) at 20k, +0.5 (15/14) at 30k; new-house connections within 60: median diff 0 at every checkpoint (11/10/9, 13/13/4, 12/11/7, 12/10/8). Arm A also destroys *more* connections than B (median diff +6 … +16.5) and builds more.

**Against its own step 0** (the only checkpoint that passes retention), Arm A at 5,120 improves new-house rate 0.32 → 0.36 (+8 connected houses), survival 9 → 10, total deliveries +9 %, latency 64 → 48.5, while median deliveries fall 22 → 20 and misrouting rises; later checkpoints are below baseline. Arm B never exceeds its baseline on deliveries or new-house rate. So **A > B mostly because B gets worse, not because A gets much better.**

## 4. Does group identity matter? (ablation, inference only, same seeds)
Selected checkpoints are step 0 for both arms (see §6), where zeroing the group channels is a no-op by construction (zero first-layer weights) — the preregistered ablation is uninformative. As a diagnostic I ablated Arm A's two best trained checkpoints (`ablation_trained_checkpoints.json`):

| Arm A checkpoint | intact deliveries median / total | ablated | paired intact − ablated (median; higher/equal/lower) | new ≤60 intact → ablated | Gap-7 det intact → ablated | Gap-10 det |
|---|---|---|---|---|---|---|
| 5,120 | 20 / 1342 | 18 / 1252 | +0.5 (15/6/9) | 0.36 → 0.40 | 0.72 → 0.75 | 0.79 → 0.82 |
| 10,240 | 8 / 697 | 5 / 610 | +1.0 (16/6/8) | 0.30 → 0.24 | 0.68 → 0.68 | 0.72 → 0.69 |

Group-channel first-layer weights: |w| mean 0.0024 (5k) → 0.0065 (30k) vs 0.10 on the v1 channels (not by itself proof of non-use). Combined with the ablation (small, inconsistent, within pair noise), **the group-aware representation is not supported by this experiment**: Arm A's edge over Arm B is not reproduced by removing the group information from Arm A itself, and A's own multi-house gains over step 0 are marginal.

## 5. Forgetting diagnosis (no scope expansion)
* Timing: deterministic Gap-7 already −7 points at 5,120 decisions (both arms), −15 at 10,240, collapse (A) between 10k and 20k. Gap-10 deterministic in Arm B collapsed by 5,120.
* PPO statistics per 5k block (Arm A / B): approx-KL 0.024 → 0.012 / 0.019 → 0.010; clip fraction 0.14 → 0.07 / 0.12 → 0.04; policy entropy **rises** in A (1.73 → 2.43 nats) and is flat in B (1.69 → 1.42–1.54); value loss 6.3 → 1.7 / 4.6 → 4.1; explained variance mostly ≤ 0.1 (0.57 in A's last block). Updates were well-behaved — no instability, early KL stops as usual. Gradient norms are not logged by SB3 and were not added.
* Action distributions in training rollouts (A): S7 WAIT/ADD/REMOVE 0.14/0.77/0.09 → 0.05/0.64/0.30; S13 0.07/0.90/0.03 → 0.07/0.76/0.18. (B): S13 ≈ 0.00/0.98/0.01 throughout.
* Does the S13 objective reward WAIT? Not in this setting: WAIT share in S13 training actions stayed ≤ 7 % (A) and ≤ 1 % (B), WAIT probability on fixed S7 observations never exceeded 0.03. The S14 WAIT collapse came from S13-only training; with S7 rewards in the stream the degradation takes the form of over-building/removal loops and a flatter (higher-entropy) policy whose argmax is no longer the useful ADD.
* No correctness bug was found; the preregistered experiment was kept.

## 6. Checkpoint selection (preregistered order, identical for both arms)
1 reject retention failures → only step 0 remains in each arm; 2–5 moot. **Selected: step 0 (the migrated S8-C) for both Arm A and Arm B.** Rationale: every trained checkpoint fails the S7 retention gate; the rule does not allow selecting on multi-house return. No test seeds inspected.

## 7. Verdict against the success criteria
* Passes S7 retention: **no trained checkpoint** (either arm).
* Improves multi-house over step 0: only Arm A at 5,120, marginally (new-house rate +0.045, survival +1, total deliveries +9 %, median deliveries −2), and that checkpoint fails retention by 0.03 on Gap-7 (within CI).
* Not via WAIT/reward exploitation: confirmed (no WAIT inflation; return tracked deliveries).
* Group representation supported: **no** (ablation effect ≈ 0; weights tiny; A > B driven by B's decline).
* **S15 fails.** Mixed training answers question 1 negatively for this recipe — 50/50 decision mixing with plain PPO does not hold deterministic long-route competence past ~5k decisions, although it keeps the stochastic Arm A policy at 0.75+ Gap-7 and avoids the WAIT collapse. Question 2: no evidence yet that the agent uses destination identity. One seed per arm; the differences quoted are directions on 30 and 60/39 episodes, not established effects.

## 8. Recommendation for S16 (needs approval; none of it was done)
The limiting factor is the training recipe, not the observation. Candidates, in order: (a) evaluate retention on the *stochastic* policy as well and make the mode of use explicit — Arm A's stochastic Gap-7 (0.75–0.97) survived all 30k decisions, so "deterministic collapse" may partly be an argmax artefact of a flatter policy; (b) a behaviour-regularised continuation (KL/BC toward the migrated step 0, or trunk freezing) — explicitly excluded here; (c) a larger receptive field so group channels can act beyond 3 tiles — architecture change, excluded here; (d) more seeds before any conclusion. Not recommended: a 100k continuation of either S15 arm, or any native evaluation.

**STOPPED. No S16 work started.**
