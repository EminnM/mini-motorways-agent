# S8 — Controlled long-route curriculum pilot (synthetic only)

**Result in one paragraph.** Under matched budgets (99,840 decisions per arm, one seed each, identical network / PPO hyperparameters / sparse +1 reward / evaluation tasks) only **Arm C (pretrained S1 CNN + curriculum)** learned long routes: held-out TEST completion at k = 2/3/5/7/10 rose from the zero-shot S1 level (≈ 91/89/51/19/0–3 % stochastic) to **97/99/99/91/72 %** (stochastic; 98/95/91/91/69 % deterministic). **Arm A (direct on {7,10})** never saw a reward in 99,840 decisions (0 successes in 1,174 training episodes; 0 % at k ≥ 3 on TEST). **Arm B (random-init curriculum)** learned slowly and weakly (27/21/20/18/0 % stochastic TEST). Success in C is much more efficient than the zero-shot policy (≈ 2× the minimum number of ADDs instead of ≈ 3.3×) but is not minimal, and about half of the k = 10 successes rely on edges whose native legality is unknown. One seed per arm, k = 10 TEST has 32 tasks: these are mechanistic findings, not statistical superiority claims.
Nothing native was run (no game launch), no reward / observation / mask change, no second seed, no extra arm, no budget extension. S1–S7 artifacts untouched; everything is in `research/s8_long_route_training/`.

## 1. Setup actually used
* Arms (`research/s7_long_route/configs/arm_{A,B,C}.yaml`; hashes in each `plan.json`): A random init, trains only on T = uniform {k=7, k=10}; B random init, stages [2]→[3]→[5]→[7]→T; C = B initialised from the frozen S1 seed-13 checkpoint (SHA 41c1e085…, strict load, fresh Adam/timestep/constant lr 3e-4).
* Stage rule (preregistered in the config, unchanged): leave a stage when the last 100 training episodes have success ≥ 0.7 and ≥ 4,096 decisions were spent, or after 12,288 decisions (a multiple of the 512-decision rollout, the closest to 12,000).
* PPO: n_steps 512, batch 256, epochs 4, target_kl 0.03, lr 3e-4, ent 0.01, γ 0.99; same training seed 11 for all arms (A and B start from bit-identical weights); `SpatialMaskablePolicy`, 73,416 parameters.
* Budget: 195 rollouts = **99,840 decisions** per arm (≤ 100k cap); smoke runs not counted. Evaluation points (multiples of 512 closest to the requested ones): 0, 10,240, 25,088, 50,176, 75,264, 99,840; validation set (279 tasks: 60/60/60/60/39 per k), deterministic **and** stochastic (seed 0). The global torch RNG is saved/restored around each evaluation and parameters/optimizer/timesteps are hashed before and after (unchanged in every evaluation of every run).
* Selection rule written before training (`PREREGISTERED_SELECTION.json`, SHA b167de20…): validation, stochastic policy, maximise mean(k7, k10) success; ties → higher efficiency → lower churn → earlier checkpoint. TEST evaluated once per arm on the selected checkpoint (A: step 10,240 — all checkpoints tied at 0 % for k ≥ 7, so the tie-break picked the earliest-efficient one; B: step 50,176; C: step 99,840).

## 2. Smoke tests (512 decisions each, not counted; `smoke/`)
All three PASS: configurations initialise; C's checkpoint loads strictly (16 tensors, bitwise equal to the source); the optimizer starts empty and PPO changes the parameters (hash differs, Adam state non-empty); all parameters and losses finite (e.g. A: approx_kl 0.016, value_loss 0.019; C: approx_kl 0.020); 0 rejected (illegal) actions in 1,536 smoke steps, 0 reward/boundary violations (reward ∈ {0,1}, reward 1 ⇔ terminated, never terminated∧truncated, episode length ≤ horizon); evaluation left the training state unchanged; checkpoints reload with identical parameters and identical predictions.

## 3. Training and validation curves (`validation_curves.png`, `ppo_diagnostics.png`, `summary.json`)
Validation success %, stochastic policy (deterministic in brackets for k=7/10), k = 2 / 3 / 5 / 7 / 10:
| decisions | A | B | C |
|---|---|---|---|
| 0 | 0/0/0/0/0 | 0/0/0/0/0 | 90/82/58/22/0 (0/3) |
| 10,240 | 2/0/0/0/0 | 0/0/0/0/0 | 100/98/77/17/0 (7/0) |
| 25,088 | 2/0/0/0/0 | 8/10/5/0/0 | 98/97/72/50/13 (45/21) |
| 50,176 | 3/2/0/0/0 | 22/30/25/13/5 (5/0) | 93/83/75/70/28 (48/41) |
| 75,264 | 7/3/0/0/0 | 17/22/18/13/3 | 92/93/88/75/44 (63/44) |
| 99,840 | 10/3/0/0/0 | 20/20/25/12/3 (12/5) | 95/100/90/83/82 (83/77) |
PPO diagnostics: policy entropy fell A 4.3→2.3, B 5.1→2.4, C 2.5→1.7 nats; approx_kl stayed ≤ 0.025 (below the 0.03 target); clip fraction 0.02–0.19; explained variance A ≈ 0 (nothing to predict: no reward ever seen), B 0.5–0.9, C 0.1–0.8 and noisy.

## 4. Curriculum stage progression
| arm | stage → decisions spent (reason, last-100 success) |
|---|---|
| A | single stage {7,10}: 99,840 decisions, 0 successes |
| B | k=2: 12,288 (budget, 0 %); k=3: 12,288 (budget, 6 %); k=5: 12,288 (budget, 34 %); k=7: 12,288 (budget, 14 %); {7,10}: 50,688 — **no stage was ever left on the success threshold** |
| C | k=2: 4,096 (threshold, 98 %); k=3: 4,096 (threshold, 99 %); k=5: 4,096 (threshold, 75 %); k=7: 12,288 (budget, 59 %); {7,10}: 75,264 |
B's curriculum progressed only because of the budget cap: it entered each harder stage without having learned the previous one, and its first real learning appeared between 10k and 25k decisions (k=2 success 0→8 %). This is a curriculum-progression weakness of the random-init arm, not an implementation fault (rejected actions: 0 in all arms).

## 5. Held-out TEST performance (selected checkpoints; n per k = 88 / 100 / 100 / 93 / 32; `runs/*/final_test_v2.json`)
Success %, stochastic [deterministic], with 95 % Wilson intervals for the two target lengths:
| k | 2 | 3 | 5 | 7 | 10 |
|---|---|---|---|---|---|
| A | 2.3 [0] | 0 [0] | 0 [0] | 0 [0] | 0 [0] |
| B | 27.3 [26.1] | 21.0 [20.0] | 20.0 [22.0] | 18.3 [12.9] (CI ≈ 8–21 det) | 0 [0] |
| **C** | **96.6 [97.7]** | **99.0 [95.0]** | **99.0 [91.0]** | **91.4 [91.4]** (CI 84–96) | **71.9 [68.8]** (CI 55–84 / 51–82) |
| S1 zero-shot reference (S7, s13 stochastic) | 90.9 | 89 / 80 | 51 / 56 | 19.4 | 0 / 3.1 |
Paired by task (same TEST configurations): at k = 7, C succeeds on 68 tasks where B fails and B on 0 where C fails (17 both); A on 0; at k = 10, C 23 vs B 0 vs A 0; B beats A on 17 (k=7) and 20 (k=5) tasks, A beats B on none.

## 6. Efficiency, partial progress, churn (TEST)
| | k=2 | k=3 | k=5 | k=7 | k=10 |
|---|---|---|---|---|---|
| C sto: mean efficiency (k/#ADDs) on successes | 0.57 | 0.54 | 0.45 | 0.48 | 0.47 |
| C det: efficiency | 0.67 | 0.63 | 0.54 | 0.58 | 0.68 |
| B sto: efficiency | 0.34 | 0.30 | 0.33 | 0.29 | – |
| S1 zero-shot s13 sto (S7): efficiency | 0.56 | 0.42 | 0.39 | 0.30 | – |
| C sto: irrelevant ADDs per successful episode | 2.9 | 4.4 | 8.2 | 9.1 | 11.6 |
| S1 zero-shot s13 sto: irrelevant ADDs / episode (all) | 4.7 | 9.0 | 23.7 | 41.4 | 53.5 |
| C sto: success within k+2 ADDs | 59 % | 44 % | 14 % | 5 % | 0 % |
| C det: success within k+2 ADDs | 74 % | 49 % | 21 % | 10 % | 16 % |
| C sto: success within 2k ADDs | 59 % | 47 % | 38 % | 44 % | 34 % |
| C sto: first ADD reduces remaining distance | 0.53 | 0.48 | 0.36 | 0.46 | 0.50 |
| C sto: max route progress (RANDOM: 0.18–0.48) | 0.98 | 1.00 | 1.00 | 0.96 | 0.92 |
| C sto: road churn (REMOVEs + house-edge swaps) / episode | 0.7 | 1.1 | 1.2 | 4.0 | 11.3 |
| B sto: road churn / episode | 1.2 | 3.6 | 15.1 | 28.6 | 55.5 |
Action mix at k=10 (stochastic): C WAIT 18 % / ADD 68 % / REMOVE 13 %; B 7 / 74 / 19 %; A 12 / 58 / 29 %. Mean ADDs per episode in C: 5.5 / 7.8 / 13.6 / 19.4 / 31.4.
**Targeted construction or coverage?** C is clearly more targeted than the zero-shot policy and than RANDOM (RANDOM completes 0 % at k ≥ 7): ~3.6× fewer irrelevant ADDs per episode than the zero-shot policy at k=7 (11.6 vs 41.4), churn 7× lower than B, first ADD useful about half the time (RANDOM 0.05–0.16). But successes still use about twice the minimum number of edges (only 5 % at k=7 and 0 % at k=10 within k+2 ADDs), so the behaviour looks like repeated local construction toward the destination with substantial redundancy, not minimal-path planning. I do not claim planning.

## 7. Forgetting
No harmful forgetting in C: stochastic validation k=2 was 90 % at step 0, 100 % at 10k, 92–98 % afterwards and 95 % at the end (TEST 96.6 %); k=3 82→100 %; k=5 58→90 %. The stage-4 distribution contains only k ∈ {7,10}, yet the short routes were retained (deterministic k=2 fluctuated 77–100 % between checkpoints). B shows mild forgetting after its best point: k=3 30 % (50k) → 20 % (99.8k), k=2 22 → 20 %, while k=5/7 stayed flat; A has nothing to forget.

## 8. Synthetic/native legality risk (diagnostic only; masks unchanged)
Evidence: 24 native start states; S7-legal ADD edges that were native-legal by class (**corrected** definition, `legality_evidence_v2.json`): interior dry edges 2,513/2,513 (100 %); window-boundary edges 79 % (1,045/1,322); water-endpoint edges 71 % (169/238). Legality of arbitrary later states is not established, so anything outside "interior dry" is labelled **unknown**, never "illegal".
* Share of ADDs by class (C stochastic, TEST): k=7: 83 % interior / 17 % boundary / 1 % water; k=10: 64 / 33 / 2 %. B and A use more boundary edges at k=10 (B 42 %, A 37 %).
* Successful routes whose new edges are all interior-dry ("supported legal"): C 92 / 91 / 86 / 82 / **52 %** at k = 2 / 3 / 5 / 7 / 10 (deterministic 91 / 93 / 88 / 82 / 64 %); the rest use at least one unknown-legality edge. So roughly half of the k = 10 successes (and ≈ 1 in 5 at k=7) cannot be assumed to be natively valid; with 71–79 % per-edge legality in the evidence, a route with several such edges is unlikely to be fully valid natively (not computed exactly).
* Correction made after the pilot: my first label for "window boundary" also flagged tiles adjacent to the destination, which made every route look unsupported (0 %). I fixed the definition, archived the old one (`s8_eval_v1_boundary_definition.py`), recomputed the evidence, and re-ran the TEST evaluation of the same three selected checkpoints once more with the corrected labels; all success counts reproduced bit-for-bit (`success_reproduces_original_run` true for every k and mode), confirming the evaluation is deterministic. The original `final_test.json` files are kept.

## 9. Runtime and throughput
Three processes in parallel, 3 torch threads each. The brief benchmark chosen before launch (`bench_*.json`): predicted per-process 34 decisions/s (1 thread), 48 (2), 55 (3), 54 (4); solo 56 (1 thread) / 117 (6 threads) → parallel×3 predicted aggregate ≈ 165 vs sequential 117. **Measured** training speed was lower: 38.7 (A), 38.9 (B), 42.5 (C) decisions/s per process (aggregate ≈ 120) because evaluation phases and the three update phases contend. Wall-clock of the main pilot (including 6 validation evaluations × ≈ 60–150 s and the TEST evaluation): A 2,933 s, B 2,910 s, C 2,582 s = **48.9 min** (cap 75 min; no deadline stop); the corrected-label TEST re-run took ≈ 4 min extra. Validation evaluations took 90–152 s for A and B versus 61–96 s for C (A/B episodes run to the horizon more often).

## 10. Hypotheses
* **H8a (direct sparse-reward learning): not supported.** A saw no reward in 1,174 episodes; validation/TEST success at k ≥ 3 stayed 0 and only entropy fell. Consistent with RANDOM completing 0 % at k ≥ 7: no initial signal to bootstrap from → an exploration failure.
* **H8b (curriculum benefit, B vs A): weakly supported.** B learned something where A learned nothing (paired: 20 / 17 tasks at k = 5 / 7 with 0 reverse), but only to ≈ 20 % and 0 % at k = 10; B never met a stage threshold, so its curriculum degenerated into a fixed time schedule. Evidence of curriculum *progression failure* for the random-init arm.
* **H8c (pretraining benefit, C vs B): strongly supported in this single-seed comparison.** C reaches 91 % / 72 % at k = 7 / 10 vs B 18 % / 0 %; paired 68 vs 0 and 23 vs 0 tasks. Caveat: C starts with an available reward signal (90 % at k = 2, 22 % at k = 7), so the benefit is "the pretrained local gap-closing prior makes the sparse reward discoverable"; pretraining and curriculum cannot be separated because no pretrained-without-curriculum arm was allowed.
* **H8d (all fail): refuted** for C, true for A, largely true for B.
Alternative explanations not excluded: a single seed; k=10 TEST n=32; only one map; validation/TEST drawn from the same scenes (different house positions); and C's improvement may partly reflect adaptation to S7-specific rules (window, house-edge replacement) that differ from native.

## 11. Recommendation
Because one arm improved substantially, run the **small native zero-shot transfer test** you described, with a strict budget — nothing else first:
* Policy: Arm C final checkpoint (SHA 85a9cb75…, `runs/arm_C_s11/checkpoints/step_0099840.zip`), inference only, **stochastic and deterministic**, against the S1 seed-13 checkpoint as the paired reference, on **12 native normal-start seeds not used in S4/S6 TEST** (3 game instances in parallel; est. ≈ 25 min).
* Metrics: deliveries, ≥ 1 useful connection, time to first connection/delivery, survival, ADD / REMOVE / swap counts, `terminal_penalty_missing`; do **not** compare returns (S5/S6 showed they are contaminated). Report the share of ADDs outside "interior dry" and whether any connection rests on them.
* Interpretation rule fixed beforehand: transfer is "supported" only if C connects/delivers on clearly more seeds than the S1 reference with a gap larger than seed noise; otherwise the limitation is the synthetic/native mismatch (spawns, time dynamics, legality), not long-route learnability.
A synthetic follow-up that this pilot cannot answer (and is not proposed to run now): pretrained-direct (C without curriculum) to separate pretraining from curriculum, and a second seed.

## 12. Reproduction and artifacts
```
python experiments/s8_train.py --arm A|B|C --seed 11 --smoke                              # smoke (not counted)
python experiments/s8_train.py --arm A|B|C --seed 11 --threads 3 --deadline-min 75 --reserve-min 7    # main pilot, three processes in parallel
python experiments/s8_test_reeval.py A|B|C ; python experiments/s8_report.py             # corrected-label TEST re-run, aggregation
```
Saved under `research/s8_long_route_training/`: `PREREGISTERED_SELECTION.json`, `runs/arm_*/` (plan.json with configuration + source/manifest hashes, `eval_validation.jsonl`, per-task validation rows, `rollouts.jsonl`, `stages.jsonl`, `sb3/progress.csv`, `selection.json`, `final_test*.json`, per-task TEST rows, checkpoints + `SHA256.txt`, `train_status.json`), `smoke/`, `bench_*.json`, `legality_evidence*.json`, `summary.json`, `paired_test_comparison.json`, figures, logs. Selected checkpoint hashes (prefix): A step 10,240 `742611d4`, B step 50,176 `4065f195`, C step 99,840 `85a9cb75`. Training seed 11 and evaluation seed 0 for all arms; validation and TEST manifests as frozen in S7 (`manifests/SHA256.txt`).
