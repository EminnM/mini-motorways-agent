# S9 — Native zero-shot transfer of the S8 long-route curriculum policy

**Result (exploratory, 12 seeds, inference only).** The S8 Arm-C policy (P2) transfers: it established at least one useful house–destination connection on **11/12** unused native normal-start seeds in both deterministic and stochastic mode, versus **1/12** (deterministic) and **3/12** (stochastic) for the S1 seed-13 baseline (P1). It delivered on 9/12 (det) and 11/12 (sto) seeds versus 1/12 and 3/12. Total deliveries over the 12 seeds: P1 det 4, P1 sto 66, P2 det 284, P2 sto 482. P2 connected on 4/5 of the seeds that need ≥ 8 added edges, where P1 connected on none. Residual weaknesses are large: heavy house-edge swapping and road-state cycling persist, two deterministic episodes connect only ~5 decisions before game over, one seed with k = 10 is never connected, and every episode but one still ends in game over. No PPO, optimizer step, reward/mask/observation change, anti-cycle filter, or game modification; S1–S8 artifacts untouched.
Raw data: `research/s9_native_transfer/` (per-episode rows, per-step trajectories for all 48 episodes, prechecks, twin check, summary.json, logs).

## 1. Checkpoints and pre-launch validation (`prechecks.json`; all passed)
* **P1** S1 CNN seed 13 `runs/2026-10-08_140406_syn_s1_cnn_s13/checkpoints/final.zip`, SHA-256 `41c1e085622b8f0d…` (trained on synthetic Gap-1 only).
* **P2** S8 Arm C `research/s8_long_route_training/runs/arm_C_s11/checkpoints/step_0099840.zip`, SHA-256 `85a9cb75cf287d24…` — the checkpoint chosen by the preregistered validation rule (selection.json), equal to the hash recorded with the S8 TEST evaluation.
* Both load strictly into the S4 spatial architecture (16 tensors / 73,416 parameters, bitwise), observation Box(−1,1,(11666,)) and Discrete(3111) match the native encoder; masked probabilities finite and normalised on 144 saved native states with zero mass on illegal actions; deterministic actions identical across two independent loads and always legal; no training or game process was running. Checkpoints were not modified.
* **Instrumentation non-interference** (live): a 60-decision rollout with the full S9 instrumentation was identical (actions, rewards, observation hashes) to the plain rollout; plain-vs-plain was also identical (`twin_noninterference.json`).

## 2. Seeds and procedure
Seeds `500000 + 7919·i`, i = 0…11 (fixed rule written before any episode; no outcome-based choice): 500000, 507919, 515838, 523757, 531676, 539595, 547514, 555433, 563352, 571271, 579190, 587109. Checked against EVAL_SEEDS, VAL_SEEDS, 9101–9110, the S7 capture seeds and a text scan of `research/`, `experiments/`, `python/`, `runs/`: no overlap (`seeds.json`). Conditions: P1 det, P1 sto, P2 det, P2 sto; native normal start, 1.0 s decisions, 600-decision horizon, native reward and termination; the stochastic episodes use torch seed `1000 + seed index` for both policies (identical sampling seed, trajectories diverge as soon as the policies differ). Action = exactly `model.predict(obs, native mask)`; nothing is filtered or overridden.
**Native wall-clock:** the budget clock started with the twin check (epoch 1791478710); all four conditions finished by +576 s ≈ **9.6 min** wall-clock (cap 25), four game instances in parallel; summed instance time 1,274 s for the 48 episodes + 33 s twin. A first launch was killed after ≈ 100 s because my carpark-stub detection (diagnostic only) was wrong; its rows were discarded and nothing was selected on its outcomes.

## 3. Primary outcomes (12 seeds each; medians with max)
| | P1 det | P1 sto | P2 det | P2 sto |
|---|---|---|---|---|
| seeds with ≥ 1 useful connection | 1 | 3 | **11** | **11** |
| useful connections / episode (median, max) | 0, 2 | 0, 5 | 2, 9 | 5.5, 13 |
| seeds with ≥ 1 delivery | 1 | 3 | **9** | **11** |
| deliveries / episode (median, max) | 0, 4 | 0, 36 | 8.5, 85 | 38, 146 |
| first connection decision (median; n) | 86 (1) | 26 (3) | 21 (11) | 23 (11) |
| ADDs before first connection (median; n) | 9 (1) | 26 (3) | 21 (11) | 23 (11) |
| irrelevant ADDs before first connection (median; n) | – | 19 (1) | 4 (6) | 15 (9) |
| first delivery decision (median; n) | 96 (1) | 36 (3) | 27 (9) | 39 (11) |
| connection → first delivery (decisions) | 10 | 9–14 | 11–14 | 8–21 |
| distinct edges constructed (median) | 11.5 | 43.5 | 36 | 81.5 |
Definitions are those of S4/S6: a connection counts a house whose road component contains a destination of its colour group (`connectivity_potential`), so roads merely near a destination do not count. Later connections include houses that spawn later and join the network. "Irrelevant ADDs before the first connection" = ADDs before the connection minus new edges on the shortest road path house→carpark, available only where that path could be computed (n shown).
By required length (initial minimum number of ADDs from the native mask, first house; 3 seeds have no path under the initial native mask, unresolved): k ≤ 7 (4 seeds): P1 det 1/4, P1 sto 3/4, P2 4/4 both modes; k ≥ 8 (5 seeds): P1 0/5 both modes, P2 4/5 both modes; unknown-k seeds (3): P1 0/3, P2 3/3. Initial k per seed: 500000: 7, 523757: 10, 531676: 7, 539595: 11, 555433: 11, 563352: 10, 571271: 8, 579190: 6, 587109: 5 (matches the 4–12 range of S7).

## 4. Secondary outcomes
Survival (decisions, median / max): P1 det 150/154, P1 sto 152/293, P2 det 180/463, P2 sto 314/600 (one P2 sto episode reached the horizon with 146 deliveries and 17 houses). Native score equals deliveries here. Terminal penalty missing (flag from S6): P1 0/12 and 0/12, P2 1/12 and 1/12 — returns are therefore not used as an outcome. Action shares (WAIT/ADD/REMOVE): P1 det 24/76/0 %, P1 sto 3/96/0, P2 det 11/88/1, P2 sto 4/90/5. Mean legal actions: P1 det 145, P1 sto 46, P2 det 114, P2 sto 69 (see §6 for the collapse once the road budget is used). Mean masked policy entropy: 2.9 / 1.5 / 1.6 / 1.3 nats.

## 5. Paired per-seed comparison (P2 − P1, same seed; modes kept separate; exploratory)
Per-seed table (k0 = initial required edits; conn / first-connection step / deliveries / decisions survived):
| seed | k0 | P1 det | P1 sto | P2 det | P2 sto |
|---|---|---|---|---|---|
| 500000 | 7 | 0 / – / 0 / 151 | 5 / 26 / 36 / 293 | 7 / 8 / 45 / 317 | 9 / 23 / 42 / 315 |
| 507919 | ? | 0 / – / 0 / 151 | 0 / – / 0 / 151 | 2 / 145 / 0 / 151 | 4 / 23 / 31 / 300 |
| 515838 | ? | 0 / – / 0 / 145 | 0 / – / 0 / 145 | 2 / 75 / 4 / 145 | 2 / 65 / 4 / 145 |
| 523757 | 10 | 0 / – / 0 / 153 | 0 / – / 0 / 153 | 0 / – / 0 / 153 | 0 / – / 0 / 153 |
| 531676 | 7 | 0 / – / 0 / 143 | 2 / 27 / 11 / 207 | 6 / 11 / 55 / 361 | 13 / 16 / 146 / 600 |
| 539595 | 11 | 0 / – / 0 / 150 | 0 / – / 0 / 150 | 4 / 145 / 0 / 150 | 6 / 29 / 61 / 453 |
| 547514 | ? | 0 / – / 0 / 154 | 0 / – / 0 / 154 | 1 / 84 / 4 / 154 | 2 / 81 / 4 / 154 |
| 555433 | 11 | 0 / – / 0 / 140 | 0 / – / 0 / 128 | 1 / 72 / 2 / 140 | 9 / 28 / 45 / 349 |
| 563352 | 10 | 0 / – / 0 / 154 | 0 / – / 0 / 154 | 8 / 21 / 85 / 463 | 5 / 32 / 36 / 312 |
| 571271 | 8 | 0 / – / 0 / 149 | 0 / – / 0 / 149 | 9 / 11 / 61 / 392 | 10 / 12 / 40 / 331 |
| 579190 | 6 | 2 / 86 / 4 / 140 | 5 / 17 / 19 / 216 | 2 / 10 / 15 / 205 | 4 / 18 / 21 / 256 |
| 587109 | 5 | 0 / – / 0 / 142 | 0 / – / 0 / 142 | 2 / 16 / 13 / 211 | 6 / 23 / 52 / 354 |
* Connections: P2 connected where P1 did not on **10 seeds (det)** and **8 seeds (sto)**; P1 connected where P2 did not on **0**. Deliveries: P2 delivered where P1 did not on 8 seeds (det) and 8 (sto), the reverse on 0. Median paired difference in deliveries: +7.5 (det), +33.5 (sto); in decisions survived: +32.5 (det), +153.5 (sto); in distinct edges: +23 (det), +34.5 (sto).
* Time and ADDs to first connection can be paired on only 1 (det) and 3 (sto) seeds because P1 rarely connects (P2 −76 decisions on the one det seed; −3, −11, +1 on the sto seeds): too few to say anything.
* Descriptive sign tests over seeds (not independent of the policy mode, 12 seeds): P2 > P1 in connection, 10 vs 0 (det, exact two-sided p ≈ 0.002) and 8 vs 0 (sto, p ≈ 0.008). This is an exploratory contrast, not a definitive estimate.

## 6. Native legality, edits and cycling (`summary.json`, `raw/*.json.gz`)
* The native mask is authoritative and was never overridden. `accepted=false` occurred in 0 steps for P1 and 1 step each for the two P2 conditions — the terminal-on-entry step of the penalty-missing episode (S5 mechanism). No P1/P2 action was rejected otherwise.
* **No-op ADDs** (accepted, road network unchanged): median 0 except P2 sto 2 per episode. **ADDs with a water endpoint:** median 0 (P1 sto 1.5, P2 sto 2); ADDs adjacent to destination/carpark: 0. So the synthetic-only "unknown legality" edge classes (water, window boundary) were not a practical obstruction natively; the policy used almost no bridges.
* **House-edge swaps** (an ADD touching a house tile that removes the previous edge): median swaps / median ADDs = 130/132 (P1 det), 0.78 (P1 sto), 0.76 (P2 det), 0.47 (P2 sto) — i.e. swapping still dominates all four conditions, least for P2 sto. Road-state oscillation steps (s_t = s_{t−2} ≠ s_{t−1}): median 121 (P1 det), 57 (P1 sto), 122 (P2 det), 102 (P2 sto); maximal A,B,A,B runs 3.5 / 8 / 4.5 / 15. Cycling is therefore **not** removed by the curriculum; the policy connects despite it, mostly early in the episode.
* Unmasked-logit diagnostics (fraction of steps whose unmasked top-1 is illegal: 0.99–1.0 in all conditions) are **not interpretable**: logits of masked-out actions receive no gradient in masked PPO, so they are arbitrary. They give no evidence for or against a legality preference; I do not use them.
* After the first connections the legal action set shrinks sharply (e.g. P2 sto seed 507919: 147 → ~8–12 legal actions from decision ≈ 30 on). This is consistent with the 25-piece road budget being used up (only house-edge swaps and removals remain legal; S7 rule) but concrete was not logged, so this is an inference (LIKELY, not confirmed).
* More houses: the median maximum house count per episode is 4 (P1, both modes), 4.5 (P2 det) and 8.5 (P2 sto), up to 17 in the longest P2 episode; S8 training had a single house.

## 7. Representative trajectories (`raw/`)
* **P2 sto, seed 531676 (k0 = 7), success:** connected at decision 16, 146 deliveries, survived to the 600-decision horizon, 17 houses, 13 simultaneous connections; kinds of the first 30 actions: 21 ADDs, 1 REMOVE, 8 more ADDs — steady construction, remaining distance to the first house 0 from decision ≈ 20.
* **P1 sto, seed 500000 (k0 = 7), the best baseline episode:** connection at decision 26, 36 deliveries, game over at 293. **P1 det, same seed:** 11 WAITs, then 130 house-edge swaps in 139 ADDs; remaining distance 7 → 5, no connection, 0 deliveries.
* **P2 failure, seed 523757 (k0 = 10), both modes:** no connection; remaining distance stuck at 6–8 (det; 142 swaps in 152 ADDs, 5 cycles) and reached 2 (sto; 96 swaps in 135 ADDs) — a sequential failure plus swap cycling.
* **P2 det, seeds 507919 and 539595:** the first connection occurs at decision 145 of ≈ 150 (game over follows immediately) — a late completion, zero deliveries.

## 8. Hypotheses
* **H9a (successful long-route transfer): supported, exploratorily.** P2 connects on 11/12 seeds (both modes) versus 1/12–3/12, including 4/5 long-route (k ≥ 8) seeds where P1 connects on none; deliveries and survival rise accordingly. Caveats: 12 seeds, a handful of native placements (initial k between 5 and 11), single training seed for P2, "connections" include later-spawned houses.
* **H9b (partial transfer): partially.** Edits are far more directed than P1 (first connection after a median 21–23 ADDs; P2 det irrelevant ADDs before the connection median 4), but efficiency is still poor and 55–76 % of ADDs are house swaps.
* **H9c (transfer failure): not supported.**
* **H9d (downstream limitation): partly supported.** After connecting, P2 delivers (median connection→delivery 11–12 decisions) yet 23 of the 24 P2 episodes still end in game over (one reached the horizon); survival rises from ≈ 150 to 180–314 decisions, not to unlimited play; the shrinking legal set (probable road-budget exhaustion), spawning of many more houses and continued swapping are plausible but unproven causes.
Failure categories for P2's non-successes (indicators, not a single attributed cause): sequential failure and cycling (seed 523757), late completion (two P2 det episodes), downstream (all game-overs after deliveries). **Legality mismatch** is not indicated by anything observed (no rejected actions, almost no water/boundary use); **distribution shift** (more houses, spawn timing, budget exhaustion) is present but its causal role is not isolated.

## 9. Recommendation
Because the curriculum policy transfers, the controlled next step toward full-game learning is a **small native PPO fine-tuning pilot starting from this checkpoint**, using the S4 infrastructure and the S9 protocol as the readout — not a larger synthetic study:
1. Arms: N1 = P2 (S8-C) initialisation, N2 = P1 (S1 seed 13) initialisation, same PPO hyperparameters, native normal start, native reward only; 10,240 decisions each, one seed each, run in parallel (≈ 1–1.5 h at the measured native speed), evaluated on a fresh set of ≥ 12 seeds at 0 and final decisions with the S9 metrics (connections, deliveries, time to first connection, swaps, `terminal_penalty_missing`), not on return.
2. Before that, one cheap addition worth approving: log `concrete` and the number of legal ADDs per step (measurement only), to settle whether road-budget exhaustion explains the legal-set collapse, and decide whether the missing terminal penalty should be fixed (reward change — needs approval) before any native reward-driven learning.
3. Optional low-cost confirmation (≈ 12 min): repeat this inference-only comparison on 24 further unused seeds with P2 only versus the same P1 data, to firm up the effect size; and a future synthetic extension of S7 with several spawning houses and a finite road budget, because swapping and shrinking legality are the dominant observed gaps.

## 10. Reproduction
```
python experiments/s9_prechecks.py                         # checkpoint / interface checks (no game)
python experiments/s9_runner.py --twin                     # instrumentation non-interference (native, ~30 s)
python experiments/s9_runner.py --who p1det|p1sto|p2det|p2sto --deadline-min 22 --stop-new-after-min 18    # four processes in parallel
python experiments/s9_analyze.py
```
Source hashes (prefix): `s9_runner.py a8d55499`, `s9_prechecks.py 819f2dcf`, `s9_analyze.py 28a90e7a`, `s7_oracle.py ad311411`, `anticycle.py 82254f32` (used read-only for road signatures/cycle counts), `diagnostics.py 5e760d30`.
