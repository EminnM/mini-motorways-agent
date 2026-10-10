# Native RL replication — two independent hybrid PPO runs vs S31 (environment `postfix_v8`)

Preregistration: `research/s34_replication/PREREGISTRATION.md` (written before any run). Code: `research/s34_replication/code/{train_rep,eval_rep,analyze_rep}.py`; S31 wrapper/policy unchanged. Data: `research/s34_replication/{runs,eval_dev,eval_fresh,results.json}`. (Directory is named `s34_replication`; the subsequent matched-EDIT ablation brief reuses the label "S34" and lives in `research/s34_matched_edit_ablation/` — see `MATCHED_EDIT_ABLATION_S34.md`.)

## 0. Verdict
**The S31 result does not replicate as a robust effect. Preregistered decision: NOT robust.** Three training runs with identical architecture, hyper-parameters, reward, seeds-distribution and 100,352-decision budget, differing only in the PPO seed (31 / 131 / 231), give 100k stochastic advantages over planner B on the 8 fresh seeds (1,500-decision horizon, mean over 3 rollouts) of **+318 (S31), +114 (rep1), −20 (rep2)** deliveries against B's 567; seeds higher 6/8, 6/8, 4/8. The rule required every run to beat B with ≥5/8 seeds and the worst run to be ≥ half the best; rep2 fails both parts. Across-run SD of the advantage (≈170) is larger than the mean advantage (≈137). Deterministic 100k: +260 / 0 / −30. Legality, reward accounting and environment version were valid in all 541 evaluation episodes and 670 training episodes (0 illegal actions, Σr = score + penalty everywhere, `postfix_v8` throughout).

## 1. What was run
* **Training**: `train_rep.py` = S31 `train.py` with only run name, instance prefix and PPO seed parametrised; pre-flight asserts the deployed DLL hash (2aa42367…) and the `playableAreaAtTime` export. rep1 (seed 131) and rep2 (seed 231), 100,352 decisions each, checkpoints 0/24,576/49,152/73,728/100,352, all preserved. Wall 150 / 152 min (4 headless instances each, run one after the other).
  * *Attempt 1 aborted* (archived in `runs/aborted_attempt1/` with README): both runs launched concurrently (8 instances + 2 trainers on 12 cores) each lost one worker (rep2 at 26,820 decisions, rep1 at 44,896) and hung in cleanup; stderr was lost because the launch went through a pipe. Cause attributed to the env's 120 s IPC timeout under contention. Both runs were restarted from scratch with the same seeds, sequentially and detached with file logging; attempt 1's partial checkpoints are not used anywhere.
* **Evaluation A — development benchmark** (12 designated seeds, 600-decision cap, det + sto with S31's torch seeds): 50k and 100k of all three runs, post-fix; planner B post-fix (S33) as baseline.
* **Evaluation B — decision benchmark** (8 fresh seeds 900000 + 7919·i, verified unused; 1,500-decision horizon fixed in advance; det ×1 and sto ×3 rollouts per seed with torch seeds 20000 + 1000·r + i, identical across policies): planner B; 50k and 100k of all three runs. Two S31 rollout-2 processes crashed at launch time (one 120 s IPC timeout, one `no_active_episode` STEP error) while other instances were being started; they were re-run from the lost seed onward with the same torch seeds. All 200 fresh-benchmark episodes are present.

## 2. Training trajectories (training seeds, stochastic collection, 600 cap; thirds of each worker's log)

| run | third | score mean | len | surv 600 | destroyed/ep | PLAN/EDIT/WAIT |
|---|---|---|---|---|---|---|
| S31 (seed 31) | 1 / 2 / 3 | 90.6 / 94.0 / 103.3 | 445 / 447 / 468 | 11 / 15 / 21 | 28.9 / 16.6 / 14.1 | 0.59→0.68 / 0.39→0.29 / 0.02 |
| rep1 (seed 131) | 1 / 2 / 3 | 97.6 / 92.8 / 94.6 | 456 / 447 / 453 | 16 / 11 / 13 | 31.0 / 27.5 / 23.7 | 0.54→0.58 / 0.43→0.41 / 0.02 |
| rep2 (seed 231) | 1 / 2 / 3 | 94.5 / 84.3 / **77.2** | 451 / 430 / 413 | 12 / 5 / 5 | 36.8 / 34.7 / **39.6** | 0.54→**0.34** / 0.44→**0.64** / 0.03 |

S31's signature (destroyed connections falling, PLAN share rising, score rising) appears only in S31. rep1 is flat. rep2 *drifts the other way*: it edits more and more (EDIT share 0.64 by the last third), destroys more and scores less over training, with normal-looking PPO diagnostics (KL ≈ target, explained variance 0.81, entropy stable). The optimisation is stable but what it converges toward depends on the seed.

## 3. Development benchmark (12 designated seeds, 600 cap, vs planner B = 615; CI = bootstrap 95 % on per-seed mean)

| run / ckpt | det total (Δ, H/E/L) | sto total (Δ, H/E/L, CI) | destroyed det / sto |
|---|---|---|---|
| S31 50k | 844 (+229, 6/2/4) | **1206** (+591, 8/2/2, [+16, +78]) | 117 / 184 |
| S31 100k | 770 (+155, 6/4/2) | 1110 (+495, 8/1/3, [+14, +70]) | 191 / 59 |
| rep1 50k | 692 (+77, 4/5/3) | 872 (+257, 6/1/5, [−3, +49]) | 6 / 221 |
| rep1 100k | 673 (+58, 4/6/2) | 995 (+380, 7/1/4, [−5, +67]) | 88 / 355 |
| rep2 50k | 853 (+238, 6/2/4) | 601 (−14, 6/2/4, [−33, +25]) | 272 / 402 |
| rep2 100k | 540 (−75, 3/2/7) | 899 (+284, 7/2/3, [−7, +53]) | 206 / 531 |

Only S31's cells have CIs excluding zero. These seeds informed the S31 go/no-go and the "50k is best" choice, so they are a development benchmark here, not evidence.

## 4. Decision benchmark (8 fresh seeds, 1,500 horizon, vs planner B = 567)

Planner B: 40 / 35 / 39 / 47 / 85 / 122 / 126 / 73, every episode ends by game-over between decisions 290 and 528; no policy of any kind reached the 1,500 horizon in any of the 200 episodes, so the cap was never binding and all gains are again episode-length gains.

| run / ckpt / mode | total (sto: Σ of per-seed means) | Δ vs B | H/E/L | CI95 mean | rollout totals | within-seed SD | destroyed / 24 eps |
|---|---|---|---|---|---|---|---|
| S31 50k det | 508 | −59 | 5/0/3 | [−29, +5] | — | — | 236 (8 eps) |
| S31 50k sto | 976 | +409 | 6/0/2 | [+10, +97] | 961 / 1065 / 903 | 30.7 | 484 |
| **S31 100k det** | 827 | +260 | 5/0/3 | [−6, +71] | — | — | 87 |
| **S31 100k sto** | 885 | **+318** | 6/0/2 | [+4, +81] | 741 / 796 / 1119 | 50.4 | 531 |
| rep1 50k det | 685 | +118 | 3/1/4 | [−18, +65] | — | — | 365 |
| rep1 50k sto | 890 | +323 | 4/0/4 | [−0, +91] | 813 / 1011 / 845 | 35.9 | 728 |
| **rep1 100k det** | 567 | **0** | 2/1/5 | [−31, +33] | — | — | 254 |
| **rep1 100k sto** | 681 | **+114** | 6/0/2 | [−18, +39] | 729 / 529 / 785 | 34.5 | 778 |
| rep2 50k det | 860 | +293 | 4/0/4 | [−21, +107] | — | — | 287 |
| rep2 50k sto | 726 | +159 | 7/0/1 | [−5, +42] | 718 / 671 / 789 | 37.5 | 1344 |
| **rep2 100k det** | 537 | **−30** | 4/0/4 | [−32, +24] | — | — | 377 |
| **rep2 100k sto** | 547 | **−20** | 4/0/4 | [−28, +19] | 568 / 581 / 492 | 31.4 | 1133 |

**Training-run variation vs rollout variation (kept separate, nothing pooled):**
* Across the 3 runs, 100k sto advantage over B: {+318, +114, −20}, SD **170**; 100k det: {+260, 0, −30}, SD 159; 50k sto: {+409, +323, +159}, SD 127.
* Within a run, across the 3 stochastic rollouts of the same seed: SD **31–50 deliveries per seed** (≈ ±90–140 on an 8-seed total). Single rollout totals of the same checkpoint range e.g. 741–1119 (S31 100k) and 529–785 (rep1 100k).
* So one training run's single evaluation can differ from another's by ~300 deliveries purely from seed-of-training, and a single rollout can swing ~±200 from sampling. The S31 numbers were one run and one rollout per seed and sat at the favourable end of both distributions.

**Preregistered decision rule** (every run's 100k sto beats B with ≥5/8 seeds higher, worst ≥ ½ best): advantages {+318, +114, −20}, higher-seeds {6, 6, 4} → **robust = False**.

## 5. Other measurements
* Survival to horizon: 0 / 200 fresh episodes for every policy; planner B 0 / 8. Episode length medians (sto, 100k): S31 494, rep1 436, rep2 367 vs B 366.
* PLAN/EDIT/WAIT (fresh, 100k sto): S31 0.59/0.37/0.04, rep1 0.50/0.47/0.02, rep2 0.25/0.70/0.04. More editing tracks worse outcomes across runs.
* Destroyed functional connections per 24 episodes (100k sto): S31 531, rep1 778, rep2 1133 (planner B 0).
* Illegal actions: 0 everywhere. Boundary-crossing actions cannot occur under `postfix_v8` (mask) and were additionally asserted in the S33 regression; the fresh evaluations ran under the same build (hash asserted at start of every process).
* Operational: launching several game instances simultaneously while others are mid-STEP caused two evaluation crashes (120 s IPC timeout at launch; one `no_active_episode` STEP error) and the attempt-1 worker deaths; staggered, detached launches ran 500+ episodes without incident. Stochastic rollouts are not bit-reproducible even with the same torch seed, because the game's rare state transient (S21) desynchronises the sampling stream on some seeds — relevant for the ablation's common-random-numbers design.

## 6. Interpretation
1. The hybrid's edge over planner B is real for S31's weights but is **a property of that particular run, not of the method**: with the same recipe, one replication is a coin flip (rep1: 6/8 seeds higher but CI spans zero and one rollout below B), the other is indistinguishable from or slightly worse than planner B and strictly more destructive.
2. What varies across runs is how much the policy edits and how destructive those edits are; what is common to all runs is that no policy learned anything that changes the throughput regime (deliveries/decision pinned, no colour separation, no survival past ~650 decisions on any fresh seed).
3. The learning signal that is reproducible is weak: all three runs keep KL/entropy/value diagnostics healthy, yet only one moves the training return in the right direction. 100k decisions at this reward is apparently inside the regime where PPO seed dominates outcome.

## 7. Recommendation on further investment in this hybrid (as trained)
**Not robust enough to justify building on these weights.** Before any further training budget is spent the open question is *where* S31's advantage comes from — learned selection of edits, learned timing of PLAN vs EDIT, or merely stochastic perturbation of planner B — which is the subject of the matched-rate random-EDIT ablation (`MATCHED_EDIT_ABLATION_S34.md`). No new training was started.
