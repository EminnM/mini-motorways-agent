# S20 — Goal-conditioned road-construction distillation (behaviour cloning of the S18-H executor's CONSTRUCTION skill)

Scope kept: no PPO / RL, no native game, no checkpoint touched, no existing file modified, no interaction with the running S19 experiment, 2 CPU threads (`torch.set_num_threads(2)`), one compact model, one training configuration, no sweep. Everything new lives under `research/s20_construction_distillation/` (`code/{common,scenes,demos,model,rollout,train,evaluate,checks}.py`, `data/` scenes + demonstrations, `runs/` checkpoint, curves, evaluation, checks) and this report. Train wall-clock is reported in §4 (the 20-minute box was used in two pieces, see there).

What is distilled: the executor's **construction skill only** — given an explicit goal (one house tile + the destination tiles of its colour), emit the legal primitive ADD actions that connect it while preserving existing functional connections, and say DONE when the goal is functionally connected. The planner's **strategy** (which house to serve, when to wait, when to remove, resource reasoning) is NOT in the data: the executor is wrapped so that it can only ever plan CONNECT for the given house (`common.GoalExecutor`, built from the real `connect_candidates(..., houses=[goal])` + `score_all`; checked in §6).

## 1. Dataset

### 1.1 Scene generator (`code/scenes.py`)
Flat-map curriculum on the S13 synthetic map (`MultiHouseEnvV2`, v2 observation, `lam = mu = 0` so no demand / deliveries; concrete set to 80 so resources never bind). One map seed → one scene:
* a land region (A = x∈[−2,7]×y∈[−6,10], B = x∈[−14,−4]×y∈[−5,10], C = x∈[−17,18]×y∈[3,10]; the S16 region x∈[−9,9]×y∈[−6,4] is NOT water-free — the river column x=−3 runs through it — so three water-free rectangles are used instead);
* 2–4 destination blocks (carpark column + 2×2 destination, S16 `_dest_block`; distinct colours among 4 native ids; extra blocks are t=1 events because the S13 env merges all destination tiles of one struct list);
* 2–6 one-tile houses (≥2 colours, every house colour has a destination);
* pre-existing networks: the S18-H executor (GoalExecutor) run for a random subset (0..all) of the non-goal houses, one at a time in random order — so maps contain several connected networks, shared segments, other-colour roads and building obstacles;
* 0–3 random legal plain road stubs (redundant / disconnected, away from houses);
* an optional prefix of the goal route: `correct` (planner prefix for the true colour, 1–5 edges, always shorter than the route) or `wrong` (planner prefix built while the goal house temporarily carries a decoy colour — an incorrect prefix the policy must recover from);
* acceptance: goal house disconnected, planner route length (path tiles − 1) in [2, 14], ≥ 1 ADD needed, oracle-reachable.

**Water**: the env's build window is at least 14×8 and always reaches the river, so water tiles are inside every window (`water_in_window_frac = 1.0`). The guarantee kept instead: no road in any demonstrated state touches water (scenes with a pre-built or prefix road on water, or a planner route through water, are rejected: `water_road_prebuild` 203 + `water_route` 14 of 1,800 train seeds). Bridges / motorways / upgrades / traffic / delayed removal / pending spawns do not exist in these scenes (the S13 env has no delayed-removal or pending mechanics for roads at all; this is a documented limitation of the synthetic env, not something excluded by sampling).

Splits BY MAP SEED (asserted disjoint, none equals an S13 native test seed): train 20001–20900 ∪ 24001–24900 (the second range was added because the acceptance rate is ~62 %), val 21001–21150, test 22001–22400.

| split | seeds | scenes accepted | rejected (route-len / water / goal already connected) | regions A/B/C | houses mean | pre-connected houses mean | existing edges mean (max) | existing-edge bins 0–9/10–19/20–29/30–39/40+ | prefix none/correct/wrong |
|---|---|---|---|---|---|---|---|---|---|
| train | 1,800 | 1,115 | 384 / 217 / 84 | 423/436/256 | 3.93 | 1.14 | 17.0 (54) | 291/431/264/108/21 | 577/354/184 |
| val | 150 | 102 | 32 / 9 / 7 | 35/41/26 | 4.21 | 1.51 | 19.6 (56) | 25/31/25/14/7 | 63/21/18 |
| test | 400 | 232 | 88 / 54 / 26 | 71/91/70 | 3.96 | 1.13 | 16.7 (47) | 59/95/47/26/5 | 119/75/38 |

Goal route length (path tiles − 1) is spread over 2..14 in every split (train histogram 2:52, 3:81, 4:108, 5:119, 6:119, 7:116, 8:123, 9:78, 10:81, 11:81, 12:56, 13:49, 14:52; test 2:8 … 14:16). Planner construction steps (ADDs needed at task start, = route length minus reused and prefix edges) range 1..14 (train mean 5.56).

**Multiple-route fraction**: with the S16 oracle idea (every legal frontier ADD on some minimum-concrete route), ≥ 2 distinct shortest routes exist in 99.7 % / 99.0 % / 100 % of train / val / test scenes (mean valid-set size 3.6): on a Manhattan grid with concrete-only cost almost every task has several shortest routes, so the flag is uninformative and the evaluation breaks down by valid-set size (2–3 vs 4+) instead.

### 1.2 Demonstrations (`code/demos.py`)
Per scene the GoalExecutor is rolled for the single goal; per decision: goal-conditioned observation (v2 tiles 777×21 + 2 goal channels [goal house tile = 1, goal-colour destination tiles = 1] = 777×23, stored exactly as uint8 ×6, + the 11 globals as float32; total 17,882 floats), extended legal mask (3,111 env actions + DONE, index 3111), planner action, remaining concrete, done flag, oracle valid set. The terminal (connected) state is labelled DONE.

| split | trajectories | transitions | completed | ADD labels | DONE labels | WAIT labels | REMOVE labels | steps mean (max) | steps == planned ADDs |
|---|---|---|---|---|---|---|---|---|---|
| train | 1,115 | 7,310 | 1,115 (100 %) | 6,195 | 1,115 | 0 | 0 | 5.56 (14) | 1,111 |
| val | 102 | 682 | 102 | 580 | 102 | 0 | 0 | 5.69 (14) | 102 |
| test | 232 | 1,599 | 232 | 1,367 | 232 | 0 | 0 | 5.89 (14) | 230 |

WAIT frequency in the planner's construction behaviour: **0** (the planner never waits in these tasks; WAIT is kept legal for the learned policy). Generation: scenes 175 s, demonstrations 112 s.

## 2. Model (`code/model.py`)
`GoalPolicy`: 23-channel input on the 21×37 grid; trunk = 3×3 convolutions (32, 64, 64, 64, 64, 64) with dilations (1, 1, 2, 4, 4, 1) + ReLU (the S8-C pattern plus dilation so the receptive field, **27×27**, spans a 14-edge route); per-tile 1×1 head → 4 logits per tile in native order (E-add, E-remove, S-add, S-remove) = action indices 1..3108; [mean-pool, max-pool, 11 globals] → Linear(64) → ReLU → 4 logits [WAIT, upgrade0, upgrade1, DONE]. Output layout [WAIT, 3108 tile logits, upg0, upg1, DONE] = 3,112 logits, illegal ones masked to −1e9. **182,344 parameters.** Trained from scratch (no S8-C weights).

## 3. Training (`code/train.py`)
Cross-entropy of the planner action over legal actions, Adam lr 1e-3, batch 128, seed 0, best-val (imitation accuracy) checkpoint. Validation metrics: loss, imitation accuracy (masked argmax == planner action), valid-set accuracy (masked argmax ∈ oracle valid set; {DONE} for terminal states), DONE precision / recall.

Curve (val):

| phase | epoch | train loss | val loss | imitation acc | valid-set acc | DONE prec / rec |
|---|---|---|---|---|---|---|
| bc | 0 | 5.39 | 4.70 | 0.150 | 0.150 | 0.15 / 1.00 |
| bc | 1 | 4.01 | 3.43 | 0.161 | 0.167 | 0.15 / 0.82 |
| bc | 2 | 2.68 | 2.26 | 0.415 | 0.496 | 0.45 / 0.43 |
| bc | 3 | 1.79 | 1.75 | 0.532 | 0.606 | 0.56 / 0.61 |
| bc | 4 | 1.27 | 1.40 | 0.598 | 0.695 | 0.63 / 0.49 |
| bc | 5 | 0.93 | 1.13 | 0.672 | 0.746 | 0.61 / 0.53 |
| bc_resume | 0 | 0.80 | 0.94 | 0.733 | 0.805 | 0.60 / 0.82 |
| bc_resume | 1 | 0.61 | 0.90 | 0.748 | 0.805 | 0.56 / 0.88 |
| bc_resume | 2 | 0.50 | 0.86 | 0.767 | 0.831 | 0.67 / 0.75 |
| bc_resume | 3 | 0.41 | 0.83 | 0.783 | 0.848 | 0.67 / 0.86 |
| bc_resume | 4 | 0.36 | 0.82 | **0.793** | **0.848** | 0.58 / 0.96 |

Wall-clock: BC phase 605 s (6 epochs; it was CPU-starved by the dataset checks I ran concurrently, ~110 s/epoch instead of ~65 s), then the same run/configuration was resumed for the remaining box: 5 epochs, 328 s (fresh Adam moments, same lr/batch). **Total training wall 944 s ≈ 15.7 min**, 11 epochs; the loss was still decreasing when the box closed (under-trained, see §8).

Offline accuracy of the final checkpoint (`runs/offline_accuracy.json`):

| split | imitation acc | valid-set acc | non-terminal only: imitation / valid-set | DONE precision / recall |
|---|---|---|---|---|
| train | 0.897 | 0.937 | 0.882 / 0.929 | 0.80 / 0.98 |
| val | 0.793 | 0.848 | 0.764 / 0.828 | 0.58 / 0.96 |
| test (unseen seeds) | **0.819** | **0.876** | 0.797 / 0.863 | 0.68 / 0.95 |

The ~6-point gap between imitation and valid-set accuracy is the planner's arbitrary tie-break among equally short routes (valid-set size 3.6 on average): the policy often picks a different but equally good edge.

**DAgger**: rule = run one DAgger round only if the autonomous greedy completion rate on 100 train scenes is clearly (> 10 points) below the val imitation accuracy. After the BC phase: autonomous completion 0.74 vs imitation 0.67 → not triggered; after the resume: 0.77 vs 0.79 → not triggered. **No DAgger round was performed**; the code path (`train.dagger_collect`, relabelling every visited state with a fresh GoalExecutor plan) exists but was not exercised.

## 4. Autonomous rollouts on the 232 held-out TEST scenes (`code/evaluate.py`, `runs/eval_test.json`)
Paired task instances (same restored state, same goal, same budget = max(6, 3 × planner steps) for every policy). A rollout ends when the goal house is functionally connected (success), when the policy emits DONE (premature if not connected → failure) or at the budget. Illegal actions are counted and replaced by WAIT (none occurred). "DONE at completion" = the learned policy is queried once more in the connected state.

| policy | completion [95 % CI] | legal-action rate | premature DONE | DONE at completion | concrete per completed connection (planner cost on the same tasks) | steps / planner steps on completed | steps == planner | existing connections destroyed (tasks) | other-house driveway swaps | removes | water ADDs |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **A planner** (goal-conditioned S18-H) | **1.000** [1.000, 1.000] | 1.0 | 0 | — | 5.50 (5.50) | 1.000 | 99.1 % | **0** (0) | 0 | 0 | 2 |
| **B learned, greedy** | **0.677** [0.616, 0.737] | 1.0 (raw unmasked argmax legal 21 %) | **70** | 97.5 % | 5.27 (5.17) | 1.020 | 89.8 % | 3 (3) | 35 | 0 | 0 |
| B learned, sampling (T=1) | 0.582 [0.522, 0.647] | 1.0 | 95 | 94.1 % | 4.86 (4.73) | 1.029 | 85.2 % | 2 (2) | 33 | 0 | 1 |
| B-diag greedy, DONE masked (same checkpoint, termination only by connection / budget) | **0.918** [0.884, 0.953] | 1.0 | — | — | 5.81 (5.26) | 1.096 | — | 6 (6) | — | 0 | 0 |
| C frozen S8-C (v1 obs, **no goal**) | 0.091 [0.056, 0.129] | 1.0 | — | — | 5.24 (2.24) | 2.34 | 33 % | **30** (27) | 430 | 17 | 9 |
| RANDOM legal ADD/REMOVE | 0.000 [0.000, 0.000] | 1.0 | — | — | — | — | — | 22 (21) | 83 | 112 | 281 |

Paired per-scene differences (bootstrap over scenes, 2,000 resamples):

| comparison | mean Δ completion [95 % CI] | wins / losses |
|---|---|---|
| B greedy − A planner | −0.323 [−0.384, −0.263] | 0 / 75 |
| B-diag (DONE masked) − A planner | −0.082 [−0.116, −0.047] | 0 / 19 |
| B-diag − B greedy | +0.241 [+0.185, +0.297] | 56 / 0 |
| B greedy − B sampling | +0.095 [+0.039, +0.155] | 34 / 12 |
| B greedy − C S8-C | +0.586 [+0.517, +0.655] | 141 / 5 |
| B greedy − RANDOM | +0.677 [+0.616, +0.737] | 157 / 0 |
| C S8-C − A planner | −0.909 [−0.944, −0.871] | 0 / 211 |

**Reading.** 70 of the 75 greedy failures are premature DONE: the termination head, not the construction head, is the dominant failure. With DONE masked the same network completes 91.8 % of the test tasks autonomously, at 1.10× the planner's steps and +0.55 concrete per connection, destroying 6 existing connections in 232 tasks (all via driveway swaps of other houses — the planner's functional-connection veto is not learned, it is imitated statistically). Greedy decoding beats sampling. S8-C caveat: it receives the map but no goal, so its 9 % is "the goal house happened to be among the houses it connected within the budget" (it also destroys 30 existing connections and swaps other houses' driveways 430 times); the comparison is only meaningful as "what a goal-less primitive policy achieves on the same tasks".

### 4.1 Breakdowns (completion, n, 95 % CI)

By goal route length (path tiles − 1):

| route length | n | A | B greedy | B-diag DONE masked | B sampling | C S8-C |
|---|---|---|---|---|---|---|
| 2–4 | 46 | 1.00 | 0.74 [0.61, 0.87] | 0.91 [0.83, 0.98] | 0.72 | 0.24 |
| 5–7 | 78 | 1.00 | 0.69 [0.59, 0.79] | 0.94 [0.88, 0.99] | 0.62 | 0.10 |
| 8–10 | 50 | 1.00 | 0.72 [0.60, 0.84] | 0.98 [0.94, 1.00] | 0.60 | 0.00 |
| 11–14 | 58 | 1.00 | 0.57 [0.45, 0.69] | 0.84 [0.74, 0.93] | 0.41 | 0.03 |

By planner construction steps (ADDs needed at task start):

| planner steps | n | B greedy | B-diag | B sampling |
|---|---|---|---|---|
| 1–2 | 50 | 0.70 [0.56, 0.82] | 0.98 [0.94, 1.00] | 0.70 |
| 3–5 | 59 | 0.76 [0.64, 0.86] | 0.92 [0.83, 0.98] | 0.75 |
| 6–8 | 63 | 0.67 [0.56, 0.79] | 0.95 [0.89, 1.00] | 0.49 |
| 9–14 | 60 | 0.58 [0.47, 0.70] | 0.83 [0.73, 0.92] | 0.42 |

By number of pre-existing edges at task start (networks of other houses, stubs, prefix):

| existing edges | n | B greedy | B-diag | B sampling | C S8-C |
|---|---|---|---|---|---|
| 0–9 | 59 | 0.81 [0.71, 0.90] | 0.93 [0.86, 0.98] | 0.71 | 0.05 |
| 10–19 | 95 | 0.72 [0.63, 0.81] | 0.94 [0.88, 0.98] | 0.62 | 0.04 |
| 20–29 | 47 | 0.57 [0.43, 0.70] | 0.89 [0.79, 0.98] | 0.45 | 0.21 |
| 30+ | 31 | 0.45 [0.29, 0.61] | 0.87 [0.74, 0.97] | 0.42 | 0.13 |

Other cuts (B greedy / B-diag): valid-set size 2–3: 0.67 / 0.93, 4+: 0.68 / 0.91 (route ambiguity does not hurt); prefix none 0.63 / 0.90, correct prefix 0.77 / 0.96, **wrong prefix (recovery) 0.63 / 0.89** — the policy recovers from an incorrect prefix about as well as from no prefix; region A 0.72 / 0.93, B 0.66 / 0.92, C 0.66 / 0.90; 2/3/4 destination colours 0.69 / 0.65 / 0.72 (B-diag 0.92 / 0.89 / 1.00) — colour discrimination does not degrade with more colours. Generalisation to unseen map seeds: offline test accuracy (0.819) is above val (0.793) and the autonomous numbers above are all on unseen seeds; the train-subset probe (0.77 greedy completion on 100 train scenes) is in the same range as test (0.68), so there is no large train/test gap beyond the train/val offline gap (0.90 vs 0.79 imitation).

### 4.2 Failure examples (B greedy, `runs/eval_test.json: failures_B_greedy`)
seed 22002 (region B, route 7, 32 existing edges): 6 ADDs then DONE one edge short; 22003 (C, route 11, planner 10 steps): DONE after 8 steps; 22015 (C, route 14): DONE after 3 steps; 22024 (A, route 12 with a correct 5-edge prefix, 7 steps left): DONE after 1 step; 22011 (A, route 11): 16 steps used (budget 33) then DONE, with one other-house driveway swap. Pattern: premature DONE is most frequent on long routes and on crowded maps where the goal house's partial road already touches many roads — the global-pooled DONE head sees "lots of road near the goal" and fires; the per-tile construction head itself keeps choosing valid edges (B-diag). The 5 non-DONE failures and the 19 B-diag failures are budget exhaustion on 9–14-step routes on crowded maps (the policy oscillates between two valid frontiers or swaps the goal driveway repeatedly), and 6 B-diag tasks swapped another house's driveway.

## 5. Checks (`code/checks.py` → `runs/checks.json`, all pass)
| check | result |
|---|---|
| C1 no leakage: train/val/test seed sets disjoint (1,800 / 150 / 400 seeds), none equals an S13 native test seed, every scene's seed in its split | pass |
| C2 goal channels contain only the goal house tile and the goal-colour destination tiles | pass on all 9,591 recorded transitions |
| C3 no strategic choice imitated: every plan in every demonstration targets the goal house (1,449 / 1,449 trajectories); 0 REMOVE labels; 0 WAIT labels | pass |
| C4 masks: every label legal under its recorded mask; uint8 ×6 tile encoding exact; `restore_env` reproduces generation-time obs + mask + metadata (20/20 re-generated scenes); 0 illegal actions emitted by any policy in 6 × 232 test rollouts | pass |
| C5 DONE labelled iff the goal is functionally connected, exactly once per trajectory, at its end | pass |
| C6 `GoalExecutor(goal=None)` emits the same 60-action sequence as the original `Executor` on 3 full-planner episodes (the wrapper changes nothing but the candidate set) | pass |
| C7 planner label ∈ oracle valid set on all 8,142 non-terminal transitions (the lexicographic planner route is always a minimum-concrete route) | pass |

Note on A's `water_adds = 2`: in 2 test tasks the executor (k_alternatives = 2) chose an alternative route containing a bridge edge where the scene filter had checked only the alt-0 route; the learned policy built 0 water edges.

## 6. Design (not implemented): the learned executor as an optional action family for strategic RL
* Action families for a strategic agent: **PLAN** (call the S18-H planner for a chosen house; the planner returns a candidate, the deterministic executor builds it), **LEARNED BUILD** (give the learned executor a goal = house + colour and let it run until DONE / budget; its DONE head is the family's termination signal), **EDIT** (single primitive ADD/REMOVE), **WAIT**, **INFRASTRUCTURE** (bridge / motorway / upgrade choice — typed, unsupported until the integration exposes them; §3 of the S18-H report). The strategic policy chooses the family and the goal; the family executes a macro-action over several decisions.
* Planner usage is a learned choice, not an annealed schedule: PLAN and LEARNED BUILD are both always available; their relative use is whatever the strategic return prefers (PLAN is exact but costs compute and is rule-bound; LEARNED BUILD is cheap and can be improved with RL). No curriculum that forces the learned family.
* No shaping against non-transport roads: the strategic reward stays the game reward (deliveries, game over); roads that serve no connection are not penalised directly — the concrete counter already makes them costly. The learned executor's own objective stays imitation (+ optional DAgger / RL fine-tuning on completion), never a road-count penalty.
* Safety wrapper for LEARNED BUILD: mask DONE until the goal is connected OR keep a step budget (3× the planner's estimate, as here), and veto any ADD that the S11 functional model predicts to disconnect a connected house — §4 shows both are needed (premature DONE, 6 destroyed connections).

## 7. Answers
1. **Can a neural policy learn the planner's basic construction skill?** Yes: 182 k parameters, 7.3 k transitions, 11 epochs → 82 % exact-action and 88 % valid-set accuracy on unseen seeds, 0 illegal actions, and 92 % autonomous completion when termination is handled externally.
2. **Does it generalise to maps with many existing roads?** Partly: completion falls from 0.81 (0–9 existing edges) to 0.45 (30+) with the learned DONE, but only from 0.93 to 0.87 with DONE masked — crowding mostly confuses the termination head, not the routing head.
3. **Does it complete tasks autonomously rather than imitate single actions?** Yes for construction: 92 % full completions without planner intervention (DONE masked), 68 % with its own DONE; on completed tasks it needs the planner's exact step count 90 % of the time and recovers from wrong prefixes (0.89).
4. **How much efficiency is lost vs the planner?** On completed tasks: +2 % steps and +0.1 concrete (greedy with own DONE), +10 % steps and +0.55 concrete per connection (DONE masked); 3–6 destroyed existing connections per 232 tasks vs 0 for the planner; 32 (own DONE) or 8 (masked) points of completion.
5. **What fails first as complexity grows?** Premature DONE (70 of 75 greedy failures), then budget exhaustion on 9–14-step routes (B-diag 0.83 vs 0.92–0.98 for shorter), then other-house driveway swaps on crowded maps; route ambiguity and colour count do not hurt.
6. **Is it ready as an optional tool for strategic RL?** As a construction primitive wrapped with an external termination rule and the functional-connection veto: yes (92 % completion, cheap); as a self-terminating macro-action: not yet (DONE precision 0.68).

## 8. Limitations
* Synthetic only (S13 env: no traffic / vehicles, no capacity, demand turned off here); no delayed-removal / pending mechanics exist in the env, so none are covered; the window-based legality rule is the fitted approximation.
* Water: inside every build window but never under a demonstrated road; the policy's behaviour on bridges is untested (0 water ADDs observed).
* One model, one configuration, one seed; no hyper-parameter or architecture search; the model was still improving when the time box closed (train loss 0.36, val 0.82) and the first BC phase was slowed by concurrent checks; the resume used fresh Adam moments.
* No DAgger round (rule not triggered), so the compounding-error question is answered only by the autonomous rollouts; the DONE head was trained on 1 positive per trajectory (13 % of labels) with no class weighting — the cheapest next fix is an external termination rule or a DONE-specific loss weight, not a bigger model.
* The valid-set oracle uses concrete-only cost; the planner's secondary length tie-break is not part of "valid" (so valid-set accuracy is slightly generous for equal-concrete longer routes).
* The S8-C comparison is interface-limited: v1 observation (no colour), no goal, stochastic as in S13.
* Scene acceptance is ~60 % (route-length and water filters); the curriculum is deterministic per seed but the rejected seeds bias the accepted set towards shorter-than-region-diagonal routes.

STOPPED here: no S21, nothing native, no RL.
