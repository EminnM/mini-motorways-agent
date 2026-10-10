# S7 — Native-grounded long-route task family: Gates A, B, C (preparation only)

**Status: Gate A passed (11/11), Gate B done, Gate C prepared. No PPO training of any kind was run** (no native training, no synthetic training; the throughput benchmark times components only and never calls `learn()`). No native game/reward change; S1–S6 artifacts untouched (S7 is a separate family in `python/synthetic_rl/s7*.py`; the S1/S3 generators and manifests are not modified).
All data/configs/raw results: `research/s7_long_route/`. Native wall-clock used: ~25 s (one capture run of 24 seeds, read-only except two tiny ADD/REMOVE concrete-cost probes).

## 0. What the native game turned out to look like (grounding captures, 24 normal-start seeds)
* The map terrain is **identical across seeds** (127 water tiles = 16.3%, no mountain/rail). Trees (8) do not block (68/94 tree-touching edges legal; the policy observation encodes trees as 0).
* A normal start has one 2×2 destination with a carpark pair; one carpark tile carries a road **stub** to a road tile (goal tile). The first house appears after the first decision and **already has a driveway edge** to a neighbouring tile. Neither pre-existing edge costs concrete.
* ADD legality: both endpoints inside a per-scene **build window** (a tile set of ~86–112 tiles around the house/destination), not destination/carpark, not both water; edges with one water endpoint are often legal (bridge-like; 1 bridge in the globals). Edges touching the house tile are legal and **replace** the driveway (S5). REMOVE: legal only where the action tile has exactly one road edge; house edges never removable; the stub is removable only from the carpark tile and then has no effect (19/19 native).
* Road budget: concrete 25, 1 per ADD, +1 per REMOVE (native probe 25→24→25).
* Native placements need **4–12 required ADDs (median 7, mean 7.4)**; house-to-goal Manhattan distance 5–13 (median 8). Later in an episode more houses spawn (e.g. a second house at decision ≈11 in seed 205907) — not modelled.

## 1. Native-to-synthetic distribution comparison (`native_vs_synthetic.json`, `obs_equivalence.json`, `replay_check.json`)
| | native (24 captures) | S7 (frozen test manifests) |
|---|---|---|
| required ADDs | 4–12, median 7 | by design 2 / 3 / 5 / 7 / 10 (test n = 88 / 100 / 100 / 93 / 32) |
| house→goal Manhattan | 5–13 (median 8) | k=2: 2, k=3: 4, k=5: 6, k=7: 7, k=10: 11 (medians) |
| legal ADDs at start | 135–176 (median 162) | medians 175–182 (superset by ~10%) |
| legal REMOVEs at start | 0–1 | 0–1 |
| terrain / window sizes / water | fixed map, 86–112-tile windows, 16.3% water | the same data (copied) |
| scenes / house tiles | 21 distinct scenes, 11 distinct house tiles | 21 scenes; houses drawn from all dry window tiles (more varied than native spawning) |
| first-action behaviour of the 3 frozen S1 CNNs on the identical start state | native-encoded observation vs S7 observation: **0 differing entries** over all 24 states | same first-action kind 24/24 (s11), 24/24 (s12), 23/24 (s13); differences come only from the ~10% mask superset |

Meaningful differences (documented, not fixed): no spawning of further houses, no time pressure / overcrowding game-over / −10 terminal penalty / traffic / upgrades; synthetic mask over-permissive on window-boundary and one-water edges (ADD precision 91.6%, recall 100% after the adapter fix below); bridges are not limited to 1; tests of observation equality used the slimmed capture, which omitted per-tile destination-demand and vehicle-count fields, so channels 12–14 are **not** compared (zero in the S7 obs; likely zero at episode start).
**Behavioural check (teacher-forced replay of the native deterministic-CNN trajectories, S5):** seed 182150: the S7 argmax equals the native action on 108/141 steps (77%); seed 205907: 15/141 — it agrees for the first 10 decisions (both WAIT), then the native game spawns a **second house** (decision 11) which S7 lacks, after which states diverge (46 native actions are illegal in S7 because they target that second house). So S7 reproduces the start-state decision but not the later dynamics.
**Adapter fix made after this check (before any training):** v1 forbade every water-endpoint edge, but native policies use them (recall 95.2%); the rule "both endpoints in the native window, not both water" reproduces 100% of native-legal ADDs on the 24 captures. v1 results are archived in `v1_water_blocked/`; the preregistered horizon rule was fixed before either version's results. (The v1 baselines had been observed when this fix was made; the fix was motivated by the native replay, not by the baseline numbers, and the baselines below are the v2 re-run.)

## 2. Generator and oracle validation — Gate A, 11/11 pass (`gate_a_results.json`)
2,872 tasks (train 2,180 / validation 279 / test 413; disjoint by (scene, house)).
1. Every task solvable and the claimed minimum correct: independent **networkx** shortest path on a graph built only from the live legal-ADD mask + roads decoded from the observation equals the stored k, the generator's Dijkstra and the route length on all 2,872; the shortest route replayed in forward / reverse / random order (8,616 replays) yields rewards 0,…,0,+1 and terminates exactly at the k-th ADD.
2. Exhaustive search through the real `env.step` (all legal actions incl. WAIT/REMOVE): k=2 minimum found = 2 on 30/30; for k=3 no sequence of ≤2 actions wins on 4 tasks (84,667 env steps). For k≥5 the lower bound is the shortest-path argument (a winning sequence must add every non-initial edge of a house–carpark path).
3. **No single-action shortcut:** 500,680 single legal actions from the start states of all tasks: 0 reward or terminate.
4. Rewards: one required edge withheld → reward 0, not terminal, not connected (4,726 routes); WAIT-only episodes truncate at 10·k with reward 0; completion reward exactly +1 and terminal.
5. Connectivity independence: env connectivity == observation-only BFS == networkx components == (reward>0) on 52,524 visited states; route-minus-one-edge never connects.
6. Native-mask parity on the 24 captures: ADD recall 100%, precision 91.6%; REMOVE sets identical 24/24; WAIT legal 24/24; upgrade slots illegal 24/24.
7. House edge replacement (edge count and concrete unchanged), house edges not removable, stub REMOVE rule and no effect, budget 25 then ADD only where it replaces the house edge.
8. **No leakage:** observation unchanged when k / task id / split metadata change; bit-identical to an independent rebuild from the scene file + (house, driveway); all non-native channels zero; changing only the driveway direction changes 4 entries, all road channels; the env stores no route.
9. Alternative routes accepted: a route avoiding one edge of the shortest route completes with +1 on 575/575 tasks; every task has ≥2 shortest routes (median 6, cap 60). Note: the open lattice makes shortest routes plentiful, so "which route" is not constrained.

## 3. Reward and observation leakage
Reward: +1 exactly at house–carpark connectivity, 0 otherwise, no shaping; observation: native 11,666 floats, action space 3,111; see tests 4, 5 and 8 above.

## 4. Baselines by route length (Gate B; frozen test manifest, preregistered horizon 10·k, no training)
Success rate (%) — n per length: 88 / 100 / 100 / 93 / 32. WAIT 0 everywhere.

| policy | k=2 | k=3 | k=5 | k=7 | k=10 |
|---|---|---|---|---|---|
| RANDOM LEGAL (5 seeds) | 2.0 | 0.6 | 0.2 | 0.0 | 0.0 |
| S1 CNN s11, deterministic | 54.5 | 42.0 | 7.0 | 9.7 | 0.0 |
| s11 stochastic (2 seeds) | 80.7 / 83.0 | 76 / 77 | 34 / 33 | 14.0 / 10.8 | 0 / 0 |
| s12, deterministic | 75.0 | 51.0 | 25.0 | 2.2 | 6.2 |
| s12 stochastic | 59.1 / 58.0 | 40 / 43 | 29 / 24 | 3.2 / 5.4 | 3.1 / 0 |
| s13, deterministic | 80.7 | 44.0 | 6.0 | 0.0 | 0.0 |
| s13 stochastic | 90.9 / 90.9 | 89 / 80 | 51 / 56 | 19.4 / 19.4 | 0 / 3.1 |

(Wilson 95% CI examples, s13 stochastic seed 0: k=5 [41,61]; k=7 [13,29]; k=10 [0,11].) The k=10 cell rests on 32 tasks.

## 5. Partial progress and efficiency (s13 as the best checkpoint; the others in `baselines_*.json`)
| k | 2 | 3 | 5 | 7 | 10 |
|---|---|---|---|---|---|
| s13 stochastic: success with exactly k ADDs | 26% | 8% | 1% | 0 | 0 |
| mean irrelevant ADDs / episode (s13 sto) | 4.7 | 9.0 | 23.7 | 41.4 | 53.5 |
| mean construction efficiency k/#ADDs on successes | 0.56 | 0.42 | 0.39 | 0.30 | – |
| max route progress reached (s13 sto) | 0.94 | 0.93 | 0.70 | 0.54 | 0.44 |
| RANDOM max route progress | 0.18 | 0.22 | 0.33 | 0.41 | 0.48 |
| first ADD reduces remaining distance (s13 sto / random) | 0.47 / 0.05 | 0.29 / 0.01 | 0.26 / 0.05 | 0.16 / 0.12 | 0.19 / 0.16 |
| WAIT share (s13 det / sto) | 0.36 / 0.03 | 0.74 / 0.04 | 0.73 / 0.10 | 0.62 / 0.15 | 0.66 / 0.28 |
Observations: (i) success falls steeply with k, and at k ≥ 5 successes need many irrelevant edits (efficiency ≤ 0.4, ≥ 24 irrelevant ADDs), i.e. they look like coverage plus local completion, not targeted construction; (ii) the "max progress" metric is **not** discriminative at long k — RANDOM reaches 0.41–0.48 by flooding edges, the same as the CNN at k=10; success (not progress) separates them (RANDOM 0% vs CNN up to 19% at k=7); (iii) deterministic policies WAIT for most of an episode once no local gap is visible (36–93% WAIT; s12 instead swaps house edges 23–77 times per episode at k = 5–10); sampling matters; (iv) these success levels are consistent with the one native stochastic success in 4 episodes (S4) — S7 at k≈7–10 predicts roughly 0–20%.
**Gate B conclusion:** Gap-7 and Gap-10 are not solved, neither efficiently nor reliably (k=7 ≤ 19%, k=10 ≤ 6%), so a curriculum experiment is still scientifically meaningful.

## 6. Training configurations A / B / C (`configs/arm_{A,B,C}.yaml`, `experiments/s7_train.py`; dry-run validated, NOT run)
Common to all arms: identical `SpatialMaskablePolicy` (73,416 parameters), PPO hyperparameters of S1/S3 (n_steps 512, batch 256, epochs 4, target_kl 0.03, lr 3e-4, ent 0.01, γ 0.99), reward +1 / 0, no intermediate or oracle signal, no demonstrations, validation manifest every 10,000 decisions (deterministic), TEST manifest once on the final checkpoint, **same total decision budget** (proposed 100,000).
* **A direct:** random init, trained only on the target distribution T = uniform over required lengths {7, 10}.
* **B curriculum:** random init, stages [2] → [3] → [5] → [7] → T with the preregistered advancement rule: leave a stage when the last 100 training episodes have success ≥ 0.7 and ≥ 4,096 decisions were spent, or at 12,000 decisions regardless; the last stage runs to the end. (Deviation from the literal list 2→3→5→7→10: the last stage is T = {7,10} so that A and B/C end on the same training distribution.) Stage budgets are logged to `stages.jsonl`.
* **C pretrained curriculum:** B initialised from the frozen S1 seed-13 checkpoint (SHA 41c1e085…) via the strict loader of S4 (policy+value copied; fresh Adam, fresh timestep counter, constant learning rate so no schedule difference).
Matched compute: all arms receive exactly the same number of environment decisions; the differing allocation across stages is reported, not hidden. No claim of curriculum superiority will be made from unmatched budgets.

## 7. Throughput benchmark (`benchmark_*.json`; components only, no learning, throwaway weights)
| | solo | 3 processes in parallel (each) |
|---|---|---|
| S7 env, random actions | 8,021 steps/s | 4,300–5,600 steps/s |
| env + CNN stochastic inference | 403 steps/s | ~255 steps/s |
| one PPO minibatch (256) forward+backward+Adam | 1,087 ms | ~1,630 ms |
| predicted PPO training speed (512-step rollout + 4 epochs × 2 minibatches) | 51 decisions/s | 34 decisions/s (aggregate ~102) |
| predicted wall-clock per 100,000 decisions | ≈ 32 min | ≈ 49 min |
The solo prediction agrees with the measured S1 training speed (55 decisions/s). The 11 validation evaluations of a run add ≈ 6–8 min.

## 8. Proposed training budget and stopping criteria
* **Pilot:** 3 arms × 1 seed × 100,000 decisions, run as three parallel processes: ≈ 55 min wall-clock, ≈ 2.6 CPU-hours total. A second seed per arm only if the pilot is informative (≈ another hour). **Maximum authorised in this proposal: 6 runs × 100,000 = 600,000 decisions (≈ 2 h wall).** Anything larger (e.g. 300,000 per run) needs a new decision.
* **Abort a run:** NaN/inf; policy entropy < 0.05 nats for two consecutive evaluations; SB3 `approx_kl` > 0.15 for 3 consecutive updates; throughput < 15 decisions/s.
* **Interpretation rules fixed in advance (TEST, final checkpoints, Wilson CIs; per-seed results are reported separately):** curriculum helps (H7c) only if B or C beats A on k=7 and k=10 success by ≥ 15 points and the gain is not explained by more irrelevant edits (exact-k and efficiency reported); sparse reward too hard (H7d) if no arm exceeds its own step-0 validation success by ≥ 10 points at k ≥ 5 within the budget; a result is "inconclusive" otherwise and extends nothing automatically.
* First action after approval: a 512-decision smoke run of `s7_train.py` (the training loop itself, including the stage callback, has only been dry-run, not exercised).

## 9. Known transfer risks
1. **Spawns:** native adds houses over time; S7 has one house — policies trained here never see competing demands or houses appearing mid-route.
2. **No time pressure / terminal penalty / game over,** and a different horizon (10·k vs ~143 native decisions).
3. **Mask superset (precision 91.6%) and unlimited bridges:** the policy may learn to rely on actions the native game refuses.
4. **House-edge swapping** is modelled, so loops are possible in S7 too; deterministic evaluation will under-report.
5. **Window as a static tile set** (1 of 141 native ADDs fell outside the capture-time window); maps are one fixed terrain, so geometry generalisation beyond this map is untested.
6. **Coverage vs construction:** at long k success can arise from flooding edges (RANDOM progress equals CNN progress at k=10), so success must be read together with efficiency and exact-k metrics.
7. **Unobserved channels:** demand / vehicle channels were not compared with the native game (capture omitted them).

## 10. Hypotheses and recommendation
* **H7a (reusable heuristic, blocked by native distribution shift):** *not supported as the main explanation.* The native start observation is identical to the S7 start observation (0 differing entries) and the CNN's first action agrees (23–24/24); S7 zero-shot success at k≈7–10 (≤ 19%) matches the single native stochastic success (1/4 episodes). Unresolved: native deterministic ADD-heavy looping (93% ADD, 6% WAIT) versus S7 deterministic WAIT-heavy behaviour (36–93% WAIT) — the dynamics (spawns, changing channels) differ beyond the start state.
* **H7b (long routes need new sequential capability):** *consistent with Gate B* — success collapses with k and long-k successes are inefficient — but not tested.
* **H7c / H7d:** open; they are the pilot's question.
**Recommended first training experiment (needs approval):** the 3-arm, 1-seed, 100,000-decision pilot above (A vs B vs C, matched budgets), followed by an **inference-only native check** (S4/S6 tooling) of the final checkpoints on normal-start seeds. The smallest experiment that separates "genuine long-horizon limitation" from "synthetic/native mismatch" is arm **C alone** (≈ 35 min solo) plus that native check: if C solves k=7/10 efficiently in S7 but native deliveries stay near zero, the mismatch (spawns, time dynamics, deterministic behaviour) dominates; if C does not improve over its own zero-shot score, the limitation is learnability of long sparse sequences (H7d or missing capability H7b). Running A and B in parallel costs almost nothing extra and yields the curriculum comparison.

## 11. Reproduction (all inference/offline; no training)
```
python experiments/s7_capture_native.py --n 24          # native read-only capture (25 s)
python experiments/s7_build_scenes.py
python python/synthetic_rl/s7_tests.py                  # Gate A, 11 checks, writes manifests + PREREGISTERED_PROTOCOL.json
python experiments/s7_native_comparison.py; python experiments/s7_obs_equivalence.py; python experiments/s7_replay_check.py
python experiments/s7_baselines.py --who wait_random|cnn11|cnn12|cnn13          # Gate B
python experiments/s7_benchmark.py --tag solo           # Gate C component timings
python experiments/s7_train.py --config research/s7_long_route/configs/arm_C.yaml --seed 11 --dry-run   # prepared, not run
```
Manifest SHA-256 (v2): train `55fe4300…`, validation `eadd6712…`, test `6a9e8cd9…` (`manifests/SHA256.txt`).
