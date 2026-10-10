# S13 — Native Noise Floor + Multi-House Synthetic Environment

Scope kept: no PPO training, no checkpoint update, no native env/reward change, no new guard, no native run other than S13a (151 s of the 300 s allowed), no change to S7/S8 code, S14 **not started**.
Artifacts: `research/s13_multi_house/` (raw replicate logs, `noise_floor.json`, `scenarios.json`, `validation.json`, `fidelity_probe.json`, `baselines.json`, `baselines_episodes.json`, `exploits.json`, `colour_blind.json`, `throughput.json`, `S14_PROPOSED_CONFIG.yaml`).
Code (all new): `experiments/s13_runner.py`, `s13_noise.py`, `s13_build_scenarios.py`; `python/synthetic_rl/s13_env.py`, `s13_controller.py`, `s13_tests.py`, `s13_fidelity.py`, `s13_baselines.py`, `s13_colourblind.py`.

## 1. Part A — native noise floor (S13a)

**Run.** Frozen S8-C, stochastic, native mask, **no guard**, the same 12 S9 game seeds (500000 + 7919·i), `NORMAL_START`, `max_decisions=600`, `decision_delta=1.0`. Torch sampling seed = **6000 + i** (S11/S12 used 1000 + i). 151 s native wall-clock. Raw logs: `raw/replicate_<seed>.json.gz`.

Per seed (deliveries): S11 control / S12 control / **S13 replicate** | S11 guard / S12 conserve

| game seed | S11 ctrl | S12 ctrl | **S13 replicate** | S11 guard | S12 conserve |
|---|---|---|---|---|---|
| 500000 | 42 | 42 | 40 | 137 | 87 |
| 507919 | 32 | 31 | 14 | 108 | 50 |
| 515838 | 4 | 4 | 74 | 5 | 4 |
| 523757 | 0 | 0 | 0 | 0 | 0 |
| 531676 | 146 | 146 | 67 | 81 | 54 |
| 539595 | 101 | 61 | 138 | 106 | 154 |
| 547514 | 4 | 4 | 129 | 5 | 4 |
| 555433 | 45 | 37 | 13 | 41 | 43 |
| 563352 | 38 | 38 | 40 | 27 | 81 |
| 571271 | 68 | 68 | 28 | 72 | 97 |
| 579190 | 20 | 20 | 21 | 90 | 26 |
| 587109 | 52 | 52 | 13 | 63 | 46 |

Steps, first connection, peak connections, zero-concrete fraction, spawned-house-connected-within-60 and first divergence per seed are in `noise_floor.json`. Sampling seeds: S11/S12 `1000+i`, S13 `6000+i`; native init identical (`NORMAL_START`, 600 decisions, `decision_delta=1.0`). Metric note: the exact action-sequence agreement is 9/12 for S12-vs-S11 control (same sampling seed) and 0/12 for the replicate against either control.

Set summaries (12 episodes each):

| set | deliveries total | median | decisions median | zero-concrete frac (median) | spawned houses connected ≤60 |
|---|---|---|---|---|---|
| S11 control | 552 | 40 | 313 | 0.627 | 43/100 |
| S12 control | 503 | 37.5 | 313 | 0.635 | 39/96 |
| **S13 replicate** | 577 | 34 | 291 | 0.642 | 48/98 |
| S11 guard (treated) | 735 | 67.5 | 403.5 | 0.690 | 58/114 |
| S12 conserve (treated) | 646 | 48 | 348 | 0.630 | 45/101 |

**Two sources of variation, separated where the data allow.**

* *Native non-determinism with the sampling seed held fixed* (S12 control vs S11 control, identical seeds and policy): 9 of 12 action sequences identical; 3 diverged at decisions 241, 12 and 253, with delivery differences −1, −40, −8. So the native game alone produces a visible difference in 25 % of re-runs.
* *Sampling-seed variation* (S13 replicate vs S12 control, same game seed, independent sampling seed): trajectories diverge at decision 0–3 in every seed (it is a different action sequence, not a perturbation). Delivery difference: sum +74, **median −1, median |diff| 31.5, SD 54.9**, range −79 … +125 (e.g. seed 515838: 4 vs 74; 547514: 4 vs 129; 531676: 146 vs 67); 5 higher / 1 equal / 6 lower. Decision-count difference: median |Δ| 118 decisions.
  Sampling variation therefore dwarfs native non-determinism, and single-seed outcomes on this task are very noisy.

The data do not allow a finer decomposition (no repeated runs with identical sampling seed beyond the S11/S12 pair), and none was added.

**Reassessment of S11 / S12.**

| comparison (treated − control) | Σ deliveries | median | higher/equal/lower | Σ without the 3 largest gains |
|---|---|---|---|---|
| S11 guard − S11 control | +183 | +2.5 | 8/1/3 | −58 |
| S11 guard − S13 replicate | +158 | +21 | 7/1/4 | −102 |
| S12 conserve − S12 control | +143 | +6 | 7/3/2 | −38 |
| S12 conserve − S13 replicate | +69 | +23 | 8/1/3 | −88 |

* Treated outcome lies **outside** the range of the three control realisations on 6/12 (S11) and 7/12 (S12) seeds. If treated and control were exchangeable draws, a fourth draw falls outside the range of three about 50 % of the time (2/4), i.e. ≈6/12 — which is what is observed.
* The aggregate gains (+183, +143) are no larger than about one SD of the 12-seed sum of ordinary control-to-control differences (≈ 55·√12 ≈ 190). This is a rough scale, **not a confidence bound** (12 paired differences, heavy-tailed, three realisations).
* The paired medians (+2.5 / +6 against the matched control; +21 / +23 against the independent replicate) are not stable across control choices, and the aggregate gain collapses to negative when the three largest gains are removed — a few episodes dominate.
* **Conclusion:** the S11/S12 delivery and survival improvements are *not distinguishable from sampling variability* with the data available. S11's mechanical result (521 destroyed connections → 0 under the guard) is a property of the veto rule and stands; it did not translate into a demonstrated performance gain. No effect of either guard on deliveries is established, and none is refuted. No additional control replications were run.

## 2. Part B — S13b multi-house synthetic environment

`python/synthetic_rl/s13_env.py` (`MultiHouseEnv`). Separate from S7 (S7/S8 files untouched; verified by test 15). 37×21 grid, 15 tile channels, 11 globals, 11,666 floats, `Discrete(3111)`, native-compatible masks.

### 2.1 Specification
* **Scenarios** (`scenarios.json`): 36 = 12 native game seeds × 3 realised native spawn timelines (S13 replicate, S12 control, S12 conserve). Each holds the seed's initial structures/stub edge and a list of timed spawn events (house or destination group, tiles, colour group, free driveway edge). Spawns are replayed on a fixed schedule (they were seed-driven in native); if the agent has built on a spawn site the structure is shifted by the smallest offset that fits (0 skipped spawn events when all 36 scenarios are replayed with WAIT; skips were not counted inside the other baseline episodes). Terrain = S7 water tiles. Splits by **game seed**: test seeds 515838, 539595, 563352, 587109 (fixed before any evaluation); validation 507919, 579190; train the other six (see S14 config).
* **Concrete**: 25 at start; plain (non-house) ADD −1; house-edge ADD free; REMOVE +1; top-ups +20/+30/+10/+20 at decisions 141/281/421/561 (measured, time-driven, identical across seeds/policies); ADDs without concrete are masked (house-edge swaps stay legal).
* **Edges**: Active / Mothballed. Adding a different edge at a house makes the new edge Active and the old Active edge Mothballed (an earlier Mothballed edge is deleted); re-adding the Mothballed edge reactivates it and deletes the displaced Active one. Connectivity uses Active edges only; destinations and carparks connect by adjacency; a house counts as connected iff its component contains a destination of its own colour group.
* **Mask**: ADD legal iff both endpoints in the build window, neither endpoint is a destination/carpark, not both water, not two houses; REMOVE only on a tile with exactly one edge, never a house edge; stub edges: REMOVE offered from the carpark tile and has no effect (native quirk, measured). Build window: rectangle `max(14,w+3) × max(8,h+3)` anchored one tile left of and level with the top of the structure bounding box, fitted to native legal-ADD extents.
* **Demand** (per destination, per decision): +1 with probability 0.22 from zero / 0.055 otherwise; −1 with probability `DOWN[n][d]` where n = connected same-colour houses (0…4+) and d = demand 1 or ≥2 (table = empirical, `s13_env.py`, from ≈14k native destination-steps). Overcrowding flag at demand ≥ 7; game over after 50 consecutive decisions with any destination overcrowded.
* **Deliveries**: every connected house delivers with probability 0.0436 per decision (the measured native rate, linear in connected houses, independent of pins).
* **Reward**: `r_t = deliveries_t − 10·[game over at t]`. Continuing, delivery-based; no per-edge, distance, shortest-path, road-count or progress term. Horizon 600.
* **Observation**: native encoding: static terrain, content type, "has group" flag (identical for houses and destinations — **colour is not observable**), road states (Active 2/3, Mothballed 3/3 as in the native encoder), destination demand (min(d/50,1)) and overcrowding flag, native cumulative-hour/day/week/score/concrete globals. Vehicles, mountains, rails, lights, roundabouts = 0.
* **Incompatibility check (spec: stop if essential state is inexpressible):** all mechanics are expressible in the existing channels. One native limitation is carried over rather than fixed: the native encoder carries no colour, so the house↔destination assignment is hidden from the policy (§2.5). This is not a new incompatibility with frozen S8-C; it is the same information the policy has in native play.

### 2.2 Native-grounding fidelity matrix

| mechanic | native evidence | synthetic implementation | fidelity |
|---|---|---|---|
| Initial concrete 25; plain ADD −1; house-edge ADD 0; REMOVE refund | S5–S12 logs (concrete deltas by action class) | exact costs; REMOVE always refunds 1 | **exact** (cost); refund **approximate** (native +1 in ≈82 %, 0 otherwise) |
| Weekly top-ups 20/30/10/20 at decisions 141/281/421/561 | 12 replicate + 24 S12 episodes, identical | same | **exact** for the horizon 600; rule behind amounts unknown |
| Construction rejected without concrete | native mask (S12) | masked | **exact** |
| Active/Mothballed, replacement and reactivation | S11 probe (1,375/1,375 predictions matched) | implemented as measured | **exact** (state semantics) |
| Functional connectivity = Active only | S11 | same | **exact** |
| House tiles are leaves with one Active edge | S11/S12 | same | **exact** |
| Build window | 1,436 native legal-ADD extents | fitted rule; median max-edge error 1 tile, p90 4, exact window 12 %; legal-ADD count 1.37× native median (p10–p90 1.10–1.71×) | **approximate** (slightly too permissive) |
| REMOVE only on dead-end tile; stub quirk | S5/S6/S12 | same | **exact** |
| Water: edge illegal only if both endpoints water; bridges | native mask; S9 | same; no bridge-count limit | **approximate** (bridge budget not modelled; native bridge global stays 1) |
| Spawn timing/positions/colours | 36 native timelines | replay of a real timeline; relocation if blocked | **approximate** (timeline set is small and was produced under S8-C-driven games; spawns appear seed-driven but this is not proven independent of the player) |
| Demand dynamics | 14k destination-steps from 12 S8-C episodes | empirical transition table | **approximate** (fitted on one policy's states; may shift for other policies) |
| Overcrowding threshold 7, ≈50-decision grace | 12/12 episodes, median 54.5, min 49 | 7 and 50 exactly | **approximate** (grace length is a median) |
| Deliveries linear in connected houses, 0.0436 | 3,867 decisions | Bernoulli 0.0436 | **approximate** (per-house rates vary 0.035–0.064 by state) |
| −10 terminal penalty | missing in ≈9 % of native terminal episodes (unresolved race) | always applied | **approximate** (deliberately consistent) |
| Destinations without colour group (12 events) | native logs | never generate demand | **unknown** (role unclear) |
| Traffic / vehicles / road capacity | native vehicles observed | none | **unsupported** |
| Weekly upgrade choices, bridges as a resource, roundabouts/lights | native forced upgrade at week boundaries | none | **unsupported** |
| Observation colour | native encoder: only a has-group flag | same (hidden) | **exact** |
| Concrete top-ups after decision 561 | not observed within 600 decisions | unused (+20 per 140 if the horizon were extended) | **unknown** |

### 2.3 Validation (`validation.json`, `python/synthetic_rl/s13_tests.py`): **15 / 15 passed**
External BFS/union-find (not the env's own connectivity) on hand-built scenarios:
1 multi-edge route connects a house (0,0,1 houses connected after each edge; env = external) · 2 second house spawns after first is connected, first stays connected · 3 concrete: plain −1, house-edge swap 0, WAIT 0, REMOVE +1, illegal action 0 · 4 no plain ADD with 0 concrete (masked), house-edge swap still possible · 5 top-ups (141,20),(281,30),(421,10),(561,20) · 6 swap makes (M,A), reactivation (A,—), third edge deletes the Mothballed one · 7 Mothballed route edges give no connectivity and 0 deliveries in 30 decisions · 8 connecting a second house through the existing trunk keeps the first connected at every step and leaves its edges unchanged · 9 deliveries 10/10 → 0/10 after replacing the critical driveway → 10/10 after reactivating; a dead-end branch removal leaves them intact; mid-route REMOVE is illegal by native rule (so the only way to destroy a trunk is a house-edge replacement — also the S11 failure mode) · 10 unmet demand: mean demand at t=100 and game-over fraction higher when disconnected than with two connected houses · 11 ignoring houses: 10/10 episodes end in game over, return exactly −10 (median 189 decisions) · 12 400 ADD/REMOVE steps: reward 0, concrete stays bounded · 13 observation: unused channels zero, observation byte-identical when hidden labels (colour, spawn time, scenario id) are changed · 14 shapes 11,666 / 3,111 / 37×21 / 15 / 11, float32 · 15 S7/S8 source files older than the S13 start, S7 env still 11,666/3,111, frozen checkpoint sha 85a9cb75cf287d24, 73,416 parameters, legal inference on both S7 and S13 observations.
Test-writing note: three tests initially failed because of mistakes in the tests (not the env) and were corrected; no environment rule was relaxed to pass a test. Test 9's original "remove a mid-route edge" was dropped because the native rule forbids it.

### 2.4 Reward and exploit tests (`exploits.json`; 36 scenarios each, 1 episode per scenario)

| policy | return mean (median) | deliveries median | survived 600 | new houses connected ≤60 |
|---|---|---|---|---|
| WAIT | −10.0 (−10) | 0 | 0/36 | 0/102 |
| endless random construction (never WAIT) | −10.0 (−10) | 0 | 0/36 | 0/100 |
| house-edge swap churn | −10.0 (−10) | 0 | 0/36 | 0/102 |
| **connect only the first house, then WAIT** | **0.7 (−1)** | 9 | 0/36 | 11/154 |
| reference controller | **150.8 (145.5)** | 145.5 | 29/36 | 287/295 |

Connecting only the first house scores ~0.7 versus ~151 for the controller and never survives; endless construction and swap churn score the same as WAIT. No exploit found among these three; this does not prove none exists (an adaptive learner is a stronger adversary than three hand-written ones).

### 2.5 Baselines, inference only (`baselines.json`; 36 scenarios × 3 environment seeds = 108 episodes; test split = 36)

| policy | return mean / median | deliveries median | decisions median | survived 600 | connected fraction by quarter | new houses ≤60 | zero-concrete frac | connections destroyed (median/ep) | plain ADDs (median) |
|---|---|---|---|---|---|---|---|---|---|
| WAIT | −10 / −10 | 0 | 170 | 0/108 | 0 | 0/277 | 0 | 0 | 0 |
| RANDOM (uniform over legal) | −10.0 / −10 | 0 | 170 | 0/108 | ≈0 | 3/273 | 0.46 | 0 | 88 |
| **frozen S8-C stochastic** | **27.2 / 7.5** | 17.5 | 261 | 15/108 | 0.29, 0.39, 0.39, 0.42 | 221/587 (38 %) | 0.54 | 41 | 56 |
| scripted controller | 141.7 / 153.5 | 156.5 | 600 | 74/108 | 0.89, 0.96, 0.95, 0.96 | 835/857 (97 %) | 0 | 0 | 26 |

Test-split values (36 episodes): S8-C return mean 16.6 / median −1.5, deliveries median 8.5, 4/36 survive, new houses ≤60 52/147 (35 %); controller return mean 141.9, 23/36 survive, 288/294.

**Solvability.** The (privileged) reference controller keeps ~96 % of houses connected, never loses a connection, uses ≈26 plain edges and survives 600 decisions in 69 % of episodes (the remainder are stochastic demand excursions: with several connected houses the demand random walk has near-zero drift, so overcrowding for 50 decisions still happens). The task is solvable by a policy that knows the colours.

**Does the synthetic S8-C resemble native S8-C?** (native control sets: S11/S12/S13 replicate)

| quantity | native S8-C | synthetic S8-C |
|---|---|---|
| deliveries median | 34–40 | 17.5 |
| decisions median | 291–313 | 261 |
| zero-concrete fraction | 0.63–0.64 | 0.54 |
| new houses connected ≤60 | 41–49 % | 38 % |
| connections destroyed per episode | ≈43 (S11: 521/12) | 41 (median) |
Qualitatively similar failure signature (heavy churn, concrete exhaustion, under half of new houses connected in time); the synthetic game is harder in raw deliveries (about half). Not a match in magnitude; it is not claimed to be.

**Colour-blind lower bound** (`colour_blind.json`): a controller that targets the nearest destination of *any* colour and cannot learn from failure reaches return 30.8 mean (median 14), survives 4/36, versus 141 for the reference controller on the same episodes. So the hidden colour assignment matters a lot; that bound is crude (the agent could, in principle, use demand/overcrowding channels to infer assignments, and the CNN is stateless).

### 2.6 Throughput (`throughput.json`; no optimizer)
Env alone, random policy: 1,807 steps/s. Env + frozen S8-C inference (1 thread): 324 steps/s → 100k decisions of rollouts ≈ 309 s. Batched inference-only: 779 obs/s. Optimizer time was not measured in S13; the S8 benchmark (3-way parallel, 2 threads each) predicted ≈ 2,100 s per 100k decisions including updates.

## 3. Part C — S14 (prepared, not run)
`S14_PROPOSED_CONFIG.yaml`: init from frozen S8-C, same CNN and PPO hyper-parameters, one seed (14), 100,352 decisions (196 updates), 45-minute cap (training + periodic evaluation), training on 6 game seeds (18 scenarios), validation on 2 game seeds (6 scenarios × 5 eval seeds), test on the 4 pre-declared game seeds, evaluated once. Preregistered selection: eligibility = S7 validation Gap-7/Gap-10 success ≥ 0.90 × step 0; then max fraction of newly spawned houses connected ≤60; ties → deliveries → fewer destroyed connections → earlier. Success: test-split gain ≥ 0.10 absolute in the primary metric in ≥ 9/12 scenarios, deliveries +25 % in ≥ 9/12, destroyed connections not higher, zero-concrete during spawns not higher, S7 test retention ≥ 0.90. Estimated cost ≈ 41–45 min (5 min rollouts + ≈28 min updates by the S8 benchmark + ≈8 min evals); the 45-minute cap includes evaluation so the estimate is at the cap. No native zero-shot evaluation is part of S14.

## 4. Limitations that bear on the decision
1. Spawn timelines and demand statistics come from native games played by one policy (S8-C) and, for 36 timelines, 12 game seeds. A policy that behaves very differently may face a different spawn/demand process natively; this is a model assumption, not a verified invariance.
2. Colour is hidden by the native observation. If the learnable part of the assignment problem is small, S14 gains will be capped (colour-blind bound above).
3. The window and demand models are approximations (matrix above); the synthetic game is harder than native in raw deliveries.
4. 6 training game seeds is a small map set; overfitting to those maps is possible. Held-out game seeds and the S7 regression are the guard.
5. S13a shows native single-seed outcomes carry SD ≈ 55 deliveries per episode, so any later native confirmation needs far more episodes than S9–S12 used.
6. Calibration leakage: the demand table, delivery rate and window rule were fitted on the S13 replicate logs of all 12 game seeds, including the four test seeds. The test split is held out for *policy* evaluation and for map/spawn timelines, not from the environment's fitted parameters; this is unlikely to matter at 14k samples but is not zero.
7. Unexamined: sensitivity of baseline rankings to the demand/delivery parameters (no parameter sweep was run).

## 5. Go / no-go

**Question: is the new synthetic environment sufficiently faithful, solvable and informative to justify training S8-C for sustained multi-house gameplay?**

**GO, conditional, for S14 as specified — a synthetic-only run; nothing about native performance is promised.**
* *Faithful enough:* every mechanic the brief required is implemented with documented fidelity; 15/15 independent tests pass; frozen S8-C shows the same qualitative failure signature as in native (churn ≈41 vs ≈43 per episode, 54 % vs 63 % zero-concrete, 38 % vs ≈45 % of new houses connected in time). Not faithful in magnitude (about half the native deliveries) or in traffic/upgrades.
* *Solvable:* the scripted controller connects 96 % of houses, destroys nothing, and survives 600 decisions in 69 % of episodes at ≈26 plain edges; WAIT, random, endless construction and first-house-only fail.
* *Informative:* S8-C is clearly below the controller (return 27 vs 142; 15/108 vs 74/108 survival), so there is headroom; reward has no shaping and the three exploit probes failed.
* *Why only conditional:* (a) colour is not observable, so part of the headroom may be unreachable for a stateless CNN (colour-blind bound 31 vs 141); a small S14 gain should therefore be read as a possible ceiling, not as proof the approach fails; (b) S13a shows the native game cannot confirm modest gains with the sample sizes used so far, so S14 can only be judged on synthetic held-out scenarios and the S7 regression, and a native claim needs a separately approved, much larger evaluation.
* *Not blocking:* no fidelity or reward problem found that justifies a NO-GO. The weakest points (demand model fitted on one policy; permissive window) are listed above.

**Proposed S14 run:** exactly `S14_PROPOSED_CONFIG.yaml`; cost ≈ 41–45 min wall-clock, one seed, no native game; stop at 100,352 decisions, the 45-minute cap, non-finite loss/KL abort, or two consecutive S7-regression failures; checkpoint chosen by the preregistered rule on validation data only; test split evaluated once.

**STOPPED here. S14 has not been started and needs separate approval.**
