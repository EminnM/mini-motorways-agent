# S19 — Hybrid hierarchical RL over S18-H candidates (synthetic MultiHouseEnv)

Scope kept: no native game launched, no existing file modified, no checkpoint touched, no S8-C initialisation, no training sweeps. All new files are under `research/s19_hybrid_rl/` (`code/{strategic_env,policy,train,evaluate,checks}.py`, `runs/s19/{train_log.json, best.pt, last.pt, ckpt_*.pt, train_stdout.txt}`, `evaluation.json`, `evaluation_episodes.json`, `checks.json`, `checks_stdout.txt`) plus this report. One training configuration, one seed (19), 20-minute hard budget. The S18-H modules (`research/hierarchical_planner/code/*`) and `python/synthetic_rl/*` are imported read-only.

## 0. Headline answers

| question | answer |
|---|---|
| (1) Does hierarchical RL learn to select meaningful investments? | Only weakly. The policy learned a legal, non-degenerate selection rule (never WAITs when a house can be connected, prefers the cheap route: zeroing the cost features costs −8 median deliveries / −3 survivals), but its held-out performance is statistically indistinguishable from choosing uniformly at random among the same candidates. |
| (2) Can it compete with the greedy planner? | It matches it, it does not beat it: test deliveries median 171.5 (greedy argmax) / 163 (stochastic) vs A 163 / B 168.5; paired C−A mean +15.0 [−1.9, +34.7], C−B +8.3 [−7.2, +26.9]; survival 25/36 vs 24 (A), 27 (B), 30 (random strategic). All CIs include zero. |
| (3) Does it discover behaviour that differs from the heuristic? | Marginally: on real-choice states C agrees with A's exact candidate 57.5 % of the time and with A's target house 96 %; 125 of its 138 disagreements are a different *route* to the same house (it selects a Dijkstra alternative in 13 % of CONNECTs and lays 31.5 vs 25 plain edges). Nothing in the outcome metrics shows this is better. |
| (4) What prevents improvement? | The strategic task, not the trainer: 92 % of strategic states have exactly one legal action (WAIT), the remaining 8 % have 2–4 near-equivalent routes, and the random strategic policy D already reaches 176.5 median deliveries / 30/36 survival. Survival is dominated by the demand random walk (S13 §2.5), so the per-episode noise (SD ≈ 55 deliveries) swamps any route-choice effect. The candidate set (CONNECT/WAIT only; REMOVE candidates never existed in these episodes) leaves no strategic lever to pull. |
| (5) Minimum next step toward native traffic-aware strategic learning | Expose a native per-candidate traffic signal (vehicle count / trip time on the route tiles, available in the raw IPC obs channel 14) and a candidate kind whose effect is *not* already determined by connectivity (motorway or reroute of a congested link); without a new action kind and a traffic-dependent reward, a learned selector has nothing to learn over the S18-H heuristic. |

## 1. Environment definition (`code/strategic_env.py`)

* **Base**: `MultiHouseEnv` (S13b; reward +1 per delivery, −10 once at game over, horizon 600, top-ups +20/+30/+10/+20 at decisions 141/281/421/561). Scenarios: the 36 S13 scenarios; `split` field: test game seeds 515838/539595/563352/587109 held out (12 scenarios), validation game seeds 507919/579190 (6 scenarios, used only for checkpoint selection), train = the other 18.
* **Visible state only**: the policy sees the S18-H `WorldModel` (built from `env.content/houses/dests/edges/free/concrete/total/t/water/action_masks()`) plus `env.score`. `env.pending` and the scenario event list are never read (the only env readers are `WorldModel.__init__`, `global_features(wm, env.score)` and the gym plumbing in `StrategicEnv.reset/step`).
* **Strategic decision point**: whenever no plan is executing and no WAIT interval is running. Candidates = S18-H `candidates.generate(wm, k_alternatives=2, banned, protected, include_motorway=False)` minus its WAIT object: CONNECT routes for every disconnected house (Dijkstra, cost lexicographic (concrete, length), k = 2 alternatives by edge-penalty re-search) and REMOVE-redundant dead-end edges (if any). The executor's oscillation windows (`banned`/`protected`, 30 decisions) are passed through unchanged.
* **Candidate ordering rule (stable geometric key, never score)**: `(kind, house x, house y, goal x, goal y, route length, first added edge (x, y, dir), full edge tuple)`; REMOVE after CONNECT, keyed by its edge. Cap K = 24 slots (truncation never occurred: 0 truncated candidates in training and evaluation); slot 24 = WAIT, always legal.
* **Validity mask**: a CONNECT slot is legal iff `feasibility == 'feasible'` (cost ≤ concrete) and `confidence > 0` and `not would_destroy_functional_connection`; REMOVE legal iff it destroys nothing. Infeasible/destroying candidates stay visible (features) but masked.
* **Commit**: selecting a candidate hands it to `CommittedExecutor` (a subclass of the S18-H `Executor` with the heuristic `plan()` call removed; all S18-H guarantees kept: one legal primitive action per decision re-checked against the env mask, would-destroy veto, oscillation ban/protect, invalidation reasons `house_gone / spawn_on_route / road_state_changed / edge_no_longer_buildable / would_destroy / mask_changed / remove_invalid`). The plan runs until completed / abandoned / invalidated. WAIT advances the env by a FIXED interval of 5 primitive decisions.
* **Step return**: `(obs', R_undiscounted, terminated, truncated, info)` with `info['duration']` = primitive decisions elapsed (≥ 1 always), `info['reward_disc'] = Σ_j γ^j r_j` (within-step discounting), `info['kind']`, `info['slot']`, `info['outcome']`. Per strategic action the controller records `(t0, slot, kind, candidate key, alt index, cost, route length, n_legal, duration, outcome)`.
* **Primitive-policy adapter** `StrategicPolicy(chooser).act(env) -> action` lets the identical `s13_baselines.episode`-style metric code score every policy (A, B, C, D, WAIT).

### Observation / features
Global (12): concrete/25, concrete/total, decisions-to-next-known-top-up/140, top-up amount/30, t/600, houses connected/10, houses unconnected/10, max destination demand/7, mean demand/7, overcrowding flag, functional (non-stub Active) edges/60, score/300.
Per candidate (16): is_connect, is_remove, concrete cost/10, route length/20, new edges/20, reused Active edges/20, newly served houses (connectivity delta)/3, served group's destination demand/7, served group overcrowded, disruption flag (would break a functional connection), travel-cost proxy (path length)/20, feasible, affordability margin (concrete − cost)/10, waited since the house spawned (min 60)/60, alt index/2, reuse fraction. Ablation groups: cost = {cost, margin}, demand = {group demand, overcrowded}, disruption = {flag}.

## 2. Policy (`code/policy.py`)
Shared MLP: global context `h_g = tanh(MLP(g))` (64); candidate encoding `h_c = tanh(MLP([c ; h_g]))` (2 × 64) → one logit per slot; WAIT logit `MLP(h_g)`; value `MLP([h_g ; mean_legal(h_c) ; max_legal(h_c)])`. Masked logits (−1e9). Parameters 20.5 k. Equivariance verified numerically (CH1): permuting slots permutes the logits with max |Δ| = 0.0, WAIT logit Δ = 0.0, value Δ = 2.4e-7.

## 3. SMDP handling and trainer (`code/train.py`)
PPO-clip with GAE, written for the semi-Markov setting:
`δ_t = R_t + γ^{d_t} V(s_{t+1}) − V(s_t)`, `A_t = δ_t + γ^{d_t} λ A_{t+1}`, `R_t = Σ_{j<d_t} γ^j r_j` (γ per primitive decision, λ per strategic transition; at λ = 1 this is exactly the discounted SMDP return — CH2 unit test: error vs brute force 6.6e-8, hand-computed case 4.8e-8, reduction to standard GAE for d ≡ 1 5.7e-8). Terminal and horizon transitions bootstrap with 0 (the horizon is part of the state via t/600). Stock `MaskablePPO` was not used (it ignores durations).
Config: γ = 0.99, λ = 0.95, lr 3e-4, clip 0.2, entropy 0.01, value coef 0.5, grad-norm 0.5, 4 epochs × minibatch 256, 16 episodes per iteration (4 worker processes × 1 torch thread, scenarios cycled through the 18 train scenarios, random env seeds), advantages normalised, rewards × 0.1 for the optimiser only, seed 19, validation every 5 iterations on the 6 validation scenarios × 3 env seeds (7000+100·idx+s) with argmax; checkpoint selected by validation deliveries median (tie → mean). Hard stop at 1200 s.

### Training curves (`runs/s19/train_log.json`, 231 iterations, 3,696 episodes, 421,133 strategic transitions, 2.06 M primitive decisions, 1,203.7 s wall, 5.2 s/iteration)

| iterations | train deliveries mean | survival | entropy (real-choice states) | value loss | WAIT freq | CONNECT freq | REMOVE freq | mean duration |
|---|---|---|---|---|---|---|---|---|
| 0–19 | 157.5 | 0.79 | 0.96 | 0.225 | 0.922 | 0.078 | 0 | 4.89 |
| 20–59 | 159.6 | 0.80 | 0.84 | 0.125 | 0.922 | 0.078 | 0 | 4.89 |
| 60–119 | 157.1 | 0.79 | 0.67 | 0.127 | 0.922 | 0.078 | 0 | 4.89 |
| 120–179 | 159.3 | 0.80 | 0.62 | 0.119 | 0.921 | 0.079 | 0 | 4.89 |
| 180–231 | 157.5 | 0.81 | 0.62 | 0.113 | 0.922 | 0.078 | 0 | 4.89 |

Validation deliveries median by checkpoint: 147.0 (iter 0), then oscillating 131–147.5 with no trend; best = iteration 185 (147.5 median, 135.4 mean, 15/18 survived) — i.e. not better than the untrained network at iteration 0 (147.0, 15/18). Mean approx-KL per update 8e-5, clip fraction 0.09 %: the policy barely moved per update; it sharpened (entropy 0.96 → 0.62 on real-choice states) without changing outcomes. Mean entropy over all states is 0.04–0.13 because ~92 % of states have exactly one legal action.

## 4. Held-out test results (`evaluation.json`; 12 test scenarios × 3 env seeds = 36 episodes per policy, env seeds 7000+100·idx+s identical for every policy and identical to S13/S18-H)
Metric code: `evaluate.episode_ext` = verbatim copy of `s13_baselines.episode()` accounting + new-house latency, WAIT-primitive fraction, illegal-primitive count and the strategic summary; `summarize` = `s13_baselines.summarize` + extensions. A and B reproduce the S18-H table exactly (163 / 168.5, 24 / 27 survived).

| policy (test, 36 ep.) | deliveries median / mean | return mean | survived 600 | new houses ≤ 60 | latency mean (dec.) | destroyed | plain ADDs median | deliv./plain ADD | zero-concrete frac | abandoned plans | WAIT freq (strategic) | CONNECT / REMOVE | mean duration | alt routes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| WAIT | 0 / 0 | −10 | 0 | 0/88 | — | 0 | 0 | 0 | 0 | — | 1.0 (primitive) | — | — | — |
| A greedy planner (S18-H) | **163 / 148.0** | 144.7 | 24 | 280/280 (100 %) | 3.42 | 0 | 25 | 5.71 | 0 | 0 / 295 | n/a (0.932 primitive) | — | — | — |
| B resource-aware (S18-H) | **168.5 / 154.8** | 152.3 | **27** | 295/295 (100 %) | 3.43 | 0 | 25.5 | **5.87** | 0 | 0 / 305 | n/a (0.934 primitive) | — | — | — |
| C learned, stochastic | **163 / 153.4** | 150.6 | 26 | 311/313 (99.4 %) | 5.88 | 0 | 27 | 5.14 | 0 | 0 / 323 | 0.918 | 323 / 0 | 4.93 | 106/323 (33 %) |
| C learned, greedy (argmax) | **171.5 / 163.1** | 160.0 | 25 | 321/327 (98.2 %) | 5.83 | 0 | 31.5 | 5.61 | 0 | 0 / 325 | 0.921 | 325 / 0 | 4.93 | 43/325 (13 %) |
| D random strategic | **176.5 / 157.4** | 155.8 | **30** | 317/318 (99.7 %) | 10.04 | 0 | 25 | 5.31 | 0 | 1 / 326 | 0.919 | 326 / 0 | 4.93 | 106/326 (33 %) |

Eligible-house denominators differ because longer-surviving episodes see more spawns. The 3-decision latency gap between C/D and A/B is the fixed 5-decision WAIT interval (a spawn during a WAIT macro waits up to 5 decisions).

**Paired per-episode comparisons on the test split (bootstrap 95 % CI, 10,000 paired resamples)**

| comparison | deliveries mean diff [CI] | deliveries median diff [CI] | wins / losses / ties | survival diff | plain ADDs diff |
|---|---|---|---|---|---|
| C stochastic − A | +5.4 [−18.3, +30.2] | −0.5 [−5, +3] | 17 / 18 / 1 | +0.056 [−0.14, +0.25] | +2.7 [−0.9, +6.8] |
| C stochastic − B | −1.4 [−24.1, +22.1] | −1.0 [−8, +3] | 15 / 19 / 2 | −0.028 [−0.22, +0.17] | +1.8 |
| C stochastic − D | −4.0 [−22.8, +15.1] | −0.5 [−3, +3] | 15 / 18 / 3 | −0.111 [−0.25, +0.03] | −0.8 |
| C greedy − A | +15.0 [−1.9, +34.7] | +1.5 [−4, +7] | 19 / 15 / 2 | +0.028 [−0.17, +0.19] | **+3.8 [+0.8, +7.6]** |
| C greedy − B | +8.3 [−7.2, +26.9] | 0.0 [−10, +3] | 16 / 17 / 3 | −0.056 [−0.22, +0.11] | +2.9 |
| C greedy − D | +5.6 [−8.5, +22.7] | −2.0 [−5, +4.5] | 14 / 21 / 1 | −0.139 [−0.31, +0.03] | +0.4 |

The only paired difference whose CI excludes zero is that C greedy lays ~4 more plain edges than A (it picks longer alternative routes); no delivery or survival difference is resolvable. Per-game-seed medians (9 episodes each): seed 515838 A 94 / B 94 / C-st 97 / C-gr 94 / D 76; 539595 167 / 196 / 196 / 201 / 217; 563352 164 / 164 / 157 / 168 / 166; 587109 166 / 175 / 177 / 174 / 182 — the ordering changes from seed to seed.

Other splits (context; C's training data = train split): train (54 ep.) deliveries median A 170 / B 143 / C-st 146.5 / C-gr 146.5 / D 152.5, survival 43 / 38 / 36 / 39 / 40; validation (18 ep.) 142.5 / 146 / 139 / 147.5 / 141.5, survival 17 / 17 / 15 / 15 / 16. C is not better than random-strategic even on the scenarios it trained on.

## 5. Selection frequencies
Across the 36 test episodes C (greedy) took 4,120 strategic actions: 92.1 % WAIT, 7.9 % CONNECT, 0 % REMOVE; D random: 91.9 / 8.1 / 0. The WAIT share is structural: a state has more than one legal action only while a house is disconnected (325 of 4,156 states on C's trajectories = 7.8 %). REMOVE was never selected by any policy because no REMOVE candidate ever existed: the executor builds only routes whose plain edges all lie between a house driveway and a destination component, so there are never dead-end plain edges to remove (the same was observed in S18-H). Effectively the learned problem is "which of 2–4 CONNECT routes, or WAIT". C greedy selects a Dijkstra alternative (alt index > 0) in 43/325 CONNECTs (A: 4/9 in the smoke episode only on exact ties), the stochastic policy in 106/323; no candidate key was ever selected twice in an episode (no re-selection after abandonment; 0 abandoned plans).

## 6. Diagnostics and checks (`checks.json`; CH1–CH4, CH7 have pass criteria: 5/5 pass; CH5, CH6, CH8, CH9 are reported diagnostics)

| check | result |
|---|---|
| CH1 equivariance | pass (max logit Δ 0.0, WAIT Δ 0.0, value Δ 2.4e-7 over 20 random permutations) |
| CH2 discounting | pass (brute-force SMDP return at λ = 1: 6.6e-8; hand case 4.8e-8; d ≡ 1 reduces to standard GAE 5.7e-8) |
| CH3 train/test contamination | pass: the training log contains exactly the 18 train scenario ids; validation used the 6 validation ids; no test id appears anywhere in training or selection |
| CH4 action-mask correctness | pass: 0 masked candidates selected (C stochastic, C greedy, D; training log also 0 over 421 k transitions), 0 illegal primitive actions under the env mask for every policy (WAIT, A, B, C ×2, D: 6 × 36 × ~600 decisions), executor `illegal_suppressed` 0 |
| CH5 WAIT exploitation | WAIT chosen in **0 of 325** real-choice states (greedy) — the policy never idles when a house can be connected. Reward per primitive decision: WAIT steps 0.286 vs CONNECT steps 0.229, which reflects that WAIT steps occur when more houses are already connected (deliveries are independent of the action), not idling for reward. The −0.62 correlation between episode WAIT frequency and deliveries (stochastic) is a confound (episodes with more spawned houses have more CONNECT actions and more deliveries). No exploitation. |
| CH6 repeated equivalent routes | 0 repeated candidate keys in 36 + 36 episodes; alternatives selected 43/325 (greedy), 125/314 (stochastic); 0 abandoned plans |
| CH7 candidate-ordering leakage | pass: with candidate slots randomly permuted at every decision, all 36 test episodes produced identical decision sequences (by candidate key) and identical metrics (deliveries median 171.5 both) |
| CH8 policy collapse / agreement | see below |
| CH9 feature ablations | see below |

**Agreement (CH8).** On C-greedy's own test states (4,156 states, 325 with a real choice): exact-candidate agreement C = A 57.5 %, C = B 68.9 %, same target house as A 96.0 % (B 97.5 %), same kind 100 % (never WAIT vs CONNECT disagreements); for reference A = B 84.0 % on the same states. On A's states (A run as the chooser through the same executor, 316 real-choice states): C-argmax = A 61.4 %, C-argmax = B 70.9 %. Of C's 138 disagreements with A, 125 are "same house, different route", 13 "different house first", 0 involve WAIT. So C is *not* a copy of A (it is closer to B, whose scorer also rewards reuse/demand), but the difference is confined to route choice among near-equivalent alternatives.

**Feature ablations at inference (CH9, test split, greedy).**

| zeroed features | deliveries median (Δ) | mean (Δ) | survived (Δ) | new-house ≤ 60 | episodes with changed decisions |
|---|---|---|---|---|---|
| none | 171.5 | 163.1 | 25 | 98.2 % | — |
| cost (concrete cost, affordability margin) | 163.5 (−8.0) | 151.2 (−11.9) | 22 (−3) | 98.1 % | 23/36 |
| demand (group demand, overcrowded) | 171.5 (0) | 159.9 (−3.1) | 25 (0) | 97.2 % | 1/36 |
| disruption flag | 171.5 (0) | 163.1 (0) | 25 (0) | 98.2 % | 0/36 |

The cost features are the only ones the policy uses materially; the disruption flag is identically 0 for every legal candidate (destroying candidates are masked), so it carries no information by construction; demand changes one decision in 36 episodes.

## 7. Reward note (S13 reward vs native score)
Training return = S13 reward: +1 per delivery, −10 once at game over (continuing reward, no shaping). Natively the score counts deliveries, and the −10 terminal penalty is missing in ≈ 9 % of native terminal episodes (unresolved race, `NATIVE_NOISE_AND_MULTI_HOUSE_S13.md` §fidelity table), so the synthetic return is on average slightly more pessimistic than a native score-based return would be. Deliveries and survival are logged and reported separately from the training return everywhere above (training log fields `deliveries_mean`, `survival_rate` vs `return_raw_mean`; evaluation fields `deliveries` vs `return`); with 0 destroyed connections and 0 abandoned plans, return = deliveries − 10·[game over] in every episode here.

## 8. What prevents further improvement
1. **Strategic task (dominant).** 92 % of strategic states are WAIT-only; the remaining 8 % offer 2–4 CONNECT routes for at most 1–2 disconnected houses. Every policy that never idles — including uniform random over legal candidates — connects 98–100 % of new houses within 60 decisions and reaches the same delivery/survival band (D: 176.5 median, 30/36). Survival is then decided by the demand random walk (S13 §2.5), not by route choice; per-episode SD ≈ 55 deliveries means a 36-episode paired test resolves only differences ≳ 20 deliveries, and the true effect of route choice is far smaller (deliveries are independent of route length in the synthetic env).
2. **Candidate set.** CONNECT and WAIT are the only kinds that ever occur (REMOVE candidates never exist); every CONNECT candidate reaches the same house, so the only strategic degrees of freedom are house order (when ≥ 2 houses are disconnected, rare) and route length/reuse, neither of which affects the synthetic reward beyond concrete accounting — and concrete never ran out (zero-concrete fraction 0 for every policy).
3. **Training implementation.** The SMDP trainer is correct by unit test and the policy did sharpen (entropy ↓, cost features used); it is not the bottleneck: per-update KL 8e-5 and clip fraction 0.1 % show the advantage signal is essentially noise. 20 minutes / 3,696 episodes is small, but the validation curve shows no trend to extrapolate, and the task-level argument above says more data would not change the conclusion.
4. **Synthetic env.** No traffic, vehicles or capacity: route length is a proxy with no effect on delivery rate, so the one thing a learned evaluator could add over the heuristic (traffic awareness) has no signal here. The hierarchical scaffolding (candidates, executor, mask, SMDP credit assignment) is validated; the learning problem it exposes is empty.

## 9. Roadmap: motorways, bridges, roundabouts, traffic lights, congestion-aware rerouting as strategic candidates (no implementation)
Integration gaps (from `HIERARCHICAL_PLANNING_BASELINE.md` §3/§12 and `MECHANICS.md`): `Entrypoint.cs` `ActionKind` = {Wait, Add, Remove, ChooseUpgrade} only; `ApplyAction` dispatches to `TryAddRoad / TryClearTile / TrySelectUpgrade`; there is no motorway, bridge, roundabout or traffic-light placement action in `Discrete(3111)`; the raw IPC observation carries per-tile `motorwayDirections`, `hasTrafficLight`, `hasRoundabout`, colour `groupIndex` and destination demand, but the encoder (`mini_motorways_env.py:86-137`) drops colour and `motorwayDirections`; upgrade globals are ratios only and the two `ChooseUpgrade` slots are blind (the observation does not say which upgrade each slot offers); bridge consumption is only observable through the legality probe.
Required, in order:
1. **Action exposure (plugin)**: add `ActionKind.AddMotorway(from tile, to tile)`, `RemoveMotorway`, `PlaceRoundabout(tile)`, `PlaceTrafficLight(tile)`, `RemoveUpgrade(tile)` to `Entrypoint.cs`, wired to the native `AddMotorwayEdit`/`MotorwayModel` and the light/roundabout edit classes, each with a legality probe (inventory > 0, tile eligibility, `bridgesAvailable` for water ADDs) and a cost report; extend the legal-action list accordingly (the Python mask must stay authoritative).
2. **Observation exposure**: keep `motorwayDirections`, `groupIndex`, upgrade inventories as counts (not ratios), and the *contents* of the pending upgrade choice (so `ChooseUpgrade` becomes a typed strategic candidate "take motorway / take roundabout / ..." instead of a blind slot); keep per-tile vehicle count (channel 14) and add per-route trip-time or queue length if the game exposes it (needed for any congestion term).
3. **World model / candidates**: typed edges `motorway` (already declared in `world_model.EDGE_TYPES`, feasibility currently `unsupported`); candidate kinds `MOTORWAY(house cluster → destination, cost = 1 inventory, benefit = bypassing N road tiles)`, `BRIDGE` = water CONNECT with an explicit inventory prerequisite, `ROUNDABOUT/LIGHT(intersection tile)` with the intersection's observed vehicle count as the feature, `REROUTE(congested edge)` = CONNECT re-search with the congested edges banned plus a REMOVE of the old link (would-destroy veto must become "would-destroy-for-longer-than-k-decisions" since a reroute necessarily disconnects briefly, or be built as add-then-remove). Each needs a disruption flag and an executor plan type (multi-step: place, verify, then remove).
4. **Reward/eval**: natively the score is deliveries and congestion shows up as trip time and overcrowding; the synthetic env cannot provide this, so the native adapter (§9 of the S18-H report: `WorldModel` from the raw IPC JSON) comes first, then the S19 trainer can be pointed at the native env unchanged (it only needs the candidate features + durations).
5. **Minimum next step**: write the raw-observation adapter and run A/B natively on matched unseen seeds (S18-H §9 plan) to learn whether route choice has *any* traffic-mediated effect on native deliveries; only if it does is there a learning target for the selector.

## 10. Caveats
One training seed, one configuration, 20 minutes; synthetic connectivity-driven env with no traffic (route length is a proxy with no effect on deliveries); demand noise dominates survival so 36 test episodes resolve only large effects; the fixed 5-decision WAIT interval adds ~2.4 decisions of new-house latency relative to the per-decision planners (irrelevant to the 60-decision criterion but visible in the latency column); REMOVE candidates never arose, so the REMOVE path of the strategic env is exercised only by construction (mask/feature code), not by episodes; feature ablations and agreement rates are computed on the held-out test split after selection (no decision was made on them). Nothing native was run. STOPPED here.
