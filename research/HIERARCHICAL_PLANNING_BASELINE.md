# S18-H — Deterministic hierarchical planning agent for Mini Motorways (synthetic baseline)

Scope kept: no PPO, no reward change, no native game launched, no checkpoint touched, no existing file modified. All new files live under `research/hierarchical_planner/` (`MECHANICS.md`, `baselines.json`, `baselines_episodes.json`, `baselines_log.txt`, `tests.json`, `code/{world_model,candidates,planner,executor,tests,run_baselines}.py`). Development was time-boxed (~60 min of active work); anything not established in that time is marked unsupported.

## 1. Schema

```
WorldModel (visible state only; python: code/world_model.py)
  construction grid : per tile -> water, content (House/Destination/Carpark/None), colour group, in_window, degree, roads{N,E,S,W: A|M|None}
                      legality: legal_now(kind, edge) = the env mask (authoritative); structurally_buildable(edge) = placement rule without the concrete counter
  transportation    : nodes = tiles; typed edges 'road' (Active) / 'stub' / 'motorway' (typed, unsupported); Mothballed edges tracked separately and
                      excluded from connectivity; union-find components over ACTIVE edges + destination/carpark adjacency (S11 / s13_env rule);
                      houses {group, spawn_t}, destinations {tiles, group, carparks, demand}; connected house set; predicted_active_after(adds, removes)
                      applies the native house-edge replacement rule; route length = travel-cost PROXY (no traffic model exists)
  resources         : concrete, total, t, next_topup(t) -> (decisions until 141/281/421/561, amount)   [time-driven, known from the clock]
Candidate (code/candidates.py): kind CONNECT|REMOVE|WAIT|MOTORWAY, targets, adds (build order, house end first), removes, path, concrete_cost, route_len,
                      reused_edges, connectivity_delta, would_destroy_functional_connection, lost, prerequisites, feasibility, confidence, score, score_terms
Executor (code/executor.py): plan, done_adds, recent_added/removed (oscillation window 30), stats {plans, completed, abandoned, abandon_reasons,
                      waits_for_concrete, compute_s, illegal_suppressed, destroy_vetoes}, log
```
The planner reads `env.content / houses / dests / edges / free / concrete / total / t / water / action_masks()` — the same facts a human sees (colours included). It never reads `env.pending`, the scenario event list or the RNG (verified by inspection: `WorldModel.__init__` is the only env access point).

## 2. Verified mechanics
Full table with `file:line` evidence in `research/hierarchical_planner/MECHANICS.md`. Key verified facts:
* `Discrete(3111)` = WAIT + per-tile ADD/REMOVE for E/S edges + 2 `ChooseUpgrade` slots (`Entrypoint.cs:2469, 6256-6292`; `layout.py:50-64`). `ActionKind = {Wait, Add, Remove, ChooseUpgrade}`; `ApplyAction` (`Entrypoint.cs:2952-2968`) dispatches only to `TryAddRoad`, `TryClearTile`, `TrySelectUpgrade`. **There is no motorway, bridge, roundabout or traffic-light placement action.**
* REMOVE is legal only on a tile with exactly one road direction because `ClearTileExplicit` clears the whole tile (`Entrypoint.cs:2514-2522, 2598-2621`).
* Native raw observation carries per-tile `motorwayDirections`, `hasTrafficLight`, `hasRoundabout`, colour `groupIndex` and destination demand (`Entrypoint.cs:3190-3226`); the encoded 11,666-float vector keeps lights/roundabouts/vehicle count but drops colour and motorway directions (`mini_motorways_env.py:86-137`). Upgrade globals (`bridges/tunnels/motorways/lights/roundabouts available/total`, pending choices) are exposed as ratios only.
* Concrete, Active/Mothballed, build window, overcrowding and delivery rates: as measured in S11–S13 (`NATIVE_NOISE_AND_MULTI_HOUSE_S13.md` §2.2), used unchanged by the synthetic env.

## 3. Motorway / upgrade support matrix

| feature | native game | raw IPC obs | encoded obs | executable action | synthetic env | planner |
|---|---|---|---|---|---|---|
| plain road ADD / REMOVE | yes | yes | yes | yes | yes | yes |
| bridge (road over water) | yes, budgeted | globals only | global ratio | implicit in ADD (native probe decides) | water edge legal unless both ends water, no budget | treated as plain ADD; budget unknown |
| tunnel | yes | globals + `underMountain` | channel 2 + ratio | implicit | no mountains | n/a |
| motorway | yes (`AddMotorwayEdit`, `MotorwayModel`) | per-tile directions + globals | global ratio only | **none** | none | typed candidate, `feasibility='unsupported'` |
| traffic light | yes | `hasTrafficLight` + globals | channel 8 + ratio | **none** | none | none |
| roundabout | yes | `hasRoundabout` + globals | channel 9 + ratio | **none** | none | none |
| weekly upgrade choice | yes (forced 2-option choice) | pending count + totals | pending count | 2 blind slots | none (top-ups time-driven) | not exercised (synthetic has no choice) |

## 4. Candidate generation (`code/candidates.py`)
* **CONNECT**: for each disconnected house (ordered by spawn time), goals = every non-structure tile of an Active component that contains a destination of the house's colour; Dijkstra over tiles with lexicographic cost (concrete, length): Active edge 0, house edge 0 (free), new plain edge 1; never through other houses or structure tiles; water allowed unless both endpoints water. k = 2 alternatives by adding +3 length-penalty to the previous route's new edges and re-searching (distinct edge sets only). Each candidate gets `would_destroy_functional_connection` from the S11-style functional model (`WorldModel.lost_connections`), `feasibility='needs_concrete'` with a prerequisite when cost > concrete, and `confidence=0` if the predicted post-route connectivity does not include the house.
* **REMOVE**: every dead-end plain edge that the mask allows removing and whose removal (functional model) loses no connected house (refund 1).
* **WAIT** always.
* **MOTORWAY**: one typed object per disconnected house with a same-colour destination, `feasibility='unsupported'`, prerequisites listed; never scored above −∞.

## 5. Scoring function (`code/planner.py`, all terms logged per plan)
```
CONNECT: 10·Δconnected + 0.8·pins(target group) + 4·[group overcrowded] − 0.6·cost·scarcity + 1.5·reused/len − 0.15·len(proxy)
         + 1.0·min(60, waited)/60 − 0.8·max(0, cost − concrete);  −∞ if would_destroy or confidence 0
scarcity = 1 + shortfall/(next_topup_amount+1) · min(1, decisions_to_next_topup/140),  shortfall = other houses' cheapest routes − concrete left after this spend
REMOVE : +2.0 if some CONNECT is unaffordable by ≤ 3 concrete, −0.5 base      WAIT: 0      MOTORWAY: −∞
greedy (baseline A): 100 − cost − 0.01·len, −100 if unaffordable (nearest house first, no demand / scarcity / reuse reasoning)
```
Deterministic: fixed candidate order, max score, ties by kind order, lower cost, shorter route, edge tuple. The planner may spend everything (no reserve term); the scarcity multiplier only bites when other known needs exceed what is left before a distant top-up.

## 6. Executor (`code/executor.py`)
Persistent plan; one primitive action per decision; progress tracked by observed road state (`done_adds`); invalidation checks each decision — house gone / spawn on route / planned edge no longer buildable / already-built edge changed state / remaining ADDs would destroy a connection / REMOVE no longer legal — each abandons with a logged reason and replans; a plan whose next plain ADD is masked only because concrete is 0 is kept (`waits_for_concrete`, next top-up known). Oscillation guard: edges REMOVEd in the last 30 decisions are banned from ADD routes, edges ADDed in the last 30 are protected from REMOVE. Every emitted action is re-checked against the mask (else WAIT) and against the functional model (an ADD predicted to disconnect a connected house is vetoed and the plan abandoned with reason `would_destroy`).

## 7. Baseline comparison (synthetic MultiHouseEnv, 36 scenarios × 3 env seeds = 108 episodes per policy; seeds `7000+100·idx+s`, identical to S13; same `s13_baselines.episode()` metric code)
Reference rows are copied from `research/s13_multi_house/baselines.json` (same seeds/metrics). Test split = game seeds 515838/539595/563352/587109 (36 episodes); no weight was tuned on them (the only change during development was fixing the sign of the greedy score, found on two train/validation episodes).

| policy | episodes | deliveries median (all / test) | return mean (all / test) | survived 600 (all / test) | new houses connected ≤60 (all / test) | connections destroyed (total) | plain ADDs median | deliveries per plain ADD | zero-concrete frac | abandoned plans | compute ms/decision (median, max) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| WAIT | 108 | 0 / 0 | −10 / −10 | 0/108 / 0/36 | 0/277 / 0/88 | 0 | 0 | 0 | 0 | — | — |
| frozen S8-C stochastic (S13) | 108 | 17.5 / 8.5 | 27.2 / 16.6 | 15/108 / 4/36 | 221/587 (38 %) / 52/147 | 5,807 | 55.5 | 0.32 | 0.54 | — | — |
| scripted reference controller (S13, privileged) | 108 | 156.5 / 162.5 | 141.7 / 141.9 | 74/108 / 23/36 | 835/857 (97 %) / 288/294 | 0 | 25.5 | 5.29 | 0 | — | — |
| **A greedy nearest-house (this work)** | 108 | 158 / 163 | 151.5 / 144.7 | **84/108** / 24/36 | **852/852 (100 %)** / 280/280 | 0 | 25 | 5.80 | 0 | 0 of 914 | 0.67, 1.16 |
| **B resource-aware planner (this work)** | 108 | 152 / **168.5** | 149.2 / **152.3** | 82/108 / **27/36** | **855/855 (100 %)** / 295/295 | 0 | 24.5 | 5.85 | 0 | 0 of 911 | 0.66, 1.42 |
| C planner with motorway reasoning | — | UNTESTED | — | — | — | — | — | — | — | — | — |

Split detail (B): train 54 ep. deliveries median 143, 38/54 survive; validation 18 ep. 146, 17/18; test 36 ep. 168.5, 27/36. (A: train 170, 43/54; validation 142.5, 17/18; test 163, 24/36.)
Eligible-house denominators differ between policies because eligibility depends on episode length (an episode that survives longer sees more spawns).

Reading: both planners match or slightly exceed the privileged scripted controller on every metric (survival 82–84 vs 74; 100 % vs 97 % of new houses connected within 60 decisions; 5.8 vs 5.3 deliveries per plain ADD; 0 destroyed connections), with no privileged information. A vs B differ by < 1 survival per 36 and a few deliveries median — within the synthetic demand noise (survival is dominated by the demand random walk, S13 §2.5); B is better on the test split (27 vs 24 survive, 168.5 vs 163) and worse on train (38 vs 43), so **no ordering between A and B is claimed**. The remaining non-survival (~24 %) matches the controller's and is attributed to stochastic demand excursions with all houses connected (every B episode ends with every house connected; `connected_end_median` 8 = `houses_median`). C is untestable: no executable motorway action exists (§3).

## 8. Tests (`code/tests.py` → `tests.json`): 6 / 6 pass
| test | result |
|---|---|
| T1 action legality | 19,491 emitted actions over 36 episodes, 0 illegal under the env mask |
| T2 Active/Mothballed connectivity | `WorldModel.connected == env.connected_houses()` at every one of the 19,491 decisions |
| T3 resource accounting | 312 completed CONNECT plans: predicted concrete cost == observed spend (top-ups subtracted) in 312/312 |
| T4 candidate→primitive execution | 312 plans selected, 312 completed (house connected), 0 abandoned, 0 in progress at horizon |
| T5 connection preservation | 0 connections destroyed in 36 episodes; adversarial case: alternative house-edge ADD flagged by the model and confirmed to disconnect in the env (model prediction == env outcome) |
| T6 determinism | identical action sequence on two runs with the same env seed |
Test-writing note: T3 initially reported 10 paired "mismatches" (1↔2 at adjacent decisions); this was an attribution bug in the test (the executor starts the next plan and emits its first ADD in the same call in which it records completion), fixed by attributing each step's concrete delta to the plan active after `act()`. No planner rule was changed.

## 9. Native evaluation PLAN (not run; needs approval)
* Policy: `Executor(Weights())` on the raw native observation. Required adapter (not written): build `WorldModel` from the raw IPC JSON (`tiles[].contentType/groupIndex/road/overWater`, `destination.totalDemand`, `globals.concreteAvailable`) and from the native legal-action list — the `s11_guard.Ctx` reader already does the content/group/edge part; the clock-driven top-up schedule is the same. Handle the forced `ChooseUpgrade` state (take slot 0; the slots are blind).
* Design: frozen S8-C vs planner on **matched unseen game seeds** (not 500000+7919·i), `NORMAL_START`, 600 decisions, `decision_delta=1.0`, no guard, one run each per seed. Variance caveat (S13a): single-seed delivery SD ≈ 55, so an aggregate difference < ~190 deliveries over 12 seeds is not interpretable; a planner-vs-S8-C gap of the synthetic size (~150 vs ~20 per episode) would be ≈ 8 SD per seed and detectable with 12 seeds, but a planner-vs-controller-sized difference would need ≫ 50 seeds. Primary metrics: deliveries, survival, new-house-within-60, destroyed connections (expected 0 by construction), zero-concrete fraction. Budget: ~150 s native wall per 12 seeds per policy (S13a), plus adapter work. Mechanics the synthetic env lacks and which could bite natively: traffic/vehicle capacity (route-length proxy), bridge budget, the stricter native window (synthetic is 1.37× more permissive), the 18 % of REMOVEs that do not refund, and upgrade choices. **Request: approval to (1) write the raw-observation adapter, (2) run 12 matched unseen seeds for both policies.**

## 10. Future learned evaluator (proposal only)
Replace the hand-set weights with a learned value V(WorldModel features, candidate features) trained on logged (state, candidate, outcome) triples from the synthetic env: features = the logged `score_terms` plus demand/overcrowding state of all groups, decisions to next top-up, concrete, candidate route length/reuse/cost, number of pending houses; target = discounted deliveries over the next 140 decisions (one top-up period) or survival to the next top-up. The executor and the functional-connectivity veto stay rule-based (safety and legality are not learned); the evaluator only re-ranks candidates, so it can never emit an illegal or connection-destroying action. Validation: ranking agreement with the hand scorer, then synthetic held-out seeds, then the native plan of §9. A traffic-aware term would require a native vehicle-count signal (`visibleVehicleCount`, exposed in channel 14) that the synthetic env cannot provide.

## 11. Limitations
* Synthetic only; no traffic, no vehicles, no road capacity — route length is a proxy and the planner's intersections are untested against congestion. Every number above comes from a game model whose demand/delivery dynamics were fitted to one policy's native logs (S13 §4).
* The build window and the legality rule are the fitted synthetic approximations; natively the planner would rely on the legal-action list instead (the `structurally_buildable` fallback used while concrete is 0 would need the native list cached).
* Survival differences between A, B and the controller are within demand noise; only the gap to S8-C is large.
* k-alternative routes are generated but, because the scorer prefers lower cost and the alternatives are strictly longer, the alternative was selected only when it tied (e.g. `alt=1` at t=11 in the smoke log) — the alternative machinery is exercised but not shown to matter.
* REMOVE candidates were generated but never selected in 216 planner episodes (concrete never ran out: `waits_for_concrete_total = 0`); the refund logic is covered only by its unit-level checks in T5/T3, not by episodes.
* No oscillation occurred to exercise the ban/protect windows beyond their presence.
* Weights are hand-set; no sweep.

## 12. Integration gaps found
1. **No motorway placement action** in `Discrete(3111)` (`Entrypoint.cs:2469, 2952`); native has `AddMotorwayEdit` / `AddMotorwayAction` / `MotorwayModel`. Per-tile `motorwayDirections` is in the raw JSON but not in the encoded vector.
2. **No traffic-light / roundabout placement actions**, although both are observed (channels 8, 9) and have native edit classes.
3. **Upgrade choice is blind**: the two `ChooseUpgrade` slots are legal/forced but the observation carries only the pending count, not which upgrade each slot offers.
4. **Colour group is dropped by the encoder** (has-group flag only); the planner needs the raw `groupIndex`, which the raw IPC JSON provides.
5. Bridge budget (`bridgesAvailable`) is a global ratio only; whether a given water ADD will consume a bridge is not exposed except through the legality probe.

## 13. Recommended next step
Write the raw-observation adapter (≈ 100 lines, reuse `s11_guard.Ctx` + `edges_from_raw`) and run the §9 native comparison on 12 matched unseen seeds for planner vs frozen S8-C only (the synthetic gap is large enough to be visible at that sample size); defer motorway/roundabout/traffic-light work until the integration exposes placement actions and their option contents. STOPPED here; nothing further was run.
