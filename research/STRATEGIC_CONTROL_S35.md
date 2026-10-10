# S35 — Strategic control on top of planner B (environment `postfix_v8`)

> **Addendum (after `PLANNER_GEOMETRY_S35.md`)**: this experiment ran on planner B *before* the unreached-destination fix. Its premise — that planner B's game-overs were demand-driven with nothing left to build — was wrong: the planner was ignoring newly spawned destinations that no house of their colour could reach (candidate list `[WAIT]` while 47 roads were available). With the fix, planner B goes from 485 to 2,084 deliveries on these same 10 seeds (2.1 → 5.7 weeks). The NO-GO below is therefore a result about the defective baseline; whether strategic choice points exist on the fixed planner has not been measured.

Question: are there strategic road decisions, above planner B's execution, that are worth learning? Code: `research/s35_strategic_control/code/{strategic,run,analyze}.py`; data: `research/s35_strategic_control/{eval,eval_v1_tiebreak_artefact,results.json}`. S31–S34 checkpoints, code and reports untouched. No training.

## 0. Decision: **NO-GO for strategic RL**
Both gate conditions fail on a preregistered 10-seed native experiment (60 episodes, horizon 1,000):
1. *Materially different outcomes* — the strategic choices change the executed road network in 8–9 of 10 episodes (edge-set Jaccard vs planner B 0.57–0.84 at decisions 100–400), but change the **outcome** in only 2 of 10 seeds; on the other 8 the episode ends at the same decision with the same deliveries, because the game-over is a demand-overflow event that road topology does not touch.
2. *Credible evidence some choices are better* — the two seeds that moved went −104 deliveries (dedicated route, seed 1007922: a 570-decision game cut to 310) and +10 (seed 1047517). Nothing beat planner B on aggregate under any rule (default −97, d2-only −87, random rollouts +12 / −94 / −98), and the single large effect is negative.
The decisions themselves are rarely live: house priority (D1) had a real choice at 3–5 of ~72 decision points per condition; route style (D2) at 13–18. The success criterion was to find strategic decisions worth learning; none were found in this regime.

## 1. Phase 1 — candidate decisions, inspected against planner B and the native state
| candidate | live in native play? | verdict |
|---|---|---|
| Which disconnected house to connect first (D1) | Almost never: planner B connects each house as it appears, so there is seldom more than one disconnected house (3–5 live points in 720 decisions per condition) | implemented; no headroom |
| Which colour / destination gets priority | Collapses into D1 (priority among disconnected houses); otherwise there is nothing to prioritise | folded into D1 |
| Defer construction to conserve resources / space | Rejected: episodes end with 35–38 concrete unspent on average (planner B 37.3) and 0 "waits for concrete"; resources are never binding | rejected |
| Repair / restructure an existing connection | Rejected: 0 abandoned plans, 0 destroy vetoes, every house connected at game-over in all 60 episodes; nothing to repair | rejected |
| Route style: reuse the network vs a dedicated road avoiding mixed-colour components (D2) | Live at 13–18 of ~72 points; the only decision with a real alternative | implemented |
Game-over cause (same in all 60 episodes, every condition): a destination of one colour overcrowds (demand counter 10, `isOvercrowding`) while all houses are connected and concrete is left over. Three of the ten seeds die at decisions 140–144 with 2–5 deliveries (the colour-timing failure seen on the 12 development seeds); the rest at 289–570.

## 2. Phase 2 — implementation (`strategic.py`)
* `StrategicExecutor(Executor)`: planner B's executor with exactly one step replaced — *which* candidate becomes the plan when the executor has none. Candidates, scoring, one-primitive-per-decision execution, invalidation/abandon logic, destroy vetoes, mask legality and upgrade handling are planner B's own code (the `act` body is copied verbatim apart from the selection call).
* Decision point = executor has no plan **and** at least one CONNECT candidate is feasible now (route found, affordable, destroys nothing). Otherwise planner B's own choice is used (safe fallback: WAIT / REMOVE / unaffordable CONNECT exactly as before).
* D1 options: `planner` (planner B's score and tie-break), `demand` (house whose colour has the most pins / is overcrowded), `cheapest`. D2 options: `reuse` (planner B's route) or `dedicated` (Dijkstra with every active edge of a mixed-colour component banned; offered only if it exists, is affordable, differs from the reuse route and destroys nothing).
* Rules: `PlannerRule` (transparency check), `FixedRule(priority, route)` — default = `demand + dedicated`, `d1only` = `demand + reuse`, `d2only` = `planner + dedicated` — and `RandomRule` (uniform over the live options, seeded per rollout).
* Action space, masks and the native stepping are untouched; the controller never emits a primitive edit of its own. Legality: 0 illegal actions in all 71 episodes; every ADD is inside the v8 playable rectangle by construction of the mask.
* Transparency verified: `wrapper` (PlannerRule) reproduced planner B on **10/10 seeds** — identical deliveries, lengths and edge sets at every snapshot.
* A first run (`eval_v1_tiebreak_artefact/`) used a different tie-break among equal-score candidates than planner B; that alone shifted outcomes by −88 on one seed, identically under every rule. Fixed (planner B's full tie-break key) before the reported run; kept as evidence that native outcomes are sensitive to tie-break-level route differences.

## 3. Phase 3 — experiment (preregistered before running)
Seeds 1000003 + 7919·i, i < 10 (unused anywhere before S35; the 12 development and 8 fresh seeds were not touched). Horizon 1,000 decisions. Conditions: A planner B; B `default` rule; C `random` rule × 3 rollouts; plus `d2only` and the transparency `wrapper`. 60 strategic episodes + 10 planner B ≈ 20k decisions, 12 min on 3 instances (as estimated).

### Results (10 seeds each)
| condition | deliveries Σ | median | length median | survived | concrete left | all houses connected at end | overcrowded-destination game-overs | decision points (D1 live / D2 live / chose dedicated) | episodes with a network differing from B |
|---|---|---|---|---|---|---|---|---|---|
| planner B | 485 | 36 | 298 | 0 | 37.3 | 6 (10 at ≤ 1 house short) | 10 | — | — |
| wrapper (planner rule) | 485 | 36 | 298 | 0 | 37.3 | 6 | 10 | 74 (3 / 15 / 0) | 0 |
| default (demand + dedicated) | 388 | 38 | 304 | 0 | 35.8 | 6 | 10 | 72 (4 / 14 / 14) | 9 |
| d2only (planner + dedicated) | 398 | 38 | 304 | 0 | 35.5 | 6 | 10 | 72 (5 / 14 / 14) | 9 |
| random r0 | 497 | 38 | 307 | 0 | 35.0 | 6 | 10 | 73 (3 / 18 / 12) | 8 |
| random r1 | 391 | 38 | 304 | 0 | 36.0 | 6 | 10 | 74 (4 / 14 / 8) | 6 |
| random r2 | 387 | 36 | 298 | 0 | 37.7 | 6 | 10 | 70 (5 / 13 / 10) | 8 |

### Paired per-seed differences vs planner B (deliveries; H/E/L; bootstrap CI on the mean)
| condition | Σ | H/E/L | CI95 | per-seed diffs | identical outcome seeds |
|---|---|---|---|---|---|
| default | −97 | 1/7/2 | [−32, +3] | 0, **−104**, 0, 0, 0, 0, +10, −3, 0, 0 | 7 |
| d2only | −87 | 2/7/1 | [−31, +4] | 0, −104, 0, 0, 0, 0, +10, +7, 0, 0 | 7 |
| random r0 | +12 | 2/8/0 | [0, +3] | 0, +2, 0, 0, 0, 0, +10, 0, 0, 0 | 8 |
| random r1 | −94 | 1/8/1 | [−31, +3] | 0, −104, 0, 0, 0, 0, +10, 0, 0, 0 | 8 |
| random r2 | −98 | 1/8/1 | [−31, +2] | 0, −104, 0, 0, 0, 0, 0, +6, 0, 0 | 8 |

### Do the strategic decisions change the executed network? Yes.
Active-edge Jaccard vs planner B at decision 100: 0.71–0.84 (identical on 4–7 of 10 seeds); at 200: 0.65–0.84; at 300: 0.57–0.74; at 400: 0.59–0.70 (never identical). Dedicated routes were chosen 8–14 times per condition at a mean extra cost of 1.5–2.6 concrete; two-thirds of them cost the same as the reuse route.

### Do the changed networks change gameplay? Almost never.
Outcomes are identical on 7–8 of 10 seeds despite different roads. The two seeds that respond:
* **1007922**: the dedicated route taken at decision 68 (cost 6, same as reuse) turns planner B's longest game (570 decisions, 144 deliveries) into a 310-decision, 40-delivery game — the same overcrowding death everyone else has, arriving 260 decisions earlier. Random r0 happened to pick reuse there and stayed on planner B's trajectory (+2).
* **1047517**: a dedicated route at decision 0 adds 59 decisions and 10 deliveries (289 → 348).
So the choices are causal on those seeds, but the effect that exists is the dedicated route making things worse on the one seed where planner B was doing well.

## 4. Failure cases and limitations
* The controller has very little to decide: D1 is live at ≈ 5 % of decision points and D2 at ≈ 20 %, and 70 % of all decisions are WAIT regardless of rule.
* Native transients and tie-breaks move outcomes as much as the strategic choices (the archived v1 run; the +10 seed appearing under rules that chose nothing different there). With 10 seeds and effects concentrated in 1–2 of them, no rule can be separated from noise except the one large negative case.
* "Dedicated route" is a component-level proxy for colour separation; it cannot keep traffic apart at the destination's own tiles, and the game's routing (static, density-blind, S25) gives no reward for separation anyway.
* Horizon 1,000 was never binding (longest episode 570). Deterministic conditions ran once per seed (planner play is reproducible apart from the rare transient); the random rule had 3 rollouts.
* Boundary/camera: every road is inside the native playable rectangle at its decision (mask + v8 checks); the camera's on-screen boundary is the same rectangle plus a 1-tile margin, driven by the interpolated expansion time (`CameraView.Tick`, `_PLAYABLE_AREA`), so a legal road can only appear outside the frame during the camera's own zoom/position lerp. Per-road boundary distances are now logged by the runner (`adds`, `adds_on_boundary`, `adds_outside_area` = 0 by construction).

## 5. Phase 4 — gate and what this means
NO-GO as stated in §0. The same picture as S21/S28/S29 and S34 from a third angle: in the regime the game reaches by week 2–3, road decisions — whether learned primitives (S31–S34) or strategic choices on top of a safe planner (S35) — do not move the delivery count, because the binding constraint is the number of cars of the overcrowding colour versus its pin rate, which no road action changes. No RL formulation is proposed; none would have a signal to learn from on these decision points.

If the programme continues toward higher native scores, the evidence points away from road construction altogether and toward the two levers the current interface does not exploit: the weekly upgrade choice (found inert for the *current* agents in M1, not shown inert in general — e.g. a roundabout or traffic light at the overcrowded destination's approach) and timing of when a colour's second destination is allowed to attach (not controllable). Both are outside this brief's constraints and are listed only so the next decision is made with the full picture. Not started.
