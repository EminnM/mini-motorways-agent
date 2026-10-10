# S35 — Planner B: the decision-446 failure, road geometry, and the destination fix (environment `postfix_v8`)

Covers both briefs: the focused investigation of the WAIT at decision 446 (new destination with pins, no connection) and the road-geometry / resource-efficiency questions. Code: `research/hierarchical_planner/code/{world_model,candidates,planner,executor}.py` (opt-in additions, defaults unchanged), `research/s35_strategic_control/code/{diag_446,geom,run,analyze_fix}.py`; data: `research/s35_strategic_control/{diag_1007922_446.json,diag_1007922_566.json,geom_1007922_327.json,eval/plannerB*,}`.

## 0. Summary
* **The decision-446 WAIT is a real low-level executor defect in planner B, not a resource or river problem.** A newly spawned destination that no house of its colour can reach is never targeted: planner B's candidate generator is house-centric (it routes *disconnected houses*), and every house of that colour was already "connected" to the colour's older destination. The new destination's pins rise monotonically (1 → 10 over decisions 446–562) and end the game at 570 while the planner holds 47–59 road tiles, 1 bridge, and a 3–4-tile legal route exists from each house of that colour. This mechanism, not demand-vs-car-supply, is what has been killing planner B at "week 2–3" throughout S21–S35.
* **Smallest viable correction, implemented and tested** (opt-in `Weights(serve_unreached_dests=True)`; S18–S34 behaviour bit-identical with the flag off): destination-targeted CONNECT candidates for unreached destinations, completed when house and destination share a component. On the 10 S35 seeds: deliveries **485 → 2,084** (+1,599; 7 seeds higher, 3 identical early colour-timing deaths; CI on the per-seed mean [+84, +232]), median episode 298 → 798 decisions (**2.1 → 5.7 game weeks**), 0 illegal actions, concrete still 36 left on average.
* **Road geometry**: planner B's route search is lexicographic (new tiles first, then length), so it will take any free detour over a short route that costs tiles — intentional design, no turn or straightness term, no congestion term, never re-optimises a connected network. At decision 327 the "navy-blue" case is real (two houses drive 21–22 tiles where a 9–10-tile path costs 8 tiles, with 39 tiles unspent; detour ratio 2.2–2.3×). A scalar objective `length + λ·new_tiles` was implemented (`Weights(route_lambda=λ)`); it changes routes but **does not improve native outcomes** (fix+λ=1 −72 and fix+λ=4 −168 vs fix alone; λ=4 without the fix −103 vs planner B). **GO for the destination fix as the new planner B baseline; NO-GO for the geometry objective; no RL.**

## 1. Decision 446 — the investigation (seed 1007922, planner B, headless reproduction = the visible run's trajectory)
State at decision 446 (`diag_1007922_446.json`; day 22, week 3; the visible run's clock reads one day later because real-time pacing shifts the clock snap):
1. **Detection**: the planner's world model sees the destination — block at (−1,−2)…(0,−1), colour group 1, demand 1 at 446 — and its component contains **0 houses of group 1** (`houses_of_group_in_component: 0`). It is disconnected in the planner's own terms, but nothing consumes that fact: `connect_candidates` iterates houses and skips every house in `wm.connected`; all 12 houses are connected (the three group-1 houses to the older group-1 destination at (−12,3)). Candidate list at 446 = `[WAIT]`.
2. **Matching houses**: three group-1 houses at (−4,3), (−5,5), (−4,6), all connected elsewhere. None is considered: not because of reachability but because of the "already connected" filter.
3. **Paths**: the mask-authoritative Dijkstra from each of those houses to the new destination's own stub tiles finds routes of length 9–10 with **3–4 new tiles** (the rest reuses the existing network). No candidate was generated, so nothing was accepted or rejected.
4. **River / resources**: the found routes need no bridge (bridgesAvailable 1 anyway); 47 road tiles available (85 total); the executor had 0 `waits_for_concrete`. Legal ADDs at 446: 407, of which 12 touch water.
5. **Behaviour**: the planner returned WAIT at every decision from 434 to 570 (`plans 10 / completed 10 / abandoned 0`), i.e. "nothing to do" — not "waiting for a resource", not "thinks it is connected" (it never evaluates destinations at all).
6. **Pins = demand**: `destination.totalDemand` for that block rises 1, 2, 3, 4, 5, 6, 7 (overcrowding flag on at 522), 8, 9, 10 → game-over at 570, while the served destinations fluctuate 0–3. The adjacent block (1,−2)…(2,−1), exported with `groupIndex null` until it is coloured at ~538, becomes a second new destination (group 0) with the same fate starting.

**Road inventory**: the exported `concreteAvailable/concreteTotal` (47/85) is the simulation's `UpgradeDatabase._availableUpgrades[Concrete]` / `_totalUpgrades` (`Entrypoint.cs:3376`, `UpgradeDatabase.cs:62–70`), the quantity every road placement is charged against (`AddRoadEdit.cs:67–74`, 1 per plain segment, `GameBehaviourModel.GetConcreteCostForConnection`). The HUD stack displays `GetAvailableOrDraftUpgradeCount(Concrete)` from the client copy of the same model (`UpgradeBarClient.cs:138–147`), which should read the same 47; a "7" on screen would have to be a stale client copy under real-time pacing or a partly occluded "47". The runner now prints the exported inventory every 25 decisions in visible mode so the next viewing can be checked against the HUD directly.

## 2. Phase 1 — how planner B chooses routes (`candidates.py`, `planner.py`, `executor.py`)
| question | answer | design or bug |
|---|---|---|
| 1. New-road cost | 1 per new plain edge, 0 for a house driveway edge (`route_to_candidate`), matching the game's charge | design, correct |
| 2. Existing vs new | Dijkstra key is the tuple **(concrete, length)**: an existing Active edge costs (0, 1), a new edge (1, 1). Concrete is strictly primary, so any route with fewer new tiles beats any shorter route, however long | design — "roads are scarce" — with no bound on the detour |
| 3. Travel distance / time | Only as `route_len` with weight `W_LEN = 0.15` in the *scoring* of already-generated candidates; never in the search. No travel-time or speed model | design limitation |
| 4. Straight corridors / turns | No turn penalty or straightness preference; corridors come from Dijkstra's fixed neighbour order (N, E, S, W) breaking ties into straight runs | artefact |
| 5. Congestion / shared roads / demand | Demand (pins, overcrowding) enters only the *house-priority* score (`W_DEMAND`, `W_OVER`); no congestion, bottleneck or shared-road term anywhere | design limitation |
| 6. Optimising a connected network | Never: candidates exist only for disconnected houses (and, with the fix, unreached destinations); REMOVE only for dead-ends | design |
| 7. Disconnected destinations / crossings | Not handled at all before the fix (§1). Water: water-to-water edges unbuildable in the geometric rule; with concrete > 0 the native mask is authoritative, so a bridge is used when the game allows it; no explicit bridge accounting | **bug** (destinations); design (crossings) |
Alternatives: the k = 2 alternative routes penalise the previous route's *new* edges and re-search, so a route with 0 new edges has no alternative — the shortcut is never even generated as a candidate for the scorer to weigh.

## 3. Phase 2 — the two visible failures
**Navy-blue detour (decision 327)** (`geom_1007922_327.json`; the colour-group index is not the on-screen colour name, so the match is by position/behaviour):
| house | group | drive over existing roads | shortest legal path | new tiles for it | breaks connectivity | detour |
|---|---|---|---|---|---|---|
| (−5,−1) | 3 | 22 | 10 | 8 | no | 2.2× |
| (−5,0) | 3 | 21 | 9 | 8 | no | 2.3× |
| (−2,2) | 0 | 16 | 10 | 4 | no | 1.6× |
| (−14,2) | 0 | 8 | 6 | 2 | no | 1.3× |
| 6 other houses | — | = shortest | — | 0 | — | 1.0× |
Concrete available 39, so the 8-tile shortcut was affordable; planner B never considered it (0-new-tile route wins lexicographically; no alternative generated). The shortcut preserves all connectivity.

**South-east new destination (decision 446)**: detected but never targeted (§1); legal paths of 9–10 tiles with 3–4 new tiles from each matching house; 0 bridges needed; 47 tiles available (far more than needed, whether the HUD showed 7 or 47); WAIT because the candidate list was empty.

## 4. Phase 3 — alternative objective (isolated, opt-in)
`Weights(route_lambda=λ)` switches the Dijkstra key to **(length + λ·new_tiles, new_tiles)**. λ is in tiles of driving per road tile: λ = 4 is the exchange rate the scorer already uses (`W_COST/W_LEN = 0.6/0.15`); λ = 1 values a road tile at one tile of driving (aggressive shortcutting). Everything else — native mask, inventory (the executor still waits at concrete 0), bridges, playable area, destroy vetoes — is unchanged. With λ = 4 the decision-327 detour still wins (22 vs 10 + 4·8 = 42); with λ = 1 the shortcut wins (18 vs 22).

## 5. Phase 4 — paired native comparison (10 S35 seeds, horizon 1,000, deterministic; ≈ 40 episodes, 25 min, as estimated)
| condition | deliveries Σ | median length | weeks (median) | roads built Σ | concrete left (mean) | episodes ending by overcrowding | Δ deliveries vs planner B (H/E/L, CI) |
|---|---|---|---|---|---|---|---|
| planner B | 485 | 298 | 2.1 | 296 | 37.3 | 10/10 | — |
| planner B + λ=4 | 382 | 298 | 2.1 | 270 | 36.5 | 10/10 | −103 (0/9/1) |
| **planner B + destination fix** | **2,084** | **798** | **5.7** | 601 | 36.1 | 10/10 | **+1,599 (7/3/0, [+84, +232])** |
| fix + λ=1 | 2,012 | 790 | 5.6 | 644 | 29.6 | 10/10 | +1,527 (7/3/0, [+73, +230]) |
| fix + λ=4 | 1,916 | 746 | 5.3 | 589 | 33.9 | 10/10 | +1,431 (7/3/0, [+68, +219]) |

Per seed (deliveries / decisions / concrete left):
```
seed        planner B       fix          fix+λ4       fix+λ1       λ4 only
1000003     41/315/40   255/799/28   295/831/25   256/783/30    41/315/40
1007922    144/570/59   375/942/57   190/660/54   375/949/12    41/310/51
1015841      3/141/18     3/141/18     3/141/18     3/141/18     3/141/18
1023760     88/443/36   358/883/44   318/832/29   350/877/39    88/443/36
1031679      5/144/35     5/144/35     5/144/35     5/144/35     5/144/35
1039598      2/140/15     2/140/15     2/140/15     2/140/14     2/140/15
1047517     36/289/52   104/504/42   104/504/42    46/348/41    36/289/52
1055436     94/452/37   373/903/36   371/903/35   430/963/32    94/452/37
1063355     36/298/40   342/881/29   329/857/32   279/832/24    36/298/40
1071274     36/299/41   267/797/57   299/839/54   266/797/51    36/299/41
```
* **Fix vs planner B**: +1,599 deliveries; every seed that is not a decision-140 colour-timing death improves by +68 to +306 and survives 215–583 decisions longer. Roads built double (296 → 601) because the planner now has something to build; inventory stays slack (36 left).
* **Geometry objective**: λ changes routes (fix+λ1 builds 644 roads and leaves 29.6 concrete) but not outcomes in its favour: vs the fix alone, λ=1 −72 (2/4/4), λ=4 −168 (2/4/4); λ=4 without the fix −103 (one seed, the same early death the "dedicated route" produced in the strategic experiment). Shorter geometric trips did not translate into deliveries, as the brief warned they might not.
* **Game-over causes after the fix** (`dest_state_end`): still overcrowding in 10/10, but now at week 5–7 with 23–37 houses, typically several destinations of several colours at demand 7–10 simultaneously (e.g. seed 1000003: four blocks of colours 0 and 2 at 8–9) and 1–8 houses not yet connected at the end — late-game saturation rather than one ignored destination. The executor's `would_destroy` abandons appear in the fix runs (3–141 per episode on three seeds): with more roads, more new driveways would mothball a connecting driveway, and the executor correctly refuses — a secondary limitation worth knowing, not addressed here.
* **Colour-timing deaths** (3 seeds at 140–144, 2–5 deliveries): unchanged by anything; the first destination of a colour arrives before any house of it exists and overflows regardless of roads.

## 6. Decision gate
* **A. Is planner B making inefficient route choices?** Yes, by construction (concrete-first lexicographic search, no alternative when the free route exists): 2.2× detours occur. **B. Correctable cheaply?** Yes in roads (8 tiles with 39 spare). **C. Does correcting it improve native gameplay?** **No** — three λ arms, none better than the fix alone, one worse than planner B. Stop tuning λ. The low-level defect that *did* matter was the unreached-destination gap, and fixing it quadruples deliveries and nearly triples survival on the same seeds.
* **Recommendation**: adopt `serve_unreached_dests=True` as planner B's default for everything from here (keep `False` available for reproducing S18–S34; their reports stand as records of the pre-fix planner). Re-baseline before any further comparison: every planner-B number in S21–S35, and the hybrid's advantage in S31–S34, was measured against a planner that ignored new destinations. The strategic-controller NO-GO (`STRATEGIC_CONTROL_S35.md`) was reached on the same defective baseline; its "no live choice points" finding should be re-read after the fix (the fixed planner builds twice as many roads and survives to week 6, where congestion and late unconnected houses — not the ignored destination — end the game). No RL, no training started.
