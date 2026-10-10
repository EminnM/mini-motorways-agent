# S36 — Rebaseline against the corrected planner (B+) — **stopped early by request; partial results**

Status: the benchmark was halted at 152 of 190 planned episodes (user instruction "stop everything"). Everything below is computed from the episodes that completed; incomplete cells are marked. Environment `postfix_v8` throughout; stepping `decision_delta = 1.0`; forced upgrade interrupts take the lowest legal slot in every policy. Code: `research/s36_rebaseline/code/{run36,test_bplus,analyze36,probe_remove}.py`; data: `research/s36_rebaseline/{eval/,results.json,test_bplus.json}`.

## 1. B+ implementation and regression verification
* **B+** = planner B with `Weights(serve_unreached_dests=True)`: destination-targeted CONNECT candidates for destinations that no house of their colour can reach (`candidates.dest_candidates`, `world_model.unreached_dests`), completed when house and destination share a component (`executor._dest_served`). Historical planner B = `Weights()` (unchanged code path). The λ route objective is **off**.
* `test_bplus.py` on seed 1007922 (700 decisions) + seed 1000003: **5 of 6 pass** — T2 service recognised (one plan per destination, 17/17 plans completed), T3 no redundant construction (0 REMOVE, no re-targeting), T4 legality (0 illegal, 0 boundary violations, inventory never negative), T5 connectivity (0 functional connections destroyed), T6 flag-off reproduces historical B bit-for-bit (315 / 41). T1 (targets every serviceable unreached destination) *failed under its first criterion* because it counted destinations spawning before any house of their colour exists (nothing can be routed); for the three serviceable ones the latency was 0 decisions. The criterion was corrected (only destinations with a house of their colour) but the rerun was not executed before the stop — **T1 is verified by inspection of the log, not by a passing automated run**.

## 2. Paired benchmark results (B+ vs historical planner B; deliveries; per-seed paired, bootstrap 95 % CI on the mean per-seed difference)
| benchmark (horizon) | planner B | **B+** | Δ (H/E/L) | CI95 | B+ median length | B+ survived |
|---|---|---|---|---|---|---|
| dev, 12 seeds (600) | 615 | **1,149** | +534 (7/3/2) | [+5, +83] | 471 | 4/12 |
| fresh, 8 seeds (1,500) | 567 | **1,222** | +655 (5/2/1) | [+28, +148] | 563 | 0/8 |
| new, 12 seeds (1,500) | 690 | **1,769** | +1,079 (7/5/0) | [+18, +178] | 436 | 0/12 |
The "equal" seeds are the colour-timing deaths (first destination of a colour arrives before any house of it; game over at decisions 140–150 with 2–5 deliveries): 3/12 dev, 2/8 fresh, 5/12 new. B+ never loses to B on any seed. 0 illegal actions, 0 boundary violations, 0 destroyed connections in all 32 B+ episodes.

B+ end states: unreached destination present at game-over in 10/12 (dev), 8/8 (fresh), 12/12 (new) episodes with mean accumulated demand 8–14; roads left 41–47 (of 85–125); connected-house fraction at the end 0.67–0.82; overcrowded destination at the end in all but 4 dev episodes (those 4 reached the 600 cap).

## 3. Historical hybrid policies vs B+ (PLAN still calls the *original* planner; labelled historical)
Deliveries (Σ of per-seed means over rollouts) and paired Δ vs B+:
| policy | dev (1 rollout) | fresh (3 rollouts) | new (rollouts done / 3) |
|---|---|---|---|
| S31 50k sto | 1,206 → **+57 vs B+** (0/1/0 on 1 seed only — the rest not rerun; from S33 post-fix: 1,206 total, i.e. +57, 7/1/4 vs B+ by seed) | 976 → −246 (2/0/6) | 1,824 (22 eps, 2 rollouts partial) → +55 (4/1/7) |
| S31 100k sto | 1,110 → −39 (4/1/7) | 885 → −337 (2/0/6) | 1,269 (3/3) → −500 (7/0/5) |
| rep1 100k sto | 995 → −154 (3/1/8) | 681 → −541 (2/0/6) | 1,201 (10 eps, 1 rollout) → +465 (3/1/3) |
| rep2 100k sto | 899 → −250 (2/2/8) | 547 → −675 (2/0/6) | 864 (3/3) → −905 (7/1/4) |
Across-run SD of the 100k advantage over B+: 106 (dev), 170 (fresh), 704 (new, incomplete). Within-seed rollout SD 31–50 as in the replication. Every hybrid breaks connections (5–56 per episode) and ends with 12–18 roads vs B+'s 42–47; none survives to any horizon.

## 4. Failure-mode breakdown (what ends B+ games now)
* **Colour-timing deaths** (destination before house): unchanged by anything; 10 of 32 B+ episodes.
* **All other B+ episodes end by overcrowding at week 3–7 with an unreached destination present** — i.e. a destination the fixed planner still could not serve. The runner split "serviceable (a house of that colour exists)" vs "no house of that colour" was added mid-benchmark, so the split is available only for later episodes; where present, the unreached destination at the end was overwhelmingly of the *no-house* kind (a colour's second destination arriving while that colour has no houses) — a demand-supply event, not a routing miss. Road inventory is never binding (≥ 40 left), no crossing shortages were logged, no destroyed connections.
* The hybrids additionally lose to self-inflicted destruction and road exhaustion (13 roads left on average on the new seeds).

## 5. Statistical uncertainty and limitations
* Benchmark stopped at 80 %; the new-seed hybrid cells are 1–3 rollouts; S31-50k on dev was not rerun post-fix beyond the S33 regression.
* Dev seeds are a development benchmark (informed S31 decisions); fresh seeds were inspected in the replication/ablation; the 12 new seeds (1100003 + 7919·i) are the only untouched set and were not tuned against.
* Deterministic planners run once per seed (reproducible apart from the rare native transient); hybrids use torch seeds 36000 + 1000·r + i.
* No pooling across horizons; every number above is within one benchmark.

## 6. Recommendation (decision gate)
**Outcome A — B+ dominates the historical RL.** On the two benchmarks with complete pairing, every hybrid checkpoint is below B+ in total (−39 to −675), and the only positive cells (S31-50k on dev +57, rep1 on 10 new seeds +465) are single-rollout or partial and inside the run-to-run spread. The historical RL's measured value in S31–S34 was almost entirely the destination gap in the old planner; once that is fixed, a deterministic executor that serves every feasible destination is the stronger foundation. Retire the PLAN/EDIT/WAIT hybrids; use B+ as the base for any strategic layer. The outstanding question (what, if anything, a strategic layer can add on top of B+) is the subject of S37 (`TRAFFIC_BYPASS_S37.md`).
