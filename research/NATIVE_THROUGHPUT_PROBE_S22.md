# S22 — Native Throughput Probe: does road topology change deliveries when connectivity and demand are held comparable?

Preregistered in `research/s22_throughput/PREREGISTRATION.md` before execution; code `research/s22_throughput/code/{throughput_probe,analyze}.py`; raw per-decision logs `research/s22_throughput/raw/` (48 episodes + 2 smoke); results `results.json`. Native wall-clock ≈ 7 min (4 instances in parallel) + 1 min smoke. No RL, no architecture work, no tuning against outcomes (one execution-order fix to the parallel-route edit list after the smoke run, documented in the code).

## 1. Design (as preregistered)
12 designated seeds × 4 matched episodes from `RESET(seed)`: **V0** planner B; **V0dup** identical no-edit duplicate; **V1 parallel** = planner B to decision 60, then a deterministic offset copy of the longest trunk with end connectors (≤ 8 plain edges, connectors first), then planner B; **V2 dense** = same prefix, then ≤ 8 legal edges meshed around the trunk (loop-closing edges first), then planner B. Edits executed one per decision only if legal under the native mask. Primary outcome *R* = deliveries per connected-house-decision inside the **matched window** `[61, min(end, first spawn divergence vs V0)]`; primary comparison V2 − V0 paired per seed; noise reference |V0dup − V0|; decision rule: effect claimed only if ≥ 9/12 seeds agree in sign and the median |ΔR| exceeds 2× the duplicate-control median.

## 2. Equivalence checks (all passed)
* All four variants have byte-identical raw observations (tiles incl. vehicle counts + globals) at every decision ≤ 60 on **12/12 seeds**; V1/V2 first differ at decision 61 (the first edit), as intended. Connectivity and destination demand at T0 are therefore identical by construction.
* V0dup vs V0: identical at every decision for 11/12 seeds; on seed 500000 the hashes diverge at decision 447 with identical final deliveries (105). Whole-episode totals V0 = V0dup = 615. **Native variation under a deterministic policy is near zero** (the large S13a variability was sampling noise of the stochastic policy plus the rare game-side divergence).
* Edits: V1 60/60 legal, V2 81/81 legal; 0 illegal actions; **0 functional connections destroyed** in any episode; concrete spent after T0: V0 5, V1 9.5, V2 10 (medians).

## 3. Primary result: not measurable as designed — the edits move the next spawn almost immediately
On 10/12 seeds the first building after the edits spawns on a different tile (or is a different building) within 5–24 decisions of T0, so the matched windows are 5–24 decisions long (two seeds: 84–93) and contain 0–3 deliveries. ΔR(V2 − V0): 0 positive / 3 negative / 7 zero (2 undefined, no connected-house exposure in the window); ΔR(V1 − V0): 2 positive / 0 negative / 10 zero; duplicate control |ΔR| = 0 on all seeds. **Decision rule not met → no detectable topology effect on throughput at this scale, and more importantly no valid test of it**: the design cannot hold the subsequent city constant without snapshot forking.

| seed | first spawn after edits, V0 | V1 | V2 | V0's spawn tile on/adjacent to an edited road? (V1 / V2) |
|---|---|---|---|---|
| 500000 | House (−9,0) @73 | House (−8,0) | House (−8,0) | no / yes |
| 507919 | House (−4,−3) @78 | same | House (−7,−1) | – / no |
| 515838 | House (−6,−3) @65 | same | House (−8,−1) | – / no |
| 523757 | House (−8,5) @68 | (−8,4) | (−8,4) | yes / yes |
| 531676 | House (−8,−5) @78 | (−8,−6) | (−8,−6) | yes / yes |
| 539595 | Destination (−14,6) @79 | House (−4,5) | House (−5,4) | yes / no |
| 547514 | House @? | same | same | – |
| 555433 | Destination (−14,7) @69 | House (−6,5) | House (−4,5) | no / no |
| 563352 | House (−5,3) @84 | (−7,3) | (−5,4) | no / no |
| 571271 | Destination (−13,−5) @65 | (−12,−6) | (−12,−6) | yes / yes |
| 579190 | House (−4,5) @77 | Destination (−14,7) | House group 0 (−15,5) | no / no |
| 587109 | House (−6,4) @78 | (−7,5) | Destination (−5,2) @68 | no / yes |

21 of 24 intervention episodes show a displaced (or type-changed) first spawn; in 9 the original spawn tile was on or adjacent to an edited road (direct blocking of the candidate/driveway), in 12 it was not — consistent with the PRNG-stream shift the S21 audit predicted (one random draw per feasible candidate; removing candidates re-aligns the stream). Spawn **timing** is unchanged (same decision), only placement/type.

## 4. Secondary (whole-episode, confounded by spawn divergence)
Totals: V0 615, V0dup 615, V1 471, V2 487 deliveries; medians 39 / 39 / 39.5 / 38.5; survival 1/12 (V0, V0dup) vs 0/12. Paired per seed the outcomes swing both ways by amounts far larger than any plausible throughput effect: seed 563352 V0 172 → V1 39 / V2 36; 587109 V2 128 vs V0 38; 555433 V1 102 vs 40; 531676 V2 82 vs 38; 579190 88 → 34 / 28. These are consequences of different cities (where the next houses/destinations landed), not of trip time on the edited network. Matched-window R values where defined are 0.02–0.06 for all variants — the same range as before, with no systematic difference.

## 5. Evidence, uncertainty, limitations
* **Established:** deterministic-policy native play is reproducible decision-for-decision (11/12 full episodes identical); eight plain road edges placed next to a trunk change the next spawn placement in ~90 % of cases without changing its timing; the planners' delivery rate per connected house (0.03–0.06 in short windows) is unchanged by the edits in the few decisions that remain comparable.
* **Not established:** whether denser/parallel topology changes throughput. The S21 cross-policy observation (S8-C's dense churned networks 0.043 vs planner trees 0.029–0.031 per house-decision) remains a correlation across different cities and policies, not a causal topology effect.
* **Limitations:** one episode per cell; the matched windows are too short by design once spawns diverge; only per-tile vehicle counts observed; the edit rules are generic; K = 8 edges may be too few to change trip times materially even if the confound were removed.

## 6. One recommended next decision
The throughput question cannot be answered without holding the spawn process fixed, and the spawn process is exactly what road occupancy perturbs. Both questions therefore need the same instrument: **approve and deploy the DIAG_FORK snapshot/fork op** (patch, affected files, backup/rollback and the four equivalence-evidence steps are in `research/spawn_harness/plugin_patch/APPLY_AND_ROLLBACK.md`). With it, the throughput probe becomes: fork at T0, run V0/V1/V2 branches with the *same* restored state, and compare deliveries in the window before the first spawn — where the spawn is now also an outcome that can be attributed to blocking vs PRNG shift via the harness's confound test. Without DIAG_FORK, the only remaining lever is to make edits far from any spawn candidate, which cannot be verified from the observation. Spawn pilot stays deferred, as instructed; no RL or architecture work is proposed.
