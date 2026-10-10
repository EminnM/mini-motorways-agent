# S23 — Fork-based topology intervention: preregistration (written before execution)

**Design.** 12 designated seeds; main line = planner B via normal STEP to decision 60 (`NORMAL_START`, `decision_delta` 1.0); one `DIAG_FORK` per seed with branches, all restored from the same snapshot:
`base` (no edits), `base2` (no edits; negative control), `parallel` (V1 rule of S22: offset copy of the longest trunk, connectors first, ≤ 8 edges), `dense` (V2 rule of S22: mesh around the trunk, loop-closing first, ≤ 8 edges). Edit sets are computed deterministically from the anchor observation and the native legal mask with the S22 code (`research/s22_throughput/code/throughput_probe.py`, unchanged). Edits are applied inside the branch (one per 0.3/n s), then **WAIT continuation for 60 decisions**; observations exported at checkpoints 1, 5, 10, 20, 30, 45, 60. Interventions are not tuned after seeing results.

**Primary outcome.** Paired change in cumulative deliveries over the 60-decision horizon: `Δ_full(branch) = score_60(branch) − score_60(base)`, per seed, for `parallel` and `dense`; reference `Δ_full(base2)`.

**Secondary outcomes.** Per-tile vehicle counts (total and per connected house) at checkpoints; first overcrowding decision and overcrowded-checkpoint fraction; first spawn divergence vs base (decision, type, tile) and the PRNG confound classification; concrete cost (concreteAtRestore − concreteAfterEdits, must equal accepted edits); connectivity (connected-house set at each checkpoint; must be unchanged by the edits at checkpoint 1); terminal step.

**Two analyses.** (A) *Pre-divergence*: deliveries, vehicles and overcrowding up to the last checkpoint ≤ the first spawn divergence (or 60 if none) — isolates immediate traffic effects of the topology; (B) *full 60-decision*: combined effect of topology + spawn perturbation + traffic. Full-horizon differences are **not** attributed to throughput where spawn histories differ.

**Checks per branch (all must hold or the branch is flagged).** `editsAccepted == len(actions)`; concrete delta == accepted edits (0 for base/base2); initial connectivity (connected houses at checkpoint 1) identical across the four branches; base vs base2: spawns, PRNG, score and vehicle totals compared and any difference reported (not assumed zero).

**Reporting.** Paired per-seed tables, medians/means, sign counts, bootstrap 95 % CIs of the mean paired difference (10k resamples), fraction of branches with spawn divergence, raw fork responses preserved (`raw/fork_<seed>.json.gz`).

**Limitation acknowledged in advance.** WAIT continuation over 60 decisions with 2–5 connected houses yields ~0.03–0.06 deliveries per house-decision, i.e. ≈ 5–15 deliveries per branch; if the resulting differences are within the base/base2 noise or too few to compare, that is reported as a limitation and the protocol is not extended.
Budget: 12 forks × 4 branches × 60 decisions + main lines ≈ 8–10 min native.
