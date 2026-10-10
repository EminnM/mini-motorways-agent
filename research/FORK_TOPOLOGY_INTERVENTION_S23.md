# S23 — Fork-based topology intervention (approved design): results

Preregistered in `research/s23_fork_topology/PREREGISTRATION.md` before execution. Code `code/run_fork_topology.py`, `code/analyze.py`; raw fork responses with observations at checkpoints preserved in `raw/fork_<seed>.json.gz` (12 files); `results.json`. Native wall-clock 122 s (one game instance, 12 forks × 4 branches × 60 decisions). Interventions = the unchanged S22 rules; nothing tuned.

## 1. Checks (requirement 5)
* **Legality:** every requested edit was accepted by the game: parallel 5/5, 0/0, 0/0, 8/8, 6/6, 8/8, 0/0, 8/8, 8/8, 8/8, 5/5, 4/4; dense 8, 3, 3, 8, 8, 8, 3, 8, 8, 8, 8, 8 (all accepted). On the three seeds where no house was connected at the anchor (507919, 515838, 547514) the parallel rule has no trunk and yields 0 edits; dense yields 3.
* **Resource accounting:** concrete at checkpoint 1 = concrete at restore − accepted edits on 24/24 intervention branches (e.g. 16→8, 14→6, 19→11). Note: the `concreteAfterEdits` field read immediately after the last edit still showed the pre-edit value whenever the 8 edits were applied within 0.3 s — the game's `EditTileCommand` is *scheduled* and debits concrete on a later frame (visible by checkpoint 1). The analyzer's "concrete_ok" flag used that early read and is therefore false on those branches; the checkpoint-1 value is the correct accounting.
* **Initial connectivity:** the connected-house set at checkpoint 1 is identical across the four branches on 12/12 seeds (edits never touch house edges).
* **Negative control (base vs base2):** identical on all 12 seeds — same spawns, same PRNG at every checkpoint, same score, same per-tile vehicle totals at every checkpoint, deliveries difference 0. Native reproducibility inside the fork is exact at this horizon; no noise to subtract.

## 2. Primary outcome — paired change in cumulative deliveries over 60 decisions (WAIT continuation)
| branch | mean Δ | median Δ | +/0/− (12 seeds) | bootstrap 95 % CI of mean |
|---|---|---|---|---|
| parallel − base | +0.08 | 0 | 2 / 8 / 2 | [−0.50, +0.67] |
| dense − base | −0.25 | 0 | 2 / 6 / 4 | [−0.92, +0.42] |
| base2 − base | 0 | 0 | 0 / 12 / 0 | [0, 0] |
Base deliveries over the horizon: mean 3.3, median 4 (0 on the three seeds with no connected house; 4–6 elsewhere). Per-seed Δ (parallel, dense): 500000 (0, 0), 507919 (0, 0), 515838 (0, 0), 523757 (0, −1), 531676 (0, 0), 539595 (−1, −1), 547514 (0, 0), 555433 (−2, −3), 563352 (+2, −1), 571271 (+2, +2), 579190 (0, +1), 587109 (0, 0).

## 3. Two analyses (requirement 3)
**(A) Pre-spawn-divergence.** First spawn after the edits differs from base in 9/12 parallel and 11/12 dense branches, at decision 5–24 (base2: 0/12). The pre-divergence windows (last checkpoint ≤ divergence) are 5–20 decisions on 9–10 seeds and 60 on the rest. Within them: deliveries difference 0 on 12/12 (parallel) and 10/12 (dense; −1 on 500000 and 579190 at the 10-decision checkpoint); vehicle totals differ by ≤ 2. **Immediate traffic effect: none detectable** — but with 2 connected houses and ≤ 20 decisions the window holds 0–2 deliveries, so only a gross effect could have shown.
**(B) Full 60 decisions** (topology + spawn perturbation + traffic combined): all |Δ| ≤ 3, both signs, CIs straddle zero. Per requirement 4 these are **not attributed to throughput**: 9–11 of 12 intervention branches have a different spawn history.

## 4. Secondary outcomes
* Spawn divergence fraction: parallel 0.75, dense 0.92, base2 0.00. RNG-confounded (PRNG differs before any spawn difference — stream shift from the edits): parallel 0.33, dense 0.58; the remaining divergences occur with equal PRNG up to the divergence (direct blocking or driveway feasibility).
* Vehicle counts at decision 60: 0–2 per branch (base 3 seeds non-zero, dense 6); with 2 houses the network carries ≤ 2 visible vehicles at a time — traffic is far from saturation in every branch.
* Overcrowding: first appears at checkpoint 45 on the three 0-connected seeds, identically in all four branches; never on the others. No branch terminated within 60 decisions.
* Concrete cost = accepted edits (above). Connectivity never changed by an edit.

## 5. Limitation (requirement 8) and verdict
The approved WAIT-continuation protocol yields 0–6 deliveries per branch over 60 decisions at anchor 60 (2 connected houses, ≈0.03–0.05 deliveries per house-decision), and the spawn divergence truncates the clean comparison window to 5–20 decisions on most seeds. A 30 % per-house throughput change — the size of the S21 cross-policy gap — would amount to ~1 delivery here, below what the design can resolve. **Result: no detectable effect of topology on deliveries in this regime, and the experiment was underpowered to detect the hypothesised effect size.** The protocol was not extended or changed. What the run did establish with exact controls: (i) the fork's negative control is noise-free at this horizon; (ii) eight plain road edges change the next spawn in 75–92 % of cases, about half of them through PRNG-stream shift alone; (iii) immediate traffic effects of these topologies within 5–20 decisions are nil at this network load.

## 6. Exactly one recommended next experiment
Repeat the same fork design — unchanged rules, branches base/base2/parallel/dense, WAIT continuation, spawn-divergence accounting — at a **later anchor where the network is loaded**: planner-B main line to decision 200 (6–9 connected houses on most seeds; the three seeds with no matching destination by then are excluded in advance), horizon 90. Expected base deliveries ≈ 15–25 per branch, so a 30 % throughput difference (~5–7 deliveries) becomes resolvable against a zero-noise control, and the pre-divergence window carries several deliveries even when spawns diverge after 5–20 decisions. Cost ≈ 12 × (200-decision main line + 4 × 90-decision branches) ≈ 6 min native. No RL, no spawn pilot, no architecture work.
