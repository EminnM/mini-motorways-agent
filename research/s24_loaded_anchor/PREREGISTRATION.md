# S24 — Loaded-anchor fork topology experiment: preregistration (written before any fork or outcome is seen)

**Design.** 12 designated seeds; planner-B main line via normal STEP to decision **200**; one `DIAG_FORK` per eligible seed with branches `base`, `base2`, `parallel`, `dense`; **90-decision WAIT continuation**; observations exported at checkpoints 5, 10, 20, 30, 40, 50, 60, 70, 80, 90. Interventions are the unchanged S22 rules (`throughput_probe.edits_parallel/edits_dense`, ≤ 8 edges), computed from the anchor observation and the native legal mask. No tuning on outcomes.

**Anchor inspection (pass 1, before forking).** For every seed, at decision 200 record: the functional road graph, houses/destinations/connectivity, the candidate edit sets, and the per-tile vehicle counts over main-line decisions 170–200 (30 observations; "bottleneck tiles" = the 5 road tiles with the highest summed counts).

**Eligibility rules (per seed; all required, decided from pass 1 only).**
E1 ≥ 2 houses functionally connected at the anchor.
E2 Traffic present: summed visible vehicle count over decisions 170–200 ≥ 10 (i.e. ≥ 1 vehicle visible in ≥ a third of the observations).
E3 For the **parallel** branch: the edit set is non-empty, every edge legal, and after the edits the added edges form a **distinct route**: the subgraph of added edges connects two distinct tiles of the existing functional graph (a cycle through ≥ 3 added edges). For the **dense** branch: ≥ 2 added edges close cycles (both endpoints already road tiles, or chain to them), so the cyclomatic number of the functional graph rises by ≥ 2. A branch failing its rule is marked *structurally invalid*; a seed with both branches invalid is ineligible; a seed with one valid branch is run with the other branch reported but excluded from the primary analysis.
E4 Plausible usability: for at least one connected house, the shortest house→(own-colour destination stub) path after the edits is strictly shorter than before, or at least one added edge lies on a shortest path of equal length (so the game's shortest-path routing can select it) — "usable"; otherwise the branch is run but flagged *not plausibly usable* and excluded from the primary analysis (reported separately).
No substitute edits are generated for failing seeds.

**Primary outcome.** Paired cumulative deliveries over the 90-decision horizon (`score_90 − score_anchor`), parallel − base and dense − base over the eligible+valid+usable branches; `base2 − base` as the negative control.
**Manipulation checks** (from checkpoint observations): (M1) summed visible vehicle count on the added road tiles across the 10 checkpoints (must be > 0 for the intervention to count as "used"); (M2) occupancy on the bottleneck tiles, intervention vs base (redistribution = decrease); (M3) structural distinct-route check re-evaluated on the branch's own road graph at checkpoint 5; (M4) congestion: total vehicles and max per-tile count per checkpoint; overcrowding first checkpoint and fraction.
**Secondary.** First spawn divergence and PRNG classification (harness), connectivity set at every checkpoint (must not lose houses), concrete at checkpoint 5 = anchor − accepted edits, terminal step, vehicle totals.
**Analyses.** (A) pre-spawn-divergence window (last checkpoint ≤ first spawn divergence); (B) full 90 decisions. Full-horizon differences are not attributed to traffic alone when spawn histories differ.
**Stop rule.** Fewer than 6 eligible seeds, or M1 = 0 on more than half of the valid branches → report design failure; do not invent further variants.
**Reporting.** Paired per-seed tables, medians/means, sign counts, bootstrap 95 % CIs, manipulation-check tables, raw traces preserved (`raw/anchor_<seed>.json.gz`, `raw/fork_<seed>.json.gz`).
Budget ≈ 12 × 200-decision main line twice (inspection + fork pass) + eligible × 4 × 90 ≈ 15 min native.
