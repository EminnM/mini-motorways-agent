# S27 — Phase-1 native spawn-steering pilot (approved): results

Protocol as approved (S26 + corrections): 10 fresh seeds (600000 + 7919·i) × anchors 60/150, planner-B main line, one `DIAG_FORK` per anchor with `base`, `base2`, `block_far`, `block_near` (3-edge disconnected segments, farthest / nearest to a destination block, deterministic tie-breaks), 90-decision WAIT continuation, observations at checkpoints 10/30/60/90. Placements fixed before any outcome. Code `research/spawn_harness/code/{spawn_pilot,run_spawn_pilot,analyze_spawn_pilot}.py`; raw fork responses `pilot/raw/pilot_<seed>_<anchor>.json.gz`; `pilot/results.json`. Native wall-clock 230 s. No RL, no architecture, no follow-on native experiment.

## 1. Forks run and predefined handling
17 of 20 forks run. Skipped (reported, not replaced): 623757/150 and 655433/150 — episode ended before the anchor; 647514/150 — far and near segments not separated by ≥ 3 tiles. Far/near distances to the nearest destination: far 5–11 tiles, near 2–3.
**Checks (17/17):** base ≡ base2 (spawns, PRNG at every checkpoint, score, terminal); both interventions 3/3 edits accepted; concrete at checkpoint 10 = restore − 3 (base: −0). No stop condition triggered.
**Metric handling:** in 7/17 forks at least one new house's same-colour destination did not exist at the anchor; per the predefined rule the distance to the nearest destination of any colour was used and flagged. Every fork produced ≥ 1 house spawn in the horizon (1–9 spawns), so no fork lost its metric.

## 2. PRNG and candidate-set accounting (correction applied)
* `prngAfterEdits` equals base's in 17/17 forks for both blocks: placing the roads consumes no PRNG draws; any stream change happens inside later spawn events (one draw per feasible candidate).
* PRNG state differed from base *before* the first spawn difference (stream shift from an earlier event with an unchanged outcome) in 3/17 far and 4/17 near forks — these forks are flagged "confounded" and their spawn differences cannot be attributed to the blocked tiles.
* Empty buildable tiles occupied: 4 per intervention (3 edges), both blocks — equal counts, but as you noted this does not equalise the candidate-set change: whether a tile is in the placer's weighted candidate set is not observable, and the classification below shows near tiles were far more often involved.

## 3. Observed effects (index-matched spawns vs base; 54 base spawns across the 17 forks)
| branch | UNCHANGED | BLOCKED (base tile or 4-neighbour occupied) | SHIFTED (base tile free, different winner) | spawn-count change | type/timing changes |
|---|---|---|---|---|---|
| base2 | 55/55 | 0 | 0 | 0 | 0 |
| block_far | 38 | 3 | 13 | −1 (one fork) | } 5 total |
| block_near | 25 | 6 | 24 | +2 (one fork) | } |
Blocking near destinations perturbs spawning about twice as often (29/54 changed vs 16/54), and most changes are **non-local shifts** (SHIFTED ≫ BLOCKED): the base winner's tile was still free but a different tile won — consistent with the candidate-count re-alignment of the placer's per-candidate draws (S21 audit), not with the specific tile being taken.

## 4. Primary exploratory comparison (seed level; anchors averaged within seed; 10 seeds)
Metric = mean Manhattan distance of new houses to the nearest same-colour destination at the anchor (lower = better); Δ = branch − base.
* **block_far − block_near:** values [0.25, 0.25, −1.0, 0.0, −3.25, −0.58, 0.5, 0.5, 0.33, 0.67]; mean −0.23, median +0.25, 6 positive / 1 zero / 3 negative, bootstrap 95 % CI [−1.01, +0.33]. (Steering would predict a consistently negative value.)
* block_far − base: mean −0.08, 2 positive / 5 zero / 3 negative, CI [−0.30, +0.13]. block_near − base: mean +0.15, 3 positive / 3 zero / 4 negative.
* Anchor-specific: at anchor 60 (10 forks) the far block changed nothing on 8 forks; effects (both signs) arise mostly at anchor 150 where more spawns occur. One large case (631676/60, near block +5.5 tiles) dominates the near effect.
**Observation:** spawn locations change often, but the *direction* of the change is not controlled by where the block is placed; the paired far-vs-near difference is centred near zero with a CI spanning ±1 tile. This is an observed null at n = 10 seeds, not a demonstration that steering is impossible.

## 5. Opportunity cost
Planner-estimated connection cost (Dijkstra, S18-H rules) for all unconnected houses at checkpoint 90, branch − base, net of the 3 concrete spent: block_far mean −2.8 (CI [−4.5, −1.2]), negative in 9/10 seeds; block_near mean −2.5 (CI [−4.65, −0.10]), negative in 9/10. In words: the interventions cost 3 concrete and on average saved nothing in later connection cost (two forks showed a saving of +8 and +6 after cost; several showed increases because the added road or the shifted spawn lengthened routes).

## 6. Interpretation — observed vs causal
*Observed:* three plain road edges change where (and occasionally whether/what) the next buildings spawn in 30–55 % of spawn events, with exact reproducibility of the control; the change is mostly non-local; placing the block far from or near to destinations does not produce opposite, systematic shifts in the strategic metric; the estimated net value is negative.
*Causal claims warranted:* road occupancy perturbs the spawn process (confirmed again under exact controls). *Not warranted:* that blocking steers spawns in a chosen direction (no directional effect; PRNG re-alignment inside the spawn event is not controlled and dominates the changes), or that steering could pay for itself (net negative under the predefined accounting).
*Limitations:* 10 seeds (17 forks), two anchors per seed averaged; single 3-edge segments; the metric ignores future destinations (7/17 forks had houses with no same-colour destination yet); WAIT continuation; the placer's candidate set is unobservable, so blocked-vs-shifted is inferred from tile occupancy.

## 7. One recommended next decision
**Close spawn steering as a strategic lever in this regime, and close the native road-intervention line of work with it.** Together with S25 (routing static, density-blind) and S26 (deliveries bounded by per-destination demand/car supply), the pilot shows that cheap road placement changes the city's future only through an effectively random re-alignment of the spawn draw, with no controllable direction and a negative expected return at this scale. A stronger follow-up (more seeds, multi-segment blocks, planner continuation) is not justified by a null with a CI of ±1 tile and a negative cost balance. The validated fork harness, adapter and planner remain available; the residual, untested lever identified in S26 — trip length when houses per destination ≤ 1 — should be examined from existing logs before any new native run, and nothing is started without approval.
