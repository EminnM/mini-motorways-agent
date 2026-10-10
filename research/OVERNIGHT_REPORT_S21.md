# Overnight report — native evaluation, fork harness, S19/S20 results

## 1. What was done (in order)
1. **Adapter issues closed.** (a) ChooseUpgrade interrupts: when WAIT is masked the adapter takes the lowest legal action and keeps the plan (offline test T4; live: 1 interrupt per episode at decision 141, handled on all 24 planner episodes). (b) Mothballed handling: live probe on seed 507919 injected a driveway replacement — the adapter saw the old driveway go Mothballed one decision later, S11 model agreement 100 %, the planner re-activated it the next decision (`live_mothballed_probe.json`). (c) **A third issue surfaced and was fixed in the adapter layer**: the planner's geometric build-window rule (fitted in S13) admitted edges the native game refuses (e.g. a driveway onto a water tile), so on seed 507919 the executor abandoned 107 plans in 120 decisions ("mask_changed"). Fix: when the view is native, `structurally_buildable` defers to the native legal mask (house edges always; plain edges when concrete > 0). No planner heuristic was changed. Re-probe: 3 plans, 0 abandons. Adapter tests 6/6 (`adapter_tests.json`), live smoke 150 decisions 0 mismatches.
2. **Preregistered native evaluation executed** (`native_eval.json`, raw per-decision logs in `research/s21_native_check/raw/`): S8-C stochastic (fresh sampling seeds 7000+i), planner A, planner B × 12 designated seeds, 600-decision cap, three parallel game instances, 335–392 s each; **no policy was tuned**.
3. **Fork harness built and tested offline**, plugin op prepared but **not applied** (permission classifier blocked edits to the shared plugin source and DLL; nothing was changed — verified).
4. **Spawn pilot protocol written** (`research/spawn_harness/PILOT_PROTOCOL.md`); not executed.
5. S19 and S20 workers finished; results summarised below.

## 2. Native evaluation results (12 seeds, one episode per policy; S13a noise floor: per-seed SD ≈ 55 deliveries, 12-seed-sum SD ≈ 190)

| metric | S8-C | planner A (greedy) | planner B (resource-aware) |
|---|---|---|---|
| deliveries total / median | 533 / 45 | 383 / 36.5 | 615 / 39 |
| decisions median; survived 600 | 340; 0/12 | 298; 0/12 | 305.5; 1/12 |
| new houses connected ≤60 decisions | 39/99 (39 %) | 66/80 (83 %) | 84/98 (86 %) |
| connection latency (median of medians) | 10.5 | 3.75 | 3.0 |
| connected / houses at end (median) | 6 / 10 | 7.5 / 7.5 | 8.5 / 8.5 |
| destroyed functional connections | 676 | 0 | 0 |
| concrete spent (median); zero-concrete fraction | 81; 0.59 | 21; 0.00 | 22; 0.00 |
| illegal actions; abandoned plans | 0; – | 0; 0/72 | 0; 0/88 |
| **deliveries per connected-house-decision** | **0.0434** | **0.0292** | **0.0311** |
| functional edges per connected house (end) | 9–41 | 3.9–9 | 3.9–9 |

Paired per-seed deliveries: A − S8-C sum −150, median −6.5, 5 higher / 7 lower; B − S8-C +82, median +3, 7/5; A − B −232, median 0 (5 ties: identical action sequences on 5 seeds). Planner totals (383, 615) bracket the three earlier S8-C control totals (503–577); each policy exceeds all three S8-C control realisations on 3–4 of 12 seeds, which is what exchangeable noise predicts (~3). **No delivery or survival difference between the planners and S8-C is distinguishable from noise.** Per-seed table in `native_eval.json`.

**Interpretation (the actual reality check).** The planners do exactly what they were built to do — fast, legal, non-destructive, cheap connectivity — and it does not translate into more deliveries or longer survival natively. Two mechanisms are visible in the logs:
* *Throughput, not connectivity, limits deliveries.* On the planners' minimal trees (4–5 functional edges per connected house) each connected house delivers ≈30 % less per decision than on S8-C's dense, churned networks (0.029–0.031 vs 0.043). The S13 synthetic model (fixed 0.0436 per connected house, no traffic) is therefore wrong for sparse networks; S13's planner advantage (160 vs 8.5) was an artefact of that model. Native deliveries depend on trip time/congestion — the traffic dimension the synthetic environment does not have.
* *Colour timing.* On seeds 507919, 515838 and 547514 all three policies end at decisions 145–154 with 2 of 4 houses connected: the other two houses belong to a colour whose destination has not spawned yet (S14 audit: a group's destination can arrive 150+ decisions after its first house), and the single existing destination overcrowds. Nothing a connectivity planner can do; S8-C's longer episode on 515838 (397 decisions) is one stochastic realisation.
* Caveat: one episode per policy per seed; aggregate differences < 190 are not interpretable (S13a).

## 3. Infrastructure completed
* `research/s21_native_check/code/`: `native_adapter.py` (raw-observation → S18-H world model; native-legality patch; upgrade interrupts), `adapter_tests.py` (6/6), `live_smoke.py`, `live_mothballed_probe.py`, `native_eval_runner.py`, `native_eval_analyze.py`.
* `research/spawn_harness/code/harness.py`: branch specs (base, base2 duplicate-restore, block, sham), DIAG_FORK request builder, parser for DIAG_FORK and the existing DIAG_SPAWN records, `equivalent_initial_state`, `rng_confounded` (PRNG differs after the edits or at a checkpoint at/before the first spawn difference), spawn displacement / set metrics, intervention cost, score deltas, legal-segment planning from a raw observation. `tests.py` **4/4**: duplicate restores equivalent on all 19 recorded pilot forks; RNG confounding detected (12/19 C branches show PRNG differences, 3 of them before any spawn difference under the stricter rule; 4 B branches); synthetic fixtures (clean vs confounded vs non-equivalent); branch plans legal on 24 real native states.
* `research/spawn_harness/plugin_patch/DIAG_FORK.cs`: the generic fork op (snapshot current state, run arbitrary branch action lists with WAIT continuation, log spawns, PRNG after edits and at checkpoints, concrete, score, terminal step). **Not applied, not built, not deployed** — needs your permission to edit `Entrypoint.cs`, run `dotnet build` and copy the DLL (backup of the deployed DLL first). The plugin source and deployed DLL are unchanged (verified).

## 4. S19 — hybrid hierarchical RL (worker report, `HYBRID_HIERARCHICAL_RL_S19.md`)
Strategic SMDP env over S18-H candidates (stable geometric ordering, K-slot mask, fixed 5-decision WAIT), equivariant candidate-scoring policy, SMDP PPO with γ^duration (unit-tested), 20-min training (231 iterations, 421k strategic transitions). Held-out test (36 episodes): learned greedy 171.5 / stochastic 163 median deliveries vs A 163, B 168.5, **random strategic choice 176.5**; every paired CI straddles zero; 92 % of states offer only WAIT, REMOVE never arises, concrete never runs out. All checks pass (equivariance, discounting, no contamination, masks, no ordering leakage, no WAIT exploitation). Verdict: the candidate set + executor carries the intelligence; there is no reward-relevant strategic choice to learn in the synthetic environment. Consistent with §2: the synthetic task is solved by connectivity alone.

## 5. S20 — goal-conditioned construction distillation (worker report, `GOAL_CONDITIONED_DISTILLATION_S20.md`)
7,310 planner demonstrations on flat maps with pre-existing networks (split by map seed); 182k-param goal-conditioned conv policy, BC only, 15.7 min. Unseen seeds: 82 % action imitation / 88 % valid-set accuracy, 0 illegal actions. Autonomous completion: 68 % with its own DONE head (70/75 failures = premature DONE), **92 % with external termination** (planner 100 %, S8-C without a goal 9 %, random 0 %); 3–6 destroyed connections per 232 tasks vs 0 for the planner; degrades with crowding only through the DONE head (0.93 → 0.87 with external termination). DAgger not triggered. Usable as a wrapped construction primitive with an external termination rule and the S11 veto; not yet as a self-terminating macro-action.

## 6. Unresolved risks
1. **Synthetic-to-native validity**: the S13 delivery model is now shown to mis-rank policies natively (dense churned networks out-deliver minimal trees per house). Any further synthetic conclusion about deliveries needs a throughput term that is measured natively first.
2. **Throughput mechanism not identified**: per-house delivery rate differs, but the cause (trip length, shared-segment congestion, vehicle count, destination overcrowding timing) was not isolated; the native export gives only per-tile vehicle counts.
3. **Spawn harness untestable live** until the DIAG_FORK op is deployed (permission); the sham control's RNG-neutrality may be rare with far segments (only 1 of 24 real states had a "far" segment ≥ 8 tiles) — a candidate-aware sham (plugin-side exposure of the spawn-candidate set) may be required.
4. Single episode per seed; native non-determinism (3/12 reruns diverge).
5. The native-legality patch replaces the planner's geometric rule by the mask only when concrete > 0; routes planned at concrete 0 can still contain natively-illegal edges and will be abandoned at the first top-up (0 occurrences in 24 episodes, but possible).

## 7. Single highest-value next experiment
**Native throughput probe (before any spawn work):** on the 12 seeds, fork at the first all-connected state of planner B (needs DIAG_FORK) or, without the plugin change, run three short paired native episodes per seed with *scripted* network variants of the same connectivity — minimal tree (planner B), tree + one parallel segment per shared trunk, S8-C-like dense grid — and measure deliveries per connected-house-decision, destination overcrowding time and per-tile vehicle counts. This isolates whether road density/trip length drives the 30 % per-house delivery gap, which is (i) the reason the planners did not beat S8-C, (ii) a decision problem a connectivity planner cannot solve, and (iii) a prerequisite for making the spawn-steering question meaningful (spawn geometry matters only through trip length and congestion). Cost ≈ 15 min native. Spawn pilot second, after DIAG_FORK is approved and deployed.

**Approvals needed:** (a) apply/build/deploy `DIAG_FORK.cs` (plugin edit + DLL copy, with backup); (b) the throughput probe above; (c) the spawn pilot per `PILOT_PROTOCOL.md` once (a) is done. Nothing else is running.
