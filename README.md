# Mini Motorways agent research

> **Disclaimer.** Independent research and educational project. Not affiliated with, endorsed by, or supported by Dinosaur Polo Club. It requires your own legitimately purchased copy of *Mini Motorways*; no game files, assets, binaries or decompiled game source are included or distributed here — only our own code and our own experiment data. Modifying or instrumenting the game may be subject to the game's and Steam's terms; use at your own responsibility. Nothing here affects other players (single-player game, no online component).

Research programme for building an agent that plays the native **Mini Motorways** (Unity) game through an in-process plugin, with a deterministic planner, reinforcement-learning experiments, and a long series of audited native experiments. Everything is documented as phase reports under `research/`; start with **[`research/SYSTEM_OVERVIEW.md`](research/SYSTEM_OVERVIEW.md)** (how the stack works) and the phase table below.

## Status (October 2026)
* **Planner B+** (`research/hierarchical_planner/code/`, `Weights(serve_unreached_dests=True)`) is the strongest agent: it connects every house and every reachable destination, obeys the native legal mask and the revealed playable area, never destroys a working connection, and survives 5–7 game weeks (e.g. 1,769 deliveries over 12 fresh seeds vs 690 for the original planner B).
* The PLAN/EDIT/WAIT **hybrid PPO** line (S31–S34) is retired: its advantage over the original planner came almost entirely from a planner defect (ignored new destinations) and does not replicate across training seeds; against B+ it is behind on every complete benchmark.
* A **traffic-aware bypass** (opt-in `Weights(bypass=True)`) is implemented and demonstrated on one case; its value is unproven (S37, stopped).
* Work was halted at the user's request with S36 at 80 % and S37 in progress; both reports state exactly what is partial.

## Project status: discontinued
The author stopped the project here (October 2026). It is published as-is for anyone who wants to pick it up. Nothing is being maintained.

### What works
* Lock-step control of the real game (headless or visible) with a native legal-action mask, verified reward accounting and reproducible deterministic play.
* Planner B+ plays legally and safely for 5–7 game weeks; its regression tests (`research/s36_rebaseline/code/test_bplus.py`) pass apart from one test whose criterion was corrected but not rerun.
* A complete evaluation pipeline (paired seeds, bootstrap CIs, separate reporting of training-run and rollout variation) and ~1,000 logged native episodes.

### Open problems (unsolved, in rough order of importance)
1. **The game is lost to colour timing, not to roads.** Every agent dies when a colour's destination demand outruns its car supply — either immediately (a colour's first destination spawns before any house of it; ~25 % of seeds end at decision ~140 with 2–5 deliveries) or at week 5–7 (several colours overcrowd at once). No road action changes car supply; routing is static and density-blind (S25). The only untested levers are the weekly upgrade choice (found inert for the agents tried, M1) and whatever a human does differently in weeks 5+.
2. **No agent reaches week 8.** Horizon 1,500 (≈ 10.7 weeks) was never reached by anything; the best episodes end around decision 1,070.
3. **RL did not add value over a correct planner.** The hybrid PPO's gains over the original planner were the planner's own bug (ignored new destinations); against the fixed planner every checkpoint is behind (S36, partial). Three training seeds gave three different policies (replication). Colour separation was never learned by any architecture (S16/S17).
4. **The bypass idea is unresolved.** The mechanism works (S37: detects a 2× detour, builds it, cars use it, the targeted colour stops piling up pins), but its effect on outcomes was measured on one confounded seed and fired on only 1 of 13 further seeds under the rules tried. The proposed next step (continuous expected-benefit score instead of thresholds; then possibly a small learned build/don't-build decision) was not implemented.
5. **S36 rebaseline is 80 % complete**; the new-seed hybrid cells have 1–3 rollouts.
6. **Known smaller issues:** the planner's route search is concrete-first and never re-optimises a connected house (deliberate, but produces 2× detours); the executor sometimes refuses routes because a new driveway would mothball a connecting one (`would_destroy` abandons, up to 141 in one episode); a rare native state transient makes stochastic rollouts non-reproducible even with fixed torch seeds; launching more than ~4–5 game instances at once causes IPC start-up timeouts; the HUD road-count badge appears to show only the last digit of the inventory (the exported number is the simulation's).

### If you continue
Start from `research/CORRECTED_PLANNER_REBASELINE_S36.md` and `research/TRAFFIC_BYPASS_S37.md`; the environment version to use is plugin build `postfix_v8` (hash in `research/spawn_harness/backup/SHA256.txt`); do not tune on the 12 development seeds (500000 + 7919·i), the 8 fresh seeds (900000 + 7919·i) or the 12 new seeds (1100003 + 7919·i) — pick a new family.

## Layout
| path | what |
|---|---|
| `game_plugin/DoorstopBootstrap/Entrypoint.cs` | the in-game plugin: lock-step IPC (RESET/STEP/…), native legal-action mask, state export, diagnostics (`DIAG_FORK`). Deployed build hashes in `research/spawn_harness/backup/SHA256.txt` (current `postfix_v8`). |
| `python/mini_motorways_env.py`, `python/synthetic_rl/` | Gym env over the IPC (`Discrete(3111)`, observation encoders v1/v2), models, synthetic twin environment |
| `experiments/gamectl.py` | launching/stopping headless or visible game instances |
| `research/hierarchical_planner/code/` | planner B / B+: world model, candidates (connect, unreached-destination, bypass, remove), scorer, executor |
| `research/s21_native_check/code/native_adapter.py` | native observation → planner world model; native-legality patch; upgrade-interrupt handling |
| `research/s31_hybrid_native_rl/` | hybrid env/policy/trainer/evaluator (retired line; checkpoints preserved) |
| `research/s34_replication/`, `research/s34_matched_edit_ablation/` | replication of S31 and the random-EDIT ablation |
| `research/s35_strategic_control/`, `research/s36_rebaseline/`, `research/s37_traffic_bypass/` | strategic wrapper, planner geometry/fix, corrected-planner rebaseline, bypass |
| `research/*.md` | phase reports (below) |

Not committed (see `.gitignore`): game process logs and IPC directories, decompiled game source used for read-only audits, legacy multi-hundred-MB training runs and checkpoints, third-party tool downloads, plugin build outputs.

## Phase reports (chronological)
| phase | report | one line |
|---|---|---|
| S21 | `NATIVE_REALITY_CHECK_S21.md`, `OVERNIGHT_REPORT_S21.md` | native adapter validated; planner B baseline 615 deliveries / 12 seeds |
| S22–S25 | `FORK_INSTRUMENTATION_VALIDATION_S22B.md`, `NATIVE_THROUGHPUT_PROBE_S22.md`, `FORK_TOPOLOGY_INTERVENTION_S23.md`, `LOADED_ANCHOR_FORK_S24.md`, `ROUTING_MECHANICS_AUDIT_S25.md` | fork harness; routing is static and density-blind; deliveries capped by car supply |
| S26–S29 | `RETROSPECTIVE_AND_SPAWN_PROTOCOL_S26.md`, `SPAWN_STEERING_PILOT_S27.md`, `TRIP_LENGTH_RETROSPECTIVE_S28.md`, `AGENT_CAPABILITY_GAP_MEMO_S29.md` | spawn steering closed; capability gap memo |
| M1 / M2 | `UPGRADE_AWARE_PLANNER_M1.md`, `MOTORWAY_CAPABILITY_M2.md` | upgrade choice inert; motorway interface validated, no use in regime |
| S30 | `NATIVE_MIDGAME_BENCHMARK_FEASIBILITY_S30.md` | midgame benchmark feasibility |
| S31 | `HYBRID_NATIVE_RL_S31.md` | hybrid PLAN/EDIT/WAIT PPO: +654 over planner B on dev seeds, conditional GO |
| S32 / S33 | `PLAYABLE_AREA_LEGALITY_AUDIT_S32.md`, `PLAYABLE_AREA_FIX_S33.md` | south-edge boundary gap closed (plugin v8); no material effect |
| replication | `NATIVE_RL_REPLICATION.md` | S31 does not replicate across PPO seeds (+318 / +114 / −20) |
| S34 | `MATCHED_EDIT_ABLATION_S34.md` | learned edits > random edits, not reliably > no edits; stop hybrid |
| S35 | `STRATEGIC_CONTROL_S35.md`, `PLANNER_GEOMETRY_S35.md` | strategic wrapper NO-GO on the old planner; **planner B defect found and fixed (B+: 485 → 2,084 on 10 seeds)** |
| S36 | `CORRECTED_PLANNER_REBASELINE_S36.md` | B+ dominates historical RL (partial, stopped at 80 %) |
| S37 | `TRAFFIC_BYPASS_S37.md` | bypass mechanism works on the navy-blue case; outcome value inconclusive (stopped) |

## Running things
Requires the game installed with the Doorstop bootstrap and the plugin DLL deployed (`game_plugin/DoorstopBootstrap`, build to `bin/Release/net472/`, copy to `…/Mini Motorways/Bootstrap/`). Python 3.14, `stable-baselines3`, `sb3-contrib`, `numpy`, `torch`.
```bash
# planner B+ on one seed, visible, real-time (viewing only)
python research/s35_strategic_control/code/run.py --cond plannerBfix --idx 1 --visible
```
```bash
# planner B / B+ / historical hybrid checkpoints on a benchmark (headless)
python research/s36_rebaseline/code/run36.py --bench new --policy plannerBplus
```
```bash
# B+ regression tests
python research/s36_rebaseline/code/test_bplus.py
```
Operational note: launch game instances one at a time (staggered, detached, logging to files); more than ~5 concurrent instances or simultaneous launches cause IPC start-up timeouts.
