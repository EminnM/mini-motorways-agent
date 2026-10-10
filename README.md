# Mini Motorways agent research

> **Disclaimer.** Independent research and educational project. Not affiliated with, endorsed by, or supported by Dinosaur Polo Club. It requires your own legitimately purchased copy of *Mini Motorways*; no game files, assets, binaries or decompiled game source are included or distributed here — only our own code and our own experiment data. Modifying or instrumenting the game may be subject to the game's and Steam's terms; use at your own responsibility. Nothing here affects other players (single-player game, no online component).

Research programme for building an agent that plays the native **Mini Motorways** (Unity) game through an in-process plugin, with a deterministic planner, reinforcement-learning experiments, and a long series of audited native experiments. Everything is documented as phase reports under `research/`; start with **[`research/SYSTEM_OVERVIEW.md`](research/SYSTEM_OVERVIEW.md)** (how the stack works) and the phase table below.

## Status (October 2026)
* **Planner B+** (`research/hierarchical_planner/code/`, `Weights(serve_unreached_dests=True)`) is the strongest agent: it connects every house and every reachable destination, obeys the native legal mask and the revealed playable area, never destroys a working connection, and survives 5–7 game weeks (e.g. 1,769 deliveries over 12 fresh seeds vs 690 for the original planner B).
* The PLAN/EDIT/WAIT **hybrid PPO** line (S31–S34) is retired: its advantage over the original planner came almost entirely from a planner defect (ignored new destinations) and does not replicate across training seeds; against B+ it is behind on every complete benchmark.
* A **traffic-aware bypass** (opt-in `Weights(bypass=True)`) is implemented and demonstrated on one case; its value is unproven (S37, stopped).
* Work was halted at the user's request with S36 at 80 % and S37 in progress; both reports state exactly what is partial.

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
