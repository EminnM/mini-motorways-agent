# Mini Motorways research system — how everything works

A plain-language map of the whole stack: the game hook, the Python environment, what the agent sees and can do, the planner, the learning experiments, the evaluation protocol, and what the programme has established so far. Pointers go to the code and the phase reports; nothing here replaces those reports as the record of results.

## 1. The native game hook (`game_plugin/DoorstopBootstrap/Entrypoint.cs`)
* Mini Motorways (Unity/C#) is launched with a Doorstop bootstrap that loads our plugin DLL into the game process. Deployed copy: `…/Mini Motorways/Bootstrap/MiniMotorwaysBootstrap.dll`; builds are versioned by SHA-256 in `research/spawn_harness/backup/SHA256.txt` (current: **v8**, `2aa42367…`, = `postfix_v8`; v7 = `prefix_v7`).
* The plugin runs the game in **lock-step** with Python over a file-based IPC (`request.json` / `response.json` in a per-instance directory). Ops: `HELLO`, `RESET(seed, max_decisions, decision_delta, start_mode)`, `STEP(action, [realtime], [motorway])`, `MOTORWAY_CHECK`, and diagnostics (`DIAG_FORK` state snapshot/branching used by the fork harness, `DIAG_*`).
* One **decision** = one Python action followed by `decision_delta` = 1.0 s of simulated game time. A game week is 140 decisions (weekly upgrade top-ups land at decisions 141/281/421/561). The game can run headless (`-batchmode -nographics`, ~14 decisions/s per instance) or visible with real-time pacing for viewing.
* Each STEP returns the raw observation (tiles with content/colour group/road state per side, destination demand & overcrowding flags, `globals` with concrete counters, pending upgrades, motorways, `playableAreaAtTime`), the legal-action mask, reward components (`scoreDelta` = deliveries this step, `terminalPenalty` = −10 on game-over) and `terminated`.
* Legality is computed natively by probing the game's own `TileEditor` for every tile/direction, restricted (since v8) to road segments whose **both** endpoints lie inside the game's revealed playable area at the current expansion time (`City.GetSimulationPlayableAreaAtTime`). Applying an action re-validates it. Forced `ChooseUpgrade` interrupts (weekly) are the only time WAIT is illegal; the UI is suppressed and the agent must pick an upgrade slot.
* Verified mechanics that shape everything (S21–S29): routing is static shortest-path, density-blind, no splitting; cars = 2 per house; deliveries per decision are capped (~0.06) by pin generation and car supply; road occupancy perturbs future spawns via the game PRNG but not directionally; motorways are placeable through the interface but no eligible placement arises in the regime we reach; weekly upgrade choice is inert for the current agents.

## 2. The Python environment (`python/mini_motorways_env.py`, `experiments/gamectl.py`)
* `gamectl.GameInstance(name, headless)` launches one game process with its own IPC directory; `MiniMotorwaysEnv` is a Gym env over that IPC (`reset(seed)`, `step(action)`, `action_masks()`, `last_raw_observation`).
* **Action space `Discrete(3111)`**: index 0 = WAIT; 1…3108 = per tile (37×21 grid, x∈[−18,18], y∈[−10,10]) × {E, S} × {ADD, REMOVE} — i.e. add/remove the road segment from a tile to its east or south neighbour; 3109–3110 = ChooseUpgrade slot 0/1. Encoding in `python/synthetic_rl/layout.py` (`action_index`, `decode_action`); the native decode is identical.
* **Observation v1** (`encode_observation`, 11,666 floats): per-tile channels (content, colour group, road per side, water, demand, …) + globals. **v2** (`python/synthetic_rl/obs_v2.py`, 16,328 floats) adds six "group slot" one-hot channels so a colour is identified consistently across an episode (`GroupSlotTracker`).
* **Reward** = native score delta (deliveries) per decision, −10 on game-over; no shaping in anything from S31 on. Reward accounting is asserted per episode (Σ rewards = score + penalty).
* Episode ends on game-over (a destination's pin counter overflows) or at a decision cap (600 in training, 1,000–1,500 in later evaluations).

## 3. Planner B — the deterministic executor (`research/hierarchical_planner/code/`, adapter `research/s21_native_check/code/native_adapter.py`)
* `world_model.py`: a visible-information snapshot — tiles, houses with colour group and spawn time, destination blocks with demand, road edges (Active / Mothballed), concrete, clock, legal mask; functional connectivity = union-find over Active edges (a house is "connected" when its component contains a destination of its colour).
* `candidates.py`: typed candidates — CONNECT (Dijkstra from a disconnected house to any tile of a component holding its colour's destination; lexicographic cost (concrete, length); up to 2 alternative routes by penalised re-search), REMOVE (dead-end road whose removal breaks nothing), WAIT. Each carries its edge list, cost, predicted connectivity change and a "would destroy a functional connection" flag.
* `planner.py`: explicit weighted score (connectivity, demand/overcrowding of the target colour, concrete cost × scarcity vs the known weekly top-up schedule, reuse, path length, waiting time) → best candidate.
* `executor.py`: executes one primitive per decision, tracks progress, abandons and re-plans on invalidation (spawn on route, mask change, would-destroy), never emits an action the mask forbids, never destroys a functional connection, no add/remove oscillation.
* `native_adapter.py`: builds the world model from the raw native observation and makes the planner defer to the native mask for buildability; handles upgrade interrupts by taking the lowest legal slot.
* Behaviour: connects every house as it appears (there is almost never a backlog), spends ~30 % of decisions building and ~70 % waiting, dies at ~300 decisions (just over 2 weeks) with all houses connected and 35–50 concrete unspent, when the first destination of a colour that has gained a second destination block overcrowds.

## 4. Learning agents
* **S8-C / S14 PPO** (`python/synthetic_rl/models_v2.py` `SpatialExtractorV2`, MaskablePPO from sb3-contrib): a spatial CNN over the tile channels with per-tile ADD/REMOVE heads and a pooled head for WAIT/upgrades; trained on a synthetic multi-house environment (S13) with native-matched noise, migrated to obs v2 (`research/s14_group_observation/s8c_migrated_v2_step0.zip`).
* **Hybrid PLAN/EDIT/WAIT (S31)** (`research/s31_hybrid_native_rl/code/`): `HybridNativeEnv` adds action 3111 = PLAN ("let planner B choose this decision's native action"); EDIT = any native ADD/REMOVE the mask allows; WAIT. Policy = S14 extractor with the pooled head widened to (WAIT, upg0, upg1, PLAN), PLAN bias 6.0 at init. Trained 100,352 native decisions on 4 headless instances, training seeds 700000 + 7919·i (i < 40).
* **Replications** (`research/s34_replication/`): same recipe with PPO seeds 131 and 231. **Matched-rate random-EDIT ablation** (`research/s34_matched_edit_ablation/`): conditions A (hybrid), B (hybrid with its chosen edits replaced by random legal edits), C (planner B + random edits at the hybrid's rate).
* **Strategic controller (S35)** (`research/s35_strategic_control/code/strategic.py`): a wrapper that substitutes *which* planner candidate becomes the plan (house priority; reuse vs dedicated route) while planner B does all execution; deterministic rules and a random rule; retired-architecture infrastructure left intact.

## 5. Evaluation protocol and seed hygiene
* **Designated development seeds**: 500000 + 7919·i, i = 0..11 (12 seeds; used for every go/no-go up to S31 → now a development benchmark only). **Fresh benchmark seeds**: 900000 + 7919·i, i < 8 (replication/ablation; inspected since). **S35 seeds**: 1000003 + 7919·i, i < 10. Training seeds are disjoint from all of these; S13 synthetic test seeds 515838/539595/563352/587109 are never tuned on.
* Paired comparison per seed, bootstrap 95 % CI on the mean per-seed difference, H/E/L counts; stochastic policies get several rollouts with controlled torch seeds, and training-run variation is reported separately from rollout variation (never pooled). Noise floor: ≈ 55 deliveries SD per seed for stochastic native play (S13), ≈ 31–50 within-seed rollout SD on the fresh benchmark; deterministic play is reproducible decision-for-decision apart from a rare native state transient.
* Every evaluation asserts the deployed DLL hash (environment version), 0 illegal actions and exact reward accounting; episodes are written as `json.gz` with per-decision traces where instrumented.
* Operational rule learned the hard way: launch game instances one at a time / staggered and detached with file logging; 8+ concurrent instances plus trainers caused IPC timeouts and worker deaths.

## 6. What the programme has established (chronological, with reports in `research/`)
| phase | report | result |
|---|---|---|
| S13–S15 | synthetic env, obs v2, mixed PPO | native-matched synthetic training works; limited transfer |
| S16/S17/S17-G | CNN / attention / GNN colour-routing diagnostics | NO-GO: no architecture learned colour separation from the signal available |
| S18-H | deterministic planner | planner B; later the executor for everything |
| S21 / overnight | `NATIVE_REALITY_CHECK_S21.md`, `OVERNIGHT_REPORT_S21.md` | native adapter validated; planner B baseline on 12 seeds (615 deliveries) |
| S22–S25 | fork harness, topology probes, routing audit | routing static & density-blind; deliveries capped by car supply; road redundancy closed |
| S26–S29 | spawn steering, trip-length retrospective, capability gap memo | spawn steering closed; the binding constraint is houses-vs-destinations, not road design |
| M1 / M2 | upgrade-aware planner, motorways | upgrade choice inert; motorway interface works but no strategic value in regime |
| S31 | `HYBRID_NATIVE_RL_S31.md` | hybrid beats planner B on 12 seeds (best cell +654); conditional GO |
| S32 / S33 | `PLAYABLE_AREA_LEGALITY_AUDIT_S32.md`, `PLAYABLE_AREA_FIX_S33.md` | south-edge boundary crossings were possible; fixed in v8; no material effect on results |
| replication | `NATIVE_RL_REPLICATION.md` | NOT robust: 100k advantage over B = +318 / +114 / −20 across three PPO seeds |
| S34 | `MATCHED_EDIT_ABLATION_S34.md` | learned edits > random edits at the same rate, but not reliably > no edits; randomising edits triggers runaway editing; recommend stop |
| S35 | `STRATEGIC_CONTROL_S35.md`, `PLANNER_GEOMETRY_S35.md` | strategic choice points rare on the old planner; **planner B defect found (ignored unreached destinations) and fixed → B+: 485 → 2,084 deliveries on 10 seeds** |
| S36 | `CORRECTED_PLANNER_REBASELINE_S36.md` | B+ dominates every historical hybrid checkpoint (stopped at 80 %, partial) |
| S37 | `TRAFFIC_BYPASS_S37.md` | bypass mechanism works (navy-blue case); outcome value inconclusive (stopped) |

## 7. The picture in one paragraph
The agent acts once per simulated second by adding or removing single road segments (or waiting), sees the whole revealed map as tile channels, and is rewarded per delivery. Planner B connects houses safely and dies after about two game weeks when a colour's demand outruns its car supply — a limit set by the game's static routing and car count, not by the roads. Reinforcement learning on top of planner B can stretch survival by a week or so on some training seeds, but the effect is not reproducible across seeds and does not come from anything like a routing strategy. The remaining open question — whether any *strategic* road decision has headroom at all in this regime — is what S35 tests.

## 8. Where things live
`game_plugin/` (C# plugin, build to `bin/Release/net472/`), `python/` (env, layout, encoders, models, synthetic env), `experiments/` (instance control, legacy phase scripts), `research/<phase>/code` + `research/<phase>/eval*` (per-phase code and raw episode data), `research/*.md` (phase reports), `research/spawn_harness/backup/` (DLL backups and hashes), `logs/appsrc/` (decompiled game source used for read-only audits).
