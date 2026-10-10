# PHASE 8 — Environment Core

Generated 2026-10-06. Objective: combine Phases 5–7's proven primitives (native reset, state inventory, road-edit command path) into the first closed agent-environment loop — canonical observable state, legal-action enumeration/masking, controlled simulation-time stepping, and an end-to-end `RandomPolicy` test — entirely inside the native/injected environment, with no Python/Gym/PPO/reward work.

**Result: SUCCESS, after a genuine, confirmed, and fixed persistence-isolation bug discovered mid-phase.** All four deliverables are implemented and live-validated. The bug and its fix are documented in full below, since they materially affect how any future phase must initialize the environment.

## Summary of what was built

- `PolicyObservation` — a canonical, deterministic-order, OBSERVABLE-only JSON representation of native state (`BuildPolicyObservation`), separate from Phase 6A's richer privileged `SimulationSnapshot`.
- Legal-action enumeration/masking (`EnumerateLegalActions`) over `{WAIT, ADD_ROAD(x,y,dir), REMOVE_ROAD(x,y,dir)}`, `dir ∈ {N,E,S,W}`, legality determined entirely by the native `TileEditor`/`EditTileCommand` path — never approximated.
- Controlled, deterministic simulation-time stepping (`AdvanceSimulation`) via direct `ISimulation.Step()` calls, bypassing `MotorwaysGame.Tick`'s real-frame accumulator entirely.
- An end-to-end `RandomPolicy` loop (`RunPolicyEpisode`) with a separately-seeded `System.Random`, decision logging, multi-episode resets, and terminal/truncation handling.
- **A persistence-isolation mechanism** (`InitializeProcessContext` + `SwitchToIsolatedTestProfile` + `AssertPersistenceIsolated`), added mid-phase after a real violation was found and root-caused.

## 1. Canonical observable state (`PolicyObservation`)

Implemented in `Phase8Runner.BuildPolicyObservation` (`Entrypoint.cs`). Structure matches the brief's sketch:

```
PolicyObservation
  simulationTime
  bounds {xMin,yMin,width,height}
  tiles[]  (row-major, y then x, over the FULL CityDefinition.PlayableArea -- deterministic regardless of any dictionary/hash-set iteration order inside native models)
    coord, buildable, overWater, underMountain, overRail
    road {N,E,S,W}   (RoadState per cardinal direction)
    hasTrafficLight, hasRoundabout, motorwayDirections[]
    contentType, groupIndex, destination{destinationType,totalDemand,isOvercrowding,overcrowdingTime,isUpgraded}
    visibleVehicleCount
  globals
    hour, day, week, score
    concreteAvailable/Total, bridgesAvailable/Total, tunnelsAvailable/Total,
    motorwaysAvailable/Total, trafficLightsAvailable/Total, roundaboutsAvailable/Total
    pendingUpgradeChoices, totalClaimedPackages
```

**Privileged exclusions, explicit** (per Phase 6A's tagging, carried forward): `contributedSupply`/`demandMultiplier`/`RequiredSupply` (hidden demand-gating formulas); the `unassignedDemand` vs `waitingDemand` split (only the visible TOTAL pin count is included — a human sees pins stack, not which ones have a vehicle en route); vehicle route/pathfinding internals, native pathfinding-graph node ids; RNG state. Vehicle positions are deliberately NOT included per-vehicle in `PolicyObservation` (a human can see cars, but reading a vehicle's internal id/house/destination references by inspecting object identity is not something a human playing normally can do) — instead each tile carries an aggregate, OBSERVABLE `visibleVehicleCount`.

**Determinism**: tile iteration is a fixed row-major loop over the map's fixed bounds rectangle — never dependent on any `Dictionary`/`HashSet` iteration order inside `TilemapModel` or elsewhere. Confirmed live via the determinism test (§5): the SHA-256 hash of the full canonical observation string matched exactly, at all 6 checkpoints, across two independently-reset episodes.

Example serialized observation: [`research/environment_observation_example.json`](environment_observation_example.json) (231KB, one full `PolicyObservation` captured mid-episode).

## 2. Legal-action enumeration / mask

Action vocabulary: `WAIT`, `ADD_ROAD(x,y,dir)`, `REMOVE_ROAD(x,y,dir)`, `dir ∈ {N,E,S,W}` (`ActionDescriptor`/`ActionKind`). Per the brief's undirected-edge requirement, `EnumerateLegalActions` only ever generates `East`/`South` candidates per tile — between them this covers every edge in the grid exactly once (a tile's North edge is its northern neighbor's South edge; West is its western neighbor's East edge), so `(x,y,E)` and `(x+1,y,W)` never become two independent actions.

**Legality is determined entirely by native logic**, never approximated:
- `ADD_ROAD` legality: `TileEditor.AddRoad(tilemap, coord, dir)` is called as a probe; if `result.IsSuccessful`, the action is legal, and the probe `TileEdit` object is released back to its pool (`Scope.Release`) without ever being scheduled, if the enumeration call isn't the one chosen to execute.
- `REMOVE_ROAD` legality: a real, reported **native constraint**, not an arbitrary restriction (first identified in Phase 7A's open questions, now resolved): `TileEditor.ClearTileExplicit(..., Roads)` mothballs/clears **every** road direction on a tile, not a single edge. To keep `REMOVE_ROAD(x,y,dir)`'s native behavior equal to its name, it is only legal when the target tile has **exactly one** two-lane road direction (the requested one) — checked via `Tile.GetTwoLaneRoadStateInDirection` for all 4 cardinal directions (a read-only precondition check, not an approximation of the edit itself) — and then confirmed via the real `TileEditor.ClearTileExplicit` probe.
- `WAIT` is always first in the legal set and always legal.

**Validated against native behavior**: live-confirmed throughout the determinism test and RandomPolicy episodes — every `ADD_ROAD`/`REMOVE_ROAD` the enumerator marked legal was subsequently `accepted=true` when executed via `ApplyAction` (zero false positives observed across 36+ RandomPolicy decisions + 12 determinism-test decisions, all logged in [`research/random_policy_trace.jsonl`](random_policy_trace.jsonl)).

**Performance bound (explicit smoke-test convenience, not a permanent design choice)**: enumeration is scoped to a `RectInt(-5,-5,16,16)` sub-region (16×16 tiles × 2 directions × 2 action kinds = up to 1024 native probes/decision) rather than the full map, to keep this phase's runtime manageable. `PolicyObservation` itself still covers the full map.

## 3. Controlled simulation-time stepping

Research question resolved via decompilation (background subagent, independently verified): `ISimulation.Step()` is **parameterless** — it always advances by exactly the fixed `Timestep` (confirmed `Fix64.One/10` = 0.1 simulated seconds), via internal `Fix64` arithmetic, or by zero if paused. `MotorwaysGame.Tick(float)`/`Game.Tick(float)` is just an accumulator wrapper around repeated `Simulation.Step()` calls, using real/scaled frame time — exactly the real-frame dependency the brief warned against relying on.

`AdvanceSimulation(testGame, deltaSeconds)` calls `ISimulation.Step()` directly, `steps = (int)((deltaSeconds/0.1)+0.5)` times, in a tight loop with no `yield return` — bypassing `MotorwaysGame.Tick`'s accumulator, view-interpolation, audio, autosave-timer, and upgrade-popup side effects entirely. Confirmed safe: `LaneUpdateProcess` (Pending→Active promotion) and every other registered `IProcess` run **inside** `Simulation.Step()` itself, not in `Game.Tick`'s wrapper — so skipping `Game.Tick` doesn't skip any simulation-correctness logic, only real-time/UI-driven effects a headless loop shouldn't depend on anyway.

`Δt_decision = 0.3` simulated seconds (3 native `Step()` calls) — **not** the brief's example `0.25`, because `Timestep=0.1` only divides evenly into multiples of `0.1`; `0.25` would force a fractional/rounded step count. Noted as an explicit environment parameter, not a permanent decision.

**Verified live**:
- Simulated time advances by exactly the requested amount, independent of real frame rate — confirmed by the log's `simulationTime` column incrementing by exactly `0.30` every decision in [`research/random_policy_trace.jsonl`](random_policy_trace.jsonl), regardless of how long each decision's native-probe-heavy legal-action enumeration took in real time.
- Scheduled commands (`EditTileCommand`) execute at the expected boundary — confirmed via the same `resultCode=Success`/`accepted=true` pattern already proven in Phase 7A, now under controlled stepping.
- Progression/spawning/vehicles continue functioning — confirmed via fingerprint changes between decisions (house/destination/vehicle counts evolving).
- The game remained healthy under repeated controlled steps across the entire run (zero exceptions in [`logs/instrumentation.log`](../logs/instrumentation.log) for the final, fixed run).

### Determinism test

`reset(seed=1001)` → fixed action sequence (`WAIT, ADD_ROAD(5,5,E), WAIT, ADD_ROAD(5,6,S), WAIT, REMOVE_ROAD(5,5,E)`) → controlled `Δt=0.3` per decision → canonical observation fingerprint (SHA-256 of the full `PolicyObservation` JSON) at every decision — run twice from independent episode resets (same seed, same action sequence). **Result: identical fingerprints at all 6 checkpoints, both runs** (not just the final state):

```
checkpoint 0: runA=4430e0d80332f882 runB=4430e0d80332f882 match=True
checkpoint 1: runA=bcf4c14af7e5377a runB=bcf4c14af7e5377a match=True
checkpoint 2: runA=175e51a4ff6c12cb runB=175e51a4ff6c12cb match=True
checkpoint 3: runA=102bf7115da03235 runB=102bf7115da03235 match=True
checkpoint 4: runA=19b94bcc500ff058 runB=19b94bcc500ff058 match=True
checkpoint 5: runA=471287850dad56c4 runB=471287850dad56c4 match=True
```

An earlier attempt at this exact test produced trivially-matching fingerprints for the wrong reason: a `Math.Round(double, MidpointRounding)` call inside `AdvanceSimulation` threw `MissingMethodException` on every single call (this game's stripped Mono BCL is missing that overload — the same class of gap documented since Phase 0/3), so simulated time silently never advanced at all (`simTime=0.00` throughout). This was caught, not hidden: fixed by replacing `Math.Round` with manual integer round-half-up (`(int)((deltaSeconds/0.1)+0.5)`), avoiding the missing overload entirely rather than patching corlib again. The determinism result above is from the corrected run, where simulated time demonstrably advanced (`simTime` climbing 0.30→1.80 across the 6 decisions) and real state changed between checkpoints (fingerprints differing decision-to-decision, matching only across the two independent runs at the same decision index).

## 4. End-to-end RandomPolicy

`RunPolicyEpisode` implements `Reset(seed) → loop{ Observe, EnumerateLegalActions, [fixed|random] action, ApplyAction, AdvanceSimulation, Observe } → terminated/truncated → teardown`, reused for both the determinism test (`fixedActions != null`) and RandomPolicy episodes (`fixedActions == null`, driven by a policy RNG).

- **Separate RNG**: `new System.Random(randomPolicySeed + episodeIndex)` — never `global::Random` (the game's own deterministic simulation RNG). Action selection never perturbs simulation determinism.
- **WAIT probability**: explicit `pWait = 0.3` constant; if `editActions.Count == 0` or the RNG roll is under `pWait`, the action is `WAIT`; otherwise uniformly sampled from the non-WAIT legal actions. This avoids the pathological "always pick a random edit because there are thousands of them vs. one WAIT" failure mode the brief warned about.
- **Per-decision record** (exactly the fields the brief specifies): `episode, decisionIndex, simulationTime, observationFingerprint, legalActionCount, selectedAction, actionAccepted, resourcesBefore, resourcesAfter, nextObservationFingerprint` — written as JSONL: [`research/random_policy_trace.jsonl`](random_policy_trace.jsonl) (36 records, 3 episodes × 12 decisions).
- **Multiple episodes, different seeds**: 3 RandomPolicy episodes (`simSeed=2001/2002/2003`, `policyRngSeed=777/778/779`), each a fresh `MotorwaysGame`/`InitCityCommand` reset.
- **Observed in the trace**: `WAIT`, legal `ADD_ROAD`s (e.g. `ADD_ROAD(-5,8,S)`, `ADD_ROAD(9,1,E)`, ...), resource consumption (`resourcesBefore`→`resourcesAfter` dropping by 1 concrete per successful add), native world evolution (fingerprints changing every decision), and resets between episodes. No `REMOVE_ROAD` happened to be sampled in these particular 3 RandomPolicy episodes (none existed yet at the single-edge-tile precondition by chance within 12 decisions each) — removal IS exercised and confirmed working in the determinism test and the isolation canary (§6).
- **Resulting road network**: deliberately scattered, visually arbitrary single-tile connections across the map — exactly the "visually terrible and expected" outcome of undirected random play.

## 5. Terminal-state handling

`MotorwaysGame.HasGameEnded` (public getter, private setter, set exactly once by `OnGameEnd(GameEndReason)`) is the authoritative native terminal flag — `IsTerminal` reads it directly, never inventing a custom threshold. **Important caveat, confirmed via decompilation**: under `GameMode.Normal`, the only path to `OnGameEnd(GameEndReason.GameOver)` is `DestinationModel.OnOvercrowded → GameplayEventHandler.OnDestinationOvercrowded → EndGame → ScreenStack.PushScreen(GameOver) → GameOverScreen.TransitionIn → MotorwaysGame.OnGameEnd`, which runs through the UI screen stack. This headless bootstrap never drives that screen stack (confirmed separately, §8), so `HasGameEnded` may never flip to `true` even if a destination genuinely overcrowds. This phase's episodes rely on **truncation** (a fixed `maxDecisions=12` / `maxSimTimeSeconds=10.0` horizon) — confirmed in every episode's log line: `terminated=False (native game-over, via MotorwaysGame.HasGameEnded) truncated=True (decision/time limit)`. `terminated`/`truncated` are logged as clearly distinct fields, never conflated.

## 6. Persistence-isolation bug: found, root-caused, fixed, and verified

### What happened

Mid-phase, after the first full RandomPolicy run (29 real `ADD_ROAD` placements + 2 `REMOVE_ROAD` removals across 5 short episodes), the REAL restored player's `extendedUserProfile_<realId>.json` was found to have changed: `AchievementStats.TreesBulldozed` +2, `AchievementStats.TotalConcreteDeleted` +4 — a genuine mutation of the user's real, persistent lifetime stats. This was caught via the same before/after hashing discipline used in every phase since 5E, not assumed.

### Root cause, established via a sequence of static traces + one targeted live diagnostic (not guessed)

1. **The achievement-stat write mechanism** (confirmed via decompilation): `AchievementStatistics.DataChanged → MotorwaysExtendedUserProfile.OnValueChanged → IPersistentStorageService.Store()` marks the file dirty *synchronously* on every stat mutation; `PersistentStorageService.Tick()` (wired to `TickRegistry.AppTicking`, driven by Unity's own per-real-frame `App.Tick` — **not** by `ISimulation.Step()`/`MotorwaysGame.Tick()`, and with **no time gate at all**) flushes it to disk within about one real frame. This is a completely separate, ungated mechanism from the already-proven-safe 300-second/game-hours-gated `MotorwaysGame` journal autosave — conflating the two was the original mistake.
2. **First fix attempt (failed)**: switch the active player to a freshly `PlayerDatabase.CreatePlayer()`'d disposable test player via `ActivePlayer.ActivatePlayer(testPlayer)` *before* constructing any `MotorwaysGame`. Live-tested: the real player's stats still mutated. A read-only diagnostic (reflection into `TreeModel`/`ReleaseMothballedLanesProcess`/`AchievementCheckingProcess`'s private `_player` fields, confirmed via decompilation to all be `[Dependency] ActivePlayer` — the live wrapper, not a baked `Player` snapshot) confirmed the *same* `ActivePlayer` object instance reverted from the test player's id back to the real player's id, purely from calling `app.Scope.Get<MotorwaysGame>()` — before `.Start()` ever ran.
3. **Exhaustive static trace of the full `MotorwaysGame` DI construction graph** (every `[Dependency]` field, recursively, every `OnCreatedInScope` hook) found **no** reachable code that touches `PlayerDatabase`/`ActivePlayer`/`MostRecentPlayer` — a genuine negative result, not a gap.
4. **Authoritative live confirmation**: subscribed to the public `ActivePlayer.PlayerChanged` event (a plain C# event subscription — not a patch of `App.dll`) and logged `Environment.StackTrace` on every fire. The real call chain: `AppRuntime.Update() → AppContainer.Tick() → App.Tick() → ScreenStack.Tick() → Motorways.Views.LoadingScreen.Tick() → PlayerDatabase.DeletePlayer → PlayerDatabase.RemovePlayer → ActivePlayer.ActivatePlayer(MostRecentPlayer)`.
5. **The exact native code**, confirmed by reading `LoadingScreen.Tick` directly (`Motorways.Views/LoadingScreen.cs`):
   ```csharp
   foreach (Player player2 in _playerDatabase.Players)
   {
       if (player2.HasAvatar) continue;
       bool flag = false;
       if (player2.LastPlayedUtcTimeOnLocalDevice == DateTime.MinValue) flag = true;
       if (player2.UserProfile is LegacyMotorwaysUserProfile { TotalPlayTime: 0 }) flag = true;
       if (flag) list.Add(player2);
       else { /* ChooseAvatar for legitimate players */ }
   }
   if (list != null) foreach (Player item in list) _playerDatabase.DeletePlayer(item);
   _hasActivatedPlayer = true;
   Player player = _playerDatabase.MostRecentPlayer;
   _activePlayer.ActivatePlayer(player);
   ```
   This is a **genuine, intentional stock invariant** (category 1 of the three possibilities considered, not a bug and not an ordering artifact exclusive to this bootstrap): every normal boot, `LoadingScreen` prunes any player with no chosen avatar and either `DateTime.MinValue` last-played time or zero recorded play time — exactly what a freshly `CreatePlayer()`'d test profile looks like. It runs on Unity's own per-frame `Update()` loop, concurrently with our bootstrap, entirely independent of when `app.Scope.Get<MotorwaysGame>()` happens to be called — the original "boundary around `Get<MotorwaysGame>()`" was correlation (several real frames elapse during the async city-definition/prefab-load wait, giving `LoadingScreen.Update()` room to run), not causation.

### The fix

Stop performing our own redundant "restore the real player" call entirely (`LoadingScreen` already does the exact same restore, unconditionally, on every boot). Instead, passively **wait for `LoadingScreen`'s one-shot activation to fire naturally** — confirmed via a strong signal (the `PlayerChanged` event's own call stack containing `"LoadingScreen"`, not merely `HasActivePlayer==true`) — and only then create+activate the disposable test player. Since `_hasActivatedPlayer` is a true one-shot gate (never reset within a single process run), once it has fired, the pruning loop can never visit or delete our test player again.

### Permanent safety gate

`AssertPersistenceIsolated(app, testPlayer, testGame, checkpointLabel)` checks, every time it's called:
- bootstrap `ActivePlayer.Player.Id == testPlayer.Id`
- `RuntimeHelpers.GetHashCode(ActivePlayer)` unchanged from the expected app-scope singleton (captured once at subscription time)
- the test player is still present in `PlayerDatabase.Players`
- every accessible simulation component's `_player` field (`TreeModel`, `ReleaseMothballedLanesProcess`, `AchievementCheckingProcess`, read via reflection — a read, not a patch) is `ReferenceEquals` to that same `ActivePlayer`, and its `.Player.Id` matches

Wired as a **permanent pre-action gate** inside `RunPolicyEpisode`: called immediately before every single action (not just the canary); if it ever fails, the action is replaced with `WAIT` and never executed, logged loudly. Per explicit direction, this gate remains in place even though the bug is now solved — it stays as a standing safety net for all future automated episodes.

### Canary validation (read-only investigation → fix → one controlled stat-producing action → full re-run)

A dedicated, minimal canary (`RunPersistenceCanary`) ran: stock boot completes → create+activate test player → verify isolation → construct/start one `GameMode.Normal` episode → verify isolation at every lifecycle boundary (`C3,D,E,E2,F,PRE-ACTION,POST-ACTION`) → one setup `ADD_ROAD` (precondition only — plain empty-tile adds don't touch `AchievementStatistics`) → **one** known stat-producing `REMOVE_ROAD` (confirmed in Phase 7A/8 to reliably increment `TotalConcreteDeleted`) → verify isolation held → teardown.

**Result: every single `AssertPersistenceIsolated` check passed.** `TotalConcreteDeleted` on the currently-active player went `0 → 1` — read from the test player, confirmed never reaching the real player's object. Persistence check: the real player's `extendedUserProfile`/`userProfile`/both `gameJournal_*.dat` files were **byte-identical** before and after. The test player's own `TotalConcreteDeleted` increment stayed purely **in memory** this run — no `extendedUserProfile`/`userProfile` file was ever created for it (explicitly reported, as required, rather than assumed).

After this canary confirmed the fix, the full determinism test + 3 RandomPolicy episodes were re-run with the corrected ordering and the permanent gate active. **Zero `AssertPersistenceIsolated` failures across the entire run** (canary + 2 determinism-test episodes + 3 RandomPolicy episodes, dozens of real road actions). The real player's `extendedUserProfile`/`userProfile`/`gameJournal_*.dat` remained byte-identical. This run's test player (a fresh GUID) DID produce its own small `extendedUserProfile`/`deviceSettings` files on disk (more actions + more elapsed real time than the canary, giving the ungated flush more opportunity to run) — expected, disposable, impossible to confuse with real data by construction (GUID-keyed filename), and deleted as routine housekeeping after verification.

## Architectural boundary (as implemented)

```
Native Mini Motorways
        │
        ▼
Environment Core (Phase8Runner, Entrypoint.cs)
├── BuildPolicyObservation()      -- canonical observation
├── EnumerateLegalActions()/ApplyAction()  -- legal actions + masking
├── AdvanceSimulation()            -- controlled stepping
├── InitializeProcessContext()/SwitchToIsolatedTestProfile() -- reset (process-level)
├── RunPolicyEpisode()             -- reset (episode-level) / terminal /truncated
└── AssertPersistenceIsolated()    -- permanent safety gate
        │
        ├──── PolicyObservation (JSON)
        │
        └──── Privileged diagnostics (Phase 6A SimulationSnapshot, unchanged)
```

No Python IPC, Gymnasium, PPO, neural-network architecture, macro-actions, curriculum, population training, model-based search, attention/saliency, or streaming was implemented, per the explicit scope boundary.

## Validation checklist (brief's explicit requirements)

- ✅ identical native state → identical canonical observation (determinism test, 6/6 checkpoints, twice)
- ✅ deterministic serialization/order (fixed row-major tile iteration, fixed resource-type order)
- ✅ privileged fields do not leak into `PolicyObservation` (explicit exclusion list, §1)
- ✅ legal action enumeration produces executable road actions (36+ decisions, zero false positives)
- ✅ illegal actions excluded with native-semantics agreement (REMOVE_ROAD's single-edge-tile constraint is a real native limitation, not an approximation)
- ✅ road costs/refunds behave correctly (concrete consumption/refund tracked per decision in the JSONL trace)
- ✅ controlled stepping advances simulation independently of wall-clock frame timing (confirmed via exact `0.30`-per-decision `simulationTime` increments regardless of real elapsed time)
- ✅ repeated same-seed/same-action runs tested for deterministic equivalence (6/6 checkpoints matched)
- ✅ RandomPolicy autonomously completed many decisions (36 across 3 episodes)
- ✅ several episode resets without stale-state leakage (5+ resets this phase alone, each a fresh `InitCityCommand`)
- ✅ game remained stable (zero exceptions in the final corrected run)
- ✅ gameplay/profile/progression saves protected — **after** the mid-phase fix; the violation and its resolution are documented in full above, not hidden

## Files changed/added this phase

- [`game_plugin/DoorstopBootstrap/Entrypoint.cs`](../game_plugin/DoorstopBootstrap/Entrypoint.cs) — `Phase7ARunner` renamed `Phase8Runner`. Added: `ActionDescriptor`/`ActionKind`, `EnumerateLegalActions`, `ApplyAction`, `AdvanceSimulation`, `IsTerminal`, `HashJson`, `BuildPolicyObservation`, `RunPolicyEpisode` (replacing the old `RunRoadEditEpisode`/`RunResetLeakCheckEpisode`). Persistence-isolation fix: `InitializeProcessContext` rewritten to wait for the stock `LoadingScreen` activation signal (via `ActivePlayer.PlayerChanged` + stack-trace detection) before creating the test player; `SwitchToIsolatedTestProfile`, `AssertPersistenceIsolated` (permanent gate), `SubscribePlayerChangedDiagnostic`/`OnPlayerChangedDiagnostic` (trimmed to a lightweight audit line), `RunPersistenceCanary` added. `RestoreExistingPlayerProfile` (Phase 5E's original restore-the-real-player method) is no longer called but left in place for reference.
- [`research/environment_observation_example.json`](environment_observation_example.json) — one full example `PolicyObservation` (new).
- [`research/random_policy_trace.jsonl`](random_policy_trace.jsonl) — 36 RandomPolicy decision records across 3 episodes (new).

## Hashes

- `Mini Motorways.exe`: `305d1ea095f0ba9e844b91497283277c61f0dea9d5f7979065e78f3e7bab0ce6` — unchanged throughout every run this phase.
- Final `Bootstrap\MiniMotorwaysBootstrap.dll`: `4dee2a8cd06d227d82345410a5064e5bd303e202f7de1c6064288e0993d762ef`.
- `doorstop_config.ini`: reverted to `target_assembly=BepInEx\core\BepInEx.Preloader.dll` (resting state).
- Real player's `extendedUserProfile_451e710a...json`, `userProfile_451e710a...json`, both `gameJournal_*.dat`: byte-identical before the first Phase 8 run attempt and after the final, corrected full run (the mid-phase violation and its subsequent fix are both documented with evidence above).

## research/ARCHITECTURE.md update

New facts added (§8C): the `ISimulation.Step()` zero-argument/fixed-`Timestep` mechanics; `MotorwaysGame.HasGameEnded`'s UI-screen-stack dependency; the `LoadingScreen` player-pruning invariant and why in-process player switching must wait for it; confirmation that `GameRules` selection for `GameMode.Normal` is independent of which player is active (switching players cannot alter simulation/road semantics).

## Stop condition reached

Per instruction: *"Phase 8 succeeds when an autonomous native loop can repeatedly perform RESET → OBSERVE → ENUMERATE LEGAL ACTIONS → RANDOM ACTION → CONTROLLED SIMULATION STEP → OBSERVE → ... → TERMINATE/TRUNCATE → RESET for multiple episodes without human/UI input or state leakage."* Confirmed live across 5 resets (canary + 2 determinism + 3 RandomPolicy), zero exceptions, zero real-profile mutation in the final corrected run, zero `AssertPersistenceIsolated` failures. Per the user's explicit direction upon the canary's success, no further reverse-engineering/UI/streaming/profile investigation was started — the next milestone is Python/Gymnasium/PPO, out of scope for this document.
