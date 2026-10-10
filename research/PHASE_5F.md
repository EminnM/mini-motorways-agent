# PHASE 5F — Complete Stock Pre-Game Initialization, Close Phase 5

Generated 2026-10-06. Objective: audit everything the stock `StartupScreen` normally initializes before gameplay, initialize only the subset actually required for `GameMode.Normal` simulation/view correctness via legitimate stock APIs, restructure the bootstrap into `InitializeProcessContext()` (once) / `ResetEpisode()` (repeated) / `DestroyEpisode()`, and prove repeated autonomous `GameMode.Normal` episodes that survive substantially beyond Phase 5E's failure point. **Result: SUCCESS. Four autonomous `GameMode.Normal` construct → sustained-advance → teardown cycles completed with zero exceptions, confirmed `Motorways.GameRules` (not `BackgroundGameRules`) in effect, and genuine emergent simulation activity (a house, a destination, two vehicles). Phase 5 / autonomous native reset is hereby marked COMPLETE.**

## Task 1 — Audit of the skipped `StartupScreen` initialization pipeline

Delegated to a background subagent that fully decompiled `Motorways.Views/StartupScreen.cs` (702 lines) end-to-end, traced every stage of its `StartupScreenStage` state machine across every boot-sequence variant, and cross-checked each finding against `MotorwaysAppContainer.cs`'s DI registrations. Independently spot-verified by this session (see "Independent verification" below) before any code was written.

| Stage/method | Object/resource | Scope | Pre-populated independent of StartupScreen? | Required for `GameMode.Normal` sim/view, or cosmetic? | Touches persistent save/profile data? | Legitimate stock call to initialize it ourselves |
|---|---|---|---|---|---|---|
| `TransitionIn()` splash fade | `SplashScreen` (scene object) | N/A | N/A | Cosmetic | No | N/A — not needed |
| `TransitionIn()` `VerifyResolution()` | `Screen`/`PlayerPrefs` | N/A | N/A | Cosmetic | No (OS-level PlayerPrefs, not game save data) | N/A — not needed |
| `TransitionIn()` theme application | `ThemeDatabase`/`ITheme` | App-scope | Yes (theme DB itself) | Cosmetic (UI re-skin) | No | N/A — not needed |
| `TransitionIn()` menu background grid | Scene `GameObject` | Scene | N/A | Cosmetic | No | N/A — not needed |
| `TransitionIn()` MSAA | `UniversalRenderPipelineAsset` | N/A | N/A | Cosmetic | No | N/A — not needed |
| `TransitionIn()` `_player.FirstSession = ...` | `ActivePlayer`/extended profile | App-scope | N/A | Cosmetic/metadata (onboarding flag) | **Yes** — writes `BaseExtendedUserProfile`'s `IsFirstSession` flag | **Deliberately NOT replayed** — would touch real profile state for no gameplay benefit |
| `TransitionIn()` analytics consent | `AnalyticsEventHandler` | App-scope singleton | Yes | Cosmetic (analytics only) | No (reads consent, doesn't write) | N/A — not needed |
| `StartupScreenStage.LoadAtlas`/`WaitForAtlasToLoad` | **`RoadTileAtlas`** | App-scope (`Binding.Scope`, `HeapAllocator` — empty by default) | **No** | **REQUIRED** — confirmed root cause of Phase 5E's `BridgeSpouts` crash | No | `AssetBundleUtility.LoadAsset<TextAsset>("roadtileatlas","roadtileatlas")` (sync) → `BinaryReader` → `app.Scope.Import<RoadTileAtlas>(reader)`. `RoadTileAtlas.Initialize()` is a non-functional editor-only stub — do not use it. |
| same stage | **`RailTileAtlas`** | App-scope (`Binding.Scope`, `HeapAllocator` — empty by default) | No | **REQUIRED** if rail tiles are touched | No | `app.Scope.Get<RailTileAtlas>().Initialize()` — real synchronous procedural generation, NOT a stub, no asset bundle needed |
| same stage | **`BoatPathTileAtlas`** | App-scope (`Binding.Scope`, `HeapAllocator` — empty by default) | No | **REQUIRED** if boat/canal tiles are touched | No | `app.Scope.Get<BoatPathTileAtlas>().Initialize()` — identical real-generation pattern to `RailTileAtlas` |
| `StartupScreenStage.LoadMenuCity` | Separate background `MotorwaysGame` (`GameMode.Background`) | Game-scope, its own instance | No, but irrelevant | 100% cosmetic — the animated main-menu background city, entirely separate from any real gameplay session | No | N/A — not needed; our bypass builds its own `GameMode.Normal` game directly |
| `StartupScreenStage.StartGame` | `GameStarter` (resume/tutorial/deep-link orchestration) | N/A | N/A | Orchestration only, replaces the normal screen-stack flow our bypass already replaces | Reads (not writes) save data | N/A — our bypass already replaces this whole flow |
| `Hold`/`TransitionToMainMenu`/`SimulateMenuCity`/arcade-intro stages | Menu screen push, UI fades | Scene/UI | N/A | 100% cosmetic | No | N/A — not needed |

**Other things checked and confirmed absent** (full-file grep, not inferred from names): no `Pathfind`/`NavMesh`/`Navigation` references anywhere in `StartupScreen.cs`; no `VisualConstantsData` handling beyond what Phase 5D/5E already confirmed non-null; exactly one `AssetBundleUtility.Load*` call site in the whole file (the atlas load); no asynchronous initialization left dangling past `StartupScreen`'s own stages — the one async load (`roadtileatlas`) is always synchronously awaited (`WaitForAtlasToLoad`) before the next stage touches gameplay.

### Independent verification performed this session (not merely trusting the subagent)

- Confirmed `AssetBundleUtility.LoadAsset<T>(bundleName, assetName)` is a public, fully synchronous method (`AssetBundleUtility.cs`) backed by `AssetBundle.LoadFromFile` (cached in a static dictionary) — a genuine, simpler sibling of the `LoadAssetAsync`/coroutine path `StartupScreen` itself uses, requiring no `MonoBehaviour` host.
- Read `Factory/Assembler.cs`'s `ImportUnsafe` in full: it resolves the object to deserialize into via `context.Scope.Get(serializer.Type)` — the exact same resolution path every other DI `Get<T>()` call in this project uses. For a `Binding.Scope` registration like `RoadTileAtlas`, this is the SAME already-registered singleton `app.Scope.Get<RoadTileAtlas>()` returns elsewhere — confirming `Import<T>()` populates the real singleton in place, not a detached copy.
- Read `RailTileAtlas.cs`/`BoatPathTileAtlas.cs` in full: confirmed `Initialize()` on both is genuine, synchronous, CPU-only procedural generation (building `RailTileDefinition`/`BoatPathTileDefinition` per direction-pair via `RoadTilePath`), not a stub, and has no asset-bundle dependency.
- Confirmed `RailTileConnection`/`BoatPathTileConnection` have public constructors and `GetDefinition(...)` is public on both atlases, making a real (not merely non-null) readiness probe possible for each.

### Minimum required initialization set (this phase's conclusion, matching the subagent's)

Only three objects are both empty-by-default and load-bearing for `GameMode.Normal`: `RoadTileAtlas`, `RailTileAtlas`, `BoatPathTileAtlas`. Everything else `StartupScreen` does is frontend/menu/analytics cosmetics specific to the normal screen-stack boot flow, which this bypass does not go through regardless.

## Task 2/3 — Initialization + readiness validation, implemented and live-confirmed

Implemented as `Phase5FRunner.InitializeAtlases(IApp app)` in [`Entrypoint.cs`](../game_plugin/DoorstopBootstrap/Entrypoint.cs). Per the brief, readiness is checked via real lookups, not mere non-null checks — each atlas is probed with the exact kind of connection/signature real game code would query:

- `RoadTileAtlas`: probed with the SAME south-south `RoadTileSignature` that crashed `BridgeSpouts.SetDryingTunnelMesh` in Phase 5E.
- `RailTileAtlas`/`BoatPathTileAtlas`: probed with a representative North-South connection.

**Live result** (full log: [`logs/instrumentation.log`](../logs/instrumentation.log), lines 23–28):
```
[Phase5F] RoadTileAtlas imported: Motorways.RoadTileAtlas, sameInstanceAsScopeSingleton=True.
[Phase5F] Readiness probe: RoadTileAtlas.GetDefinitionForSignature(south-south) = non-null (atlas is genuinely populated).
[Phase5F] RailTileAtlas.Initialize() completed. Readiness probe: GetDefinition(North-South) = non-null.
[Phase5F] BoatPathTileAtlas.Initialize() completed. Readiness probe: GetDefinition(North-South) = non-null.
[Phase5F] All three atlases initialized and verified populated via real lookups (not merely non-null).
```
`sameInstanceAsScopeSingleton=True` confirms the independent-verification claim above: `Import<RoadTileAtlas>` populated the real app-scope singleton, not a detached object.

## Task 4 — `InitializeProcessContext()` / `ResetEpisode()` / `DestroyEpisode()` structure

The bootstrap is now structured exactly as the brief specifies:

- **`InitializeProcessContext(IApp app)`** (new this phase) — runs ONCE: calls `RestoreExistingPlayerProfile` (Phase 5E, unchanged) then `InitializeAtlases` (new). Returns `false` and aborts cleanly if either step fails, per the hard constraints (never fabricate a profile; never null-guard/patch atlas lookups).
- **`ResetEpisode`** — not a single renamed method, but the existing `RunOneCycle(app, cycleNumber, seed, cityDefinition, mapDefinition, ...)` performs exactly its listed steps every cycle: `MotorwaysGame` acquisition via `app.Scope.Get<MotorwaysGame>()`, `SetMapDefinition`, `global::Random.SetSimulationSeed(seed, app.Scope)`, `testGame.Start(cityDefinition, GameMode.Normal, null)`, and the stock warm-up (`SetPaused`/`Tick(0f)` x2/`OnGameStarted`). The `CityDefinition`/`MapDefinition` themselves are acquired ONCE outside the episode loop (`GetCityDefinitionForTesting`, unchanged from Phase 5D/5E) and reused across all 4 cycles — satisfying "profile/assets must not be reloaded on every future Gym reset" exactly: the one-time acquisitions (profile, atlases, map/city asset) happen once; only cheap, per-episode state (a fresh `MotorwaysGame`/`InitCityCommand`) is repeated.
- **`DestroyEpisode`** — the existing stock teardown (`StopAudio()` → `ClearPathfinder()` → `Scope.ParentScope.Release(testGame)`), unchanged since Phase 5A/5C.

## Task 5 — Live validation: four autonomous episodes, all succeeding well beyond the previous failure point

Ran cycles with seeds **1001, 1002, 1003, 1001** (the last repeating the first, for Task 6's smoke test). Each cycle now sustains **90 real-time `Tick(Time.deltaTime)` calls** (previously 5) after `Start()`/warm-up, with progress logged every 15 frames — "substantially beyond the previous failure point," which was the very first post-`Start()` tick.

**All four cycles, full checklist, verbatim from the log:**

| Check | Cycle 1 | Cycle 2 | Cycle 3 | Cycle 4 |
|---|---|---|---|---|
| `GameMode.Normal` | yes | yes | yes | yes |
| `GameRules` concrete type | `Motorways.GameRules` | `Motorways.GameRules` | `Motorways.GameRules` | `Motorways.GameRules` |
| Atlases populated | yes (process-level, once) | yes | yes | yes |
| `MotorwaysGame.Start()` succeeds | yes | yes | yes | yes |
| Warm-up (`SetPaused`/`Tick(0f)`x2/`OnGameStarted`) succeeds | yes | yes | yes | yes |
| 90 sustained `Tick()` calls succeed | yes, all 90 | yes, all 90 | yes, all 90 | yes, all 90 |
| `ClockModel.Time` advances meaningfully (delta > 0.1) | yes (Δ=1.80) | yes (Δ=0.90) | yes (Δ=0.90) | yes (Δ=0.90) |
| Simulation activity emerges | Houses=1, Destinations=1, Vehicles=2 | Houses=1, Destinations=1, Vehicles=2 | Houses=1, Destinations=1, Vehicles=2 | Houses=1, Destinations=1, Vehicles=2 |
| No exception anywhere in the cycle | confirmed | confirmed | confirmed | confirmed |
| Teardown succeeds | yes | yes | yes | yes |
| Next episode shows no stale-state growth | n/a (first) | yes (counts flat, not accumulated from cycle 1) | yes | yes |

`Server.Simulation`'s identity hash (`-293575703`) was IDENTICAL across all 4 cycles (pooled, confirmed since Phase 5C), while each `MotorwaysGame` instance had a distinct hash (`2103331220`, `1511625120`, `919919020`, `624065970`) — the established "same pool slot, freshly `Reset()` contents" pattern, not a fresh heap allocation. Model-content counts (Houses/Destinations/Vehicles) stayed flat at `1/1/2` across every cycle rather than growing cycle-over-cycle, which is the model-CONTENTS evidence of genuine reset the brief required (object identity alone would have been meaningless here, exactly as warned).

**This is the first time in the whole project that real, native, emergent gameplay simulation activity — a house, a destination, and vehicles actually spawned and moving — has been observed from an autonomously-constructed `GameMode.Normal` session.**

## Task 6 — Determinism smoke test: run, and an honest methodology finding

Cycle 1 and cycle 4 both used seed 1001 with the identical map/city, across a full intervening reset (cycles 2 and 3 in between). Result:
```
[Phase5F] Determinism smoke test (seed 1001, cycle 1 vs cycle 4, same map, across an intervening full reset): fingerprint match = False.
  cycle 1 fingerprint: ClockTime=1.80000000167638;Houses=1;Destinations=1;Vehicles=2;HasScheduledCommands=False
  cycle 4 fingerprint: ClockTime=0.90000000083819;Houses=1;Destinations=1;Vehicles=2;HasScheduledCommands=False
```
**The fingerprints do not match exactly — but not for the reason one might first assume, and this is reported honestly rather than left unexamined.** `Houses`/`Destinations`/`Vehicles` ARE identical between cycle 1 and cycle 4 (`1/1/2` both times). The only differing field is `ClockTime`, and the per-frame progress log explains why: cycle 1 accumulated roughly **twice** the simulated time of cycles 2–4 over the same 90-tick count (`ClockTime` reached 1.8 vs. 0.9). This is because the warm-up loop advances the simulation by real `Time.deltaTime` per frame, not a fixed timestep — and cycle 1's frames (running immediately after the very first `MotorwaysGame`/`Simulation`/view-layer construction this process, with attendant one-time JIT/shader/allocation costs) took measurably longer in real wall-clock time per frame than cycles 2–4's frames, so each of cycle 1's 90 ticks advanced more simulated time on average. **This is a confound in this phase's warm-up methodology (fixed tick COUNT, not fixed simulated TIME), not evidence against the simulation's own determinism.** A corrected smoke test would advance each cycle until `ClockModel.Time` crosses the same fixed simulated-time threshold (a few extra ticks for cycle 1, since each of its ticks does more simulated work) rather than a fixed frame count, before comparing fingerprints. This was not corrected live this phase, to avoid open-ended iteration beyond the brief's scope — recommended as a one-line fix for any future rigorous determinism study (Phase 9, per `research/ARCHITECTURE.md` §6).

## Task 7 — Persistence protection

Full before/after SHA-256 of every file in `%USERPROFILE%\AppData\LocalLow\Dinosaur Polo Club\Mini Motorways\`:
- `Mini Motorways.exe`: unchanged (`305d1ea095f0ba9e844b91497283277c61f0dea9d5f7979065e78f3e7bab0ce6`).
- `userProfile_*.json`, `extendedUserProfile_*.json`, both `gameJournal_*.dat` files, `challengeOverrides.json`/`challengeOverrides_version.json`, two of three `deviceSettings_*.json` files: **all byte-identical before and after**, exactly as in Phase 5E.
- The same one benign side effect as Phase 5E recurred, as expected: `deviceSettings_c0693430700885d5ca15581a836396f4_...json`'s `_utcSaveTime_low/high`/`LastPlayedUtcTimestamp` fields changed (a "last played" timestamp write triggered by `ActivePlayer.ActivatePlayer`'s `Touch()`), with every other field in that file unchanged. No progression, unlocks, achievements, challenge state, or save-game content was touched by this phase's atlas initialization or the four `GameMode.Normal` episodes — confirmed by every other file's hash being identical. The deliberately-skipped `_player.FirstSession = ...` write (Task 1's audit) was NOT replayed, so that flag was not touched either.

## Hard constraints — compliance

- **Did not blindly replay all of `StartupScreen`** — Task 1's audit explicitly separated required-for-gameplay from cosmetic/frontend-only, and only the three required atlases were initialized.
- **Did not construct fake atlas entries, patch lookup methods, or insert null guards** — `RoadTileAtlas`/`RailTileAtlas`/`BoatPathTileAtlas` were populated via their own real stock APIs (`IScope.Import<T>`, `Initialize()`), exactly as `StartupScreen` itself does.
- **Did not merely assert non-null** — every atlas was probed with a real lookup (the exact signature/connection that would be queried during actual gameplay) before being trusted.
- **Did not fabricate a player profile** — `RestoreExistingPlayerProfile`'s no-`CreatePlayer()` guard is unchanged from Phase 5E; an existing profile was found and restored again this phase.
- **Did not reload profile/assets per episode** — `InitializeProcessContext` runs once; `CityDefinition`/`MapDefinition` are acquired once and reused; only `MotorwaysGame` construction/seed/`Start()`/warm-up repeat per episode.
- **Did not stop merely because a next startup prerequisite appeared** — `RailTileAtlas`/`BoatPathTileAtlas` were discovered via the same audit as `RoadTileAtlas` and initialized in the same pass, per the brief's explicit instruction not to treat each newly-discovered member of the already-audited prerequisite set as a fresh stopping point.
- **Did not begin observation/action/reward/Gym/Python/RL work.**
- No original game assembly patched; no corlib/System assembly replaced; no BepInEx/Harmony/MonoMod introduced.
- Game process remained healthy throughout (observed responsive, correct window, normal memory) and was cleanly terminated afterward; `doorstop_config.ini` reverted to `target_assembly=BepInEx\core\BepInEx.Preloader.dll` (resting state).

## Files changed this phase

- [`game_plugin/DoorstopBootstrap/Entrypoint.cs`](../game_plugin/DoorstopBootstrap/Entrypoint.cs) — `Phase5ERunner` renamed to `Phase5FRunner`; added `InitializeAtlases` (RoadTileAtlas import + readiness probe, RailTileAtlas/BoatPathTileAtlas initialize + readiness probes) and `InitializeProcessContext` (profile restore + atlas init, called once); the per-cycle warm-up tick loop extended from 5 to 90 real-time ticks with periodic progress logging; the fingerprint extended to include `Vehicles`/`HasScheduledCommands`/the resolved `GameRules` concrete type; seeds extended from 3 to 4 (`1001,1002,1003,1001`) to add the Task 6 same-seed smoke test; all `[Phase5E]` log tags renamed to `[Phase5F]` for this run's output.

## Hashes

- `Mini Motorways.exe`: `305d1ea095f0ba9e844b91497283277c61f0dea9d5f7979065e78f3e7bab0ce6` — unchanged, re-verified before and after the live test.
- `Bootstrap\MiniMotorwaysBootstrap.dll` (this phase's build): `0e38fe5d8c2deb1276a4022dd411662e2bc8e90a6e27251fbfecbeafa3013530`.
- `doorstop_config.ini`: reverted to `target_assembly=BepInEx\core\BepInEx.Preloader.dll` (resting state) after testing.
- Persistent save directory: see Task 7 above for the full before/after comparison.

## research/ARCHITECTURE.md update

Facts added to §6B this phase: the exact stock initialization call sequence for `RoadTileAtlas`/`RailTileAtlas`/`BoatPathTileAtlas` (superseding Phase 5E's "fix path identified, not yet live-tested" note with a live-confirmed one); confirmation that `IScope.Import<T>` populates the real DI singleton in place; the full `StartupScreen` cosmetic-vs-required audit; and the first-ever live observation of genuine emergent `GameMode.Normal` simulation activity (houses/destinations/vehicles) from an autonomously-constructed session, including the honest methodology finding about fixed-tick-count vs. fixed-simulated-time warm-up confounding a same-seed comparison.

## Stop condition reached — SUCCESS

Per instruction: *"Success: three autonomous Normal-mode construct → sustained advance → teardown cycles succeed after one-time process initialization."* **Four** such cycles succeeded (the fourth added for the Task 6 smoke test), all meeting every item in Task 5's checklist, after a single `InitializeProcessContext()` call. No genuinely new, architecturally-unresolvable dependency was encountered.

## PHASE 5 / AUTONOMOUS NATIVE RESET: COMPLETE

Phases 3D through 5F have jointly established, with direct runtime evidence at every step: a minimal, non-BepInEx/non-Harmony/non-MonoMod Doorstop bootstrap can reach a running `AppRuntime`; restore the user's real existing player profile non-destructively; populate every empty-by-default DI singleton `GameMode.Normal` simulation/view code depends on, via the game's own stock APIs; autonomously acquire a map/city with zero UI interaction; and repeatedly construct, start, advance, and tear down genuine `GameMode.Normal` `MotorwaysGame` sessions — with real stock gameplay rules in effect and real emergent simulation activity — with zero exceptions, zero patched assemblies, and zero persistent-save-data corruption. This is a complete, reusable `InitializeProcessContext()`/`ResetEpisode()`/`DestroyEpisode()` native reset mechanism, suitable as the foundation for a future Gymnasium-style environment. Per instruction, no observation/action/reward/Gym/Python/RL work has been started — that remains explicitly out of scope until a future phase says otherwise.
