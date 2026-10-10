# PHASE 5E — Native Profile Initialization

Generated 2026-10-06. Objective: find and invoke the stock, non-destructive path that turns `ActivePlayer.HasActivePlayer` from `false` to `true` (Phase 5D's confirmed `GameMode.Normal` blocker), using an EXISTING saved profile only — never fabricating a new one — then retry the Phase 5C/5D lifecycle under `GameMode.Normal`. **Result: the profile blocker is SOLVED and confirmed live — `testGame.Start(cityDefinition, GameMode.Normal, null)` now completes without throwing, for the first time in this project. A second, distinct, previously-latent blocker was immediately exposed one step later (inside the first `Tick()`), and its exact root cause and stock fix path have been identified via decompilation, but not yet invoked live.** Per the explicit stop condition ("EITHER success OR identify the next exact dependency/blocker with runtime evidence"), this phase stops here with the second outcome.

## Task 1 — Static analysis: tracing the stock profile-restore path

Delegated to a background subagent (decompiling `App.dll` fresh, no assumptions from names). Full chain found with concrete call-site evidence:

- **`ActivePlayer`'s private `_player` field has exactly one writer**: `ActivatePlayer(Player newActivePlayer)` (also declared on `IActivePlayer`). It (un)subscribes `DataChanged`/`SavedGamesChanged`, fires `PlayerChanged`, calls `CheckAchievementsRetroactively()` and `Touch()` — all in-memory.
- **Callers of `ActivatePlayer(`**, found by grepping the full decompiled project: `Motorways.Views/LoadingScreen.cs` (the real startup path, unconditional on every launch), `Motorways.Views/ProfileSelectScreen.cs` (two call sites, reachable only from explicit manual profile-switch/creation UI button clicks), and `PlayerDatabase.RemovePlayer` (re-activation after a deletion, not relevant here).
- **`PlayerDatabase.MostRecentPlayer`** is a pure, side-effect-free read: it scans the in-memory `_players` dictionary (populated earlier by `AddUserProfile`/`AddDeviceSettings`/`AddSavedGame`, themselves invoked only while the storage layer reads already-existing on-disk/cloud files) and returns the entry with the latest `DeviceSettings.LastPlayedUtcTime`. `PlayerDatabase.CreatePlayer()` is the only method that fabricates a brand-new `Player` (`Guid.NewGuid()`-based id) — confirmed as the one path to avoid.
- **The real startup orchestrator is `Motorways.Views.LoadingScreen`, not `ProfileSelectScreen`/`ProfileCreationScreen`.** `LoadingScreen.TransitionIn` calls `_storage.LoadAll(OnPlayerDataLoaded)`; its `Tick()` then runs, unconditionally, on every launch:
  ```csharp
  Player player = _playerDatabase.MostRecentPlayer;
  if (player == null) { player = _playerDatabase.CreatePlayer(); ... }
  _activePlayer.ActivatePlayer(player);
  ```
  `StartupScreen` (pushed only after this activation) already assumes `_player` is populated. `ProfileSelectScreen`/`ProfileCreationScreen` are reachable only from explicit manual UI button clicks — the single-existing-profile case never routes through them at all.

This session additionally decompiled the two concrete building blocks directly (`IPersistentStorageService.LoadAll(Action)`, `PersistentStorageService`/`LocalFileStorage`/`SteamCloud`) and confirmed:

- `LocalFileStorage.LoadAll` (used by `WindowsBaseEnvironment`) is **fully synchronous**: it reads every recognized file in `_hardwareCapabilities.PersistentStoragePath`, calls each file's `IStorableTypeHandler.ProcessLoadedStorable` (which populates `PlayerDatabase` as a side effect), then invokes `loadCompleteCallback` — all before returning.
- `SteamCloud : LocalFileStorage` (the type actually registered by `WindowsSteamEnvironment`, confirmed as this build's real provider — `WindowsBaseEnvironment`'s plain `LocalFileStorage` registration is overridden for Steam) reads Steam Cloud files synchronously via `SteamworksApi.GetCloudFiles()`/`ReadCloudFile()`, then calls `base.LoadAll(...)` — so the full chain is synchronous end-to-end on this build too.
- `IPersistentStorageService`, `PlayerDatabase`, and `IActivePlayer`/`ActivePlayer` are all registered as ordinary `Binding.Scope` singletons in `AppContainer.cs` (lines 116/119/121) — directly resolvable via `app.Scope.Get<T>()` from our bootstrap, exactly like every other DI-resolved type used in this project since Phase 5A.

**Conclusion**: the legitimate, non-destructive restore chain is `app.Scope.Get<IPersistentStorageService>().LoadAll(callback)` → `app.Scope.Get<PlayerDatabase>().MostRecentPlayer` (pure read) → `app.Scope.Get<ActivePlayer>().ActivatePlayer(existingPlayer)`. All three calls are plain public API, reachable from our existing bootstrap with zero reflection and zero patching.

## Task 2/3 — Live probe: profile restoration, invoked

Implemented as `Phase5ERunner.RestoreExistingPlayerProfile` in [`Entrypoint.cs`](../game_plugin/DoorstopBootstrap/Entrypoint.cs), called at the start of the lifecycle coroutine, before anything else touches `GameMode.Normal`. Per the hard constraint, it explicitly checks `PlayerDatabase.MostRecentPlayer == null` and **aborts without calling `CreatePlayer()`** if no existing profile is found — no profile is ever fabricated by this code.

**Live result** (full log in [`logs/instrumentation.log`](../logs/instrumentation.log), lines 15–23):
```
[Phase5E] IPersistentStorageService.LoadAll callback fired = True (elapsed 0,000s -- confirms the synchronous path decompiled above).
[Phase5E] PlayerDatabase.PlayerCount after load = 1.
[Phase5E] Found existing player via MostRecentPlayer: Id=451e710aba7246fdb57170f197f21b1e, LocaleId=tr, LastPlayedUtcTime=6.10.2026 10:52:00.
[Phase5E] ActivePlayer.ActivatePlayer(existingPlayer) completed without throwing. HasActivePlayer = True.
```
The user's real, pre-existing profile (`userProfile_451e710aba7246fdb57170f197f21b1e.json`, confirmed present on disk beforehand) was loaded and activated — not a fabricated one. `ActivePlayer.HasActivePlayer` confirmed `True` both immediately after activation and again inside each of the three test cycles (lines 34, 58, 82).

## Task 4 — Retry Phase 5C/5D lifecycle under `GameMode.Normal`

Ran 3 cycles, seeds 1001/1002/1003, same map (`LosAngeles`), identical to Phase 5D's structure except for the profile restore and the seed set. **Result, decisively new**:

```
[Phase5E] Cycle 1: testGame.Start(cityDefinition, GameMode.Normal, null) completed without throwing.
```
**This is the first time in the whole project that `MotorwaysGame.Start()` under `GameMode.Normal` has completed without throwing.** The exact `CameraView.Initialize` `NullReferenceException` that blocked Phase 5D is gone — directly attributable to the profile restore (`ActivePlayer.ZoomLevel` no longer dereferences a null `_player`).

A **new, distinct** `NullReferenceException` then occurred one step later, during the immediately-following `testGame.Tick(0f)` call (all three cycles, identical stack):
```
System.NullReferenceException: Object reference not set to an instance of an object
  at Motorways.BridgeSpouts.SetDryingTunnelMesh (Motorways.RoadTileDefinition deadEndMesh) ... BridgeSpouts.cs:23
  at Motorways.Views.RoadView.OnCreatedInScope (Factory.IScope scope) ... RoadView.cs:95
  at Factory.Assembler+TypeAssembler`1[T].Assemble/.Create ...
  at Motorways.Views.DeadEndRoadView.Initialize (...) ... DeadEndRoadView.cs:164
  at Motorways.Views.TileView.ShowDeadEnd (...) ... TileView.cs:1345
  at Motorways.Views.TileView.RebuildDynamicRoads (...) ... TileView.cs:945
  at Motorways.Views.TileView.Tick (...) ... TileView.cs:404
  ... at Motorways.MotorwaysGame.Tick (System.Single frameTime) ... MotorwaysGame.cs:1117
```

### Root cause, identified via decompilation (static only — not yet live-tested)

`RoadView.OnCreatedInScope` (RoadView.cs:109-113) does:
```csharp
using RoadTileSignature roadTileSignature = scope.Get<RoadTileSignature>();
roadTileSignature.AddConnection(new RoadTileConnection(new RoadTileNode(TileDirection.South), new RoadTileNode(TileDirection.South)));
bridgeSpouts.SetDryingTunnelMesh(_roadTileAtlas.GetDefinitionForSignature(roadTileSignature));
```
Unlike the sibling method `RoadView.SetSignature` (which null-checks the same kind of lookup via `Diagnostics.Verify(definitionForSignature != null, ...)`), this call passes `_roadTileAtlas.GetDefinitionForSignature(...)`'s result **directly and unguarded** into `SetDryingTunnelMesh`, which immediately dereferences it (`deadEndMesh.mesh.roadMesh`). `RoadTileAtlas.GetDefinitionForSignature` (RoadTileAtlas.cs) is a plain dictionary lookup (`_signatureToDefinition.TryGetValue(...)`) that returns `null` whenever the signature isn't present — which is **always**, if the dictionary was never populated.

Decompiling `RoadTileAtlas` itself confirms it has **no self-population logic at all** — it is marked `[Factory.Serializable(1)]` and is registered in `MotorwaysAppContainer.cs:263` as an ordinary `HeapAllocator`-backed `Binding.Scope` singleton (a fresh, empty instance by default). Its one-time population is performed by **`Motorways.Views.StartupScreen`**, a stock screen our bypass never runs (exactly analogous to Phase 5D/5E's `LoadingScreen`/profile finding). The exact mechanism, found in `StartupScreen.Tick()`:
```csharp
// CurrentStage == StartupScreenStage.LoadAtlas:
_atlasAsyncLoadResult = AssetBundleUtility.LoadAssetAsync("roadtileatlas", "roadtileatlas", this);

// CurrentStage == StartupScreenStage.WaitForAtlasToLoad, once loaded:
TextAsset textAsset = _atlasAsyncLoadResult.asset as TextAsset;
using BinaryReader reader = new BinaryReader(new MemoryStream(textAsset.bytes));
roadTileAtlas = _appScope.Import<RoadTileAtlas>(reader);   // <-- deserializes INTO the scope's existing singleton
...
_appScope.Get<RailTileAtlas>().Initialize();
_appScope.Get<BoatPathTileAtlas>().Initialize();
```
`IScope.Import<T>(BinaryReader reader)` (`Factory/IScope.cs:31`, implemented `Factory/Scope.cs:216`) is a **public** interface method — confirmed callable exactly like every other `Scope.Get<T>()` call already used throughout this project, with no reflection and no patching. `AssetBundleUtility.LoadAssetAsync` is the same public async-load utility already used (as `LoadPrefabAsync`) for `CityDefinition` prefabs since Phase 5C/5D.

**This is the same category of problem Phase 5D/5E has now solved twice: a one-time stock-screen data-population step our bypass skips because it starts from an already-booted `AppRuntime` instead of running the full `LoadingScreen` → `StartupScreen` boot sequence.** The fix path is concrete and fully identified, but — per this phase's explicit stop condition and to avoid scope creep into untested territory — it was **not** implemented or invoked live this phase. Note also that `StartupScreen` additionally initializes `RailTileAtlas` and `BoatPathTileAtlas` in the same stage; these are likely similarly-empty and may surface as further blockers once `RoadTileAtlas` itself is fixed (rail/boat-specific tiles, not exercised by this phase's plain-road signature).

### Reset-verification across cycles

All three cycles failed identically at the same point (`constructionSucceeded` never reached `true`), so no fingerprint data was produced and the model-contents reset check could not run this phase:
```
[Phase5E] Reset verification: 0/3 cycles completed successfully.
[Phase5E] Fewer than 2 cycles succeeded -- reset verification across cycles skipped.
```
Teardown (`StopAudio`/`ClearPathfinder`/`Scope.ParentScope.Release`) completed without throwing after each of the three failed cycles — consistent with every prior phase's teardown behavior.

## Task 5 — Persistent-state protection

A background subagent, independent of the live test, traced every disk-write path reachable from `Start()`/`Tick(0f)`/teardown and found it **SAFE** for this phase's test shape: `TrySave(GameJournalMotive.Autosave)` is gated by a real-time 300-second threshold (`Time.time`, immune to `Tick(0f)`'s zero frame-delta) or a game-time-hours threshold a handful of zero-delta ticks cannot cross; `ActivePlayer.LocalSavedGame`'s setter (the only path to `_storage.Delete()/Store()`) is reachable only via that blocked autosave or `OnGameEnd()`'s `DeleteLocalSave()` (not called by this test); teardown touches no storage. This was confirmed empirically:

**Before/after SHA-256 of every file in `%USERPROFILE%\AppData\LocalLow\Dinosaur Polo Club\Mini Motorways\`:**
- `Mini Motorways.exe`: unchanged (`305d1ea095f0ba9e844b91497283277c61f0dea9d5f7979065e78f3e7bab0ce6`).
- `userProfile_451e710aba7246fdb57170f197f21b1e.json`, `extendedUserProfile_451e710aba7246fdb57170f197f21b1e.json`, both `gameJournal_*.dat` files, `challengeOverrides.json`, `challengeOverrides_version.json`, two of the three `deviceSettings_*.json` files — **all byte-identical before and after.**
- **One exception, found and reported honestly rather than glossed over**: `deviceSettings_c0693430700885d5ca15581a836396f4_451e710aba7246fdb57170f197f21b1e.json` (the device-settings file for this machine) changed hash. Diffing its (obfuscated/spaced-out, but JSON-structured) contents shows **only** two fields changed: `_utcSaveTime_low`/`_utcSaveTime_high` (an internal last-saved timestamp pair) and `LastPlayedUtcTimestamp`. No gameplay field (`ZoomLevel`, `VolumeSetting`, `NightMode`, controller mappings, etc.) changed. This is the same "you played at this time" metadata write any ordinary launch to the main menu produces — `ActivePlayer.ActivatePlayer`'s `Touch()` call updates `DeviceSettings.LastPlayedUtcTime` in memory, and evidently something in the device-settings path (not decompiled by either subagent, likely a periodic settings-dirty-flag flush distinct from the gameplay-save path they traced) flushes it to disk promptly rather than waiting for the 300s autosave gate. **This slightly corrects the earlier "SAFE" prediction**: that investigation covered `LocalSavedGame`/game-journal writes correctly, but did not anticipate this one lightweight settings-timestamp write. No progression, unlocks, achievements, challenge state, or save-game content was affected — confirmed by every other file's hash being identical — but this is flagged explicitly rather than silently absorbed into "nothing changed."

## What this phase establishes

**Established, with direct runtime evidence:**
- The stock existing-profile-restore chain (`IPersistentStorageService.LoadAll` → `PlayerDatabase.MostRecentPlayer` → `ActivePlayer.ActivatePlayer`) works exactly as decompiled, restores the user's REAL existing profile, and is safe to invoke from this bootstrap.
- `GameMode.Normal`'s player-profile blocker (Phase 5D) is **solved**: `MotorwaysGame.Start(cityDefinition, GameMode.Normal, null)` now completes without throwing, for the first time in this project.
- A second, previously-masked blocker (`RoadTileAtlas` never populated because `StartupScreen` never ran) is now exposed and its exact root cause and fix path are identified via decompilation, with a concrete call sequence (`AssetBundleUtility.LoadAssetAsync("roadtileatlas","roadtileatlas",this)` → `IScope.Import<RoadTileAtlas>(reader)`).
- A minor, narrowly-scoped persistent-state side effect (one device-settings timestamp write) was detected via before/after hashing and is fully characterized as non-destructive to progression/save data — but it is a real deviation from "nothing is written" and is documented as such rather than ignored.

**Not established (explicitly out of scope, per this phase's stop condition):**
- Whether invoking `IScope.Import<RoadTileAtlas>(reader)` (and `RailTileAtlas.Initialize()`/`BoatPathTileAtlas.Initialize()`) live actually resolves this blocker — not attempted this phase.
- The 3-cycle determinism/reset-via-model-contents check (Task 4's second half) — blocked by this new failure; no fingerprints were produced.
- Whether any further blockers exist beyond `RoadTileAtlas` once it's fixed (rail/boat atlases are a named, concrete suspicion, not yet investigated).

## Stop condition reached

Per instruction: *"EITHER (A) stock profile init + autonomous GameMode.Normal construct→start→advance→teardown succeeds repeatedly, OR (B) identify the next exact native dependency/blocker with runtime evidence."* Outcome (B): the profile blocker is solved, and the next exact blocker (`RoadTileAtlas` never populated, because `StartupScreen`'s `LoadAtlas`/`WaitForAtlasToLoad` stages never ran) has been identified with full runtime evidence (live stack trace) and a concrete, decompiled stock fix path. This phase stops here, as instructed, rather than implementing/testing that fix or proceeding to observations/actions/Gymnasium/IPC/reward/RL.

## Hard constraints — compliance

- **Did not fabricate a profile.** `RestoreExistingPlayerProfile` explicitly checks `PlayerDatabase.MostRecentPlayer == null` and aborts (logging and stopping) rather than calling `PlayerDatabase.CreatePlayer()`. In this live run, an existing profile WAS found, so this path was not exercised, but the code and the log line documenting the abort case are both in place for future reference.
- **Did not mutate/delete/reset the user's real save/profile data** in any way that affects gameplay state: confirmed via before/after SHA-256 of every file in the persistent-storage directory. One non-gameplay side effect (a device-settings timestamp) was found and is explicitly documented above, not hidden.
- **Did not patch, null-guard, or disable** `CameraView`, `ActivePlayer`, `RoadView`, `BridgeSpouts`, or `RoadTileAtlas` — the new failure was investigated via decompilation only; nothing was changed to make it disappear.
- **Did not switch back to `GameMode.Background`** to route around either blocker.
- No road/action injection, no reward design, no PPO/RL, no Python IPC, no Gymnasium, no observation design, no streaming — none attempted.
- No original game assembly patched; no corlib/System assembly replaced; no BepInEx/Harmony/MonoMod introduced.
- Game process remained healthy throughout the test and was cleanly terminated afterward; `doorstop_config.ini` reverted to `target_assembly=BepInEx\core\BepInEx.Preloader.dll` (resting state).

## Files changed this phase

- [`game_plugin/DoorstopBootstrap/Entrypoint.cs`](../game_plugin/DoorstopBootstrap/Entrypoint.cs) — `Phase5DRunner` renamed to `Phase5ERunner`; added `RestoreExistingPlayerProfile` (the profile-restore chain, with the no-fabrication guard); the restore is now called, and gates, the start of the lifecycle coroutine; seeds array changed from Phase 5D's single-seed probe to the full `{1001u, 1002u, 1003u}` 3-cycle run; the cycle-1-vs-cycle-4 same-seed determinism check (which assumed 4 cycles with a repeated seed) was replaced with a 3-cycle model-contents reset-verification pass (not exercised this phase, since all 3 cycles failed identically).

## Hashes

- `Mini Motorways.exe`: `305d1ea095f0ba9e844b91497283277c61f0dea9d5f7979065e78f3e7bab0ce6` — unchanged, re-verified before and after the live test.
- `Bootstrap\MiniMotorwaysBootstrap.dll` (this phase's build): `b7c5666915723e74e7b0b914a6af9029ae3f268358ea59a50da31188f97d3a3e`.
- `doorstop_config.ini`: reverted to `target_assembly=BepInEx\core\BepInEx.Preloader.dll` (resting state) after testing.
- Persistent save directory: see Task 5 above for the full before/after file-hash comparison.

## research/ARCHITECTURE.md update

Facts added to §6B this phase (all experimentally established via live probes except where noted static-only):
- The stock existing-profile-restore chain (`IPersistentStorageService.LoadAll` → `PlayerDatabase.MostRecentPlayer` → `ActivePlayer.ActivatePlayer`), confirmed live to restore the user's real profile and resolve the Phase 5D `ActivePlayer.HasActivePlayer` blocker.
- `GameMode.Normal`'s `MotorwaysGame.Start()` now completes successfully once a profile is active — confirmed live for the first time.
- `RoadTileAtlas`/`RailTileAtlas`/`BoatPathTileAtlas` are registered as empty DI singletons, populated only by `StartupScreen`'s `LoadAtlas`/`WaitForAtlasToLoad` stages (`AssetBundleUtility.LoadAssetAsync("roadtileatlas","roadtileatlas",this)` + `IScope.Import<RoadTileAtlas>(reader)`) — identified via decompilation as the next blocker a bypass that skips `StartupScreen` will hit, with the exact stock, non-UI fix path documented (not yet live-tested).
- A device-settings file's save-timestamp field is written promptly (not gated by the 300s autosave threshold that governs gameplay saves) as a side effect of `ActivePlayer.ActivatePlayer`/`Touch()` — a minor correction to the Phase 5E persistence-safety investigation's scope.

## Recommended next step (not started)

**Phase 5F**: implement and live-test the `RoadTileAtlas` population call (`AssetBundleUtility.LoadAssetAsync("roadtileatlas","roadtileatlas",this)` → `app.Scope.Import<RoadTileAtlas>(reader)`, plus `app.Scope.Get<RailTileAtlas>().Initialize()`/`app.Scope.Get<BoatPathTileAtlas>().Initialize()`) from the bootstrap, before constructing any test `MotorwaysGame`. If that resolves this blocker, retry the full Task 4 (3-cycle `GameMode.Normal` run with model-contents reset verification) that this phase could not complete.
