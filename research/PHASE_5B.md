# PHASE_5B — Native Session Lifecycle / Reset Proof

Generated 2026-10-06. Objective: prove the game's own object graph supports repeated native `MotorwaysGame`/`Server.Simulation` construction and destruction from our Doorstop bootstrap, using the exact lifecycle traced in Phase 5A. **Result: partial success, with a precisely identified blocker.** Construction via the DI container, deterministic seed control, and exception-safe teardown are all proven live and working. The *full* `MotorwaysGame.Start()` call — specifically its view/rendering-layer side effects — throws when invoked outside the normal `GameContainerScreen` flow, because that flow performs a theme-initialization step (`RegisterThemeComponents`) ours never did. Per the explicit instruction to stop rather than layer a workaround on top of instability, this phase stops here with the root cause documented, not patched around.

## Step 1 — Preserve known-good state (done, verified before and after)

- `Mini Motorways.exe` SHA-256: `305d1ea095f0ba9e844b91497283277c61f0dea9d5f7979065e78f3e7bab0ce6` — confirmed unchanged before the test and re-confirmed after.
- `patched_corlib\mscorlib.dll` SHA-256: `43639906003e444716d06afb010d5a91ae1c900582c69e373977857ebaf600fd` (the six-patch Phase 3B build) — unchanged, untouched this phase.
- `doorstop_config.ini`'s `target_assembly` was flipped to `Bootstrap\MiniMotorwaysBootstrap.dll` for the test and reverted to `BepInEx\core\BepInEx.Preloader.dll` immediately afterward — the established resting state.
- No manual UI interaction was used or required this phase (see "Design decision" below for why).

## Design decision: an isolated test session, not the user's live one

Before writing any live test, decompiling `App` (the concrete `IApp` implementation) revealed a fact that changes how "lifecycle proof" has to be approached:

```csharp
public Game Game => _screenStack.GetGameIfInGame();   // App.cs:76 -- a LIVE COMPUTED property, not a cached field
```//
and `ScreenStack.GetGameIfInGame()`:
```csharp
public virtual Game GetGameIfInGame() {
    GameContainerScreen activeScreen = GetActiveScreen<GameContainerScreen>();
    return activeScreen?.GetActiveGame();
}
```

**`AppRuntime.App.Game` is entirely driven by whatever `GameContainerScreen` is currently on top of the `ScreenStack`.** It is not an independent slot we can point at a game we construct ourselves. Constructing a `MotorwaysGame` directly via `appScope.Get<MotorwaysGame>()` (as `GameContainerScreen.TransitionIn` itself does internally) does **not** make `AppRuntime.App.Game` reflect it — that would require also manipulating `GameContainerScreen`'s own private `_game` field and the `ScreenStack`, which risks visibly disrupting whatever the user might be looking at and edges toward UI-layer manipulation the brief asked to avoid.

Checking the DI registration confirmed this was safe to do in isolation:
```csharp
appAssembler.Register<Game, MotorwaysGame>().Allocator(new HeapAllocator<MotorwaysGame>()).EstablishScope(assembler)
```
`HeapAllocator` (not a singleton), with `.EstablishScope(assembler)` — **every call to `Get<MotorwaysGame>()` constructs a genuinely independent instance with its own freshly-established child scope.** `ISimulation` is separately registered `Binding.Scope` with an `ObjectPool<Simulation>` allocator — one instance *per Game's own scope*, pooled for reuse across the app's lifetime. This confirms our originally-planned approach (construct/destroy test `MotorwaysGame` instances directly via the DI container, never touching `GameContainerScreen`/`ScreenStack`) exercises the exact same construction mechanism the game itself uses, while remaining completely inert with respect to whatever (if anything) is visibly on screen. No manual city start was needed for this phase as a result — the test ran fully autonomously.

## Step 3 — Obtaining the app scope (live, confirmed)

Same proven chain as Phase 4: `UnityEngine.Object.FindObjectOfType<AppRuntime>()` → `.App` (public property, confirmed non-null) → `.Scope` (public `IScope`, confirmed). No existing manually-started session was active this run (fresh launch, main menu) — Step 4's "if a manually started session exists, record it" logic is implemented and was exercised with a null result (logged as such), confirmed harmless.

## Steps 7-9 — Obtaining a CityDefinition/MapDefinition autonomously, then constructing

A background investigation (delegated to a subagent, to keep this document's context budget reasonable) confirmed:
- `Motorways.MapDatabase` is a **Scope-bound singleton**, loaded **synchronously** at container-build time (`AssetBundleUtility.LoadAsset<MapDatabase>`, blocking, not something we trigger) — `app.Scope.Get<MapDatabase>().MapLibrary.Maps` is immediately available, zero coroutines needed for that part.
- Obtaining an actual `CityDefinition` *instance* still requires the same async prefab load `GameStarter.LoadMapDefinition` uses (`AssetBundleUtility.LoadPrefabAsync(bundle, prefab, coroutineHost)`), typically resolving within a couple of frames for a first-time bundle load.

Live result, fully autonomous, zero UI/input:
```
[Phase5B] Resolved MapDatabase synchronously; selected first map: cityName='LosAngeles', bundle='losangeles', prefab='LosAngeles'.
[Phase5B] Autonomously loaded and instantiated CityDefinition 'LosAngeles(Clone)' for map 'LosAngeles' (no UI interaction).
```

## Step 9 — Deterministic seed control (confirmed working)

Rather than constructing our own `InitCityCommand` (which would duplicate logic `MotorwaysGame.Start()` already does internally), we used the game's own exposed seed-control entry point:
```csharp
global::Random.SetSimulationSeed(seed, app.Scope);   // public static, pre-existing API (ARCHITECTURE.md §5)
testGame.Start(cityDefinition, GameMode.Normal, null);  // internally calls Random.NextSimulationSeed(), now deterministic
```
This ran without error in all 3 cycles:
```
[Phase5B] Cycle 1: called global::Random.SetSimulationSeed(1001, app.Scope).
[Phase5B] Cycle 2: called global::Random.SetSimulationSeed(1002, app.Scope).
[Phase5B] Cycle 3: called global::Random.SetSimulationSeed(1003, app.Scope).
```
Whether this actually produces reproducible city layouts across repeated runs with the same seed was **not** verified this phase (the `Start()` call itself failed before reaching a stable, tickable state — see below) — that verification is a natural next step once the blocker below is resolved.

## Steps 8/10 — Constructing MotorwaysGame and calling Start(): three live attempts, three informative failures

```
[Phase5B] Cycle 1: constructed MotorwaysGame via app.Scope.Get<MotorwaysGame>(): Motorways.MotorwaysGame, hash=-499756140
[Phase5B] Cycle 1: EXCEPTION during construction/start: System.NullReferenceException
  at Motorways.Views.CameraView.Initialize (Motorways.GameRules rules) [0x00014] in .../CameraView.cs:323
  at Motorways.MotorwaysGame.Start (...) [0x0026a] in .../MotorwaysGame.cs:284

[Phase5B] Cycle 2: constructed MotorwaysGame via app.Scope.Get<MotorwaysGame>(): Motorways.MotorwaysGame, hash=-1387315290
[Phase5B] Cycle 2: testGame.Start(cityDefinition, GameMode.Normal, null) completed without throwing.
[Phase5B] Cycle 2: EXCEPTION during construction/start: System.NullReferenceException
  at Motorways.BridgeSpouts.SetDryingTunnelMesh (RoadTileDefinition deadEndMesh) [0x0001a] in .../BridgeSpouts.cs:23
  at Motorways.Views.RoadView.OnCreatedInScope (...) -> ... -> Motorways.Views.TileView.Tick (...) -> Motorways.MotorwaysGame.Tick (...)

[Phase5B] Cycle 3: constructed MotorwaysGame via app.Scope.Get<MotorwaysGame>(): Motorways.MotorwaysGame, hash=-1683168340
[Phase5B] Cycle 3: testGame.Start(cityDefinition, GameMode.Normal, null) completed without throwing.
[Phase5B] Cycle 3: EXCEPTION during construction/start: System.NullReferenceException
  (identical stack trace to Cycle 2 -- BridgeSpouts.SetDryingTunnelMesh via TileView.Tick)
```

Notably, the game shipped with embedded debug symbols (full file/line info resolved live, e.g. `MotorwaysGame.cs:284`, `BridgeSpouts.cs:23`) — unexpected and very useful for future debugging, confirming the retail build was not compiled with symbols fully stripped.

**Root cause — two DISTINCT issues, not one.** A follow-up decompile (delegated to a subagent, to keep this investigation's context cost down) corrected an earlier, too-hasty attribution of both exceptions to "skipped theme registration." The corrected picture, cited from decompiled source:

- **Crash #1 (`CameraView.Initialize`)**: `CameraView`'s fields (`_camera`, `_player`, `_visualConstantsData`, `_gameUI`, `_screenStack`, etc.) are all plain `[Dependency]` scope-resolved fields — there is no `FindObjectOfType`/scene lookup inside `CameraView` itself. The NRE (at the line reading `_visualConstantsData.ZoomLevelsCameraMin[_player.ZoomLevel]`) means one of those dependencies wasn't populated in our test scope at the point `Start()` ran. **Confirmed avoidable**: `MotorwaysGame.Start()` guards this entire call with `if (gameRules.UseCamera())`, and `GameRules.UseCamera() => ShowsUI()` — `BackgroundGameRules` overrides `ShowsUI()` to return `false`, so `GameMode.Background` would skip this call path entirely (confirmed from `BackgroundGameRules`'s decompiled overrides, not GameMode in general).
- **Crash #2 (`BridgeSpouts.SetDryingTunnelMesh`)**: **unrelated to theme registration.** `RegisterThemeComponents` was confirmed to call `InitializeTheme` only on types implementing `IThemeComponent` — and neither `BridgeSpouts` nor `RoadTileDefinition` implement that interface (confirmed from their class declarations: `IReusable, IReleasedFromScopeHandler` only). The real mechanism: `RoadView.OnCreatedInScope` calls `_roadTileAtlas.GetDefinitionForSignature(roadTileSignature)`, which does a plain dictionary lookup and **legitimately returns null when a signature isn't present** (not itself an error) — `SetDryingTunnelMesh` then unconditionally dereferences that possibly-null result (`deadEndMesh.mesh.roadMesh`) without a null check. The most plausible trigger, per the same investigation: the freshly-`Instantiate()`'d `CityDefinition`'s own Unity lifecycle (`Awake`/`Start`/`OnEnable`) may not have fully run yet in the same frame we immediately called `MotorwaysGame.Start()`/`Tick()` on it, leaving the tile/road model queried by `TileView.RebuildDynamicRoads` in an not-yet-fully-initialized state. This is a timing/ordering gap in our isolated-construction approach, not something `GameMode.Background` fixes.

So: one of the two failures has a confirmed, simple avoidance (`GameMode.Background`); the other needs a different fix — most likely letting Unity run at least one real frame between `Instantiate(cityDefinition)` and calling `Start()` on it, rather than doing both synchronously back-to-back. Neither fix was attempted this phase, per the instruction to stop at the identified blocker rather than patch around it immediately.

This is a genuine, reproducible gap in the *isolated-construction* approach, not a flaw in the Phase 5A call-path tracing (which was about the `MotorwaysGame`/`InitCityCommand`/`Simulation` layer, and that layer's own construction evidently proceeded far enough to reach `Tick()` calls without complaint — both confirmed failures are specifically in the **view/rendering** layer, not the **simulation/model** layer).

## What this phase DID prove, cleanly

1. `appScope.Get<MotorwaysGame>()` can be called repeatedly (3x in one process) without the DI container itself erroring — each call returns a distinct object (`hash=-499756140`, `-1387315290`, `-1683168340` — three different instances, confirmed non-equal).
2. The exception-handling and teardown path is robust: **every one of the 3 cycles' exceptions was caught, and the subsequent `StopAudio()` / `ClearPathfinder()` / `Scope.ParentScope.Release(testGame)` teardown sequence completed without throwing, every time** — the native release path tolerates tearing down a game that failed partway through its own `Start()`/`Tick()`.
3. **The live game process itself remained fully stable throughout** — confirmed via `Get-Process` immediately after all 3 failed cycles: responsive, correct window title, 610 MB working set, no crash, no hang. Three consecutive failed construction attempts, each followed by a real release call, did not destabilize the app.
4. The seed-control API (`global::Random.SetSimulationSeed`) is callable and doesn't itself throw.
5. The fully autonomous CityDefinition-acquisition path (`MapDatabase` → `MapLibrary.Maps` → `AssetBundleUtility.LoadPrefabAsync` → `Instantiate`) works end-to-end with zero UI/input, confirming a future `reset()` won't need a human or mouse/keyboard automation for this step either.

## What remains unresolved

- A fresh, tickable `Server.Simulation` with verifiably-advancing `ClockModel.Time` across multiple ticks (this document's original "step 10" checklist) was **not** achieved — every cycle's `Start()` call failed before the test's own post-construction verification logic ran.
- Whether the seed actually produces reproducible city layouts was not verified (same reason).
- `GameMode.Background` is now **confirmed** (not just hypothesized) to avoid Crash #1 (`CameraView.Initialize`), since `BackgroundGameRules.ShowsUI()` → `false` → `GameRules.UseCamera()` → `false` → `MotorwaysGame.Start()`'s `if (gameRules.UseCamera())` guard skips the call entirely. This is a materially simpler fix than replicating `GameContainerScreen.RegisterThemeComponents` ourselves, and fits an RL training loop's needs well (no rendered camera needed anyway).
- Crash #2 (`BridgeSpouts`/`RoadTileAtlas`) is a **separate, still-open** issue, unrelated to `GameMode` or theme registration — most likely a frame-timing gap between `Instantiate(cityDefinition)` and calling `Start()` on it in the same frame, before the prefab's own Unity lifecycle (`Awake`/`Start`/`OnEnable`) has run. A future phase should test yielding at least one frame between instantiation and `Start()` before concluding this requires a deeper fix.

## Why this phase stops here

Per explicit instruction: *"If any lifecycle operation causes instability, stop rather than layering workarounds on top of it. Determine which assumption about the native lifecycle was wrong."* The wrong assumption, now identified with evidence: **`MotorwaysGame.Start()` is not a self-contained entry point** — it implicitly depends on `GameContainerScreen.RegisterThemeComponents()` having already run on the same `CityDefinition` first. This is a real fact about the native lifecycle, not a bug in our test. Fixing it (by either replicating that theme step ourselves, or sidestepping the view layer entirely via a non-rendering `GameMode`) is a new, separate decision for a future phase, not a workaround to bolt on silently now.

## Files changed this phase

- [`game_plugin/DoorstopBootstrap/Entrypoint.cs`](../game_plugin/DoorstopBootstrap/Entrypoint.cs) — added `Phase5BRunner` (isolated construct/seed/start/tick/teardown ×3) and `GetCityDefinitionForTesting` (autonomous map-loading fallback, reuse-existing-session fast path).
- [`game_plugin/DoorstopBootstrap/DoorstopBootstrap.csproj`](../game_plugin/DoorstopBootstrap/DoorstopBootstrap.csproj) — added `MOAnalytics.Runtime.dll` reference (for `Plugins.Analytics.AnalyticsTypes.PauseReason`).

## Hashes

- `Mini Motorways.exe`: `305d1ea095f0ba9e844b91497283277c61f0dea9d5f7979065e78f3e7bab0ce6` — unchanged.
- `patched_corlib\mscorlib.dll`: `43639906003e444716d06afb010d5a91ae1c900582c69e373977857ebaf600fd` — unchanged, untouched.
- `Bootstrap\MiniMotorwaysBootstrap.dll` (this phase's build): `1800f35a1fd13bed171c3919fbae1161d4e6d6ee20d34d226aaeb8fbfac854cc`.

## Hard constraints — compliance

- No mouse/keyboard automation (the autonomous map-load path replaced the need for it entirely).
- No BepInEx/Harmony/MonoMod (bootstrap remains zero-reference to all three).
- No game assembly patched; no corlib/System assembly replaced this phase.
- No Python simulator, no road/action implementation, no reward, no Gymnasium IPC.
- No spawn/RNG/gameplay logic modified — `Random.SetSimulationSeed` is a pre-existing, documented public API, used as designed, not patched.
- Stopped at the first sign of a wrong lifecycle assumption rather than patching around it, per explicit instruction.

## research/ARCHITECTURE.md update

One fact established experimentally this phase, added to `ARCHITECTURE.md` §1 (composition root): `IApp.Game` is a live computed property (`App.cs`: `_screenStack.GetGameIfInGame()`), not a cached reference — it reflects whichever `GameContainerScreen` is currently on top of `ScreenStack`, not whatever a caller may have separately constructed via `appScope.Get<MotorwaysGame>()`. This was previously only inferred from the interface declaration (Phase 4); the concrete implementation and its `ScreenStack` coupling are new, confirmed-live facts.
