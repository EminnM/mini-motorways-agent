# PHASE_4 — Locating the Live Simulation Object

Generated 2026-09-28. Objective: find and obtain a reference to the actual live `Server.ISimulation` instance driving the real game, via the proven Phase 3D Doorstop bootstrap. **Result: the access chain is fully identified and verified live — `AppRuntime.App` resolves successfully inside the running game — but `IApp.Game` (and therefore `Game.Simulation`) is confirmed null while sitting at the main menu, because no game/city session is active.** No simulation instance exists yet to obtain a reference to. This is a genuine, verified fact about the game's live state, not a flaw in the access chain.

## Step 1 — Revisit static recon: composition root and DI classes (read-only, decompiled in full)

Newly decompiled this phase (not previously read to this depth): `AppContainer.cs`, `Motorways.MotorwaysAppContainer` (partially), `IApp.cs`, `Factory.IScope.cs`, `Game.cs` (base class of `MotorwaysGame`). Combined with Phase 2's existing `AppRuntime.cs` and `Motorways.MotorwaysGame` decompiles, the full composition-root chain is now mapped end-to-end:

```
AppRuntime : MonoBehaviour                         (Unity scene object, Phase 2 decompile)
    private AppContainer _container;                (instantiated: new MotorwaysAppContainer())
    public IApp App => _container?.App;              PUBLIC property, CONFIRMED

AppContainer (abstract base of MotorwaysAppContainer)
    public IApp App { get; private set; }             set via: App = AppScope.Get<IApp>() in CreateApp()
    public IScope AppScope { get; private set; }

IApp (interface)
    IScope Scope { get; }
    Game Game { get; }                                 PUBLIC property, CONFIRMED
    ...

Game (base class; runtime type is Motorways.MotorwaysGame : Game)
    [Dependency] protected ISimulation _simulation;    DI-injected field (Factory/Scope system, same pattern as every class documented in ARCHITECTURE.md)
    public ISimulation Simulation => _simulation;      PUBLIC property, CONFIRMED
    [Dependency] public IScope Scope { get; private set; }  PUBLIC property -- a second route to City/CityModel/ClockModel etc. via Scope.Get<T>()
```

**Full chain**: `AppRuntime.App.Game.Simulation` — three consecutive public property reads, zero private-field reflection, zero patching of any kind.

## Step 2 — Where is the simulation instance actually created?

`Game._simulation` is marked `[Dependency]`, the exact same dependency-injection attribute documented throughout `ARCHITECTURE.md` for every simulation-facing class (`BuildingSpawningProcess`, `City`, `BuildingPlacer`, etc.). This confirms it is populated by the `Factory`/`Scope` container at object-assembly time (`scope.Assemble(...)`), not hand-constructed in a traceable constructor call we'd need to hunt down further — and because it is the *same* DI scope that injects `ISimulation` into every other simulation-facing class, `Game.Simulation` is guaranteed to be the literal same object instance those other classes already use internally. This satisfies the brief's "do not assume a singleton exists — verify from decompiled code": the mechanism is DI-scope-based sharing, not a named singleton pattern, and this was confirmed by reading the actual `[Dependency]` attribute usage, not inferred from behavior.

`AppContainer.CreateApp()`(decompiled in full) shows exactly how `IApp` itself is obtained: `App = AppScope.Get<IApp>();` — a scope resolution, not a `new App()` call directly in view. The Game-level scope (a presumably a *child* scope created per city session, based on the `IScope.AddChildScope` method seen in `IScope`'s interface and the "Scope" ubiquity throughout `ARCHITECTURE.md`) is where `Game` and its `[Dependency]`-injected `ISimulation` actually get assembled — this per-session child-scope creation was not traced to its exact call site this phase (out of scope for what was needed to answer Step 1-4; the important fact — that `Game.Simulation` is the authoritative live reference once it exists — is established either way).

## Step 3 — Can the bootstrap load/reference `App.dll` types? Yes, confirmed live.

The Phase 3D bootstrap ([`game_plugin/DoorstopBootstrap/`](../game_plugin/DoorstopBootstrap/)) was extended with references to `App.dll`, `UnityEngine.CoreModule.dll`, and `FixedMath.dll` (needed for `Fix64`, the return type of `ISimulation.Timestep`) — still zero references to BepInEx, Harmony, or MonoMod. It compiles and, live in the running game process, successfully resolves `AppRuntime`, `IApp`, and (when non-null) would resolve `Game` and `Server.ISimulation` — all by ordinary typed C# property access. `Assembly.Load("App")` was already confirmed working in Phase 3D; this phase went further by actually walking the live object graph, not just checking assembly identity.

## Step 4 — Live verification: the chain works, but no session is active (verified, not assumed)

Two live tests were run (each following the established one-change-at-a-time, health-checked protocol):

**Test 1** — subscribed to `UnityEngine.SceneManagement.SceneManager.sceneLoaded` (a standard public .NET event; subscribing to it is not a patch of any existing method) to defer the probe until Unity's engine and scene system are ready, since `Doorstop.Entrypoint.Start()` runs far too early to call Unity APIs directly. Result:
```
SceneManager.sceneLoaded fired: mode=2
AppRuntime not found in this scene yet.
SceneManager.sceneLoaded fired: mode=Additive
AppRuntime found: AppRuntime
AppRuntime.App is null (container not started yet).
```
`AppRuntime` was found on the second scene load, but `.App` was still null at that exact instant — a timing artifact (Unity's own `AppRuntime.Start()`, which calls `_container.CreateApp()`, had not yet run in that same frame).

**Test 2** — added a plain `MonoBehaviour` (`SimulationPoller`), attached to a new `GameObject` via the standard public `AddComponent<T>()` API (again, not a patch — this is exactly how every Unity script normally gets attached to a scene), polling once per `Update()` frame (throttled logging every ~5s, not per-frame) for up to 1800 frames (~30s). Result:
```
Created polling MonoBehaviour ... to wait for App/Game/Simulation to become available.
Still waiting (frame 300):  App=True, Game=False, Simulation=False
Still waiting (frame 600):  App=True, Game=False, Simulation=False
Still waiting (frame 900):  App=True, Game=False, Simulation=False
Still waiting (frame 1200): App=True, Game=False, Simulation=False
Still waiting (frame 1500): App=True, Game=False, Simulation=False
Still waiting (frame 1800): App=True, Game=False, Simulation=False
Gave up waiting for Game.Simulation after 1800 frames. App=True, Game=False.
```
`App` resolved successfully and stayed non-null for the full 30-second window — confirming the container/App layer is live and stable. `Game` stayed null the entire time. **Conclusion, directly observed, not inferred**: this game's main menu does **not** run a live background city/simulation (an earlier working hypothesis, based on a `BuildingPlacer` code comment mentioning "menu city," was considered but is now contradicted by this direct live measurement — corrected rather than left standing). `IApp.Game` is genuinely only populated once a player starts or resumes an actual city session.

## Current status: chain proven correct, blocked only on "no active session," not on any technical obstacle

There is no more static analysis or code-path investigation that would change this outcome — the fact that `Game` is null right now is a true statement about the live game's current state (sitting at the main menu), not a gap in our approach. To actually obtain a non-null `ISimulation` reference, a city session needs to be active when the probe runs. That requires either:

1. **A human manually starts or resumes a city** while this instrumentation is installed and running (the poller already waits up to 30s; that window can be lengthened, or the probe re-armed on every future scene load instead of giving up).
2. **This agent navigates the game's own main menu via mouse input** (computer-use / standard OS-level clicks on the already-visible game window — the same input surface a human player uses, not a code-level action injection into the simulation) to start a new city or resume an existing save, then re-run the read-only probe.

Both are consistent with the phase's "read-only inspection" and "do not mutate simulation state" constraints — starting a city session is normal use of the software (equivalent to a human clicking play), not a mutation performed *on* the simulation object once it exists, and no `ISimulation` method that changes state (`Step()`, `ScheduleCommand()`, etc.) has been or would be called. Neither was attempted this phase without checking first, since it's a step beyond what "investigate the access chain" strictly asked for, and because it's the kind of action (navigating UI, however innocuous) worth confirming rather than assuming.

## Hard constraints — compliance

- **No original game assembly modified**: confirmed, `Mini Motorways.exe` SHA-256 unchanged (`305d1ea095f0ba9e844b91497283277c61f0dea9d5f7979065e78f3e7bab0ce6`) throughout.
- **No mscorlib/System.Core replacement**: none attempted or needed this phase; the one new BCL gap hit in Phase 3D-style testing (`Scene.name`/`Scene.buildIndex` both stripped) was worked around in our own code (dropped from the log line) rather than patching the assembly.
- **BepInEx state preserved**: `doorstop_config.ini`'s `target_assembly` was repointed to our bootstrap for testing and reverted back to `BepInEx\core\BepInEx.Preloader.dll` afterward — the resting state matches Phase 3B/3D's baseline exactly.
- **No repeated launches after a crash**: no crash occurred this phase; both live tests completed cleanly with the process healthy throughout (599-602 MB, responsive) and were stopped by normal `taskkill` once their data was collected.
- **No simulation mutation**: only property getters were read (`App`, `Game`, `Simulation`, `Timestep`, `IsPaused`, `HasAnyScheduledCommands`, `Scope`) — no method with a side effect was called.
- **Rollback**: trivial and already executed — `doorstop_config.ini` reverted (one line), no other file touched. `Bootstrap\MiniMotorwaysBootstrap.dll` updated in place with the Phase 4 code (harmless, unreferenced while `target_assembly` points at BepInEx).

## Files changed

- [`game_plugin/DoorstopBootstrap/DoorstopBootstrap.csproj`](../game_plugin/DoorstopBootstrap/DoorstopBootstrap.csproj) — added `UnityEngine.CoreModule.dll`, `App.dll`, `FixedMath.dll` references.
- [`game_plugin/DoorstopBootstrap/Entrypoint.cs`](../game_plugin/DoorstopBootstrap/Entrypoint.cs) — added the `SceneManager.sceneLoaded`-based deferred probe and the `SimulationPoller` `MonoBehaviour`.
- New decompiled reference files under `research/static_recon/phase4/`: `AppContainer.cs`, `Motorways_MotorwaysAppContainer.cs`, `IApp.cs`, `Factory_IScope.cs`, `Game.cs`.

## Hashes

- `Mini Motorways.exe`: `305d1ea095f0ba9e844b91497283277c61f0dea9d5f7979065e78f3e7bab0ce6` (unchanged)
- `Bootstrap\MiniMotorwaysBootstrap.dll` (final Phase 4 build): `fc03963b3361d851eba043fb1c489192c8f75a372fa3f33f78394caf5245374e`

## Phase 4B — live result with an active city session (SUCCESS)

Per direction, UI input was **not** automated. The user manually started/resumed a city in the Mini Motorways window while the Phase 3D bootstrap (repointed at via `doorstop_config.ini`'s `target_assembly`, exactly as in every prior test) was running. The `SimulationPoller` `MonoBehaviour` was extended (still zero Harmony/MonoMod/BepInEx references, still only public-property reads) to, once `Game.Simulation` first becomes non-null, switch into a sampling mode: read `Motorways.Models.ClockModel.Time` (resolved via `Game.Scope.Get<ClockModel>()`, the same public DI-scope accessor documented in `ARCHITECTURE.md`) once every 2 real seconds, 12 times, logging each reading plus a reference-identity check (`ReferenceEquals` against the first-obtained `ISimulation` reference) and `IsPaused`.

Live log (`logs/instrumentation.log`, this run):
```
[MiniMotorwaysAI] simulation reference obtained
Real seconds waited for an active city session: 215.2
IApp resolved: App
Game resolved: Motorways.MotorwaysGame
ISimulation concrete type: Server.Simulation
ISimulation runtime hash (identity check across future samples): 1238087616
Simulation.Timestep: 0.1000000001
Simulation.IsPaused: False
Sample #0 @ realtime+215.2s: ClockModel.Time = 0
Sample #1 @ realtime+217.2s: sameInstanceAsFirst=True, IsPaused=False, ClockModel.Time=0
Sample #2 @ realtime+219.3s: sameInstanceAsFirst=True, IsPaused=False, ClockModel.Time=1.1000000001
Sample #3 @ realtime+221.3s: sameInstanceAsFirst=True, IsPaused=True,  ClockModel.Time=1.8000000017
Sample #4 @ realtime+223.3s: sameInstanceAsFirst=True, IsPaused=True,  ClockModel.Time=1.8000000017
Sample #5 @ realtime+225.3s: sameInstanceAsFirst=True, IsPaused=True,  ClockModel.Time=1.8000000017
Sample #6 @ realtime+227.3s: sameInstanceAsFirst=True, IsPaused=True,  ClockModel.Time=1.8000000017
Sample #7 @ realtime+229.3s: sameInstanceAsFirst=True, IsPaused=True,  ClockModel.Time=1.8000000017
Sample #8 @ realtime+231.3s: sameInstanceAsFirst=True, IsPaused=True,  ClockModel.Time=1.8000000017
Sample #9 @ realtime+233.3s: sameInstanceAsFirst=True, IsPaused=True,  ClockModel.Time=1.8000000017
Sample #10 @ realtime+235.3s: sameInstanceAsFirst=True, IsPaused=False, ClockModel.Time=2.7000000025
Sample #11 @ realtime+237.4s: sameInstanceAsFirst=True, IsPaused=False, ClockModel.Time=6.7000000062
Sample #12 @ realtime+239.4s: sameInstanceAsFirst=True, IsPaused=False, ClockModel.Time=10.70000001
[MiniMotorwaysAI] sampling complete
```

**Success criterion, checked point by point:**
1. **A real city session is active** — confirmed: `Game` transitioned from null (main menu) to a real `Motorways.MotorwaysGame` instance once the user started/resumed a city.
2. **`AppRuntime.App.Game.Simulation` is non-null** — confirmed, concrete runtime type `Server.Simulation`.
3. **It is the actual live simulation instance, not a copy** — confirmed by `ReferenceEquals` returning `True` on all 12/12 samples spanning ~24 seconds; if a new instance were created or returned each poll, this would read `False`.
4. **A read-only value changes as the real game advances** — confirmed unambiguously: `ClockModel.Time` climbs 0 → 1.1 → 1.8 → (frozen for six samples) → 2.7 → 6.7 → 10.7, and the "frozen" stretch exactly coincides with `IsPaused=True` — i.e. the value doesn't just increase, it tracks the game's actual pause state, which is strong independent confirmation this is the real simulation clock and not a static counter of our own.

No gameplay-mutating method was ever called (`Step()`, `ScheduleCommand()`, `AddProcess()`, etc.) — only the pre-existing public getters `Simulation`, `Scope`, `IsPaused`, `Timestep`, and `Get<ClockModel>()`, plus `.Time` on the returned model.

**Phase 4 is now fully closed.** Rollback executed: `doorstop_config.ini` reverted to `BepInEx\core\BepInEx.Preloader.dll`, test process terminated normally, `Mini Motorways.exe` hash re-confirmed unchanged. Proceeding to Phase 5A.
