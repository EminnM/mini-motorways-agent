# PHASE_5A — Tracing the City-Start Call Path (Investigation Only)

Generated 2026-10-06. Objective: trace the exact internal call sequence Mini Motorways itself uses to go from "menu/new-game selection" to a live, populated `AppRuntime.App.Game.Simulation`, so this can eventually become an automated, deterministic RL `reset()` — invoking the game's own native machinery, not a hand-built simulator. **Investigation only — nothing in this phase was executed against the live game. All findings are from decompiled source**, cross-checked against the live object graph already proven reachable in Phase 4/4B.

## Correction from Phase 4

Phase 4B's live test revealed `Game.Simulation`'s concrete runtime type is **`Server.Simulation`**, not `Motorways.Simulation` as the original brief's wording assumed. Noted here rather than silently corrected, per the engagement's "verify, don't assume" discipline — the brief's phrasing was a reasonable guess given the `Motorways.*` naming convention used almost everywhere else in the codebase, but the actual simulation *engine* class lives in the generic `Server` namespace (alongside `ISimulation`, `IProcess`, `Command` — the engine-agnostic core documented in `ARCHITECTURE.md` §1), separate from the `Motorways.*` game-specific layer built on top of it.

## The full call chain, decompiled end-to-end

```
Player clicks "New Game" / "Continue" on a menu screen (not traced further -- UI-level,
out of scope; this phase picks up from GameStarter, which every menu path funnels into)
        │
        ▼
GameStarter.StartFromMapDefinition(mapDefinition, mode, ...)      [fresh game]
  or    .StartFromSavedGame(mapLibrary, save, ...)                [resume/load]
        │  kicks off an ASYNC asset-bundle load of the map prefab
        │  (AssetBundleUtility.LoadPrefabAsync) -- NOT synchronous,
        │  takes at least one frame; GameStarter.CanStart becomes
        │  true once _cityDefinitionLoader.HasValue
        ▼
GameStarter.Start(screenStack, appScope)
        │  Instantiates the CityDefinition prefab GameObject
        │  (UnityEngine.Object.Instantiate), then:
        │
        ├─ fresh game: screenStack.PushScreen(InGame, screen => screen.PrepareForMap(cityDefinition, mapDefinition, mode, mapChallenge))
        └─ saved game: save.DeserializeGame(cityDefinition) -> validates -> screenStack.PushScreen(_startScreen, screen => screen.PrepareForNewGame(cityDefinition, mapDefinition, motorwaysGame, mapChallenge, startPaused))
        │
        ▼
GameContainerScreen.PrepareForMap(...) / .PrepareForNewGame(...)
        │  Just stores fields (_newCity, _newMapDefinition, _gameMode, _newMapChallenge)
        │  and sets _startGameOnTransition = true. Does NOT construct the game yet --
        │  construction is deferred until the screen's transition-in animation fires.
        ▼
GameContainerScreen.TransitionIn(outScreen)        <-- THE ACTUAL CONSTRUCTION POINT
        │  if (_startGameOnTransition) {
        │      _game = _appScope.Get<MotorwaysGame>();     // DI-constructs a fresh MotorwaysGame + its whole child-scope object graph
        │      _game.SetMapDefinition(_newMapDefinition);
        │      _game.Start(_newCity, _gameMode, _newMapChallenge);   // <-- see below
        │      _game.SetPaused(false, AnalyticsTypes.PauseReason.TransitionIn);
        │      _game.Tick(0f);                              // x2, zero-delta warm-up ticks
        │      _game.Tick(0f);
        │      _gameUIScreen = _game.Scope.Get<GameUIScreen>();
        │      _game.OnGameStarted();
        │  }
        ▼
MotorwaysGame.Start(CityDefinition cityDefinition, GameMode mode, MapChallenge mapChallenge, bool replaceExistingRules = false)
        │  _simulation already exists (constructed empty as part of the DI scope
        │  graph when appScope.Get<MotorwaysGame>() ran) -- this method POPULATES it.
        │  model = _simulation.GetModel<CityModel>();  // null for a genuinely fresh Game
        │  gameRules = CreateRulesForMode(Scope, mode);   // GameMode enum -> GameRules subclass, via Scope.Get<T>()
        │  if (!TryLoadSimulationJournal(gameRules)) {    // false outside Editor/SoakTest builds -- confirmed, this is the retail-build path
        │      if (mapChallenge == null)
        │          command = InitCityCommand.CreateNormalCity(Scope, cityName, cityDefinition, mode, gameRules, Random.NextSimulationSeed());
        │      else
        │          command = InitCityCommand.CreateChallengeCity(Scope, cityName, cityDefinition, mode, gameRules, mapChallenge);
        │      _simulation.ScheduleCommand(command);   // <-- NOT executed yet -- scheduled for the next simulation Step()
        │  }
        ▼
InitCityCommand.Execute(ISimulation simulation)    <-- runs when the simulation actually steps
        (triggered by the two _game.Tick(0f) warm-up calls back in TransitionIn, which
         drive the simulation forward through its scheduled-command queue)
        │  cityModel = simulation.Scope.Get<CityModel>();
        │  cityModel.pseudorandomGenerator.Seed = _seed;      // <-- THE determinism-critical line
        │  simulation.AddModel(cityModel);
        │  _city.Initialize(_cityDefinition, _rules);
        │  simulation.AddProcess(...)  x24   // every IProcess from ARCHITECTURE.md §3, in this exact order
        │  simulation.AddModel(CityPlanModel / DemandModel / TilemapModel / ClockModel / ScoreModel / ...)
        │  cityModel.startOffset = cityDefinition.GenerateCityStartOffset(cityModel.pseudorandomGenerator);  // seeded
        │  _city.GenerateCityLayout();        // seeded (Phase 2: uses cityModel.pseudorandomGenerator)
        │  _city.SetupTrainNetwork(simulation);
        │  _city.SetupBoatPathNetwork(simulation);
        │  _city.PopulateTrees(simulation);
```

## Determinism: the seed path is now fully traced, not assumed

`InitCityCommand.CreateNormalCity(scope, cityName, cityDefinition, mode, rules, seed)` takes the seed as a plain `uint` parameter (widened to the command's internal `ulong _seed` field). The *only* place this seed is currently populated, in the normal game-start path, is `Random.NextSimulationSeed()` — the top-level `Random` helper documented as non-deterministic in `ARCHITECTURE.md` §5 (it wraps `System.Random(Environment.TickCount)`).

This means: **the game's own code already has a clean, explicit seed parameter sitting right at the API boundary we need.** A deterministic `reset(seed=N)` does not require touching `PseudorandomGenerator`, `CityModel`, or any simulation internals directly — it only requires constructing our own `InitCityCommand` with our own chosen `uint` seed instead of letting `MotorwaysGame.Start()` call `Random.NextSimulationSeed()`. Confirmed via `InitCityCommand.CreateNormalCity`'s public static factory signature and `InitCityCommand.Execute`'s direct use of `_seed` to set `cityModel.pseudorandomGenerator.Seed`.

## Teardown: the exact reverse path, also traced

```
GameContainerScreen.ReleaseGame()
    _game.StopAudio();
    _game.ClearPathfinder();
    _game.Scope.ParentScope.Release(_game);   // returns Game + its whole child-scope graph to the DI allocator/pool
    _game = null;

GameContainerScreen.CleanupPreviousGame()
    if (_newCity != null) { Object.Destroy(_newCity.gameObject); _newCity = null; }
    if (_game != null) ReleaseGame();
```
`GameContainerScreen.PrepareForMap`/`PrepareForNewGame` both already call `CleanupPreviousGame()` defensively if a previous game wasn't properly torn down — confirming the engine itself expects and handles "start a new session while one might already exist," which is exactly what a repeated `reset()` needs to do safely.

## Map/city selection — the other required input

`GameStarter.StartFromMapDefinition` requires a `MapDefinition` (not literally the same as the final `CityDefinition` — `MapDefinition` is the lightweight catalog entry, e.g. from `MapLibrary.GetMapByName(name)`; `CityDefinition` is the actual instantiated prefab asset it points to, loaded async via `AssetBundleUtility.LoadPrefabAsync(mapDefinition.mapAssetBundle, mapDefinition.mapPrefabName, coroutineHost)`). This confirms map selection is **not synchronous** — any `reset()` implementation needs to either wait a frame (or more) for this async load the first time a given map is used, or keep a `CityDefinition` prefab instance around and reuse/re-instantiate it directly, skipping `GameStarter` for repeat resets on the same map.

`GameMode` (`Normal`, `Endless`, `Expert`, `Creative`, `Tutorial`, `Background`, `Movie`, `Cinematic`) maps 1:1 to a `GameRules` subclass via `MotorwaysGame.CreateRulesForMode`, itself just a `Scope.Get<T>()` resolution — confirmed exhaustively from the decompiled switch statement, not inferred.

## What remains unknown / out of scope for this phase

- The exact UI-level call from a menu screen's button handler into `GameStarter` was not traced (explicitly out of scope — Phase 5A picks up at `GameStarter`, which is itself engine/UI-agnostic and the right level to call directly, bypassing menu UI entirely for an automated reset).
- Whether `appScope.Get<MotorwaysGame>()` (the construction call) can be invoked a *second* time in the same process after a `ReleaseGame()` teardown without issues has not been tested live — this is the first thing Phase 5B (if authorized) should verify empirically, exactly like every other phase in this engagement: smallest possible live test, one change at a time, read-only verification before any state-changing attempt.
- `CityDefinition.GenerateCityStartOffset`, `City.GenerateCityLayout`, `SetupTrainNetwork`/`SetupBoatPathNetwork`/`PopulateTrees` were not individually re-decompiled this phase (several were already covered in Phase 2's `ARCHITECTURE.md`); their seeded-RNG usage is already documented there and is consistent with what this phase found in `InitCityCommand.Execute`.

## Proposed `reset()` architecture for the future Gymnasium wrapper

This is a **design proposal based on the traced call path above** — nothing here has been executed; it is the planned shape of Phase 5B/6 work, written now while the call path is fresh, per the instruction to end Phase 5A with this proposal.

```
reset(seed: int, map_name: str = "<default>", mode: GameMode = Normal) -> initial_observation:

    1. [one-time per process, or per map] Ensure a CityDefinition instance for `map_name`
       is already instantiated (GameStarter.LoadMapDefinition's async bundle load,
       awaited once; cached thereafter -- avoids repeating the async load every reset).

    2. If a previous session's Game exists (our bootstrap's own tracked reference,
       not necessarily GameContainerScreen's, since we may run headless of normal
       screen flow):
           game.StopAudio()
           game.ClearPathfinder()
           game.Scope.ParentScope.Release(game)
       -- mirrors GameContainerScreen.ReleaseGame() exactly.

    3. game = appScope.Get<MotorwaysGame>()        // fresh Game + simulation object graph
       game.SetMapDefinition(mapDefinition)

    4. Construct our OWN InitCityCommand instead of letting MotorwaysGame.Start() pick
       a random seed:
           command = scope.Get<InitCityCommand>()
           command.Initialize(cityName, cityDefinitionInstance, mode, gameRules, seed)
           game.Simulation.ScheduleCommand(command)
       -- this is the one deliberate deviation from the stock call path: replacing
       `InitCityCommand.CreateNormalCity(..., Random.NextSimulationSeed())`
       with our own chosen `seed`, calling the exact same InitCityCommand machinery
       otherwise untouched.

       Open question for Phase 5B to verify live: can we call `MotorwaysGame.Start(...)`
       as-is and simply pre-empt/override the seed some other way (e.g. temporarily
       stubbing what `Random.NextSimulationSeed()` returns), instead of duplicating
       the command-construction step ourselves? Either achieves the same determinism
       goal; which is less invasive needs a live test, not a guess.

    5. game.SetPaused(false, ...)
       game.Tick(0f)   // x2, exactly as GameContainerScreen does -- drains the
       game.Tick(0f)   // scheduled InitCityCommand through the simulation's command
                        // queue and lets every process/model populate itself.
       game.OnGameStarted()

    6. Track `game` (and `game.Simulation`) as the active session for subsequent
       step()/observation calls. No GameContainerScreen, no ScreenStack, no UI
       required for any of the above -- everything in steps 3-5 is pure App.dll/
       Server.* API, confirmed reachable via the same AppRuntime.App.Game chain
       Phase 4 already proved live.

    7. Return the first observation by reading (not mutating) simulation state --
       reusing whatever read-only accessors Phase 4's probe already established
       (Scope.Get<ClockModel>(), simulation.GetModels<T>(), etc.) plus whatever
       Phase 6's state-extraction work defines as the actual observation schema.
```

Properties this design preserves, by construction (because it reuses the game's own machinery rather than re-implementing it):
- **Determinism**: the only randomness source for city generation (`PseudorandomGenerator`, seeded once via `InitCityCommand`) is explicitly controlled; everything downstream (`GenerateCityLayout`, spawn weighting, etc.) is seeded-RNG-driven per `ARCHITECTURE.md` §4-§5, so a fixed seed should reproduce identical city layouts — to be empirically verified in a future phase (`reset(seed=123)` twice, diff the resulting state), not assumed.
- **Fidelity**: every spawn rule, road-adjacency effect, traffic behavior, and all the "unintuitive interaction" surface area documented in `ARCHITECTURE.md` remains exactly the shipped game's own code — nothing here constructs a parallel simulation.
- **No gameplay mutation performed in this phase**: this section is a design proposal only; no `InitCityCommand`, `ScheduleCommand`, or `Get<MotorwaysGame>()` call has been executed by us against the live game. The only live interaction in this entire investigation remains Phase 4B's read-only probe.

## Files changed this phase

None — Phase 5A is pure static analysis. New decompiled reference files under `research/static_recon/phase5a/`: `Motorways_GameStarter.cs`, `Motorways_Commands_InitCityCommand.cs`, `Motorways_CityBootstrap.cs`, `GameContainerScreen.cs`, `MotorwaysGame_full.cs`.

## Hard constraints — compliance

- No UI automation, no mouse/keyboard coordinates used or proposed for the actual reset mechanism (the proposal explicitly bypasses `ScreenStack`/menu UI entirely, calling `GameStarter`/`MotorwaysGame`/`InitCityCommand` directly instead).
- No separate simulator built or proposed — the entire design is "invoke the game's own object graph," confirmed via decompiled source at every step.
- No gameplay mutated: zero live calls this phase beyond what Phase 4B already did (read-only).
- `Mini Motorways.exe` and all previously-recorded hashes remain unchanged (no live interaction occurred this phase to risk them).
