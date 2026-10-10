# Phase 9A — Visible Environment (presentation integration)

**Status: ACHIEVED.** RandomPolicy visibly controlled the actual native Mini Motorways game
through the stock `GameContainerScreen` frontend, while Phase 8's environment core
(observation, legal-action masking, controlled stepping, persistence isolation) remained fully
authoritative and deterministic throughout. This required one architectural change from the
first two attempts documented below: letting the stock frontend boot (`LoadingScreen` →
`StartupScreen` → `MainMenuScreen`) complete entirely on its own, with zero intervention, before
doing anything of our own.

## 1. Summary of all three attempts

| Attempt | Change | Result |
|---|---|---|
| 1 | Push `GameContainerScreen` almost immediately after boot | `NullReferenceException` in `StartupScreen.StopMenuGameAudio()` — `_menuGame` not yet constructed |
| 2 | Reflectively wait up to 30s for `StartupScreen._menuGame != null` before pushing | Timed out — our own early atlas/test-player intervention stalled `StartupScreen`'s own stage machine before it could reach `LoadMenuCity` |
| 3 | Wait for the ENTIRE stock boot (`LoadingScreen → StartupScreen → MainMenuScreen`) to finish naturally, with no intervention of any kind, before creating a test profile, verifying atlases, or attaching anything | **Succeeded** |

Attempt 3 required one further correction mid-attempt: the first wait-signal design
(`GetActiveScreen<StartupScreen>() == null && GetActiveScreen<LoadingScreen>() == null`) itself
timed out at 60s even though the real boot had already finished — because `ScreenStack`'s
`GetActiveScreen<T>()` searches a pooled `_activeScreens` list that screen instances never
leave once created. The corrected signal uses `ScreenStack.GetTopVisibleScreen()` /
`HasVisibleScreens()`, which read a *different* internal list (`_visibleScreens`) that
`StartScreenTransitions` removes the outgoing screen from immediately on a non-additive
`PushScreen` — this is the signal that actually reflects what the player would see. See §3.

## 2. Architectural change: wait for a genuinely stock boot

Per the user's explicit instruction, our bootstrap no longer does *anything* — no manual atlas
initialization, no test-player creation, no frontend push — before the stock boot reaches a
stable main menu. `InitializeProcessContext` now:

1. Subscribes to `ActivePlayer.PlayerChanged` immediately (diagnostic only, no action taken).
2. Calls `WaitForStockFrontendBootComplete`, which waits — with no sleeps, no intervention, just
   `yield return null` each frame — until the native boot has genuinely finished.
3. Only then confirms the stock `LoadingScreen` one-shot player-activation signal (`PlayerChanged`
   stack trace containing `"LoadingScreen"`) was observed during that wait — a redundant
   safety check layered on top of (2), not a substitute for it.
4. Only then creates and activates the disposable test player.
5. Only then verifies (read-only probes, no `Initialize()`/`Import()` calls of our own) that
   `RoadTileAtlas`/`RailTileAtlas`/`BoatPathTileAtlas` are already populated — by the stock
   `StartupScreen.LoadAtlas`/`WaitForAtlasToLoad` stage, which has already run to completion by
   this point.

This fixes attempt 2's actual root cause: that attempt still created the test player and
initialized atlases ourselves almost immediately after the `LoadingScreen` signal (frame 7,
well before `StartupScreen`'s own stage machine could progress), which apparently interfered
with or raced against `StartupScreen`'s own progress toward `LoadMenuCity`. Attempt 3 does none
of that — it is a passive observer until the stock UI is already sitting at a stable main menu.

## 3. Step 1 — the stock-boot-completion signal (and a signal bug found and fixed)

**Final signal** (`WaitForStockFrontendBootComplete`):

```csharp
IScreen topVisible = screenStack.HasVisibleScreens() ? screenStack.GetTopVisibleScreen() : null;
MainMenuScreen mainMenu = topVisible as MainMenuScreen;
bool stockBootComplete = mainMenu != null && !mainMenu.IsTransitioningIn();
```

`GetTopVisibleScreen()`/`HasVisibleScreens()` are public `ScreenStack` methods reading its
private `_visibleScreens` list. Confirmed by decompiling `ScreenStack.cs`:
`StartScreenTransitions` removes the outgoing screen from `_visibleScreens` synchronously,
inside the same non-additive `PushScreen` call that pushes the incoming screen — not gated on
any transition animation completing. Since `StartupScreen.TransitionToMainMenu()` (the only
stock code path that ever pushes `MainMenuScreen`) is itself only reached after
`StartupScreen`'s `MainMenuSequence` stage array has already executed `LoadAtlas →
WaitForAtlasToLoad → LoadMenuCity → Hold` in order, `MainMenuScreen` becoming the topmost
visible screen is sufficient, direct proof that the entire stock boot — including atlas
loading and `_menuGame` construction — has completed. No reflection into private fields is
needed for this signal.

**A first version of this signal was wrong and is documented for the record**: it used
`ScreenStack.GetActiveScreen<T>()` and required `GetActiveScreen<StartupScreen>() == null &&
GetActiveScreen<LoadingScreen>() == null`. Live evidence showed `MainMenuScreen` was already
present and fully transitioned in (`IsTransitioningIn=False`) at 45 seconds, yet the wait still
timed out at 60 seconds reporting `StartupScreen=still active, LoadingScreen=still active`.
Decompiling `ScreenStack.GetActiveScreen(Type)` showed it searches `_activeScreens`, a list of
every *pooled* screen instance ever enabled — which screens do not leave once added. This list
is not a reliable "is this the current screen" signal; `_visibleScreens` (exposed via
`GetTopVisibleScreen`) is. This was corrected before the successful attempt.

## 4. Step 2 — attachment semantics (confirmed, unchanged from the original static trace)

- **`GameContainerScreen.PrepareForMap(...)`** is the only safe public entry point for
  attaching a frontend when no game yet exists: it constructs and starts a new `MotorwaysGame`
  itself, inside `TransitionIn`, exactly once.
- **`PrepareForNewGame(...)`** was confirmed unsafe (would call `.Start()` a second time on an
  already-started game, causing `_simulation.Subscribe(_view)` to re-fire `OnModelAdded` for
  every existing model with no deduplication — duplicate views). Not used.
- There is no public API to hand an already-separately-constructed `MotorwaysGame` to
  `GameContainerScreen` without one of these two paths, so "constructing the Phase 8 game" for
  this run means: set the seed via `global::Random.SetSimulationSeed` beforehand (the same
  native seed-control entry point used in every prior phase), then let `PrepareForMap`'s own
  internal `_appScope.Get<MotorwaysGame>()` + `.Start(...)` build it — this is the identical
  construction Phase 8's headless episodes use, just invoked by the stock screen instead of by
  us directly.
- The originally-diagnosed `StopMenuGameAudio()` `NullReferenceException` hazard is sidestepped
  by construction now: `GameContainerScreen.TransitionIn` still calls
  `screenStack.GetActiveScreen<StartupScreen>()?.StopMenuGameAudio()`, and
  `GetActiveScreen<StartupScreen>()` still returns the pooled (non-null) instance — but by the
  time `WaitForStockFrontendBootComplete` has confirmed a stable `MainMenuScreen`, that
  instance's `_menuGame` is already constructed (confirmed live via a read-only reflective
  diagnostic immediately before pushing `GameContainerScreen`: `"its _menuGame = non-null
  (StopMenuGameAudio is safe)"`), so the call no longer NREs.

## 5. Step 3 — clock ownership (HARD invariant, verified twice)

**Confirmed via decompilation**: `GameContainerScreen.Tick(deltaTime)` calls
`_game.Tick(deltaTime)` every real frame while active and `!_gameSuspended` — a second,
real-frame-driven clock that would race against Phase 8's controlled `ISimulation.Step()`
calls. Mitigated via the existing public `GameContainerScreen.SetGameSuspended(true)` (a stock
mechanism — `ShouldTickGame()` returns false — not a patch), called immediately after
attachment, before any `AdvanceSimulation` call of our own.

**Verified twice, live, this run:**

1. **Idle-frame check** (`AttachGameToFrontend`): immediately after `SetGameSuspended(true)`,
   recorded `simTime=0.0000` and a `PolicyObservation` fingerprint, then waited 15 real frames
   with zero `AdvanceSimulation` calls, then re-recorded both. Result: `simTime=0.0000 (was
   0.0000)`, fingerprint unchanged — **PASS, no duplicate stepping detected**.
2. **Per-decision step-count check** (`RunVisibleRandomPolicyEpisode`): every one of the 10
   `AdvanceSimulation` calls (each `decisionDelta=0.3`s, i.e. exactly 3 controlled
   `ISimulation.Step()` calls) was followed by comparing the actual `ClockModel.Time` advance
   against the expected `0.3000`. All 10 decisions logged `stepCountOk=True`; final `simTime`
   after decision 9 was exactly `3.00` (10 × 0.3, exact).
3. **Independent external confirmation**: a separate, pre-existing diagnostic poller (unrelated
   to the Phase 9A code, sampling `ClockModel.Time` every ~2s) recorded `ClockModel.Time =
   3.0000000028` at `+13.9s`, `+15.9s`, `+17.9s`, and `+19.9s` real time — i.e. completely
   unchanged for 6+ seconds after the RandomPolicy loop finished calling `AdvanceSimulation`,
   from a measurement path with no knowledge of our own instrumentation.

## 6. Step 4 — identity/integrity proof (all confirmed)

- **Before attachment**: no `MotorwaysGame` existed yet (nothing to attach to — `PrepareForMap`
  constructs it internally; see §4). Seed `2026` set via `global::Random.SetSimulationSeed`.
- **After attachment**: `attachedGame` hash `-312078926`, `Server.Simulation` hash
  `-1415864896`, `ReferenceEquals(app.Game, attachedGame) = True` (`app.Game` hash also
  `-312078926`) — **the exact same `MotorwaysGame`/`Server.Simulation` the stock frontend
  considers active is the one Phase 8 is driving.**
- **No unexpected simulation-time advancement**: confirmed in §5.
- **No unexpected `PolicyObservation` mutation**: the fingerprint only changed in step with our
  own applied actions (`415020ab94105e66` at attachment → `84c8ae3cf8fb83a2` after 7 real
  `ADD_ROAD` actions across the loop) — never between our own calls.
- **Persistence isolation remained PASS** at every single checkpoint this run: `C3`, `D`, `E`,
  `E2`, `F`, `PRE-ACTION`, `POST-ACTION` (canary), `post-attach`, and all 10
  `visible-pre-action-decisionN` checkpoints, plus `visible-post-loop`. Zero failures.

## 7. Step 5 — visible RandomPolicy run (the Phase 9A milestone)

10 decisions, `decisionDelta=0.3`s each, `pWait=0.3`, seed `90125` for the policy RNG, action
bounds `(-5,-5,16,16)`:

| # | simTime | action | legalActionCount | accepted |
|---|---|---|---|---|
| 0 | 0.30 | `ADD_ROAD(-5,-2,S)` | 481 | True |
| 1 | 0.60 | `WAIT` | 480 | True |
| 2 | 0.90 | `WAIT` | 480 | True |
| 3 | 1.20 | `ADD_ROAD(10,4,S)` | 480 | True |
| 4 | 1.50 | `WAIT` | 480 | True |
| 5 | 1.80 | `ADD_ROAD(-3,10,S)` | 480 | True |
| 6 | 2.10 | `ADD_ROAD(0,-3,S)` | 480 | True |
| 7 | 2.40 | `ADD_ROAD(7,-3,S)` | 480 | True |
| 8 | 2.70 | `ADD_ROAD(-2,10,E)` | 480 | True |
| 9 | 3.00 | `ADD_ROAD(8,6,S)` | 480 | True |

Every action was `accepted=True` via the native `TileEditor.AddRoad`/`ClearTileExplicit` path
(unchanged since Phase 7A). The run paced at 30 idle real frames (~0.5s) between decisions for
human-viewable speed, with `GameContainerScreen` rendering normally throughout (its own
per-frame `_game.Tick` suspended; Phase 8's `AdvanceSimulation` the sole clock).

**Visual evidence**: screenshot/screen-capture access to the `Mini Motorways` window was
requested via `request_access` twice this session (once before this attempt, once mid-session
when the user asked to see visuals) and was explicitly denied by the user both times. Per this
session's non-retry policy for denied tool calls, no screenshot evidence exists for this
document — all evidence here is log-based (identity hashes, simulation time, action results,
fingerprints) and file-hash-based (persistence verification). The log evidence is unambiguous
on every invariant the brief asked for; it is simply not a picture.

## 8. Persistence verification

- **Game exe hash**, before and after: `305D1EA095F0BA9E844B91497283277C61F0DEA9D5F7979065E78F3E7BAB0CE6`
  (unchanged across this and all prior Phase 9A attempts this session).
- **Real player persistence**, before and after this run: `extendedUserProfile_
  451e710a...json` = `BA6AFCB4DFCF8B0FEC27B4AF6D380993A73876B5A86B7C902E86922DF809473C`;
  `userProfile_451e710a...json` = `1E39344254F5D01FD524183635E389F1F7352142B428FFC44FE760061
  C916C83`; both `gameJournal_*.dat` files unchanged — byte-identical before/after. Only
  `deviceSettings_c0693430..._451e710a...json` changed (expected timestamp-only write).
- `doorstop_config.ini` reverted to `target_assembly=BepInEx\core\BepInEx.Preloader.dll`; game
  process terminated after the run.

## 9. Bootstrap code changes this phase (`game_plugin/DoorstopBootstrap/Entrypoint.cs`)

- `InitializeAtlases` replaced with `VerifyAtlasesPopulatedByStock` — read-only probes only, no
  `Import()`/`Initialize()` calls (the stock `StartupScreen` boot already did this work).
- New `WaitForStockFrontendBootComplete` — the Step 1 signal described in §3, called before any
  of our own initialization.
- `InitializeProcessContext` restructured to call `WaitForStockFrontendBootComplete` first, then
  confirm the redundant `LoadingScreen` `PlayerChanged`-stack-trace signal, then create the test
  player, then verify atlases — in that order, with no action before the full stock boot signal.
- `AttachGameToFrontend`'s pre-push hazard check changed from "wait up to 30s for
  `StartupScreen._menuGame != null`" (attempt 2, removed) to a one-shot diagnostic read of
  `_menuGame` with an abort-only-if-still-null gate (attempt 3) — no waiting/intervention, since
  `WaitForStockFrontendBootComplete` already guarantees it should be non-null by this point.
- Added the Step 3 clock-ownership idle-frame check (§5.1) and per-decision step-count check
  (§5.2) to `AttachGameToFrontend`/`RunVisibleRandomPolicyEpisode`.

## 10. Remaining limitations

- No visual/screenshot evidence (user declined screen-capture access twice; see §7).
- The run used a short, 10-decision episode at a human-viewable pace, not an extended session —
  sufficient to meet the Phase 9A milestone as specified, not a stress test.
- `research/ARCHITECTURE.md` §8D has been corrected and updated to reflect this phase's actual
  findings (the `_activeScreens` vs `_visibleScreens` distinction, the final working signal, and
  the achieved outcome) in place of the two-failure writeup from the previous attempt.

## 11. Stop condition

Per the brief: RandomPolicy is now visibly controlling the actual native game while Phase 8
remains authoritative and deterministic. Phase 9A is complete. Per explicit instruction, no PPO,
no streaming/OBS/FFmpeg infrastructure, no telemetry overlays, no custom renderer, and no further
reverse-engineering follow from this — next steps require new direction.
