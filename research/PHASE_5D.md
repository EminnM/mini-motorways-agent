# PHASE_5D — Normal-Mode Headless Reset

Generated 2026-10-06. Objective: determine whether Phase 5C's autonomous lifecycle works with `GameMode.Normal` (preserving actual gameplay semantics), now that `CityDefinition.CompileTilemap()` is known to be required. **Result: Normal mode still fails, but the exact remaining dependency has been precisely identified and empirically confirmed — it is unrelated to `GameMode` or rendering entirely.** Per the explicit stop condition ("prove success, or identify the exact remaining dependency"), this phase stops here with a confirmed diagnosis rather than switching back to Background mode or patching around it.

## Minimal-change test, as instructed

Starting from the exact known-good Phase 5C code, the only change made was `GameMode.Background` → `GameMode.Normal` in the single `testGame.Start(cityDefinition, mode, null)` call — `CompileTilemap()`, the DI construction path, and the seed-setting call were left untouched.

**Result: identical failure to Phase 5B, at the identical line**, despite `CompileTilemap()` now being in place:
```
System.NullReferenceException: Object reference not set to an instance of an object
  at Motorways.Views.CameraView.Initialize (Motorways.GameRules rules) [0x00014] in CameraView.cs:323
  at Motorways.MotorwaysGame.Start (...) in MotorwaysGame.cs:284
  at Doorstop.Entrypoint+Phase5DRunner+<RunOneCycle>d__3.MoveNext () [0x000f2]
```
This confirms, empirically rather than by assumption: `CompileTilemap()` and `GameMode.Background` were solving **two different problems** in Phase 5C. `CompileTilemap()` alone does not touch whatever `GameMode.Background` was actually sidestepping (`UseCamera() → false`, skipping the `CameraView.Initialize` call entirely). Per instruction, this failure was investigated rather than assumed to be the old Phase 5B issue recurring unchanged — and it turned out to be a **different**, more specific root cause than either prior phase had identified.

## Root-cause investigation: three rounds of narrowing, each empirically checked

**Round 1 (static, subagent-delegated)**: decompiled `CameraView` in full. The exact crash line is:
```csharp
playerOrthoZoom = _visualConstantsData.ZoomLevelsCameraMin[_player.ZoomLevel];
```
Two candidate null fields: `_visualConstantsData` (`VisualConstantsData`) or `_player` (`ActivePlayer`). A plausible-but-unconfirmed hypothesis was offered: `VisualConstantsData`'s DI registration in `AppContainer.CreateAppAssembler()` is **unguarded** (`AssetBundleUtility.LoadAsset<VisualConstantsData>("core","VisualConstantsData")`, no `Diagnostics.Verify` null-check, unlike `GameCamera`'s registration which IS guarded) — so it was flagged as the likelier candidate.

**Round 2 (live, empirical — this hypothesis was WRONG, and the test caught it directly)**: added a direct, read-only probe (`app.Scope.Get<VisualConstantsData>()`) BEFORE constructing any test game. Result:
```
[Phase5D] Diagnostic: app.Scope.Get<VisualConstantsData>() = non-null, ZoomLevelsCameraMin.Count=5
```
Non-null, with a sensible `Count=5`. A second probe inside `RunOneCycle`, reading `testGame.Scope.Get<VisualConstantsData>()` (the exact same resolution `CameraView` would perform, from the exact same scope `CameraView` lives in), confirmed:
```
[Phase5D] Cycle 1: Diagnostic: testGame.Scope.Get<VisualConstantsData>() = non-null, sameInstanceAsAppScope=True
```
**`VisualConstantsData` is ruled out, definitively, by direct measurement** — the Round 1 hypothesis, while plausible from static reading alone, was wrong, and the live test is what caught it. This is reported honestly rather than left standing, consistent with this whole investigation's practice of correcting earlier guesses when evidence contradicts them.

**Round 3 (static + live, decisive)**: with `VisualConstantsData` ruled out, the only remaining candidate at that exact line is `_player.ZoomLevel`. Decompiling `Motorways.ActivePlayer` directly (`_player` is declared as type `ActivePlayer`, confirmed from `CameraView`'s field list) found:
```csharp
public int ZoomLevel {
    get { return MotorwaysDeviceSettings.ZoomLevel; }
    ...
}
public MotorwaysDeviceSettings MotorwaysDeviceSettings => (MotorwaysDeviceSettings)_player.DeviceSettings;
                                                                              // ^ ActivePlayer's OWN private backing field, type Player -- NOT the same "_player" as CameraView's field!
public Player Player => _player;
public bool HasActivePlayer => _player != null;   // <-- this guard existing elsewhere in the SAME class confirms null is an expected, handled state
```
`ActivePlayer.ZoomLevel` → `MotorwaysDeviceSettings` → dereferences `ActivePlayer`'s own internal `_player` (a `Player` object representing the currently-loaded player profile) → `.DeviceSettings`. If no profile has been loaded, that internal field is null, and the chain throws exactly the observed `NullReferenceException`, with the throw site correctly attributed to `CameraView.Initialize` (the outermost call, since `ZoomLevel`'s getter is a trivial one-liner the JIT can report as having thrown from its caller's IL offset).

**Live confirmation, decisive**:
```
[Phase5D] Cycle 1: Diagnostic: ActivePlayer.HasActivePlayer = False
```
**Confirmed. No player profile is loaded or active in this bypass context.** This is the exact remaining Normal-mode dependency.

## Why this is NOT a GameMode or rendering issue

`GameMode.Background`'s success in Phase 5C worked by a different, orthogonal mechanism entirely: `BackgroundGameRules.ShowsUI() → false` → `GameRules.UseCamera() → false` → `MotorwaysGame.Start()`'s `if (gameRules.UseCamera())` guard **skips the `CameraView.Initialize()` call outright** — it never fixes the underlying null-player condition, it just avoids the one code path that happens to touch it. This matters for the task's explicit rendering-vs-renderless question: **a missing player profile is not a rendering dependency at all.** Even a fully-rendered `GameMode.Normal` session, displayed into a real window or a virtual display (Xvfb, per the Ubuntu deployment target), would hit this exact same `NullReferenceException` unless a player profile is loaded first — because in the real/stock game, `ProfileSelectScreen`/`ProfileCreationScreen` (class names already catalogued in `ARCHITECTURE.md` from Phase 2) run as a mandatory step **before** the main menu or any gameplay screen ever becomes reachable, populating `ActivePlayer`'s internal state as a side effect. Our bypass starts directly from an already-booted `AppRuntime`/main-menu state and has never gone through that flow.

## What this phase does and does not establish

**Established, with direct evidence:**
- `CompileTilemap()` alone is insufficient for `GameMode.Normal` — the Phase 5B `CameraView` failure is unrelated to `CompileTilemap()`/tilemap compilation at all.
- The exact failing expression (`ActivePlayer.ZoomLevel` → `MotorwaysDeviceSettings` → null internal `Player`) and the exact missing precondition (`ActivePlayer.HasActivePlayer == False`) are both confirmed by direct, minimal, read-only live probes — not inferred from a single stack trace alone.
- This dependency is orthogonal to `GameMode` and orthogonal to rendering — it is a **player-profile-loading** precondition, confirmed via `HasActivePlayer`.

**Not established (explicitly out of scope for this phase, per the stop instruction):**
- How to autonomously load/select/create a player profile without UI interaction. `ProfileSelectScreen`/`ProfileCreationScreen` were catalogued by name in Phase 2 but not decompiled this phase — that is the natural next investigation, not attempted here.
- Whether, once a profile is loaded, `GameMode.Normal` succeeds cleanly the rest of the way (no further Normal-mode-specific failures were reached, since the test aborts at the very first `CameraView.Initialize()` call inside `Start()`).
- The three-cycle run with seeds 1001/1002/1003, the "inspect model/state values to verify `Reset()` actually cleared episode state" check, and the stronger same-seed determinism experiment (advancing far enough for real spawns) were **not** performed this phase — all of them require a successful `Start()` call first, which Normal mode does not yet achieve.

## Stop condition reached

Per instruction: *"Stop after proving that [success] or identifying the exact remaining Normal-mode dependency."* The exact remaining dependency has been identified and empirically confirmed: **an active player profile must be loaded (`ActivePlayer.HasActivePlayer == true`) before any `GameMode.Normal` session can call `MotorwaysGame.Start()` successfully**, because `CameraView.Initialize()` unconditionally reads player-profile-backed device settings. This phase stops here, as instructed, rather than attempting to load/fake a profile, patch `CameraView`, or fall back to `GameMode.Background`.

## Hard constraints — compliance

- Did not switch back to `GameMode.Background` as "the solution" — it remains documented (Phase 5C) as a mechanism-only proof, not reused here to paper over this phase's finding.
- Did not patch, null-guard, or disable `CameraView`, `ActivePlayer`, or any rendering component — the three diagnostic probes added this phase are pure reads (`Get<T>()`, property getters) with zero mutation, added specifically to test hypotheses rather than to make the failure disappear.
- No road/action injection, no reward design, no PPO/RL, no Python IPC, no Gymnasium, no observation design, no streaming — none attempted.
- No original game assembly patched; no corlib/System assembly replaced; no BepInEx/Harmony/MonoMod introduced.
- Game process remained healthy and responsive throughout every test this phase (604-622 MB working set, responsive, correct window — confirmed via `Get-Process` after each of the 4 live test runs).

## Files changed this phase

- [`game_plugin/DoorstopBootstrap/Entrypoint.cs`](../game_plugin/DoorstopBootstrap/Entrypoint.cs) — `Phase5CRunner` renamed to `Phase5DRunner`; `GameMode.Background` reverted to `GameMode.Normal`; seeds array temporarily reduced to a single probe seed (`{1001u}`) to isolate the hypothesis cleanly before any multi-cycle run was warranted; three diagnostic read-only probes added (`VisualConstantsData` from app scope, `VisualConstantsData` from game scope with identity comparison, `ActivePlayer.HasActivePlayer`).

## Hashes

- `Mini Motorways.exe`: `305d1ea095f0ba9e844b91497283277c61f0dea9d5f7979065e78f3e7bab0ce6` — unchanged, re-verified before and after all four live test runs this phase.
- `doorstop_config.ini`: reverted to `target_assembly=BepInEx\core\BepInEx.Preloader.dll` (resting state) after testing.

## research/ARCHITECTURE.md update

One fact added to §6B this phase (experimentally established via direct live probes, correcting an initial static-analysis-only hypothesis that was tested and found wrong): the `GameMode.Normal` `CameraView.Initialize()` failure is caused by a missing active player profile (`ActivePlayer.HasActivePlayer == false`), not by `VisualConstantsData` resolution — confirmed non-null and identical across app- and game-scope by direct measurement. This is unrelated to `GameMode` or rendering; it is a player-profile-loading precondition that the stock game satisfies via a mandatory `ProfileSelectScreen`/`ProfileCreationScreen` flow before any gameplay screen is reachable.

## Recommended next step (not started)

Decompile `ProfileSelectScreen`/`ProfileCreationScreen`/`ActivePlayer`'s profile-loading methods to find the smallest legitimate, code-level (no UI) call that populates `ActivePlayer`'s internal `Player` reference — analogous to how Phase 5C/5D found `MapDatabase`/`AssetBundleUtility.LoadPrefabAsync` as the legitimate autonomous path for city/map loading. This is the natural Phase 5E.
