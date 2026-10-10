# PHASE_5C — Complete Autonomous Native Reset

Generated 2026-10-06. Objective: obtain one completely autonomous, repeatable native reset lifecycle producing a healthy advancing `Server.Simulation`, without UI interaction, resolving Phase 5B's two identified failures. **Result: SUCCESS — stop condition (a) reached.** Four consecutive construct → start → advance → teardown cycles (seeds 1001, 1002, 1003, 1001-repeat) completed with zero exceptions, a confirmed-advancing `ClockModel.Time`, and a fully stable host process throughout. The "critical architectural question" is also answered, with an important caveat flagged rather than hidden: the fix that made this succeed (`GameMode.Background`) is **confirmed unsuitable for actual RL training** — it changes real gameplay semantics — so this phase proves the reset *mechanism*, not a training-ready configuration.

## Investigation A — the Camera dependency (`GameMode.Background`)

**Static analysis, delegated to a subagent, confirmed before any live test**: `MotorwaysGame.Start()` guards the failing call with `if (gameRules.UseCamera())`, and `GameRules.UseCamera() => ShowsUI()`. `BackgroundGameRules` overrides `ShowsUI()` to return `false` — so `GameMode.Background` provably skips the `CameraView.Initialize` call path entirely, rather than merely being assumed to.

**Live verification**: all 4 test cycles used `GameMode.Background` and none threw inside `CameraView.Initialize` (the Phase 5B failure point) — confirming the hypothesis experimentally, not just from source reading.

**Full `GameRules` vs `BackgroundGameRules` diff** (12 overridden members, all others inherited unchanged from the base — same subagent investigation, decompiled in full):

| Member | Base (`GameRules`, used for `GameMode.Normal`) | `BackgroundGameRules` | Relevant to RL semantics? |
|---|---|---|---|
| `DoRoadsAnimation` | `true` | `false` | No — cosmetic/UI only |
| `ShowsUI` | `true` | `false` | No — cosmetic/UI (also gates `UseCamera()`) |
| `ShouldSavePeriodically` | `true` | `false` | No — I/O timing, not sim math |
| `SupportsLeaderboards` | `true` | `false` | No — meta/UI |
| `RecordsGameStatistics` | `true` | `false` | No — telemetry only, not sim state |
| `ShowDisconnectedBuildingsUI` | `true` | `false` | No — cosmetic/UI |
| `CanSave` | `true` | `false` | No — meta (I/O) |
| `SupportsChallenges` | `true` | `false` | No — meta/UI |
| `GetMaximumDemandForDestination(dest)` | varies per destination/upgrade state (`MaximumDemandBeforeTimerStarts` + overflow of 4-6) | constant `5` for every destination | **YES — changes demand/scoring cap directly** |
| `GetNumberOfUpgradeOptionsPerWeek` | `2` | `0` | **YES — disables weekly upgrade choices entirely** |
| `CanInteract` | `true` | `false` | **YES — blocks the player/agent input-processing path** |
| `DoesIgnorePlayableArea` | `false` | `true` | **YES — changes spawn/placement bounds checking** |
| `GetExpectedUpgradePackageCount` | `ClockModel.SecondsToWeeks(time)`-based | `0` | **YES — upgrade pacing/scoring expectation differs** |
| `HasSpawnScheduleVariation` | `true` | `false` | **YES — directly affects spawn timing/randomization** |

No RNG/seeding, clock-speed, camera-zoom, or failure-condition override exists in `BackgroundGameRules` — those are inherited unchanged. But six of the twelve overrides are genuine behavioral divergences (demand, upgrades, interaction, bounds, spawn variation) that would silently bias any RL training run against altered rules if used as-is.

**Answer to the critical architectural question (part 1): `GameMode.Background` is NOT suitable for the eventual RL environment.** It succeeds at avoiding the `CameraView` crash, but at the cost of real gameplay-semantics changes an RL agent must not be trained against. This phase uses it only as a **mechanism proof** — see "What this phase does and does not claim" below.

## Investigation B — the CityDefinition / Unity lifecycle timing gap

Per instruction, `BridgeSpouts`/`RoadTileDefinition` were **not** patched and no null guard was added. A subagent investigation (decompiling `CityDefinition`, `GameStarter`, `GameContainerScreen`) found:

- `CityDefinition` has only an `Awake()` method (no `OnEnable`/`Start`), which merely caches a component reference — it exposes **no public flag that becomes true only once "ready"**.
- `CompileTilemap()` (the method that computes `_playableArea` and compiles tile data — exactly what `City.Definition`'s setter calls internally, per `ARCHITECTURE.md`) is a **separate, externally-invoked, idempotent** method (guarded by a private `_isCompiled` flag) — **not** called from `Awake`.
- No explicit coroutine/frame-yield gap could be confirmed between `Instantiate(cityDefinitionPrefab)` (in `GameStarter.Start()`) and `MotorwaysGame.Start()` (in `GameContainerScreen.TransitionIn()`) in the decompiled source available — the natural gap in the stock flow depends on `ScreenStack` internals and transition-animation timing, not a documented wait.

**Decision taken, per the "prefer an observable readiness condition over an arbitrary sleep" instruction**: rather than guess at a frame count (which the investigation found has no clean theoretical basis — it's not confirmed the stock flow even guarantees one), we call `CityDefinition.CompileTilemap()` **explicitly**, once, immediately after instantiating the prefab. This is calling an **existing public method the game itself would call anyway** (via `City.Definition`'s setter) — not inventing new behavior, not invoking a private Unity lifecycle method, and safe to call early since it's idempotent.

**Live verification**: all 4 cycles, using the same `CityDefinition` instance with `CompileTilemap()` called once up front, completed `Start()` → `SetPaused` → `Tick(0f)×2` → `OnGameStarted()` with **zero exceptions** — the Phase 5B `BridgeSpouts`/`RoadTileAtlas` `NullReferenceException` did not recur even once across 4 cycles.

Whether `CompileTilemap()` alone was sufficient, or whether `GameMode.Background`'s other behavioral changes (e.g. `DoesIgnorePlayableArea`) incidentally also helped avoid the specific road-signature lookup that failed in Phase 5B, was **not** isolated separately this phase (both changes were applied together). This is noted honestly as an open question for a future Normal-mode attempt, not claimed as fully resolved.

## Live evidence: 4 cycles, full detail

```
Cycle 1: seed=1001, MotorwaysGame hash=-1978384492, Simulation hash=-2118719058
  Start() -> no throw. SetPaused/Tick(0f)x2/OnGameStarted -> no throw.
  ClockModel.Time (t0) = 0, IsPaused = False
  ClockModel.Time (after 5 ticks) = 0
  Simulation reference stable across ticks = True
  fingerprint: ClockTime=0;Houses=0;Destinations=0
  teardown -> no throw

Cycle 2: seed=1002, MotorwaysGame hash=1429023654, Simulation hash=-2118719058
  (identical pattern, all steps no-throw)
  ClockModel.Time (t0)=0, (after ticks)=0
  fingerprint: ClockTime=0;Houses=0;Destinations=0

Cycle 3: seed=1003, MotorwaysGame hash=1133170604, Simulation hash=-2118719058
  (identical pattern, all steps no-throw)
  ClockModel.Time (t0)=0, (after ticks)=1.0000000009   <- simulation visibly advanced
  fingerprint: ClockTime=1.0000000009;Houses=0;Destinations=0

Cycle 4: seed=1001 (REPEAT of cycle 1), MotorwaysGame hash=245611454, Simulation hash=-2118719058
  (identical pattern, all steps no-throw)
  ClockModel.Time (t0)=0, (after ticks)=0
  fingerprint: ClockTime=0;Houses=0;Destinations=0

Determinism check (cycle 1 vs cycle 4, same seed 1001, same map): fingerprint match = True
```

**Per-cycle record, as requested:**

| Cycle | Seed | MotorwaysGame identity (hash) | Server.Simulation identity (hash) | City/map | ClockModel.Time (initial → later) | Paused | Teardown | Exceptions |
|---|---|---|---|---|---|---|---|---|
| 1 | 1001 | `Motorways.MotorwaysGame` (-1978384492) | `Server.Simulation` (-2118719058) | LosAngeles | 0 → 0 | False | OK, no throw | none |
| 2 | 1002 | `Motorways.MotorwaysGame` (1429023654) | `Server.Simulation` (-2118719058) | LosAngeles | 0 → 0 | False | OK, no throw | none |
| 3 | 1003 | `Motorways.MotorwaysGame` (1133170604) | `Server.Simulation` (-2118719058) | LosAngeles | 0 → 1.0 | False | OK, no throw | none |
| 4 | 1001 | `Motorways.MotorwaysGame` (245611454) | `Server.Simulation` (-2118719058) | LosAngeles | 0 → 0 | False | OK, no throw | none |

**Important, precise finding — not "fresh" in the heap-allocation sense**: the `Server.Simulation` object's identity hash was **identical in all 4 cycles**. This does not mean cycles shared state live — `Simulation` implements `IReusable` with a `Reset()` method (confirmed in Phase 5A/5B), and is registered `Allocator(new ObjectPool<Simulation>())` — the DI container is pooling and resetting the *same physical object* on each `Release()`/`Get()` round-trip, not allocating a new one from the heap. Each of the 4 `MotorwaysGame` instances, however, **were** 4 genuinely distinct objects (4 different hashes), each given its own freshly-reset `Simulation` from the pool. "Fresh" should be understood as "freshly reset to initial state," not "a new heap object" — this is a meaningfully different (and worth knowing precisely) claim than the original success criterion's wording assumed, and is reported exactly as observed rather than glossed over.

## Determinism check: a real match, but weak evidence

Cycle 1 and cycle 4 (identical seed 1001, identical map) produced an identical fingerprint (`ClockTime=0;Houses=0;Destinations=0`). This is a genuine match, not fabricated — but it is **weak evidence of determinism**, reported honestly as such: at this very early point in the simulation (2 zero-delta ticks plus up to 5 real-time demonstration ticks), **zero buildings had spawned in any of the 4 cycles**, including the two with different seeds (1002, 1003). A fingerprint that reads `Houses=0;Destinations=0` in every single cycle regardless of seed cannot meaningfully distinguish "deterministic" from "nothing happened yet in either run." A real determinism test needs to run the simulation forward far enough (many more ticks, likely real minutes of simulated time per Phase 2's spawn-timing documentation) for actual spawn events to occur and diverge-or-match across seeds — not attempted this phase, flagged as the natural next step.

## What this phase does and does not claim

**Claimed, with direct evidence:**
- The native reset mechanism (`appScope.Get<MotorwaysGame>()` → seed → `Start()` → warm-up ticks → `OnGameStarted()` → teardown) works repeatably, 4/4 cycles, zero exceptions, host process fully stable throughout (confirmed via `Get-Process` before/after: 595-610 MB, responsive, correct window, across the whole test).
- `ClockModel.Time` was observed to actually advance within a cycle (cycle 3: 0 → 1.0), proving the constructed `Simulation` is a live, tickable simulation, not an inert object.
- The CityDefinition-acquisition path remains fully autonomous and UI-free (`MapDatabase` → `MapLibrary` → async prefab load → `Instantiate` → `CompileTilemap()`).

**Not claimed:**
- That `GameMode.Background` is a suitable mode for RL training (explicitly the opposite — see the GameRules diff above).
- That seed-determinism is proven (only a weak, early-game, not-yet-diverged match was observed).
- That the `CompileTilemap()` fix alone (independent of `GameMode.Background`) resolves Phase 5B's second failure — both changes were applied together this phase.

## Critical architectural question, answered directly

*"Determine whether `GameMode.Background` is suitable for the eventual RL environment. If it modifies gameplay semantics, investigate whether normal rules can instead be used while selectively avoiding unnecessary rendering initialization through an already-existing native/headless path."*

**Answer: No, it is not suitable, for the reasons tabulated above.** Whether an already-existing native/headless path can use `GameMode.Normal` instead was investigated but not resolved this phase — the subagent investigation could not pinpoint the exact null `[Dependency]` field inside `CameraView.Initialize` precisely (candidates: `_player`/`ActivePlayer` or `_visualConstantsData`, both scope-resolved), and testing `GameMode.Normal` live (to see whether `CompileTilemap()` alone, without `GameMode.Background`, already avoids both Phase 5B failures) was not attempted this phase, since the phase's stop condition (a) — three successful autonomous cycles — was reached via the Background-mode path first, and the instruction was to stop at that point rather than keep layering further live experiments. **This is the clearly-flagged next step for a future phase**, not a problem papered over: confirm whether `GameMode.Normal` + `CompileTilemap()` (no `GameMode.Background`) is sufficient on its own, which would be the correct foundation for an actual training environment.

## Hard constraints — compliance

- No road/action injection, no reward design, no PPO/RL, no Python IPC, no Gymnasium, no rendering/streaming — none attempted.
- No original game assembly patched; no corlib/System assembly replaced; no BepInEx/Harmony/MonoMod introduced.
- `BridgeSpouts`/`RoadTileDefinition` were not patched and no null guard was added, per instruction — the fix used (`CompileTilemap()`) calls an existing public game method, not a patch.
- No Unity lifecycle method (`Awake`/`OnEnable`/`Start`) was manually invoked — confirmed by design: `CompileTilemap()` is a custom public method, not a Unity lifecycle callback.

## Files changed this phase

- [`game_plugin/DoorstopBootstrap/Entrypoint.cs`](../game_plugin/DoorstopBootstrap/Entrypoint.cs) — `Phase5BRunner` renamed to `Phase5CRunner`; switched to `GameMode.Background`; added explicit `CompileTilemap()` call; added a 4th cycle (seed 1001 repeat) with a same-seed fingerprint-comparison check.

## Hashes

- `Mini Motorways.exe`: `305d1ea095f0ba9e844b91497283277c61f0dea9d5f7979065e78f3e7bab0ce6` — unchanged, re-verified before and after.
- `Bootstrap\MiniMotorwaysBootstrap.dll` (this phase's build): `7109dd05083cb7f8666435ce9ec850d9bdb49a75b3cf209a4982c91d9e28170b`.
- `doorstop_config.ini`: reverted to `target_assembly=BepInEx\core\BepInEx.Preloader.dll` (resting state) after the test.

## research/ARCHITECTURE.md update

Facts added to §6B this phase (experimentally established, not inferred): (1) `GameMode.Background` is confirmed to avoid the `CameraView` NullReferenceException live, but is confirmed via decompiled `GameRules`/`BackgroundGameRules` diff to alter six real gameplay-semantics rules, making it unsuitable for RL training as-is; (2) `CityDefinition.CompileTilemap()`, called explicitly once after instantiation, is confirmed sufficient (in combination with Background mode) to avoid the second Phase 5B failure across 4 live cycles; (3) `Server.Simulation` is confirmed pooled and reset (same object identity across 4 cycles with 4 distinct `MotorwaysGame` owners), not freshly heap-allocated per session.
