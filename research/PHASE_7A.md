# PHASE 7A — Native Road Action Smoke Test

Generated 2026-10-06. Objective: prove that injected code can perform one legitimate road edit through the game's native semantic command/edit system — not by mutating `TilemapModel`/lane/pathfinding structures directly, and not by fabricating internal state the normal edit system would construct itself — and observe the resulting state change. **Result: SUCCESS. One road was added, verified active, removed, verified reverted, and a subsequent episode reset was verified NOT to leak the edit — all through the exact stock `TileEditor` → `EditTileCommand` → `ISimulation.ScheduleCommand` path real player input uses, with zero exceptions and zero persistent-save-data impact.**

This directly answers the phase's one question: **yes** — external agent code, running from this project's Doorstop bootstrap, can cause a legitimate Mini Motorways road action through the game's own simulation interfaces, with no UI, no patched assemblies, and no synthesized internal state.

## Task 1 — Static trace of the normal gameplay path

Delegated to a background subagent (full decompilation), independently spot-verified by this session (`ClientUpgradeDatabase`'s DI wiring, `TileEditor`/`EditTileCommand`/`AddRoadEdit`/`ClearTileEdit`'s registration scope, `GameBehaviourModel.GetConcreteCostForConnection`'s cost formula, `Tile.CanSetNodeState`/`GetTwoLaneRoadStateInDirection` existence) before any code was written.

### `EditTileCommand` (`Motorways.Commands/EditTileCommand.cs`)

```csharp
public class EditTileCommand : Command, IReleasedFromScopeHandler
{
    [Dependency] private TilemapModel _tilemap;
    [Dependency] private UpgradeDatabaseModel _upgradeDatabase;
    private TileEdit _edit;

    public override void Execute(ISimulation simulation)
    {
        if (Diagnostics.Verify(_upgradeDatabase.ApplyEdit(_edit, _tilemap), ...))
            Diagnostics.Verify(_edit.ApplyToTilemap(_tilemap), ...);
        _edit.ApplyToSimulation(simulation);
    }

    public static EditTileCommand Create(IScope scope, TileEdit edit) { ... }
}
```

`EditTileCommand : Server.Command` — the exact same base class, and the exact same `ISimulation.ScheduleCommand(Command)` submission mechanism, already proven for `InitCityCommand` since Phase 5A. This confirms the prior architecture doc's PROBABLE speculation outright.

### `AddRoadEdit` (`Motorways/AddRoadEdit.cs`)

```csharp
public static AddRoadEdit Create(IScope scope, ITilemap tilemap, Vector2Int originCoordinates,
    TileDirection direction, CityDefinition cityDefinition)
```

Only ONE edge at a time, between two tiles that must already be adjacent (`DestinationCoordinates` is derived from `originCoordinates`/`direction` via `TileUtilities.GetAdjacentCoordinates` — there is no multi-tile-road-in-one-edit path). `ApplyToAffectedTile` sets `RoadState.Pending` (not `Active` directly) on both tiles. `ApplyToUpgradeDatabase` computes `GameBehaviourModel.GetConcreteCostForConnection(...)` — **confirmed via decompilation: connecting two plain empty (`TileContentType.None`/`Tree`) tiles costs exactly 1 concrete** — and either consumes `UpgradeType.Concrete` or un-mothballs an existing connection, failing the whole command (silently, no tilemap mutation) if insufficient.

### `ClearTileEdit` (`Motorways/ClearTileEdit.cs`)

```csharp
public static ClearTileEdit Create(IScope scope, Vector2Int coordinates, ITilemap tilemap,
    Tile.TileChangePermissions changePermissions = Tile.TileChangePermissions.Full)
```

NOT a narrow single-edge inverse of `AddRoadEdit` — it mothballs **every** road direction on the target tile (computed via `GetRoadDirectionsToMothball`). The static analysis predicted the removed road would end up `RoadState.Mothballed` (lanes kept, just inactive) rather than disappearing. **Live testing corrected this** — see Task 5.

### Real callers — confirmed

`Motorways/TileEditor.cs` (`[Dependency] IScope`, `GameBehaviourModel`, `City`, `ClientUpgradeDatabase`) is the sole producer, via its public `AddRoad(ITilemap, Vector2Int, TileDirection)` / `ClearTileExplicit(ITilemap, Vector2Int, ClearTileOfType, TileChangePermissions)` methods, returning a `TileEditResult {resultCode, edit, errorPosition}`. Real gameplay callers: `Motorways.Actions/DrawRoadAction.cs` (mouse/touch drag-to-build) and `Motorways.Actions/ControllerDrawRoadAction.cs` (gamepad), via `MotorwaysPlayerAction.AddTileEdit` → `ScheduleClientTileEdit`:

```csharp
EditTileCommand command = EditTileCommand.Create(base.Scope, clientTileEdit.edit);
_simulation.ScheduleCommand(command);
```

The `ClientTileEdit`/`TilemapView` wiring surrounding this in the real UI code is pure view-layer presentation bookkeeping (road-preview animation) — confirmed to perform no simulation mutation itself, and safely skippable for a bootstrap that only cares about simulation state.

### Preconditions (found in `TileEditor.AddRoad`)

Both tiles must exist/be creatable; both must be `CityDefinition.TileIsBuildable`; carpark/destination/house/water/mountain/rail special-casing (each producing a different edit type or a rejection code, not relevant to a plain tile-to-tile road); `Tile.CanSetNodeState(..., RoadState.Pending)` on both sides; sufficient `UpgradeType.Concrete` (or an existing mothballed connection to reactivate). `EditTileCommand.Execute` re-checks the concrete-cost success at execution time, independently of `TileEditor.AddRoad`'s construction-time check.

### Submission mechanism — confirmed

`ISimulation.ScheduleCommand(Command command)` (`Server/ISimulation.cs:33`) — identical in kind to `InitCityCommand`. No other queue exists.

### Removal/promotion groundwork

`RoadState.Pending → Active` promotion is automatic: `LaneUpdateProcess.Step` (one of the 26 processes registered by `InitCityCommand`, confirmed in Phase 6A's process registry) unconditionally calls `TilemapModel.ActivateUnblockedPendingLanes()` every simulation step — no extra manual call needed beyond letting the simulation tick at least once after the command executes.

## Task 2 — Live test: build → verify → remove → verify → reset → verify

Chose tile `(5,5)` → `TileDirection.East` → `(6,5)` (far from the LosAngeles map's pre-existing destination footprint and from where houses had been observed spawning in Phase 6A's identical-seed run) as the known-empty adjacent pair, using native state (`TilemapModel.GetTile` returning `null` for both, confirming untouched tiles) rather than guessing blind.

Implemented in [`Entrypoint.cs`](../game_plugin/DoorstopBootstrap/Entrypoint.cs) as `Phase7ARunner.TryAddRoad`/`TryClearTile` (going through `testGame.Scope.Get<TileEditor>().AddRoad(...)`/`.ClearTileExplicit(...)` → `EditTileCommand.Create` → `ISimulation.ScheduleCommand`, exactly the traced path — no direct `TilemapModel`/lane mutation anywhere) plus a `DescribeRoadState` read-only probe (tile existence, `Tile.GetTwoLaneRoadStateInDirection`, concrete available/total, total active-lane count).

**Full live result** (complete log: [`logs/instrumentation.log`](../logs/instrumentation.log)):

| Step | `roadStateTowardDest` | `concreteAvailable/total` | `totalActiveLanes` | Notes |
|---|---|---|---|---|
| Before | `None` | `25/25` | `29` | Both tiles didn't exist yet (`GetTile` → null) |
| After `AddRoad` + 10 ticks | **`Active`** | `24/25` | `33` | `TileEditor.AddRoad` → `resultCode=Success`; `EditTileCommand` scheduled=`True`; Pending→Active promotion confirmed automatic |
| After `ClearTileExplicit(Roads)` + 10 ticks | **`None`** | `25/25` | `31` | Concrete fully refunded; road reverted (see correction below) |
| +120 more ticks (2s) | `None` (unchanged) | `25/25` | `31` (unchanged) | `ClockModel.Time` 1.1s→2.6s, `Houses=1/Destinations=1/Vehicles=2` stable throughout, zero exceptions |
| Final, before teardown | `None` | `25/25` | `31` | Matches post-clear state exactly — no drift |
| Reset-leak-check episode (seed 1002), t0 | **`None`** | `25/25` | **`29`** | Fresh `MotorwaysGame`; both tiles `null` again (`originTileExists=False`) — back to the EXACT pre-edit baseline |

**Every item in the brief's live-test checklist is confirmed**:
- **Road state changed at the intended tile(s)**: `None → Active`, confirmed via the tile's own `GetTwoLaneRoadStateInDirection`, not inferred.
- **Relevant connectivity updated**: active lane count jumped `29 → 33` (+4: the new bidirectional connection plus its wiring into the two adjacent `RoadChunkModel`s) when the road was added.
- **No unrelated corruption**: `Houses`/`Destinations`/`Vehicles` counts and `GameRules` concrete type stayed exactly as expected throughout (matching Phase 6A's steady-state for this seed/timeframe); the concrete-cost accounting matched the statically-predicted formula exactly (1 concrete for a plain empty-tile connection).
- **Simulation continues ticking normally**: 120 further ticks, zero exceptions, `ClockModel.Time` advancing.
- **Removal via the legitimate path, verified**: `TileEditor.ClearTileExplicit(..., ClearTileOfType.Roads)` → `EditTileCommand` → `ScheduleCommand`, succeeded, state reverted.
- **Native episode reset verified with no leak**: a fresh `MotorwaysGame` (seed 1002) shows the identical pristine baseline (`None`, `29` active lanes, tile objects not yet created) at the exact coordinates episode 1 edited.

### Correction to the static prediction (reported honestly, per this project's practice)

The static trace predicted `ClearTileEdit` would leave the road `RoadState.Mothballed` (lane structurally kept, just inactive) rather than fully reverted. **Live testing shows `roadStateTowardDest` reading `None` after the clear, not `Mothballed`**, with concrete fully refunded to `25/25` (exactly the pre-add value). The most likely explanation (not independently re-verified by a second decompilation pass this phase, flagged as an open question below) is that `GetRoadDirectionsToMothball`/`ApplyToAffectedTile` fully releases a connection back to `None` when it was the tile's only road and had no other structure depending on it — i.e. "mothballed" is the state for roads kept-but-paused as part of a larger network change (e.g. during a roundabout upgrade), while clearing an isolated, freshly-placed, never-reconnected road reverts it cleanly. This is a genuine correction, caught by live evidence exactly as this project's methodology intends — not glossed over.

One side detail, also reported rather than hidden: `totalActiveLanes` settled at `31` after the clear and stayed at `31` through the following 120 ticks and into teardown — 2 more than the pre-edit baseline of `29`. This is NOT attributable to our edit (which fully reverted, per the tile-level check above) — it matches Phase 6A's independently-observed pattern of a house's driveway lanes appearing a few ticks after warm-up in this same seed/map. The reset-leak-check episode's `29` (not `31`) at its own t0 confirms this: it's ordinary background city-building activity from THIS episode's own `BuildingSpawningProcess`, reset away cleanly in the next episode along with everything else, not a sign of leakage from our edit.

## Task 3 — Visual confirmation

Not performed this phase — the brief marks this "where practical," and this session's live tests run the game process without an attached interactive view (no `preview_start`/browser pane applicable to a native Win32 game window, and attaching one was judged unnecessary extra risk/scope for a smoke test already fully verified through native model-state reads, which are strictly more precise than a screenshot). Flagged as skipped-by-judgment, not silently omitted.

## Persistence protection

Full before/after SHA-256 of every file in the persistent-storage directory: `Mini Motorways.exe` and all gameplay/profile/save files (`userProfile_*`, `extendedUserProfile_*`, both `gameJournal_*.dat`, `challengeOverrides*.json`) byte-identical; the same one benign device-settings timestamp write recurred (as in every phase since 5E). Note this is a meaningful confirmation specific to this phase: the road edit DID mutate real simulation state (`TilemapModel`, `UpgradeDatabaseModel`) in memory, yet still produced no additional persistent-storage writes — consistent with the Phase 5E finding that `TrySave`/journal writes are gated by the 300-second real-time autosave threshold, which this short test never approached.

## Hard constraints — compliance

- **Did not directly mutate `TilemapModel`, road collections, lane graphs, or pathfinding structures** — every mutation went through `TileEditor.AddRoad`/`ClearTileExplicit` → `EditTileCommand.Create` → `ISimulation.ScheduleCommand`, the real stock path.
- **Did not synthesize internal state the normal road-edit system would ordinarily maintain** — `AddRoadEdit`/`ClearTileEdit` instances were obtained from `TileEditor`'s own validating factory methods, not hand-constructed with guessed internal fields.
- Did not build a generic action API, action masking, path planner, RL action space, Python IPC, Gymnasium, reward, or PPO — exactly one add and one remove, read verified via native state each time.
- No original game assembly patched; no corlib/System assembly replaced; no BepInEx/Harmony/MonoMod introduced; no UI/input used.
- Game process remained healthy throughout and was cleanly terminated; `doorstop_config.ini` reverted to the BepInEx resting state.

## Files changed this phase

- [`game_plugin/DoorstopBootstrap/Entrypoint.cs`](../game_plugin/DoorstopBootstrap/Entrypoint.cs) — `Phase6ARunner` renamed to `Phase7ARunner`; added `TryAddRoad`/`TryClearTile` (the stock edit/command submission helpers) and `DescribeRoadState` (read-only per-tile/direction probe); `RunEpisodeWithSnapshots` renamed `RunRoadEditEpisode` and extended with the add→verify→remove→verify sequence (shortened the trailing sustained-tick count from 360 to 120, since this phase's focus is the edit, not long-run dynamics); added `RunResetLeakCheckEpisode` (a second, independent episode checking the edited tile for leakage).
- [`research/state_snapshot_example.json`](state_snapshot_example.json) — regenerated from this phase's road-edit episode (7 samples: t0, after-add, after-clear, plus 4 periodic samples during the trailing tick loop) — supersedes Phase 6A's example file with one that also demonstrates a road-state transition, not just entity appearance.

## Hashes

- `Mini Motorways.exe`: `305d1ea095f0ba9e844b91497283277c61f0dea9d5f7979065e78f3e7bab0ce6` — unchanged.
- `Bootstrap\MiniMotorwaysBootstrap.dll` (this phase's build): `9f58265b6c598e5c47c7b7b6eaad6a913a0bd237d9d504010c5155a14f0ce42c`.
- `doorstop_config.ini`: reverted to `target_assembly=BepInEx\core\BepInEx.Preloader.dll` (resting state).

## research/ARCHITECTURE.md update

§7 ("ROAD / ACTION-INJECTION PATH") updated: the previously PROBABLE `EditTileCommand.Create(scope, edit)` + `ScheduleCommand` path is now CONFIRMED live, with the exact `TileEditor.AddRoad`/`ClearTileExplicit` entry points, preconditions, and the `ClearTileEdit`-reverts-to-`None`-not-`Mothballed` correction recorded.

## Open questions

1. Why the cleared road reverted to `RoadState.None` rather than the statically-predicted `Mothballed` was not re-confirmed via a second decompilation pass (e.g. reading `GetRoadDirectionsToMothball`/`Tile.SetNodeState`'s exact branching for an isolated, never-reconnected road) — flagged as a real, reported discrepancy rather than resolved.
2. Whether `AddRoadEdit`/`ClearTileEdit` behave identically when the target tile already has OTHER roads on it (this phase deliberately chose a pristine, previously-untouched tile pair to isolate the one edge being tested) — not exercised this phase.
3. Diagonal connections, bridge/tunnel-triggering connections (`Passage.WillConnectionStartPassage`), and the house/carpark/destination-adjacent special cases in `TileEditor.AddRoad` were not live-tested — only the plain tile-to-tile case the brief asked for.
4. Visual/in-window confirmation (Task 3's "where practical") was not attempted.

## Stop condition reached — SUCCESS

Per instruction: *"Stop after build → verify → remove → verify → reset succeeds, or after identifying the exact blocker."* All five stages succeeded, each with direct native-state evidence, not assumption. This phase answers its one question affirmatively: external agent code, via this project's bootstrap, CAN cause a legitimate Mini Motorways road action through the game's own stock simulation interfaces. Per the brief, no generic action API, action masking, path planner, action space, Python IPC, Gymnasium, reward, or PPO work was started.
