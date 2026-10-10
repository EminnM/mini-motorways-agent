# PHASE 6A — Native Simulation State Inventory

Generated 2026-10-06. Objective: build a complete, structured inventory of useful native simulation state exposed by the real `Server.Simulation`, via static analysis plus read-only runtime probes — explicitly NOT an RL observation-space design. **Result: the model graph is mapped (model registry, per-entity fields, relationships, two parallel connectivity layers, three coexisting coordinate representations), tagged OBSERVABLE/PRIVILEGED/UNCERTAIN throughout, and demonstrated live via a 13-sample `SimulationSnapshot` series from one sustained `GameMode.Normal` episode, showing coherent state evolution (entities appearing, demand appearing, lane count growing) with zero exceptions and zero persistent-save-data impact beyond the now-familiar benign device-settings timestamp.**

Per the brief's explicit constraint, the proven Phase 5F process-initialization/reset lifecycle was **not modified** except for the one change strictly necessary for read-only state access: the 4-cycle determinism-smoke-test episode shape was replaced with a single, longer (360-tick, ~6.2 simulated seconds) sustained episode so multiple snapshots could be captured as state evolved. `InitializeProcessContext`, `InitializeAtlases`, `RestoreExistingPlayerProfile`, and `GetCityDefinitionForTesting` are byte-for-byte unchanged from Phase 5F.

## Methodology

Three background subagents statically decompiled disjoint clusters of the native model graph in parallel (established delegation pattern from prior phases), cross-checked against an authoritative fourth read of `InitCityCommand.cs` (the actual model/process registry). Several of their specific claims (public accessor names, exact types) were independently spot-verified against the decompiled source by this session before being used in the live snapshot code — listed under "Independent verification" below. The live snapshot code itself is the strongest verification: every field it reads either compiled and returned sensible values, or the build/run would have failed outright.

## 1. The master model/process registry (`InitCityCommand.cs`, correcting prior documentation)

**Correction**: earlier phase notes (Phase 5A/5C) stated `InitCityCommand` registers "24 `IProcess`es." Full decompilation this phase found **26**, listed in registration order: `ClockProcess`, `BuildingSpawningProcess`, `BuildMotorwaysProcess`, `BuildRoundaboutsProcess`, `LaneUpdateProcess`, `GenerateDemandProcess`, `DispatchVehiclesProcess`, `TrafficLightAlternatingProcess`, `VehiclePathfindingProcess`, `IntersectionEvaluatingProcess`, `TrainSpawningProcess`, `BoatSpawningProcess`, `TrainMovementProcess`, `BoatMovementProcess`, `OpenTrainCrossingsProcess`, `VehicleSpawningProcess`, `VehicleMovementProcess`, `ParkVehiclesProcess`, `TutorialProgressionProcess`, `ReleaseMothballedLanesProcess`, `ReleaseMotorwaysProcess`, `TilePermanenceUpdatingProcess`, `EfficiencyCalculationProcess`, `FailureStateProcess`, `UpgradeAwardingProcess`, `AchievementCheckingProcess`. This is corrected here rather than silently carried forward, per this project's practice of flagging and fixing earlier inaccuracies.

**Models registered directly in `InitCityCommand`** (the "global/singleton framework models" — importantly, this is NOT where per-entity models like `HouseModel`/`VehicleModel`/`TileModel` are registered; those are added dynamically later by `BuildingSpawningProcess`, `City.GenerateCityLayout()`, `TileModel.Initialize`, and similar call sites, not traced in detail this phase):

| Model | Always/conditional | Purpose | Covered in this inventory |
|---|---|---|---|
| `CityModel` | always | City-level singleton: name, `GameMode`, `GameRules`, `pseudorandomGenerator`, `startOffset` | §2 (bounds/rules), §4 (RNG) |
| `CityPlanModel` | always | Not fully decompiled this phase — holds `groupHouseCounts[6]` at minimum (used by `HouseModel.GroupIndex`); full purpose is an **open question** | Partial |
| `DemandModel` | always | Global demand/supply bookkeeping | §3 |
| `TilemapModel` | always | Grid/tilemap container | §2 |
| `ClockModel` | always | In-game time/week/progression | §5 |
| `ScoreModel` | always | Player score | §3 |
| `UpgradeDatabaseModel` | always | Granted/pending upgrade packages | §3 |
| `ActiveChallengesModel` | always | Challenge-mode metadata (challenge type, time window, initial seed) | **Uncategorized** — not traced in depth this phase |
| `GameBehaviourModel` | always | Cross-cutting game-mode behavior switches, incl. `CanGameOver` | §3 (failure gate), included in live snapshot |
| `SnapshotModel` | conditional (`Feature.ValidateSimulationDeterminism`) | Dev/testing determinism-validation infrastructure | Out of scope by design |
| `IntersectionDecisionDatabaseModel` | conditional (`Feature.RecordIntersectionDecisions`) | Debug recording of intersection routing decisions | Out of scope by design |
| `EasterEggModel` | conditional (`Feature.WhatTheCarEasterEgg`) | Easter-egg state | Out of scope by design |

Achievements have **no native model** at all: `AchievementCheckingProcess` depends on an externally-injected `AchievementDatabase` (a `[Dependency]`, not simulation state) plus `CityModel`/`ActiveChallengesModel`/`ActivePlayer` — achievement tracking is a hybrid that reads simulation state but is not itself part of the deterministic simulation.

## 2. Tilemap / grid / roads / motorways / bridges / tunnels / traffic lights / roundabouts / playable bounds

- **Grid**: `TilemapModel` (GetModel-reachable) holds a **sparse `Dictionary<Vector2Int, TileModel>`** — not a dense array. Unplaced tiles simply don't exist as objects yet. World position is `coordinates * TileWidth` (`TileWidth = Fix64(2)`), via `TilemapModel.GetWorldPositionForCoordinates`. Diagonal tile-corner nodes (`TileCornerModel`) exist separately for diagonal road/roundabout connections.
- **Playable bounds**: `CityDefinition.PlayableArea` (public `RectInt`) — a MonoBehaviour-held value (NOT a simulation `IModel`; reached via the `CityDefinition`/`City` object, e.g. `testGame.StartedWithCityDefinition.PlayableArea` in our bootstrap), computed once in `CompileTilemap()`. Confirmed live: LosAngeles map reports `{xMin:-18, yMin:-10, width:37, height:21}`. `GameRules.AllowSpawningAtMapEdges`/`DoesIgnorePlayableArea()` modulate enforcement per game mode. The exact placement-validation enforcement code path was **not traced** (open question, both subagents independently flagged this).
- **Roads (model layer)**: the drivable unit is `LaneModel` (`RoadTileConnection` = input/output `RoadTileNode` direction pair through a tile), owned by a `RoadChunkModel` (one per tile/corner, representing the **intersection**, not a directional segment). Lifecycle: shared `RoadState` enum — `None/Planned/Pending/Active/Mothballed` (+ combo flags `Live`, `VisiblyActive`, `ActiveOrPending`). "Pending" = drawn but not yet wired into the live graph; "Mothballed" = temporarily replaced (e.g. during a roundabout upgrade) but recoverable. Confirmed live via `GetModels<LaneModel>()`: the LosAngeles map starts with 29 active lanes (pre-existing baked road network), growing to 31 once the first house's driveway connects.
- **Motorways**: `MotorwayModel` (confirmed `GetModels<MotorwayModel>()`-reachable in code) is a two-endpoint object (`StartCoordinates`/`EndCoordinates` + directions); the "body" between endpoints is NOT tile-by-tile modeled — length is Chebyshev distance for cost purposes only.
- **Bridges/Tunnels**: `PassageModel` (confirmed `GetModels<PassageModel>()`-reachable) wraps a plain `Passage` object (`UpgradeType` Bridge/Tunnel, `StartCoordinates`/`EndCoordinates`, `CrossingCoordinates` list, `Length`, `IsComplete`) — purely a cost/bookkeeping object layered over normal tiles; there is no separate bridge-tile model.
- **Traffic lights**: `TrafficLightModel` (nested on `RoadChunkModel.TrafficLight`) — `greenLightPairs`/`ActivePair`/`BlockedLanes`/`amberLightsOn` OBSERVABLE (visible light state); `durationOnCurrentPair`/the `WeightForPair` priority heuristic PRIVILEGED (internal AI, never shown to the player).
- **Roundabouts**: `RoundaboutModel` — fixed 2×2 footprint (`Roundabout.cs` hard-codes the 4 corner-direction offsets), `OriginCoordinates`/`CenterCoordinates` OBSERVABLE, `ReplacedConnections` (pre-roundabout road state, kept for refund accounting) PRIVILEGED.
- **Connectivity — two distinct layers, this is the most important finding for any future action space**:
  1. **Managed lane graph** (`LaneModel.InboundLanes`/`OutboundLanes`) — built incrementally as roads are placed, OBSERVABLE in effect (visible on screen) though the object-graph itself is an internal detail.
  2. **Native pathfinding graph** — `Motorways/Pathfinder.cs` (a DI service, NOT an `IModel`, NOT `ISimulation`-reachable) wraps a P/Invoke-based native C/C++ library (`ExternPathfinding.cs`: `NativeAddNode`, `NativeAddEdge`, `NativeGetPathCost`, etc.). `LaneModel.PathfindingStartNodeId`/`PathfindingEndNodeId` key into it; edge costs (`NormalLaneCostMultiplier=10`, `MothballLaneCost=100000`, turn/U-turn/carpark penalties) are entirely PRIVILEGED and **opaque to static C# analysis** — the native binary itself would need to be analyzed to fully characterize this graph. This is flagged as a hard limit of this phase's methodology, not a gap to paper over.

## 3. Houses, destinations, vehicles, demand, traffic, colors, score, failure, resources, upgrades

- **Houses** (`HouseModel`, `GetModels`-reachable): **no stable ID field** — identity is via object reference or its `tileModel.Coordinates`. `GroupIndex` (plain `int`, 0–5) IS the "color" mechanic — there is no named color enum. `waitingVehicles`/`ownedVehicles` OBSERVABLE; `DrivewayLane`/internal realignment bookkeeping PRIVILEGED. **No explicit lifecycle/connectedness field** — "connected to a destination" is entirely emergent from the demand/supply calculation, not a stored flag (flagged as an open question: the located demand formula (`CalculateSupplyContributionFromHouseToDestination`) appears to be a pure distance calc, not an explicit road-network reachability check — worth a closer look in a future phase).
- **Destinations** (`DestinationModel`, `GetModels`-reachable): also no stable ID. `DestinationType` enum = `{Destination, TrainStation, BoatTerminal}`, independent of `GroupIndex` (color). Demand is a `List<int>` of group-index "pins" (`unassignedDemand`/`waitingDemand`, both OBSERVABLE — exactly the stacked-pin icons the player sees). `contributedSupply`/`demandMultiplier`/`RequiredSupply`/`demandLevelUpTime` are all PRIVILEGED internal gating values never shown as numbers. **Failure state is destination-local, not global**: `IsOvercrowding` (`TotalDemand > MaximumDemandBeforeTimerStarts`) and `CurrentFrame.OvercrowdingTime` (the shrinking-ring countdown) are BOTH **OBSERVABLE** — directly shown in the UI, correcting an initial assumption that overload timers would be privileged. The exact trigger-to-`EndGame()` chain is: `IsOvercrowding` → (a not-yet-located process ticks `OvercrowdingTime` down) → `DestinationModel.OnOvercrowded()` → gated by `GameBehaviourModel.CanGameOver` (PRIVILEGED/debug-togglable bool) → `GameplayEventHandler.OnDestinationOvercrowded` → gated again by `GameRules.CanDestinationsOvercrowd` → `EndGame()`. Confirmed live: `GameBehaviourModel.CanGameOver = true` in our test episode (the default).
- **Vehicles** (`VehicleModel`, `GetModels`-reachable): **does** have a stable `int id` (process-lifetime counter, not persisted). `house`/`destination`/`lastVisitedDestination` references + `behaviorState` enum (`WaitingForDestination`, `DrivingToDestination`, `ParkingAtDestination`, `ParkedAtDestination`, `DrivingHome`, `RealigningDriveway`) are OBSERVABLE and together fully describe a vehicle's "trip" — **there is no separate Trip/Journey/Demand object**; a trip in progress is just a vehicle with a non-null `destination`. Position is `CurrentFrame.lane` (a `LaneModel` reference) + `CurrentFrame.distanceAlongLane` (a 1D scalar) — a third, topological coordinate representation (see §4) — converted to a world `Vector2Fixed` via the public `LaneModel.PositionAtDistance(distance)` method, confirmed live. Routing internals (`path`/`returnPath`/`repathUrgency`/congestion-"shoving" fields) are all PRIVILEGED.
- **Traffic/queues**: no dedicated Queue/Congestion model exists. The closest thing is `RoadChunkModel.inboundVehicles` (an explicit per-intersection arbitration queue, PRIVILEGED) — otherwise congestion is purely emergent from vehicle positions/speeds and the leading/blocking-vehicle fields on `VehicleModel.Frame`.
- **Score**: `ScoreModel` (GetModel-reachable) — `Score` (OBSERVABLE, +1 per delivered vehicle, confirmed `0` at t=0 and throughout our short episode since no delivery occurred yet) and a separate `EfficiencyScore`/`CurrentEfficiencyMilestone` pair (Endless-mode milestone system, decays over time, skipped while upgrades are pending).
- **Resources** (`UpgradeDatabase`/`UpgradeDatabaseModel`, GetModel-reachable): `GetAvailableUpgradeCount(UpgradeType)`/`GetTotalUpgradeCount(UpgradeType)` are the real player-facing resource counters. **Confirmed live**: the starting grant is `Concrete: 25/25`, `Bridge: 1/1`, everything else `0/0` — this is the `AwardStartingPackages()` call, and matches exactly what a human player sees as their opening resource allotment (directly OBSERVABLE, cross-checked against known game behavior).
- **Upgrades** (`UpgradeDatabaseModel`): distinct from the raw counters above — `TotalClaimedPackages`/`pendingUpgradeChoices`/`numChoicesMade` track the weekly upgrade-offer pipeline; internal scheduling/weighting arrays (`_claimedPackageCounts`, `timesUpgradePresented`) are PRIVILEGED (hidden RNG-weighting of what gets offered next).
- **`UpgradeType` enum** (the resource/upgrade kind, unrelated to house/destination color): `Concrete, Bridge, Motorway, TrafficLight, Roundabout, Tunnel, House, Destination, DoubleDestination` (+ `Count` sentinel).

## 4. Clock / week / progression

- `ClockModel.Time` is in **simulated seconds**, with `SecondsPerHour = 5/6` (so `Hour = Time / (5/6)`, `Day = Hour/24`, `Week = Hour/168`). Confirmed live: our episode's `Time` climbed from `0` to `6.2000000057742` seconds over 6.2 real seconds at normal real-time-to-sim-time ratio (expected — Normal mode's `GetClockSpeedMultiplier()` returns `1`), landing at `Hour=7` by the end.
- There is **no separately-stored "current week"** — it's always derived from time. Critically, the weekly upgrade-grant schedule rides on a SEPARATE time track, `ClockModel.ExpansionTime` (not `Time`), which can be independently paused (`expansionTimeManuallyPaused`, gated also by `GameRules.CanExpansionTimeContinue`) while the main clock keeps running — confirmed in our live snapshot, `expansionTime` tracked `time` exactly throughout (both advanced together, since neither pause flag was set).
- Three independent pause layers exist: `ISimulation.IsPaused` (simulation-wide, short-circuits the whole step), `ClockModel.isPaused` (gates `Time` only), `ClockModel.expansionTimeManuallyPaused` (gates `ExpansionTime` only, on top of the rules-level gate). All three confirmed `false` throughout our test.

## 5. Stable identifiers and the canonical coordinate system

- **No universal ID scheme.** The common base `Model<TFrame,TObserver>` has no `Id`/`Guid` field at all. Identity is ad hoc per type: `VehicleModel` has its own explicit `int id` (needed for its high-cardinality dynamic population); `HouseModel`/`DestinationModel` have **no ID field** — identity is by object reference or by `TileModel.Coordinates`. Singleton models (`CityModel`, `ClockModel`, `ScoreModel`, etc.) need no ID since `GetModel<T>()` returns the one instance.
- **Three coexisting position representations, confirmed live, each used for a different purpose**:
  1. **Discrete tile/grid coordinate**: `UnityEngine.Vector2Int` (int x,y) — used pervasively at the model level (`TileModel.Coordinates`, `HouseModel`/`DestinationModel` footprints, `MotorwayModel`/`PassageModel` endpoints). This directly contradicts an earlier-phase assumption that model code avoids `UnityEngine` types entirely — `Vector2Int` is integer-based and therefore fully deterministic, unlike `UnityEngine.Vector2/3` (float, view-layer only). **This is the canonical discrete/grid coordinate for any future environment design.**
  2. **Continuous world position**: `FixMath.Vector2Fixed`/`Vector3Fixed` (fixed-point) — e.g. `TileModel.WorldPosition`, `CityModel.startOffset`. **Canonical for continuous/rendering-adjacent position.**
  3. **Topological lane-relative position**: `VehicleModel.CurrentFrame.lane` (a `LaneModel` reference) + `distanceAlongLane` (`Fix64` scalar) — a vehicle's PRIMARY position representation is not a coordinate at all, but "which lane, how far along it," convertible to a world `Vector2Fixed` via `LaneModel.PositionAtDistance(distance)` (confirmed live, used in our snapshot's `vehicles[].worldPosition`).
- Recommendation for a future environment: use `Vector2Int` as the canonical discrete action/observation coordinate (matches the player's own mental model — "which tile"), and `Vector2Fixed` (derived, e.g. via `PositionAtDistance` for vehicles) only where continuous position is specifically needed.

## 6. The `SimulationSnapshot` structure, implemented and live-validated

Implemented as `Phase6ARunner.BuildSnapshotJson` in [`Entrypoint.cs`](../game_plugin/DoorstopBootstrap/Entrypoint.cs) — every line is a plain property/field read or a `Get<T>()`/`GetModel<T>()`/`GetModels<T>()` resolution; nothing mutates simulation or persistent state. Structure (see [`research/state_snapshot_example.json`](state_snapshot_example.json) for the full live example, 13 samples):

```
SimulationSnapshot
  sampleIndex, realElapsedSeconds
  time        { time, hour, day, week, expansionTime, isPaused, simulationIsPaused }
  score       { score, efficiencyScore, currentEfficiencyMilestone }
  bounds      { xMin, yMin, width, height }                      -- CityDefinition.PlayableArea
  resources[] { type, available, total }                         -- one entry per UpgradeType
  upgrades    { totalClaimedPackages, pendingChoices, numChoicesMade }
  failureState{ canGameOver }
  houses[]    { coord, groupIndex, waitingVehicles, ownedVehicles }
  destinations[] { footprint[], groupIndex, destinationType, unassignedDemand, waitingDemand, totalDemand, isOvercrowding, overcrowdingTime, isUpgraded }
  vehicles[]  { id, houseCoord, behaviorState, worldPosition, speed }
  roadSummary { totalLanes, planned, pending, active, mothballed, motorwayCount, passageCount, trafficLightCount, roundaboutCount }
  motorways[] { id, number, state, start, end }
  passages[]  { upgradeType, start, end, length, isComplete }
```

Deliberately NOT included (out of scope per the brief): no tensor/grid encoding of the full tilemap, no per-lane dump (summarized by state instead — a fresh city's lane count is small, but a built-up one could be large; a future phase can add a capped per-lane list if needed), no `TrafficLightModel`/`RoundaboutModel` per-instance detail (counts only, since none existed in this short episode — fields are inventoried in §2 for when a future test builds one).

Numeric formatting uses `CultureInfo.InvariantCulture` explicitly — this machine's locale is Turkish, and earlier phases' log output already showed comma decimal separators (e.g. `ClockModel.Time=0,9000000008`); without this, the JSON would have been invalid.

## 7. Live validation — one sustained episode, state evolving coherently

Ran one `GameMode.Normal` episode (seed 1001, LosAngeles map), 360 real-time ticks (~6.2 simulated seconds), capturing a snapshot every 30 ticks plus one immediately after warm-up (13 total). Full log: [`logs/instrumentation.log`](../logs/instrumentation.log); full data: [`research/state_snapshot_example.json`](state_snapshot_example.json).

**Observed coherent evolution** (cross-referencing consecutive snapshots):
- **t=0** (sample 0): 1 pre-existing `DestinationModel` (LosAngeles map's baked-in destination, footprint `[[-10,-2],[-10,-1],[-9,-2],[-9,-1]]`), 0 houses, 0 vehicles, 29 active lanes (the map's pre-built road network), `Concrete: 25/25`, `Bridge: 1/1` available (starting resource grant).
- **t≈0.7s** (sample 1): a house appears at `[-2,-5]` with 2 owned/waiting vehicles (both `WaitingForDestination`, stationary, world position `[-4.2,-10.7]`/`[-3.8,-10.7]`), and active lane count grows 29→31 (the new house's driveway).
- **t=0.7s→3.7s** (samples 1–7): stable — same house/vehicles, destination's `unassignedDemand` still 0.
- **t≈4.2s** (sample 8): the destination's `unassignedDemand` becomes `1` — the first demand "pin" appears.
- **t=4.2s→6.2s** (samples 9–12): demand stays at `1`; vehicles remain `WaitingForDestination` with unchanged position/zero speed through the end of this short episode — i.e., dispatch hadn't happened yet within this window. This is itself a useful, honestly-reported observation: a 6-second episode is long enough to see entities and demand appear, but not long enough to see a full house→destination delivery cycle complete. A future phase wanting to observe `DispatchVehiclesProcess`/vehicle movement/delivery/score increment end-to-end will need a longer sustained run (or to advance `ClockModel`/`Random` state faster) than this phase's proof required.
- `ClockModel.Time`/`ExpansionTime` tracked each other exactly throughout (both unpaused, Normal mode's ×1 speed multiplier) — `Hour` climbed from `0` to `7` as expected from the `5/6` seconds-per-hour conversion confirmed in §4.
- `GameRules` concrete type confirmed `Motorways.GameRules` (genuine Normal-mode rules, not `BackgroundGameRules`) throughout, consistent with Phase 5F.

**Manual cross-check against known game behavior**: the starting resource grant (`Concrete 25, Bridge 1`, everything else 0) and the destination's pin-based demand display both match what a human player would observe in a fresh city — these are not surprising or implausible values, which is itself a (informal) sanity check that the snapshot is reading real, meaningful state rather than nonsense.

## Persistence protection

Full before/after SHA-256 of every file in the persistent-storage directory, same pattern as every phase since 5E: `Mini Motorways.exe` and all gameplay/profile/save files (`userProfile_*`, `extendedUserProfile_*`, both `gameJournal_*.dat`, `challengeOverrides*.json`) byte-identical; the same one benign device-settings timestamp write (`_utcSaveTime`/`LastPlayedUtcTimestamp` only) recurred, as expected and previously documented in Phase 5E/5F. No progression, unlocks, achievements, statistics, or challenge state was touched.

## Hard constraints — compliance

- **Did not modify the proven process initialization/reset lifecycle** beyond the one necessary change (one longer episode instead of four short ones, to allow multiple snapshots over time) — `InitializeProcessContext`/`InitializeAtlases`/`RestoreExistingPlayerProfile`/`GetCityDefinitionForTesting` are unchanged from Phase 5F.
- **Did not design an observation space, tensor encoding, reward, Gymnasium environment, Python IPC, or action injection, and did not train anything** — the snapshot is a diagnostic/research JSON dump, explicitly not compressed into tensors.
- **Did not exclude privileged state from the inventory** — PRIVILEGED fields (internal routing costs, demand-gating formulas, RNG state, hidden scheduling weights) are catalogued throughout §2–5 specifically because they may be useful later for critics/auxiliary heads/mechanics-discovery, per the brief.
- No original game assembly patched; no corlib/System assembly replaced; no BepInEx/Harmony/MonoMod introduced; no UI/input used.
- Game process remained healthy throughout and was cleanly terminated; `doorstop_config.ini` reverted to the BepInEx resting state.

## Files changed/added this phase

- [`game_plugin/DoorstopBootstrap/Entrypoint.cs`](../game_plugin/DoorstopBootstrap/Entrypoint.cs) — `Phase5FRunner` renamed to `Phase6ARunner`; the 4-cycle determinism loop replaced with a single 360-tick sustained episode (`RunEpisodeWithSnapshots`); added `BuildSnapshotJson` and small JSON-escaping/invariant-formatting helpers (`Inv`, `JStr`, `JVec2Int`, `JBool`); added a `using FixMath;` directive for `Vector2Fixed`.
- [`research/state_snapshot_example.json`](state_snapshot_example.json) — 13 live-captured snapshots from the validation episode (new).

## Hashes

- `Mini Motorways.exe`: `305d1ea095f0ba9e844b91497283277c61f0dea9d5f7979065e78f3e7bab0ce6` — unchanged.
- `Bootstrap\MiniMotorwaysBootstrap.dll` (this phase's build): `6589e3cdfb77553813e91600495463f453628b406705e30a2878c1cecddf2502`.
- `doorstop_config.ini`: reverted to `target_assembly=BepInEx\core\BepInEx.Preloader.dll` (resting state).

## Open questions (carried forward explicitly, not resolved this phase)

1. `CityPlanModel`'s full field set/purpose was not decompiled (only its `groupHouseCounts[6]` use was confirmed incidentally).
2. The exact call sites where `HouseModel`/`DestinationModel`/`TileModel`/`RoadChunkModel`/`LaneModel`/`RailTileModel`/`BoatPathTileModel`/`CarparkModel` are dynamically `AddModel`-registered (confirmed NOT in `InitCityCommand`) were not individually traced.
3. The exact process/trigger point where `DestinationModel.OvercrowdingTime` reaching its limit calls `OnOvercrowded()` was not located (likely in a `Motorways.Processes.*` class, possibly `FailureStateProcess` given its name, not yet opened).
4. Whether house→destination "connectivity" requires actual road-network reachability or is purely a distance formula (the located `CalculateSupplyContributionFromHouseToDestination` looks like a pure distance calc) is unresolved — worth checking against `GenerateDemandProcess`/`DispatchVehiclesProcess` directly.
5. The native pathfinding graph's internal structure (node merging, multi-lane intersection representation) is opaque to C# decompilation and would need native-binary analysis or targeted runtime logging of `AddNode`/`AddEdge` calls to fully map.
6. `IntersectionDecisionDatabaseModel`, `ActiveChallengesModel`, and the exact per-`GameRules`-subclass `GetClockSpeedMultiplier()` values were not individually decompiled in depth.
7. This phase's 6.2-second episode did not run long enough to observe a full dispatch→delivery→score-increment cycle; a future phase investigating `DispatchVehiclesProcess`/`VehicleMovementProcess`/`ParkVehiclesProcess` end-to-end should plan for a longer sustained run.

## Stop condition reached

Per instruction: *"Stop once the useful native state graph is mapped and demonstrated live."* The model registry, per-entity fields with OBSERVABLE/PRIVILEGED/UNCERTAIN tags, cross-entity relationships, the three coexisting coordinate systems, and a working read-only `SimulationSnapshot` capturing coherent state evolution across 13 live samples are all delivered. No RL/Gym/action/reward work was started, per the explicit brief.
