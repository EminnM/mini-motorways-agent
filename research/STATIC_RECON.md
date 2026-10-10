# STATIC_RECON — Phase 2 Methodology

Generated 2026-09-28. Companion to [`ARCHITECTURE.md`](ARCHITECTURE.md) — this file covers *how* the recon was done, tool versions, and a confidence-graded index of what was and wasn't examined, so the method is repeatable and auditable.

## Tooling

- **Decompiler**: [ILSpy](https://github.com/icsharpcode/ILSpy)'s official command-line frontend, `ilspycmd`, installed as a per-user `dotnet tool`:
  ```
  dotnet tool install -g ilspycmd --version 9.1.0.7988
  ```
  (`dotnet tool install -g ilspycmd` with no version pinned to the .NET 9 SDK's default failed with a package-metadata error; version `9.1.0.7988` — matching the installed .NET 9.0.305 SDK's major version — installed and ran cleanly. A newer version, `11.1.0.9782`, is available upstream but was not used, to avoid an unplanned toolchain change mid-recon.)
- **Why ilspycmd over dnSpy/other GUI decompilers**: purely read-only, scriptable, no GUI dependency in this environment, and critically — **it never loads or executes the target assembly**, it only parses IL metadata and reconstructs C#-like source text. This satisfies the "read-only analysis only" constraint given for this phase; no `Assembly.Load`/reflection-execution path was used anywhere.
- **No modification of any file under the Mini Motorways install directory occurred.** All decompiler output was written under `research/static_recon/` in the workspace, never back into the Steam install path.
- **Command pattern used throughout**:
  ```bash
  ilspycmd -l c "<Managed>/App.dll" > app_classes.txt          # full class list
  ilspycmd -l i "<Managed>/App.dll" > app_interfaces.txt       # full interface list
  ilspycmd -t "<Fully.Qualified.TypeName>" "<Managed>/App.dll" -r "<Managed>" > types/<Name>.cs
  ```
  `-r "<Managed>"` (reference path) was passed on all per-type decompiles so the decompiler could resolve cross-assembly types (e.g. `FixMath.Fix64`, `UnityEngine.Vector2Int`) for cleaner output; it does not affect correctness of the read, only formatting/resolution quality.

## What was actually decompiled this phase (full source read)

All under `research/static_recon/types/`:

| File | Type | Lines | Read in full? |
|---|---|---|---|
| `Motorways_Processes_BuildingSpawningProcess.cs` | `Motorways.Processes.BuildingSpawningProcess` | 1182 | Yes |
| `Motorways_BuildingPlacer.cs` | `Motorways.BuildingPlacer` | 990 | Yes |
| `Motorways_Processes_GenerateDemandProcess.cs` | `Motorways.Processes.GenerateDemandProcess` | 436 | Yes |
| `Motorways_City.cs` | `Motorways.City` | 425 | Yes |
| `Motorways_Models_CityPlanModel.cs` | `Motorways.Models.CityPlanModel` | 386 | Not yet read in full — decompiled, skim only |
| `Random.cs` | top-level `Random` (static helper) | 171 | Yes |
| `AppRuntime.cs` | top-level `AppRuntime` (MonoBehaviour) | 170 | Yes |
| `PseudorandomGenerator.cs` | top-level `PseudorandomGenerator` | 93 | Yes |
| `Motorways_AddRoadEdit.cs` | `Motorways.AddRoadEdit` | 110 | Decompiled, not transcribed in detail |
| `Motorways_Models_ClockModel.cs` | `Motorways.Models.ClockModel` | 119 | Yes |
| `Motorways_Processes_ClockProcess.cs` | `Motorways.Processes.ClockProcess` | 39 | Yes |
| `Motorways_CitySpawningLayerData.cs` | `Motorways.CitySpawningLayerData` | 36 | Yes |
| `Motorways_BuildingSpawningTileWeights.cs` | `Motorways.BuildingSpawningTileWeights` | 38 | Yes |
| `Motorways_DensityTile.cs` | `Motorways.DensityTile` | 24 | Yes |
| `Motorways_SpawnRampParameters.cs` | `Motorways.SpawnRampParameters` | 14 | Yes |
| `Server_ISimulation.cs` | `Server.ISimulation` (interface) | 40 | Yes |
| `Server_IProcess.cs` | `Server.IProcess` (interface) | 13 | Yes |
| `Server_Command.cs` | `Server.Command` (abstract base) | 33 | Yes |
| `ProcessInputEventCommand.cs` | top-level `ProcessInputEventCommand` | 35 | Yes |
| `Motorways_Commands_EditTileCommand.cs` | `Motorways.Commands.EditTileCommand` | 49 | Yes |
| `Motorways_TileEdit.cs` | `Motorways.TileEdit` (abstract base) | 68 | Yes |
| `Motorways_MotorwaysGame.cs` (partial, `Tick()`/`TickDuringTransition()` region) | `Motorways.MotorwaysGame` | 1120 total, ~95 read | Partial — only the `Tick()` method region (lines 839–927) |

`app_classes.txt` (1497 entries) and `app_interfaces.txt` were both generated in full and grepped extensively; not every line was individually read, but the full text was searched for every concept category listed in the task brief (Game, Simulation, Map, City, Building, House, Destination, Road, Spawner, Traffic, Vehicle, Resource, Score, RNG, Save, Load, Week, Day).

## Method actually followed (vs. the prescribed broad→narrow approach)

1. **Broad**: dumped full class list (`-l c`) and interface list (`-l i`) for `App.dll` — 1497 classes, confirmed namespace structure (`Motorways.*` dominant), confirmed the `AOTTarget_*` naming convention as a free index into real type names.
2. **Grep-narrowed**: searched the class dump for every concept in the brief's checklist (`City|Simulation|Clock|Tile|Road|House|Destination|Vehicle|Score|Random|Rng|Save|Spawn|Traffic|Motorway|Resource|Week|Passenger|Bridge|Carpark|Lane`), which is what surfaced `Motorways.Processes.BuildingSpawningProcess` and `Motorways.Processes.GenerateDemandProcess` as the clear highest-value spawn-related targets, and `Motorways.Models.*`/`Motorways.Commands.*`/`Motorways.*Edit` as the state/action surface.
3. **Targeted decompile**: pulled the ~20 types in the table above individually via `-t`, prioritized by: (a) direct name match to the brief's "especially investigate" list, (b) follow-the-reference — e.g. `BuildingSpawningProcess` referencing `BuildingPlacer`, `CityPlanModel`, `DemandModel`, `ClockModel` as `[Dependency]` fields is what pulled those in next, matching the brief's explicit instruction to "follow actual references/call graphs" rather than stop at name mismatches.
4. **Cross-checked against runtime evidence where possible**: none yet — this phase is 100% static analysis; no runtime verification (comparing decompiled logic against actual observed game behavior) has been performed, since no instrumentation exists yet (that's Phase 3+). All CONFIRMED gradings in `ARCHITECTURE.md` mean "confirmed by reading the actual decompiled source," not "confirmed against live game behavior."

## Confidence-grading discipline applied

Per the brief's explicit instruction not to infer behavior from names alone:
- Every claim in `ARCHITECTURE.md` is backed by an inline code excerpt or a direct paraphrase of decompiled logic, not a name-based guess.
- Places where only the class *name* is known (the ~20 `Processes.*` types not individually decompiled, `GameRules` subclasses, `Motorways.Commands.InitCityCommand`/`SnapshotCommand`) are explicitly listed under "What remains UNKNOWN" in `ARCHITECTURE.md` rather than described as if understood.
- The RNG section (§5) is the clearest example of the brief's specific warning being honored: `FixedMath.dll`'s presence does *not* imply blanket determinism — the investigation found and distinguished two separate RNG systems (`PseudorandomGenerator`, deterministic; `Random`, wall-clock-seeded `System.Random`) rather than assuming determinism from the library's presence alone.

## Scope boundaries honored this phase

- No BepInEx installed, no hooks created, no `.exe`/`.dll` in the Steam install directory touched (verified: only new files created were under `research/` and `research/static_recon/` in the separate workspace directory, `C:\Users\TKA\Desktop\mini moto\`).
- No RL environment or simulator code written.
- `ilspycmd` itself runs entirely outside the game process — it was never launched, and the game was not running during this phase (consistent with `INSTALLATION.md`'s "not running" finding, unchanged).

## Suggested next static-recon targets (not yet done, for a future Phase 2 continuation or Phase 6)

- `Motorways.GameRules` and its subclasses (`EndlessGameRules`, `ExpertGameRules`, etc.) — needed to know which game modes actually exercise the road-demolition-for-spawn escape hatch, and what `GetClockSpeedMultiplier()`/`AllowSpawningAtMapEdges` etc. actually return per mode.
- `Motorways.Commands.InitCityCommand` and `Motorways.Commands.SnapshotCommand` — directly relevant to Phase 9 reset/determinism.
- `Motorways.Models.RoadChunkModel`, `Motorways.Models.LaneModel`, `Motorways.TilePathfinder`/`Motorways.Pathfinder`, `Motorways.Pathfinding.ExternPathfinding` (the "Extern" name is worth checking — could indicate a native/non-managed pathfinding call, which would be a determinism and instrumentation wrinkle worth flagging early) — needed for a full ROAD/traffic path writeup to the same depth as the spawn path.
- `AppContainer`/`MotorwaysAppContainer`/the `Factory`/`Factory.Pools` DI framework itself — needed before Phase 3 can know how to reach a live `IScope`/`ISimulation` instance from an injected BepInEx plugin.
- Full field-by-field read of `Motorways.Tile`/`RoadTileNode`/`RoadState` — currently only understood at the "what it's used for" level via call sites in `BuildingPlacer`, not its own definition.
