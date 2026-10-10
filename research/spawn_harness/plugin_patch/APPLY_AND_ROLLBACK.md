# DIAG_FORK — exact change, affected files, backup/rollback, equivalence evidence (for the conditional approval; NOTHING applied yet)

## Exact patch
1. `game_plugin/DoorstopBootstrap/Entrypoint.cs` — two insertions, both additive:
   * In the IPC dispatch switch (currently lines 4083–4094), after `case "DIAG_SPAWN": return HandleIpcDiagSpawn(obj);` add
     `case "DIAG_FORK": return HandleIpcDiagFork(obj);`
   * Immediately before `private string HandleIpcDiagSnapshot(` (currently line 5180) insert the two methods in `DIAG_FORK.cs` verbatim (`RunForkBranch`, `HandleIpcDiagFork`). They call only existing members: `TearDownIpcGameForDiag`, `RestoreGameFromBytes`, `CaptureExtras`, `ExportSim`, `SpawnKeys`, `ApplyAction`, `AdvanceSimulation`, `IsTerminal`, `ActionIndexToDescriptor`, `StateFingerprint`, `ScriptedDiagStep`, `JStr`, `JBool`, `Inv`, fields `_ipcGame`, `_ipcActionBounds`, `_ipcDecisionDeltaSeconds`, `_diagRestoreMode`, `_diagUseExtras`. No existing line is modified; RESET/STEP/HELLO paths are untouched, so training/evaluation behaviour cannot change unless the new op is called.
2. Build: `cd game_plugin/DoorstopBootstrap && dotnet build -c Release` → `bin/Release/net472/MiniMotorwaysBootstrap.dll` (dotnet SDK 9.0.305 present; same csproj as the deployed build).
3. Deploy: copy that DLL to `C:\Program Files (x86)\Steam\steamapps\common\Mini Motorways\Bootstrap\MiniMotorwaysBootstrap.dll` while no game instance is running.

## Affected files
`game_plugin/DoorstopBootstrap/Entrypoint.cs` (source), `game_plugin/DoorstopBootstrap/bin|obj/Release/net472/*` (build output), the deployed DLL above. Nothing in `python/`, `experiments/`, `research/` (apart from new files under `research/spawn_harness/`).

## Backup / rollback
* Before applying: copy `Entrypoint.cs` → `research/spawn_harness/backup/Entrypoint.cs.pre_fork`; copy the deployed DLL → `research/spawn_harness/backup/MiniMotorwaysBootstrap.dll.deployed_pre_fork` and record both SHA-256 values in `backup/SHA256.txt`. (The deployed and built DLLs are currently byte-identical, sha256 7ba7147eceb1fdee…, 191,488 bytes, built 2026-10-08 00:21.)
* Rollback = copy the two backups back and verify the SHA-256; takes seconds; no other state is involved.

## Evidence that the op preserves native state equivalence (to be produced after deployment, before any pilot)
1. **Regression of the normal path:** re-run `research/s21_native_check/code/live_smoke.py --seed-idx 0 --decisions 150` with the new DLL; the per-decision observation hashes must equal those logged by the S22 V0 episode of seed 500000 for the first 150 decisions (same deterministic policy) → RESET/STEP unchanged.
2. **Restore fidelity:** `DIAG_FORK` with branches `base`, `base2` (both empty) at anchors 30 and 90 on 3 seeds: `harness.equivalent_initial_state` must report identical spawns, PRNG at every checkpoint and score (the existing DIAG_SPAWN machinery gave 19/19 on the stopped pilot; `prngAtRestore` must also equal `prngAtSnapshot`).
3. **Fork vs straight play:** the `base` branch's spawn sequence and score at checkpoints must equal a straight STEP continuation (WAIT actions) of the same episode from the same anchor (one seed), within the known residual-divergence caveat (two unexplained cases in `research/cf_probe/RESULTS.md`).
4. **Edit accounting:** a `block` branch with 3 legal ADDs must show `editsAccepted = 3` and `concreteAtRestore − concreteAfterEdits = 3`.
If 1–3 fail, roll back and report; the pilot is not started.
