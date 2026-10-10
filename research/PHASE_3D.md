# PHASE_3D — Minimal Bootstrap: Bypassing Harmony/MonoMod Entirely

Generated 2026-09-28. **SUCCESS** — the phase's exact success criterion was reached: a minimal managed assembly, written and controlled entirely by us, executed inside the live Mini Motorways process and logged `[MiniMotorwaysAI] bootstrap executed`, with the process ID in the log matching the actual live game process. Stopped immediately per instructions; BepInEx's Phase 3B state fully restored and re-verified afterward.

## Reframing (per the brief): the real objective was never "make a mod loader work"

Phases 3/3A-3C were all attempts to get a *general-purpose mod loader* (BepInEx, then MelonLoader) fully operational, and both hit genuine blockers deep in third-party dynamic-codegen/reflection machinery (Harmony/MonoMod's `DynamicMethodDefinition`, and Mono's own native attribute decoder, respectively). Phase 3D steps back to the actual goal — execute our own code, then read live objects — and asks whether a mod loader is even necessary for that. It is not.

## Step 1 — Is Harmony initialization fundamental to BepInEx's plugin loading? (read-only investigation, no changes made)

Decompiled `BepInEx.Preloader.PreloaderRunner` and `BepInEx.Preloader.Preloader` in full (not previously read to this depth in Phase 3B). Findings:

- `PreloaderRunner.PreloaderMain()` conditionally calls `ConsoleSetOutFix.Apply()` (the call that hit our Phase 3B blocker) only `if (Preloader.ConfigApplyRuntimePatches.Value)` — a config-toggleable boolean, default `true`.
- However, `Preloader.Run()` — called unconditionally right after — itself calls `HarmonyInteropFix.Apply()` **unconditionally**, with no config gate at all:
  ```csharp
  public static void Run() {
      InitializeHarmony();          // just sets an env var, MONOMOD_DMD_TYPE
      HarmonyInteropFix.Apply();    // Harmony.CreateAndPatchAll(...) -- UNCONDITIONAL
      ...
  }
  ```
  `HarmonyInteropFix.Apply()` (decompiled in full) calls `Harmony.CreateAndPatchAll(typeof(HarmonyInteropFix), ...)` directly — this alone triggers `HarmonyLib.AccessTools`'s static constructor and would hit the exact `DynamicMethodDefinition`/`UnverifiableCodeAttribute` blocker from Phase 3B, **regardless of any config setting**. So no, the standard `BepInEx.Preloader.dll` cannot be configured around this — Harmony initialization is baked into `Preloader.Run()` unconditionally.
- **But** — critically — the actual mechanism that makes BepInEx's chainloader run at all does **not** use Harmony. `Preloader.Run()` registers a patcher via `AssemblyPatcher.AddPatcher(new PatcherPlugin { TargetDLLs = () => new[] { "UnityEngine.CoreModule.dll" }, Patcher = PatchEntrypoint, ... })`, and `PatchEntrypoint` (decompiled in full) uses **Mono.Cecil directly** to rewrite `UnityEngine.CoreModule.dll`'s `Application` type's static constructor in memory, inserting calls to `BepInEx.Chainloader.Initialize()`/`.Start()` at the very top of it. This is a pure Cecil IL-rewrite, executed before Mono ever runs the original `.cctor` — Harmony plays no role in it.

**Answer to Step 1**: Plugin loading's *core mechanism* (Cecil-based entrypoint injection into `Application..cctor`) does **not** fundamentally require Harmony. Harmony is only pulled in by `BepInEx.Preloader.dll`'s own unconditional `HarmonyInteropFix.Apply()` call, which exists to work around assembly-loading path issues unrelated to entrypoint injection itself. Bypassing it means not using `BepInEx.Preloader.dll` as Doorstop's `target_assembly` at all — not modifying it.

## Step 2 — Can Doorstop invoke a minimal bootstrap directly, with zero BepInEx/Harmony/MonoMod? **Yes, empirically confirmed.**

Doorstop's own contract (documented, and already exercised successfully in every Phase 3/3B test) is simply: *any* managed assembly containing a `static void Doorstop.Entrypoint.Start()` method can be named directly in `doorstop_config.ini`'s `target_assembly` — there is nothing BepInEx-specific about this mechanism; BepInEx.Preloader.dll is just the assembly the official install points it at by convention.

A new project, [`game_plugin/DoorstopBootstrap/`](../game_plugin/DoorstopBootstrap/), was created with **zero references** beyond the implicit BCL — no `BepInEx.dll`, no `0Harmony.dll`, no `MonoMod.*`, no `UnityEngine.dll`. `doorstop_config.ini`'s `target_assembly` was repointed (backed up first, see Rollback) from `BepInEx\core\BepInEx.Preloader.dll` to `Bootstrap\MiniMotorwaysBootstrap.dll`.

First test (verbose Doorstop build, for full diagnostic visibility) confirmed the mechanism working immediately:
```
Opening assembly: ...\Bootstrap\MiniMotorwaysBootstrap.dll
Opened Assembly DLL (6144 bytes); opening its main image
Image opened; loading included assembly
Assembly loaded; looking for Doorstop.Entrypoint:Start
Invoking method 0157C600
Error invoking code!
Error message: System.MissingMethodException: Method not found: int System.Diagnostics.Process.get_Id()
```
This is not a Harmony/MonoMod-related failure at all — it's the same class of stripped-mscorlib gap seen throughout Phase 3B (`System.Diagnostics.Process.Id`'s getter is missing from this build), and it occurred inside **our own trivial code**, proving Doorstop had already successfully loaded and begun executing our assembly. Per this phase's hard constraint ("do not continue the mscorlib/System.Core patching experiments"), this was fixed in **our own code** instead of mscorlib: replaced `Process.GetCurrentProcess().Id` with a direct `[DllImport("kernel32.dll")] GetCurrentProcessId()` P/Invoke (a different, always-present code path that doesn't touch the stripped managed `Process` type at all).

Second test — **full success**:
```
[2026-09-28 23:08:54.860] === MiniMotorwaysBootstrap: Doorstop.Entrypoint.Start() invoked directly (no BepInEx, no Harmony, no MonoMod) ===
[2026-09-28 23:08:54.887] [MiniMotorwaysAI] bootstrap executed
[2026-09-28 23:08:54.898] Process ID: 32052
[2026-09-28 23:08:54.905] Executing assembly: MiniMotorwaysBootstrap, Version=1.0.0.0, Culture=neutral, PublicKeyToken=null
[2026-09-28 23:08:54.913] Executing assembly location: C:\Program Files (x86)\Steam\steamapps\common\Mini Motorways\Bootstrap\MiniMotorwaysBootstrap.dll
[2026-09-28 23:08:54.921] App.dll assembly identity: App, Version=0.0.0.0, Culture=neutral, PublicKeyToken=null
[2026-09-28 23:08:54.929] === MiniMotorwaysBootstrap: Start() completed without throwing ===
```

**Answer to Step 2**: Yes, unambiguously. This is the smallest possible managed entrypoint that satisfies Doorstop's contract.

## Proof this executed inside the actual live game process (not a simulated/external context)

- The PID logged by our own code (`32052`, obtained via a direct kernel32 call from inside the process) was cross-checked against `Get-Process -Id 32052` run externally at the same moment, which reported `Responding: True`, `MainWindowTitle: "Mini Motorways"`, ~602 MB working set — the same live, fully-rendered game window, confirmed from outside the process at the same time our code ran inside it.
- `Assembly.Load("App")` succeeded and returned a real assembly identity (`App, Version=0.0.0.0...`) matching the game's actual main logic assembly (documented in `ARCHITECTURE.md`) — this assembly is only loadable from inside the live game's AppDomain (it's part of `Mini Motorways_Data\Managed`, resolved via the Mono search path Doorstop set up for this exact process), not from an external tool.
- Doorstop's own diagnostic log (`doorstop_<hash>.log`, generated by the same official upstream verbose build used throughout this investigation, not authored by us) independently corroborates the exact sequence: proxy loaded → Mono domain started → our assembly opened → `Doorstop.Entrypoint:Start` located and invoked.

## Steps 3-6: not attempted, per the success criterion's explicit stop instruction

The brief's success criterion states plainly: *"The ideal Phase 3D result is: `[MiniMotorwaysAI] bootstrap executed`. Nothing more is required. If successful, STOP."* That exact marker was produced and verified live. Steps 3 (loading/referencing App.dll types beyond identity), 4 (reflection-based live-object discovery), 5 (polling thread investigation), and 6 (the MonoMod blocker itself) were **not** investigated this phase, per the explicit instruction not to proceed automatically to simulation hooks or further investigation once the bootstrap succeeded.

## Why Harmony/MonoMod was bypassed (summary)

Because the code that actually makes external code run inside this game (Doorstop's native proxy-DLL + Mono-domain-hijack mechanism) has no dependency on Harmony or MonoMod whatsoever — those only enter the picture once `BepInEx.Preloader.dll` specifically is chosen as the `target_assembly`, because *that particular assembly's own code* unconditionally initializes Harmony as part of its startup (Step 1). Pointing Doorstop at a different, minimal assembly with no such dependency sidesteps the entire problem area without needing to patch, configure around, or otherwise fight with Harmony/MonoMod at all.

## Files added

```
game_plugin/DoorstopBootstrap/
    DoorstopBootstrap.csproj   -- net472, zero package/project references
    Entrypoint.cs              -- namespace Doorstop { public static class Entrypoint { public static void Start() { ... } } }
```
Built output installed at `Bootstrap\MiniMotorwaysBootstrap.dll` in the game directory (new folder, new file — nothing overwritten).

## Hashes

- `Mini Motorways.exe`: `305d1ea095f0ba9e844b91497283277c61f0dea9d5f7979065e78f3e7bab0ce6` — **unchanged**, re-verified after the full test sequence.
- `Bootstrap\MiniMotorwaysBootstrap.dll` (final, working version): `55312d021bd3fe3350c744cb5f177474e9c08f8d71f816185730f2f06a2cb5a2`.
- All 26 BepInEx-related files (from the Phase 3C backup manifest, `backups/bepinex_state_manifest.sha256`): **all hash-identical** after this phase's revert, re-verified by diffing the full restored-file hash set against the original manifest.

## Rollback (executed and verified)

1. `doorstop_config.ini`'s `target_assembly` line reverted from `Bootstrap\MiniMotorwaysBootstrap.dll` back to `BepInEx\core\BepInEx.Preloader.dll` (the pre-edit value was independently backed up to `backups/doorstop_config.ini.bepinex_target.bak` before any change, as an extra safety net beyond the file edit itself being trivially reversible).
2. The verbose Doorstop build used for diagnostics was swapped back to the standard BepInEx-bundled `winhttp.dll`/`version.dll` (byte-identical to the Phase 3B backup, confirmed by hash).
3. Stray `doorstop_*.log`/`preloader_*.log`/`output_log.txt` diagnostic files removed.
4. `Bootstrap\MiniMotorwaysBootstrap.dll` was **left in place** (harmless, unreferenced once `target_assembly` points elsewhere again) as a ready-to-use artifact for whoever picks this up next.
5. Full verification: re-hashed every file under `BepInEx\`, `patched_corlib\`, plus the four root-level proxy/config files, and diffed the complete hash set against the original Phase 3C backup manifest — **zero differences**.
6. Final launch performed once more with the restored BepInEx configuration: confirmed fully playable (601 MB, responsive, correct window title) — i.e., this phase leaves the game in exactly the Phase 3B working state, with a proven, ready-to-reactivate minimal-bootstrap path as a new option alongside it.

## What this changes for the project going forward

We now have **two live, independently working entry points** for getting our own code to run inside the game:
1. **BepInEx (Phase 3B state)**: reaches deep into a real, feature-rich plugin framework (config system, logging, chainloader) but is currently blocked before any plugin's own code runs, by Harmony/MonoMod's dynamic-codegen dependency on a BCL member this build's `mscorlib.dll` lacks.
2. **Direct Doorstop bootstrap (this phase)**: proven to execute *our own arbitrary managed code* inside the live process right now, with no blocker at all for basic execution and reflection-based read access to loaded assemblies (`App.dll` confirmed reachable). It lacks BepInEx's conveniences (no Harmony-based hooking, no plugin framework), but for the stated next objective — read-only access to `MotorwaysGame`/`ISimulation`/`Simulation`/`Clock`/`CityModel` via reflection rather than Harmony patches — this path has no known blocker yet and is the more promising one to continue from.
