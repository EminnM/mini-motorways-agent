# LIVE_INSTRUMENTATION — Phase 3 Results

Generated 2026-09-28. Records what was proven to work live in the real game process, exactly where the chain currently breaks, and the diagnostic evidence for each claim. Steps 5-8 of the Phase 3 brief (first read-only hook, simulation lifecycle observation, live DI access, first state proof) were **not reached** — see Current Blocker below. Everything below Steps 1-2 is honest reporting of a partial result, not a success being overstated.

## STEP 3 — Hello-world plugin (built, not yet proven live)

Source: [`game_plugin/HelloWorldPlugin/Plugin.cs`](../game_plugin/HelloWorldPlugin/Plugin.cs), project [`game_plugin/HelloWorldPlugin/HelloWorldPlugin.csproj`](../game_plugin/HelloWorldPlugin/HelloWorldPlugin.csproj).

- A minimal `BepInPlugin` (`BaseUnityPlugin` subclass) whose `Awake()` does nothing but log: plugin version, process ID, `Application.unityVersion`, product name/version, and `App.dll`'s assembly identity (via `Assembly.Load("App")`), to a dedicated file at `C:\Users\TKA\Desktop\mini moto\logs\instrumentation.log` (separate from BepInEx's own `LogOutput.log`), plus a line through BepInEx's own `Logger.LogInfo`.
- Targets `net472` (not `net35` — the SDK-style project could not resolve `net35` reference assemblies in this environment; `net472` compiled cleanly and is a supported BepInEx 5 target framework).
- Builds cleanly: `dotnet build -c Release` → `bin\Release\net472\MiniMotorwaysInstrumentation.dll`, 0 warnings, 0 errors.
- Installed at `BepInEx\plugins\MiniMotorwaysInstrumentation.dll` in the live game directory.
- **Has not yet executed** — see Current Blocker. No gameplay hooks (Steps 5-8) have been written yet, per the brief's explicit instruction not to proceed past hello-world until it's confirmed live.

## STEP 4 — Verify live execution: PARTIAL, chain traced precisely via Doorstop's own diagnostic log

Rather than guessing, the exact point of failure was captured directly from Doorstop's own runtime trace (obtained by temporarily swapping in the upstream "verbose" Doorstop build, `doorstop_win_verbose_4.5.0.zip` from the official `NeighTools/UnityDoorstop` releases, which writes a `doorstop_<hash>.log` per launch — this file is not present in normal operation and was deleted again before leaving the install in its final state). This is the actual, literal execution trace from the live process, not an inference:

```
Doorstop started!
...
Proxy loaded
CONFIG: ... [config parsed correctly, target_assembly resolved to BepInEx\core\BepInEx.Preloader.dll]
Doorstop enabled!
Installing IAT hooks
Hooks installed, marking DOORSTOP_DISALBE = TRUE
Got mono_jit_init_version at 729C0000
Loaded all runtime functions

Starting mono domain "Unity Root Domain"
Runtime version: v4.0.30319
...
Opening assembly: ...\BepInEx\core\BepInEx.Preloader.dll
Opened Assembly DLL (43008 bytes); opening its main image
Image opened; loading included assembly
Assembly loaded; looking for Doorstop.Entrypoint:Start
Invoking method 00C0A4C8
Error invoking code!
Error message: System.MissingMethodException: Method not found: void System.IO.File.WriteAllText(string,string)

Done
```

This confirms, with direct evidence rather than assumption, that the following links in the required chain **do** work:
```
Mini Motorways starts                                    CONFIRMED
        ↓
BepInEx's Doorstop proxy loads (WINHTTP.dll/VERSION.dll)  CONFIRMED (also independently confirmed via live process module enumeration, see BEPINEX_INSTALL.md)
        ↓
Doorstop installs IAT hooks, obtains mono_jit_init_version CONFIRMED
        ↓
Doorstop starts a Mono domain, loads BepInEx.Preloader.dll CONFIRMED
        ↓
Doorstop locates and invokes Doorstop.Entrypoint:Start     CONFIRMED (invocation happens; the invoked code itself then throws)
        ↓
BepInEx.Preloader's own startup code                       FAILS — MissingMethodException before it can chainload plugins
        ↓
our plugin loads / our managed code executes                NOT YET REACHED
```

So the injection mechanism itself — the hardest, most uncertain part of Phase 3 — is proven to work end-to-end into the live Mono domain of the real running game. What is not yet proven is our own plugin code executing, because BepInEx's own preloader (not our code) fails first.

## Root cause (CONFIRMED, not guessed)

`Mini Motorways_Data\Managed\mscorlib.dll` is a Unity-linker-stripped build — the game's own code never calls `File.WriteAllText(string,string)`, so Unity's managed-stripping build step removed that specific overload from the shipped assembly. `BepInEx.Preloader.dll` (version 5.4.23.5) calls that exact overload during its own startup (almost certainly for its own preloader log file), and Mono's method-resolution fails hard with `MissingMethodException` since the method genuinely does not exist in the loaded `mscorlib.dll`, unrelated to anything about our plugin.

## Attempted fix and its outcome (both tried live, both reverted)

The standard, BepInEx-documented remedy for exactly this class of failure is Doorstop's `dll_search_path_override` config option, pointed at a folder of **unstripped** framework assemblies for the exact Unity version, fetched from BepInEx's own official corlib mirror (`https://unity.bepinex.dev/corlibs/2022.3.62.zip` — an exact patch-version match for this game's `2022.3.62f2`).

- **Attempt A** — full unstripped bundle (`mscorlib.dll`, `System.dll`, and 13 other BCL assemblies) in `unstripped_corlib\`, referenced via `dll_search_path_override`: fixed the `File.WriteAllText` error, but the *unstripped* `System.dll` then threw a **new** exception during `BepInEx.Preloader`'s startup — `TypeInitializationException` → `DllNotFoundException: System.Native`, from `TimeZoneInfo.TryGetLocalTzFile` trying to use a Unix-style native interop shim (`Interop+Sys`) that has no Windows counterpart in this Mono runtime. This is worse: the **whole game process crashes** (confirmed via `UnityCrashHandler32.exe` spawning and the main process's memory staying at ~320 KB instead of reaching the normal ~600 MB, and never becoming responsive), rather than the previous behavior of doorstop failing gracefully and the game still launching normally afterward.
- **Attempt B** — narrowed to *only* the unstripped `mscorlib.dll` (removing the other 14 files, on the theory that `System.dll` was the specific culprit): **same crash, same stack trace**. The `TimeZoneInfo`/`System.Native` bug is inside `mscorlib.dll` itself, not `System.dll` — confirmed by elimination, not assumption.
- **Reverted**: `dll_search_path_override` was set back to blank. The `unstripped_corlib\mscorlib.dll` file itself was left on disk (harmless, unreferenced) in case a future attempt wants to try a more surgical fix (e.g. patching only the one missing method into the game's own `mscorlib.dll` via Mono.Cecil, rather than swapping the whole assembly).

This is a genuine, evidence-backed STOP condition per the brief ("if a hook causes crashes, roll back the change, record what changed / what crashed / logs, revert to last known good") — not a case of guessing or hammering blindly. Both the crash and the revert were verified live, each time checking the process actually reached a healthy ~596 MB responsive window before moving on.

## STEP 5-8: not reached

Since BepInEx's own preloader does not yet complete successfully, no plugin code — ours or anyone else's — has executed in the live process yet. Consequently:
- No low-risk read-only hook (`AppRuntime.Update`/`MotorwaysGame.Tick`) has been installed.
- No simulation lifecycle observation has been attempted.
- No live DI/`ISimulation` access has been attempted.
- No live state has been read or verified.

All of these remain exactly where the static reconnaissance (`ARCHITECTURE.md`) left them: identified as plausible targets, not yet exercised against the live process.

## Game stability / safety compliance

- The exe hash was re-verified as unchanged (`305d1ea095f0ba9e844b91497283277c61f0dea9d5f7979065e78f3e7bab0ce6`) after all experimentation.
- No gameplay code was patched; no RNG, spawn logic, roads, or save files were touched (per the brief's explicit prohibition list) — the entire chain up to the current blocker is loader/preloader-level, before any hook could even be installed.
- The install was deliberately left in the **safer of the two known states**: doorstop present but its preloader fails gracefully (game fully playable, instrumentation inert) rather than the crashing state.
- Every launch performed was manual, one at a time, with explicit process/log verification between attempts — no automated repeated-crash loop occurred.

## Update: Phase 3B (see [`PHASE_3B.md`](PHASE_3B.md) for full detail)

Option 1 below was pursued and succeeded far beyond its original narrow scope: six individually-verified surgical Mono.Cecil patches to a **copy** of `mscorlib.dll` (never the original file) took the preloader chain from "crashes on its very first error-logging attempt" all the way through BepInEx's entire configuration subsystem and into Harmony's own static initialization — `BepInEx\config\BepInEx.cfg` is now created and correctly re-read across launches. The new blocker is deep inside `MonoMod.Utils.DynamicMethodDefinition` (Harmony's dynamic-codegen dependency), a fundamentally different and larger problem than the original single missing method. A related attempt to fix an intermediate `IGrouping<TKey,TElement>.Key` gap via the same technique revealed a real limitation: Mono's VTable builder rejects interface members added post-hoc via metadata patching (confirmed via a `TypeLoadException` at type-load time), even though six consecutive plain-method patches worked flawlessly. Both risky attempts (a wholesale unstripped `System.Core.dll` swap, and the narrower interface patch) were tested once each and reverted per the crash-handling protocol. Plugin execution (this document's original success criterion) has still not been reached — the current blocker is now Harmony/MonoMod's own dynamic codegen, not anything in BepInEx's own code path.

Options 2 and 3 below remain unstarted and are the live choices going forward.

## Update: Phase 3C (see [`PHASE_3C.md`](PHASE_3C.md) for full detail)

Option 3 below was pursued: MelonLoader 0.7.3 (x86, official release) was installed cleanly (BepInEx fully removed and backed up first) and tested with a minimal hello-world plugin. **Result: worse than BepInEx, not better.** MelonLoader's own native bootstrap (`MelonLoader.Bootstrap` / `BootstrapInterop.Initialize`) crashed the entire game process with `System.Reflection.CustomAttributeFormatException: Could not find a field with name CharSet`, originating inside Mono's own native custom-attribute decoder (`System.MonoCustomAttrs.GetCustomAttributesInternal`, a `wrapper managed-to-native` internal call) — not managed code, so not patchable with the Mono.Cecil technique that worked for BepInEx's six gaps. Unlike every BepInEx failure so far, this one is a **hard crash**: the game process fully exits rather than continuing to run normally. Our plugin never loaded; the `[MiniMotorwaysAI] Plugin loaded successfully` marker was never produced. MelonLoader was fully removed and BepInEx restored — verified byte-for-byte identical to its Phase 3B state via hash comparison of all 26 files, and the game re-confirmed fully playable afterward.

**Current standing**: BepInEx remains the further-progressed, safer path (graceful degradation, game stays playable, reaches deep into Harmony/MonoMod init) versus MelonLoader (hard crash before plugin loading even begins). Plugin execution — this document's core success criterion — has still not been achieved by either loader.

## Update: Phase 3D — SUCCESS (see [`PHASE_3D.md`](PHASE_3D.md) for full detail)

This document's core success criterion — our own managed code executing inside the live game process, with proof — **has now been achieved**, via a different route than Phases 3-3C attempted. Rather than continuing to fight BepInEx's or MelonLoader's own internal Harmony/MonoMod dependencies, Phase 3D asked whether a general-purpose mod loader was even necessary, and found it is not: Doorstop's injection mechanism (proven working since Phase 3) can invoke *any* assembly with a `Doorstop.Entrypoint.Start()` method directly, with zero dependency on BepInEx, Harmony, or MonoMod. A minimal bootstrap assembly with no such references was built and pointed to via `doorstop_config.ini`'s `target_assembly`, and it executed successfully inside the live process:

```
[MiniMotorwaysAI] bootstrap executed
```

with the process ID logged from inside the process cross-verified against `Get-Process` observing the same live, responsive game window from outside it, and `Assembly.Load("App")` (the game's own main logic assembly) succeeding from within that context. Per the phase's explicit stop instruction, no further investigation (live object access, polling threads, etc.) was attempted this phase — those are the natural next steps, now unblocked. BepInEx's Phase 3B state was fully restored and re-verified afterward; both paths remain available.

## Next step options (not started; awaiting direction)

1. ~~Attempt a surgical binary patch~~ — **done, see Phase 3B above.** Further continuation would mean characterizing `MonoMod.Utils.DynamicMethodDefinition`'s full `System.Reflection.Emit` usage surface one missing member at a time, which is a substantially larger undertaking than the six patches applied so far.
2. Check whether a newer/different BepInEx build (or a differently-configured Preloader, e.g. disabling whatever preloader feature calls `File.WriteAllText`) avoids the problematic call path entirely. Note: the *original* File.WriteAllText call path is now fixed; this option would need to target avoiding Harmony's `ConsoleSetOutFix` runtime fix instead, if that's even configurable (not yet investigated — see PHASE_3B.md's Step 2 discussion).
3. ~~Try MelonLoader~~ — **done, see Phase 3C above. Result: worse than BepInEx (hard crash vs. graceful degradation), not currently viable without a harder native-level fix.**
4. Continue the BepInEx surgical-patch approach against `MonoMod.Utils.DynamicMethodDefinition`'s `System.Reflection.Emit` surface, now that MelonLoader has been ruled out as an easier alternative.
