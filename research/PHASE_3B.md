# PHASE_3B — Surgical BCL Patching for BepInEx Compatibility

Generated 2026-09-28. Continues directly from [`LIVE_INSTRUMENTATION.md`](LIVE_INSTRUMENTATION.md)'s blocker. This phase's success criterion (a plugin producing an unmistakable live marker) was **not** reached, but substantial, precisely bounded progress was made and is recorded here in full, per the instruction to follow the crash-handling protocol strictly rather than push through blindly.

## Method

A dedicated Mono.Cecil-based patcher tool, [`tools/mscorlib_patcher/`](../tools/mscorlib_patcher/), was built to make the smallest possible additive change to a **staged copy** of `mscorlib.dll`, never the game's original file. Workflow, repeated for every one of the six fixes below:

1. Decompile the failing method's declaring type in the game's actual shipped assembly to confirm the target member is *genuinely* absent (not just under a different overload) — evidence over assumption, per the mission's own operating principle.
2. Identify the smallest correct fix: either a thin forward to an existing overload that *is* present, or (for `Module.GetPEKind`) a constant stub whose correctness is justified by this exact game's confirmed architecture (see [`INSTALLATION.md`](INSTALLATION.md) — always x86, never ARM).
3. Emit the new member via Cecil directly into a copy of `mscorlib.dll`, decompile the *patched* copy and visually confirm the added method's IL/decompiled shape exactly matches intent.
4. Deploy only the patched copy, via Doorstop's `dll_search_path_override` config option pointed at a new `patched_corlib\` folder in the game directory — **the original `Mini Motorways_Data\Managed\mscorlib.dll` is never touched on disk**, confirmed by SHA-256 before and after every test in this phase.
5. Launch the game exactly once, check process health (`Get-Process` — window title, `Responding`, working set) and the resulting error (if any), then stop and analyze before making the next change.

## Pre-patch verification (STEP 1, as instructed)

- `Mini Motorways_Data\Managed\mscorlib.dll` SHA-256: `fe9f5c7a02c89e71d45a31ecadf9303a65c85828e625e0f2ecb13ee2ce6f75f2` — backed up byte-identical to `backups/mscorlib_original/mscorlib.dll.bak` before any inspection.
- Assembly identity: `mscorlib, Version=4.0.0.0, Culture=neutral, PublicKeyToken=b77a5c561934e089` (confirmed via `[System.Reflection.AssemblyName]::GetAssemblyName(...)`).
- `System.IO.File` in this exact assembly has 28 methods including `ReadAllText`, `ReadAllBytes`, `WriteAllBytes`, `WriteAllLines` — but genuinely **no** `WriteAllText` overload at all (confirmed by full decompile of the type, not a grep miss).
- This confirmed the original Phase 3 finding was a real, specific stripped-method gap, not a broader missing-type or missing-assembly problem — exactly the kind of situation the brief anticipated as "surgical patch viable."

## Patches applied, in the order they were discovered (each one only visible after fixing the previous one)

All six live inside `System.IO.File`, `System.Reflection.Module`, `System.IO.StreamWriter`, `System.Console`, and `System.Reflection.AmbiguousMatchException` — all in `mscorlib.dll`. Each was individually verified present-in-output via decompilation before deployment, and each was tested live exactly once before moving to the next.

| # | Missing member | Fix | Why it's correct, not just convenient |
|---|---|---|---|
| 1 | `File.WriteAllText(string, string)` | Forwards to the existing `File.WriteAllBytes(string, byte[])` via `Encoding.UTF8.GetBytes` | Byte-for-byte what real .NET's overload chain reduces to |
| 2 | `Module.GetPEKind(out PortableExecutableKinds, out ImageFileMachine)` | Constant stub: `ILOnly` / `I386` | The only caller (`BepInEx.Preloader.PlatformUtils.SetPlatform`, decompiled and read in full) uses the result **solely** to test `== ImageFileMachine.ARM`. This game and every one of its assemblies is confirmed x86 (`INSTALLATION.md`), never ARM — the constant is the true answer for this exact toolchain, not an approximation |
| 3 | `StreamWriter(string, bool, Encoding)` | Forwards to the existing `StreamWriter(string, bool, Encoding, int)` with buffer size `1024` | The real .NET 3-arg constructor is itself defined as exactly this forward with the framework's default buffer size |
| 4 | `Console.Out` (`get_Out`) | Returns the existing internal `stdout` field | `Console.Error`'s getter (`=> stdout`-equivalent for stderr) was left un-stripped in this build and follows the identical pattern; `stdout` itself was already present and used internally by `Console.WriteLine` etc. |
| 5 | `File.ReadAllLines(string)` | `ReadAllText(path).Split(['\r','\n'], StringSplitOptions.RemoveEmptyEntries)` | **Documented simplification**, not a faithful reimplementation: real `ReadAllLines` preserves blank lines as empty-string array entries; this version drops them. Deliberately accepted because the only caller is BepInEx's own line-oriented config parser reading back a file BepInEx itself wrote, where blank lines carry no meaning |
| 6 | `AmbiguousMatchException(string, Exception)` | Chains to `SystemException(string, Exception)`, then sets the same `HResult` (`-2147475171`) the class's other two constructors already set | Mirrors an existing, visible pattern in the same class rather than inventing one |

Each patch was additive only — every run's tool output includes an explicit method/property count check (e.g. "`System.IO.File` methods: 28 → 29 → 30") confirming exactly one new member per patch and zero existing members altered. Full tool source: [`tools/mscorlib_patcher/Program.cs`](../tools/mscorlib_patcher/Program.cs).

## Result of the six mscorlib patches: real, substantial forward progress

Each fix was deployed and the game launched **exactly once** per the brief's instruction, health-checked (`Get-Process`), then the resulting `preloader_<timestamp>.log` (now writable thanks to patch #1) was read for the next blocker. The failure point moved, one launch at a time, from:

```
Doorstop.Entrypoint.Start()                          [original: crashed writing its own error log]
  → BepInEx.Preloader.PlatformUtils.SetPlatform()     [fixed by patch 2]
  → BepInEx.Configuration.ConfigFile..ctor (write)    [fixed by patches 1, 3]
  → BepInEx.Preloader.PreloaderRunner.PreloaderMain() [fixed by patch 4]
  → BepInEx.Configuration.ConfigFile..ctor (re-read)  [fixed by patch 5]
  → HarmonyLib.AccessTools..cctor()                   [fixed by patch 6]
  → HarmonyLib's MonoMod.Utils.DynamicMethodDefinition -- see Blocker below
```

That is: **BepInEx's entire configuration subsystem now initializes correctly**, and the failure point is now deep inside Harmony's own dynamic-method-generation internals (`MonoMod.Utils.DynamicMethodDefinition`) — well past anything in BepInEx's own code. `BepInEx\config\BepInEx.cfg` is now created and correctly re-read on subsequent launches. This is the deepest point reached in either phase.

Every one of these six launches, and the current resting state, kept the game fully playable (~600 MB working set, responsive window, confirmed via `Get-Process`) — the graceful-failure behavior from the end of Phase 3 was preserved throughout.

## Current blocker (why Steps 4-8 were not reached)

```
System.TypeLoadException: Could not resolve type with token 01000011 from typeref
(expected class 'System.Security.UnverifiableCodeAttribute' in assembly 'mscorlib, Version=2.0.0.0, ...')
  at MonoMod.Utils.DynamicMethodDefinition..cctor()
  at HarmonyLib.MethodInvoker.GetHandler(...)
  at HarmonyLib.AccessTools..cctor()
```

`System.Security.UnverifiableCodeAttribute` (an attribute type Reflection.Emit applies to dynamically-generated assemblies) is entirely absent from this stripped `mscorlib.dll` — not just one overload, the whole type. This is inside MonoMod's dynamic-codegen machinery, used by Harmony to build fast method detours/invokers at runtime via `System.Reflection.Emit`. This is qualitatively different from patches 1-6: those were self-contained missing methods with a clear, narrow forward target; this sits inside a much larger, more intricate subsystem (dynamic assembly/type emission) that plausibly depends on many more BCL members not yet enumerated, and getting it fully working would mean characterizing MonoMod's entire Reflection.Emit usage surface rather than one method at a time.

## A second, riskier attempt and why it was reverted (crash-handling protocol followed)

To resolve the *earlier* `IGrouping<TKey,TElement>.Key` gap (hit between patches 5 and 6, inside `BepInEx.Configuration.ConfigFile`'s own LINQ usage), two approaches were tried:

1. **Wholesale swap** of the official unstripped `System.Core.dll` (from `unity.bepinex.dev`, same source as Phase 3's original mscorlib attempt) via `dll_search_path_override`, alongside the six mscorlib patches. This got further (created `BepInEx.cfg` successfully) but on the **next** launch caused the **entire game process to exit** with no crash handler and no graceful continuation — a genuine regression from the "game stays playable" baseline this phase was explicit about preserving. Root cause: the full unstripped `System.Core.dll` let Harmony/MonoMod get far enough to hit the deep `UnverifiableCodeAttribute` gap above, and *that* failure mode is apparently unrecoverable at the Mono runtime level (unlike the six patched MissingMethodExceptions, which were all cleanly caught by Doorstop's own exception handler with Unity continuing normally afterward). **Reverted immediately** — `System.Core.dll` swap removed, config reverted, game re-verified playable, per "if a hook causes a crash: stop, collect logs, revert, do not repeatedly relaunch."
2. **Narrower, surgical version**: a second small Cecil-based tool ([`tools/systemcore_patcher/`](../tools/systemcore_patcher/)) that adds *only* the missing `Key` property to `IGrouping<TKey,TElement>` (the interface) and to the concrete `Grouping<TKey,TElement>` class (backed by its own still-present `_key` field) — verified by decompilation to exactly match the official unstripped signature. This is qualitatively different from patches 1-6: it modifies an **interface member**, not a standalone method. Tested once: the game stayed **fully playable** (no full-process crash this time — a materially safer outcome than the wholesale swap), but Mono's own type loader failed with `TypeLoadException: ... VTable setup of type System.Linq.Grouping\`2[...] failed`. **Conclusion, based on direct evidence, not assumption**: Mono's runtime VTable builder does not accept an interface member added post-hoc via metadata patching the way a plain standalone method addition does — the six mscorlib patches (all plain static/instance methods or simple constructors, none touching an interface) worked flawlessly every single time, while this is the one interface-touching attempt and it is the one that failed at the type-loading level. This is a genuine, reproducible technical boundary of this patching technique, not a coding mistake in the patch itself (the emitted IL and decompiled shape were verified correct). **Reverted** — `System.Core.dll` removed from `patched_corlib\`, leaving only the six proven-safe mscorlib patches active.

Both reversions were verified: game re-launched once more with only the six mscorlib patches active, confirmed fully playable (602 MB, responsive, correct window title), and `Mini Motorways.exe`'s SHA-256 re-confirmed unchanged (`305d1ea0...`).

## What this means for Steps 2 and 3 of the brief's escalation order

- **Step 2 (BepInEx config-level avoidance)**: not applicable in the usual sense — the blocker is no longer in BepInEx's own code (which now initializes fully), it's inside Harmony/MonoMod's dynamic codegen, a third-party dependency BepInEx bundles. There is no BepInEx config option that avoids Harmony initialization (`ConsoleSetOutFix.Apply()`, which triggers it, is one of BepInEx's built-in "runtime fixes" applied unconditionally during `PreloaderMain`).
- **Step 3 (MelonLoader)**: remains a live option, not yet attempted this phase. MelonLoader uses a different (though related) detour/patching stack and might not hit the identical `DynamicMethodDefinition`/`UnverifiableCodeAttribute` gap, or might hit it in a different, more tractable place. Not started — held for explicit direction, per the brief's "stop after this phase" instruction.

## Rollback / current state

- `Mini Motorways.exe`: SHA-256 `305d1ea095f0ba9e844b91497283277c61f0dea9d5f7979065e78f3e7bab0ce6` — **unchanged** throughout.
- `Mini Motorways_Data\Managed\mscorlib.dll`: SHA-256 `fe9f5c7a02c89e71d45a31ecadf9303a65c85828e625e0f2ecb13ee2ce6f75f2` — **unchanged**, never touched on disk. Backup retained at `backups/mscorlib_original/mscorlib.dll.bak`.
- `Mini Motorways_Data\Managed\System.Core.dll`: SHA-256 `de0e43e31a22cf11bb9dd4c4662ea81a43f64f275695e5dd901d08ffdf1e4a2a` — **unchanged**. Backup retained at `backups/systemcore_original/System.Core.dll.bak`.
- `patched_corlib\` (new folder, new files only) now contains exactly one file: our six-times-patched `mscorlib.dll` (SHA-256 `1755a28c919662611f8730b70dbc41c8f03e4514c180718ff908e05f434c294f`), referenced via `doorstop_config.ini`'s `dll_search_path_override = patched_corlib`. This is the new, more-capable baseline going forward — strictly better than end-of-Phase-3's baseline (reaches much deeper before failing gracefully) and confirmed safe across every test this phase.
- Full rollback to a corlib-override-free state (if ever desired): set `dll_search_path_override =` (blank) in `doorstop_config.ini`, or delete `patched_corlib\` entirely — either reverts to Phase 3's original baseline behavior.
