# PHASE_3C — MelonLoader Evaluation

Generated 2026-09-28. Evaluates MelonLoader as an alternative to BepInEx per Phase 3B's Step 3 option. **Result: MelonLoader crashes this exact game build during its own native bootstrap, before reaching plugin-loading stage — a harder failure than anything hit with BepInEx.** BepInEx has been fully restored to its Phase 3B working state.

## Safety requirements — how each was met

1. **Preserve BepInEx state**: full byte-for-byte backup of all 26 BepInEx-related files (the `BepInEx\` folder, `patched_corlib\`, `winhttp.dll`, `version.dll`, `doorstop_config.ini`, `.doorstop_version`, `BepInEx_changelog.txt`) taken to `backups/bepinex_full_state_20260928_225145/` with a SHA-256 manifest (`backups/bepinex_state_manifest.sha256`) **before** touching anything.
2. **Original game assemblies untouched**: `Mini Motorways.exe` hash re-verified unchanged (`305d1ea0...`) at the start, immediately after the crash, and after the final restore. `mscorlib.dll`/`System.Core.dll` were never touched this phase (Phase 3B's patches live entirely in the separate `patched_corlib\` folder, which was itself backed up and restored intact).
3. **No further mscorlib/System.Core patching**: none attempted this phase — confirmed by the tool directories (`tools/mscorlib_patcher/`, `tools/systemcore_patcher/`) being untouched since Phase 3B.
4. **Complete inventory + hashes before changing loader files**: recorded above (item 1) and in the Files section below.
5. **MelonLoader compatibility for Unity 2022.3.62f2 / x86 / Mono**: verified before download — see next section.
6. **Clean test, not mixed**: BepInEx was **fully removed** (`BepInEx\`, `patched_corlib\`, `winhttp.dll`, `version.dll`, `doorstop_config.ini`, `.doorstop_version`, `BepInEx_changelog.txt` all deleted, confirmed via directory listing showing only the four original game files remaining) before installing MelonLoader.
7. **Smallest possible Hello World plugin**: [`game_plugin/MelonHelloWorld/Plugin.cs`](../game_plugin/MelonHelloWorld/Plugin.cs) — a single `MelonPlugin` subclass whose `OnInitializeMelon()` does nothing but log identifying information.
8. **Unmistakable load marker**: the plugin logs the literal string `[MiniMotorwaysAI] Plugin loaded successfully` (never reached — see Result).
9. **No simulation hooks**: none written or attempted.
10. **Crash handling**: followed exactly — logs collected immediately, reverted immediately, documented here, did not attempt to patch around it or relaunch repeatedly.

## MelonLoader version and compatibility evidence

- **Version**: 0.7.3, the current latest release, `MelonLoader.x86.zip` from the official `LavaGang/MelonLoader` GitHub repository.
- **Integrity**: downloaded file's SHA-256 (`f389672bc234ed134b878492f057527e7029b9022936bbf99ecd43d0d55f51c3`) matched the digest published in the GitHub Releases API for that exact asset.
- **Architecture**: confirmed x86/32-bit via `dumpbin /headers` on the package's `version.dll` proxy (`14C machine (x86)`), matching this game's confirmed x86 architecture.
- **Mono compatibility**: the package ships separate `net35\`, `net472\`, and `net6\` runtime variants; `net6\` additionally carries `Il2CppInterop.*` assemblies used only for IL2CPP games. This game is Mono (confirmed `INSTALLATION.md`), so the `net472\` variant applies — no IL2CPP-specific dependency needed, and the official README states the heavier ".NET 6.0 Desktop Runtime" requirement applies only to Il2Cpp games, not Mono ones.
- **Unity 2022.3.62f2 specifically**: no explicit per-Unity-version compatibility statement was found in MelonLoader's changelog or README (unlike BepInEx 5.4.23.5's explicit "fixes Unity 2022.3.62f3" note) — MelonLoader's compatibility claims are broader/version-untargeted ("most Unity Mono/IL2CPP games"), so this was a genuine open question resolved empirically, per the mission's evidence-over-assumption principle.

## Installation (clean, BepInEx fully removed first)

Files added to the game root, all confirmed new (not overwriting anything):
```
MelonLoader\                      (net35\, net472\, net6\, Dependencies\, Documentation\)
version.dll                       (MelonLoader's doorstop-equivalent proxy DLL, x86, confirmed via dumpbin)
Plugins\MiniMotorwaysMelonHelloWorld.dll   (our hello-world plugin, net472)
```
`MelonLoader.dll`'s public API was decompiled and read before use, not assumed from memory (`MelonPlugin`, `MelonInfoAttribute`, `MelonPlatformAttribute`, `UnityInformationHandler` — all confirmed present with exact signatures used in the plugin).

## Loader initialization evidence — it did start, further than expected

MelonLoader's own log (`MelonLoader\Logs\26-9-28_22-57-30.log`, and the identical `MelonLoader\Latest.log`) confirms its native/managed bootstrap bridge genuinely executed inside the live process (colored console output visible in stdout, proving MelonLoader's console subsystem initialized):

```
[22:57:30.478] [ERROR] [MelonLoader.Bootstrap] System.Reflection.CustomAttributeFormatException: Could not find a field with name CharSet
  at (wrapper managed-to-native) System.MonoCustomAttrs.GetCustomAttributesInternal(...)
  at System.MonoCustomAttrs.GetCustomAttributesBase (...)
  at System.MonoCustomAttrs.GetCustomAttributes (...)
  at System.RuntimeType.GetCustomAttributes (...)
  at MelonLoader.NativeLibrary`1[T]..ctor (System.IntPtr ptr) [...] in .../MelonLoader/Utils/NativeLibrary.cs:143
  at MelonLoader.InternalUtils.BootstrapInterop.Initialize (System.IntPtr bootstrapHandle) [...] in .../MelonLoader/InternalUtils/BootstrapInterop.cs:95
```

## Result: hard crash, not a graceful failure — qualitatively worse than BepInEx's blocker

The game process **fully terminated** — confirmed via `tasklist` showing no `Mini Motorways.exe` process and no `UnityCrashHandler32.exe` watchdog either, within seconds of the error being logged. This is a materially different failure mode from every BepInEx failure encountered in Phase 3/3B, all of which were caught by Doorstop's own exception handler and left the game running normally afterward. Here, the crash originates inside `System.MonoCustomAttrs.GetCustomAttributesInternal` — a **native Mono runtime internal call** (`wrapper managed-to-native`), not managed code we could patch with the Cecil technique used in Phase 3B. It is triggered while `MelonLoader.NativeLibrary<T>`'s constructor inspects a delegate type's custom attributes (almost certainly `UnmanagedFunctionPointerAttribute`, which carries a `CharSet` field) to prepare a native function-pointer binding — and this exact stripped `mscorlib.dll` is missing that field from the attribute type's own metadata, which the Mono VM's own attribute-blob decoder cannot tolerate, unlike a missing *method* (which merely throws a catchable `MissingMethodException` from managed code).

**Plugin load evidence**: none. `Plugins\MiniMotorwaysMelonHelloWorld.dll` was never loaded — MelonLoader's own bootstrap failed before reaching the plugin-discovery stage.

**Whether the game remains playable**: **no**, while MelonLoader was installed — this is the first time in Phases 3/3A/3B/3C that a loader installation made the game fail to launch at all, rather than launching normally with instrumentation simply inert. This is why safety requirement #10 (immediate revert) was followed rather than attempting any patch.

## Why this is harder to fix than the Phase 3B blocker

Phase 3B's blocker (`MonoMod.Utils.DynamicMethodDefinition` needing `System.Security.UnverifiableCodeAttribute`) was already inside dynamic-codegen territory and flagged as a larger, more open-ended problem than the six successful patches. This MelonLoader failure is a level below even that: it's inside Mono's own C-implemented custom-attribute reader, which decodes attribute blobs against the *declaring attribute type's own field layout* read from metadata — there is no managed method to add via Cecil that would fix this, because the failure isn't "a method is missing," it's "the VM's internal attribute parser can't find a field it expects to exist on an attribute type applied somewhere in MelonLoader's own compiled code." Fixing it would require either (a) patching the actual `System.Runtime.InteropServices.UnmanagedFunctionPointerAttribute` type in this mscorlib to have the missing `CharSet` field with correct layout/metadata semantics the runtime's native decoder expects (a much more delicate class of patch than anything attempted in Phase 3B, since it's read by native code, not managed code), or (b) finding a way to make MelonLoader avoid ever inspecting that particular attribute on this platform.

## Rollback: executed and fully verified

1. Confirmed process fully stopped (crash was already a full exit — no forced kill needed for MelonLoader itself, though the earlier BepInEx test process from before this phase was force-killed as routine cleanup).
2. Deleted `MelonLoader\`, `Plugins\`, `version.dll`, and the `UserData\` folder MelonLoader created on its brief run.
3. Restored all 26 backed-up BepInEx files (`BepInEx\`, `patched_corlib\`, `winhttp.dll`, `version.dll`, `doorstop_config.ini`, `.doorstop_version`, `BepInEx_changelog.txt`) from `backups/bepinex_full_state_20260928_225145/`.
4. **Verified the restore byte-for-byte**: re-hashed every restored file and diffed the hash set against the original pre-3C backup manifest — **all hashes matched exactly**, confirming the restore is not just "the same files" but bit-for-bit identical to the Phase 3B working state.
5. Re-launched the game once: confirmed fully playable (598 MB working set, responsive, correct window title) — i.e. BepInEx's Phase 3B state (six mscorlib patches, preloader reaching Harmony/MonoMod before its own blocker) is fully restored and working exactly as it was left.
6. `Mini Motorways.exe` SHA-256 re-confirmed unchanged (`305d1ea095f0ba9e844b91497283277c61f0dea9d5f7979065e78f3e7bab0ce6`) throughout.

## Conclusion

MelonLoader is **not currently a viable path** for this exact game build without solving a harder, native-runtime-level BCL gap than anything addressed so far — and solving it would require patching an attribute type's field layout in a way that Mono's own native VM code reads correctly, a materially more delicate operation than the plain-method additions that worked reliably in Phase 3B. BepInEx, despite its own unresolved blocker (deep in Harmony/MonoMod dynamic codegen), remains the further-progressed and safer of the two paths: it degrades gracefully on failure, keeps the game playable, and reaches substantially further into its own initialization before failing.
