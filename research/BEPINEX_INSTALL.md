# BEPINEX_INSTALL — Phase 3 Installation Record

Generated 2026-09-28. Documents the exact BepInEx/Doorstop installation performed into the live Mini Motorways install directory, what was added/modified, and rollback instructions.

## Pre-installation verification (STEP 1)

- Re-read [`INSTALLATION.md`](INSTALLATION.md); re-hashed `Mini Motorways.exe` immediately before installing — SHA-256 `305d1ea095f0ba9e844b91497283277c61f0dea9d5f7979065e78f3e7bab0ce6`, **unchanged** from the Phase 0 record.
- Confirmed via `tasklist` that the game was not running before any file was touched.
- Confirmed BepInEx `5.4.23.5` (`win_x86` build) matches: Windows, x86 (32-bit), Unity 2022.3.62f2, Mono — per [`TOOLCHAIN.md`](TOOLCHAIN.md)'s prior analysis (release notes explicitly cite Unity 2022.3.62f3 compatibility, one point-release from our 2022.3.62f2).
- Confirmed via directory listing that **no existing BepInEx/doorstop files were present** in the install directory before this phase — a clean install, not a replacement.
- Backup created before any modification: `backups/pre_bepinex_20260928/Mini Motorways.exe.bak` (byte copy of the untouched executable) and `backups/install_dir_listing_before.txt` (full pre-install directory listing).

## BepInEx version and source

- **Version**: 5.4.23.5 (`BepInEx_win_x86_5.4.23.5.zip`)
- **Source**: official `BepInEx/BepInEx` GitHub repository, release tag `v5.4.23.5`, asset uploaded by `ManlyMarco` (recognized BepInEx maintainer)
- **Integrity**: downloaded file's SHA-256 (`37651c79e40d6f909572a4f461ac25350bb3ef8fe7fbd29f1aa8791a33b84c82`) matched the digest published in the GitHub Releases API for that exact asset — confirmed authentic, not a reupload/mirror.

## Files added to the game install directory (all new — nothing overwritten)

| Path | Purpose |
|---|---|
| `BepInEx\core\*.dll` (+`.xml`) | BepInEx core runtime, Harmony, MonoMod, Mono.Cecil — unmodified from the official release zip |
| `BepInEx\plugins\MiniMotorwaysInstrumentation.dll` | Our hello-world plugin (see [`LIVE_INSTRUMENTATION.md`](LIVE_INSTRUMENTATION.md)) |
| `winhttp.dll` | Doorstop proxy DLL (loaded because `UnityPlayer.dll` statically imports `WINHTTP.dll` — confirmed via `dumpbin /imports`) |
| `version.dll` | A second copy of the same Doorstop proxy DLL, installed under the alternate name per BepInEx's own documented troubleshooting step (`UnityPlayer.dll` also statically imports `VERSION.dll`) |
| `doorstop_config.ini` | Doorstop configuration (see below for the one setting changed from default) |
| `.doorstop_version` | Doorstop version marker (`4.5.0`) |
| `BepInEx_changelog.txt` | BepInEx's changelog, renamed on copy to avoid colliding with any existing `changelog.txt` (there was none, but done defensively) |
| `unstripped_corlib\mscorlib.dll` | An unstripped `mscorlib.dll` for Unity 2022.3.62, fetched from the official `unity.bepinex.dev` corlib mirror — **currently not referenced** (see Known Issue below; the config line that would activate it is commented back out) |

**No file that shipped with the original game was modified, renamed, or deleted.** `Mini Motorways.exe`, `App.dll`, `UnityPlayer.dll`, and every other original asset are byte-identical to the Phase 0 record (exe hash re-verified after all experimentation — see Final State below).

## Configuration changes

`doorstop_config.ini` (itself a new file, not an original game file) currently holds:
```ini
[General]
enabled = true
target_assembly = BepInEx\core\BepInEx.Preloader.dll
redirect_output_log = true      ; changed from the shipped default (false), for diagnostics
boot_config_override =
ignore_disable_switch = false

[UnityMono]
dll_search_path_override =      ; tried "unstripped_corlib" during troubleshooting, reverted — see Known Issue
debug_enabled = false
```
`redirect_output_log = true` was left enabled — it only causes Unity's own log to additionally be written to `output_log.txt` in the game folder, which is useful for future debugging and has no gameplay effect. Everything else is default.

## Installation is filename-driven proxying, not binary patching

BepInEx 5 on a Mono Unity game uses Doorstop: `winhttp.dll`/`version.dll` are legitimate Windows DLL names that `UnityPlayer.dll` already imports at load time (confirmed via `dumpbin /imports UnityPlayer.dll` — it statically imports both `VERSION.dll` and `WINHTTP.dll`). Windows' standard DLL search order checks the executable's own directory before `System32`, so placing a same-named, same-architecture DLL there causes ours to load instead of (and forward to) the real system one. This is why no existing file needs to be touched — confirmed empirically via live process-module inspection (see below), not just assumed from documentation.

## Empirical confirmation the proxy loads (STEP 2 verification)

32-bit-process-aware module enumeration (`Get-Process -Id <pid> | Select Modules`, run from **32-bit** PowerShell — 64-bit PowerShell cannot correctly enumerate a WOW64 process's module list, a real pitfall hit during this phase) confirmed, for a live running `Mini Motorways.exe`:
```
WINHTTP.dll   C:\Program Files (x86)\Steam\steamapps\common\Mini Motorways\WINHTTP.dll   <- ours
WINHTTP.dll   C:\WINDOWS\system32\WINHTTP.dll                                            <- real one, loaded by our proxy to forward to
```
confirming the proxy DLL is genuinely loaded into the live game process, not merely present on disk.

## Rollback procedure

1. Ensure the game is not running (`tasklist`).
2. Delete: `BepInEx\` (entire folder), `winhttp.dll`, `version.dll`, `doorstop_config.ini`, `.doorstop_version`, `BepInEx_changelog.txt`, `unstripped_corlib\`.
3. Verify `Mini Motorways.exe`'s SHA-256 still equals `305d1ea095f0ba9e844b91497283277c61f0dea9d5f7979065e78f3e7bab0ce6` (it was never touched, so this should already hold).
4. As a second line of defense, Steam's "Verify integrity of game files" restores anything unexpected from the depot.

This has **not** been executed — the install is currently left in place (inert; see Known Issue) for continued Phase 3 work.

## Update (Phase 3B): surgically patched, no longer the Phase 3 baseline

Phase 3B replaced the blank `dll_search_path_override` with a **surgically patched** `mscorlib.dll` (six small, individually-verified additive fixes via Mono.Cecil — see [`PHASE_3B.md`](PHASE_3B.md) for the full account), deployed non-destructively via the same `dll_search_path_override` mechanism, in a new folder:

```ini
[UnityMono]
dll_search_path_override = patched_corlib
```

`patched_corlib\mscorlib.dll` (SHA-256 `1755a28c919662611f8730b70dbc41c8f03e4514c180718ff908e05f434c294f`) is a copy — the original `Mini Motorways_Data\Managed\mscorlib.dll` was never modified on disk (verified unchanged, SHA-256 `fe9f5c7a02c89e71d45a31ecadf9303a65c85828e625e0f2ecb13ee2ce6f75f2`, backed up at `backups/mscorlib_original/mscorlib.dll.bak`). This patched copy fixes the original `File.WriteAllText` gap plus five further gaps discovered one at a time as the preloader chain got progressively further (see PHASE_3B.md for the complete list and justification of each). The original wholesale "official unstripped corlib" workaround described below was superseded by this narrower approach and is no longer in effect.

A wholesale unstripped `System.Core.dll` swap was also tried in Phase 3B to fix a `IGrouping.Key` gap, and a narrower interface-only patch after that — both were tested once and **reverted** after revealing real problems (a full-process crash for the wholesale swap; a Mono VTable-loading failure for the narrower interface patch). `Mini Motorways_Data\Managed\System.Core.dll` remains completely unmodified (SHA-256 `de0e43e31a22cf11bb9dd4c4662ea81a43f64f275695e5dd901d08ffdf1e4a2a`, backed up at `backups/systemcore_original/System.Core.dll.bak`); `patched_corlib\` currently contains only the patched `mscorlib.dll`.

## Known issue history (original Phase 3 finding, now partially superseded)

`BepInEx.Preloader.dll` originally threw `System.MissingMethodException: Method not found: void System.IO.File.WriteAllText(string,string)` before it ever reached our plugin, because this Unity build's shipped `mscorlib.dll` has that overload stripped by Unity's managed-code linker. The first documented fix tried (Doorstop's `dll_search_path_override` pointed at the *official, wholesale* unstripped `mscorlib.dll`) fixed that specific error but introduced a worse one — a hard crash from an unrelated bug in the unstripped assembly's `TimeZoneInfo` implementation. That wholesale approach was abandoned in favor of the six-patch surgical approach described above, which avoids the regression entirely.

## Final state verification

- `Mini Motorways.exe` SHA-256: `305d1ea095f0ba9e844b91497283277c61f0dea9d5f7979065e78f3e7bab0ce6` — **unchanged**.
- `Mini Motorways_Data\Managed\mscorlib.dll` and `System.Core.dll` — **both unchanged** on disk; all patching happens in the separate `patched_corlib\` folder via Doorstop's search-path override, never in place.
- Game launches, reaches a full Direct3D-rendered, responsive window titled "Mini Motorways", ~600 MB working set — **fully playable**, verified via `Get-Process` (`Responding: True`) after every experiment in both Phase 3 and Phase 3B, including the current baseline.
- BepInEx's preloader now runs substantially further than at the end of Phase 3 — through its entire configuration subsystem and into Harmony's own initialization — before hitting the current blocker (deep inside `MonoMod.Utils.DynamicMethodDefinition`, a third-party dynamic-codegen dependency, not BepInEx's own code). See [`PHASE_3B.md`](PHASE_3B.md) for the complete chain and next-step options.
- No plugin code has executed yet.
