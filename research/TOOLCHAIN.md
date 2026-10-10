# TOOLCHAIN — Phase 1 Instrumentation Route

Status: CONFIRMED unless marked otherwise. Generated 2026-09-28.

## Summary

```
Game build:      Mini Motorways, Steam buildid 25381668, Unity 2022.3.62f2, x86 (32-bit), Mono scripting backend
Unity version:   2022.3.62f2 (7670c08855a9)
Runtime:         Mono (mono-2.0-bdwgc.dll bundled), NOT IL2CPP
Recommended loader: BepInEx 5.4.23.5 (win_x86 build)
Loader version:  5.4.23.5 (released 2026-02-08, LTS branch)
Reason:          Least invasive, most mature, actively-maintained loader for a Mono/x86 Unity 2022.3 game. Managed hooking (Harmony) rather than native/IL2CPP interop — matches the fact that App.dll/Assembly-CSharp.dll are real managed assemblies with PDBs.
Known compatibility issues: none identified for this exact combination — see Evidence below.
Installation procedure: see below (non-destructive, doorstop-based).
Rollback procedure: delete injected files, restore backed-up doorstop config; see below.
```

## Why BepInEx over MelonLoader (or raw Mono debugger attach)

Both BepInEx 5 and MelonLoader officially support Mono x86 Unity games (CONFIRMED via web search of both projects' docs/releases). Recommending **BepInEx 5** because:

1. **Direct version evidence.** BepInEx 5.4.23.5's release notes explicitly call out "fixes for Unity 2022.3.62f3 compatibility" (CONFIRMED via GitHub release notes, fetched 2026-09-28). Our installed build is Unity **2022.3.62f2** — same `2022.3.62` patch train, one Unity point-release apart. This is the closest documented compatibility statement available for any loader against this exact engine build.
2. **Asset exists for our exact architecture.** `BepInEx_win_x86_5.4.23.5.zip` is a real, published asset on that release (CONFIRMED via GitHub API `releases/tags/v5.4.23.5`).
3. **Non-invasive injection mechanism.** BepInEx on Mono uses **Doorstop**, which sets an environment variable (`DOORSTOP_ENABLE`) / loads via `winhttp.dll` proxy placed next to the game executable, causing Mono to load a managed entrypoint DLL before the game's own assemblies run. This requires **zero modification of any existing game file** — `Mini Motorways.exe`, `App.dll`, `UnityPlayer.dll` etc. are untouched on disk. Only new files are added to the install directory (`winhttp.dll`, `doorstop_config.ini`, `BepInEx/` folder). This satisfies the "preserve the original game" safety rule better than binary patching or DLL redirection that modifies an existing file.
4. **Harmony-based managed hooking.** Because the game is Mono (not IL2CPP), BepInEx plugins can use standard Harmony patches directly against `App.dll` types/methods reflectively — no metadata-to-C#-stub regeneration step (which IL2CPP would require via Il2CppInterop/Cpp2IL) is needed. This is a meaningfully shorter, lower-risk path from "hook exists" to "can read game state," matching the Phase 2→4 progression in the plan.
5. **PDB availability.** `App.pdb` ships alongside `App.dll` in the install. A decompiler (dnSpy/ILSpy/dotPeek) will resolve original method/field names and even source line numbers, which is a major static-recon advantage BepInEx's Harmony patching can exploit directly (patch by original method name, not an obfuscated one).

MelonLoader remains a documented fallback (PROBABLE viable, not yet tested) if BepInEx 5 turns out to have an undiscovered incompatibility with this specific build — same Doorstop-style non-destructive injection model, also supports Mono x86.

## Installation procedure (not yet executed — pending Phase 3 "hello world")

1. Download `BepInEx_win_x86_5.4.23.5.zip` from the official GitHub release (`https://github.com/BepInEx/BepInEx/releases/tag/v5.4.23.5`), verify it's fetched from the official `BepInEx/BepInEx` repo (not a mirror/reupload).
2. **Back up** the following before touching the install directory (per safety rules), copying into `backups/`:
   - `Mini Motorways.exe` (already hashed above, but keep a byte copy too since Doorstop's `winhttp.dll` sits beside it)
   - Record directory listing of the install root (to detect any unexpected file additions/removals later)
3. Extract the zip contents into `C:\Program Files (x86)\Steam\steamapps\common\Mini Motorways\` (game root, same folder as `Mini Motorways.exe`). This adds:
   - `winhttp.dll` (Doorstop proxy — new file, not a replacement of any existing DLL)
   - `doorstop_config.ini`
   - `BepInEx\` folder (core managed DLLs, `plugins\`, `config\`, `core\`)
4. Launch the game once through Steam to let BepInEx generate `BepInEx/LogOutput.txt` and `BepInEx/config/BepInEx.cfg`.
5. Verify `BepInEx/LogOutput.txt` shows a clean chainloader startup with no exceptions, and that the game itself still runs and is playable normally (regression check).
6. Only then proceed to Phase 3 (minimal hello-world plugin DLL in `BepInEx/plugins/`).

This procedure is **not yet performed** — Phase 0/1 is reconnaissance only per the task instructions. Executing it is the first action of Phase 3, and will only happen after this report is reviewed.

## Rollback procedure

Because installation only adds files (does not overwrite `Mini Motorways.exe`, `App.dll`, or any other original asset):
1. Delete `winhttp.dll`, `doorstop_config.ini`, and the `BepInEx/` folder from the game root.
2. Verify via SHA-256 that `Mini Motorways.exe` still matches the hash recorded in [`INSTALLATION.md`](INSTALLATION.md) (`305D1EA0...`) — confirms no in-place modification occurred.
3. Alternatively, Steam's "Verify integrity of game files" can restore any accidentally-touched file from the depot as a second line of defense.

## Known compatibility risks (UNKNOWN / to verify empirically in Phase 3)

- HYPOTHESIS: Steam's `dpcPlatform.dll` or `steam_api.dll` native plugins could behave differently under Doorstop injection order — untested.
- HYPOTHESIS: Anti-cheat/DRM — no evidence of any anti-cheat layer in the file inventory (no EAC/BattlEye/VAC-specific files found beyond standard `steam_api.dll`), so this is LOW risk but not zero. Mini Motorways is a single-player-only city-builder with no competitive multiplayer, which further lowers likelihood of anti-cheat.
- UNKNOWN: whether the Steam Turkish-language client config affects Doorstop's working directory assumptions — likely irrelevant, flagged for completeness only.

## Stop condition check

No stop condition triggered:
- Toolchain identified with direct version-compatibility evidence (not guessed).
- Asset for the exact required architecture (`win_x86`) confirmed to exist.
- Installation procedure is non-destructive and reversible.

Proceeding recommendation: begin Phase 3 (minimal BepInEx "hello world" plugin) only after this report is reviewed, since Phase 3 is the first step that actually touches the install directory.
