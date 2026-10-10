# INSTALLATION — Phase 0 Environment Audit

Status: CONFIRMED unless marked otherwise. Generated 2026-09-28.

## Host

- OS: Windows 11 Pro 10.0.26200
- Steam library: `C:\Program Files (x86)\Steam\steamapps\common`
- Game currently running: NO (checked via `tasklist`, no `Mini Motorways.exe` process found)

## Game installation

- Install path: `C:\Program Files (x86)\Steam\steamapps\common\Mini Motorways`
- Executable: `Mini Motorways.exe` (650,240 bytes)
- Steam AppID: `1127500` (from `appmanifest_1127500.acf`)
- Depot: `1127501`, manifest `4423824400427792133`
- Build ID: `25381668`
- Last updated (Steam epoch): `1790054164`
- Installed language: Turkish (`UserConfig.language = "turkish"`) — CONFIRMED from acf; relevant because UI text strings will be Turkish, not English, for any OCR/vision-based work later.
- Publisher: Dinosaur Polo Club (from `app.info`)
- SizeOnDisk: 255,145,951 bytes (~243 MB)

## Engine / runtime

- Unity version: **2022.3.62f2** (build `7670c08855a9`), CONFIRMED via `Mini Motorways.exe` FileVersionInfo (`ProductVersion: 2022.3.62f2 (7670c08855a9)`)
- Executable architecture: **x86 (32-bit)**, CONFIRMED by reading the PE header `IMAGE_FILE_HEADER.Machine` field of `Mini Motorways.exe` (`0x014C` = IMAGE_FILE_MACHINE_I386). Corroborated by presence of `UnityCrashHandler32.exe` and a `Plugins\x86\` (not `x86_64\`) folder.
- Scripting backend: **Mono**, CONFIRMED — not IL2CPP. Evidence:
  - `Mini Motorways_Data\Managed\` contains real managed .NET assemblies (e.g. `App.dll`, `Assembly-CSharp.dll`, `mscorlib.dll`, `UnityEngine.CoreModule.dll`, ~100 DLLs total), each with a matching `.pdb`.
  - There is **no** `GameAssembly.dll` and **no** `global-metadata.dat` anywhere under the install directory (IL2CPP signature files, absent).
  - `MonoBleedingEdge\EmbedRuntime\mono-2.0-bdwgc.dll` and `MonoBleedingEdge\etc\mono` are present — the Mono runtime is bundled.
- `boot.config`:
  ```
  wait-for-native-debugger=0
  hdr-display-enabled=0
  gc-max-time-slice=3
  build-guid=030acf5ec5f1457f99db607bcc557595
  ```

## Relevant managed assemblies (Mini Motorways_Data\Managed)

Game-logic-relevant (PROBABLE, based on name/size — not yet decompiled/verified):

| File | Size | Notes |
|---|---|---|
| `App.dll` | 2,727,936 bytes | Has matching `App.pdb` (1.05 MB). By far the largest custom assembly — PROBABLE main game/simulation logic lives here, not in `Assembly-CSharp.dll`. |
| `Assembly-CSharp.dll` | 4,608 bytes | Tiny — PROBABLE thin/near-empty, real logic elsewhere (i.e. `App.dll`). |
| `Assembly-CSharp-firstpass.dll` | 124,928 bytes | Has `.pdb`. |
| `DinoPoloClub.Analytics.Runtime.dll` | 17,408 bytes | Likely telemetry/analytics, not simulation. |
| `MOAnalytics.Runtime.dll` | 12,288 bytes | Likely telemetry/analytics. |
| `FixedMath.dll` | 3,309,568 bytes | PROBABLE deterministic fixed-point math library — highly relevant if the simulation uses fixed-point math for determinism (common in Dinosaur Polo Club / Unity strategy games with seeded reproducibility). Worth prioritizing in Phase 2 static recon. |
| `Rewired_Core.dll`, `Rewired_Windows.dll` | 2.0 MB / 421 KB | Third-party input library (Rewired) — relevant for Phase 7 action injection (input simulation) and Phase 13 (mouse/keyboard). |
| `Newtonsoft.Json.dll` | 387,072 bytes | JSON — possibly used for save files. |
| `NaughtyAttributes.Core.dll` | 16,896 bytes | Unity inspector attribute helper, not gameplay-relevant. |
| `G-Audio.dll` | 235,008 bytes | Audio middleware, not gameplay-relevant. |
| `Easing.dll` | 8,192 bytes | Animation easing, likely UI/VFX only. |

App.dll having a full PDB alongside it is notable: it means decompilers (dnSpy, ILSpy) will be able to show original file/line info, and if a debugger can attach to Mono, we may get proper breakpoints/symbols — this significantly improves Phase 2/3/4 prospects versus IL2CPP reverse engineering.

## Native plugins (`Mini Motorways_Data\Plugins\x86\`)

| File | Size | Purpose (PROBABLE) |
|---|---|---|
| `steam_api.dll` | 280,728 | Steamworks SDK |
| `dpcPlatform.dll` | 183,808 | PROBABLE Dinosaur Polo Club's own cross-platform abstraction layer |
| `decompressAudio.dll` | 137,216 | Audio codec |
| `lib_burst_generated.dll` | 94,720 | Unity Burst-compiled native code |
| `Rewired_DirectInput.dll` | 11,264 | Rewired input backend |

## File hashes (SHA-256, for drift detection / rollback verification)

```
Mini Motorways.exe:
  305D1EA095F0BA9E844B91497283277C61F0DEA9D5F7979065E78F3E7BAB0CE6

Mini Motorways_Data\Managed\App.dll:
  76C9D9069D61160A5C20AFA62D33C0272173AB46682DB28214374E56B557C638

Mini Motorways_Data\Managed\Assembly-CSharp.dll:
  23D8457DD53C64A04F2A08298F8ED489D57B61B2ABFE2A94948F71AE7E199B51

UnityPlayer.dll:
  28C9FE8F3C1E580B636205B8C567B90F49CA56D910BD22A102CF5461D7179322
```

Note: `App.dll`'s embedded `FileVersion`/`ProductVersion` are both `0.0.0.0` (UNKNOWN — the assembly doesn't carry a meaningful version stamp). The Steam build ID (`25381668`) and these hashes are therefore the authoritative version fingerprint for this install; re-check hashes after any Steam update before trusting prior research notes.

## Save / config locations (UNKNOWN — not yet located)

Not yet searched. Likely candidates for Phase 2+ investigation: `%USERPROFILE%\AppData\LocalLow\Dinosaur Polo Club\Mini Motorways\`, or Steam Cloud. Not touched in this phase (per safety rules — no destructive save access without explicit need).

## Initial toolchain implication

Because this is a **32-bit Mono** build (not IL2CPP), the modding ecosystem's Mono-x86 support path applies, not the IL2CPP native-hooking path. See [`TOOLCHAIN.md`](TOOLCHAIN.md) for the specific loader recommendation and compatibility check.

## Stop conditions check

- Installation located: YES
- Build type determined: YES (Mono, x86)
- Required files present: YES (`Managed/*.dll` with matching `.pdb`s, native plugins, boot.config)
- Toolchain compatibility: see TOOLCHAIN.md — not blocked, proceeding to Phase 1 deliverable.

No STOP condition triggered. Proceeding to Phase 1.
