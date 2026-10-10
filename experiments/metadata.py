"""Reproducibility metadata. Anything that cannot be determined is recorded as null/"unknown", never guessed."""
import datetime
import glob
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import zipfile
from typing import Any, Dict, List, Optional

from . import paths


def sha256_file(path: str) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _run(cmd: List[str], cwd: Optional[str] = None, timeout: int = 20) -> Optional[str]:
    try:
        out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        return out.stdout if out.returncode == 0 else None
    except Exception:
        return None


def git_info(save_patch_to: Optional[str] = None) -> Dict[str, Any]:
    info = {"git_commit": None, "git_dirty": None, "git_available": False}
    top = _run(["git", "rev-parse", "--show-toplevel"], cwd=paths.ROOT)
    if not top:
        info["git_note"] = "project directory is not a git repository; see source_files / logs/source_snapshot.zip"
        return info
    info["git_available"] = True
    info["git_commit"] = (_run(["git", "rev-parse", "HEAD"], cwd=paths.ROOT) or "").strip() or None
    status = _run(["git", "status", "--porcelain"], cwd=paths.ROOT)
    info["git_dirty"] = bool(status and status.strip()) if status is not None else None
    if info["git_dirty"] and save_patch_to:
        diff = _run(["git", "diff", "HEAD"], cwd=paths.ROOT, timeout=60) or ""
        untracked = _run(["git", "ls-files", "--others", "--exclude-standard"], cwd=paths.ROOT) or ""
        with open(save_patch_to, "w", encoding="utf-8") as f:
            f.write(diff)
            if untracked.strip():
                f.write("\n# untracked files (content not included in this patch):\n")
                for line in untracked.splitlines():
                    f.write(f"# {line}\n")
    return info


def source_files() -> List[str]:
    files = glob.glob(os.path.join(paths.PYTHON_DIR, "*.py"))
    files += glob.glob(os.path.join(paths.ROOT, "experiments", "**", "*.py"), recursive=True)
    files += glob.glob(os.path.join(paths.ROOT, "experiments", "configs", "*.yaml"))
    files += [paths.ENTRYPOINT_CS, os.path.join(paths.ROOT, "game_plugin", "DoorstopBootstrap", "DoorstopBootstrap.csproj")]
    return sorted(f for f in files if os.path.isfile(f))


def source_hashes() -> Dict[str, str]:
    return {os.path.relpath(f, paths.ROOT).replace("\\", "/"): sha256_file(f) for f in source_files()}


def source_tree_sha256(hashes: Dict[str, str]) -> str:
    h = hashlib.sha256()
    for k in sorted(hashes):
        h.update(f"{k}:{hashes[k]}\n".encode())
    return h.hexdigest()


def save_source_snapshot(zip_path: str):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in source_files():
            z.write(f, os.path.relpath(f, paths.ROOT))


def game_build() -> Dict[str, Any]:
    out = {"game_build_id": None, "unity_version": None, "steam_manifest": None}
    manifest = os.path.join(paths.STEAM_APPS, f"appmanifest_{paths.STEAM_APP_ID}.acf")
    try:
        txt = open(manifest, encoding="utf-8", errors="replace").read()
        m = re.search(r'"buildid"\s+"(\d+)"', txt)
        out["game_build_id"] = m.group(1) if m else None
        out["steam_manifest"] = manifest
    except OSError:
        pass
    try:
        data = open(os.path.join(paths.GAME_DIR, "Mini Motorways_Data", "globalgamemanagers"), "rb").read(4096)
        m = re.search(rb"(\d{4}\.\d+\.\d+[fpb]\d+)", data)
        out["unity_version"] = m.group(1).decode() if m else None
    except OSError:
        pass
    out["game_exe_sha256"] = sha256_file(paths.GAME_EXE)
    return out


def cpu_name() -> Optional[str]:
    out = _run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name)"])
    return out.strip() if out and out.strip() else (platform.processor() or None)


def library_versions() -> Dict[str, Optional[str]]:
    v = {"python": sys.version.split()[0]}
    for mod, key in (("gymnasium", "gymnasium"), ("stable_baselines3", "stable_baselines3"), ("sb3_contrib", "sb3_contrib"),
                     ("torch", "torch"), ("numpy", "numpy"), ("yaml", "pyyaml"), ("matplotlib", "matplotlib")):
        try:
            v[key] = __import__(mod).__version__
        except Exception:
            v[key] = None
    return v


def gpu_info() -> Dict[str, Any]:
    try:
        import torch
        if torch.cuda.is_available():
            return {"gpu_used_by_torch": True, "gpu_name": torch.cuda.get_device_name(0), "cuda_version": torch.version.cuda}
        return {"gpu_used_by_torch": False, "gpu_name": None, "cuda_version": torch.version.cuda}
    except Exception:
        return {"gpu_used_by_torch": None, "gpu_name": None, "cuda_version": None}


def environment_snapshot() -> Dict[str, Any]:
    """Everything about the code/binaries/machine, independent of one particular run."""
    hashes = source_hashes()
    return {
        "bootstrap_dll_deployed_sha256": sha256_file(paths.DEPLOYED_DLL),
        "bootstrap_dll_built_sha256": sha256_file(paths.BUILT_DLL),
        "entrypoint_cs_sha256": sha256_file(paths.ENTRYPOINT_CS),
        "source_tree_sha256": source_tree_sha256(hashes),
        "source_files": hashes,
        "game": game_build(),
        "libraries": library_versions(),
        "os": platform.platform(),
        "cpu": cpu_name(),
        "cpu_count": os.cpu_count(),
        **gpu_info(),
    }


def observation_action_shape() -> Dict[str, Any]:
    """Shapes as defined by the existing code (module constants), without launching the game."""
    try:
        import mini_motorways_env as M
        return {"observation_shape_expected": [M.NUM_TILE_FEATURES * 777 + M.NUM_GLOBAL_FEATURES],
                "tile_features": M.NUM_TILE_FEATURES, "global_features": M.NUM_GLOBAL_FEATURES,
                "action_space_size_expected": 1 + 37 * 21 * 4 + 2}
    except Exception:
        return {}


def write_json(path: str, obj: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
