"""Locations. Importing this module puts the existing stack (python/) on sys.path."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON_DIR = os.path.join(ROOT, "python")
RUNS_DIR = os.path.join(ROOT, "runs")
LOGS_DIR = os.path.join(ROOT, "logs")
CONFIGS_DIR = os.path.join(ROOT, "experiments", "configs")
GAME_DIR = r"C:\Program Files (x86)\Steam\steamapps\common\Mini Motorways"
GAME_EXE = os.path.join(GAME_DIR, "Mini Motorways.exe")
STEAM_APPS = r"C:\Program Files (x86)\Steam\steamapps"
STEAM_APP_ID = "1127500"
DEPLOYED_DLL = os.path.join(GAME_DIR, "Bootstrap", "MiniMotorwaysBootstrap.dll")
BUILT_DLL = os.path.join(ROOT, "game_plugin", "DoorstopBootstrap", "bin", "Release", "net472", "MiniMotorwaysBootstrap.dll")
ENTRYPOINT_CS = os.path.join(ROOT, "game_plugin", "DoorstopBootstrap", "Entrypoint.cs")

if PYTHON_DIR not in sys.path:
    sys.path.insert(0, PYTHON_DIR)
