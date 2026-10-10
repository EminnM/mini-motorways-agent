#!/bin/bash
# usage: snap_run.sh <restore_mode> <seed> <n_pre> <n_post>
R="/c/Users/TKA/Desktop/mini moto"; G="/c/Program Files (x86)/Steam/steamapps/common/Mini Motorways"
P=$(grep -o "Process ID: [0-9]*" "$R/logs/instr_snap.log" 2>/dev/null | grep -o "[0-9]*$" | head -1)
[ -n "$P" ] && taskkill //PID $P //F >/dev/null 2>&1
sleep 3
cp "$R/game_plugin/DoorstopBootstrap/bin/Release/net472/MiniMotorwaysBootstrap.dll" "$G/Bootstrap/MiniMotorwaysBootstrap_diag.dll"
: > "$R/logs/instr_snap.log"
cd "$G"
export MINI_MOTORWAYS_IPC_DIR='C:\Users\TKA\Desktop\mini moto\logs\ipc_snap' MINI_MOTORWAYS_LOG_PATH='C:\Users\TKA\Desktop\mini moto\logs\instr_snap.log'
(./"Mini Motorways.exe" -batchmode -nographics --doorstop-target-assembly 'Bootstrap\MiniMotorwaysBootstrap_diag.dll' > /dev/null 2>&1 &)
until grep -q "File-based IPC ready" "$R/logs/instr_snap.log"; do sleep 2; done
cd "$R/python"; python "${PROBE:-snapshot_probe.py}" "$2" "$3" "$4" "$1" 2>&1 | grep -v -i "gym\|migrat"
