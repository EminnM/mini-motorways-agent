#!/bin/bash
# Starts (or restarts) the dedicated research probe game instance (logs/ipc_snap3) on the
# probe DLL. Never touches the training instances.
R="/c/Users/TKA/Desktop/mini moto"; G="/c/Program Files (x86)/Steam/steamapps/common/Mini Motorways"
P=$(grep -o "Process ID: [0-9]*" "$R/logs/instr_snap3.log" 2>/dev/null | grep -o "[0-9]*$" | head -1)
[ -n "$P" ] && taskkill //PID $P //F >/dev/null 2>&1
sleep 3
cp "$R/game_plugin/DoorstopBootstrap/bin/Release/net472/MiniMotorwaysBootstrap.dll" "$G/Bootstrap/MiniMotorwaysBootstrap_diag.dll"
: > "$R/logs/instr_snap3.log"
cd "$G"
export MINI_MOTORWAYS_IPC_DIR='C:\Users\TKA\Desktop\mini moto\logs\ipc_snap3' MINI_MOTORWAYS_LOG_PATH='C:\Users\TKA\Desktop\mini moto\logs\instr_snap3.log'
(./"Mini Motorways.exe" -batchmode -nographics --doorstop-target-assembly 'Bootstrap\MiniMotorwaysBootstrap_diag.dll' > /dev/null 2>&1 &)
until grep -q "File-based IPC ready" "$R/logs/instr_snap3.log"; do sleep 2; done
echo "probe instance ready"
