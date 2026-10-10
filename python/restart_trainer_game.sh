#!/bin/bash
# Restart ONLY the trainer-side game instance (default IPC dir / default log),
# identified by the PID recorded in logs/instrumentation.log -- never the stream's.
L="/c/Users/TKA/Desktop/mini moto/logs/instrumentation.log"
PID=$(grep -m1 "Process ID" "$L" | sed 's/.*Process ID: //' | tr -d '\r ')
[ -n "$PID" ] && powershell -Command "Stop-Process -Id $PID -Force -ErrorAction SilentlyContinue" && echo "stopped trainer game $PID"
sleep 2
cp "/c/Users/TKA/Desktop/mini moto/game_plugin/DoorstopBootstrap/bin/Release/net472/MiniMotorwaysBootstrap.dll" "/c/Program Files (x86)/Steam/steamapps/common/Mini Motorways/Bootstrap/MiniMotorwaysBootstrap.dll" && echo "DLL deployed"
> "$L"
rm -f "/c/Users/TKA/Desktop/mini moto/logs/ipc/request.json" "/c/Users/TKA/Desktop/mini moto/logs/ipc/response.json"
powershell -Command "Start-Process -FilePath 'C:\Program Files (x86)\Steam\steamapps\common\Mini Motorways\Mini Motorways.exe' -WorkingDirectory 'C:\Program Files (x86)\Steam\steamapps\common\Mini Motorways'" 2>/dev/null
for i in $(seq 1 60); do grep -q "File-based IPC ready" "$L" 2>/dev/null && { echo "IPC_READY"; exit 0; }; sleep 2; done
echo "IPC NOT READY"; exit 1
