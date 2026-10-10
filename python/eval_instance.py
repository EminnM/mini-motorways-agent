"""Launch (or reuse) the dedicated headless EVALUATION game instance."""
import os, subprocess, sys, time
D = r"C:\Users\TKA\Desktop\mini moto\logs\ipc_eval"
L = r"C:\Users\TKA\Desktop\mini moto\logs\instrumentation_eval.log"
GE = r"C:\Program Files (x86)\Steam\steamapps\common\Mini Motorways\Mini Motorways.exe"
GD = r"C:\Program Files (x86)\Steam\steamapps\common\Mini Motorways"
for f in ("request.json", "response.json"):
    try: os.remove(os.path.join(D, f))
    except OSError: pass
open(L, "w").close()
env = os.environ.copy(); env["MINI_MOTORWAYS_IPC_DIR"] = D; env["MINI_MOTORWAYS_LOG_PATH"] = L
subprocess.Popen([GE, "-batchmode", "-nographics"], cwd=GD, env=env)
t0 = time.time()
while time.time() - t0 < 150:
    try:
        if "File-based IPC ready" in open(L, encoding="utf-8", errors="replace").read():
            print("eval instance ready"); sys.exit(0)
    except OSError: pass
    time.sleep(2)
print("eval instance NOT ready"); sys.exit(1)
