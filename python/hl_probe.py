import os, subprocess, time, sys
import numpy as np
D = r"C:\Users\TKA\Desktop\mini moto\logs\ipc_hl1"
L = r"C:\Users\TKA\Desktop\mini moto\logs\instrumentation_hl1.log"
for f in ("request.json", "response.json"):
    try: os.remove(os.path.join(D, f))
    except OSError: pass
open(L, "w").close()
env = os.environ.copy(); env["MINI_MOTORWAYS_IPC_DIR"] = D; env["MINI_MOTORWAYS_LOG_PATH"] = L
GE = r"C:\Program Files (x86)\Steam\steamapps\common\Mini Motorways\Mini Motorways.exe"
GD = r"C:\Program Files (x86)\Steam\steamapps\common\Mini Motorways"
p = subprocess.Popen([GE, "-batchmode", "-nographics"], cwd=GD, env=env)
t0 = time.time(); ok = False
while time.time() - t0 < 120:
    if p.poll() is not None: print("PROCESS EXITED early, code", p.returncode); break
    try:
        if "File-based IPC ready" in open(L, encoding="utf-8", errors="replace").read(): ok = True; break
    except OSError: pass
    time.sleep(2)
print("headless instance IPC ready:", ok, f"({time.time()-t0:.0f}s)")
if ok:
    from mini_motorways_env import IpcClient
    c = IpcClient(request_path=os.path.join(D, "request.json"), response_path=os.path.join(D, "response.json"), timeout=60)
    r = c.send("RESET", seed=31, max_decisions=600, decision_delta=1.0)
    rng = np.random.default_rng(1); ts = []
    for i in range(120):
        t = time.perf_counter(); r = c.send("STEP", action=int(rng.choice(r["legalActionIndices"]))); ts.append(time.perf_counter()-t)
    print(f"HEADLESS STEP: median {np.median(ts)*1000:.0f} ms -> {1/np.median(ts):.1f} steps/s   (pid {p.pid})")
