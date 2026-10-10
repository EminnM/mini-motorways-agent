"""Research probe: is exact native snapshot/restore (incl. simulation PRNG) feasible?
Talks to a dedicated game instance (logs/ipc_snap) running the DIAG_SNAPSHOT op."""
import json
import sys

sys.path.insert(0, ".")
from mini_motorways_env import IpcClient

IPC = r"C:\Users\TKA\Desktop\mini moto\logs\ipc_snap"
c = IpcClient(IPC + r"\request.json", IPC + r"\response.json", timeout=300)
c.send("HELLO")
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 182150
n_pre = int(sys.argv[2]) if len(sys.argv) > 2 else 30
n_post = int(sys.argv[3]) if len(sys.argv) > 3 else 20
restore_mode = int(sys.argv[4]) if len(sys.argv) > 4 else 0
c.send("RESET", seed=seed, max_decisions=1000, decision_delta=1.0)
r = c.send("DIAG_SNAPSHOT", n_pre=n_pre, n_post=n_post, restore_mode=restore_mode)
print(json.dumps(r, indent=1))
