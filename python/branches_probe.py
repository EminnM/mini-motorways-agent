"""Restored-branch determinism stress test (DIAG_BRANCHES). usage: branches_probe.py seed n_pre n_post"""
import json, sys
sys.path.insert(0, ".")
from mini_motorways_env import IpcClient
IPC = r"C:\Users\TKA\Desktop\mini moto\logs\ipc_snap"
c = IpcClient(IPC + r"\request.json", IPC + r"\response.json", timeout=600)
c.send("HELLO")
seed, n_pre, n_post = (int(a) for a in sys.argv[1:4])
c.send("RESET", seed=seed, max_decisions=5000, decision_delta=1.0)
dump_rel = int(sys.argv[4]) if len(sys.argv) > 4 else None
kw = {"dump_rel": dump_rel} if dump_rel is not None else {}
print(json.dumps(c.send("DIAG_BRANCHES", n_pre=n_pre, n_post=n_post, **kw)))
