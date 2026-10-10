"""Spawn-causality pilot driver (research only; separate probe game instance logs/ipc_snap).
Per (seed, anchor): replay a WAIT main line to the anchor, snapshot, run branches A (WAIT), A2 (duplicate WAIT),
B (local disconnected road segment), C (far control segment); every new house/destination is logged."""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mini_motorways_env import IpcClient
from eval_seeds import EVAL_SEEDS

IPC = r"C:\Users\TKA\Desktop\mini moto\logs\ipc_snap"
ap = argparse.ArgumentParser()
ap.add_argument("--seeds", default=",".join(str(s) for s in EVAL_SEEDS))
ap.add_argument("--anchors", default="30,60")
ap.add_argument("--horizon", type=int, default=180)
ap.add_argument("--start-mode", default="CONNECTED_START")
ap.add_argument("--seg-len", type=int, default=3)
ap.add_argument("--out", required=True)
a = ap.parse_args()
c = IpcClient(IPC + r"\request.json", IPC + r"\response.json", timeout=900)
c.send("HELLO")
with open(a.out, "a") as f:
    for seed in [int(x) for x in a.seeds.split(",")]:
        for anchor in [int(x) for x in a.anchors.split(",")]:
            t0 = time.time()
            c.send("RESET", seed=seed, max_decisions=100000, decision_delta=1.0, start_mode=a.start_mode, start_gap=0)
            rec = c.send("DIAG_SPAWN", n_pre=anchor, horizon=a.horizon, seg_len=a.seg_len, cand_seed=seed)
            rec.pop("id", None)
            rec.update(seed=seed, anchor=anchor, startMode=a.start_mode, wallSeconds=round(time.time() - t0, 1))
            f.write(json.dumps(rec) + "\n"); f.flush()
            msg = rec.get("skipped") or rec.get("error", "")[:120] or " ".join(f"{k}:{len(v['spawns'])}sp" for k, v in rec["branches"].items())
            print(f"seed {seed} anchor {anchor}: {msg} ({rec['wallSeconds']}s)", flush=True)
print("done")
