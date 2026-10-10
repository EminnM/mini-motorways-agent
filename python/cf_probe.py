"""Offline counterfactual-action probe (NOT wired into PPO).

For each (seed, anchor decision): replay the episode with a fixed scripted main policy
up to the anchor, snapshot the native simulation once, then for every candidate first
action restore the snapshot, apply it, continue with WAIT and record native score gained
at several horizons. Output: JSONL, one record per (seed, anchor)."""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mini_motorways_env import IpcClient

IPC = r"C:\Users\TKA\Desktop\mini moto\logs\ipc_snap"

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", default="182150,229664,205907")
ap.add_argument("--anchors", default="60,100,140,180,220,260")
ap.add_argument("--start-mode", default="CONNECTED_START")
ap.add_argument("--start-gap", type=int, default=0)
ap.add_argument("--k-rand", type=int, default=8)
ap.add_argument("--k-adj", type=int, default=12)
ap.add_argument("--k-rem", type=int, default=4)
ap.add_argument("--horizons", default="1,3,5,10,20,40,60,90")
ap.add_argument("--main-variant", type=int, default=0)
ap.add_argument("--followups", action="store_true", help="after the candidate, apply the remaining omitted (oracle) edits one per step instead of pure WAIT")
ap.add_argument("--out", required=True)
a = ap.parse_args()

horizons = [int(x) for x in a.horizons.split(",")]
c = IpcClient(IPC + r"\request.json", IPC + r"\response.json", timeout=900)
c.send("HELLO")
n = 0
with open(a.out, "a") as f:
    for seed in [int(x) for x in a.seeds.split(",")]:
        for anchor in [int(x) for x in a.anchors.split(",")]:
            t0 = time.time()
            r = c.send("RESET", seed=seed, max_decisions=100000, decision_delta=1.0, start_mode=a.start_mode, start_gap=a.start_gap)
            init = r.get("initInfo") or {}
            if isinstance(init, str):
                try: init = json.loads(init)
                except Exception: init = {}
            extra = [e for e in (init.get("omittedEdits") or "").split() if e]
            rec = c.send("DIAG_CF", n_pre=anchor, main_variant=a.main_variant, k_rand=a.k_rand, k_adj=a.k_adj, k_rem=a.k_rem,
                         h_max=max(horizons), horizons=horizons, cand_seed=seed, extra=extra, followups=1 if a.followups else 0)
            rec["omittedEdits"], rec["followups"] = extra, a.followups
            rec["seed"], rec["anchor"], rec["startMode"], rec["startGap"], rec["wallSeconds"] = seed, anchor, a.start_mode, a.start_gap, round(time.time() - t0, 1)
            rec.pop("id", None)
            f.write(json.dumps(rec) + "\n"); f.flush()
            n += 1
            print(f"seed {seed} anchor {anchor}: " + (rec.get("skipped") or f"{len(rec['cands'])} candidates, phi0={rec['phi0']}, scoreStart={rec['scoreStart']}") + f" ({rec['wallSeconds']}s)", flush=True)
print("done", n, "records ->", a.out)
