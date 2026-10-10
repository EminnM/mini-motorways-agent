"""S7 Gate B: zero-shot baselines on the frozen S7 TEST manifest (no training, no gradient update).
   python experiments/s7_baselines.py --who wait_random | cnn11 | cnn12 | cnn13
Preregistered protocol: research/s7_long_route/PREREGISTERED_PROTOCOL.json (horizon 10*k). Deterministic x1, stochastic x2 seeds, RANDOM x5 seeds."""
import argparse, json, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
import torch
torch.set_num_threads(1)
from synthetic_rl import s7, s7_eval as E
OUT = os.path.join(ROOT, "research", "s7_long_route")
CK = {"cnn11": "2026-10-08_140406_syn_s1_cnn_s11/checkpoints/step_050000.zip", "cnn12": "2026-10-08_140406_syn_s1_cnn_s12/checkpoints/step_050000.zip",
      "cnn13": "2026-10-08_140406_syn_s1_cnn_s13/checkpoints/final.zip"}
ap = argparse.ArgumentParser(); ap.add_argument("--who", required=True); ap.add_argument("--split", default="test"); a = ap.parse_args()
tasks = [json.loads(l) for l in open(os.path.join(OUT, "manifests", f"s7_{a.split}.jsonl"))]
res = {"split": a.split, "n_tasks": len(tasks), "runs": {}}
t0 = time.time()
if a.who == "wait_random":
    res["runs"]["wait"] = E.summarize(E.evaluate("wait", tasks))
    rs = [E.evaluate("random", tasks, seed=s) for s in range(5)]
    res["runs"]["random_seed0"] = E.summarize(rs[0])
    res["random_success_by_k_per_seed"] = {str(k): [E.summarize(r)[k]["success"] for r in rs] for k in s7.LENGTHS}
    json.dump(rs[0], open(os.path.join(OUT, "rows_random_seed0.json"), "w"))
else:
    from sb3_contrib import MaskablePPO
    import hashlib
    p = os.path.join(ROOT, "runs", CK[a.who]); m = MaskablePPO.load(p, device="cpu")
    res["checkpoint"] = CK[a.who]; res["sha256"] = hashlib.sha256(open(p, "rb").read()).hexdigest()
    rows = E.evaluate("model", tasks, model=m, deterministic=True)
    res["runs"]["deterministic"] = E.summarize(rows)
    json.dump(rows, open(os.path.join(OUT, f"rows_{a.who}_det.json"), "w"))
    for sd in (0, 1):
        rows = E.evaluate("model", tasks, model=m, deterministic=False, seed=sd)
        res["runs"][f"stochastic_seed{sd}"] = E.summarize(rows)
        json.dump(rows, open(os.path.join(OUT, f"rows_{a.who}_sto{sd}.json"), "w"))
res["wall_s"] = round(time.time() - t0, 1)
json.dump(res, open(os.path.join(OUT, f"baselines_{a.who}.json"), "w"), indent=1)
print(a.who, "done", res["wall_s"], "s")
