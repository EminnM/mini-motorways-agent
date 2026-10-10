"""S8: re-run the single TEST evaluation of each arm's ALREADY-SELECTED checkpoint with the corrected native-legality class definition (selection was done on validation before;
nothing here feeds back into selection or training). Success counts must reproduce the originals exactly (same seed-0 evaluation)."""
import gzip, json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
import torch
torch.set_num_threads(2)
from sb3_contrib import MaskablePPO
from synthetic_rl import s8_eval as E
arm = sys.argv[1]
d = os.path.join(ROOT, "research", "s8_long_route_training", "runs", f"arm_{arm}_s11")
old = json.load(open(os.path.join(d, "final_test.json")))
test = [json.loads(l) for l in open(os.path.join(ROOT, "research", "s7_long_route", "manifests", "s7_test.jsonl"))]
m = MaskablePPO.load(os.path.join(d, "checkpoints", old["checkpoint"]), device="cpu")
res, rows_all, same = {}, {}, {}
for mode, det in (("deterministic", True), ("stochastic", False)):
    rows = E.evaluate("model", test, model=m, deterministic=det, seed=0)
    res[mode] = E.summarize(rows); rows_all[mode] = rows
    same[mode] = {k: (res[mode][k]["success"] == old["summary"][mode][k]["success"]) for k in ("2", "3", "5", "7", "10")}
json.dump({"checkpoint": old["checkpoint"], "sha256": old["sha256"], "summary": res, "success_reproduces_original_run": same, "legality_class_definition": "v2 (corrected)"}, open(os.path.join(d, "final_test_v2.json"), "w"), indent=1)
with gzip.open(os.path.join(d, "final_test_rows_v2.json.gz"), "wt") as f: json.dump(rows_all, f)
print(arm, "reproduced:", same)
