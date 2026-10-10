"""Stratified S2 pilot selection from the captured native validation tasks (ambiguity x orientation x gap position)."""
import json
import os
import random
from collections import defaultdict

ROOT = r"C:\Users\TKA\Desktop\mini moto"
OUT = os.path.join(ROOT, "research", "synthetic_to_native")


def main(per_cell=3, seed=0):
    tasks = {t["task_id"]: t for t in (json.loads(l) for l in open(os.path.join(ROOT, "research", "gap1", "manifests", "validation.jsonl")))}
    caps = {}
    for fn in os.listdir(OUT):
        if fn.startswith("capture_validation_") and fn.endswith(".jsonl"):
            for l in open(os.path.join(OUT, fn)):
                if l.strip():
                    c = json.loads(l)
                    caps[c["task_id"]] = c
    cells = defaultdict(list)
    for tid, c in caps.items():
        t = tasks[tid]
        amb = "unique" if c["ambiguity"]["category"] == "unique" else "ambiguous"
        pos = "early" if t["gap_position_in_path"] / max(t["path_length_edits"], 1) < 0.5 else "late"
        cells[(amb, t["gap_orientation"], pos)].append(tid)
    rng = random.Random(seed)
    chosen = []
    for k in sorted(cells):
        ids = sorted(cells[k])
        rng.shuffle(ids)
        chosen += [(k, i) for i in ids[:per_cell]]
    os.makedirs(os.path.join(OUT, "manifest"), exist_ok=True)
    with open(os.path.join(OUT, "manifest", "s2_pilot.jsonl"), "w", encoding="utf-8") as f:
        for k, tid in chosen:
            rec = dict(tasks[tid])
            rec["s2_flags"] = {"carpark_category": caps[tid]["ambiguity"]["category"], "ambiguity_stratum": k[0], "orientation": k[1], "gap_position_stratum": k[2],
                               "n_adjacent_road_tiles": caps[tid]["ambiguity"]["n_adjacent_road_tiles"]}
            f.write(json.dumps(rec) + "\n")
    print({str(k): len(v) for k, v in sorted(cells.items())})
    print("selected", len(chosen))


if __name__ == "__main__":
    main()
