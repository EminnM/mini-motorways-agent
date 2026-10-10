"""S7: turn the native captures into a small, self-contained scene file (terrain + per-scene destination/stub/build-window) and measure how well
the S7 rule-based legality reproduces the NATIVE ADD/REMOVE masks on those states.   python experiments/s7_build_scenes.py"""
import gzip
import json
import os
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "python"))
from synthetic_rl import layout as L  # noqa: E402

OUT = os.path.join(ROOT, "research", "s7_long_route")


def tmap(raw):
    return {(t["coord"][0], t["coord"][1]): t for t in raw["tiles"]}


def main():
    recs = [json.loads(l) for l in gzip.open(os.path.join(OUT, "native_normal_start_captures.jsonl.gz"), "rt")]
    base = tmap(recs[0]["first_house"]["raw"])
    water = sorted(k for k, t in base.items() if t["overWater"])
    scenes, seen = [], {}
    native_states = []
    for r in recs:
        raw = r["first_house"]["raw"]
        m = tmap(raw)
        legal = set(r["first_house"]["legal"])
        H = [k for k, t in m.items() if t["contentType"] == "House"][0]
        D = sorted(k for k, t in m.items() if t["contentType"] == "Destination")
        C = sorted(k for k, t in m.items() if t["contentType"] == "Carpark")
        edges = [(k[0], k[1], d) for k, t in m.items() for d in "ES" if t["road"][d] != "None"]
        stub = [e for e in edges if "Carpark" in (m[(e[0], e[1])]["contentType"], m[L.neighbor(*e)]["contentType"])][0]
        drive = [e for e in edges if "House" in (m[(e[0], e[1])]["contentType"], m[L.neighbor(*e)]["contentType"])][0]
        carpark_tile = (stub[0], stub[1]) if m[(stub[0], stub[1])]["contentType"] == "Carpark" else L.neighbor(*stub)
        goal = L.neighbor(*stub) if (stub[0], stub[1]) == carpark_tile else (stub[0], stub[1])
        pts = set()
        for a in legal:
            if 0 < a < L.UPGRADE_BASE:
                dec = L.decode_action(a)
                if dec[0] == "ADD":
                    pts |= {(dec[1], dec[2]), L.neighbor(dec[1], dec[2], dec[3])}
        key = (tuple(D), tuple(C), stub, tuple(sorted(pts)))
        if key not in seen:
            seen[key] = len(scenes)
            scenes.append({"scene_id": len(scenes), "destination": [list(x) for x in D], "carpark": [list(x) for x in C], "stub_edge": list(stub), "carpark_tile": list(carpark_tile),
                           "goal_tile": list(goal), "window_tiles": [list(p) for p in sorted(pts)], "native_seeds": []})
        scenes[seen[key]]["native_seeds"].append(r["seed"])
        native_states.append({"seed": r["seed"], "scene_id": seen[key], "house": list(H), "driveway": list(drive),
                              "native_legal_add": sorted(a for a in legal if 0 < a < L.UPGRADE_BASE and L.decode_action(a)[0] == "ADD"),
                              "native_legal_remove": sorted(a for a in legal if 0 < a < L.UPGRADE_BASE and L.decode_action(a)[0] == "REMOVE"),
                              "n_legal": len(legal), "reset_n_legal": len(r["reset"]["legal"])})
    json.dump({"water_tiles": [list(w) for w in water], "scenes": scenes, "concrete_total": 25, "bridges_total": 1}, open(os.path.join(OUT, "scenes.json"), "w"))
    json.dump(native_states, open(os.path.join(OUT, "native_states_summary.json"), "w"))
    print("unique scenes:", len(scenes), "from", len(recs), "captures; water tiles", len(water), "; window sizes", sorted(len(s["window_tiles"]) for s in scenes))


if __name__ == "__main__":
    main()
