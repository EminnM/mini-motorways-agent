"""S6 supplement: house / destination geometry of the two paired seeds (RESET only, no stepping) vs the road edges each arm ever added."""
import json, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from experiments import paths  # noqa
import mini_motorways_env as M
from experiments import gamectl
OUT = os.path.join(ROOT, "research", "native_cnn_anticycle")
t0 = time.time()
res = {}
with gamectl.GameInstance("s6geo") as g:
    env = M.MiniMotorwaysEnv(max_decisions=600, decision_delta=1.0, request_path=os.path.join(g.ipc_dir, "request.json"), response_path=os.path.join(g.ipc_dir, "response.json"), start_mode="NORMAL_START")
    for seed in (182150, 205907):
        env.reset(seed=seed)
        raw = env.last_raw_observation
        houses = [(t["coord"][0], t["coord"][1], t.get("groupIndex")) for t in raw["tiles"] if t["contentType"] == "House"]
        dests = [(t["coord"][0], t["coord"][1], t.get("groupIndex")) for t in raw["tiles"] if t["contentType"] == "Destination"]
        carp = [(t["coord"][0], t["coord"][1]) for t in raw["tiles"] if t["contentType"] == "Carpark"]
        roads = sorted({(t["coord"][0], t["coord"][1], d) for t in raw["tiles"] for d in ("E", "S") if t["road"][d] != "None"})
        from collections import Counter
        res[seed] = {"content_types": dict(Counter(t["contentType"] for t in raw["tiles"])), "tile_keys": sorted(raw["tiles"][0].keys()), "houses": houses, "destinations": dests, "carparks": carp, "initial_road_edges": roads}
    env.close()
print("native seconds", round(time.time() - t0, 1))
from synthetic_rl import layout as L
for seed in res:
    d = res[seed]
    for arm in ("A", "B"):
        tr = json.load(open(os.path.join(OUT, f"paired_{seed}_{arm}.json")))
        added = {tuple(s["decoded"][1:4]) for s in tr["steps"] if s["decoded"][0] == "ADD"}
        def dist(e, targets):
            if not targets:
                return 10**6
            return min(abs(e[0] - x) + abs(e[1] - y) for x, y, *_ in targets) if targets else None
        dd = [dist(e, d["destinations"] + [(c[0], c[1]) for c in d["carparks"]]) for e in added]
        dh = [dist(e, d["houses"]) for e in added]
        d[f"arm_{arm}"] = {"distinct_edges_ever_added": len(added), "min_manhattan_to_destination_or_carpark": min(dd) if dd else None, "median_to_destination": sorted(dd)[len(dd) // 2] if dd else None,
                           "min_to_house": min(dh) if dh else None, "median_to_house": sorted(dh)[len(dh) // 2] if dh else None,
                           "edges_within_2_of_destination": sum(1 for x in dd if x <= 2), "edges_within_2_of_a_house": sum(1 for x in dh if x <= 2)}
    d["min_house_to_dest_manhattan"] = min([abs(h[0] - x[0]) + abs(h[1] - x[1]) for h in d["houses"] for x in d["destinations"]] or [None])
json.dump(res, open(os.path.join(OUT, "geometry.json"), "w"), indent=1)
for seed, d in res.items():
    print(seed, d["content_types"], "houses", d["houses"], "dests", d["destinations"][:4], "n_dest_tiles", len(d["destinations"]), "carparks", d["carparks"], "min house->dest", d["min_house_to_dest_manhattan"], "initial roads", len(d["initial_road_edges"]))
    for arm in ("A", "B"):
        print("  ", arm, d[f"arm_{arm}"])
