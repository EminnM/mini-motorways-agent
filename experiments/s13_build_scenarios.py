"""S13b: build the native-grounded scenario library from the instrumented native episodes (read-only data analysis, no game).
A scenario = a game seed's initial structures + one realised spawn timeline (time, structure tiles, groups, free driveway / stub edges).
Sources: S13 replicate (has the reset header) + S12 control / conserve episodes of the SAME game seeds (same initial structures, different realised timelines).
   python experiments/s13_build_scenarios.py"""
import gzip
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import s11_guard as G

SEEDS = json.load(open(os.path.join(ROOT, "research", "s9_native_transfer", "seeds.json")))["seeds"]
OUT = os.path.join(ROOT, "research", "s13_multi_house")
SRC = {"replicate": "research/s13_multi_house/raw/replicate_{s}.json.gz", "s12_control": "research/s12_concrete_conservation/raw/control_{s}.json.gz", "s12_conserve": "research/s12_concrete_conservation/raw/conserve_{s}.json.gz"}
TEST_SEED_IDX = (2, 5, 8, 11)          # fixed rule: held-out game seeds (every third seed), declared before any synthetic training or evaluation


def events_of(d):
    ev = []
    F = {tuple(e) for e in d["header"]["F0"]}
    for x in d["steps"]:
        ns = x["new_struct"]
        if not ns:
            continue
        tiles = {(s[0], s[1]) for s in ns}
        free = [e for e in (tuple(e) for e in x["fa"]) if set(G.ends(e)) & tiles]
        ev.append({"t": x["i"] + 1, "struct": [[s[0], s[1], s[2], s[3]] for s in ns], "free_edges": [list(e) for e in free]})
    return ev


def main():
    scen = []
    for k, seed in enumerate(SEEDS):
        rep = json.load(gzip.open(os.path.join(ROOT, SRC["replicate"].format(s=seed)), "rt"))
        s0 = rep["header"]["structures0"]
        F0 = rep["header"]["F0"]
        for name, path in SRC.items():
            d = rep if name == "replicate" else json.load(gzip.open(os.path.join(ROOT, path.format(s=seed)), "rt"))
            ev = events_of(d)
            scen.append({"id": f"{seed}_{name}", "game_seed": seed, "seed_index": k, "source": name, "split": "test" if k in TEST_SEED_IDX else "train", "structures0": s0, "free_edges0": F0, "events": ev,
                         "native_episode_decisions": len(d["steps"]), "native_deliveries": d["steps"][-1]["score_cum"]})
    json.dump({"test_seed_indices": list(TEST_SEED_IDX), "concrete_schedule": {"initial": 25, "weekly_topups": {"decisions": [141, 281, 421, 561], "amounts": [20, 30, 10, 20]}}, "scenarios": scen},
              open(os.path.join(OUT, "scenarios.json"), "w"))
    print(len(scen), "scenarios;", sum(1 for s in scen if s["split"] == "test"), "test;", "events per scenario median", sorted(len(s["events"]) for s in scen)[len(scen) // 2])


if __name__ == "__main__":
    main()
