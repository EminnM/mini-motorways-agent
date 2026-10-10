"""S7: native normal-start captures vs the synthetic S7 task family (no game, no training).   python experiments/s7_native_comparison.py"""
import json
import os
import statistics
import sys
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
from synthetic_rl import layout as L  # noqa: E402
from synthetic_rl import s7  # noqa: E402

OUT = os.path.join(ROOT, "research", "s7_long_route")


def q(xs):
    xs = sorted(xs)
    return {"min": xs[0], "median": statistics.median(xs), "mean": round(statistics.fmean(xs), 2), "max": xs[-1], "n": len(xs)}


def main():
    W = s7.World.get()
    ns = json.load(open(os.path.join(OUT, "native_states_summary.json")))
    nat = {"scenes": len(W.scenes), "captures": len(ns)}
    ks, md, legal_add, legal_rm, rel = [], [], [], [], []
    for n in ns:
        s = W.scenes[n["scene_id"]]
        h = tuple(n["house"])
        dr = tuple(n["driveway"])
        k = s7.remaining_distance(W, s, h, {s["stub"], dr})
        ks.append(k)
        g = s["goal"]
        md.append(abs(h[0] - g[0]) + abs(h[1] - g[1]))
        legal_add.append(len(n["native_legal_add"]))
        legal_rm.append(len(n["native_legal_remove"]))
        xs = [t[0] for t in s["win"]]
        ys = [t[1] for t in s["win"]]
        rel.append(((h[0] - min(xs)) / max(max(xs) - min(xs), 1), (h[1] - min(ys)) / max(max(ys) - min(ys), 1)))
    reach = [k for k in ks if k < 10 ** 6]
    nat.update({"required_adds_water_impassable_model": q(reach), "unreachable_without_bridge": len(ks) - len(reach), "required_adds_values": sorted(ks),
                "house_to_goal_manhattan": q(md), "legal_adds_at_first_house_state": q(legal_add), "legal_removes": q(legal_rm),
                "legal_total_incl_wait": q([n["n_legal"] for n in ns]), "distinct_house_tiles": len({tuple(n["house"]) for n in ns}), "distinct_scenes": len({n["scene_id"] for n in ns}),
                "house_rel_pos_in_window_mean": [round(float(np.mean([r[0] for r in rel])), 2), round(float(np.mean([r[1] for r in rel])), 2)],
                "window_tiles": q([len(s["win"]) for s in W.scenes]), "water_fraction": round(len(W.water) / L.N_TILES, 4)})
    sets = {sp: {k: [json.loads(l) for l in open(os.path.join(OUT, "manifests", f"s7_{sp}.jsonl")) if f'"k_min": {k},' in l] for k in s7.LENGTHS} for sp in ("train", "validation", "test")}
    syn = {}
    for k in s7.LENGTHS:
        tasks = sets["test"][k]
        legal, rm, dist, scenes, rel2, nalt = [], [], [], set(), [], []
        env = s7.LongRouteEnv(tasks, sampling="sequential")
        for i, t in enumerate(tasks):
            env.reset(options={"task_index": i})
            m = env.action_masks()
            ids = np.flatnonzero(m)
            legal.append(int(sum(1 for a in ids if 0 < a < L.UPGRADE_BASE and L.decode_action(int(a))[0] == "ADD")))
            rm.append(int(sum(1 for a in ids if 0 < a < L.UPGRADE_BASE and L.decode_action(int(a))[0] == "REMOVE")))
            h, g = tuple(t["house"]), env.scene["goal"]
            dist.append(abs(h[0] - g[0]) + abs(h[1] - g[1]))
            scenes.add(t["scene_id"])
            xs = [x[0] for x in env.scene["win"]]
            ys = [x[1] for x in env.scene["win"]]
            rel2.append(((h[0] - min(xs)) / max(max(xs) - min(xs), 1), (h[1] - min(ys)) / max(max(ys) - min(ys), 1)))
        syn[k] = {"n_test": len(tasks), "legal_adds": q(legal), "legal_removes": q(rm), "house_to_goal_manhattan": q(dist), "distinct_scenes": len(scenes),
                  "house_rel_pos_mean": [round(float(np.mean([r[0] for r in rel2])), 2), round(float(np.mean([r[1] for r in rel2])), 2)]}
    out = {"native_24_captures": nat, "synthetic_test_by_required_length": syn,
           "differences": [
               "native placements need 4-12 required ADDs (median 7; 1 of 24 would need the single native bridge across water, which this model does not allow); the S7 lengths 2/3/5/7/10 bracket the middle of that range, 2 and 3 are shorter than any native placement",
               "native houses occupy a few spawn slots (distinct house tiles / scenes above) whereas synthetic houses are drawn from every land tile of the window, so synthetic geometry is more varied than the native spawn process",
               "synthetic mask is a superset of the native one on window-boundary edges (precision ~93%) and misses ~4.8% of native-legal ADDs (one-water-endpoint 'bridge' edges)",
               "no spawning, time pressure, traffic, overcrowding game-over, upgrades or multiple houses/destinations; the policy observation channels for those are zero",
               "synthetic episodes end at success or at 10*k decisions; native episodes end at ~143 decisions via game over and have a -10 terminal penalty (not modelled; S7 reward is +1 / 0)"]}
    json.dump(out, open(os.path.join(OUT, "native_vs_synthetic.json"), "w"), indent=1)
    print(json.dumps(out["native_24_captures"], indent=0)[:1500])
    for k, v in syn.items():
        print(k, v)


if __name__ == "__main__":
    main()
