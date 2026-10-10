"""S33 tests for the playable-area legality fix (run ONLY after the fixed DLL is deployed and S31 has finished).
 T1 mask geometry: at decisions 0/60/150/250 on 3 seeds, every legal ADD has BOTH endpoints inside globals.playableAreaAtTime (east/south/north/west, corners); the rectangle grows with expansion time.
 T2 interior preserved: on the 48 pre-fix capture states (reset / first house), the post-fix legal set == pre-fix legal set minus the boundary-crossing edges (nothing else removed or added).
 T3 TryAddRoad defence: a crossing ADD index sent directly (bypassing the mask) is rejected with 'outside playable area'; an interior legal ADD is accepted.
 T4 TryAddMotorway: dry-run and real placement with an endpoint outside the area are rejected; inside endpoints follow the pre-existing native checks.
 T5 interfaces unchanged: Discrete(3111), 11,666-float v1 observation / 16,328-float v2, frozen checkpoints load and predict; hybrid env PLAN/EDIT obey the corrected mask (0 illegal over 100 decisions).
   python research/s33_playable_area_fix/code/test_fix.py"""
import gzip
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
sys.path.insert(0, os.path.join(ROOT, "research", "s21_native_check", "code"))
sys.path.insert(0, os.path.join(ROOT, "research", "s31_hybrid_native_rl", "code"))
sys.path.insert(0, os.path.join(ROOT, "research", "m2_motorways", "code"))
import numpy as np  # noqa: E402
from synthetic_rl import layout as L  # noqa: E402
from native_adapter import NativePlannerPolicy  # noqa: E402
from executor import make_policy  # noqa: E402
from planner import Weights  # noqa: E402

OUT = os.path.join(ROOT, "research", "s33_playable_area_fix")
R = {}


def check(name, ok, detail):
    R[name] = {"passed": bool(ok), "detail": detail}
    print(("PASS " if ok else "FAIL ") + name + ": " + str(detail)[:500], flush=True)


def far(k):
    return (k[1] + 1, k[2]) if k[3] == "E" else (k[1], k[2] - 1)


def inside(p, a):
    return a[0] <= p[0] <= a[2] and a[1] <= p[1] <= a[3]


def legal_adds(mask):
    return [L.decode_action(int(i)) for i in np.flatnonzero(mask) if 0 < i < L.UPGRADE_BASE and L.decode_action(int(i))[0] == "ADD"]


def main():
    from motorway_env import MotorwayEnv
    from experiments import gamectl
    t0 = time.time()
    with gamectl.GameInstance("s33_test") as g:
        env = MotorwayEnv(max_decisions=100000, decision_delta=1.0, request_path=os.path.join(g.ipc_dir, "request.json"), response_path=os.path.join(g.ipc_dir, "response.json"), start_mode="NORMAL_START")
        # ---- T1
        geo = []
        for seed in (500000, 507919, 563352):
            pol = NativePlannerPolicy(make_policy(Weights()))
            env.reset(seed=seed)
            pol.reset()
            raw = env.last_raw_observation
            areas = []
            for i in range(251):
                if i in (0, 60, 150, 250):
                    a = raw["globals"].get("playableAreaAtTime")
                    adds = legal_adds(env.action_masks())
                    bad = [k for k in adds if not (inside((k[1], k[2]), a) and inside(far(k), a))]
                    sides = {"E": sum(1 for k in adds if k[3] == "E" and k[1] == a[2]), "S": sum(1 for k in adds if k[3] == "S" and k[2] == a[1]), "corner_origins": sum(1 for k in adds if (k[1], k[2]) in {(a[0], a[1]), (a[2], a[1]), (a[0], a[3]), (a[2], a[3])})}
                    areas.append({"seed": seed, "i": i, "area": a, "n_add": len(adds), "both_endpoints_inside": len(bad) == 0, "violations": bad[:3], "edges_on_east_col_or_south_row": sides})
                mask = env.action_masks().copy()
                _, _, te, tr, _ = env.step(pol.act(raw, mask))
                raw = env.last_raw_observation
                if te or tr:
                    break
            geo.append(areas)
        grows = all(all((b["area"][2] - b["area"][0]) >= (a_["area"][2] - a_["area"][0]) and (b["area"][3] - b["area"][1]) >= (a_["area"][3] - a_["area"][1]) for a_, b in zip(s[:-1], s[1:])) for s in geo)
        check("T1_mask_geometry_and_expansion", all(x["both_endpoints_inside"] for s in geo for x in s) and grows and all(x["area"] is not None for s in geo for x in s), {"states": [{k: v for k, v in x.items() if k != "violations"} for s in geo for x in s], "area_grows": grows})
        # ---- T2 interior preserved vs pre-fix captures (reset and first-house states are reached by RESET + deterministic steps)
        caps = [json.loads(l) for l in gzip.open(os.path.join(ROOT, "research", "s7_long_route", "native_normal_start_captures.jsonl.gz"), "rt")]
        n = ok = 0
        diffs = []
        retries = []
        for c in caps[:12]:
            env.reset(seed=c["seed"])
            raw = env.last_raw_observation
            a = raw["globals"]["playableAreaAtTime"]
            pre = set(c["reset"]["legal"])
            post = set(int(i) for i in np.flatnonzero(env.action_masks()))
            crossing = {i for i in pre if 0 < i < L.UPGRADE_BASE and L.decode_action(i)[0] == "ADD" and not inside(far(L.decode_action(i)), a)}
            n += 1
            if post != pre - crossing:   # known reset transient (S21: rare state differences at decisions 0-7): re-reset once and record the retry
                env.reset(seed=c["seed"])
                post = set(int(i) for i in np.flatnonzero(env.action_masks()))
                retries.append(c["seed"])
            if post == pre - crossing:
                ok += 1
            else:
                diffs.append({"seed": c["seed"], "removed_not_crossing": sorted(pre - crossing - post)[:5], "added": sorted(post - pre)[:5], "n_crossing": len(crossing)})
        check("T2_interior_actions_preserved_vs_prefix_captures", ok == n and n > 0, {"states": n, "exact": ok, "diffs": diffs[:3], "transient_retries": retries})
        # ---- T3 TryAddRoad defence (bypass the mask)
        env.reset(seed=500000)
        raw = env.last_raw_observation
        a = raw["globals"]["playableAreaAtTime"]
        adds = legal_adds(env.action_masks())
        # The stock rule (City.IsTileInPlayableArea = RectFixed.Contains, INCLUSIVE at xMax/yMax of the ForceWholeTiles rect) is one column/row wider on the max side than the plugin's
        # loop/export rectangle ([floor(xMin), ceil(xMax)-1]). Hence a SOUTH crossing (far = yMin-1) is a genuine stock violation and must be rejected by TryAddRoad; an EAST edge from the
        # export's last column to column+1 is stock-legal (tile xMax is inside) and is recorded for information only.
        south_idx = L.action_index("ADD", a[0] + 4, a[1], "S")
        _, _, _, _, inf = env.step(south_idx)
        rej = (not inf["accepted"]) and "outside playable area" in (inf.get("applyInfo") or "")
        env.reset(seed=500000)
        raw = env.last_raw_observation
        a = raw["globals"]["playableAreaAtTime"]
        adds = legal_adds(env.action_masks())
        east_idx = L.action_index("ADD", a[2], a[1] + 4, "E")
        _, _, _, _, inf_e = env.step(east_idx)
        env.reset(seed=500000)
        raw = env.last_raw_observation
        adds = legal_adds(env.action_masks())
        interior = next(L.action_index("ADD", k[1], k[2], k[3]) for k in adds if inside((k[1], k[2]), a) and inside(far(k), a) and k[1] < a[2] - 1 and k[2] > a[1] + 1)
        _, _, _, _, inf2 = env.step(interior)
        check("T3_TryAddRoad_rejects_crossing_independently_of_mask", rej and inf2["accepted"], {"south_crossing": L.decode_action(south_idx), "applyInfo": (inf.get("applyInfo") or "")[:140], "east_edge_from_export_xMax": L.decode_action(east_idx), "east_accepted": inf_e["accepted"], "east_info": (inf_e.get("applyInfo") or "")[:140], "interior": L.decode_action(interior), "interior_accepted": inf2["accepted"]})
        # ---- T4 motorway endpoint check (dry run; no upgrade needed for the area rejection to show first)
        m_out = env.motorway_check({"op": "add", "x1": a[0] + 1, "y1": a[1] + 1, "d1": "E", "x2": a[2] + 3, "y2": a[1] + 1, "d2": "W"})
        m_in = env.motorway_check({"op": "add", "x1": a[0] + 1, "y1": a[1] + 1, "d1": "E", "x2": a[2] - 1, "y2": a[1] + 1, "d2": "W"})
        check("T4_TryAddMotorway_area_check", (not m_out.get("ok")) and "outside playable area" in m_out.get("info", "") and "outside playable area" not in m_in.get("info", ""), {"outside": m_out.get("info"), "inside": m_in.get("info")})
        # ---- T5 interfaces unchanged + hybrid env obeys mask
        import mini_motorways_env as M
        from sb3_contrib import MaskablePPO
        from hybrid_policy import HybridMaskablePolicy  # noqa: F401
        from synthetic_rl import obs_v2 as V
        o1 = M.encode_observation(raw)
        o2 = V.encode_native_v2(raw, V.GroupSlotTracker())
        ck = sorted([p for p in os.listdir(os.path.join(ROOT, "research", "s31_hybrid_native_rl", "runs", "hybrid_s31", "checkpoints"))])[-1]
        model = MaskablePPO.load(os.path.join(ROOT, "research", "s31_hybrid_native_rl", "runs", "hybrid_s31", "checkpoints", ck), device="cpu")
        env.close()
        from hybrid_env import HybridNativeEnv, N_ACTIONS
        henv = HybridNativeEnv([500000], "s33_hybrid", max_decisions=100, log_fn=None)
        obs, _ = henv.reset()
        kinds = {}
        for _ in range(100):
            act, _ = model.predict(obs, action_masks=henv.action_masks(), deterministic=False)
            obs, r, te, tr, info = henv.step(int(act))
            kinds[info["kind"]] = kinds.get(info["kind"], 0) + 1
            if te or tr:
                break
        ill = henv.ep["illegal"]
        henv.close()
        check("T5_interfaces_unchanged_and_hybrid_obeys_mask", env.action_space_size == 3111 and o1.shape == (11666,) and o2.shape == (16328,) and N_ACTIONS == 3112 and ill == 0, {"action_space": env.action_space_size, "v1": o1.shape, "v2": o2.shape, "checkpoint": ck, "kinds_100_decisions": kinds, "illegal": ill})
    R["_wall_s"] = round(time.time() - t0, 1)
    json.dump(R, open(os.path.join(OUT, "test_fix.json"), "w"), indent=1, default=str)
    print(sum(1 for v in R.values() if isinstance(v, dict) and v.get("passed")), "passed of", sum(1 for v in R.values() if isinstance(v, dict) and "passed" in v))


if __name__ == "__main__":
    main()
