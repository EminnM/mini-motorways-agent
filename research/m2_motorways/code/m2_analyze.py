"""M2 Stage B analysis: opportunities, placements, usage (vehicle counts on endpoint tiles), divergence point vs control, paired deliveries/survival.   python research/m2_motorways/code/m2_analyze.py"""
import gzip
import json
import os
import statistics as S
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from motorway_env import ROOT  # noqa: E402

OUT = os.path.join(ROOT, "research", "m2_motorways")
SEEDS = json.load(open(os.path.join(ROOT, "research", "s9_native_transfer", "seeds.json")))["seeds"]


def boot_ci(v, n=10000, seed=0):
    v = np.asarray(v, dtype=float)
    if len(v) == 0:
        return None
    rng = np.random.default_rng(seed)
    return [float(np.percentile([v[rng.integers(len(v), size=len(v))].mean() for _ in range(n)], q)) for q in (2.5, 97.5)]


def main():
    rows = []
    for s in SEEDS:
        pa, pb = (os.path.join(OUT, "raw", f"{a}_{s}.json.gz") for a in ("ruleM", "noMW"))
        if not (os.path.exists(pa) and os.path.exists(pb)):
            continue
        A, B = json.load(gzip.open(pa, "rt")), json.load(gzip.open(pb, "rt"))
        sa, sb = A["steps"], B["steps"]
        div = next((i for i, (x, y) in enumerate(zip(sa, sb)) if x["hash"] != y["hash"]), None)
        first_place = A["placements"][0]["i"] if A["placements"] else None
        usage = sum(x["veh_mw_tiles"] for x in sa[first_place + 1:]) if first_place is not None else 0
        veh_after = [sum(x["veh_total"] for x in st[first_place + 1:]) for st in (sa, sb)] if first_place is not None else [None, None]
        mw_states = sorted({m["state"] for x in sa for m in x["motorways"]})
        rows.append({"seed": s, "deliv_ruleM": sa[-1]["score_cum"], "deliv_noMW": sb[-1]["score_cum"], "dec_ruleM": len(sa), "dec_noMW": len(sb), "mw_avail_first": next((x["i"] for x in sa if x["mw_avail"] >= 1), None), "placements": [(p["i"], p["a"], p["b"], p["cheb"], p["saving"], p.get("motorwayOk")) for p in A["placements"]],
                     "opportunities": len(A["opportunities"]), "opps_with_eligible": sum(1 for o in A["opportunities"] if o["placed"]), "no_legal_ramp": sum(1 for o in A["opportunities"] if o.get("no_legal_ramp")), "best_saving_seen": max([o["best_saving"] for o in A["opportunities"]] or [0]),
                     "usage_veh_on_endpoints_after": usage, "veh_total_after_placement": veh_after, "divergence_decision": div, "first_placement": first_place, "mw_states": mw_states, "choices_identical": [c["slot"] for c in A["choices"]] == [c["slot"] for c in B["choices"]],
                     "d_deliv": sa[-1]["score_cum"] - sb[-1]["score_cum"], "d_dec": len(sa) - len(sb)})
    placed = [r for r in rows if r["placements"]]
    dd = [r["d_deliv"] for r in rows]
    ddp = [r["d_deliv"] for r in placed]
    res = {"seeds": len(rows), "seeds_with_upgrade": sum(1 for r in rows if r["mw_avail_first"] is not None), "seeds_with_placement": len(placed), "placements_total": sum(len(r["placements"]) for r in rows),
           "usage": {r["seed"]: r["usage_veh_on_endpoints_after"] for r in placed}, "paired_all": {"n": len(dd), "sum": sum(dd), "median": S.median(dd) if dd else None, "higher": sum(1 for x in dd if x > 0), "equal": sum(1 for x in dd if x == 0), "lower": sum(1 for x in dd if x < 0), "ci95_mean": boot_ci(dd)},
           "paired_placed": {"n": len(ddp), "values": ddp, "median": S.median(ddp) if ddp else None, "higher": sum(1 for x in ddp if x > 0), "equal": sum(1 for x in ddp if x == 0), "lower": sum(1 for x in ddp if x < 0), "ci95_mean": boot_ci(ddp)},
           "decisions_diff_placed": [r["d_dec"] for r in placed], "rows": rows}
    json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}))
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
