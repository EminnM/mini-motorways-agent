"""Summarise counterfactual-probe JSONL files (see cf_probe.py).

For every anchor, A_h(a) = dScore_h(a) - dScore_h(WAIT): the extra native score the
first action a earns by horizon h, relative to doing nothing, with WAIT as the
continuation (or the oracle follow-ups when the run used --followups).
Usage: python cf_analyze.py file1.jsonl [file2.jsonl ...]"""
import json
import sys
from collections import defaultdict


def load(paths):
    recs = []
    for p in paths:
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if line:
                r = json.loads(line)
                if "cands" in r:
                    r["_file"] = p
                    recs.append(r)
    return recs


def main(paths):
    recs = load(paths)
    if not recs:
        print("no usable records")
        return
    hs = recs[0]["horizons"]
    print(f"{len(recs)} anchors from {len(set(r['seed'] for r in recs))} seeds; horizons={hs}")
    print(f"start modes: {sorted(set((r['startMode'], r['startGap'], r.get('followups', False)) for r in recs))}")

    # 1. determinism of the restore itself
    dup_ok = dup_n = 0
    for r in recs:
        w = [c for c in r["cands"] if c["cat"] == "wait"][0]
        d = [c for c in r["cands"] if c["cat"] == "wait_dup"][0]
        dup_n += 1
        dup_ok += (w["dScore"] == d["dScore"] and w["phiEnd"] == d["phiEnd"])
    print(f"\nrestore determinism (WAIT vs duplicate WAIT branch): {dup_ok}/{dup_n} identical")

    # 2. per-horizon effect statistics, by category
    cats = defaultdict(list)   # cat -> list of (A vector, dphi1, dphiEnd, r)
    for r in recs:
        w = [c for c in r["cands"] if c["cat"] == "wait"][0]
        for c in r["cands"]:
            if c["cat"] in ("wait", "wait_dup"):
                continue
            A = [c["dScore"][i] - w["dScore"][i] for i in range(len(hs))]
            cats[c["cat"]].append((A, c["phi1"] - w["phi1"], c["phiEnd"] - w["phiEnd"], r))
    print("\nshare of candidate first-actions whose score differs from WAIT by horizon h (any effect), and mean A_h")
    head = "category".ljust(14) + "n".rjust(5) + "".join(f"  h={h}".rjust(9) for h in hs)
    print(head)
    for cat, rows in sorted(cats.items()):
        line = cat.ljust(14) + str(len(rows)).rjust(5)
        for i, h in enumerate(hs):
            nz = sum(1 for A, *_ in rows if A[i] != 0) / len(rows)
            line += f"{nz:8.0%} "
        print(line)
    print("\nmean A_h (signed) by category")
    print(head)
    for cat, rows in sorted(cats.items()):
        line = cat.ljust(14) + str(len(rows)).rjust(5)
        for i, h in enumerate(hs):
            m = sum(A[i] for A, *_ in rows) / len(rows)
            line += f"{m:+8.2f} "
        print(line)

    # 3. invisibility at the 1-step reward: effects that exist at long horizons but are zero at h=1..3
    allrows = [x for rows in cats.values() for x in rows]
    long_i = len(hs) - 1
    eff_long = [x for x in allrows if x[0][long_i] != 0]
    print(f"\ncandidates with a non-zero effect by h={hs[long_i]}: {len(eff_long)}/{len(allrows)} ({len(eff_long)/max(1,len(allrows)):.0%})")
    for i, h in enumerate(hs[:-1]):
        vis = sum(1 for A, *_ in eff_long if A[i] != 0)
        print(f"   of those, already visible at h={h}: {vis} ({vis/max(1,len(eff_long)):.0%})")

    # onset lag distribution
    onset = defaultdict(int)
    for A, *_ in eff_long:
        first = next((hs[i] for i in range(len(hs)) if A[i] != 0), None)
        onset[first] += 1
    print("first horizon with any score difference (among effective candidates):", dict(sorted(onset.items())))

    # 4. what the shaping potential sees vs what the score eventually does
    print("\nPhi-change vs long-horizon effect (A at h=%d):" % hs[long_i])
    buckets = defaultdict(list)
    for A, dphi1, dphiEnd, r in allrows:
        key = "phi up" if dphi1 > 0 else ("phi down" if dphi1 < 0 else "phi same")
        buckets[key].append(A[long_i])
    for key, v in sorted(buckets.items()):
        pos = sum(1 for x in v if x > 0); neg = sum(1 for x in v if x < 0)
        print(f"   {key:9s} n={len(v):4d}  mean A={sum(v)/len(v):+.2f}  positive={pos}  negative={neg}  zero={len(v)-pos-neg}")

    # 5. big movers
    print("\nlargest counterfactual effects (|A| at the longest horizon):")
    ranked = sorted(allrows, key=lambda x: -abs(x[0][long_i]))[:8]
    for A, dphi1, dphiEnd, r in ranked:
        c = None
        # find the matching candidate record for printing
        for cc in r["cands"]:
            w = [q for q in r["cands"] if q["cat"] == "wait"][0]
            if cc["cat"] not in ("wait", "wait_dup") and [cc["dScore"][i] - w["dScore"][i] for i in range(len(hs))] == A:
                c = cc
                break
        print(f"   seed {r['seed']} anchor {r['anchor']} gap {r['startGap']}: {c['cat']:12s} {c['a']:22s} A={A} dPhi1={dphi1:+d}")


if __name__ == "__main__":
    main(sys.argv[1:])
