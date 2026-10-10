"""Analyse spawn_probe.py output. Purely causal/spatial: no notion of a 'good' spawn.

Per snapshot the branches are A (WAIT control), A2 (independent duplicate WAIT), B (local disconnected road
segment), C (same geometry far away). Spawn identity = (type, group, x, y); order = (time, then key).
Primary window: spawns up to (and including) the step before the first branch reached the game-over condition,
because post-terminal timing is not trustworthy (see PROTOCOL notes). Secondary: the full horizon."""
import json
import math
import sys
from collections import defaultdict


def load(paths):
    out = []
    for p in paths:
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if line.startswith("{"):
                r = json.loads(line)
                if "branches" in r:
                    out.append(r)
    return out


def ident(s):
    return (s[1], s[2], s[3], s[4])


def seq(branch, tmax):
    return [s for s in branch["spawns"] if s[0] <= tmax and s[0] > 0]   # t==0 entries = spawned during the intervention window


def compare(ctrl, br, tmax):
    a, b = seq(ctrl, tmax), seq(br, tmax)
    ia, ib = [ident(s) for s in a], [ident(s) for s in b]
    res = {"diverged": ia != ib}
    first = None
    for i in range(max(len(a), len(b))):
        sa = a[i] if i < len(a) else None
        sb = b[i] if i < len(b) else None
        if (sa is None) != (sb is None) or (sa and ident(sa) != ident(sb)):
            ts = [x[0] for x in (sa, sb) if x]
            first = min(ts)
            break
    res["firstDivergence"] = first
    sa_set, sb_set = set(ia), set(ib)
    res["onlyInBranch"] = len(sb_set - sa_set)
    res["onlyInControl"] = len(sa_set - sb_set)
    res["nCtrl"], res["nBranch"] = len(ia), len(ib)
    d = []
    for sa, sb in zip(a, b):
        if sa[1] == sb[1]:    # same type
            d.append(abs(sa[3] - sb[3]) + abs(sa[4] - sb[4]))
    res["meanDisp"] = sum(d) / len(d) if d else 0.0
    res["maxDisp"] = max(d) if d else 0
    res["nDisplaced"] = sum(1 for x in d if x > 0)
    return res


def prng_div(ctrl, br):
    for c1, c2 in zip(ctrl["checkpoints"], br["checkpoints"]):
        if c1["prng"] != c2["prng"]:
            return c1["t"]
    return None


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0, c - h), min(1, c + h))


def binom_two_sided(k, n):
    if n == 0:
        return 1.0
    pk = [math.comb(n, i) * 0.5 ** n for i in range(n + 1)]
    return min(1.0, sum(p for p in pk if p <= pk[k] + 1e-12))


def main(paths):
    recs = load(paths)
    skipped = []
    ok = []
    for r in recs:
        ok.append(r)
    all_lines = []
    for p in paths:
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if line.startswith("{"):
                j = json.loads(line)
                if "skipped" in j or "error" in j:
                    skipped.append((j.get("seed"), j.get("anchor"), j.get("skipped") or j.get("error", "")[:60]))
    print(f"{len(ok)} snapshots analysed, {len(skipped)} skipped/errored: {skipped}")

    for window in ("pre-terminal", "full-180"):
        print(f"\n================ window: {window} ================")
        rows = defaultdict(list)
        dup_ident = dup_time = dup_prng = 0
        for r in ok:
            br = r["branches"]
            terms = [b["terminalStep"] for b in br.values() if b["terminalStep"] > 0]
            tmax = (min(terms) - 1) if (window == "pre-terminal" and terms) else r["horizon"]
            c = compare(br["A"], br["A2"], tmax)
            dup_ident += (not c["diverged"])
            sa, sb = seq(br["A"], tmax), seq(br["A2"], tmax)
            dup_time += ([s[0] for s in sa] == [s[0] for s in sb])
            dup_prng += (prng_div(br["A"], br["A2"]) is None)
            rows["B"].append((compare(br["A"], br["B"], tmax), prng_div(br["A"], br["B"]), r, tmax))
            rows["C"].append((compare(br["A"], br["C"], tmax), prng_div(br["A"], br["C"]), r, tmax))
        n = len(ok)
        print(f"duplicate-control sanity (A vs independently restored A2): identical spawn identities {dup_ident}/{n}; "
              f"identical spawn TIMES {dup_time}/{n}; identical PRNG at all checkpoints {dup_prng}/{n}")
        for name, label in (("B", "B local road"), ("C", "C far road (control)")):
            rs = rows[name]
            k = sum(1 for c, *_ in rs if c["diverged"])
            lo, hi = wilson(k, len(rs))
            firsts = sorted(c["firstDivergence"] for c, *_ in rs if c["firstDivergence"] is not None)
            med = firsts[len(firsts) // 2] if firsts else float("nan")
            pk = sum(1 for _, p, *_ in rs if p is not None)
            div = [c for c, *_ in rs if c["diverged"]]
            print(f"\n{label}: spawn trajectory differs from WAIT in {k}/{len(rs)} snapshots ({k/len(rs):.0%}, 95% CI {lo:.0%}-{hi:.0%}); PRNG state differs at some checkpoint in {pk}/{len(rs)}")
            if firsts:
                print(f"   time to first divergence (s after intervention): median {med:.0f}, min {firsts[0]:.0f}, max {firsts[-1]:.0f}; "
                      f"distribution {[round(x) for x in firsts]}")
            if div:
                print(f"   among diverged: spawns at positions not seen in control: mean {sum(c['onlyInBranch'] for c in div)/len(div):.1f} "
                      f"of {sum(c['nBranch'] for c in div)/len(div):.1f}; index-matched displacement mean {sum(c['meanDisp'] for c in div)/len(div):.1f} tiles, "
                      f"max {max(c['maxDisp'] for c in div)} tiles; displaced spawns per snapshot {sum(c['nDisplaced'] for c in div)/len(div):.1f}")
        # paired B vs C
        bdiv = [c["diverged"] for c, *_ in rows["B"]]
        cdiv = [c["diverged"] for c, *_ in rows["C"]]
        b_only = sum(1 for x, y in zip(bdiv, cdiv) if x and not y)
        c_only = sum(1 for x, y in zip(bdiv, cdiv) if y and not x)
        both = sum(1 for x, y in zip(bdiv, cdiv) if x and y)
        neither = sum(1 for x, y in zip(bdiv, cdiv) if not x and not y)
        p = binom_two_sided(min(b_only, c_only), b_only + c_only)
        print(f"\npaired B vs C: both diverge {both}, only B {b_only}, only C {c_only}, neither {neither}; exact sign test p={p:.4f}")
        # distance dependence
        pairs = sorted(((r["B"]["distToBuilding"], c["diverged"]) for (c, _, r, _) in rows["B"]), key=lambda x: x[0])
        by = defaultdict(list)
        for d, x in pairs:
            by[d].append(x)
        print("B divergence rate by distance of the segment to the nearest building:", {d: f"{sum(v)}/{len(v)}" for d, v in sorted(by.items())})
        cd = sorted(set(r["C"]["distToBuilding"] for (_, _, r, _) in rows["C"]))
        print("C segment distances to nearest building:", cd)


if __name__ == "__main__":
    main(sys.argv[1:])
