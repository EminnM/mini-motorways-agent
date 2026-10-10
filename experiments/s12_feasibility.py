"""S12 feasibility gate analysis on the control capture (edge-level logs) + the S11 trajectories. Writes research/s12_concrete_conservation/feasibility.json"""
import collections
import glob
import gzip
import json
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import s11_guard as G
sys.path.insert(0, os.path.join(ROOT, 'python'))
from synthetic_rl import layout as L

O = os.path.join(ROOT, "research", "s12_concrete_conservation")
T = lambda l: tuple(l)
CAP = {}


def uf_components(edges):
    parent = {}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for e in edges:
        a, b = G.ends(e)
        parent.setdefault(a, a)
        parent.setdefault(b, b)
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    return {t: find(t) for t in parent}


def same_partition(c1, c2, tiles):
    # equal reachability partition restricted to `tiles`
    groups1, groups2 = collections.defaultdict(set), collections.defaultdict(set)
    for t in tiles:
        if t in c1: groups1[c1[t]].add(t)
        if t in c2: groups2[c2[t]].add(t)
    return {frozenset(v) for v in groups1.values()} == {frozenset(v) for v in groups2.values()}


def bfs_len(edges, src, targets):
    adj = collections.defaultdict(list)
    for e in edges:
        a, b = G.ends(e)
        adj[a].append(b)
        adj[b].append(a)
    if src not in adj:
        return None
    seen = {src: 0}
    q = collections.deque([src])
    while q:
        u = q.popleft()
        if u in targets:
            return seen[u]
        for v in adj[u]:
            if v not in seen:
                seen[v] = seen[u] + 1
                q.append(v)
    return None


eps = {}
for f in sorted(glob.glob(os.path.join(O, "raw", "control_*.json.gz"))):
    d = json.load(gzip.open(f, "rt"))
    eps[d["seed"]] = d
out = {"episodes": len(eps)}

# 1. concrete accounting on this HELD-OUT capture (excluding top-up steps)
acc = collections.Counter()
mism = []
for s, d in eps.items():
    for x in d["steps"]:
        if x["topup"] != 0:
            continue
        dC = x["concrete"] - x["concrete_prev"]
        if x["kind"] == "ADD":
            c = x["exec_cls"]
            plain = c in ("cycle_closing", "extension", "joins_components", "new_segment")
            pred = -1 if plain else 0
            if c is not None and c.startswith("unknown"):
                acc[("ADD", "unknown_class", dC == pred)] += 1 if False else 0
                acc[("ADD_unknown_class", dC)] += 1
                continue
            acc[("ADD_plain" if plain else "ADD_other", pred, dC)] += 1
            if dC != pred and len(mism) < 6:
                mism.append((s, x["i"], c, dC, x["fa"], x["fr"]))
        elif x["kind"] == "REMOVE":
            acc[("REMOVE", dC)] += 1
        elif x["kind"] == "WAIT":
            acc[("WAIT", dC)] += 1
out["concrete_accounting"] = {str(k): v for k, v in acc.items()}
n_ok = sum(v for k, v in acc.items() if k[0] in ("ADD_plain", "ADD_other") and k[1] == k[2])
n_all = sum(v for k, v in acc.items() if k[0] in ("ADD_plain", "ADD_other"))
out["concrete_accounting_match_ADD"] = [n_ok, n_all]
out["concrete_accounting_mismatches"] = mism

# 2. non-functional (Mothballed / Pending) predictions + replacement semantics
mo = collections.Counter()
for s, d in eps.items():
    for x in d["steps"]:
        if x["kind"] != "ADD":
            continue
        c = x["exec_cls"]
        if c in ("cycle_closing", "extension", "joins_components", "new_segment") and not x["new_struct"]:
            mo[("plain_extension", "no_change_in_nonfunctional_edges", not x["nfa"] and not x["nfr"])] += 1
            mo[("plain_extension", "exactly_one_functional_edge_added_none_removed", len(x["fa"]) == 1 and not x["fr"])] += 1
        if c == "house_house_edge_replacement" and not x["new_struct"]:
            mo[("house_replacement", "functional_edge_count_unchanged", len(x["fa"]) == len(x["fr"]) == 1)] += 1
out["mothballed_and_replacement_checks"] = {str(k): v for k, v in mo.items()}

# 3. legal-ADD availability vs concrete
av = collections.Counter()
for s, d in eps.items():
    for x in d["steps"]:
        k = "zero" if x["concrete"] == 0 else "positive"
        if x["n_legal_add_next"] == 0:
            av[(k, "no_legal_add(upgrade pending or terminal)")] += 1
        elif x["concrete"] == 0:
            av[(k, "legal_add<=30" if x["n_legal_add_next"] <= 30 else "legal_add>30")] += 1
        else:
            av[(k, "legal_add>=60" if x["n_legal_add_next"] >= 60 else "legal_add<60")] += 1
out["legal_add_availability"] = {str(k): v for k, v in av.items()}

# 4. class frequencies among executed ADDs; concrete consumed; candidate-veto exposure
ex = collections.Counter(x["exec_cls"] for d in eps.values() for x in d["steps"] if x["kind"] == "ADD")
n_add = sum(ex.values())
plain = sum(ex[c] for c in ("cycle_closing", "extension", "joins_components", "new_segment"))
out["executed_ADD_classes"] = dict(ex)
out["executed_ADDs"] = n_add
out["plain_extension_ADDs(concrete consuming)"] = plain
out["cycle_closing_share_of_ADDs"] = round(ex["cycle_closing"] / n_add, 4)
out["cycle_closing_share_of_plain_extension_concrete"] = round(ex["cycle_closing"] / max(plain, 1), 4)
steps_total = sum(len(d["steps"]) for d in eps.values())
out["steps_total"] = steps_total
out["steps_with_a_vetoable_legal_ADD"] = sum(1 for d in eps.values() for x in d["steps"] if x["n_vetoable"] > 0)
out["mean_policy_mass_on_cycle_closing_when_vetoable_exists"] = round(statistics.fmean([x["vetoable_mass"] for d in eps.values() for x in d["steps"] if x["n_vetoable"] > 0] or [0]), 4)
leg_tot = collections.Counter()
for d in eps.values():
    for x in d["steps"]:
        for k, v in x["cls_n"].items():
            leg_tot[k] += v
tot_leg = sum(leg_tot.values())
out["legal_ADD_candidate_class_share"] = {k: round(v / tot_leg, 4) for k, v in sorted(leg_tot.items(), key=lambda kv: -kv[1])}
out["unknown_share_of_legal_ADD_candidates"] = round(sum(v for k, v in leg_tot.items() if k.startswith("unknown")) / tot_leg, 4)
out["unknown_share_of_executed_ADDs"] = round(sum(v for k, v in ex.items() if k and k.startswith("unknown")) / n_add, 4)
cc_conc = collections.defaultdict(int)
for s, d in eps.items():
    for x in d["steps"]:
        if x["exec_cls"] == "cycle_closing" and x["topup"] == 0:
            cc_conc[s] += x["concrete_prev"] - x["concrete"]
out["concrete_consumed_by_cycle_closing_per_episode"] = dict(cc_conc)
out["concrete_consumed_by_all_plain_extensions"] = sum(x["concrete_prev"] - x["concrete"] for d in eps.values() for x in d["steps"] if x["exec_cls"] in ("cycle_closing", "extension", "joins_components", "new_segment") and x["topup"] == 0)

# 5. redundancy replay + 6. route shortcut effect
diff_steps = 0
checked = 0
shortcut = []
ce_total = 0
for s, d in eps.items():
    F = {T(e) for e in d["header"]["F0"]}
    CE = set()
    houses, stubs_tiles = set(), set()
    for e in F:
        stubs_tiles |= set(G.ends(e))
    prev_conn = set()
    for x in d["steps"]:
        added_before = set(F)
        F -= {T(e) for e in x["fr"]}
        F |= {T(e) for e in x["fa"]}
        CE -= {T(e) for e in x["fr"]}
        if x["exec_cls"] == "cycle_closing":
            k = L.decode_action(x["action"])
            e_act = G.canon((k[1], k[2]), G.ends((k[1], k[2], k[3]))[1])
            if e_act in {T(e) for e in x["fa"]}:           # only the executed edge itself (a house spawned in the same step also adds a free driveway edge)
                CE.add(e_act)
                ce_total += 1
        for ns in x["new_struct"]:
            if ns[2] == "House":
                houses.add((ns[0], ns[1]))
        for e in x["fa"]:                                   # stubs of destinations spawned later appear together with a Carpark tile
            pass
        for ns in x["new_struct"]:
            if ns[2] in ("Carpark", "Destination"):
                stubs_tiles.add((ns[0], ns[1]))
        if CE:
            c_with, c_without = uf_components(F), uf_components(F - CE)
            tiles = set(c_with) | set(c_without)
            checked += 1
            if not same_partition(c_with, c_without, tiles):
                diff_steps += 1
        conn = {T(h) for h in x["connected_houses"]} if x["connected_houses"] is not None else prev_conn
        for h in conn - prev_conn:
            if CE:
                l1 = bfs_len(F, h, stubs_tiles)
                l2 = bfs_len(F - CE, h, stubs_tiles)
                if l1 is not None and l2 is not None:
                    shortcut.append(l2 - l1)
        prev_conn = conn
out["redundancy_replay"] = {"steps_with_cycle_edges_checked": checked, "steps_where_reachability_partition_differs_without_them": diff_steps, "cycle_closing_edges_added": ce_total}
out["route_length_gain_from_cycle_edges_at_house_connection_events"] = {"events": len(shortcut), "events_with_gain>0": sum(1 for v in shortcut if v > 0), "median_gain": statistics.median(shortcut) if shortcut else None, "max_gain": max(shortcut) if shortcut else None}
json.dump(out, open(os.path.join(O, "feasibility.json"), "w"), indent=1, default=str)
for k, v in out.items():
    print(k, ":", v)
