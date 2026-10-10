"""S10 step 1: reconstruct per-episode timelines from the SAVED S9 data only (no native run). Writes research/s10_post_connection/s9_timelines.json"""
import gzip, json, os, statistics
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S9 = os.path.join(ROOT, "research", "s9_native_transfer")
OUT = os.path.join(ROOT, "research", "s10_post_connection")
W = ["p1det", "p1sto", "p2det", "p2sto"]
seeds = json.load(open(os.path.join(S9, "seeds.json")))["seeds"]
rows = {w: {json.loads(l)["seed"]: json.loads(l) for l in open(os.path.join(S9, "rows", w + ".jsonl"))} for w in W}
COL = ["i", "action", "kind", "x", "y", "dir", "n_legal", "reward", "scoreDelta", "accepted", "edges_added", "edges_removed", "n_houses", "conn", "rem", "pmass", "top1ill"]
def steps(w, s):
    return json.load(gzip.open(os.path.join(S9, "raw", f"{w}_{s}.json.gz"), "rt"))["steps"]
out = {}
for w in W:
    for s in seeds:
        r, st = rows[w][s], steps(w, s)
        n = len(st)
        score = 0.0
        cum, spawn, conn_hist = [], [], []
        last_prog = None
        prev_h, prev_c, prev_score = st[0][12], 0, 0.0
        for x in st:
            score += x[8]
            if x[12] > prev_h: spawn.append(x[0])
            if x[13] > prev_c or x[8] > 0: last_prog = x[0]
            prev_h, prev_c = x[12], x[13]
            cum.append(score)
        adds = [x for x in st if x[2] == "ADD"]
        repl = [x for x in adds if x[11] >= 1]           # ADD that removed >=1 edge (house-edge swap or other replacement)
        legal = [x[6] for x in st]
        collapse = next((x[0] for x in st if x[6] < 40 and all(y[6] < 40 for y in st[x[0]:x[0] + 5])), None)
        deliv_steps = [x[0] for x in st if x[8] > 0]
        fs = r["diag_final_status"] or {}
        out[f"{w}_{s}"] = {"policy": w, "seed": s, "n_steps": n, "first_connection": r["diag_first_connection_step"], "first_delivery": r["diag_first_delivery_step"], "last_delivery": deliv_steps[-1] if deliv_steps else None,
                           "deliveries": score, "house_spawn_steps": spawn, "n_houses_end": st[-1][12], "connections_end": st[-1][13], "connections_max": r["diag_connections_max"],
                           "unconnected_houses_end": st[-1][12] - st[-1][13], "last_progress_step": last_prog, "decisions_after_last_progress": (n - 1 - last_prog) if last_prog is not None else None,
                           "adds": len(adds), "replacement_adds": len(repl), "removes": sum(1 for x in st if x[2] == "REMOVE"), "legal_collapse_onset(<40 legal for 5 steps)": collapse,
                           "legal_median": statistics.median(legal), "terminal": r["terminal"], "truncated": r["truncated"], "penalty_missing": r["diag_terminal_penalty_missing"],
                           "terminal_on_entry": r["diag_terminal_on_entry"], "final_applyInfo": fs.get("applyInfo"), "final_sim_advance": fs.get("sim_time_advance"), "k0": json.load(open(os.path.join(S9, "analysis_extra.json")))["k0"][str(s)],
                           "oscillation_steps": r["road_state_stats"]["two_state_oscillation_steps"], "score_by_decile": [round(cum[min(int(n * q / 10), n - 1)], 1) for q in range(1, 11)],
                           "conn_every20": [x[13] for x in st[::20]], "houses_every20": [x[12] for x in st[::20]], "legal_every20": legal[::20]}
json.dump(out, open(os.path.join(OUT, "s9_timelines.json"), "w"), indent=1)
for key in sorted(out):
    if key.startswith("p2"):
        o = out[key]
        print(key, "steps", o["n_steps"], "conn1", o["first_connection"], "deliv1", o["first_delivery"], "lastdeliv", o["last_delivery"], "deliv", o["deliveries"], "houses_end", o["n_houses_end"], "conn_end", o["connections_end"],
              "idle_after_last_progress", o["decisions_after_last_progress"], "repl/adds", f"{o['replacement_adds']}/{o['adds']}", "collapse", o["legal_collapse_onset(<40 legal for 5 steps)"], "pen_missing", o["penalty_missing"])
