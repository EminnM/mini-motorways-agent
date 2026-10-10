"""S7 Gate A: observation / policy-behaviour equivalence on the 24 captured native states (no game): native-encoded observation vs the S7 env's
observation for the SAME scene/house/driveway; and what the frozen S1 CNNs do on each."""
import gzip, json, os, sys, collections
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
import numpy as np, torch
import mini_motorways_env as M
from synthetic_rl import layout as L, s7
from sb3_contrib import MaskablePPO
OUT = os.path.join(ROOT, "research", "s7_long_route")
W = s7.World.get()
ns = json.load(open(os.path.join(OUT, "native_states_summary.json")))
recs = [json.loads(l) for l in gzip.open(os.path.join(OUT, "native_normal_start_captures.jsonl.gz"), "rt")]
names = ["buildable","water","mountain","rail","roadN","roadE","roadS","roadW","light","roundabout","content","group","demand","overcrowd","vehicles"]
CK = {"cnn11": "2026-10-08_140406_syn_s1_cnn_s11/checkpoints/step_050000.zip", "cnn12": "2026-10-08_140406_syn_s1_cnn_s12/checkpoints/step_050000.zip", "cnn13": "2026-10-08_140406_syn_s1_cnn_s13/checkpoints/final.zip"}
models = {k: MaskablePPO.load(os.path.join(ROOT, "runs", v), device="cpu") for k, v in CK.items()}
diff_ch = collections.Counter(); diff_glob = collections.Counter(); n_diff = []
acts = collections.defaultdict(list)
def kind(a): return "WAIT" if a == 0 else L.decode_action(int(a))[0]
for n, r in zip(ns, recs):
    raw = r["first_house"]["raw"]
    # native encoder expects 'destination' dicts etc.; slim raw keeps what encode_observation needs except per-tile destination/visibleVehicleCount -> default
    nat = M.encode_observation({"tiles": raw["tiles"], "globals": raw["globals"]})
    s = W.scenes[n["scene_id"]]; h = tuple(n["house"]); dr = tuple(n["driveway"])
    dd = [d for d in s7.DIRS4 if s7.canon(h, L.neighbor(h[0], h[1], d)) == dr][0]
    env = s7.LongRouteEnv([{"scene_id": n["scene_id"], "house": n["house"], "driveway_dir": dd, "k_min": 0, "task_id": "x"}], sampling="sequential")
    syn, _ = env.reset()
    d = np.flatnonzero(nat != syn); n_diff.append(len(d))
    nt = L.N_TILES * 15
    for i in d:
        if i < nt: diff_ch[names[i % 15]] += 1
        else: diff_glob[int(i - nt)] += 1
    nm = np.zeros(3111, bool); nm[r["first_house"]["legal"]] = True
    for mk, m in models.items():
        a_nat = int(m.predict(nat, action_masks=nm, deterministic=True)[0])
        a_syn = int(m.predict(syn, action_masks=env.action_masks(), deterministic=True)[0])
        acts[mk].append((kind(a_nat), kind(a_syn), a_nat == a_syn))
res = {"obs_entries_differing_per_state": {"min": min(n_diff), "median": float(np.median(n_diff)), "max": max(n_diff)},
       "tile_channels_that_differ_total_over_24_states": dict(diff_ch), "global_indices_that_differ_total": dict(diff_glob),
       "policy_first_action_kind_native_vs_s7": {mk: {"native": dict(collections.Counter(x[0] for x in v)), "s7": dict(collections.Counter(x[1] for x in v)), "same_action": sum(x[2] for x in v)} for mk, v in acts.items()}}
json.dump(res, open(os.path.join(OUT, "obs_equivalence.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
