"""S7 Gate A: teacher-forced replay of the NATIVE deterministic-CNN trajectories (S5 traces) inside S7: at every step the S7 state is rebuilt from the same actions;
we compare the frozen S1 seed-13 CNN's argmax in S7 with the action it actually took natively.  Measures behavioural (not just observational) equivalence."""
import json, os, sys, collections
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
import numpy as np
from synthetic_rl import s7, layout as L
from sb3_contrib import MaskablePPO
OUT = os.path.join(ROOT, "research", "s7_long_route")
m = MaskablePPO.load(os.path.join(ROOT, "runs", "2026-10-08_140406_syn_s1_cnn_s13", "checkpoints", "final.zip"), device="cpu")
ns = {n["seed"]: n for n in json.load(open(os.path.join(OUT, "native_states_summary.json")))}
W = s7.World.get()
res = {}
for seed in (182150, 205907):
    n = ns[seed]; h = tuple(n["house"]); dr = tuple(n["driveway"])
    dd = [d for d in s7.DIRS4 if s7.canon(h, L.neighbor(h[0], h[1], d)) == dr][0]
    env = s7.LongRouteEnv([{"scene_id": n["scene_id"], "house": n["house"], "driveway_dir": dd, "k_min": 0, "task_id": "x"}], sampling="sequential", horizon_fn=lambda k: 10 ** 6)
    tr = json.load(open(os.path.join(ROOT, "research", "native_action_reward_audit", f"loop_cnn_det_{seed}.json")))
    obs, _ = env.reset()
    agree = tot = illegal = upg = 0
    first = None
    kinds_syn = collections.Counter()
    for i, x in enumerate(tr):
        a = x["action"]
        if a >= L.UPGRADE_BASE:
            upg += 1; obs, *_ = env.step(0); continue
        mask = env.action_masks()
        a_syn = int(m.predict(obs, action_masks=mask, deterministic=True)[0])
        tot += 1
        agree += (a_syn == a)
        if a_syn != a and first is None:
            first = {"step": i, "native": L.decode_action(a), "s7": L.decode_action(a_syn), "native_action_legal_in_s7": bool(mask[a])}
        illegal += (not mask[a])
        obs, *_ = env.step(a if mask[a] else 0)
    res[seed] = {"steps": tot, "argmax_agreement": agree, "native_action_illegal_in_s7": illegal, "upgrade_steps_skipped": upg, "first_disagreement": first}
json.dump(res, open(os.path.join(OUT, "replay_check.json"), "w"), indent=1, default=str)
print(json.dumps(res, indent=1, default=str))
