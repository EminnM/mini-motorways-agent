"""S13b: how much does the unobservable house<->destination colour assignment matter? Controller identical to the reference one, but it targets the NEAREST destination of ANY colour (what an agent without colour
information would do) and never learns from failure.   python python/synthetic_rl/s13_colourblind.py -> research/s13_multi_house/colour_blind.json"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from synthetic_rl import s13_env as E, s13_controller as CT
from synthetic_rl.s13_baselines import episode, summarize, SC
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "research", "s13_multi_house")
orig = CT._plan


def blind_plan(env, house):
    saved = env.houses[house]["group"]
    best = None
    for g in {d["group"] for d in env.dests if d["group"] is not None}:
        env.houses[house]["group"] = g
        p = orig(env, house)
        if p is not None and (best is None or len(p) < len(best)):
            best = p
    env.houses[house]["group"] = saved
    return best


def blind_act(env):
    CT._plan = blind_plan
    try:
        conn = env.connected_houses()
        # a house is "done" for the blind agent when it has an Active route into ANY destination component
        return CT.act(env)
    finally:
        CT._plan = orig


rows = [episode(blind_act, idx, 9500 + idx) for idx in range(len(SC))]
res = {"colour_blind_nearest_destination": summarize(rows), "reference_controller_same_episodes": summarize([episode(CT.act, idx, 9500 + idx) for idx in range(len(SC))])}
json.dump(res, open(os.path.join(OUT, "colour_blind.json"), "w"), indent=1)
print(json.dumps(res, indent=1)[:2500])
