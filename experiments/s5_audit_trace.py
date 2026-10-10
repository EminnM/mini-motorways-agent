"""S5 audit: per-step native traces (investigation only; no fixes, no training).
  python experiments/s5_audit_trace.py loop --seeds 182150,205907      # deterministic S1 CNN (s13): repeated-ADD anatomy
  python experiments/s5_audit_trace.py terminal                        # WAIT seeds 182150 (return 0 case) / 205907 (-10 case) + logging on/off twin
  python experiments/s5_audit_trace.py random                          # replay the S4 RANDOM stream (rng seed 0, seeds in S4 order)
Everything is read-only: the full IPC response of every STEP is captured by wrapping IpcClient.send (the wrapper only stores a copy)."""
import argparse
import hashlib
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from experiments import paths  # noqa: E402,F401
import numpy as np  # noqa: E402

OUT = os.path.join(ROOT, "research", "native_action_reward_audit")
CNN = os.path.join(ROOT, "runs", "2026-10-08_140406_syn_s1_cnn_s13", "checkpoints", "final.zip")
OPP = {"E": "W", "S": "N"}


def tiles_by_coord(raw):
    return {(t["coord"][0], t["coord"][1]): t for t in raw["tiles"]}


def edge_state(raw, x, y, d):
    tb = tiles_by_coord(raw)
    a = tb.get((x, y))
    n = (x + 1, y) if d == "E" else (x, y - 1)
    b = tb.get(n)
    return {"A": a["road"][d] if a else None, "B": b["road"][OPP[d]] if b else None,
            "A_content": a["contentType"] if a else None, "B_content": b["contentType"] if b else None}


def road_edges(raw):
    s = set()
    for t in raw["tiles"]:
        c = (t["coord"][0], t["coord"][1])
        for d in ("E", "S"):
            if t["road"][d] != "None":
                s.add((c[0], c[1], d, t["road"][d]))
    return s


def obs_channels_changed(o0, o1):
    d = np.flatnonzero(o0 != o1)
    nt = 777 * 15
    names = ["buildable", "water", "mountain", "rail", "roadN", "roadE", "roadS", "roadW", "light", "roundabout", "content", "group", "demand", "overcrowd", "vehicles"]
    tile = {}
    for i in d:
        if i < nt:
            tile[names[i % 15]] = tile.get(names[i % 15], 0) + 1
    glob = [int(i - nt) for i in d if i >= nt]
    return {"n_changed": int(d.size), "tile_channels": tile, "global_indices": glob}


class Capture:
    def __init__(self, env):
        self.last = None
        orig = env.client.send

        def send(op, **kw):
            r = orig(op, **kw)
            self.last = r
            return r
        env.client.send = send


def make_env(g):
    import mini_motorways_env as M
    env = M.MiniMotorwaysEnv(max_decisions=600, decision_delta=1.0, request_path=os.path.join(g.ipc_dir, "request.json"),
                             response_path=os.path.join(g.ipc_dir, "response.json"), start_mode="NORMAL_START")
    return env, Capture(env)


def glob_of(raw):
    gl = raw["globals"]
    return {"sim_time": raw.get("simulationTime"), "hour": gl["hour"], "day": gl["day"], "score": gl["score"]}


def step_record(i, env, cap, action, mask, raw_before, obs_before, model=None, obs_for_model=None):
    from synthetic_rl import layout as L
    rec = {"i": i, "action": int(action), "decoded": list(L.decode_action(int(action))), "legal_before": bool(mask[int(action)]), "n_legal_before": int(mask.sum())}
    dec = L.decode_action(int(action))
    if dec[0] in ("ADD", "REMOVE"):
        rec["edge_before"] = edge_state(raw_before, dec[1], dec[2], dec[3])
    rec["globals_before"] = glob_of(raw_before)
    return rec


def finish_record(rec, env, cap, raw_before, obs_before, obs_after, mask_after, reward, te, tr):
    from synthetic_rl import layout as L
    r = cap.last
    raw_after = env.last_raw_observation
    rec.update({"reward": reward, "terminated": bool(te), "truncated": bool(tr), "accepted": r.get("accepted"), "appliedAction": r.get("appliedAction"),
                "applyInfo": r.get("applyInfo"), "rewardComponents": r.get("rewardComponents"), "decisionIndex": r.get("decisionIndex"),
                "n_legal_after": int(mask_after.sum()), "globals_after": glob_of(raw_after)})
    dec = rec["decoded"]
    if dec[0] in ("ADD", "REMOVE"):
        rec["edge_after"] = edge_state(raw_after, dec[1], dec[2], dec[3])
        rec["still_legal_after"] = bool(mask_after[rec["action"]])
    e0, e1 = road_edges(raw_before), road_edges(raw_after)
    rec["road_edges_before"] = len(e0)
    rec["road_edges_after"] = len(e1)
    rec["road_edges_added"] = sorted(e1 - e0)[:6]
    rec["road_edges_removed"] = sorted(e0 - e1)[:6]
    rec["obs_changes"] = obs_channels_changed(obs_before, obs_after)
    return rec


def run_loop(a):
    from sb3_contrib import MaskablePPO
    import torch
    from experiments import gamectl
    from experiments.evaluation import UPGRADE_BASE
    os.makedirs(OUT, exist_ok=True)
    model = MaskablePPO.load(CNN, device="cpu")
    t0 = time.time()
    with gamectl.GameInstance("s5loop") as g:
        env, cap = make_env(g)
        for seed in [int(s) for s in a.seeds.split(",")]:
            obs, info = env.reset(seed=seed)
            recs = []
            done = False
            i = 0
            while not done:
                mask = env.action_masks().copy()
                raw_b = env.last_raw_observation
                with torch.no_grad():
                    t = torch.as_tensor(obs[None, :], dtype=torch.float32)
                    pol = model.policy
                    logits = pol.action_net(pol.mlp_extractor(pol.extract_features(t))[0])[0].numpy()
                ml = np.where(mask, logits, -np.inf)
                top = np.argsort(-ml)[:5]
                act, _ = model.predict(obs, action_masks=mask, deterministic=True)
                act = int(act)
                rec = step_record(i, env, cap, act, mask, raw_b, obs)
                rec["logit_chosen"] = float(logits[act])
                rec["top5"] = [[int(k), float(ml[k])] for k in top]
                obs_b = obs
                obs, r, te, tr, inf = env.step(act)
                recs.append(finish_record(rec, env, cap, raw_b, obs_b, obs, env.action_masks(), r, te, tr))
                done = te or tr
                i += 1
            json.dump(recs, open(os.path.join(OUT, f"loop_cnn_det_{seed}.json"), "w"))
            print("seed", seed, "steps", len(recs), "return", sum(x["reward"] for x in recs), f"{time.time() - t0:.0f}s", flush=True)
        env.close()


def run_terminal(a):
    import mini_motorways_env as M
    from experiments import gamectl, recording
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    with gamectl.GameInstance("s5term") as g:
        env, cap = make_env(g)
        results = {}
        for seed in (182150, 205907):
            obs, info = env.reset(seed=seed)
            recs = []
            done = False
            i = 0
            while not done:
                mask = env.action_masks().copy()
                raw_b = env.last_raw_observation
                rec = step_record(i, env, cap, 0, mask, raw_b, obs)
                obs_b = obs
                obs, r, te, tr, inf = env.step(0)
                recs.append(finish_record(rec, env, cap, raw_b, obs_b, obs, env.action_masks(), r, te, tr))
                done = te or tr
                i += 1
            results[seed] = recs
            json.dump(recs, open(os.path.join(OUT, f"terminal_wait_{seed}.json"), "w"))
            print("WAIT seed", seed, "steps", len(recs), "return", sum(x["reward"] for x in recs), f"{time.time() - t0:.0f}s", flush=True)
        # logging-on / logging-off twin on the 0-penalty seed, full terminal episode
        meta = {"initialization": "normal", "start_gap": 0, "reward_mode": "native_score", "decision_interval_seconds": 1.0, "diagnostics": True}
        twin = {}
        for mode in ("plain", "recorder_diag"):
            e2, c2 = make_env(g)
            if mode == "recorder_diag":
                e2 = recording.RecorderEnv(e2, os.path.join(OUT, "twin_recorder.jsonl"), meta)
            if os.path.exists(os.path.join(OUT, "twin_recorder.jsonl")) and mode == "plain":
                os.remove(os.path.join(OUT, "twin_recorder.jsonl"))
            o, _ = e2.reset(seed=182150)
            seq = {"obs": [], "rew": [], "term": [], "mask": []}
            d = False
            while not d:
                m = e2.action_masks()
                seq["mask"].append(hashlib.sha1(np.packbits(m).tobytes()).hexdigest())
                o, r, te, tr, inf = e2.step(0)
                seq["obs"].append(hashlib.sha1(o.tobytes()).hexdigest())
                seq["rew"].append(float(r))
                seq["term"].append(bool(te or tr))
                d = te or tr
            twin[mode] = seq
        eq = {k: twin["plain"][k] == twin["recorder_diag"][k] for k in ("obs", "rew", "term", "mask")}
        row = [json.loads(l) for l in open(os.path.join(OUT, "twin_recorder.jsonl"))]
        ref = results[182150]
        summary = {"twin_equal": eq, "steps": len(twin["plain"]["rew"]), "plain_return": sum(twin["plain"]["rew"]), "diag_return": sum(twin["recorder_diag"]["rew"]),
                   "recorder_row_return": row[0]["return"], "recorder_row_length": row[0]["length_decisions"], "recorder_row_diag_errors": row[0]["diag_errors"],
                   "recorder_row_terminal_reward_sum": row[0]["diag_terminal_reward_sum"], "same_as_first_trace": twin["plain"]["rew"] == [x["reward"] for x in ref]}
        json.dump(summary, open(os.path.join(OUT, "twin_summary.json"), "w"), indent=1)
        print("twin", summary, flush=True)
        env.close()


def run_random(a):
    from experiments import gamectl
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(0)
    t0 = time.time()
    with gamectl.GameInstance("s5rand") as g:
        env, cap = make_env(g)
        for seed in [182150, 205907, 229664, 253421]:
            obs, info = env.reset(seed=seed)
            recs = []
            done = False
            i = 0
            while not done:
                mask = env.action_masks().copy()
                raw_b = env.last_raw_observation
                act = int(rng.choice(np.flatnonzero(mask)))
                rec = step_record(i, env, cap, act, mask, raw_b, obs)
                obs_b = obs
                obs, r, te, tr, inf = env.step(act)
                recs.append(finish_record(rec, env, cap, raw_b, obs_b, obs, env.action_masks(), r, te, tr))
                done = te or tr
                i += 1
            json.dump(recs, open(os.path.join(OUT, f"random_{seed}.json"), "w"))
            print("RANDOM seed", seed, "steps", len(recs), "return", sum(x["reward"] for x in recs), f"{time.time() - t0:.0f}s", flush=True)
        env.close()


def run_hunt(a):
    """WAIT episodes back to back in ONE game process, full per-step capture, until two zero-penalty terminations are seen (max 6 episodes)."""
    from experiments import gamectl
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    zeros = 0
    with gamectl.GameInstance("s5hunt") as g:
        env, cap = make_env(g)
        for n, seed in enumerate([182150, 205907, 182150, 182150, 205907, 229664]):
            obs, info = env.reset(seed=seed)
            recs, done, i = [], False, 0
            while not done:
                if a.delay and i >= 130:
                    time.sleep(a.delay)
                mask = env.action_masks().copy()
                raw_b = env.last_raw_observation
                rec = step_record(i, env, cap, 0, mask, raw_b, obs)
                obs_b = obs
                obs, r, te, tr, inf = env.step(0)
                recs.append(finish_record(rec, env, cap, raw_b, obs_b, obs, env.action_masks(), r, te, tr))
                done = te or tr
                i += 1
            ret = sum(x["reward"] for x in recs)
            json.dump(recs, open(os.path.join(OUT, f"hunt_delay{a.delay}_ep{n}_seed{seed}.json"), "w"))
            print("episode", n, "seed", seed, "steps", len(recs), "return", ret, f"{time.time() - t0:.0f}s", flush=True)
            zeros += (ret == 0.0)
            if zeros >= 2:
                break
        env.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["loop", "terminal", "random", "hunt"])
    ap.add_argument("--seeds", default="182150,205907")
    ap.add_argument("--delay", type=float, default=0.0, help="hunt: sleep this many seconds before each step once i>=130 (lets Unity render frames between steps)")
    a = ap.parse_args()
    {"loop": run_loop, "terminal": run_terminal, "random": run_random, "hunt": run_hunt}[a.mode](a)
