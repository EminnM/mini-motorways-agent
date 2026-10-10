"""Deterministic validation suite for the Gap-1 completion-reward task (run BEFORE any training).
  python -m experiments.gap1.diagnostic_tests                 # checks 1-6, 8-10 (live game, ~10 tasks each)
  python -m experiments.gap1.diagnostic_tests --legality-sweep validation,test --shard 0/1   # check 7 (+ oracle reward) over whole splits
Task config under test: reward = completion (1.0 on the connecting transition, else 0), success terminates,
destructive failure terminates, timeout (20 decisions) truncates, no native score / terminal penalty."""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

from .. import paths
from .. import gamectl
from . import gap1_env as G
from . import tasks as T

OUT = os.path.join(paths.ROOT, "research", "gap1")
W = 37


def idx_remove(x, y, d):
    return 1 + ((y - T.YMIN) * W + (x - T.XMIN)) * 4 + (0 if d == "E" else 1) * 2 + 1


def build(ipc_dir, tasks):
    import mini_motorways_env as M
    base = M.MiniMotorwaysEnv(max_decisions=100, decision_delta=1.0,
                              request_path=os.path.join(ipc_dir, "request.json"), response_path=os.path.join(ipc_dir, "response.json"),
                              start_mode="CONNECTED_START", start_gap=0)
    env = G.Gap1Env(base, tasks, construction_window=20, continuation=0, reward_variant="completion", sampling="sequential",
                    continue_after_success=False, success_terminates=True)
    return base, env


def obs_hash(o):
    return hashlib.sha256(np.ascontiguousarray(o, dtype=np.float32).tobytes()).hexdigest()[:16]


class Rec:
    def __init__(self):
        self.res = {}

    def add(self, name, passed, detail):
        self.res[name] = {"passed": bool(passed), "detail": detail}
        print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}", flush=True)


def test9(base, env, tasks, rec, sample="first20"):
    """Layouts / correct actions vary. `sample`: 'first20' (the first 20 manifest entries; run 1) or 'random20' (20
    validation tasks drawn at random with a fixed seed; the intended design -- the first entries of a manifest share base
    layouts, so they under-represent diversity)."""
    idxs = list(range(20)) if sample == "first20" else sorted(np.random.default_rng(9).choice(len(tasks), 20, replace=False).tolist())
    houses, astars, orient, bases = set(), set(), set(), set()
    for i in idxs:
        env.reset(options={"task_index": i})
        houses.add(tuple(tasks[i]["house"]))
        astars.add(tasks[i]["a_star"]["action_index"])
        orient.add(tasks[i]["gap_orientation"])
        bases.add(tasks[i]["base_layout_signature"])
        init = base.last_init_info
        if tuple(init["house"]) != tuple(tasks[i]["house"]):
            houses.add(("MISMATCH", i))
    allsplits = {s_: G.load_manifest(os.path.join(paths.ROOT, "research", "gap1", "manifests", f"{s_}.jsonl")) for s_ in ("train", "validation", "test")}
    dist = {s_: len({t["a_star"]["action_index"] for t in v}) for s_, v in allsplits.items()}
    layouts = {s_: len({t["base_layout_signature"] for t in v}) for s_, v in allsplits.items()}
    top = {s_: round(max(np.bincount([t["a_star"]["action_index"] for t in v]).tolist()) / len(v), 3) for s_, v in allsplits.items()}
    ok = len(houses) >= 10 and len(astars) >= 10 and len(orient) == 2 and all(d >= 50 for d in dist.values()) and all(t < 0.10 for t in top.values())
    rec.add(f"9_layouts_and_correct_action_vary[{sample}]", ok,
            f"{sample}: 20 live resets -> {len(houses)} distinct houses, {len(bases)} base layouts, {len(astars)} distinct a*, orientations {sorted(orient)}; "
            f"whole splits: distinct a* {dist}, base layouts {layouts}, top single-index share {top}")


def run_suite(ipc_dir, rec: Rec):
    tasks = G.load_manifest(os.path.join(paths.ROOT, "research", "gap1", "manifests", "validation.jsonl"))
    base, env = build(ipc_dir, tasks)
    n = 10
    # 1. correct action -> reward 1 and immediate termination
    bad = []
    for i in range(n):
        env.reset(options={"task_index": i})
        a = tasks[i]["a_star"]["action_index"]
        _, r, te, tr, info = env.step(a)
        if not (r == 1.0 and te and not tr and info["gap1"]["success"] and info["gap1"]["steps_to_success"] == 1):
            bad.append((i, r, te, tr))
    rec.add("1_correct_action_reward1_terminates", not bad, f"{n} tasks; failures={bad}")
    # 2. incorrect legal road action -> 0
    bad = []
    rng = np.random.default_rng(0)
    for i in range(n):
        env.reset(options={"task_index": i})
        mask = env.action_masks()
        a_star = tasks[i]["a_star"]["action_index"]
        cands = [int(a) for a in np.flatnonzero(mask) if 1 <= a < 1 + W * 21 * 4 and (a - 1) % 2 == 0 and a != a_star]   # ADD kinds
        a = int(rng.choice(cands))
        _, r, te, tr, info = env.step(a)
        if info["gap1"]["success"]:
            continue                                 # (an incidental connection would legitimately reward; never observed)
        if not (r == 0.0 and not te and not tr):
            bad.append((i, a, r, te, tr))
    rec.add("2_incorrect_legal_road_gives_0", not bad, f"{n} tasks; failures={bad}")
    # 3. WAIT -> 0
    bad = []
    for i in range(n):
        env.reset(options={"task_index": i})
        _, r, te, tr, info = env.step(0)
        if not (r == 0.0 and not te and not tr):
            bad.append((i, r, te, tr))
    rec.add("3_wait_gives_0", not bad, f"{n} tasks; failures={bad}")
    # 4. destructive action -> 0 and terminates as failure
    tested, bad = 0, []
    for i in range(40):
        env.reset(options={"task_index": i})
        mask = env.action_masks()
        full = T.parse_edits(tasks[i]["full_path_edits"])
        a_s = (tasks[i]["a_star"]["x"], tasks[i]["a_star"]["y"], tasks[i]["a_star"]["dir"])
        legal_removes = [idx_remove(x, y, d) for (x, y, d) in full if (x, y, d) != a_s and d in ("E", "S") and mask[idx_remove(x, y, d)]]
        if not legal_removes:
            continue
        _, r, te, tr, info = env.step(legal_removes[0])
        tested += 1
        if not (r == 0.0 and te and not tr and info["gap1"]["destructive_failure"] and not info["gap1"]["success"]):
            bad.append((i, r, te, tr, info["gap1"]["destructive_failure"]))
        if tested >= 8:
            break
    rec.add("4_destructive_action_zero_and_terminates", tested >= 5 and not bad, f"{tested} destructive removals tested; failures={bad}")
    # 5. timeout -> 0 and truncation after exactly 20 decisions
    bad = []
    for i in range(5):
        env.reset(options={"task_index": i})
        rs, last = [], None
        for k in range(20):
            _, r, te, tr, last = env.step(0)
            rs.append(r)
            if te or tr:
                break
        g = last["gap1"]
        if not (len(rs) == 20 and all(x == 0.0 for x in rs) and tr and not te and g["timeout"] and not g["success"]):
            bad.append((i, len(rs), tr, te, g["timeout"]))
    rec.add("5_timeout_zero_and_truncates_at_20", not bad, f"5 tasks; failures={bad}")
    # 6. no native score / terminal penalty leaks (injected native values must not reach the reward)
    leaks = []
    env.reset(options={"task_index": 0})
    orig = base.step

    def injected(action, score_delta=4.0, penalty=-10.0, terminal=False):
        o, r, te, tr, info = orig(action)
        info = dict(info)
        info["rewardComponents"] = {"scoreDelta": score_delta, "terminalPenalty": penalty if terminal else 0.0}
        return o, (score_delta + (penalty if terminal else 0.0)), te or terminal, tr, info
    base.step = lambda a: injected(a, 4.0, -10.0, False)
    _, r, te, tr, info = env.step(0)
    if r != 0.0:
        leaks.append(("native_score_delta_leaked", r))
    env.reset(options={"task_index": 1})
    base.step = lambda a: injected(a, 0.0, -10.0, True)
    _, r, te, tr, info = env.step(0)
    if not (r == 0.0 and te and info["gap1"]["native_terminal"]):
        leaks.append(("terminal_penalty_leaked_or_not_terminal", r, te))
    base.step = orig
    # live: rewards over many random-policy episodes must be exactly {0.0, 1.0}
    seen = set()
    rng = np.random.default_rng(1)
    for i in range(30):
        env.reset(options={"task_index": 20 + i})
        done = False
        while not done:
            legal = np.flatnonzero(env.action_masks())
            _, r, te, tr, info = env.step(int(rng.choice(legal)))
            seen.add(float(r))
            done = te or tr
    if not seen <= {0.0, 1.0}:
        leaks.append(("unexpected_reward_values", sorted(seen)))
    rec.add("6_no_native_reward_or_terminal_penalty", not leaks, f"leaks={leaks}; live reward values over 30 random episodes={sorted(seen)}")
    # 8. observation carries no correct-action metadata
    obs, info = env.reset(options={"task_index": 3})
    from mini_motorways_env import encode_observation
    same = np.array_equal(obs, encode_observation(base.last_raw_observation))
    shape_ok = obs.shape == base.observation_space.shape == (11666,) and env.observation_space is base.observation_space
    rec.add("8_observation_is_the_unchanged_policy_observation", same and shape_ok,
            f"obs shape {obs.shape}; equals encode_observation(native observation): {same}; task fields only in info: {sorted(k for k in info if 'task' in k)}")
    test9(base, env, tasks, rec, "first20")
    # 10. determinism: same task + same action sequence -> identical results
    bad = []
    for i in (0, 5, 9):
        runs = []
        for rep in range(2):
            obs, _ = env.reset(options={"task_index": i})
            trace = [obs_hash(obs)]
            rng = np.random.default_rng(100 + i)
            for step in range(8):
                legal = np.flatnonzero(env.action_masks())
                a = int(rng.choice(legal)) if step != 4 else tasks[i]["a_star"]["action_index"]
                obs, r, te, tr, info = env.step(a)
                trace.append((a, r, te, tr, obs_hash(obs)))
                if te or tr:
                    break
            runs.append(trace)
        if runs[0] != runs[1]:
            bad.append(i)
    rec.add("10_identical_task_and_actions_reproduce_identical_results", not bad, f"3 tasks x 2 replays; mismatching tasks={bad}")
    try:
        env.close()
    except Exception:
        pass


def legality_sweep(ipc_dir, splits, shard, rec: Rec):
    """Check 7 over entire splits: a* legal in every initial state, house not connected at reset, and the real
    completion reward fires (+1, terminate) when a* is played."""
    si, sn = (int(x) for x in shard.split("/"))
    bad, n = [], 0
    for split in splits:
        tasks = G.load_manifest(os.path.join(paths.ROOT, "research", "gap1", "manifests", f"{split}.jsonl"))
        tasks = [t for j, t in enumerate(tasks) if j % sn == si]
        base, env = build(ipc_dir, tasks)
        for i, t in enumerate(tasks):
            try:
                env.reset(options={"task_index": i})
                mask = env.action_masks()
                a = t["a_star"]["action_index"]
                connected0 = T.house_connected(base.last_raw_observation, tuple(t["house"]), [tuple(x) for x in t["destination_footprint"]])
                _, r, te, tr, info = env.step(a)
                n += 1
                if not (mask[a] and not connected0 and r == 1.0 and te and info["gap1"]["success"]):
                    bad.append((t["task_id"], bool(mask[a]), connected0, r, te))
            except Exception as e:
                bad.append((t["task_id"], repr(e)))
                break
        try:
            env.close()
        except Exception:
            pass
    rec.add(f"7_correct_action_legal_and_rewarded_in_every_initial_state[{','.join(splits)} shard {shard}]", not bad and n > 0, f"{n} configurations checked; failures={bad[:5]}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--legality-sweep", default=None, help="comma-separated splits for check 7")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--tag", default="g1diag")
    ap.add_argument("--only9", default=None, choices=["first20", "random20"], help="re-run check 9 only, with that sampling")
    ap.add_argument("--out", default=os.path.join(OUT, "completion_diagnostic_tests.json"))
    a = ap.parse_args(argv)
    rec = Rec()
    with gamectl.GameInstance(a.tag) as g:
        if a.only9:
            tasks = G.load_manifest(os.path.join(paths.ROOT, "research", "gap1", "manifests", "validation.jsonl"))
            base, env = build(g.ipc_dir, tasks)
            test9(base, env, tasks, rec, a.only9)
        elif a.legality_sweep:
            legality_sweep(g.ipc_dir, a.legality_sweep.split(","), a.shard, rec)
        else:
            run_suite(g.ipc_dir, rec)
    out = os.path.join(OUT, f"completion_diagnostic_check9_{a.only9}.json") if a.only9 else a.out if not a.legality_sweep else os.path.join(OUT, f"completion_diagnostic_legality_{a.legality_sweep.replace(',', '_')}_{a.shard.replace('/', 'of')}.json")
    json.dump(rec.res, open(out, "w"), indent=2)
    return 0 if all(v["passed"] for v in rec.res.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
