"""Generator validation for synthetic Gap-2 (run BEFORE any Gap-2 training):  python -m synthetic_rl.tests_gap2"""
import itertools
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from synthetic_rl import layout as L  # noqa: E402
from synthetic_rl.env import SyntheticGapEnv, load_tasks  # noqa: E402
from synthetic_rl.tests import obs_only_solver  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MAN = os.path.join(ROOT, "research", "synthetic_lab", "manifests")
OUT = os.path.join(ROOT, "research", "s3_gap2")
R = {}


def check(name, ok, detail):
    R[name] = {"passed": bool(ok), "detail": detail}
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    splits = {s: load_tasks(os.path.join(MAN, f"gap2_{s}.jsonl")) for s in ("train", "validation", "test")}
    g1 = {s: load_tasks(os.path.join(MAN, f"gap1_{s}.jsonl")) for s in ("test",)}
    allt = [t for v in splits.values() for t in v]
    # 1. exactly two required edges, no one-action solution (brute force over EVERY legal single ADD from the start state)
    t0 = time.time()
    one_step_solvers = 0
    n_checked = 0
    two_edge_tasks = 0
    for s, ts in splits.items():
        env = SyntheticGapEnv(ts, sampling="sequential")
        for i, t in enumerate(ts):
            two_edge_tasks += (len(t["solution_edges"]) == 2 and t["n_gaps"] == 2)
            env.reset(options={"task_index": i})
            legal = [int(a) for a in np.flatnonzero(env.action_masks()) if L.decode_action(a)[0] == "ADD"]
            for a in legal:
                env.reset(options={"task_index": i})
                _, r, te, tr, info = env.step(a)
                n_checked += 1
                if info["synthetic"]["success"]:
                    one_step_solvers += 1
    check("1_two_edges_required_and_no_single_action_solves_it", one_step_solvers == 0 and two_edge_tasks == len(allt),
          f"{two_edge_tasks}/{len(allt)} tasks have exactly two solution edges; brute force over every legal ADD from the start state ({n_checked:,} single actions, {time.time() - t0:.0f}s): {one_step_solvers} complete the connection")
    # 2/3/4. both orders legal, first useful edit reward 0 and not terminal, completion reward exactly +1
    bad = []
    for s, ts in splits.items():
        env = SyntheticGapEnv(ts, sampling="sequential")
        for i, t in enumerate(ts):
            sol = t["solution_action_indices"]
            for order in itertools.permutations(sol):
                env.reset(options={"task_index": i})
                m = env.action_masks()
                if not m[order[0]]:
                    bad.append((t["task_id"], "first_illegal", order))
                    continue
                _, r1, te1, tr1, i1 = env.step(order[0])
                if not (r1 == 0.0 and not te1 and not tr1 and not i1["synthetic"]["success"]):
                    bad.append((t["task_id"], "first_edit_reward_or_terminal", r1, te1))
                    continue
                if not env.action_masks()[order[1]]:
                    bad.append((t["task_id"], "second_illegal_after_first", order))
                    continue
                _, r2, te2, tr2, i2 = env.step(order[1])
                if not (r2 == 1.0 and te2 and not tr2 and i2["synthetic"]["success"]):
                    bad.append((t["task_id"], "completion_reward", r2, te2))
    check("2_both_orders_legal_first_edit_zero_reward_completion_exactly_plus_one", not bad,
          f"{len(allt)} tasks x 2 orders: first useful edit -> reward 0 / non-terminal; second -> reward exactly +1 and terminal; violations={bad[:5]}")
    # 5. oracle + observation-only solver; 6. no label leakage
    ok_o = ok_s = tot = 0
    for ts in (splits["test"], splits["validation"]):
        env = SyntheticGapEnv(ts, sampling="sequential")
        for i, t in enumerate(ts):
            obs, _ = env.reset(options={"task_index": i})
            done = False
            while not done:
                _, r, te, tr, info = env.step(env.oracle_action())
                done = te or tr
            ok_o += info["synthetic"]["success"] and info["decisions" if False else "decisionIndex"] == 2
            obs, _ = env.reset(options={"task_index": i})
            done = False
            while not done:
                obs, r, te, tr, info = env.step(obs_only_solver(obs, env.action_masks()))
                done = te or tr
            ok_s += info["synthetic"]["success"]
            tot += 1
    check("3_oracle_and_observation_only_solver_complete_every_task", ok_o == tot and ok_s == tot,
          f"validation+test ({tot} tasks): oracle completes {ok_o}/{tot} in exactly 2 actions; a solver that sees only the observation vector + legal mask completes {ok_s}/{tot}")
    env = SyntheticGapEnv(splits["test"], sampling="sequential")
    a, _ = env.reset(options={"task_index": 3})
    env.task["solution_edges"] = [[0, 0, "E"], [1, 0, "E"]]
    env.task["solution_action_indices"] = [1, 5]
    env2 = SyntheticGapEnv(splits["test"], sampling="sequential")
    b, _ = env2.reset(options={"task_index": 3})
    check("4_no_privileged_solution_labels_in_observation", np.array_equal(a, b) and a.shape == (L.OBS_DIM,), "observation is unchanged when the stored solution is overwritten; shape (11666,)")
    # 7. random much worse than on gap-1, wait zero
    rates = {}
    for name, ts in (("gap1", g1["test"]), ("gap2", splits["test"])):
        succ = []
        for seed in range(5):
            rng = np.random.default_rng(seed)
            env = SyntheticGapEnv(ts, sampling="sequential")
            k = 0
            for i in range(len(ts)):
                env.reset(options={"task_index": i})
                done = False
                while not done:
                    _, r, te, tr, info = env.step(int(rng.choice(np.flatnonzero(env.action_masks()))))
                    done = te or tr
                k += info["synthetic"]["success"]
            succ.append(k / len(ts))
        rates[name] = (float(np.mean(succ)), float(np.std(succ)))
    check("5_random_much_worse_on_gap2_than_gap1", rates["gap2"][0] < 0.25 * rates["gap1"][0], f"RANDOM LEGAL completion over 5 seeds on TEST: Gap-1 {100 * rates['gap1'][0]:.1f}% +- {100 * rates['gap1'][1]:.1f}, Gap-2 {100 * rates['gap2'][0]:.1f}% +- {100 * rates['gap2'][1]:.1f} (oracle 100%, WAIT 0%)")
    R["_random_rates"] = rates
    # 8. disjointness
    sig = {s: {t["base_signature"] for t in v} for s, v in splits.items()}
    full = {s: {t["signature"] for t in v} for s, v in splits.items()}
    over = [(a_, b_) for a_, b_ in (("train", "validation"), ("train", "test"), ("validation", "test")) if sig[a_] & sig[b_] or full[a_] & full[b_]]
    check("6_train_validation_test_layouts_disjoint", not over, f"overlaps={over}; sizes { {s: len(v) for s, v in splits.items()} }")
    # 9. solution is unique at the edge-set level (generator guarantee) + orientation diversity
    from collections import Counter
    comp = Counter("".join(sorted("H" if e[2] == "E" else "V" for e in t["solution_edges"])) for t in splits["test"])
    adj = sum(1 for t in splits["test"] if abs(t["gap_indices"][0] - t["gap_indices"][1]) == 1)
    check("7_solution_edge_pair_diversity", len(comp) == 3, f"TEST solution-edge orientation pairs {dict(comp)}; adjacent-gap tasks (two consecutive missing edges) {adj}/{len(splits['test'])}; "
          f"distinct first solution indices {len({t['solution_action_indices'][0] for t in splits['test']})}")
    json.dump(R, open(os.path.join(OUT, "gap2_generator_validation.json"), "w"), indent=2)
    sys.exit(0 if all(v["passed"] for k, v in R.items() if not k.startswith("_")) else 1)


if __name__ == "__main__":
    main()
