"""S6 offline validation (NO game): terminal-penalty flag + anti-cycle filter, using the saved S5 traces and the 120 saved native states.
   python experiments/s6_offline_tests.py"""
import glob
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from experiments import paths  # noqa: E402,F401
from experiments.anticycle import AntiCycleFilter, ranking_from_logits, revisit_stats  # noqa: E402

AUD = os.path.join(ROOT, "research", "native_action_reward_audit")
OUT = os.path.join(ROOT, "research", "native_cnn_anticycle")
RES = []


def test(fn):
    try:
        info = fn()
        RES.append((fn.__name__, True, info or ""))
    except AssertionError as e:
        RES.append((fn.__name__, False, f"ASSERTION: {e}"))
    except Exception as e:
        import traceback
        RES.append((fn.__name__, False, f"{type(e).__name__}: {e} {traceback.format_exc()[-300:]}"))
    print(("PASS " if RES[-1][1] else "FAIL ") + RES[-1][0] + ": " + RES[-1][2][:420], flush=True)
    return fn


def load(p):
    return json.load(open(p))


def trace_signatures(rows):
    """State signature at each decision (state BEFORE action i) reconstructed from the S5 trace's added/removed edge lists, relative to the (unknown, constant) base."""
    diff, sigs = {}, []
    for r in rows:
        sigs.append(hashlib.sha1(repr(sorted(diff.items())).encode()).hexdigest()[:16])
        assert len(r["road_edges_added"]) < 6 and len(r["road_edges_removed"]) < 6, "edge list truncated; signature not exact"
        for e in r["road_edges_added"]:
            k = tuple(e[:3])
            diff[k] = diff.get(k, 0) + 1
            if diff[k] == 0:
                del diff[k]
        for e in r["road_edges_removed"]:
            k = tuple(e[:3])
            diff[k] = diff.get(k, 0) - 1
            if diff[k] == 0:
                del diff[k]
    sigs.append(hashlib.sha1(repr(sorted(diff.items())).encode()).hexdigest()[:16])
    return sigs


LOOPS = {s: load(os.path.join(AUD, f"loop_cnn_det_{s}.json")) for s in (182150, 205907)}


def replay_filter(rows, enabled=True):
    """Drive the filter along the recorded (unfiltered) trajectory. Valid only up to and including the first intervention."""
    sigs = trace_signatures(rows)
    f = AntiCycleFilter(enabled)
    out = []
    for t, r in enumerate(rows):
        ranking = [a for a, _ in r["top5"]]
        a, iv = f.choose(sigs[t], ranking, t)
        out.append((t, ranking, a, iv))
        if iv is not None:
            break
        f.observe(sigs[t], r["action"], sigs[t + 1])
    return f, out, sigs


@test
def test_s5_oscillation_is_reproduced_from_saved_traces_in_road_state_space():
    d = []
    for s, rows in LOOPS.items():
        st = revisit_stats(trace_signatures(rows))
        assert st["two_state_cycles"] >= 1 and st["two_state_oscillation_steps"] >= 50, st
        d.append(f"seed {s}: {st}")
    return "; ".join(d)


@test
def test_filter_detects_first_cycle_and_selects_next_ranked_legal_alternative():
    d = []
    for s, rows in LOOPS.items():
        f, out, sigs = replay_filter(rows)
        t0, ranking, a, iv = out[-1]
        assert iv is not None, "no intervention found"
        assert sigs[t0] == sigs[t0 - 2] and sigs[t0 - 1] == sigs[t0 - 3] and sigs[t0] != sigs[t0 - 1], "A,B,A,B criterion not met at the intervention step"
        assert iv["original_action"] == rows[t0]["action"] == ranking[0]
        assert iv["excluded"] == [ranking[0]] and iv["outcome"] == "replaced"
        assert iv["chosen_action"] == ranking[1] and iv["chosen_rank"] == 1, "did not take the next-highest-ranked legal action"
        d.append(f"seed {s}: first detection at decision {t0}; excluded {iv['excluded']}, chose rank-{iv['chosen_rank']} action {iv['chosen_action']}")
    return "; ".join(d)


@test
def test_non_cycling_decisions_are_unchanged():
    n = 0
    for s, rows in LOOPS.items():
        f, out, sigs = replay_filter(rows)
        for t, ranking, a, iv in out[:-1]:
            assert iv is None and a == ranking[0] == rows[t]["action"], f"changed a non-cycling decision at {t}"
            n += 1
    for fpath in glob.glob(os.path.join(AUD, "hunt_*.json")):       # WAIT episodes: constant road state, never a cycle
        rows = load(fpath)
        sigs = trace_signatures(rows)
        f = AntiCycleFilter(True)
        for t, r in enumerate(rows):
            a, iv = f.choose(sigs[t], [0, 1, 2], t)
            assert iv is None and a == 0
            f.observe(sigs[t], 0, sigs[t + 1])
            n += 1
    # valid repeated construction: the same action index chosen at every step, but every step creates a NEW road state
    f = AntiCycleFilter(True)
    for t in range(30):
        a, iv = f.choose(f"new{t}", [7, 8, 9], t)
        assert iv is None and a == 7
        f.observe(f"new{t}", 7, f"new{t + 1}")
        n += 1
    return f"{n} decisions (pre-cycle prefix of both S5 traces, 8 WAIT episodes, 30 repeated-construction steps): zero interventions, action == policy argmax"


@test
def test_disabled_filter_reproduces_original_actions_exactly():
    n = 0
    for s, rows in LOOPS.items():
        sigs = trace_signatures(rows)
        f = AntiCycleFilter(False)
        for t, r in enumerate(rows):
            ranking = [a for a, _ in r["top5"]]
            a, iv = f.choose(sigs[t], ranking, t)
            assert iv is None and a == r["action"], f"seed {s} step {t}"
            f.observe(sigs[t], r["action"], sigs[t + 1])
            n += 1
        assert not f.interventions
    return f"{n}/{n} recorded actions reproduced with the filter disabled, through the full oscillation"


@test
def test_selector_ranking_equals_model_predict_on_120_saved_native_states():
    import gzip
    import numpy as np
    import torch
    from sb3_contrib import MaskablePPO
    import mini_motorways_env as M
    model = MaskablePPO.load(os.path.join(ROOT, "runs", "2026-10-08_140406_syn_s1_cnn_s13", "checkpoints", "final.zip"), device="cpu")
    ok = n = 0
    for line in gzip.open(os.path.join(ROOT, "research", "synthetic_to_native", "native_states_validation_0of1.jsonl.gz"), "rt"):
        r = json.loads(line)
        obs = M.encode_observation(r["raw"])
        mask = np.zeros(3111, bool)
        mask[r["legal_indices"]] = True
        with torch.no_grad():
            pol = model.policy
            logits = pol.action_net(pol.mlp_extractor(pol.extract_features(torch.as_tensor(obs[None], dtype=torch.float32)))[0])[0].numpy()
        rank = ranking_from_logits(logits, mask)
        pred = int(model.predict(obs, action_masks=mask, deterministic=True)[0])
        assert all(mask[a] for a in rank) and len(rank) == int(mask.sum())
        ok += rank[0] == pred
        n += 1
    assert ok == n, f"{ok}/{n}"
    return f"ranking_from_logits(...)[0] == model.predict(deterministic) on {ok}/{n} saved native states; ranking covers exactly the legal set"


@test
def test_filter_unit_cases():
    # A,B,A is not a cycle; A,A,A never is
    f = AntiCycleFilter(True)
    for s in ["A", "B", "A"]:
        assert not f.detected(s) or False
        f.hist.append(s)
    f = AntiCycleFilter(True)
    f.hist = ["A", "A", "A"]
    assert not f.detected("A")
    f.hist = ["A", "B", "A"]
    assert f.detected("B"), "A,B,A,B must be detected at the 4th state"
    f.hist = ["A", "B"]
    assert not f.detected("A"), "A,B,A is only one revisit"
    # rejection of a replacement that is known to return into the cycle
    f = AntiCycleFilter(True)
    f.hist = ["A", "B", "A"]
    f.trans = {("B", 10): "A", ("B", 11): "A", ("B", 12): "C"}
    a, iv = f.choose("B", [10, 11, 12, 13], 5)
    assert a == 12 and iv["excluded"] == [10, 11] and iv["outcome"] == "replaced" and iv["chosen_rank"] == 2, iv
    # known transition into a recognised cycle state is rejected with a reason
    f = AntiCycleFilter(True)
    f.hist = ["A", "B", "A"]
    f.cycle_states = {"X"}
    f.trans = {("B", 10): "A", ("B", 11): "X"}
    a, iv = f.choose("B", [10, 11, 12], 5)
    assert a == 12 and (11, "known_transition_into_cycle_state") in iv["rejected"], iv
    # no alternative -> original kept, recorded
    f = AntiCycleFilter(True)
    f.hist = ["A", "B", "A"]
    f.trans = {("B", 10): "A"}
    a, iv = f.choose("B", [10], 5)
    assert a == 10 and iv["outcome"] == "no_alternative"
    # detected but top action is not a known cycle-continuing action -> untouched, counted
    f = AntiCycleFilter(True)
    f.hist = ["A", "B", "A"]
    a, iv = f.choose("B", [20, 21], 5)
    assert a == 20 and iv is None and f.detected_not_continuing == 1
    # determinism + tie-breaking
    import numpy as np
    m = np.zeros(10, bool)
    m[[2, 5, 7]] = True
    lg = np.zeros(10)
    assert ranking_from_logits(lg, m) == [2, 5, 7]
    lg[7] = 1.0
    assert ranking_from_logits(lg, m) == [7, 2, 5]
    # revisit statistics
    st = revisit_stats(list("ABABABAB"))
    assert st["two_state_cycles"] == 1 and st["two_state_oscillation_steps"] == 6, st
    assert revisit_stats(list("ABCDEF"))["revisits"] == 0 and revisit_stats(list("AAAA"))["two_state_oscillation_steps"] == 0
    return "A,A,A / A,B,A / A,B,A,B / rejection reasons / no_alternative / detected-but-not-continuing / deterministic ties / cycle statistics"


def replay_diag(rows, diag_on):
    from experiments.recording import EpisodeAccumulator
    meta = {"initialization": "normal", "start_gap": 0, "reward_mode": "native_score", "decision_interval_seconds": 1.0, "diagnostics": diag_on}
    acc = EpisodeAccumulator(meta)
    acc.reset_episode(0, None, {"tiles": [], "simulationTime": rows[0]["globals_before"]["sim_time"]} if diag_on else None)
    for r in rows:
        info = {"accepted": r["accepted"], "appliedAction": r["appliedAction"], "applyInfo": r["applyInfo"], "rewardComponents": r["rewardComponents"], "decisionIndex": r["decisionIndex"]}
        acc.add_step(r["action"], r["legal_before"], r["reward"], info, r["terminated"], r["truncated"],
                     raw_after={"tiles": [], "simulationTime": r["globals_after"]["sim_time"]} if diag_on else None, mask_before=None)
    return acc.row(1, None)


@test
def test_terminal_penalty_flag_matches_s5_traces_and_never_touches_rewards():
    zero = normal = 0
    for f in sorted(glob.glob(os.path.join(AUD, "hunt_*.json")) + glob.glob(os.path.join(AUD, "terminal_wait_*.json")) + glob.glob(os.path.join(AUD, "loop_cnn_det_*.json"))):
        rows = load(f)
        on, off = replay_diag(rows, True), replay_diag(rows, False)
        for k in ("return", "native_score", "length_decisions", "terminal", "truncated", "wait_count", "add_count", "remove_count", "upgrade_count"):
            assert on[k] == off[k], (os.path.basename(f), k)
        assert abs(on["return"] - sum(r["reward"] for r in rows)) < 1e-12, "diagnostics changed/ignored the reward"
        last = rows[-1]
        expect_missing = last["rewardComponents"]["terminalPenalty"] == 0 and last["terminated"]
        assert on["diag_terminal_penalty_missing"] is expect_missing, os.path.basename(f)
        assert on["diag_terminal_on_entry"] is (last["applyInfo"] == "terminal state (not applied)")
        fs = on["diag_final_status"]
        assert fs["terminated"] and not fs["truncated"] and fs["rewardComponents"] == last["rewardComponents"] and fs["applyInfo"] == last["applyInfo"]
        if expect_missing:
            zero += 1
            assert fs["sim_time_advance"] < 0.5 and on["return"] == 0.0
        else:
            normal += 1
            assert abs(fs["sim_time_advance"] - 1.0) < 1e-6 and on["return"] == -10.0
        assert on["diag_errors"] == 0
    assert zero >= 2 and normal >= 8
    return f"{zero} zero-penalty and {normal} normal terminal traces: flag == (terminated & penalty==0); terminal_on_entry == applyInfo 'terminal state (not applied)'; sim-time advance of the final step ~0.1 s vs 1.0 s; return/length/action counts identical with diagnostics on vs off; diag_errors 0"


@test
def test_horizon_truncation_is_not_flagged_as_missing_penalty():
    rows = [{"action": 0, "legal_before": True, "reward": 0.0, "accepted": True, "appliedAction": "WAIT", "applyInfo": "WAIT (no-op)", "rewardComponents": {"scoreDelta": 0, "terminalPenalty": 0},
             "decisionIndex": i + 1, "terminated": False, "truncated": i == 4, "globals_before": {"sim_time": float(i)}, "globals_after": {"sim_time": float(i + 1)}} for i in range(5)]
    on = replay_diag(rows, True)
    assert on["diag_terminal_penalty_missing"] in (None, False) and on["diag_terminal_on_entry"] in (None, False)
    return "a horizon-truncated episode (terminated=False) is not reported as a missing penalty"


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    json.dump([{"test": n, "passed": p, "detail": d} for n, p, d in RES], open(os.path.join(OUT, "offline_test_results.json"), "w"), indent=1)
    print(f"\n{sum(r[1] for r in RES)}/{len(RES)} passed")
    sys.exit(0 if all(r[1] for r in RES) else 1)
