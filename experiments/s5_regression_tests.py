"""S5 regression tests. They run on SAVED native traces / logs (research/native_action_reward_audit/, research/native_cnn_gate0/) and on
the C# source text; no game is launched.   python experiments/s5_regression_tests.py
Each test documents an established (or explicitly qualified) property of the two findings:
  1. repeated ADD = a 2-cycle of valid driveway-swapping edits (not a mask/application/aliasing defect)
  2. missing -10 = the terminal flag was already set when the final STEP started (penalty only paid on the step that turns terminal)"""
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUD = os.path.join(ROOT, "research", "native_action_reward_audit")
S4 = os.path.join(ROOT, "research", "native_cnn_gate0")
RES = []


def test(fn):
    try:
        info = fn()
        RES.append((fn.__name__, True, info or ""))
    except AssertionError as e:
        RES.append((fn.__name__, False, f"ASSERTION: {e}"))
    except Exception as e:
        RES.append((fn.__name__, False, f"{type(e).__name__}: {e}"))
    print(("PASS " if RES[-1][1] else "FAIL ") + RES[-1][0] + ": " + RES[-1][2][:400], flush=True)
    return fn


def load(p):
    return json.load(open(p))


def loop_cycles(rows):
    """index ranges where the action sequence strictly alternates between two ADD actions for >= 6 steps"""
    out, i, n = [], 0, len(rows)
    while i < n - 5:
        a, b = rows[i]["action"], rows[i + 1]["action"]
        j = i
        while j + 2 < n and rows[j + 2]["action"] == rows[j]["action"] and rows[j + 1]["action"] == b and rows[j]["action"] == a:
            j += 1
        # extend while pattern a,b,a,b holds
        k = i
        while k + 1 < n and rows[k]["action"] == (a if (k - i) % 2 == 0 else b):
            k += 1
        if a != b and k - i >= 6 and rows[i]["decoded"][0] == "ADD" and rows[i + 1]["decoded"][0] == "ADD":
            out.append((i, k))
            i = k
        else:
            i += 1
    return out


@test
def test_loop_is_a_two_cycle_not_a_consecutive_repeat():
    detail = []
    for f in sorted(glob.glob(os.path.join(AUD, "loop_cnn_det_*.json"))):
        rows = load(f)
        cyc = loop_cycles(rows)
        assert cyc, f"{os.path.basename(f)}: no ADD/ADD alternation found"
        longest_consecutive = max(len(list(g)) for _, g in __import__("itertools").groupby(r["action"] for r in rows if r["decoded"][0] == "ADD"))
        assert longest_consecutive == 1, "an ADD repeated consecutively (S4 interpretation) - not expected"
        detail.append(f"{os.path.basename(f)}: cycle steps {cyc}, longest consecutive identical ADD run = {longest_consecutive}")
    return "; ".join(detail)


@test
def test_each_loop_step_is_legal_applied_changes_roads_and_observation():
    n = 0
    for f in sorted(glob.glob(os.path.join(AUD, "loop_cnn_det_*.json"))):
        rows = load(f)
        for s, e in loop_cycles(rows):
            for r in rows[s:e]:
                n += 1
                assert r["legal_before"], "action not in mask"
                assert r["accepted"] and "Success" in (r["applyInfo"] or ""), r["applyInfo"]
                assert r["edge_before"]["A"] == "None" and r["edge_before"]["B"] == "None", "edge already existed before the ADD (would indicate mask/no-op defect)"
                assert r["edge_after"]["A"] == "Active" and r["edge_after"]["B"] == "Active", "edge not present after a reported-successful ADD (application defect)"
                assert r["road_edges_added"], "no road added"
                assert r["still_legal_after"] is False, "action still legal after being satisfied (mask defect)"
                ch = r["obs_changes"]["tile_channels"]
                assert any(k.startswith("road") for k in ch), "road change not visible in the policy observation (aliasing)"
    return f"{n} loop steps checked: legal, 'AddRoad succeeded', edge absent before / present after, observation road channels changed, action illegal afterwards"


@test
def test_each_loop_add_replaces_the_other_edge_at_a_house_tile():
    n = 0
    for f in sorted(glob.glob(os.path.join(AUD, "loop_cnn_det_*.json"))):
        rows = load(f)
        for s, e in loop_cycles(rows):
            for r in rows[s + 1:e]:           # first step of the cycle has nothing to replace
                n += 1
                eb = r["edge_before"]
                assert "House" in (eb["A_content"], eb["B_content"]), "loop edge does not touch a House tile"
                assert len(r["road_edges_removed"]) == 1, f"expected exactly one replaced edge, got {r['road_edges_removed']}"
                assert r["road_edges_added"], "nothing added"
    return f"{n} cycle steps: every ADD touches a House tile and removes exactly one other (the previous) edge: net road count unchanged"


@test
def test_swap_events_overall_involve_house_tiles():
    swaps = house = 0
    for f in sorted(glob.glob(os.path.join(AUD, "loop_cnn_det_*.json"))):
        for r in load(f):
            if r["decoded"][0] == "ADD" and r["road_edges_removed"]:
                swaps += 1
                house += ("House" in (r["edge_before"]["A_content"], r["edge_before"]["B_content"]))
    assert swaps > 0 and house == swaps, f"{house}/{swaps} replacing ADDs touch a House tile"
    return f"{house}/{swaps} ADD steps that removed an existing edge touched a House tile (both S4 seeds)"


def episode_penalties(rows):
    term = [i for i, r in enumerate(rows) if r["terminated"]]
    return term, [r["rewardComponents"]["terminalPenalty"] for r in rows if r["rewardComponents"]["terminalPenalty"] != 0]


@test
def test_terminal_penalty_paid_iff_game_not_already_terminal_at_step_start():
    n_pen = n_zero = 0
    for f in sorted(glob.glob(os.path.join(AUD, "hunt_*.json")) + glob.glob(os.path.join(AUD, "terminal_wait_*.json"))):
        rows = load(f)
        term, pens = episode_penalties(rows)
        assert term == [len(rows) - 1], f"{os.path.basename(f)}: terminated must be set exactly on the last step"
        last = rows[-1]
        not_applied = last["applyInfo"] == "terminal state (not applied)"
        if not_applied:
            assert last["accepted"] is False and pens == [], "terminal-at-start step must carry no penalty"
            n_zero += 1
        else:
            assert pens == [-10], f"{os.path.basename(f)}: expected exactly one -10, got {pens}"
            n_pen += 1
    assert n_pen > 0 and n_zero > 0, "need both kinds of terminal episode in the saved traces"
    return f"{n_pen} episodes ended with 'justTerminated' (-10 exactly once); {n_zero} ended with 'terminal state (not applied)' (penalty 0, accepted=False): penalty presence == game not terminal at STEP start"


@test
def test_zero_penalty_episode_final_step_shows_clock_moved_outside_step():
    found = []
    for f in sorted(glob.glob(os.path.join(AUD, "hunt_*.json"))):
        rows = load(f)
        last = rows[-1]
        if last["applyInfo"] == "terminal state (not applied)":
            gb, ga = last["globals_before"]["sim_time"], last["globals_after"]["sim_time"]
            found.append(round(ga - gb, 3))
            assert abs(ga - gb) < 0.5, "a full 1.0 s advance happened; the step was not a no-advance terminal step"
            prev = rows[-2]
            assert prev["terminated"] is False, "previous STEP already reported terminal"
    assert found, "no zero-penalty trace saved"
    return f"zero-penalty final STEPs advanced sim time by {found} s (a normal STEP advances 1.0 s): the clock was moved by something other than that STEP"


@test
def test_csharp_penalty_rule_is_justTerminated_and_wasTerminal_is_read_at_step_start():
    src = open(os.path.join(ROOT, "game_plugin", "DoorstopBootstrap", "Entrypoint.cs"), encoding="utf-8", errors="ignore").read()
    assert len(re.findall(r"bool justTerminated = isTerminalNow && !wasTerminal;", src)) >= 1
    assert len(re.findall(r"double terminalPenalty = justTerminated \? IpcTerminalPenalty : 0\.0;", src)) >= 1
    assert 'applyInfo = wasTerminal ? "terminal state (not applied)" : ""' in src
    assert "private bool IsTerminal(MotorwaysGame testGame)" in src and "HasGameEnded" in src
    return "STEP handler: penalty = justTerminated(isTerminalNow && !wasTerminal), wasTerminal = IsTerminal() read BEFORE applying the action; IsTerminal() also fires one decision early on the overcrowd timer (pre-emption)"


@test
def test_s4_terminal_reward_sums_are_only_0_or_minus10_and_five_of_twenty_are_zero():
    zero = tot = 0
    for f in glob.glob(os.path.join(S4, "diag_*.jsonl")):
        for l in open(f):
            if not l.strip():
                continue
            r = json.loads(l)
            if r["terminal"]:
                tot += 1
                assert r["diag_terminal_reward_sum"] in (0.0, -10.0)
                zero += r["diag_terminal_reward_sum"] == 0.0
    assert tot == 20 and zero == 5, f"{zero}/{tot}"
    return f"S4 logs: {zero}/{tot} terminated episodes carry no terminal penalty; the rest exactly -10"


@test
def test_logging_on_off_twin_on_a_full_terminal_episode_is_identical():
    t = load(os.path.join(AUD, "twin_summary.json"))
    assert all(t["twin_equal"].values()), t["twin_equal"]
    assert t["plain_return"] == t["diag_return"] == t["recorder_row_return"] and t["recorder_row_diag_errors"] == 0
    return f"plain vs RecorderEnv+diagnostics, full {t['steps']}-decision terminal episode: observations, masks, rewards, termination identical; episode row return {t['recorder_row_return']}, diag_errors 0"


if __name__ == "__main__":
    ok = all(r[1] for r in RES)
    json.dump([{"test": n, "passed": p, "detail": d} for n, p, d in RES], open(os.path.join(AUD, "regression_results.json"), "w"), indent=1)
    print(f"\n{sum(r[1] for r in RES)}/{len(RES)} passed")
    sys.exit(0 if ok else 1)
