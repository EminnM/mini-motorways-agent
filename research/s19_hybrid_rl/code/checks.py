"""S19 diagnostics and checks -> research/s19_hybrid_rl/checks.json
  CH1 equivariance            : permuting candidate slots permutes logits identically (policy.check_equivariance)
  CH2 discounting             : SMDP GAE with gamma^duration == brute-force discounted return on a toy sequence (lambda = 1) + a hand-computed case
  CH3 train/test contamination: no test scenario id in the training log (train + validation scenarios only)
  CH4 action-mask correctness : the learned / random strategic policies never select a masked candidate; the executor never emits an illegal primitive
                                action under the env mask (counts from evaluation_episodes.json and from the runs below)
  CH5 WAIT exploitation       : WAIT frequency vs deliveries; reward per primitive decision by candidate kind; fraction of real-choice states where WAIT was chosen
  CH6 repeated equivalent routes: re-selection of an already selected candidate key; alternative-route selections
  CH7 candidate-ordering leakage: shuffle candidate slots at evaluation -> identical decisions (by candidate key) and identical metrics
  CH8 policy collapse / agreement: C's choices vs A's / B's choices on the same states (C's states and A's states)
  CH9 feature ablations       : zero cost / demand / disruption features at inference, deltas on the test split
python research/s19_hybrid_rl/code/checks.py [checkpoint]"""
import json
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import strategic_env as S  # noqa: E402
import importlib.util as _ilu  # noqa: E402
_spec = _ilu.spec_from_file_location("s19_evaluate", os.path.join(HERE, "evaluate.py"))   # explicit path: another 'evaluate' module shadows the name
EV = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(EV)
from policy import check_equivariance  # noqa: E402
from planner import Weights  # noqa: E402
from train import gae_smdp  # noqa: E402

OUT_DIR = os.path.dirname(HERE)
RUN_DIR = os.path.join(OUT_DIR, "runs", "s19")


# ---------------------------------------------------------------- CH2 discounting
def check_discounting() -> Dict[str, Any]:
    rng = np.random.default_rng(3)
    gamma = 0.9
    T = 12
    r = rng.normal(size=T)
    d = rng.integers(1, 8, size=T)
    v = rng.normal(size=T)
    adv, ret = gae_smdp(r, v, d, gamma, 1.0)
    # brute force: G_t = sum_{j>=t} gamma^{(T_j - T_t)} r_j, with T_j = cumulative elapsed primitive decisions before transition j
    elapsed = np.concatenate([[0], np.cumsum(d)[:-1]])
    brute = np.array([sum(gamma ** (elapsed[j] - elapsed[t]) * r[j] for j in range(t, T)) for t in range(T)])
    err_lambda1 = float(np.abs(ret - brute).max())
    # hand case: r=[1,2,3], d=[2,1,3], v=[0.5,0.25,0.0], gamma=0.5, lambda=0.8
    r2, d2, v2 = np.array([1.0, 2.0, 3.0]), np.array([2, 1, 3]), np.array([0.5, 0.25, 0.0])
    g, lam = 0.5, 0.8
    delta2 = 3.0 + 0 - 0.0
    delta1 = 2.0 + g ** 1 * 0.0 - 0.25
    delta0 = 1.0 + g ** 2 * 0.25 - 0.5
    a2 = delta2                                   # trace decays by gamma^d (per primitive decision) * lambda (per strategic transition)
    a1 = delta1 + g ** 1 * lam * a2
    a0 = delta0 + g ** 2 * lam * a1
    adv2, _ = gae_smdp(r2, v2, d2, g, lam)
    err_hand = float(np.abs(adv2 - np.array([a0, a1, a2])).max())
    # a Markov (d=1) sequence must reduce to standard GAE
    d1 = np.ones(T, int)
    adv_s, _ = gae_smdp(r, v, d1, gamma, 0.95)
    last, std = 0.0, np.zeros(T)
    for t in range(T - 1, -1, -1):
        vn = v[t + 1] if t + 1 < T else 0.0
        last = (r[t] + gamma * vn - v[t]) + gamma * 0.95 * last
        std[t] = last
    err_markov = float(np.abs(adv_s - std).max())
    return {"err_vs_bruteforce_lambda1": err_lambda1, "err_hand_case": err_hand, "err_markov_reduction": err_markov, "pass": max(err_lambda1, err_hand, err_markov) < 1e-5}


# ---------------------------------------------------------------- CH3 contamination
def check_contamination() -> Dict[str, Any]:
    log = json.load(open(os.path.join(RUN_DIR, "train_log.json")))
    seen = set(log["train_scenarios_seen"])
    val_seen = set()
    for v in log["validation"]:
        val_seen |= set(v.get("scenarios", []))
    bad_train = sorted(seen & S.TEST_IDS)
    bad_val = sorted(val_seen & S.TEST_IDS)
    train_ok = all(S.split3(S.SC.index(next(s for s in S.SC if s["id"] == sid))) == "train" for sid in seen)
    return {"train_scenarios_seen": sorted(seen), "n_train_scenarios": len(seen), "validation_scenarios_seen": sorted(val_seen), "test_ids_in_training": bad_train, "test_ids_in_validation": bad_val,
            "all_training_ids_are_train_split": train_ok, "pass": not bad_train and not bad_val and train_ok}


# ---------------------------------------------------------------- recorded runs for CH4-CH9
class Recorder:
    """wraps a chooser; records per decision: chosen key, n_legal, the A / B heuristic keys on the same state, C argmax key."""

    def __init__(self, chooser, pol=None, record_heuristics=True, record_c=False):
        self.ch, self.pol, self.rh, self.rc = chooser, pol, record_heuristics, record_c
        self.rows: List[Dict[str, Any]] = []

    def __call__(self, obs, ctrl):
        slot = int(self.ch(obs, ctrl))
        n_legal = int(obs["mask"].sum())
        key = ("WAIT",) if slot == ctrl.K else S.cand_key(ctrl.cur["cands"][slot])
        row = {"t": ctrl.cur["t"], "slot": slot, "key": key, "n_legal": n_legal, "legal": bool(obs["mask"][slot]), "kind": key[0]}
        if self.rh:
            for nm, w in (("A", Weights(greedy=True)), ("B", Weights())):
                s, k = S.heuristic_choice(ctrl, w)
                row[nm] = k
                row[nm + "_slot"] = s
        if self.rc and self.pol is not None:
            cs = self.pol.act(obs, deterministic=True)[0]
            row["C"] = ("WAIT",) if cs == ctrl.K else S.cand_key(ctrl.cur["cands"][cs])
        self.rows.append(row)
        return slot


def run_recorded(chooser_factory, idxs, pol=None, shuffle_seed=None, ablation=None, record_heuristics=True, record_c=False):
    rows, decisions = [], []
    for idx in idxs:
        for s in range(EV.ENV_SEEDS):
            ctrl = S.StrategicController(shuffle_rng=(np.random.default_rng(shuffle_seed + idx * 10 + s) if shuffle_seed is not None else None), feature_ablation=ablation)
            rec = Recorder(chooser_factory(), pol, record_heuristics, record_c)
            sp = S.StrategicPolicy(rec, ctrl)
            r = EV.episode_ext(sp.act, idx, S.env_seed(idx, s))
            r.update(ctrl.summary())
            rows.append(r)
            decisions.append({"scenario": r["scenario"], "env_seed": r["env_seed"], "rows": rec.rows})
    return rows, decisions


def agreement(decisions, a: str, b: str) -> Dict[str, Any]:
    """agreement between two choice columns over decision states; exact key, same target house, and on real-choice states only (n_legal >= 2)."""
    allr = [x for d in decisions for x in d["rows"] if a in x and b in x]
    multi = [x for x in allr if x["n_legal"] >= 2]
    nonwait = [x for x in multi if x[a] != ("WAIT",) or x[b] != ("WAIT",)]

    def house(k):
        return k[1] if k[0] == "CONNECT" else None

    def rate(rows, f):
        return (float(np.mean([f(x) for x in rows])) if rows else None)
    return {"n_states": len(allr), "n_real_choice_states": len(multi), "exact_all": rate(allr, lambda x: x[a] == x[b]), "exact_real_choice": rate(multi, lambda x: x[a] == x[b]),
            "same_house_real_choice": rate(multi, lambda x: house(x[a]) == house(x[b])), "same_kind_real_choice": rate(multi, lambda x: x[a][0] == x[b][0]),
            "exact_when_either_nonwait": rate(nonwait, lambda x: x[a] == x[b]), "n_either_nonwait": len(nonwait)}


def main(ckpt: str = None):
    ckpt = ckpt or os.path.join(RUN_DIR, "best.pt")
    pol = EV.load_policy(ckpt)
    res: Dict[str, Any] = {"checkpoint": os.path.relpath(ckpt, OUT_DIR)}
    t0 = time.time()
    res["CH1_equivariance"] = check_equivariance()
    res["CH2_discounting"] = check_discounting()
    res["CH3_contamination"] = check_contamination()
    # ---- evaluation rows (if present) for mask / repeat statistics
    ev_rows = json.load(open(os.path.join(OUT_DIR, "evaluation_episodes.json"))) if os.path.exists(os.path.join(OUT_DIR, "evaluation_episodes.json")) else {}
    # ---- C greedy on the test split, recorded (CH4-CH8 base run)
    test = S.TEST_IDX
    base_rows, base_dec = run_recorded(lambda: EV.learned_chooser(pol, True), test, pol, record_heuristics=True)
    stoch_rows, stoch_dec = run_recorded(lambda: EV.learned_chooser(pol, False, 77), test, pol, record_heuristics=True)
    # CH4 mask correctness
    mask = {"base_greedy_illegal_selected": int(sum(r["illegal_selected"] for r in base_rows)), "base_greedy_illegal_primitive": int(sum(r["illegal_primitive"] for r in base_rows)),
            "stochastic_illegal_selected": int(sum(r["illegal_selected"] for r in stoch_rows)), "stochastic_illegal_primitive": int(sum(r["illegal_primitive"] for r in stoch_rows)),
            "recorded_decisions_all_legal": all(x["legal"] for d in base_dec + stoch_dec for x in d["rows"]), "executor_illegal_suppressed": int(sum(r["illegal_suppressed"] for r in base_rows + stoch_rows))}
    for p, rows in ev_rows.items():
        if "illegal_selected" in rows[0]:
            mask[f"eval_{p}_illegal_selected"] = int(sum(r["illegal_selected"] for r in rows))
        mask[f"eval_{p}_illegal_primitive"] = int(sum(r["illegal_primitive"] for r in rows))
    mask["pass"] = all(v == 0 for k, v in mask.items() if k.endswith(("illegal_selected", "illegal_primitive"))) and mask["recorded_decisions_all_legal"]
    res["CH4_action_mask"] = mask
    # CH5 WAIT exploitation: gym run with per-step rewards
    env = S.StrategicEnv(test, seed=0)
    per_kind = {k: {"reward": 0.0, "decisions": 0, "steps": 0} for k in S.KINDS}
    wait_when_choice, choice_states = 0, 0
    for idx in test:
        for s in range(EV.ENV_SEEDS):
            obs, _ = env.reset(options={"scenario_index": idx, "env_seed": S.env_seed(idx, s)})
            while True:
                a = pol.act(obs, deterministic=True)[0]
                if obs["mask"].sum() >= 2:
                    choice_states += 1
                    wait_when_choice += int(a == S.WAIT_SLOT)
                obs, r, te, tr, inf = env.step(a)
                pk = per_kind[inf["kind"]]
                pk["reward"] += r
                pk["decisions"] += inf["duration"]
                pk["steps"] += 1
                if te or tr:
                    break
    wf = np.array([r["wait_frequency"] for r in stoch_rows])
    dl = np.array([r["deliveries"] for r in stoch_rows], float)
    res["CH5_wait_exploitation"] = {"reward_per_primitive_decision_by_kind": {k: (v["reward"] / v["decisions"] if v["decisions"] else None) for k, v in per_kind.items()},
                                   "steps_by_kind": {k: v["steps"] for k, v in per_kind.items()}, "real_choice_states": choice_states, "wait_chosen_in_real_choice_states": wait_when_choice,
                                   "wait_rate_in_real_choice_states": (wait_when_choice / choice_states if choice_states else None),
                                   "corr_wait_frequency_vs_deliveries_stochastic": (float(np.corrcoef(wf, dl)[0, 1]) if wf.std() > 0 else None),
                                   "wait_frequency_mean_stochastic": float(wf.mean()), "note": "WAIT frequency is dominated by WAIT-only states (all houses connected); the exploitation signal is the WAIT rate in real-choice states"}
    # CH6 repeated equivalent routes
    rep = {"greedy_repeated_keys": int(sum(r["repeated_keys"] for r in base_rows)), "greedy_alt_selected": int(sum(r["alt_selected"] for r in base_rows)), "greedy_connect_actions": int(sum(r["kind_counts"]["CONNECT"] for r in base_rows)),
           "stochastic_repeated_keys": int(sum(r["repeated_keys"] for r in stoch_rows)), "stochastic_alt_selected": int(sum(r["alt_selected"] for r in stoch_rows)), "stochastic_connect_actions": int(sum(r["kind_counts"]["CONNECT"] for r in stoch_rows)),
           "greedy_abandoned": int(sum(r["abandoned_plans"] for r in base_rows)), "stochastic_abandoned": int(sum(r["abandoned_plans"] for r in stoch_rows))}
    res["CH6_repeated_routes"] = rep
    # CH7 ordering leakage: shuffled slots
    shuf_rows, shuf_dec = run_recorded(lambda: EV.learned_chooser(pol, True), test, pol, shuffle_seed=5, record_heuristics=False)
    same_keys = [[x["key"] for x in a["rows"]] == [x["key"] for x in b["rows"]] for a, b in zip(base_dec, shuf_dec)]
    same_metrics = [(a["deliveries"], a["decisions"], a["plain_adds"]) == (b["deliveries"], b["decisions"], b["plain_adds"]) for a, b in zip(base_rows, shuf_rows)]
    # tie diagnostics: how often the argmax had an exact tie in logits (identical features)
    res["CH7_ordering_leakage"] = {"episodes": len(same_keys), "identical_decision_sequences": int(sum(same_keys)), "identical_metrics": int(sum(same_metrics)),
                                  "deliveries_median_base": float(np.median([r["deliveries"] for r in base_rows])), "deliveries_median_shuffled": float(np.median([r["deliveries"] for r in shuf_rows])),
                                  "pass": all(same_keys) and all(same_metrics)}
    # CH8 agreement
    agr = {"on_C_greedy_states": {"C_vs_A": agreement(base_dec, "key", "A"), "C_vs_B": agreement(base_dec, "key", "B"), "A_vs_B": agreement(base_dec, "A", "B")},
           "on_C_stochastic_states": {"C_vs_A": agreement(stoch_dec, "key", "A"), "C_vs_B": agreement(stoch_dec, "key", "B")}}

    def a_chooser():
        def ch(obs, ctrl):
            s, k = S.heuristic_choice(ctrl, Weights(greedy=True))
            return s if (s >= 0 and obs["mask"][s]) else ctrl.K
        return ch
    a_rows, a_dec = run_recorded(a_chooser, test, pol, record_heuristics=True, record_c=True)
    agr["on_A_states"] = {"C_argmax_vs_A": agreement(a_dec, "C", "A"), "C_argmax_vs_B": agreement(a_dec, "C", "B"), "A_as_chooser_deliveries_median": float(np.median([r["deliveries"] for r in a_rows]))}
    # kind / ordering preferences: which house does C pick vs A when they differ
    diff = [x for d in base_dec for x in d["rows"] if x["n_legal"] >= 2 and x["key"] != x["A"]]
    agr["C_vs_A_disagreements_greedy"] = {"n": len(diff), "C_wait_A_connect": sum(1 for x in diff if x["key"][0] == "WAIT" and x["A"][0] == "CONNECT"), "C_connect_A_wait": sum(1 for x in diff if x["key"][0] == "CONNECT" and x["A"][0] == "WAIT"),
                                          "both_connect_different_house": sum(1 for x in diff if x["key"][0] == "CONNECT" and x["A"][0] == "CONNECT" and x["key"][1] != x["A"][1]),
                                          "both_connect_same_house_different_route": sum(1 for x in diff if x["key"][0] == "CONNECT" and x["A"][0] == "CONNECT" and x["key"][1] == x["A"][1]),
                                          "A_choice_not_in_slots_or_masked": sum(1 for x in diff if x["A_slot"] < 0)}
    res["CH8_agreement"] = agr
    # CH9 feature ablations at inference (test split, greedy)
    base_sum = EV.summarize(base_rows)
    abl = {"none": {"deliveries_median": base_sum["deliveries_median"], "deliveries_mean": base_sum["deliveries_mean"], "survived_600": base_sum["survived_600"], "new_house_rate_60": base_sum["new_house_rate_60"],
                    "wait_frequency": base_sum["wait_frequency"]}}
    for name, cols in S.FEATURE_GROUPS.items():
        rows, dec = run_recorded(lambda: EV.learned_chooser(pol, True), test, pol, ablation=cols, record_heuristics=False)
        sm = EV.summarize(rows)
        changed = sum(1 for a, b in zip(base_dec, dec) if [x["key"] for x in a["rows"]] != [x["key"] for x in b["rows"]])
        abl[name] = {"zeroed_features": [S.CAND_NAMES[i] for i in cols], "deliveries_median": sm["deliveries_median"], "deliveries_mean": sm["deliveries_mean"], "survived_600": sm["survived_600"],
                     "new_house_rate_60": sm["new_house_rate_60"], "wait_frequency": sm["wait_frequency"], "delta_deliveries_median": sm["deliveries_median"] - base_sum["deliveries_median"],
                     "delta_deliveries_mean": sm["deliveries_mean"] - base_sum["deliveries_mean"], "delta_survived": sm["survived_600"] - base_sum["survived_600"], "episodes_with_changed_decisions": changed}
    res["CH9_feature_ablations"] = abl
    res["wall_s"] = round(time.time() - t0, 1)
    res["checks_pass"] = {k: res[k]["pass"] for k in res if isinstance(res[k], dict) and "pass" in res[k]}
    json.dump(res, open(os.path.join(OUT_DIR, "checks.json"), "w"), indent=1, default=str)
    for k, v in res.items():
        print(k, json.dumps(v, default=str)[:600])
    return res


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
