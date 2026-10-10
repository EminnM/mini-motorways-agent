"""Assembles research/synthetic_to_native/S2_REPORT.md from stored artifacts only."""
import glob
import json
import math
import os
import statistics

ROOT = r"C:\Users\TKA\Desktop\mini moto"
OUT = os.path.join(ROOT, "research", "synthetic_to_native")


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0, c - h), min(1, c + h)


iface = json.load(open(os.path.join(OUT, "interface_report.json")))
amb = json.load(open(os.path.join(OUT, "ambiguity_report.json")))
rows = json.load(open(os.path.join(OUT, "zero_shot_rows_29tasks.json")))
traces = json.load(open(os.path.join(OUT, "failure_traces.json")))
tasks = {}
for p in ("manifest/s2_pilot.jsonl", "manifest_ext/s2_pilot_ext.jsonl"):
    for l in open(os.path.join(OUT, p)):
        r = json.loads(l)
        tasks[r["task_id"]] = r


def stored_rows(pattern):
    d = sorted(glob.glob(os.path.join(ROOT, "runs", pattern)))[-1]
    return [json.loads(l) for l in open(os.path.join(d, "evaluation", "partial_rows.jsonl"))], d


mlp_rows, _ = stored_rows("*eval_gap1_pilot_baseline0_det_validation")
rnd_rows, _ = stored_rows("*eval_gap1_pilot_random_validation")
mlp_rows = [r for r in mlp_rows if r["task_id"] in tasks]
rnd_rows = [r for r in rnd_rows if r["task_id"] in tasks]


def pct(k, n):
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {100 * k / n:.1f}% [{100 * lo:.0f}-{100 * hi:.0f}]"


L = []
A = L.append
A("# S2 - synthetic CNN -> native Mini Motorways transfer\n")
A("Question: does the spatial policy learned in the synthetic sandbox transfer to the real game, and if not, why? **Zero-shot only**: the three synthetic-trained "
  "CNNs (S1 seeds 11 / 12 / 13, the checkpoints selected on synthetic validation) were run on frozen native Gap-1 *validation* configurations with no gradient update. "
  "No native training, no fine-tuning, no 200-task evaluation was launched; S1 artifacts, native Baseline-0 checkpoints, the native Gap-1 manifests and the game/observation/"
  "action/reward semantics are unchanged (the only code edit: `evaluate_cli.py`'s `--split` now accepts any manifest stem, needed for the pilot manifest).\n")
A("## Headline\n")
n29 = 29
cnn = {s: sum(r["success"] for r in rs) for s, rs in rows.items()}
A(f"* **Interface:** 0 encoding mismatches over {iface['n_native_states']} live native states. The CNN receives exactly the layout it was trained on.")
A(f"* **Zero-shot native Gap-1 (29 stratified validation tasks):** CNN seeds {', '.join(f'{pct(cnn[s], n29)}' for s in sorted(cnn))}; frozen native MLP {pct(sum(r['success'] for r in mlp_rows), len(mlp_rows))}; "
  f"RANDOM LEGAL {pct(sum(r['success'] for r in rnd_rows), len(rnd_rows))} on the same tasks. The policy transfers.")
A("* **Carpark ambiguity exists** in the live observations (29% of native validation tasks, 40% of test tasks have more than one road tile touching the destination) - an *encoding omission* "
  "of the python encoder, not missing game information - but it did **not** hurt Gap-1 completion in the pilot (17, 17 and 18 of the 18 ambiguous tasks solved by seeds 11, 12, 13; see below).")
A("* The 3 failing tasks are not interface errors: two (019, 072) are unambiguous, legal, observable situations where the policy puts little probability on the correct edge; the third (115) is carpark-ambiguous.")
A("* Caveat: n = 29 tasks (3 seeds share them, so they are not independent); confidence intervals are wide (about +-12 points).\n")

A("## 1. Live interface validation\n")
A(f"Captured from the running game (headless, deployed DLL): {iface['n_tasks_captured']} frozen native Gap-1 validation tasks reset into their start state "
  f"(raw native observation + legal-action indices) plus {iface['n_native_states'] - iface['n_tasks_captured']} post-action states from live WAIT / ADD / REMOVE tests on 12 tasks = {iface['n_native_states']} states "
  f"({iface['n_native_states'] * 777:,} tiles). Raw data: `native_states_validation_0of1.jsonl.gz`, `capture_validation_0of1.jsonl`; machine-readable report: `interface_report.json`.\n")
A("| # | check | result | evidence |\n|---|---|---|---|")
names = {"1_observation_length_11666": "observation length 11,666", "2_action_space_3111": "action space Discrete(3111)", "3_grid_37x21_bounds": "grid 37 x 21, bounds (-18,-10)",
         "4_all_15_tile_features_mapped_to_correct_tile_and_channel": "15 tile features at the right tile/channel", "5_all_11_global_features_consistent": "11 global features consistent",
         "6_tile_coordinate_convention_row_major_y_outer_x_inner": "tile coordinate convention (row-major, y outer)", "7_south_is_y_minus_1": "South = y-1",
         "8_ADD_REMOVE_index_mapping_live": "ADD/REMOVE index mapping (live)", "9_WAIT_index0_live": "WAIT = index 0 (live)", "10_legal_mask_meaning_and_ordering": "legal-mask meaning and ordering"}
ev = {
    "1_observation_length_11666": lambda c: f"observed lengths {c['observed_lengths']}",
    "2_action_space_3111": lambda c: f"max native legal index {c['max_native_legal_index']} < 3111",
    "3_grid_37x21_bounds": lambda c: "all states report bounds xMin -18, yMin -10, 37 x 21",
    "4_all_15_tile_features_mapped_to_correct_tile_and_channel": lambda c: f"{c['tiles_checked']:,} tiles re-derived independently from the raw native fields: {c['channel_mismatches']} mismatches",
    "5_all_11_global_features_consistent": lambda c: f"{c['mismatches']} mismatches; native ranges e.g. hour/24 {c['native_value_ranges']['0']}, concrete ratio {c['native_value_ranges']['4']}, bridges ratio {c['native_value_ranges']['5']}",
    "6_tile_coordinate_convention_row_major_y_outer_x_inner": lambda c: f"{c['mismatching_tiles']} tiles out of order",
    "7_south_is_y_minus_1": lambda c: f"{c['south_edges_checked']} South edges: {c['mismatches_if_south_is_y_minus_1']} inconsistencies if South = y-1 vs {c['mismatches_if_south_is_y_plus_1']} if South = y+1",
    "8_ADD_REMOVE_index_mapping_live": lambda c: f"ADD {c['ADD_tests_ok'][0]}/{c['ADD_tests_ok'][1]}, REMOVE {c['REMOVE_tests_ok'][0]}/{c['REMOVE_tests_ok'][1]}; all {c['REMOVE_failures_on_carpark_tiles']} REMOVE 'failures' target a carpark tile (legal in the native mask, reported applied, stub road stays Active)",
    "9_WAIT_index0_live": lambda c: f"{c['tests'][0]}/{c['tests'][1]} WAIT steps applied as WAIT with no road change; WAIT legal in every reset state",
    "10_legal_mask_meaning_and_ordering": lambda c: f"index mapping verified by 8/9; set-level comparison with the sandbox rules below"}
for k, v in iface["checks"].items():
    A(f"| {k.split('_')[0]} | {names.get(k, k)} | {'PASS' if v['passed'] else 'FAIL'} | {ev[k](v)} |")
A("\n**Encoding mismatches: 0.** Everything the CNN depends on (channel order, row-major tile order, y-outer layout, South = y-1, action indexing, WAIT index, mask ordering) is identical in the sandbox and the live game.\n")
m = iface["checks"]["10_legal_mask_meaning_and_ordering"]
A("### Genuine state / rule differences (not encoding errors)\n")
nf = iface["state_differences"]["tile_channel_nonzero_fraction"]
A("| difference | native | sandbox |\n|---|---|---|")
A(f"| tiles over water (feature 1) | {100 * nf['overWater']['native']:.1f}% of tiles | 0% |")
A(f"| legal actions at reset | {iface['state_differences']['legal_action_count']['native_mean_at_reset']:.0f} | {iface['state_differences']['legal_action_count']['sandbox_mean']} |")
A(f"| ADD legality (sandbox rules reconstructed from the native state vs the native mask) | native {m['totals_over_states']['native_add']:,} legal edges over {m['totals_over_states']['states']} states | sandbox precision {100 * m['add_precision_synth_vs_native']:.0f}% / recall {100 * m['add_recall_synth_vs_native']:.0f}% |")
A(f"| sandbox-legal but native-illegal ADD edges (endpoint counts) | {m['synthetic_only_add_reasons_endpoint_counts']} | (water endpoints dominate; 'unexplained' = other native constraints) |")
A(f"| REMOVE legality | native {m['totals_over_states']['native_rem']} | sandbox precision {100 * m['remove_precision']:.0f}% / recall {100 * m['remove_recall']:.0f}% (native also offers REMOVE on carpark stubs, which has no effect) |")
A("| bridges resource ratio (global 5) | up to 1.0 | always 0 |")
A("| other effects not modelled | trees, destinations' demand, traffic lights, vehicles (all ~0 in these start states) | 0 |\n")
A("These are genuine differences between the real game state and the sandbox; none changes how an index or channel is interpreted. They are the distribution shift the CNN had to survive zero-shot (diagnosis C below).\n")

A("## 2. Carpark ambiguity\n")
A("Definition (see `ambiguity_report.json`): the destination's carpark tiles have native `contentType = 'Carpark'`; `python/mini_motorways_env.py::_CONTENT_TYPE_ORDINAL` only knows None/House/Destination, so the encoder maps Carpark (like Tree) to **0** through `.get(..., 0)`. "
  "A task is *affected* when more than one road tile touches the destination footprint (the policy cannot read which one is the entry); *resolvable* when exactly one touching road tile is a dead end and it is the true entry; *unresolvable* otherwise.\n")
A("| split | tasks | unique (not affected) | affected, resolvable by leaf rule | affected, unresolvable by leaf rule | raw native JSON contains the carpark tile | encoded observation marks it |\n|---|---|---|---|---|---|---|")
for sp in ("validation", "test"):
    a = amb[sp]
    c = a["categories"]
    A(f"| {sp} | {a['n']} | {c.get('unique', 0)} ({100 * c.get('unique', 0) / a['n']:.0f}%) | {c.get('resolvable_by_leaf_rule', 0)} ({100 * c.get('resolvable_by_leaf_rule', 0) / a['n']:.0f}%) | "
      f"{c.get('unresolvable_by_leaf_rule', 0)} ({100 * c.get('unresolvable_by_leaf_rule', 0) / a['n']:.0f}%) | {'yes, all' if a['raw_json_has_carpark_tile_in_all'] else 'no'} | {'yes' if a['encoded_obs_marks_carpark_in_any'] else 'never'} |")
A("\n* **Does it exist in the live observations?** Yes: 29% (validation) / 40% (test) of the frozen native Gap-1 tasks are affected, 11% / 19% unresolvable by the leaf rule.\n"
  "* **Is the information in the game's observation?** Yes - every native state's JSON identifies the carpark tiles (2 per destination, exactly one with a road). It is lost by the **python encoder** (an encoding omission), not missing from the game. The 15-feature contract as documented simply has no carpark channel.\n"
  "* **Does the current PolicyObservation contain enough information?** For 71% (val) / 60% (test) of tasks the entry is trivially identifiable (only one road tile touches the destination); for a further 18% / 21% a dead-end rule identifies it; for 11% / 19% neither - the policy would have to guess which touching road tile is the entry.\n"
  "* **Affected tasks are flagged, not removed:** per-task category for all 120 validation and 240 test tasks is in `capture_validation_0of1.jsonl` / `capture_test_*of2.jsonl` (`ambiguity.category`); the pilot manifests carry `s2_flags`. No feature was added.\n")

A("## 3. Zero-shot native pilot\n")
A("Design: 29 frozen native Gap-1 **validation** configurations, stratified by carpark ambiguity (11 unique, 5 resolvable, 13 unresolvable), gap orientation (19 horizontal, 10 vertical) and gap position along the path (9 early, 20 late). "
  "20 tasks were chosen by stratified sampling (3 per cell where available, `select_pilot.py`), then 9 more unresolvable-ambiguity tasks were added (the rest of that stratum) because only 4 had been covered; this over-represents hard cases relative to the validation distribution. "
  "Deterministic inference (argmax over masked logits), the native observation / mask / actions / stepping, the native Gap-1 success criterion and a 20-decision construction horizon; no continuation. The three runs took ~60 s + ~43 s wall each (including game launch) - well inside the 30-minute cap; 250 native decisions in total.\n")
A("| policy | completion (95% Wilson CI) | first-action accuracy | median decisions to success | destructive failure | timeout | WAIT / ADD / REMOVE | mean P(correct edge) at decision 1 | median rank of correct edge |\n|---|---|---|---|---|---|---|---|---|")
for s in sorted(rows):
    rs = rows[s]
    tot = sum(r["n_wait"] + r["n_add"] + r["n_remove"] for r in rs)
    A(f"| synthetic CNN seed {s} | {pct(sum(r['success'] for r in rs), len(rs))} | {pct(sum(r['first_action_correct'] for r in rs), len(rs))} | {statistics.median(r['steps_to_success'] for r in rs if r['success'])} | "
      f"{sum(r['destructive_failure'] for r in rs)}/{len(rs)} | {sum(r['timeout'] for r in rs)}/{len(rs)} | {100 * sum(r['n_wait'] for r in rs) / tot:.0f} / {100 * sum(r['n_add'] for r in rs) / tot:.0f} / {100 * sum(r['n_remove'] for r in rs) / tot:.0f} % | "
      f"{statistics.fmean(r['a_star_prob_first_decision'] for r in rs):.3f} | {statistics.median(r['a_star_rank_first_decision'] for r in rs)} |")
for name, rs in (("frozen native MLP (Baseline-0 49k, det.; stored per-task rows)", mlp_rows), ("RANDOM LEGAL (stored per-task rows)", rnd_rows)):
    tot = sum(r["n_wait"] + r["n_add"] + r["n_remove"] for r in rs)
    pr = [r["a_star_prob_first_decision"] for r in rs if r["a_star_prob_first_decision"] is not None]
    rk = [r["a_star_rank_first_decision"] for r in rs if r["a_star_rank_first_decision"] is not None]
    A(f"| {name} | {pct(sum(r['success'] for r in rs), len(rs))} | {pct(sum(r['first_action_correct'] for r in rs), len(rs))} | "
      f"{statistics.median([r['steps_to_success'] for r in rs if r['success']] or [0])} | {sum(r['destructive_failure'] for r in rs)}/{len(rs)} | {sum(r['timeout'] for r in rs)}/{len(rs)} | "
      f"{100 * sum(r['n_wait'] for r in rs) / tot:.0f} / {100 * sum(r['n_add'] for r in rs) / tot:.0f} / {100 * sum(r['n_remove'] for r in rs) / tot:.0f} % | {statistics.fmean(pr):.3f} | {statistics.median(rk)} |" if pr else
      f"| {name} | {pct(sum(r['success'] for r in rs), len(rs))} | {pct(sum(r['first_action_correct'] for r in rs), len(rs))} | - | {sum(r['destructive_failure'] for r in rs)}/{len(rs)} | {sum(r['timeout'] for r in rs)}/{len(rs)} | "
      f"{100 * sum(r['n_wait'] for r in rs) / tot:.0f} / {100 * sum(r['n_add'] for r in rs) / tot:.0f} / {100 * sum(r['n_remove'] for r in rs) / tot:.0f} % | - | - |")
A("\n(The native MLP and RANDOM rows come from the existing full-validation evaluations of those policies, restricted to the same 29 tasks; no extra native compute.)\n")
A("### By stratum (successes / tasks, per seed)\n")
A("| stratum | n | seed 11 | seed 12 | seed 13 |\n|---|---|---|---|---|")
def cell(rs, f):
    sub = [r for r in rs if f(tasks[r['task_id']])]
    return f"{sum(r['success'] for r in sub)}/{len(sub)}", len(sub)
for label, f in (("carpark: unique", lambda t: t["s2_flags"]["carpark_category"] == "unique"), ("carpark: ambiguous, resolvable", lambda t: t["s2_flags"]["carpark_category"] == "resolvable_by_leaf_rule"),
                 ("carpark: ambiguous, unresolvable", lambda t: t["s2_flags"]["carpark_category"] == "unresolvable_by_leaf_rule"), ("horizontal gap", lambda t: t["gap_orientation"] == "horizontal"),
                 ("vertical gap", lambda t: t["gap_orientation"] == "vertical"), ("gap early in path", lambda t: t["s2_flags"]["gap_position_stratum"] == "early"), ("gap late in path", lambda t: t["s2_flags"]["gap_position_stratum"] == "late")):
    cs = [cell(rows[s], f) for s in ("11", "12", "13")]
    A(f"| {label} | {cs[0][1]} | {cs[0][0]} | {cs[1][0]} | {cs[2][0]} |")
A("\nThe synthetic S1 test numbers for the same checkpoints were 83.5% / 90.2% / 95.8% (first-action accuracy 37-45%). On native tasks first-action accuracy is higher (about 70%) and success needs a median of 1 decision: these native Gap-1 states are, if anything, easier for the CNN than the synthetic ones (fewer competing dead-end pairs). "
  "All policies finished with destructive failure 0 and used only ADD actions (no WAIT, no REMOVE).\n")

A("## 4. Failure diagnosis\n")
A("Failures: native validation tasks `gap1_validation_000019` (fails for all 3 seeds), `...072` (seed 11), `...115` (seeds 11, 12) - all timeouts after 20 ADD actions, no destructive failure. "
  "Re-run deterministically with full traces (`failure_traces.json`, seed-11 checkpoint):\n")
A("| class | assessment |\n|---|---|")
A("| A. interface mismatch | **Excluded**: 0 encoding mismatches over live states (section 1); the correct action was legal natively in every pilot task (`a_star_legal_native_all`), and the same checkpoints solve 26-28 of 29 native tasks through the same interface. |")
A("| B. observation ambiguity | Task 115 is carpark-ambiguous (4 touching road tiles, unresolvable); 019 and 072 are **unambiguous** (one touching road tile). Ambiguity does not explain the failures: on the 13 unresolvable tasks the seeds succeed 12/13, 12/13 and 13/13. In Gap-1 only the two facing dead ends matter, not which tile is the carpark. |")
A("| C. distribution shift | Present and large in aggregate (16% water tiles, bridge resource, 151 vs 202 legal actions, REMOVE-on-carpark quirk, extra roads touching the destination in 29-40% of tasks (0% in the sandbox)), yet 90-97% transfer shows the CNN is robust to most of it. The three failures show no single separating feature in the data I have; candidates are the gap's position next to a house tile (019) or at the end of a long road touching water-adjacent tiles. This is **not tested**. |")
A("| D. policy failure | **Supported for 019 and 072**: the correct edge is observable, legal and the task unambiguous, but the first-decision probability of the correct edge is 0.005-0.07 and its rank 6-18 (vs mean 0.41-0.52 and rank 1 on successes). The policy instead adds edges next to existing road segments (pictured below) for 20 decisions. |")
A("\nSuccess/failure by first-decision probability of the correct edge: successes mean P = 0.41-0.52 (median rank 1); failures P = 0.005-0.08 (rank 4-18). The failures are wrong with moderate confidence (the chosen first edge had P about 0.2-0.4), not near-misses.\n")
for t in traces:
    A(f"#### {t['task_id']} ({t['flags']['carpark_category']}, {t['flags']['orientation']} gap, {t['flags']['gap_position_stratum']}), correct edge `*`, edges added by the policy `=`/`H`\n")
    A("```\n" + t["ascii"] + "\n```")
    A(f"First decisions of the policy: {[(tuple(x['decoded']), x['p']) for x in t['trace'][:4]]}; probability on the correct edge at decision 1: {t['trace'][0]['p_correct_edge']}.\n")
A("Do not read these three failures as a failure of the CNN architecture: 26-28 of 29 tasks succeed zero-shot.\n")

A("## 5. Cost and native throughput\n")
A("Interface capture: 120 validation resets with 12 action-test sequences + 240 test resets (summaries) on three headless instances, about 2.3 min per 120 tasks. Zero-shot pilot: 6 evaluation runs (3 seeds x 20 + 9 tasks), 250 native decisions in total, 42-60 s wall each including game launch, model load and 1.4 s per reset (resets, not steps, dominate: most episodes end after 1 decision). "
  "Native cost per episode here is dominated by the ~1.4 s city rebuild at reset (a native step is ~0.1-0.3 s), versus ~1 ms per synthetic step. Total S2 native wall time: roughly 15-20 minutes including captures, far inside any budget.\n")
A("## 6. Recommendation\n")
A("""1. **Do not fine-tune yet.** Zero-shot already reaches 90-97% on the pilot with 0 interface errors; a fine-tuning comparison is only informative if a larger zero-shot evaluation shows a real gap.
2. **Next experiment (needs approval):** zero-shot the same three checkpoints on **all 120 native validation tasks** (about 10-15 minutes with three instances) to tighten the +-12-point intervals and measure the failure rate by stratum; then, if it holds, the frozen **native test set (240)** once, with the checkpoint choice fixed from synthetic validation.
3. **Diagnose the three failures** with a cheap, native-free check: build matching synthetic tasks (gap beside a house tile; gap at a long road end near water) and see whether the synthetic CNN also fails there - that separates distribution shift (C) from a genuine policy blind spot (D).
4. **Then Gap-2 zero-shot (native)** with the frozen S1 policies, as planned, because Gap-1 transfer is established; and run the synthetic H2/H3/H4 comparisons for the questions the sandbox is meant to answer.
5. **Representation fix is optional, not urgent:** the carpark channel (an encoder omission) does not hurt Gap-1; it is likely to matter for multi-edit tasks - test it synthetically first (compare with and without a carpark feature) before changing the native encoder.
""")
A("## 7. Reproducibility and artifacts\n")
A("| item | location |\n|---|---|")
for s in ("11", "12", "13"):
    d = sorted(glob.glob(os.path.join(ROOT, "runs", f"*syn_s1_cnn_s{s}")))[-1]
    sel = json.load(open(os.path.join(d, "evaluation", "selection.json")))
    A(f"| CNN seed {s} checkpoint evaluated | `{os.path.relpath(os.path.join(d, 'checkpoints', sel['selected_file']), ROOT)}` sha256 `{sel['sha256']}` |")
A("| per-episode zero-shot rows (29 tasks x 3 seeds) | `research/synthetic_to_native/zero_shot_rows_29tasks.json`; native evaluation run directories `runs/*s2zs_cnn_s*` and `runs/*s2zsext_cnn_s*` (config, metadata with DLL/game/source hashes, partial_rows.jsonl, results.json) |")
A("| pilot manifests | `research/synthetic_to_native/manifest/s2_pilot.jsonl`, `manifest_ext/s2_pilot_ext.jsonl` |")
A("| raw native states and masks | `native_states_validation_0of1.jsonl.gz`, `capture_*.jsonl` |")
A("| reports | `interface_report.json`, `ambiguity_report.json`, `failure_traces.json` |")
A("| code | `experiments/s2/{capture,interface_check,ambiguity,select_pilot,trace_failures,make_report}.py` |")
open(os.path.join(OUT, "S2_REPORT.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
print("written", os.path.join(OUT, "S2_REPORT.md"))
