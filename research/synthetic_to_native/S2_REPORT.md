# S2 - synthetic CNN -> native Mini Motorways transfer

Question: does the spatial policy learned in the synthetic sandbox transfer to the real game, and if not, why? **Zero-shot only**: the three synthetic-trained CNNs (S1 seeds 11 / 12 / 13, the checkpoints selected on synthetic validation) were run on frozen native Gap-1 *validation* configurations with no gradient update. No native training, no fine-tuning, no 200-task evaluation was launched; S1 artifacts, native Baseline-0 checkpoints, the native Gap-1 manifests and the game/observation/action/reward semantics are unchanged (the only code edit: `evaluate_cli.py`'s `--split` now accepts any manifest stem, needed for the pilot manifest).

## Headline

* **Interface:** 0 encoding mismatches over 156 live native states. The CNN receives exactly the layout it was trained on.
* **Zero-shot native Gap-1 (29 stratified validation tasks):** CNN seeds 26/29 = 89.7% [74-96], 27/29 = 93.1% [78-98], 28/29 = 96.6% [83-99]; frozen native MLP 2/29 = 6.9% [2-22]; RANDOM LEGAL 4/29 = 13.8% [5-31] on the same tasks. The policy transfers.
* **Carpark ambiguity exists** in the live observations (29% of native validation tasks, 40% of test tasks have more than one road tile touching the destination) - an *encoding omission* of the python encoder, not missing game information - but it did **not** hurt Gap-1 completion in the pilot (17, 17 and 18 of the 18 ambiguous tasks solved by seeds 11, 12, 13; see below).
* The 3 failing tasks are not interface errors: two (019, 072) are unambiguous, legal, observable situations where the policy puts little probability on the correct edge; the third (115) is carpark-ambiguous.
* Caveat: n = 29 tasks (3 seeds share them, so they are not independent); confidence intervals are wide (about +-12 points).

## 1. Live interface validation

Captured from the running game (headless, deployed DLL): 120 frozen native Gap-1 validation tasks reset into their start state (raw native observation + legal-action indices) plus 36 post-action states from live WAIT / ADD / REMOVE tests on 12 tasks = 156 states (121,212 tiles). Raw data: `native_states_validation_0of1.jsonl.gz`, `capture_validation_0of1.jsonl`; machine-readable report: `interface_report.json`.

| # | check | result | evidence |
|---|---|---|---|
| 1 | observation length 11,666 | PASS | observed lengths [11666] |
| 2 | action space Discrete(3111) | PASS | max native legal index 2875 < 3111 |
| 3 | grid 37 x 21, bounds (-18,-10) | PASS | all states report bounds xMin -18, yMin -10, 37 x 21 |
| 4 | 15 tile features at the right tile/channel | PASS | 121,212 tiles re-derived independently from the raw native fields: 0 mismatches |
| 6 | tile coordinate convention (row-major, y outer) | PASS | 0 tiles out of order |
| 5 | 11 global features consistent | PASS | 0 mismatches; native ranges e.g. hour/24 [0.0, 0.208], concrete ratio [0.56, 0.88], bridges ratio [0.0, 1.0] |
| 7 | South = y-1 | PASS | 480 South edges: 0 inconsistencies if South = y-1 vs 416 if South = y+1 |
| 8 | ADD/REMOVE index mapping (live) | PASS | ADD 12/12, REMOVE 7/12; all 5 REMOVE 'failures' target a carpark tile (legal in the native mask, reported applied, stub road stays Active) |
| 9 | WAIT = index 0 (live) | PASS | 12/12 WAIT steps applied as WAIT with no road change; WAIT legal in every reset state |
| 10 | legal-mask meaning and ordering | PASS | index mapping verified by 8/9; set-level comparison with the sandbox rules below |

**Encoding mismatches: 0.** Everything the CNN depends on (channel order, row-major tile order, y-outer layout, South = y-1, action indexing, WAIT index, mask ordering) is identical in the sandbox and the live game.

### Genuine state / rule differences (not encoding errors)

| difference | native | sandbox |
|---|---|---|
| tiles over water (feature 1) | 16.3% of tiles | 0% |
| legal actions at reset | 151 | 202 |
| ADD legality (sandbox rules reconstructed from the native state vs the native mask) | native 17,847 legal edges over 120 states | sandbox precision 82% / recall 98% |
| sandbox-legal but native-illegal ADD edges (endpoint counts) | {'unexplained_by_terrain_content': 1637, 'endpoint_over_water': 2062, 'endpoint_content_Tree': 73} | (water endpoints dominate; 'unexplained' = other native constraints) |
| REMOVE legality | native 154 | sandbox precision 97% / recall 60% (native also offers REMOVE on carpark stubs, which has no effect) |
| bridges resource ratio (global 5) | up to 1.0 | always 0 |
| other effects not modelled | trees, destinations' demand, traffic lights, vehicles (all ~0 in these start states) | 0 |

These are genuine differences between the real game state and the sandbox; none changes how an index or channel is interpreted. They are the distribution shift the CNN had to survive zero-shot (diagnosis C below).

## 2. Carpark ambiguity

Definition (see `ambiguity_report.json`): the destination's carpark tiles have native `contentType = 'Carpark'`; `python/mini_motorways_env.py::_CONTENT_TYPE_ORDINAL` only knows None/House/Destination, so the encoder maps Carpark (like Tree) to **0** through `.get(..., 0)`. A task is *affected* when more than one road tile touches the destination footprint (the policy cannot read which one is the entry); *resolvable* when exactly one touching road tile is a dead end and it is the true entry; *unresolvable* otherwise.

| split | tasks | unique (not affected) | affected, resolvable by leaf rule | affected, unresolvable by leaf rule | raw native JSON contains the carpark tile | encoded observation marks it |
|---|---|---|---|---|---|---|
| validation | 120 | 85 (71%) | 22 (18%) | 13 (11%) | yes, all | never |
| test | 240 | 145 (60%) | 50 (21%) | 45 (19%) | yes, all | never |

* **Does it exist in the live observations?** Yes: 29% (validation) / 40% (test) of the frozen native Gap-1 tasks are affected, 11% / 19% unresolvable by the leaf rule.
* **Is the information in the game's observation?** Yes - every native state's JSON identifies the carpark tiles (2 per destination, exactly one with a road). It is lost by the **python encoder** (an encoding omission), not missing from the game. The 15-feature contract as documented simply has no carpark channel.
* **Does the current PolicyObservation contain enough information?** For 71% (val) / 60% (test) of tasks the entry is trivially identifiable (only one road tile touches the destination); for a further 18% / 21% a dead-end rule identifies it; for 11% / 19% neither - the policy would have to guess which touching road tile is the entry.
* **Affected tasks are flagged, not removed:** per-task category for all 120 validation and 240 test tasks is in `capture_validation_0of1.jsonl` / `capture_test_*of2.jsonl` (`ambiguity.category`); the pilot manifests carry `s2_flags`. No feature was added.

## 3. Zero-shot native pilot

Design: 29 frozen native Gap-1 **validation** configurations, stratified by carpark ambiguity (11 unique, 5 resolvable, 13 unresolvable), gap orientation (19 horizontal, 10 vertical) and gap position along the path (9 early, 20 late). 20 tasks were chosen by stratified sampling (3 per cell where available, `select_pilot.py`), then 9 more unresolvable-ambiguity tasks were added (the rest of that stratum) because only 4 had been covered; this over-represents hard cases relative to the validation distribution. Deterministic inference (argmax over masked logits), the native observation / mask / actions / stepping, the native Gap-1 success criterion and a 20-decision construction horizon; no continuation. The three runs took ~60 s + ~43 s wall each (including game launch) - well inside the 30-minute cap; 250 native decisions in total.

| policy | completion (95% Wilson CI) | first-action accuracy | median decisions to success | destructive failure | timeout | WAIT / ADD / REMOVE | mean P(correct edge) at decision 1 | median rank of correct edge |
|---|---|---|---|---|---|---|---|---|
| synthetic CNN seed 11 | 26/29 = 89.7% [74-96] | 21/29 = 72.4% [54-85] | 1.0 | 0/29 | 3/29 | 0 / 100 / 0 % | 0.475 | 1 |
| synthetic CNN seed 12 | 27/29 = 93.1% [78-98] | 20/29 = 69.0% [51-83] | 1 | 0/29 | 2/29 | 0 / 100 / 0 % | 0.389 | 1 |
| synthetic CNN seed 13 | 28/29 = 96.6% [83-99] | 21/29 = 72.4% [54-85] | 1.0 | 0/29 | 1/29 | 0 / 100 / 0 % | 0.445 | 1 |
| frozen native MLP (Baseline-0 49k, det.; stored per-task rows) | 2/29 = 6.9% [2-22] | 1/29 = 3.4% [1-17] | 9.5 | 2/29 | 25/29 | 0 / 90 / 10 % | 0.009 | 73 |
| RANDOM LEGAL (stored per-task rows) | 4/29 = 13.8% [5-31] | 0/29 = 0.0% [0-12] | - | 3/29 | 22/29 | 1 / 92 / 7 % | - | - |

(The native MLP and RANDOM rows come from the existing full-validation evaluations of those policies, restricted to the same 29 tasks; no extra native compute.)

### By stratum (successes / tasks, per seed)

| stratum | n | seed 11 | seed 12 | seed 13 |
|---|---|---|---|---|
| carpark: unique | 11 | 9/11 | 10/11 | 10/11 |
| carpark: ambiguous, resolvable | 5 | 5/5 | 5/5 | 5/5 |
| carpark: ambiguous, unresolvable | 13 | 12/13 | 12/13 | 13/13 |
| horizontal gap | 19 | 17/19 | 18/19 | 19/19 |
| vertical gap | 10 | 9/10 | 9/10 | 9/10 |
| gap early in path | 9 | 7/9 | 8/9 | 8/9 |
| gap late in path | 20 | 19/20 | 19/20 | 20/20 |

The synthetic S1 test numbers for the same checkpoints were 83.5% / 90.2% / 95.8% (first-action accuracy 37-45%). On native tasks first-action accuracy is higher (about 70%) and success needs a median of 1 decision: these native Gap-1 states are, if anything, easier for the CNN than the synthetic ones (fewer competing dead-end pairs). All policies finished with destructive failure 0 and used only ADD actions (no WAIT, no REMOVE).

## 4. Failure diagnosis

Failures: native validation tasks `gap1_validation_000019` (fails for all 3 seeds), `...072` (seed 11), `...115` (seeds 11, 12) - all timeouts after 20 ADD actions, no destructive failure. Re-run deterministically with full traces (`failure_traces.json`, seed-11 checkpoint):

| class | assessment |
|---|---|
| A. interface mismatch | **Excluded**: 0 encoding mismatches over live states (section 1); the correct action was legal natively in every pilot task (`a_star_legal_native_all`), and the same checkpoints solve 26-28 of 29 native tasks through the same interface. |
| B. observation ambiguity | Task 115 is carpark-ambiguous (4 touching road tiles, unresolvable); 019 and 072 are **unambiguous** (one touching road tile). Ambiguity does not explain the failures: on the 13 unresolvable tasks the seeds succeed 12/13, 12/13 and 13/13. In Gap-1 only the two facing dead ends matter, not which tile is the carpark. |
| C. distribution shift | Present and large in aggregate (16% water tiles, bridge resource, 151 vs 202 legal actions, REMOVE-on-carpark quirk, extra roads touching the destination in 29-40% of tasks (0% in the sandbox)), yet 90-97% transfer shows the CNN is robust to most of it. The three failures show no single separating feature in the data I have; candidates are the gap's position next to a house tile (019) or at the end of a long road touching water-adjacent tiles. This is **not tested**. |
| D. policy failure | **Supported for 019 and 072**: the correct edge is observable, legal and the task unambiguous, but the first-decision probability of the correct edge is 0.005-0.07 and its rank 6-18 (vs mean 0.41-0.52 and rank 1 on successes). The policy instead adds edges next to existing road segments (pictured below) for 20 decisions. |

Success/failure by first-decision probability of the correct edge: successes mean P = 0.41-0.52 (median rank 1); failures P = 0.005-0.08 (rank 4-18). The failures are wrong with moderate confidence (the chosen first edge had P about 0.2-0.4), not near-misses.

#### gap1_validation_000019 (unique, vertical gap, early), correct edge `*`, edges added by the policy `=`/`H`

```
y= 12 ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? 
                                      
y= 11 ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? 
                                      
y= 10 . . . . . . . . . . . . . . . t 
                                      
y=  9 . . . . . . . . . . . . . . . . 
                                      
y=  8 . . . . . . . . . . D D . . . . 
                                      
y=  7 . . . . . . . . + + D D . . . . 
                      H H             
y=  6 . . . . . . . . +=+-C C . . . . 
                      H |             
y=  5 . . . H . . . +=+=+=+=+=+ . . . 
            H         H | H           
y=  4 . . . +-+-+-+-+-+-+=+=+=+ . ~ . 
            *           H             
y=  3 . . . +=H . . . +=+=+ . . . ~ . 
                                      
y=  2 . t . . . . . . . . . . . . ~ . 
                                      
y=  1 ~ . . . . . . . . . . . . . ~ . 
                                      
y=  0 ~ ~ . . . . . . . . . . . . ~ . 
      x=-17..-2   legend: + road, = / H edge added by the policy, * missing correct edge (a*), ~ water, D destination, C carpark, H house
```
First decisions of the policy: [(('ADD', -8, 5, 'E'), 0.409), (('ADD', -9, 5, 'E'), 0.408), (('ADD', -7, 5, 'E'), 0.281), (('ADD', -8, 4, 'E'), 0.332)]; probability on the correct edge at decision 1: 0.0052.

#### gap1_validation_000072 (unique, horizontal gap, early), correct edge `*`, edges added by the policy `=`/`H`

```
y= 10 ? . . . . . . . . . . . . . 
                                  
y=  9 ? . . . . . . . . . . . . . 
                                  
y=  8 ? . . . . . . . . . . . . . 
                                  
y=  7 ? . . . . . . . . . . . . . 
                                  
y=  6 ? . . . +*+-+-+ . . . . . . 
              | H H |             
y=  5 ? . . . H +=+=+=+=+ . . . . 
                H H |             
y=  4 ? . . . H=+=+=+=+=+ . . . . 
                  H |             
y=  3 ? . . . +=+=+=+=+ . . . . . 
                    |             
y=  2 ? ~ . t . . . C D D . . . . 
                                  
y=  1 ? ~ ~ . . . . C D D . . . . 
                                  
y=  0 ? ~ ~ ~ . . . . . . . . . . 
                                  
y= -1 ? ~ ~ ~ ~ . . . . . . . . . 
                                  
y= -2 ? ~ ~ ~ ~ . . . . . . . . . 
      x=-19..-6   legend: + road, = / H edge added by the policy, * missing correct edge (a*), ~ water, D destination, C carpark, H house
```
First decisions of the policy: [(('ADD', -13, 5, 'E'), 0.203), (('ADD', -12, 5, 'E'), 0.457), (('ADD', -13, 4, 'E'), 0.285), (('ADD', -12, 4, 'E'), 0.531)]; probability on the correct edge at decision 1: 0.0745.

#### gap1_validation_000115 (unresolvable_by_leaf_rule, horizontal gap, late), correct edge `*`, edges added by the policy `=`/`H`

```
y=  9 . . . . . . . . . . . . . . . . 
                                      
y=  8 . . . . . . . . . . . . . . . . 
                                      
y=  7 . . . . . . . . . . . . . . . . 
                                      
y=  6 . . . . . . . . . . . . . . . . 
                                      
y=  5 . . . . . . . . + + +-H . . . . 
                      H H             
y=  4 . . . . . . . +=+=+=+=+ ~ . . . 
                    H H H             
y=  3 . . . +*+-+-+=+=+=+=+=H ~ . . . 
            |     | H H H             
y=  2 . . . C D D +-+-+-+-+-+ ~ . . . 
                    H                 
y=  1 . . . C D D . + . . . . ~ . . . 
                                      
y=  0 . . . . . . . . . . . . ~ . . . 
                                      
y= -1 ~ . . . . . . . . . . . ~ . . . 
                                      
y= -2 ~ . . . . . . . . . . . ~ . . . 
      x=-15..0   legend: + road, = / H edge added by the policy, * missing correct edge (a*), ~ water, D destination, C carpark, H house
```
First decisions of the policy: [(('ADD', -9, 3, 'E'), 0.33), (('ADD', -8, 3, 'S'), 0.154), (('ADD', -8, 3, 'E'), 0.324), (('ADD', -8, 4, 'S'), 0.333)]; probability on the correct edge at decision 1: 0.0743.

Do not read these three failures as a failure of the CNN architecture: 26-28 of 29 tasks succeed zero-shot.

## 5. Cost and native throughput

Interface capture: 120 validation resets with 12 action-test sequences + 240 test resets (summaries) on three headless instances, about 2.3 min per 120 tasks. Zero-shot pilot: 6 evaluation runs (3 seeds x 20 + 9 tasks), 250 native decisions in total, 42-60 s wall each including game launch, model load and 1.4 s per reset (resets, not steps, dominate: most episodes end after 1 decision). Native cost per episode here is dominated by the ~1.4 s city rebuild at reset (a native step is ~0.1-0.3 s), versus ~1 ms per synthetic step. Total S2 native wall time: roughly 15-20 minutes including captures, far inside any budget.

## 6. Recommendation

1. **Do not fine-tune yet.** Zero-shot already reaches 90-97% on the pilot with 0 interface errors; a fine-tuning comparison is only informative if a larger zero-shot evaluation shows a real gap.
2. **Next experiment (needs approval):** zero-shot the same three checkpoints on **all 120 native validation tasks** (about 10-15 minutes with three instances) to tighten the +-12-point intervals and measure the failure rate by stratum; then, if it holds, the frozen **native test set (240)** once, with the checkpoint choice fixed from synthetic validation.
3. **Diagnose the three failures** with a cheap, native-free check: build matching synthetic tasks (gap beside a house tile; gap at a long road end near water) and see whether the synthetic CNN also fails there - that separates distribution shift (C) from a genuine policy blind spot (D).
4. **Then Gap-2 zero-shot (native)** with the frozen S1 policies, as planned, because Gap-1 transfer is established; and run the synthetic H2/H3/H4 comparisons for the questions the sandbox is meant to answer.
5. **Representation fix is optional, not urgent:** the carpark channel (an encoder omission) does not hurt Gap-1; it is likely to matter for multi-edit tasks - test it synthetically first (compare with and without a carpark feature) before changing the native encoder.

## 7. Reproducibility and artifacts

| item | location |
|---|---|
| CNN seed 11 checkpoint evaluated | `runs\2026-10-08_140406_syn_s1_cnn_s11\checkpoints\step_050000.zip` sha256 `c8a7930eb12e9a204be02fa69e98e0dfc6969652555c5e390839bec225186fdc` |
| CNN seed 12 checkpoint evaluated | `runs\2026-10-08_140406_syn_s1_cnn_s12\checkpoints\step_050000.zip` sha256 `980f95f7ce28bfebebdb8c60dbd832c7dc12252c2320304d61fd87dcfa8c603c` |
| CNN seed 13 checkpoint evaluated | `runs\2026-10-08_140406_syn_s1_cnn_s13\checkpoints\final.zip` sha256 `41c1e085622b8f0d2e70274b2a6f05fedc94266124c1c74f9b39645ade417682` |
| per-episode zero-shot rows (29 tasks x 3 seeds) | `research/synthetic_to_native/zero_shot_rows_29tasks.json`; native evaluation run directories `runs/*s2zs_cnn_s*` and `runs/*s2zsext_cnn_s*` (config, metadata with DLL/game/source hashes, partial_rows.jsonl, results.json) |
| pilot manifests | `research/synthetic_to_native/manifest/s2_pilot.jsonl`, `manifest_ext/s2_pilot_ext.jsonl` |
| raw native states and masks | `native_states_validation_0of1.jsonl.gz`, `capture_*.jsonl` |
| reports | `interface_report.json`, `ambiguity_report.json`, `failure_traces.json` |
| code | `experiments/s2/{capture,interface_check,ambiguity,select_pilot,trace_failures,make_report}.py` |
