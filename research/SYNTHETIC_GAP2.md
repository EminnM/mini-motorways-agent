# Synthetic Gap-2 (Phase S3) — interim: generator validation + S3-A zero-shot

Status: **S3-A done; stopped before any training** (zero-shot is high — brief says stop and ask).
Raw results: `research/s3_gap2/` (`gap2_generator_validation.json`, `s3a_zero_shot.json`, `s3a_zero_shot_rows.json`).
Code: `python/synthetic_rl/tests_gap2.py`, `python/synthetic_rl/gap2_eval.py`, `experiments/s3_zero_shot.py`. No existing S1/S2/native artifact modified; generator and manifests unchanged.

## 1. Generator validation (7/7 pass; run on all 1,300 train/val/test tasks)
| Check | Result |
|---|---|
| Exactly two required ADDs; no single action completes | 1300/1300 tasks have 2 solution edges; brute force over every legal ADD from the start state (259,039 actions): **0** complete |
| Both orders legal; first useful edit reward 0 and non-terminal; completion reward exactly +1 | 0 violations (1300 tasks × 2 orders) |
| Oracle / observation-only solver (sees only obs vector + mask) | 300/300 and 300/300 (val+test); oracle needs exactly 2 actions |
| No label leakage | observation unchanged when stored solution is overwritten |
| RANDOM legal vs Gap-1 | Gap-1 9.8% ± 1.8, Gap-2 **2.0% ± 1.0** (5 seeds, TEST); WAIT 0% |
| Splits disjoint | no layout overlap (1000/100/200) |
| Diversity | TEST orientation pairs HV 83, HH 71, VV 46; 37/200 have adjacent missing edges; 182 distinct first-edge indices |

## 2. S3-A zero-shot (frozen S1 CNNs, Gap-2 TEST n=200, 20-decision window, no gradient updates)
Baselines: WAIT 0%, RANDOM LEGAL 1.5% (seeds 1.5/0.5/2.5), oracle 100%.

| | cnn_s11 | cnn_s12 | cnn_s13 |
|---|---|---|---|
| SHA256 prefix | c8a7930e | 980f95f7 | 41c1e085 |
| Full completion, deterministic | 62.5% [56,69] | 75.5% [69,81] | 88.5% [83,92] |
| Full completion, stochastic (3 seeds) | 68.7% | 77.7% | 85.0% |
| First action is a solution edge | 51.5% | 60.0% | 62.5% |
| ≥1 correct edge added | 90.5% | 94.0% | 94.0% |
| Partial only (1 edge, never completed) | 28.5% | 22.5% | 9.5% |
| Destructive failure | 0% | 3.5% | 0% |
| Median decisions to completion | 5 | 5 | 4 |
| Success with 0 irrelevant ADDs | 11.5% | 12.5% | 26.0% |
| Success with ≤2 irrelevant ADDs | 25.0% | 29.5% | 42.5% |
| Mean irrelevant ADDs / episode | 9.2 | 7.5 | 5.3 |
| Second action correct right after a correct first edit | 21% | 22% | 39% |
| Median rank of remaining edge: decision 1 → after first edit | 1 → 5 | 1 → 5 | 1 → 2 |
| Action mix (WAIT/ADD/REMOVE) | 9/91/0 | 0/95/5 | 6/94/0 |

By orientation pair (det. completion): HH 80/87/96%, HV 55/64/81%, VV 48/78/91%.

## 3. Interpretation (H4a vs H4b/H4c) — preliminary
Zero-shot Gap-2 is far above RANDOM/WAIT, but the evidence points to **H4a: repeated local gap filling plus trial-and-error, not sequence planning**:
- Wrong ADDs are free inside the window, and policies add ~5–9 irrelevant edges per episode; only 12–26% of successes are clean (no irrelevant edits).
- After a correct first edit the correct second edit is chosen immediately only 21–39% of the time (rank rises from 1 to 2–5): the policy does not "know" the first edit made the second one decisive; it keeps sampling candidates until one connects.
- Completion given a useful first action (76–93%) is much higher than given a wrong first action, consistent with attempts at each gap in turn.
This does not demonstrate planning, reasoning or connectivity understanding. It also does not show Gap-2 is already solved in a meaningful sense: a cleaner efficiency metric (≤2 irrelevant edits) is only 25–43%.

## 4. Not run (per brief)
S3-B (from scratch) and S3-C (curriculum) were not started. Open question for the user: is further training scientifically useful given strong zero-shot? Options: (a) stop; (b) S3-B anyway as an H4b control; (c) tighten the metric first (e.g. a window-limited or irrelevant-edit-penalised evaluation, no reward change) to see whether zero-shot survives a stricter "efficient completion" test.

Throughput benchmark of CNN training not yet run (no training performed).
