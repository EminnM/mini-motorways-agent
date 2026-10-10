# S3-D Part A/B — Gap-2 construction efficiency (re-scored from the recorded S3-A rollouts)

No new simulation, no training. Source: `research/s3_gap2/s3a_zero_shot_rows.json` (per-episode rows for the three frozen S1 CNNs on Gap-2 TEST, n=200; deterministic ×1, stochastic ×3 eval seeds).
Script: `experiments/s3_efficiency.py` → `research/s3_gap2/s3_efficiency.json` (all numbers below). Gap-2 needs exactly 2 correct ADDs.

**What is exactly reconstructable.** Each row stores the episode's total ADD count, irrelevant-ADD count, WAIT/REMOVE counts, success, decisions, and whether the action right after the first correct edit was the second correct edge. An episode terminates the instant the connection completes, so "success with ≤k ADDs" is exact (a success's `n_add` is its full ADD count). Failures stay in the denominator of every budgeted rate. ADD count ≠ decision count: decisions also include WAIT/REMOVE; both are reported.
**Not reconstructable** (actions were not stored per step): per-step action sequences, which irrelevant edges were chosen, or ranks beyond the two recorded points. WAIT and RANDOM have only aggregate summaries recorded (RANDOM 1.5% completion over 3 seeds, WAIT 0%), not per-episode ADD counts, so budgeted RANDOM rates are bounded above by 1.5%, not measured. Oracle is exactly 2 ADDs / 2 decisions on 200/200 by construction.

## 1. Budgeted completion (failures counted as failures; deterministic policy, n=200)
| Policy | any (≤20 decisions) | exactly 2 ADDs | ≤3 ADDs | ≤5 ADDs | ≤10 ADDs |
|---|---|---|---|---|---|
| WAIT | 0% | 0 | 0 | 0 | 0 |
| RANDOM LEGAL | 1.5% | ≤1.5% | ≤1.5% | ≤1.5% | ≤1.5% |
| CNN s11 | 62.5% | 11.5% | 19.0% | 32.0% | 51.5% |
| CNN s12 | 75.5% | 12.5% | 21.5% | 41.0% | 66.0% |
| CNN s13 | 88.5% | **26.0%** | **36.5%** | **52.0%** | 78.5% |
| Oracle | 100% | 100% | 100% | 100% | 100% |

Stochastic (mean of 3 eval seeds): completion 68.7 / 77.7 / 85.0%; exactly 2 ADDs 6.0 / 8.0 / 10.2%; ≤3 ADDs 13.3 / 14.0 / 22.3%; ≤5 ADDs 26.7 / 28.7 / 41.7% (s11 / s12 / s13). Per-seed values are in the JSON.
Fraction of the 20-decision completion that survives a 3-ADD budget: 30% / 28% / 41% (s11 / s12 / s13, deterministic).

## 2. Efficiency among successes and over all episodes (deterministic; stochastic mean in brackets)
| | s11 | s12 | s13 |
|---|---|---|---|
| Irrelevant ADDs per successful episode (mean / median) | 4.8 / 4 (5.9) | 4.5 / 4 (6.1) | 3.9 / 3 (4.9) |
| Total ADDs per completion (mean = irrelevant + 2) | 6.8 | 6.5 | 5.9 |
| Mean total decisions per success | see JSON | see JSON | see JSON |
| Efficiency (2 / total ADDs), successes only | 0.47 (0.38) | 0.47 (0.38) | 0.55 (0.44) |
| Efficiency over all episodes, failure = 0 | 0.29 (0.26) | 0.36 (0.29) | 0.49 (0.37) |
| Successes at efficiency 1.0 / 0.5–0.99 / 0.25–0.5 / ≤0.25 | 23 / 30 / 38 / 34 (of 125) | 25 / 39 / 56 / 31 (of 151) | 52 / 41 / 45 / 39 (of 177) |
| Correct 2nd edge immediately after a correct 1st edit | 21.1% (n=180) | 21.9% (n=183) | 38.8% (n=183) |
| First action is a solution edge | 51.5% | 60.0% | 62.5% |

Sampling hurts efficiency without hurting completion: stochastic first-action accuracy is only 32–40% (s11), 34–35% (s12), 33–45% (s13), yet completion is as high or higher, i.e. extra retries make up for it.
Chance for the second-edge question is roughly 1 / (≈150–200 legal ADDs) ≈ 0.5%, so 21–39% is far above chance — the policy does place real mass on the missing edge, just not decisively.

## 3. Paired differences (same 200 test tasks, deterministic; bootstrap 95% CI over tasks)
| Comparison | any ≤20 | exactly 2 ADDs | ≤3 ADDs | ≤5 ADDs |
|---|---|---|---|---|
| s13 − s12 | +13.0 [7, 19] | +13.5 [7.5, 19.5] | +15.0 [8, 22] | +11.0 [3.5, 18.5] |
| s13 − s11 | +26.0 [19.5, 33] | +14.5 [8.5, 20.5] | +17.5 [10.5, 24.5] | +20.0 [12, 28] |
| s12 − s11 | +13.0 [6.5, 19.5] | +1.0 [−4.5, 6] | +2.5 [−4, 9] | +9.0 [1, 17] |

Only 15 of 200 tasks are solved within ≤3 ADDs by all three checkpoints; 91 by at least one. Three checkpoints share one task set, so these are paired comparisons of three training seeds, not a population estimate over seeds.

## 4. Interpretation (Part B)
* **Efficient or repeated attempts?** Primarily repeated attempts. Even the best checkpoint completes within the minimum 2 ADDs on only 26% of tasks (12–13% for the others) and needs ≥5 ADDs on about half of its successes; typical successes carry 3–5 irrelevant roads. RANDOM at 1.5% shows the task is not trivially solvable, so the policy is doing something real (its candidate edges are well above chance), but the data are consistent with "fill nearby gaps until the path connects" (H4a), not with a policy that identifies the exact two-edge completion up front. No multi-step planning claim is supported.
* **Does better completion go with better efficiency?** Partly. s13 is best on every measure. But s12 vs s11 shows completion can improve (+13 points) with no efficiency gain at 2/3 ADDs (+1 and +2.5, CIs span 0); its gain appears only at larger budgets (≥5 ADDs, +9). Three checkpoints are too few for a correlation estimate; the claim is only that completion alone overstates efficiency.
* **Best checkpoint = best first-action accuracy?** Nominally yes (62.5% vs 60.0% vs 51.5%), but s13 vs s12 is within noise (±7 points) and s12's first-action gain over s11 (+8.5) did not produce efficiency gains. First-action accuracy is not sufficient to explain efficiency; what separates s13 is the second action after a correct first edit (39% vs ~22%).
* **Budget reduction 20 decisions → 5 / 3 / 2 ADDs:** deterministic s13 keeps 59% (52/88.5), 41% and 29% of its completion; s11/s12 keep 51/54%, 30/28% and 18/17%. The larger the strictness, the larger the checkpoint spread.
* **Caveats:** one task set (n=200, CI ≈ ±7 points), three S1 training seeds, wrong ADDs are free in this environment (no road cost), so nothing in training ever penalised irrelevant edits; the efficiency gap is therefore expected, not a failure of the training signal. Efficiency was not an optimisation target.

## 5. Implication for the native step
The transferable asset is a *local connectivity prior* (find the gap, add the edge), not a verified multi-step planner. In the real game, road is a limited resource and wrong edges have consequences (churn, resource cost, traffic), so the 5–9 irrelevant ADDs seen here are the main risk to monitor — see `research/NATIVE_CNN_PILOT_PLAN.md` (action-efficiency and road-churn metrics).
