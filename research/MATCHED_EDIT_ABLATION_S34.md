# S34 — Matched-rate random EDIT control (diagnostic, environment `postfix_v8`)

Question: does the hybrid's advantage over planner B come from *learned EDIT selection*, from *learned PLAN/EDIT timing*, or merely from *stochastic perturbation* of planner B?
Code/data: `research/s34_matched_edit_ablation/{code/ablate.py,code/analyze_ablation.py,eval/,results.json}`; condition A = the replication's fresh-benchmark rollouts (`research/s34_replication/eval_fresh`). No training was run. Classified as a **diagnostic comparison** (the 8 fresh seeds had been inspected in the replication).

## 0. Summary
* **A vs B (primary)**: the trained policy with its chosen edits (A) beats the same policy with every chosen edit replaced by a uniformly random legal edit (B) by **+858 / +585 / +521** deliveries over 8 seeds (S31 / rep1 / rep2 100k), 8/8 seeds in every run, CIs far from zero. But B is not "same timing, random choice": once its edits are randomised the closed-loop policy *edits far more* (EDIT share 0.70–0.80 vs 0.37–0.70 in A), takes PLAN only 17–26 % of the time, and dies at ~140 decisions with 1–4 deliveries. So A ≫ B measures the joint effect of learned selection *and* the timing feedback it triggers; it cannot be read as "selection alone".
* **C vs planner B**: planner B plus random legal edits/waits at each run's matched rate is **≈ planner B** at the S31/rep1 rates (+20, −5 over 567; 4/4 seeds each way) and *worse* at rep2's 70 % rate (−173). Random perturbation of planner B does not reproduce the hybrid's gains.
* **A vs C** (learned policy vs random edits at the same rate on planner B): **+298 / +119 / +153**, positive in all three runs (CI excludes zero for S31 and rep2, not rep1). Learned edits are better than random edits at the same frequency — but, from the replication, that only translates into a gain over *planner B itself* for one of three runs.
* **Destructiveness**: destroyed functional connections occur only on ADD decisions. Per ADD: learned (A′) 0.13 / 0.18 / 0.22, random-on-planner-B (C) 0.23 / 0.23 / 0.17 — learned ADDs are not systematically less destructive than random ones when a network exists to destroy; B's low rate (0.02–0.07) reflects that B never builds a network.
* **Noise null**: re-running A with the same torch seeds (A′) differs from A by −33 / +3 / −13 total (CI within ±13 per seed) — an order of magnitude below every contrast above.
* **Recommendation: option 4 — stop investing in this hybrid architecture**, with the single preservable finding that learned edit selection is better than random edit selection (A > C in 3/3 runs); see §6.

## 1. Design (fixed before running)
* Checkpoints: the preregistered 100k of S31, rep1, rep2; S31 50k as a labelled secondary analysis (A/B only).
* Seeds / horizon / rollouts: the 8 fresh seeds (900000 + 7919·i), 1,500-decision horizon, 3 stochastic rollouts per seed; identical across conditions.
* RNG: policy sampling uses the replication's torch seed per (seed, rollout) (common random numbers until divergence — which occurs immediately in B/C by construction, and even in A′ on some seeds because of the game's rare state transient); replacements and kind draws use a separate `random.Random(770000 + 1000·r + i)`.
* **A** original hybrid (replication rollouts, kinds trace only). **A′** instrumented re-run of A (per-decision kind, ADD/REMOVE, destroyed delta, boundary status).
* **B** randomized EDIT: policy sampled normally; whenever the sampled action is ADD/REMOVE it is replaced by a uniform draw over the *other* legal EDIT indices of the current native mask (kept if none; never happened — 100 % replaced). PLAN, WAIT and forced upgrade choices untouched.
* **C** planner B + random EDIT: per decision the kind is drawn from the matched run's observed A shares (preregistered in the script: S31 0.580/0.382/0.031, rep1 0.503/0.473/0.018, rep2 0.254/0.696/0.044 PLAN/EDIT/WAIT; S31's from its first two rollouts, the third completed later and moved the shares by < 0.01); EDIT = uniform legal EDIT, PLAN = planner B, WAIT = 0; forced upgrades as in the hybrid env.
* Realised shares in C matched the targets to ±0.01 (table below). Checks per episode: 0 illegal actions, 0 boundary-crossing actions, Σ rewards = score + penalty — all 240 episodes passed.
* Budget: estimate 240 episodes ≈ 108k decisions ≈ 2.1 instance-hours; actual 240 episodes, ≈ 1.9 h wall on 3–4 instances. Secondary S31-50k B included (24 episodes).

## 2. Results — totals (Σ over 8 seeds of the per-seed mean over 3 rollouts; planner B = 567)

| run | cond | deliveries | len med | survived | destroyed /ep | PLAN/EDIT/WAIT | ADD share of edits | destroyed per ADD | rollout totals | within-seed SD |
|---|---|---|---|---|---|---|---|---|---|---|
| S31 100k | A | 885 | 494 | 0 | 22.1 | 0.59/0.37/0.04 | — | — | 741 / 796 / 1119 | 50 |
| | A′ | 918 | 510 | 0 | 18.8 | 0.60/0.36/0.04 | 0.79 | 0.131 | 789 / 1135 / 830 | 44 |
| | **B** | **27** | 140 | 0 | 2.1 | 0.26/**0.70**/0.04 | 0.74 | 0.026 | 36 / 32 / 13 | 5 |
| | C | 587 | 397 | 0 | 27.1 | 0.59/0.38/0.03 | 0.77 | 0.231 | 625 / 530 / 607 | 42 |
| rep1 100k | A | 681 | 436 | 0 | 32.4 | 0.50/0.47/0.02 | — | — | 729 / 529 / 785 | 35 |
| | A′ | 678 | 436 | 0 | 31.8 | 0.50/0.48/0.02 | 0.85 | 0.182 | 727 / 489 / 817 | 33 |
| | **B** | **96** | 185 | 0 | 7.8 | 0.25/**0.73**/0.02 | 0.74 | 0.071 | 70 / 82 / 136 | 11 |
| | C | 562 | 396 | 0 | 33.8 | 0.50/0.48/0.02 | 0.77 | 0.230 | 606 / 506 / 573 | 30 |
| rep2 100k | A | 547 | 367 | 0 | 47.2 | 0.25/0.70/0.04 | — | — | 568 / 581 / 492 | 31 |
| | A′ | 560 | 372 | 0 | 55.3 | 0.25/0.71/0.04 | 0.89 | 0.221 | 637 / 521 / 522 | 31 |
| | **B** | **26** | 140 | 0 | 1.6 | 0.17/**0.80**/0.03 | 0.73 | 0.017 | 14 / 51 / 12 | 3 |
| | C | 394 | 332 | 0 | 30.0 | 0.25/0.70/0.05 | 0.75 | 0.168 | 372 / 475 / 335 | 23 |
| S31 50k (secondary) | A | 976 | 506 | 0 | 20.2 | 0.62/0.37/0.00 | — | — | 961 / 1065 / 903 | 31 |
| | B | 83 | 144 | 0 | 4.2 | 0.40/0.59/0.01 | 0.76 | 0.050 | 48 / 114 / 88 | 8 |

Illegal actions 0, boundary violations 0 in all instrumented episodes (A′, B, C: 168 episodes, 70,000+ decisions). Policy-chosen edits before replacement in B were 74–75 % ADD; the replacements are drawn from the legal set, in which ADD/REMOVE proportions are whatever the mask offers (≈ 73–76 % ADD realised).

## 3. Paired contrasts (per-seed means; bootstrap 95 % CI on the mean per-seed difference; H/E/L over 8 seeds)

| contrast | S31 100k | rep1 100k | rep2 100k |
|---|---|---|---|
| A − planner B | +318, 6/0/2, [+4, +81] | +114, 6/0/2, [−18, +39] | −20, 4/0/4, [−27, +19] |
| B − planner B | −540, 0/0/8, [−94, −43] | −471, 0/0/8, [−86, −33] | −541, 0/0/8, [−95, −44] |
| **A − B (primary)** | **+858, 8/0/0, [+89, +130]** | **+585, 8/0/0, [+55, +91]** | **+521, 8/0/0, [+53, +79]** |
| A − A′ (noise null) | −33, 3/0/5, [−13, +3] | +3, 5/2/1, [−5, +4] | −13, 3/2/3, [−13, +8] |
| C − planner B | +20, 4/0/4, [−18, +24] | −5, 4/0/4, [−20, +18] | −173, 4/0/4, [−55, +8] |
| **A − C** | **+298, 7/0/1, [+14, +70]** | +119, 6/0/2, [−8, +35] | **+153, 7/0/1, [+1, +40]** |
| B − C | −560, 0/0/8 | −466, 0/0/8 | −368, 0/0/8 |
| S31 50k: A − B (secondary) | +893, 8/0/0, [+93, +135] | | |

Variation: across training runs (n = 3) A − B sums {858, 585, 521}, SD 179, all positive; A − C {298, 119, 153}, SD 95, all positive; C − planner B {+20, −5, −173}, SD 105, mixed; A − planner B {+318, +114, −20}, SD 170, mixed. Within a run, rollout-to-rollout SD per seed is 23–50 for A/A′/C and 3–11 for B (B is uniformly dead). Per-seed: A − B is positive on all 24 (run, seed) pairs; A − C on 20 of 24.

## 4. Why B collapses (what the data show, no mechanics audit)
B is not a "same decisions, random edits" counterfactual, because the policy is closed-loop: after its edits are randomised, the state it sees is not the one its edits were meant to produce, and it responds by editing again — EDIT share rises from 0.37–0.70 (A) to 0.70–0.80 (B) and PLAN share falls to 0.17–0.26, so planner B is called too rarely to connect the first houses; every B episode ends at the first unserviceable destination (~140 decisions, ≤ 4 deliveries). The 2,600–3,500 random edits per 24 episodes mostly land as isolated ADDs (74 % ADD) that never form a network — hence B's low destroyed counts (there is nothing to destroy). C, which keeps the planner's share fixed at the matched rate, does *not* collapse at the S31/rep1 rates, confirming that the collapse is the timing feedback rather than the random edits per se; at rep2's 70 % edit rate even C loses 173 deliveries.

## 5. Interpretation against the preregistered outcomes
* **A > B** holds overwhelmingly, but it is evidence that learned selection and learned timing are *entangled*, not that selection alone carries the value: B changes both. The outcome "A ≈ B → timing" cannot occur in this design because randomising selection destabilises timing.
* **C ≈ A** does **not** hold: random edits at the hybrid's own rate leave planner B unchanged (S31/rep1 rates) or hurt it (rep2 rate). The neural policy is doing more than perturbing planner B — its edits are better than random at the same frequency (A − C > 0 in 3/3 runs, 20/24 seeds).
* But the replication already showed that this "better than random" quality yields a gain over *planner B itself* in only one run of three (A − planner B: +318 / +114 / −20). The learned edits are "less bad than random" everywhere and "better than not editing" only sometimes.
* Destructiveness: with a network present (C vs A′), random ADDs destroy connections at 0.17–0.23 per ADD and learned ADDs at 0.13–0.22; no systematic protective effect of learned selection. Learned selection's advantage over random is therefore not "it avoids breaking things" — the remaining candidates are where/when it builds, which this design does not separate further.
* Effect sizes are large relative to noise (A − A′ null ≈ ±13 per seed; A − B ≈ +65 to +107 per seed; A − C ≈ +15 to +37 per seed), so the non-robustness of A − planner B in the replication is not a power problem; it is a between-training-run problem.

## 6. Recommendation
Among the four options:
1. *Continue improving learned EDIT selection* — the only signal that is positive in all three runs (A > C) supports this direction in principle, but its size (+15 to +37 deliveries per seed over random edits) is smaller than the between-run spread of the hybrid itself (SD 170 on 8 seeds), and it rests on an architecture that, in two of three runs, does not beat the planner it is built on.
2. *Focus on learning PLAN/EDIT timing instead* — not supported: timing is what breaks when selection is perturbed, and fixed-rate random editing (C) adds nothing to planner B.
3. *Replace the neural component with a simpler stochastic heuristic* — not supported: C ≈ planner B at moderate rates and < planner B at high rates; random perturbation is not where the hybrid's (occasional) gain comes from.
4. **Stop investing in this hybrid architecture — recommended.** The replication shows the hybrid-over-planner-B effect is a property of one training seed; this ablation shows the mechanism behind that effect is an entangled, state-dependent edit policy whose edits are better than random but not reliably better than nothing, that is as destructive per action as random editing, and that collapses when perturbed. Nothing here changes the throughput regime (deliveries/decision pinned, no colour separation, no survival past ~650 decisions on fresh seeds — S21/S28/S29). If any further budget is spent in this direction it should go to option 1 *only* as a narrowly scoped question — "can edit selection be made reliably better than no editing across training seeds?" — and not to further training of this policy.

No further training was launched.
