# S16 — Can the CNN Learn Colour-Aware Routing? (supervised diagnostic)

Scope kept: supervised only; no PPO, no native runs, no reward, no change to S7/S13 environments, encoders, production architecture or any checkpoint. Isolated code: `python/synthetic_rl/s16_data.py`, `s16_models.py`, `experiments/s16_train.py`. Artifacts: `research/s16_colour_routing/` — `{train,val,test}.{npz,json}`, `dataset_stats.json`, `results.json` (main run), `results_cd_only.json` (Outcome-C follow-up), `model_{A,B,C}[_cd].pt`.
Compute actually used: dataset generation 67 s; main training+evaluation 272 s; follow-up 227 s → **≈ 9.5 min** of the 15-minute cap (CPU, 4 threads).

## 1. Dataset
* Built on the unchanged S13 multi-house environment (`MultiHouseEnvV2`, v2 observation) in a water-free 19×11 land region of the S13 map. Each **geometry seed** places 2–3 destination blocks (native 2×2 destination + carpark column + stub, distinct colours, random native ids) and 2–3 houses (≥ 2 colours). All non-query houses are pre-connected by the privileged controller (their roads are visible and act as geometric decoys); the query house's route is partially built in 0–4 steps so the frontier is often away from the house. Concrete never binds.
* **Oracle label** = the set of legal plain ADD actions leaving the query house's current Active component (its frontier) that lie on some shortest concrete-cost path to a destination of the query house's colour (Dijkstra with ground-truth groups; house-edge swaps excluded because they are free for every target). Mean valid-set size 1.31–1.38 actions. The model only sees the v2 observation; the oracle's group knowledge never enters it (gate B5/B13 of S14 still hold: `identical_obs_with_conflicting_targets = 0`).
* **Decoy** = nearest destination of another colour. **Counterfactual twin** = identical geometry with the query house recoloured to the decoy colour. Automatic check per scene: v1 observation identical, v2 different, valid sets of the correct and decoy targets disjoint → *colour-dependent*; otherwise the scene is kept for geometry training but excluded from the colour-dependence evaluation (common case: both targets start in the same frontier direction).
* Splits by geometry seed (seeds 1601/1602/1603): train 900 geometries → 1,143 samples (243 colour-dependent pairs); val 150 → 187 (37 pairs); test 400 → 502 (102 pairs). Test: 150 within-RF / 352 beyond-RF samples (within = some valid edge within 3 tiles of both the house and a correct-colour destination tile), decoy nearer in 257. Query slot 2 (75 train samples) is **held out of training** → "unseen group permutation" test (40 test samples). All test checks pass: labels feasible and non-empty; CF target sets disjoint 102/102; v1 identical 102/102 and v2 different 102/102 across twins; 0 identical observations with conflicting targets.

## 2. Models (per-tile 4-logit head over the native tile-action layout; loss = −log Σ_valid softmax over legal ADD actions; Adam 1e-3, batch 64, seed 16, best-val checkpoint)

| model | trunk | theoretical RF | empirical RF (gradient) | params |
|---|---|---|---|---|
| A existing CNN | 3×3 convs (32,64,64), dilation 1 — the S8-C trunk | 7×7 | 7 | 61,764 |
| B larger RF | 3×3 convs (32,48,48,48), dilations 1,2,4,8 | 31×31 | 31 | 61,716 |
| C group-blind | = B, channels 15–20 zeroed at input | 31×31 | 31 | 61,716 |

Sanity: C receives no group information through any other feature (v1 carries only the has-group flag; twins are v1-identical, so C is forced to chance on them — observed: 0 % pair accuracy). Models cannot exploit id ordering: observations carry appearance-order slots, native ids never enter; the held-out slot measures permutation generalisation.

## 3. Results — main run (30 epochs each; A 73 s, B 94 s, C 101 s; all curves had plateaued: val accuracy flat for the last ~10 epochs)

| test subset (n) | A existing | B dilated | C blind |
|---|---|---|---|
| all (502) valid-set accuracy | 0.669 | 0.657 | **0.697** |
| within RF (150) | 0.693 | 0.680 | 0.673 |
| beyond RF (352) | 0.659 | 0.648 | 0.707 |
| decoy nearer (257) | 0.696 | 0.665 | 0.693 |
| **colour-dependent (204)** | **0.397** | **0.392** | **0.412** |
| originals of CF pairs (102) | 0.676 | 0.598 | 0.588 |
| CF twins (102) | 0.118 | 0.186 | 0.235 |
| **CF pairs both correct (102)** | **0.020** | **0.049** | **0.000** |
| decoy chosen on colour-dependent | 0.40 | 0.36 | 0.41 |
| unseen slot (40) | 0.150 | 0.125 | 0.350 |

Paired bootstrap (95 % CI of the accuracy difference on identical examples): B − A all −0.012 [−0.050, 0.026], colour-dependent −0.005 [−0.064, 0.059], beyond-RF −0.011 [−0.060, 0.037]; B − C all −0.040 [−0.070, −0.004], colour-dependent −0.020 [−0.074, 0.039]; A − C colour-dependent −0.015 [−0.078, 0.054].

Reading: every model learns the geometric prior (≈ 67–70 % overall, where the oracle's target coincides with "nearest destination" in about half the scenes) and **none learns the colour rule**: on colour-dependent scenes all three are at 39–41 % — indistinguishable from the group-blind control — and a counterfactual pair is solved in 0–5 % of cases. The models pick the original twin's answer and keep it after the recolouring (CF-twin accuracy 12–24 %, i.e. they keep choosing the now-wrong, geometrically favoured target).

## 4. Outcome-C follow-up: is the information there and can it be learned at all?
Information: present by construction (v2 twins differ exactly in the house's slot channels; targets are disjoint; no ambiguous observations). The question is learnability under this data budget. One extra configuration (same three models, same optimiser, 60 epochs, **training restricted to the 486 colour-dependent samples**, where geometry alone caps at ~50 %): `results_cd_only.json`.

| test subset | A existing | B dilated | C blind |
|---|---|---|---|
| colour-dependent (204) | 0.392 | 0.299 | 0.309 |
| CF pairs both correct (102) | **0.118** | 0.039 | 0.000 |
| all (502) | 0.538 | 0.333 | 0.408 |
| unseen slot (40) | 0.175 | 0.150 | 0.300 |
A − C on colour-dependent: +0.083 [0.005, 0.157]; A − B: +0.093 [0.015, 0.172]. With colour-only data the *existing* 7×7 CNN shows the first weak sign of colour use (12 % of counterfactual pairs solved vs 0 % blind), the dilated network does not, and nothing generalises to the held-out slot. These are small effects on 102 pairs after ~1 minute of training per model.

## 5. Failure examples (test, colour-dependent; identical for A/B/C in most cases)
* geom 3: query house (−2,−2), correct destination cost 3, decoy cost 1 — all three models build toward the decoy (`ADD −2,1,S`) instead of `ADD 0,0,E` / `ADD 0,1,S`.
* geom 22: costs equal (4 vs 4); the original twin is solved by none (A/B choose unrelated frontier edges), the recoloured twin is answered with the original's target by C.
Further examples in `results*.json → models.*.failure_examples`.

## 6. Interpretation against the decision framework
* Not Outcome A (existing CNN performs well): A is at chance on colour-dependent decisions in the main run.
* Not Outcome B (larger RF substantially better): B ≤ A on every subset including beyond-RF (CI excludes any gain > +0.04), and B is worse in the colour-only follow-up. **The receptive field is not the bottleneck at this stage.**
* **Outcome C/D**: both group-aware models perform poorly and the group-blind control performs the same. The counterfactual results show this is not a dataset flaw (information present, targets change, ambiguity 0) but a **learning failure under small data (486 colour-dependent training samples) and a strong geometric prior**: the colour signal (two one-hot channels on a handful of tiles) is drowned out by nearest-destination regularities that already explain ~67 % of labels. The slot-generalisation results (15–35 % on the held-out slot) add that whatever is learned about slots is slot-specific, not the abstract "house slot = destination slot" rule — a one-hot slot encoding does not give a plain CNN that equivariance for free.
* Caveats: one seed per model; 102 test pairs (CI half-width ≈ ±0.09); short fixed budgets; the diagnostic is a next-edge task, not full routing.

## 7. Decision and recommendation for S17
**Answer:** with the current observation encoding and ~1k supervised examples, the existing CNN does not learn which destination a house belongs to — and enlarging the receptive field does not help. We do **not** need a larger spatial receptive field before continuing RL; we need the colour-matching rule to be learnable/expressible first.

**One recommendation (S17, needs approval):** make the house↔destination match *explicit and equivariant* in the observation rather than hoping PPO discovers it from six one-hot slot channels: a single derived, non-privileged per-tile channel "this destination tile has the same colour as the currently selected/nearest unconnected house" cannot work for a stateless policy over many houses, so the cleaner option is a slot-equivariant input transform — e.g. compute per-tile features that are symmetric across slots (for each tile: *does a same-slot destination exist, direction/offset summaries to the nearest same-slot destination* — derived from visible information only, exactly as a player reads colours) — and re-run this S16 diagnostic (same data, same budget) before any PPO. If that diagnostic passes (CF-pair accuracy ≫ 0, held-out slot ≈ seen slots), transfer S8-C by the S14 zero-init widening and resume the S15 recipe question. If it fails too, the problem is the supervised learnability of this task family, and more RL compute is not the answer.

**STOPPED. No S17 work started.**
