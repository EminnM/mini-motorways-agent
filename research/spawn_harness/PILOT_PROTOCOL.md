# Spawn-steering pilot — protocol (PREPARED, NOT EXECUTED; needs review of the controls)

**Question.** Does occupying tiles with cheap roads change where the next buildings spawn (geometry), and does that change long-run score beyond a planner that builds transport roads only?
**Verified mechanics it rests on** (S21 audit): a tile with any road state is not buildable for spawns (`Tile.cs:1094–1096, 1174–1199` via `IsTileBuildable`); placement = random-scaled argmax over feasible candidates with one PRNG draw per candidate (`BuildingPlacer.cs:580, 734, 818–835`); no road terms in the weights.

## Design (per anchor; all branches restored from ONE snapshot)
| branch | edits at t0 | purpose |
|---|---|---|
| base | none (advance 0.3 s) | reference spawn sequence, PRNG trace, score |
| base2 | none | restore-fidelity check: must equal base exactly (spawns, PRNG at every checkpoint, score) or the anchor is discarded |
| block | 3-edge disconnected road segment 2–5 tiles (Manhattan) from the nearest building, not 4-adjacent to any road/building/carpark | occupies likely spawn candidates |
| sham | 3-edge segment as far from buildings as the legal window allows | same concrete cost and road count, intended not to touch spawn candidates; **validity decided post hoc**: PRNG state equal to base right after the edits and at every checkpoint up to the first spawn difference → RNG-neutral; otherwise the anchor's sham is "confounded" and only the block-vs-base contrast with the confound flag is reported |
Continuation after the edits: WAIT for `horizon` = 120 decisions (checkpoints 30/60/90/120), i.e. the same continuation the stopped pilot used, so spawn displacement is measured without a policy in the loop. A second phase (only if phase 1 shows displacement) replays base/block with planner B continuation to measure score — that needs the DIAG_FORK op to accept a policy or Python-driven branches (open design point).

Main line to the anchor: planner B via normal STEP calls on `NORMAL_START` (so anchors are realistic cities), anchors at decisions 30 and 90; seeds = 10 fresh seeds (not the 12 evaluation seeds, not the S13 test seeds): 600000 + 7919·i, i = 0..9.

## Measurements (harness.py)
* Restore equivalence (base vs base2): spawns identical, PRNG identical at all checkpoints, score identical.
* RNG confounding per intervention branch: `rng_confounded()` — PRNG differs after the edits or at a checkpoint at/before the first spawn difference ⇒ confounded.
* Spawn outcome: index-matched Manhattan displacement of same-type spawns, set overlap / only-base / only-other counts, first-spawn-difference time, houses/dests at checkpoints.
* Cost: concrete at restore − after edits (block and sham must match), edits accepted (must be 3/3 or the branch is discarded).
* Score at checkpoints and terminal step (secondary in phase 1 — WAIT continuation makes score mostly reflect the pre-anchor network).

## Pre-specified analysis and success criterion
Primary: among anchors with (i) equivalent base/base2 and (ii) a non-confounded sham, compare block vs sham on "first spawn displaced by ≥ 2 tiles or spawn set changed within 90 decisions". Effect claimed only if block shows displacement in ≥ 70 % of such anchors AND sham in ≤ 30 %, with ≥ 12 valid anchors. Secondary: block-vs-base displacement with the confound flag reported separately; PRNG-shift frequency for block and sham.
Stop rules: discard the pilot if base/base2 non-equivalence occurs in > 2/20 anchors (restore fidelity) or if fewer than 8 anchors yield a non-confounded sham (the control is not achievable with far segments → need a candidate-aware sham, i.e. a plugin-side exposure of the spawn-candidate set, which would be a new op).

## Prerequisites
1. Apply and build the DIAG_FORK plugin op (`plugin_patch/DIAG_FORK.cs`) — blocked tonight by the shared-resource permission; deployment needs a backup of `<GameDir>/Bootstrap/MiniMotorwaysBootstrap.dll`.
2. Smoke: one anchor, branches base/base2 only → equivalence must hold; then one anchor with block/sham.
3. Budget: 20 anchors × 4 branches × 120 decisions ≈ 20 × 4 × ~8 s ≈ 11 min native wall-clock plus main lines (~20 × 30 s) ≈ 20 min total.

## Confounders and how they are handled
RNG-stream shift (post-hoc PRNG equality test, sham discarded if shifted); concrete opportunity cost (identical cost in block and sham; WAIT continuation removes any planner interaction in phase 1); Expert-rule road demolition (Normal mode assumed — to be confirmed from the RESET response's game mode before the pilot); restore residual divergences (base2 check per anchor); window-reach limits for far shams (reported as the "sham_is_far" rate; if low, the far sham is not a valid control and the pilot must wait for a candidate-aware sham).
