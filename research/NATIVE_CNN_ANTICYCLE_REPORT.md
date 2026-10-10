# S6 — Native CNN anti-cycle diagnostic (inference only)

**Result: H2 — cycle suppression, as implemented, is insufficient.** The filter broke the strict two-state oscillation (about −60% oscillation steps), but the policy settled into longer cycles over the same few road states; **no useful connection, no delivery, no survival change** in either paired seed. Two seeds is a mechanistic probe, not a performance estimate.
No PPO, optimizer step, reward/observation/mask/game change, checkpoint change, or edit to S1–S5 results (S5 regression suite 9/9 and Gate-0 offline checks re-run and still pass). **Native wall-clock: ≈ 1.7 min** (paired evaluation 48.8 s + two short geometry reads, cap 15 min).

## 1. Terminal-penalty logging (measurement only)
* `experiments/diagnostics.py`: per episode `diag_terminal_penalty_missing` (= native game-over, not truncation, with `terminalPenalty == 0`), `diag_terminal_on_entry` (= final STEP started already terminal: `applyInfo == "terminal state (not applied)"`, `accepted=false`), and `diag_final_status` (reward, terminated/truncated, accepted, appliedAction, applyInfo, reward components, decisionIndex, sim time before/after and its advance, wall interval since the previous step). Nothing is returned to or injected into the environment or reward.
* `python/mini_motorways_env.py`: one additive, read-only key `info["applyInfo"]` (already in the native response). Reward, termination and stepping untouched. Pre-change copies: `research/s5_frozen_copies/`.
* Tests (`experiments/s6_offline_tests.py`, replaying the S5 traces): 2 zero-penalty and 11 normal terminal traces flagged exactly as expected (missing ⇔ terminated & penalty 0; on-entry ⇔ `terminal state (not applied)`; final-step clock advance ≈ 0.1 s vs 1.0 s); return, length, native score and action counts identical with diagnostics on vs off; `diag_errors` 0; a horizon-truncated episode is not flagged. In the native paired run all four episodes ended normally (flag false, on-entry false, final advance 1.0 s).

## 2. Evaluation-only anti-cycle filter (`experiments/anticycle.py`)
* **State:** road-network signature = hash of every (tile, E/S side, road state) from the **raw native observation** — privileged, evaluation-only; it never reaches the policy. The filter also remembers observed (state, action) → next-state transitions within the episode.
* **Cycle criterion (road state, not action index):** with states s₀…s_t, detected ⇔ t ≥ 3 and s_t = s_{t−2} and s_{t−1} = s_{t−3} and s_t ≠ s_{t−1} (the roads really went A,B,A,B). Repeated WAIT (A,A), or a single revisit (A,B,A), never triggers; new construction always yields new states.
* **Intervention:** only if detected **and** the policy's top-ranked legal action is a known cycle-continuing action (observed transition (s_t, a) → s_{t−1}). That action is excluded for this decision only; walk the policy's own logit ranking, reject candidates whose observed transition lands in a state already recognised as part of a cycle (recorded with reason), take the first remaining legal action. If none, keep the original and record `no_alternative`. Candidates with unknown effect are accepted. Ties: lower index (same as argmax).
* **Unchanged:** native mask, network, weights, reward, observation, game. The filter chooses only among actions the native mask already allows. Period ≥ 3 cycles are not intervened on (only counted).
* Every intervention is logged: step, state, original/excluded/chosen action, chosen rank, rejections.

## 3. Offline validation (no game) — 8/8 pass (`experiments/s6_offline_tests.py`)
| Check | Result |
|---|---|
| S5 oscillation reproduced in road-state space from saved traces | seed 182150: 124 oscillation steps, seed 205907: 104 |
| Filter detects the cycle and picks the next-ranked legal alternative | first detection at decision 10 (seed 182150, excluded 1391, chose rank-1 action 1091) and 28 (seed 205907, excluded 2077, chose 2231); A,B,A,B criterion verified at the step |
| Non-cycling decisions unchanged | 1,362 decisions (pre-cycle prefixes, 8 WAIT episodes, 30 repeated-construction steps): 0 interventions, action = policy argmax |
| Filter disabled reproduces the original policy | 285/285 recorded actions through the full oscillation; selector ranking head == `model.predict` on 120/120 saved native states, ranking covers exactly the legal set |
| Unit cases | A,A,A / A,B,A not triggered; A,B,A,B triggered; rejection reasons; no_alternative; detected-but-not-continuing; deterministic ties; cycle statistics |
| Terminal flag does not affect rewards | see §1 |

## 4. Paired native evaluation
Same checkpoint (S1 seed 13, `final.zip`), validation seeds 182150 and 205907, normal start, 1.0 s decisions, 600 horizon, native reward, native legal actions, fresh `RESET(seed)` per episode, order A,B per seed. A reproduces S4/S5 exactly (143/142 decisions, same action prefix); B differs only from its first intervention on.

| | seed 182150 A | seed 182150 B | seed 205907 A | seed 205907 B |
|---|---|---|---|---|
| Deliveries | 0 | 0 | 0 | 0 |
| Useful connections (max) | 0 | 0 | 0 | 0 |
| Time to first connection / delivery | – | – | – | – |
| Survival (decisions) | 143 | 143 | 142 | 142 |
| Terminal-penalty missing / on-entry | no / no | no / no | no / no | no / no |
| First divergence (decision) | – | 10 | – | 28 |
| Filter interventions | 0 | 23 (all rank-1 replacements, 0 rejected-no-alternative) | 0 | 20 (all rank-1) |
| Oscillation steps (s_t = s_{t−2} ≠ s_{t−1}) | 124 | 48 | 104 | 42 |
| Maximal A,B,A,B runs ("cycles") | 4 long | 23 short | 2 long | 20 short |
| Road-state revisits at gaps 2 / 4 / 6 | 124 / 0 / 0 | 48 / 36 / 34 | 104 / 0 / 0 | 42 / 19 / 16 (+16 at gap 3) |
| Distinct road states visited | 18 | 24 | 29 | 32 |
| Distinct edges ever added | 10 | 13 | 22 | 25 |
| ADDs / of which revisit an earlier road state ("wasted") | 141 / 125 | 141 / 119 | 130 / 104 | 130 / 101 |
| ADDs that replaced an existing edge (swap) | 133 | 132 | 112 | 111 |
| Closest added edge to the destination (Manhattan) | 5 | 4 | 7 | 7 |
| REMOVE / road-churn (REMOVE-based) | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| Mean masked policy entropy | 2.07 | 2.44 | 2.68 | 2.60 |

"Irrelevant ADD" has no task label natively; "wasted ADD" (an ADD that returns to an earlier road state) and "swap ADD" are the operational proxies. Destinations exist from the start; houses spawn during the game (none at reset), so the traps start when the first house appears.

### First divergence and what happened next
* Seed 182150: identical to A through decision 9 (…1385, 1391, 1385); at decision 10 the filter vetoes 1391 and takes 1091; the policy later returns to the same house-tile swap, so the filter intervenes again every ~4–6 decisions (1391→1091, 1385→943, 1391→1095, 1385→1095, …).
* Seed 205907: identical through decision 27; from decision 28 the filter replaces 2077 by 2231 every 5th decision; the policy resumes the 2083/2077 swap in between.
* Resulting dynamics: a deterministic **period-4/6 (and 3) cycle** replaces the period-2 one. The fraction of decisions spent oscillating is lower, but the number of distinct states is nearly unchanged (24 vs 18, 32 vs 29) and the wasted-ADD count barely moves (119 vs 125, 101 vs 104).

## 5. Interpretation
* **H1 (cycle suppression is the main bottleneck): not supported.** Removing the two-state oscillation did not produce a connection or delivery in either seed.
* **H2 (insufficient): supported by these two seeds.** The filter breaks the cycle it targets, but the CNN's local greedy preference for house-tile edits is a wider attractor: the veto moves it to a longer cycle among 24–32 road states. Neither arm built edges within 3 tiles of the destination (closest 4–5 for seed 182150; 7 for seed 205907). Starting from nothing, the policy never extends a path across the roughly 7–10 tile house-to-destination distance.
* **H3 (mixed): not seen.** Distinct edges and states rose slightly (13 vs 10, 25 vs 22), but not in a direction that reached the destination, and no downstream metric improved.
* **Not established:** that the pretrained prior is useless in the real game (it is the same prior that produced the one stochastic success in S4: 89 distinct edges, a connection after 29 decisions); that a better filter would not help (period ≥ 3 cycles are not handled; one filter design, two seeds, one checkpoint, deterministic only). **LIKELY (untested):** the deterministic policy's coverage is too narrow — it never leaves the neighbourhood of the first house — and discovery in S4 came from sampling-driven coverage. The gap size is also a plausible limit: the synthetic policy was trained on Gap-1/2 paths, whereas a normal start needs a path of roughly 7–10 edges from scratch.

## 6. Recommended next experiment
Smallest step that answers the open causal question without training: the **inference-only stochastic-CNN vs RANDOM delivery-rate measurement** already proposed in S4/S5 (≈16 validation seeds each, scored on deliveries and connections via the new terminal-penalty flag rather than return, ≈20–30 min). It tests whether the pretrained prior's *sampling* behaviour produces connections more often than random coverage, which the deterministic anti-cycle result suggests is where any usable signal lives. Do **not** extend the anti-cycle filter (a period-n variant would only chase longer attractors). Requires your approval; nothing was started.

## 7. Reproduction and artifacts
```
python experiments/s6_offline_tests.py                       # 8/8, no game
python experiments/s6_paired_eval.py --seeds 182150,205907 --cap-min 15
python experiments/s6_geometry.py                            # RESET-only geometry read (destination positions)
```
Raw: `research/native_cnn_anticycle/` (`paired_<seed>_<A|B>.json` with per-step action/road signature/intervention, `offline_test_results.json`, `geometry.json`). Source hashes (sha256 prefix): `mini_motorways_env.py 28c42997`, `diagnostics.py 5e760d30`, `anticycle.py 82254f32`, `s6_offline_tests.py f1fd632c`, `s6_paired_eval.py 1d52f5bf`. Patch record: `experiments/_patch_s6.py`.
