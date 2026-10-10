# S5 — Native repeated-ADD and missing-terminal-penalty audit

Investigation only. No fix, reward/mask/observation change, native-game change, PPO or checkpoint change was made. Added: trace/regression tooling only (`experiments/s5_audit_trace.py`, `experiments/s5_regression_tests.py`).
Evidence: `research/native_action_reward_audit/` (per-step native traces with the full IPC response fields; `regression_results.json`; `twin_summary.json`). Labels: **CONFIRMED** (directly observed / checked in code), **LIKELY** (consistent with all evidence, mechanism not directly observed), **UNRESOLVED**.
**Native wall-clock used: ≈ 7.1 min** (single game instance at a time, four short runs, including launches), under the 15-minute cap.

## Headline
* **Repeated ADDs (CONFIRMED):** not a mask, application or observation defect. The S4 "one ADD repeated 67 times, state unchanged" reading was **wrong**. The deterministic CNN falls into a **2-cycle**: two different ADD actions that each *replace the other* at a House tile (the native game keeps one road connection per house tile). Category **E (expected native semantics) + D (policy failure: deterministic argmax cycles forever)**; there is also a sandbox-fidelity gap (the sandbox forbids edits touching the house tile).
* **Missing −10 (CONFIRMED mechanism at the reward-code level; cause of the trigger LIKELY):** the penalty is paid only on the STEP that *turns* the game terminal. When the game is already terminal when the next STEP starts, that STEP returns `terminated=true`, `accepted=false`, `applyInfo="terminal state (not applied)"` and penalty 0. It is a **native-side race**, not a wrapper, logging, or truncation artifact, and it is **not a deterministic function of seed and actions**.
* Neither finding explains why most normal-start policies never make a connection. That remains a reinforcement-learning/exploration problem; these are engineering defects layered on it.

## 1. Repeated ADD actions

### Method
Reproduced S4's deterministic S1-seed-13 CNN on S4 seeds 182150 and 205907 (143 / 142 decisions, same lengths and returns as S4), logging every step: action, decoded edit, mask legality before, `accepted`/`appliedAction`/`applyInfo` from the raw STEP response, the edge's road state on both tiles before and after, the full set of road edges before and after (added/removed), which observation channels changed, whether the action stays legal, raw logit and masked top-5.

### Findings
1. **CONFIRMED — it is a 2-cycle, not a consecutive repeat.** Longest run of the same ADD = 1. Seed 182150: from step 7 the policy alternates `ADD(-5,-1,E)` (index 1385) and `ADD(-4,-1,S)` (index 1391), 67 and 66 times; seed 205907: `ADD(-15,4,E)` (2085) / `ADD(-15,4,S)` (2087), 32 each (plus a second pair, 2083/2077). 242 loop steps over the two seeds (of 285 decisions). S4's "67 repetitions" was the per-index count of one member of the pair.
2. **CONFIRMED — every loop step is a valid, applied, observable edit.** 242/242 loop steps: action in the mask; `TileEditor.AddRoad succeeded (resultCode=Success)`; the edge is absent on both tiles before and Active on both after; the road-edge set changes; the road channels of the policy observation change; the action is illegal afterwards (the mask is correct). Hence not **A** (mask), **B** (application) or **C** (aliasing).
3. **CONFIRMED — each ADD replaces the other edge.** Adding edge X removes edge Y and vice versa; the net road count stays constant (e.g. step 7 adds (-5,-1,E) and removes (-4,-1,S); step 8 adds (-4,-1,S) and removes (-5,-1,E)). All 245 ADD steps in both traces that removed an existing edge touch a House tile (237/237 inside the cycles). **LIKELY:** the native rule is "one road connection per house tile"; I did not read the game rule itself.
4. **CONFIRMED — the policy cannot leave the cycle deterministically.** The observation returns to the same two states, so the logits repeat per cycle (e.g. 6.02 for X, 4.06 for Y; runner-up logit 0.1–1.9 lower), and argmax picks the swap again. Time-driven features (vehicles, demand) change by at most a few entries and do not alter the argmax. Stochastic sampling escapes (S4: 51 distinct edges vs 13).
5. **CONFIRMED — sandbox/native difference.** The synthetic env allows ADD only when both endpoints are free (not house/destination/carpark) and gives the house a fixed driveway; in the native game edits touching the house tile are legal and swap the driveway. The S1 policy never saw these actions in training, which is consistent with (not proof of) it finding them attractive.
6. **Carpark REMOVE quirk: unrelated (CONFIRMED by action type).** The loops contain only ADDs at House tiles; the carpark quirk was REMOVE on a carpark stub reported applied with no change.

**Classification: E + D**, mask/application/aliasing ruled out. Not established: *why* the CNN prefers the swap (UNRESOLVED), and whether swapping is ever useful for connectivity.

## 2. Missing terminal penalty

### Code (CONFIRMED by reading `Entrypoint.cs`, STEP handler, and by `test_csharp_penalty_rule…`)
`wasTerminal = IsTerminal(game)` is read **before** the action is applied; after advancing, `justTerminated = isTerminalNow && !wasTerminal`; `terminalPenalty = justTerminated ? −10 : 0`. If `wasTerminal` is already true the handler applies nothing, advances nothing (`applyInfo = "terminal state (not applied)"`) and still returns `terminated=true`. `IsTerminal` is `HasGameEnded` **or** the overcrowding timer being within one decision (+0.05 s) of `MaxOvercrowdTime` (a deliberate one-decision pre-emption).

### Traces (live, WAIT policy, per-step full response)
* Normal terminal (9 traces): last step `applyInfo="WAIT (no-op)"`, `accepted=true`, sim time advanced 1.0 s, `terminalPenalty=-10`, `terminated=true`, the previous step `terminated=false`. Exactly one −10 per episode; never two.
* Zero-penalty terminal (2 traces, seed 182150): last step `applyInfo="terminal state (not applied)"`, `accepted=false`, reward components `{scoreDelta 0, terminalPenalty 0}`, `terminated=true, truncated=false`, previous step `terminated=false`; sim time moved **0.1 s** (141.9 → 142.0) instead of 1.0 s, i.e. the clock had moved by something other than that STEP before it started.
* Ended through native game-over every time (`truncated=false`, horizon 600 never reached, no process reset). Score 0 in all (no deliveries), so scoreDelta is not involved.
* S4 logs: 5/20 terminated episodes carry no penalty; all others exactly −10 (no episode has any other value).

### Determination
| Hypothesis | Verdict |
|---|---|
| Intended reward semantics | Unlikely: the design comment/handler pays −10 once at game over; skipping it is a side effect of `!wasTerminal`. (LIKELY unintended) |
| Environment-wrapper (Python) bug | **No (CONFIRMED):** the raw native response already has `terminalPenalty 0` with `terminal state (not applied)`; Python forwards it unchanged. |
| Logging/accounting bug | **No (CONFIRMED):** returns from the diagnostic rows, plain env sums and raw responses agree; the logging-on/off twin (full 143-decision terminal episode) was identical, including the 0 return. |
| Episode-boundary/truncation artifact | **No (CONFIRMED):** native `terminated`, not truncation. |
| Native-side race: terminal state reached between STEP requests | **CONFIRMED** (the STEP starts already terminal). **LIKELY** that a stray 0.1 s clock tick outside STEP pushes the overcrowd timer into the pre-emption window; the source of that tick (the Unity frame driver / pause re-assertion) is **UNRESOLVED**. |

Occurrence is **not** determined by seed and actions: seed 182150 gave −10 in 4 plain WAIT traces (and in the CNN trace) and 0 in two back-to-back no-delay twin episodes and in two traces with a 0.5 s pause before steps ≥130; seed 205907 always −10. The dependence on step cadence is non-monotonic and unexplained (UNRESOLVED); only the seeds whose game-over falls near the pre-emption boundary seem exposed (S4: 182150, plus RANDOM episodes ending at 130 and 137).

## 3. Reproduction and minimal tests
* Loop (live, ≈1 min): `python experiments/s5_audit_trace.py loop --seeds 182150,205907` → traces `loop_cnn_det_*.json`.
* Terminal (live, ≈1.5 min): `python experiments/s5_audit_trace.py terminal` (WAIT traces + logging twin); `hunt [--delay 0.5]` (back-to-back WAIT episodes, full capture) reproduced the zero-penalty case.
* Regression tests, **no game needed** (saved traces, S4 logs, C# source text): `python experiments/s5_regression_tests.py` → **9/9 pass**: loop is a 2-cycle; each loop step is legal/applied/road-changing/observation-changing/illegal afterwards; each ADD replaces one edge at a House tile (245/245 swaps touch a House); penalty paid iff the game was not already terminal at STEP start; zero-penalty last step advanced the clock 0.1 s; C# rule text; S4 logs 5/20 zero else −10; logging-on/off twin identical on a full terminal episode.
* **S4 non-interference gap closed:** plain env vs `RecorderEnv`+diagnostics over a full terminal episode (143 decisions): observations, legal masks, rewards and termination identical; episode row return matches; `diag_errors=0`. Caveat: this pair happened to be a zero-penalty episode, so the penalty-present terminal path is covered by the separate traces, not by a twin.

## 4. Effect on S4 validity
| Issue | S4 results affected? |
|---|---|
| Repeated-ADD 2-cycle | **Interpretation corrected** (the S4 report said a single ADD repeated with no state change; wrong). The deterministic CNN's 0/4 outcome is an artifact of the swap cycle, so "deterministic CNN never connects" says little about the policy's ability; the stochastic CNN result (1/4 with 41 deliveries) is unaffected. Gate 0 and the noninterference checks are unaffected. |
| Missing penalty | **Mean returns in the S4 table are contaminated** (WAIT −7.5 instead of −10 etc.); do not compare returns across policies or against the frozen baselines' returns. Deliveries, connections, survival, action statistics are unaffected. The +31 episode is correct (41 − 10). |

## 5. Would each compromise PPO training?
* **Loop:** not by itself — PPO samples, so it does not freeze in a deterministic cycle; but swap edits are available at every house, cost decisions without progress, and the pretrained prior favours them (UNRESOLVED how strongly under sampling). Deterministic evaluation of any checkpoint will show the artifact; evaluate stochastically too. No evidence it blocks learning.
* **Missing penalty:** adds ±10 reward noise on ≈25% of terminations (5/20 observed, 2/11 in new traces), independent of the policy in non-delivering episodes, so it mostly shifts value targets and variance; it weakens but does not remove the survival signal. Moderate risk, not a blocker. It matters more if survival differences are used as the learning signal.

## 6. Proposed fixes — NOT implemented
1. *Measurement-only (zero behavioural risk, can be done immediately):* record `terminal_penalty_missing` (final STEP `applyInfo == "terminal state (not applied)"`) per episode in diagnostics, and report S4-style tables with and without those episodes.
2. *Native-side fix (needs approval — changes game/reward behaviour):* pay the penalty on the first STEP that reports `terminated=true`, e.g. track a "terminal already reported" flag instead of `wasTerminal`.
3. *Python-side equivalent (reward change, needs approval):* when a response has `terminated`, `terminalPenalty==0` and `applyInfo=="terminal state (not applied)"`, add −10 in the wrapper.
4. *Loop:* no environment change warranted (E). Options for later, none requested: sampling-based evaluation alongside argmax; a sandbox-fidelity change letting house-tile edits swap the driveway (would require retraining S1; forbidden now). No anti-loop heuristic proposed.

## 7. Smallest justified next action
Neither issue is the reason connections are rare. The engineering issues are real but second-order: (a) the loop is a deterministic-evaluation artifact and a sandbox gap, (b) the missing penalty is reward noise. The decision-relevant gap is still the one from S4: whether a sampling pretrained CNN reaches deliveries more often than RANDOM. Smallest next step: approve fix 1 (logging only) and the S4-proposed inference-only measurement (≈16 validation seeds each for RANDOM and the stochastic CNN, scored on deliveries/connections, not return, ≈20–30 min). Hold fixes 2/3 until/unless a PPO pilot is approved; then apply one of them before the pilot.
