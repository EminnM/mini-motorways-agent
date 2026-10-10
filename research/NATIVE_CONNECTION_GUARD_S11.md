# S11 — Native connection-preservation causal probe

**Result in one paragraph.** An evaluation-only guard that vetoes only those ADDs that would disconnect a currently connected house removed **all** connection destruction (521 destroyed connections in the 12 control episodes → **0** in the 12 guarded episodes; every observed loss was caused by an ADD, none by REMOVE / WAIT / upgrade). Deliveries rose from 552 to 735 in total (+33 %; 14.2 → 16.7 deliveries per 100 decisions) and the guard produced more deliveries on 8/12 seeds, equal on 1, fewer on 3. But the paired median difference is small (+2.5 deliveries; median survival difference 0 decisions), the gain is concentrated in three seeds (+95, +76, +70), three seeds got worse (−65, −11, −4), and the concrete budget is at zero for 63 % (control) / 69 % (guard) of decisions with ≥ 10 consecutive empty decisions in **24/24** episodes. Best supported reading: **H11b — destructive churn is a real, now-verified cause of lost work, but removing it is not sufficient; concrete inefficiency and slow adaptation to spawned houses remain the binding limits.** Exploratory (12 seeds, one checkpoint, native non-determinism); the guard is a diagnostic intervention, not a proposed solution. No training, no reward / mask / observation / game change; S1–S10 untouched. **Native-active wall-clock ≈ 8.4 min** (cap 10): side-effect twin 21 s, two 4-episode counterfactual-model validation runs 81 s + 70 s, a 35 s semantics probe, the 24-episode paired experiment 296 s.

## 1. Guard: exact semantics (`experiments/s11_guard.py`)
* Evaluation-only filter on the policy's choice; the CNN receives the identical observation in both conditions. The guard reads the **raw native observation** (privileged) and never feeds anything back to the network, the reward or the game.
* Control: `predict(obs, native mask)` (stochastic). Guard: `predict(obs, native mask minus vetoed ADDs)` — i.e. sampling from the original masked policy distribution **conditioned on the permitted actions** (verified: total variation 0.041 vs the analytic renormalised distribution over 3,000 draws, no vetoed action ever drawn; with an empty veto set the guarded sampler consumes the RNG exactly like the control, so paired seeds stay paired). WAIT / REMOVE / UPGRADE are never vetoed; the native mask is not modified; vetoed actions are not replaced by WAIT (the remaining policy mass decides).
* Veto rule: for each native-legal ADD, predict the **functional** road graph after the action and veto iff some house that is functionally connected to a destination of its group before the action would not be afterwards (all affected houses are protected, none exempted). Counterfactual model of one ADD on a house tile h (functional = road state Active / VisiblyActive): afterwards h's active edge set is exactly the new edge (the displaced driveway is either deleted or left **Mothballed**, which is non-functional either way); an ADD that touches no house is a plain extension (adding edges cannot disconnect anything, so only house-touching ADDs are evaluated). Unknown cases — an edge between two houses, a house with ≥ 2 active edges, a destination/carpark endpoint — are logged, never vetoed (0 occurred in the experiment).
* Water/bridge-like edges follow the native mask and get no extra rule.

## 2. Validation before and during the experiment
**Offline (no game), 12/12 pass** (`offline_test_results.json`): destroying replacement vetoed; preserving replacement allowed; route extension allowed; new-house connection allowed and connects it; multi-house loss protects all; water edge = plain extension; unknown counterfactuals not vetoed; no connected houses ⇒ permitted = native mask (also on the 24 real captured native states: raw observation and mask untouched, 0 vetoes); WAIT / REMOVE / upgrade never vetoed; Mothballed edges are not functional connections; conditional-sampling checks above.
**Native counterfactual validation — and a model failure.** My first model (any non-"None" state is a road; a house's single edge is deleted on replacement; archived `s11_guard_v1_any_state.py`) was validated on four unused seeds (600000, 607919, 615838, 623757; control episodes, 1,418 executed ADDs) and **failed**: 308/696 replacement predictions and 282/357 "no-op" predictions were wrong; as a veto it would have had precision 15 % (243 harmless vetoes vs 44 correct) and recall 56 %. A 35 s probe showed why: the displaced house edge is often left **Mothballed** (non-functional but not "None"), and re-adding it reactivates it. I changed the model to functional (Active) edges and re-ran the validation on the same four seeds: **1,375/1,375** known predictions on non-spawn steps matched the observed Active-edge changes; the guard verdict separated all 1,418 executed ADDs perfectly (1,085 allowed with no loss, 333 vetoed with a loss; no loss from REMOVE / WAIT / upgrade). Caveat: the semantic probe used the first of those validation seeds, so that check is partly in-sample; the out-of-sample check is the main experiment: on the 12 evaluation seeds the model mispredicted 1 of ≈ 3,500 ADDs per condition.
**Side effects.** A 60-decision paired rollout with the guard computed but not applied was identical to the plain rollout (actions, rewards, observation hashes), but two plain rollouts of the same seed were *not* identical to each other (native non-determinism), so this twin is weak evidence by itself. Additional evidence: the guard code works on copies (offline tests show raw observation and mask unchanged), and the instrumented control episodes reproduce the original uninstrumented S9 stochastic S8-C episodes **exactly (steps and deliveries) on 7/12 seeds** and closely on three more.

## 3. Native experiment (12 S9 seeds, paired, stochastic S8-C `step_0099840.zip`, SHA 85a9cb75cf287d24)
Seeds 500000 + 7919·i (i = 0…11), torch sampling seed 1000 + i in both conditions, 8 processes in parallel, all 24 episodes completed (no cut-off). Source hashes: `s11_guard.py 66960e92`, `s11_runner.py 68b6ae0c`, `s11_offline_tests.py b259b2c1`, `s11_analyze.py e7d6e915`.
**Definitions.** Connections are counted **functionally** (Active / VisiblyActive edges). The legacy any-state count used in S9/S10 overstates them: at episode end 71 (legacy) vs 63 (functional) connected houses in the control runs. Consequently S10's connection-loss count (54 in two episodes) was an under-count; the functional count is 521 in the 12 control episodes.

| metric | control | guard | paired difference (seeds guard > = <) |
|---|---|---|---|
| deliveries, median (total) | 40 (552) | 67.5 (735) | median +2.5 (8 / 1 / 3) |
| deliveries per 100 decisions | 14.2 | 16.7 | |
| survival, median decisions (total) | 313 (3,877) | 403.5 (4,390) | median 0 (5 / 3 / 4) |
| episodes with ≥ 1 delivery | 11/12 | 11/12 | |
| episodes reaching the 600-decision horizon | 1 | 0 | |
| peak functional connections, median (total) | 6 (68) | 7.5 (84) | +0.5 (6 / 5 / 1) |
| connections destroyed (total) | 521 (11/12 seeds) | **0** | −36.5 median (0 / 1 / 11) |
| steps in which a destructive ADD was chosen | 521 | 0 | |
| unconnected houses at termination, median (total) | 3 (47) | 3.5 (52) | 0 (4 / 4 / 4) |
| concrete = 0 for ≥ 10 consecutive decisions | 12/12 | 12/12 | |
| fraction of decisions with zero concrete, median | 0.63 | 0.69 | +0.03 (11 / 1 / 0) |
| first connection decision | identical on all 11 seeds that connect (22 median) | | |
| newly spawned houses connected within 60 decisions | 43/100 (43 %; 10 censored) | 58/114 (51 %; 22 censored) | |
| terminal-penalty-missing (flag, not rewritten) | 1 | 1 | |

Per-seed (control / guard): 
| seed | decisions | deliveries | connections destroyed | peak connections | unconnected at end | zero-concrete fraction | first connection |
|---|---|---|---|---|---|---|---|
| 500000 | 315 / 582 | 42 / **137** | 32 / 0 | 9 / 13 | 0 / 6 | 0.686 / 0.751 | 22 / 22 |
| 507919 | 296 / 539 | 32 / **108** | 23 / 0 | 6 / 11 | 2 / 5 | 0.733 / 0.772 | 22 / 22 |
| 515838 | 145 / 145 | 4 / 5 | 4 / 0 | 2 / 2 | 2 / 2 | 0.483 / 0.503 | 64 / 64 |
| 523757 | 153 / 153 | 0 / 0 | 0 / 0 | 0 / 0 | 4 / 4 | 0.51 / 0.51 | never |
| 531676 | 600 / 423 | **146** / 81 | 64 / 0 | 12 / 7 | 7 / 9 | 0.662 / 0.716 | 15 / 15 |
| 539595 | 508 / 511 | 101 / 106 | 13 / 0 | 9 / 11 | 5 / 3 | 0.579 / 0.605 | 23 / 23 |
| 547514 | 154 / 154 | 4 / 5 | 6 / 0 | 2 / 2 | 2 / 2 | 0.643 / 0.649 | 80 / 80 |
| 555433 | 349 / 338 | 45 / 41 | 111 / 0 | 8 / 8 | 3 / 3 | 0.576 / 0.601 | 27 / 27 |
| 563352 | 311 / 268 | 38 / 27 | 70 / 0 | 5 / 6 | 3 / 1 | 0.611 / 0.698 | 31 / 31 |
| 571271 | 459 / 440 | 68 / 72 | 93 / 0 | 6 / 8 | 11 / 8 | 0.691 / 0.75 | 11 / 11 |
| 579190 | 233 / 453 | 20 / **90** | 41 / 0 | 3 / 10 | 5 / 3 | 0.597 / 0.682 | 16 / 16 |
| 587109 | 354 / 384 | 52 / 63 | 64 / 0 | 6 / 6 | 3 / 6 | 0.709 / 0.724 | 22 / 22 |
Paired delivery differences: +95, +76, +70, +11, +5, +4, +1, +1, 0, −4, −11, −65. Total +183; the three largest gains alone add +241, so the other nine seeds sum to −58 deliveries — the median (+2.5) is the honest summary; three seeds (515838, 523757, 547514) end by ≈ decision 145–154 before the guard can matter.

## 4. Guard statistics and secondary diagnostics
* Guarded episodes: the guard vetoed something in 3,862 of 4,390 decisions (88 %); the action actually changed (compared with the action the control sampler would have drawn from the same RNG state) in 1,142 decisions (26 %); the policy's probability mass on vetoed actions had a median per-seed mean of 0.285 when vetoes existed. In the control runs the guard *would have* vetoed something in 3,327 decisions and the control executed 521 destructive ADDs (≈ 15 % of its 3,497 ADDs; 20 % of its 2,602 replacement ADDs). Unknown counterfactual cases: 0.
* Action mix (control / guard): ADD 90.2 % / 85.5 %, REMOVE 4.8 % / 6.6 %, WAIT 4.4 % / 7.3 %, upgrade 0.6 % / 0.6 %. Net-growth ADDs 25.6 % / 27.3 % of ADDs; replacement ADDs 2,602 vs 2,730 in total (destructive 521 → 0, so non-destructive replacements rose from 2,081 to 2,730): the guard converts destructive swaps into harmless ones rather than into construction — **churn as a decision sink is unchanged** (a guard that prevents disconnection but leaves the policy swapping will not by itself add roads).
* Concrete: at first connection identical in both conditions (median 3; the guard cannot act before a connection exists); zero-concrete fraction rose slightly (guarded episodes live longer at an empty budget); concrete at termination median 6.5 vs 1; median legal ADD count 16 vs 19.5.
* Legal ADD / REMOVE counts and concrete are logged every decision in `raw/`.

## 5. Representative trajectories
* **Seed 500000 (largest gain):** control and guard are identical until the first destructive swap; control ends at 315 decisions with 42 deliveries and 6 connections, the guard keeps 5 → 10 → 13 connections, survives to 582 and delivers 137; both spend most decisions at zero concrete.
* **Seed 531676 (largest loss):** the control episode is the exceptional S9/S10 success (600 decisions, 146 deliveries, 12 connections); the guarded run plateaus at 4–6 connections from decision ≈ 120 to 360 and dies at 423 with 81 deliveries. With stochastic divergence and one realisation per seed I cannot say whether this is the guard's effect (it removes swaps that sometimes re-routed a house) or ordinary variation (H11d is therefore *not excluded* for this seed).
* **Seeds 515838 / 523757 / 547514:** first connection at decisions 64 / never / 80, game over at 145–154: nothing for the guard to protect.

## 6. Hypotheses
* **H11a (destructive churn is an important causal bottleneck): partly supported.** The mechanism is confirmed (100 % of observed losses were ADD-induced; the guard removes them exactly) and deliveries rise in 8/12 seeds and +33 % overall, but the median paired effect is small and the improvement is concentrated in 3 of 12 seeds.
* **H11b (contributes, not sufficient): best supported.** Connections are preserved yet unconnected houses at the end (47 vs 52), spawn adaptation (43 % vs 51 % within 60 decisions) and the empty budget (≥ 10 consecutive zero-concrete decisions in 24/24 episodes; 63 % → 69 % of decisions) are essentially unchanged.
* **H11c (not dominant): not supported** — the effect is not zero.
* **H11d (adverse effects): possible in 3/12 seeds** (−65, −11, −4), not demonstrable with one stochastic realisation per seed.
Limits: 12 seeds, one checkpoint, stochastic policy, native non-determinism (paired trajectories diverge after the first guard intervention), the "functional" definition (Pending edges are not counted; Mothballed excluded) is an inference from the observation, validated against deliveries only indirectly.

## 7. Decision gate
**Does protecting existing connections materially improve the policy's ability to sustain native gameplay?** Not in the median — modestly in aggregate (+33 % deliveries, +13 % decisions survived, 521 → 0 destroyed connections) and strongly in a minority of seeds. Because the remaining limits are shared by both conditions, the evidence points more strongly toward **concrete inefficiency** (empty budget most of the time, in every episode) and **slow adaptation to spawned houses** than toward churn alone; traffic limits remain unmeasured.
**Recommended smallest training experiment (needs approval; not started):** a single-seed synthetic continuation of S8-C (≈ 100k decisions, ≈ 45 min at the S8 speed) in an *extended S7* that contains the mechanics now known to matter — several houses spawning on a native-like schedule, a finite road budget with weekly top-ups, Active / Mothballed house-edge replacement, and a continuing delivery stream as the only reward (+1 per delivery while connected; no shaping). In that world a destroyed connection and a wasted concrete piece cost deliveries naturally, so connection preservation and budget efficiency can be learned without a privileged filter. Then evaluate the checkpoint with the S9 protocol on fresh seeds. A cheaper pre-check, if you prefer one more native probe first (≈ 10 min): an evaluation-only guard that conserves concrete (blocks ADDs that exceed the route's requirement) on the same seeds, to see whether budget — rather than churn — carries the larger effect.

## 8. Reproduction
```
python experiments/s11_offline_tests.py                                # 12 offline checks
python experiments/s11_runner.py --twin                                # side-effect twin
python experiments/s11_runner.py --cond control --seeds val --idx 0    # counterfactual-model validation (seeds 600000+7919 i), idx 0..3
python experiments/s11_probe_house_semantics.py 600000 0               # Mothballed-state probe
python experiments/s11_runner.py --cond control|guard --seeds s9 --idx 0,1,2   # (and 3,4,5 / 6,7,8 / 9,10,11), 8 processes in parallel
python experiments/s11_analyze.py
```
Artifacts: `research/s11_connection_guard/` (per-decision records for all episodes in `raw/`, `s11_summary.json`, `offline_test_results.json`, `twin_noninterference.json`, `probe_600000.json`, `s11_guard_v1_any_state.py`, logs).
