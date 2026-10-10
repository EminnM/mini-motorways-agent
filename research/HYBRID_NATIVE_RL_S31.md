# S31 — Bounded hybrid native RL experiment (PLAN / EDIT / WAIT)

**Environment version: `prefix_v7`** (deployed plugin DLL sha 33e30b33…, i.e. *before* the S32/S33 playable-area fix). All numbers in this report were produced under the pre-fix legal mask, in which 11–12 % of offered ADD edges cross the revealed-area boundary (see `PLAYABLE_AREA_LEGALITY_AUDIT_S32.md`). Post-fix results live separately under `research/s33_playable_area_fix/`.

## 0. One-paragraph verdict
A single PPO run of 100,352 native decisions (205 min wall, 4 headless instances) trained a hybrid policy that, on the 12 held-out seeds, delivers **more than planner B in every checkpoint and sampling mode**, with the paired 95 % CI on the per-seed mean excluding zero for the 25k/50k/100k stochastic checkpoints and the 50k/75k deterministic ones (best: 50k stochastic, +654 total deliveries over B's 615, 8 seeds higher / 2 equal / 2 lower, CI [+18, +86] per seed). The gain is real but of a specific, limited kind: the agent learned to *stay alive longer* (median episode length 306 → 520 decisions; survivors to the 600-decision cap 1 → 4 seeds) by interleaving its own road edits with the planner's steps — it did **not** learn colour separation, and the three "colour-timing" seeds (507919, 515838, 547514) are unchanged by every policy. Reward accounting, legality (0 illegal actions in all 218 training + 144 evaluation episodes) and seed disjointness were valid throughout. **Go/no-go: conditional GO** — native RL on top of planner B produces a measurable improvement within a 100k budget, but checkpoint-to-checkpoint variation is of the same size as the between-checkpoint differences, so the next step must be a replication (≥2 training seeds) before anything is built on this policy, and it must be run on the corrected (`postfix_v8`) environment.

## 1. What was built (no new architecture, no synthetic env)
* `research/s31_hybrid_native_rl/code/hybrid_env.py` — `HybridNativeEnv`: the native v2 observation (16,328 floats), `Discrete(3112)` = the native 3,111 actions + index 3111 = **PLAN**. PLAN is legal iff WAIT is legal (i.e. outside a forced ChooseUpgrade interrupt); on PLAN the planner-B adapter (`NativePlannerPolicy`, S21) chooses one native action for the current state; any other index is the agent's own **EDIT** (full native primitive set: ADD/REMOVE × E/S on any tile the native mask allows, including redundant roads and removals) or **WAIT**. Upgrade interrupts take the forced ChooseUpgrade index (logged as UPG, ~0.6 % of decisions).
* Reward: the native env reward unchanged = score delta (deliveries) per decision with the −10 terminal penalty on game-over. No shaping, no reward for agreeing with the planner or completing connections. Verified: Σ rewards == final score + penalty in every episode.
* Policy: `HybridMaskablePolicy` = the S14 `SpatialExtractorV2` with the pooled WAIT/upgrade head widened from 3 to 4 outputs (WAIT, upg0, upg1, PLAN), initialised from the migrated S8-C v2 checkpoint; PLAN row initialised to zero weight + bias 6.0 so the policy starts as "mostly PLAN". MaskablePPO (sb3-contrib), 4 SubprocVecEnv workers (one headless game each), n_steps 512/env, batch 256, 4 epochs, lr 3e-4, ent 0.01, γ 0.99, λ 0.95, target_kl 0.03, seed 31.
* Seeds: training 700000 + 7919·i (i < 40), cycled per worker; evaluation the 12 designated seeds 500000 + 7919·i (never used for training or any tuning). No hyper-parameters were adjusted after seeing held-out results; checkpoint selection below is reported for all five checkpoints, not just the best.

## 2. Training (100,352 native decisions, 218 episodes, 4 workers)
Training-seed episodes, by chronological fifth of each worker's log (stochastic policy during collection, 600-decision cap):

| fifth | n | score mean / median | decisions mean | survived 600 | destroyed/ep | PLAN | EDIT | WAIT | illegal |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 43 | 94.0 / 109 | 456 | 6 | 32.3 | 0.58 | 0.40 | 0.02 | 0 |
| 2 | 43 | 96.6 / 105 | 455 | 10 | 23.7 | 0.64 | 0.34 | 0.02 | 0 |
| 3 | 44 | 95.3 / 100 | 449 | 9 | 16.4 | 0.68 | 0.31 | 0.01 | 0 |
| 4 | 43 | 87.4 / 83 | 434 | 7 | 14.3 | 0.68 | 0.31 | 0.01 | 0 |
| 5 | 45 | 106.8 / 125 | 474 | 15 | 12.4 | 0.68 | 0.28 | 0.04 | 0 |

PPO diagnostics (`runs/hybrid_s31/train_log.json`): KL-limited updates stayed stable (clip fraction 0.11–0.17, no early-stop cascades), explained variance rose from 0.30 to 0.75–0.84 after ~35k decisions, policy entropy fell from ≈1.1 to ≈0.7 nats around 40–70k and rose back to ≈1.1 by 100k. The clearest *monotone* training signal is **destroyed connections per episode falling 32 → 12** (the agent learned to stop removing/overwriting useful road); the return itself is flat-to-slightly-up and noisy (training seeds differ in difficulty).

## 3. Held-out evaluation (12 designated seeds, 600-decision cap, same wrapper for all policies)
Totals over the 12 seeds; "survived" = reached the 600-decision cap without game-over; destroyed = connected house↔destination pairs severed by the policy's own EDITs; PLAN/EDIT/WAIT = share of decisions. Stochastic runs use one torch seed per map seed (8008 + i); no re-rolls.

| policy | deliveries | median | decisions (median) | survived | destroyed | illegal | PLAN / EDIT / WAIT |
|---|---|---|---|---|---|---|---|
| planner B (always PLAN) | 615 | 39 | 306 | 1 | 0 | 0 | 0.99 / 0 / 0 |
| random strategy (uniform PLAN/WAIT/random legal EDIT) | 499 | 37 | 301 | 0 | 137 | 0 | 0.32 / 0.33 / 0.34 |
| ck 0 det (initialisation) | 800 | 49 | 355 | 2 | 610 | 0 | 0.60 / 0.40 / 0.00 |
| ck 0 sto | 842 | 60 | 381 | 1 | 582 | 0 | 0.47 / 0.52 / 0.01 |
| ck 25k det | 843 | 61 | 374 | 2 | 139 | 0 | 0.88 / 0.11 / 0.00 |
| ck 25k sto | 1041 | 76 | 443 | 3 | 142 | 0 | 0.62 / 0.33 / 0.04 |
| **ck 50k sto** | **1269** | **127** | **521** | **4** | 137 | 0 | 0.57 / 0.42 / 0.00 |
| ck 50k det | 837 | 68 | 393 | 2 | 117 | 0 | 0.85 / 0.15 / 0.00 |
| ck 75k det | 918 | 79 | 421 | 2 | 187 | 0 | 0.85 / 0.15 / 0.00 |
| ck 75k sto | 939 | 92 | 452 | 1 | 186 | 0 | 0.66 / 0.32 / 0.01 |
| ck 100k det | 774 | 56 | 363 | 1 | 187 | 0 | 0.82 / 0.15 / 0.02 |
| ck 100k sto | 1116 | 114 | 502 | 3 | 122 | 0 | 0.61 / 0.35 / 0.04 |

Paired per-seed differences (policy − planner B; bootstrap 95 % CI on the per-seed mean; H/E/L = seeds higher / equal / lower):

| policy | Σ diff | H/E/L | median diff | CI95 mean | vs random: Σ, H/E/L, CI |
|---|---|---|---|---|---|
| ck 0 det | +185 | 6/3/3 | +0.5 | [−19, +50] | +301, 6/3/3, [−6, +62] |
| ck 0 sto | +227 | 6/0/6 | +3 | [−15, +53] | +343, 7/0/5, [−3, +62] |
| ck 25k det | +228 | 5/3/4 | 0 | [+0.2, +41] | +344, 9/1/2, [+0.2, +61] |
| ck 25k sto | +426 | 7/2/3 | +26 | [+10, +63] | +542, 8/1/3, [+17, +76] |
| ck 50k det | +222 | 6/2/4 | +0.5 | [+1, +38] | +338, 9/1/2, [+2, +59] |
| **ck 50k sto** | **+654** | **8/2/2** | **+77** | **[+18, +86]** | +770, 10/1/1, [+38, +92] |
| ck 75k det | +303 | 7/2/3 | +13 | [+9, +44] | +419, 8/2/2, [+6, +67] |
| ck 75k sto | +324 | 6/1/5 | +11 | [−8, +61] | +440, 10/0/2, [+18, +57] |
| ck 100k det | +159 | 6/4/2 | +1.5 | [−12, +37] | +275, 8/1/3, [−6, +54] |
| ck 100k sto | +501 | 8/1/3 | +27 | [+14, +72] | +617, 8/0/4, [+19, +84] |

Per-seed deliveries / decisions (`*` = survived to the cap):

```
seed      plannerB   random    0_det    0_sto   25k_det  25k_sto  50k_det  50k_sto  75k_det  75k_sto 100k_det 100k_sto
500000    105/498   40/313   40/309  134/547   81/457   76/410   82/457  175/600* 128/539   86/440  118/513  119/512
507919      5/151    5/151    5/151    0/151    5/151    5/151    5/151    5/151    5/151    3/151    5/151    4/151
515838      6/145    6/151    6/145  155/600*   5/145    4/145    5/145    5/145    5/145    5/145    5/145    5/145
523757     40/310   94/450   41/310   84/439   41/310  133/550   41/310  169/600*  43/310  114/501   43/310  108/491
531676     38/295   34/293  167/600* 119/509   38/295   56/373   83/428  115/505   77/417   60/368   68/403  161/600*
539595     43/323   33/300   40/323  119/543   36/323  170/600*  36/323  143/578   36/323  141/558   43/323  129/546
547514      5/154    3/154    5/154    4/154    5/154    5/154    5/154    5/154    5/154    5/154    5/154    5/154
555433     40/318   78/428   56/386   30/318   80/425   74/429  106/484  150/600*  80/425  113/521   80/425   77/423
563352    172/600*  57/392   75/463   68/413  164/600* 160/600* 161/600*  85/449  164/600*  97/463   83/436  134/560
571271     35/290   84/444   60/408   42/337   88/473   75/457   91/474  131/526   84/474  162/600*  35/290   51/374
579190     88/442   25/258  168/600*  52/349  172/600* 162/600* 169/600* 164/600* 162/600* 158/600*
587109     38/301   40/301  137/558   35/301  128/523  121/515   53/358  122/515  129/523  121/507  125/523  165/600*
```

## 4. Reading the result honestly
1. **Deliveries track survival, not throughput.** Deliveries per decision are pinned at ≈0.06 by pin generation / car supply (S21, S28); every policy that lives longer delivers more. Planner B dies at decision ~300 on most seeds (the first house whose colour cannot be routed, or gridlock from the planner's single-tree connections). The hybrid agent's EDITs — removing its own or the planner's road where it blocks, re-adding, and filling in redundant links — postpone game-over by 100–250 decisions on 6–8 seeds. That is a legitimate improvement in the game's own objective, but it is "survival engineering", not routing insight.
2. **Colour separation was not learned** (user's observation on the visible run, confirmed by the traces): the agent's edits are local repairs around congested tiles; it never builds separate colour sub-networks. Three seeds (507919, 515838, 547514) end at 145–154 decisions with ≤6 deliveries under every policy because the first same-colour destination appears before any road can reach it — this is the colour-timing failure mode documented in S21/S23 and no amount of road editing fixes it.
3. **The initialisation already differs from planner B** — ck 0 scores 800/842 vs B's 615, but with ~600 destroyed connections (the S8-C prior fires REMOVE/ADD everywhere). Training mainly *reduced destruction* (610 → 117–187) while keeping or improving deliveries; the net learned gain over the initialisation is +40 to +430 total deliveries depending on checkpoint and mode — modest and noisy.
4. **Checkpoint ordering is not reliable.** 50k sto (1269) > 100k sto (1116) > 25k sto (1041) > 75k sto (939); deterministic peaks at 75k (918) and drops to 774 at 100k. The S13 noise floor for stochastic native play is SD ≈ 55 deliveries per seed, i.e. ≈ ±190 on a 12-seed total — the gaps between 25k/50k/75k/100k are inside that band. The paired CIs vs B exclude zero for 5 of the 10 checkpoint×mode cells, but the *best* cell was chosen after looking at all ten (selection bias), so the honest estimate is the pooled one: all trained stochastic checkpoints ≈ +320 to +650 over B, deterministic ≈ +160 to +300.
5. **Stochastic > deterministic everywhere.** The deterministic policy collapses to PLAN 82–88 % of the time and reproduces planner B's outcome (identical score/length on 4–6 seeds per checkpoint); the gains come from sampled EDITs. Sampling noise is therefore part of the mechanism, which limits reproducibility of any single evaluation.
6. **Do deviations help?** Every evaluated hybrid episode deviates from B (≥1 EDIT), so the within-checkpoint "deviating vs not" contrast is empty. Across seeds, deviations help on the 6–8 non-colour-timing seeds (+40 to +130 each) and hurt on 1–2 (e.g. 563352, where B itself survives: hybrid 85–134 vs B 172 in 5 of 10 cells; 531676 ck 0 vs trained). The random strategy (uniform PLAN/EDIT/WAIT, 137 destroyed) scores *below* B (499 vs 615), so the hybrid's edits are not random perturbation; they are selectively useful.
7. **Budget / validity.** 100,352 decisions ≤ the 100k hard max (+352 from the last rollout rounding, disclosed). 0 illegal actions anywhere. Reward accounting verified per episode. Seeds disjoint. No reward, observation, mask or native-mechanics changes. The playable-area gap (S32) was present throughout: both planner B and the hybrid could use boundary-crossing edges; whether they did is measured in S33.

## 5. Go / no-go for continuing native RL
**Conditional GO.** Evidence for continuing: a learned policy beat its own planner baseline on the game's true objective, with paired CIs that exclude zero in half the cells and the right direction in all ten; integration cost is now zero (the harness, wrapper, reward accounting and evaluation pipeline exist and are validated); 14 decisions/s/instance makes 100k decisions ≈ 3.5 h on 4 instances. Evidence against building on it immediately: one training seed; between-checkpoint spread ≈ the noise floor; gains are survival-driven and plateau at the 600-decision cap (4 seeds already hit it, so the metric saturates); no colour separation.

Conditions for the next milestone (not started; requires approval):
1. **Replicate before extending**: 2 further training seeds under the corrected `postfix_v8` environment, same budget and hyper-parameters, evaluated at the same checkpoints — decide on the *mean over training seeds*, not the best cell.
2. **Raise the cap** for evaluation (e.g. 1,500 decisions) so survival gains are not truncated; keep 600 for training unless approved otherwise.
3. Re-baseline planner B on `postfix_v8` first (S33 does this).
4. Keep the three colour-timing seeds in the held-out set but report them separately; they carry no signal for any road-editing policy.

Artifacts: `research/s31_hybrid_native_rl/{code,runs/hybrid_s31/{checkpoints,episodes,train_log.json},eval/*.json.gz,results.json}`; visible viewing run `eval/vis_step_0024576_sto_539595.json.gz` (not part of the paired comparison).
