# S14 — Group-Aware Observation: Root Cause, Encoding v2, Checkpoint Migration, Feasibility

Scope kept: no native PPO, no native reward/mechanics change, no inference-time guard, no privileged/future information, no channel repurposing, S8-C not overwritten (migrated copy is a new file), no 100k run, no native evaluation. One synthetic feasibility run of 10,240 decisions (Gate D).
Artifacts: `research/s14_group_observation/` — `gate_a.json`, `gate_b.json`, `gate_c.json`, `gates_summary.json`, `info_loss.json`, `channel_audit.json`, `s8c_migrated_v2_step0.zip`, `runs/feasibility_s14/` (feasibility.json, progress.csv, checkpoints, per-episode rows), `logs/`.
Code (all new; S1–S13 files untouched): `python/synthetic_rl/obs_v2.py`, `models_v2.py`, `s14_env.py`, `s14_tests.py`, `experiments/s14_train.py`.

## 1. Root cause: where group identity is lost

| stage | what carries identity | verified by |
|---|---|---|
| 1 Unity objects | `HouseModel.GroupIndex`, `DestinationModel.GroupIndex` (int). It is the **theme colour group**: `MotorwaysThemeDatabase.MAX_THEME_COLOR_GROUPS`, `GenerateDemandProcess._allocatedColorGroups = new bool[6]`, `Motorways.Themes.ColorGroup`, `PerGroupMaterialBindings.RenderersWithColor` (the sprite colour). Demand generation and trips match `house.GroupIndex == destination.GroupIndex` (`GenerateDemandProcess.cs:216–323`). | static recon sources |
| 2 Native state extraction | plugin reads both models per tile | `Entrypoint.cs:3211–3219` |
| 3 PolicyObservation JSON | `"groupIndex": <int or null>` **per tile** — the integer is already exported | `Entrypoint.cs:3223`; 48 captured states: 24/24 house tiles and 192/192 destination tiles carry it (gate A) |
| **4 Python decoding** | **`mini_motorways_env.py:114`: `out[base+11] = 1.0 if groupIndex is not None`** — the integer is reduced to a has-group flag | **identity lost here** |
| 5 Synthetic S7 / S13 | replicate the v1 flag (`F_GROUP = 1.0` on house and destination tiles) — faithful to v1, hence equally blind | `s7.py:220–222`, `s13_env.py` |
| 6 CNN | consumes the 15 channels; never sees identity | `models.py` |

So the diagnosis in the S13 report is confirmed: the loss is a Python-side encoding decision, not a plugin or game limitation. The plugin's existing JSON already suffices; **no native change is needed**.

### Native identity verification (Gate A — 5/5 pass, `gate_a.json`)
* Exists internally and is readable: yes (above).
* Visible to the human: yes — it is the building colour, the game's core visual rule.
* Stable: 0 house reassignments and 0 destination group changes across 3,867 native decisions (12 S13 replicate episodes).
* Consistent houses↔destinations: all 24 captured houses and all 120 logged house spawns have a destination of the same group already present (or spawning in the same decision).
* Reuse / meaning of the ids: within an episode an id is never reassigned; across episodes the same small integers are reused as labels. The order of first appearance is **not** the id order (e.g. 0, 1, 3, 2 in 3 of 12 seeds) — ids are arbitrary labels and must not be encoded as numbers. At most 4 simultaneous groups were observed; capacity is 6.
* Oddity: in 2/12 episodes a 2×4 destination block has an upper 2×2 whose tiles report `contentType=Destination` with `groupIndex=null` (no content model). Treated as group-less in both encoders; documented, not resolved.
* Not exposed: scheduled future buildings (`CityPlanModel.scheduledBuildings`), demand internals, RNG. The v2 encoder uses only the per-tile `groupIndex` of currently visible structures and the episode's own history.

## 2. Observation-channel audit (`channel_audit.json`)

| channel | finding | severity | action |
|---|---|---|---|
| 11 has-group | **identity collapsed to a flag** (root cause) | **high** | fixed by v2 (channels appended; 11 kept) |
| 10 content ordinal | Carpark (96/96 capture tiles) and Tree (384) encode as 0 = empty land; the route endpoint carpark is invisible (only its stub edge is visible) | medium | not changed in S14 (would alter channel semantics and break S8-C equivalence); candidate for a later migration |
| 13 overcrowding | flag only; `overcrowdingTime` (distance to game over) is in the JSON but not encoded | medium | candidate extra channel later; not added now |
| 14 vehicles | native-only; synthetic envs always 0 | medium (sim/native mismatch) | documented |
| 4–7 roads | Active 0.667 vs Mothballed 1.0 are distinct → Active/Mothballed handling is **correct** (not a collapse); the ordinal order is odd; Pending never observed | low | none |
| 12 demand/50 | uses 0–0.2 of range (max demand 10) | low | none |
| g0 hour/24 | cumulative, unclipped (reaches 28.4) | low | none |
| g4 concrete ratio | ratio hides absolute count after top-ups | low | none |
| 0–3, 8–9, g1–3, g5–10 | correct / constant on this map; motorwayDirections, isUpgraded, destinationType unencoded | none/low | none |
| CNN | 7×7 receptive field per tile logit; only pooled features reach WAIT/value. Group channels can inform a tile only about houses/destinations within 3 tiles | architectural | unchanged; noted for later |

Nothing is present synthetically that the native policy cannot see (gate B6). Present natively but absent synthetically: vehicles, pending upgrade choices.

## 3. Encoding v2 (`obs_v2.py`)

* Layout: **v1 unchanged (15 tile channels + 11 globals) followed by 6 appended one-hot per-tile group-slot channels** → 21 channels, 777×21+11 = **16,328 floats**. Action space unchanged, Discrete(3111); masks unchanged. The 11,666-float layout is not preserved; this is an explicit migration, not a repurposing. (Fitting 6 one-hot channels into the old shape would have required overwriting channels 2/3/8/9 which are legitimately zero only on this map — rejected.)
* **Slots, not ids.** Slot = rank of a group's first appearance among visible structures in the current episode (tie-break: row-major position of its first tile). Properties: invariant under any consistent relabeling of native ids (tested); stable within an episode (never reassigned); new groups take the next free slot; a house gets the same slot as its destinations; derived from what is visible now plus the episode's own visual history (what a player remembers), nothing future. Raw integer ids are never encoded as values.
* The native v2 encoder `encode_native_v2(raw, tracker)` calls the **frozen v1 encoder** and appends slots from the plugin's per-tile `groupIndex`; the synthetic `MultiHouseEnvV2` (subclass of the untouched S13 env) applies the same rule to its own labels. S7 tasks have one group → `s7_v1_to_v2` maps every has-group tile to slot 0, so the frozen S7 evaluators can score a v2 model through an adapter.
* Receptive-field caveat (audit): with the S1 architecture, group channels help only locally; this phase does not change the network.

### Gate B — encoding correctness (9/9 pass, `gate_b.json`)
B1 relabeling invariance on 48 native states (48/48 identical) and along 4 whole synthetic episodes with relabeled native ids (0 differing steps) · B2 171 newly spawned houses all receive the slot of their destination group · B3 7,297 states with several destinations of one group: one slot each · B7 0 slot reassignments within episodes · B4 the first 15 channels of v2 are bitwise the S13 v1 observation across an episode containing 709 Mothballed-edge steps (Active/Mothballed semantics unchanged) · B5 deleting all future spawn events changes nothing in 180 compared observations · **B6 native/synthetic agreement: 644 synthetic states rendered as native-style JSON and passed through the frozen native v1 encoder + v2 slots reproduce the synthetic v2 observation bit-for-bit (644/644)** · B8 capacity 6 ≥ 4 observed.

## 4. Information loss of v1 (`info_loss.json`)
* Native: 2,954 of 3,867 decisions (76%) have ≥2 destination groups and ≥1 house. In such states the number of house→group assignments consistent with one v1 observation has median **64** (= G^H); v2 admits exactly one.
* Synthetic counterfactuals (S13 env, 22 states with ≥2 groups and an unconnected house): the house's group was changed to another visible group → the v1 observation is identical in 22/22, the v2 observation differs in 22/22, and the correct construction target (nearest same-group destination) changes in **22/22**; the first planned edge changes in 11/22 (routes often start the same way and fork later). Five examples with coordinates are in the JSON.
* v2 resolves exactly the house↔group assignment ambiguity. It does not remove other partial observability (demand internals, trips, future spawns, upgrade packages, model-less destination tiles, local receptive field).

## 5. Checkpoint migration (Gate C — 2/2 pass, `gate_c.json`)
Procedure (`models_v2.migrate_state_dict`): all 22 tensors copied by name; `mlp_extractor.trunk.0.weight` widened from (32,15,3,3) to (32,21,3,3) with **zeros** in input channels 15–20; bias copied; optimizer fresh. 73,416 → 75,144 parameters. Saved as `s8c_migrated_v2_step0.zip` (S8-C untouched, sha 85a9cb75…).
* C1: on 408 saved observations (48 native capture states + 360 synthetic multi-house states) the migrated network's 3,111 logits and value equal S8-C's **exactly (max |Δ| = 0.0)**; greedy action identical 408/408.
* C2: frozen S7 evaluators, deterministic, validation Gap-7 (60 tasks) and Gap-10 (39): row-by-row identical outcomes through the adapter (success 0.833 / 0.769 for both).
* Stochastic step-0 evaluation on the multi-house validation set (30 episodes, same seeds): frozen S8-C on the v1 env and the migrated checkpoint on the v2 env produce identical metrics (return mean 33.97, median 12; 9/30 survive; 41/129 new houses within 60 decisions; 47.5 destroyed connections median).

## 6. Gate D — feasibility training (≤10,240 decisions, synthetic, one seed)
Run: `experiments/s14_train.py --decisions 10240 --seed 14` (`runs/feasibility_s14/`). Init = migrated checkpoint; S1/S8 PPO hyper-parameters (512/256/4, target_kl 0.03, lr 3e-4, ent 0.01, γ 0.99); training scenarios = the 18 scenarios of game seeds 500000/523757/531676/547514/555433/571271; validation = 6 scenarios of game seeds 507919/579190 × 5 env seeds, stochastic policy, fixed torch seeds; S7 Gap-7/Gap-10 validation retention (stochastic and deterministic, seed 0). Wall-clock 389 s total including all evaluations. Abort conditions never triggered (KL ≤ 0.06, all statistics finite).

**Throughput (2 threads):** rollouts 273 decisions/s; PPO update 5.4 s per 512 decisions (target-KL early stops: 3–4 of 8 minibatch passes per rollout); **70 decisions/s end-to-end → ≈1,430 s (24 min) per 100k decisions** excluding evaluation.

**Multi-house validation (30 episodes each; same seeds):**

| checkpoint | return mean / median | deliveries median | decisions median | survived 600 | new houses ≤60 | connected frac by quarter | zero-concrete | destroyed conn. (median) | plain ADDs |
|---|---|---|---|---|---|---|---|---|---|
| frozen S8-C (v1 env) | 34.0 / 12 | 22 | 291 | 9/30 | 41/129 (32%) | .35 .40 .46 .51 | 0.62 | 47.5 | 70.5 |
| migrated step 0 (v2 env) | **identical to the row above** (representation alone changes nothing: zero weights) | | | | | | | | |
| trained 5,120 | 44.2 / 24.5 | 34.5 | 350 | 8/30 | 68/147 (46%) | .35 .56 .57 .57 | 0.34 | 39 | 56.5 |
| trained 10,240 | 24.5 / 11.5 | 21.5 | 281 | 5/30 | 54/134 (40%) | .23 .48 .47 .52 | 0.25 | 24 | 51 |

Reading: at 5k decisions every primary metric moved in the intended direction (new-house connection +14 points, deliveries +57%, destroyed connections −18%, zero-concrete fraction halved); at 10k the delivery/return gains were gone while the churn and concrete improvements continued (destroyed connections −49%, zero-concrete 0.62 → 0.25). With 30 episodes and the S13a noise scale these are directions, not established effects; a 10k run says nothing about convergence.

**Learning signal:** value explained variance rose from ≈0 to 0.5–0.9; policy entropy fell (−1.36 → −0.74). The first-layer weights on the six new group channels stayed almost untouched (mean |w| 0.005 vs 0.101 on the v1 channels; max 0.035): **in 10k decisions the network barely began to use the group information** — the measured changes come from re-weighting what v1 already showed (WAIT preference, concrete use), not from group-aware routing.

**S7 retention — FAILED (`s7_forgetting_diagnosis.json`):**

| checkpoint | Gap-7 sto / det | Gap-10 sto / det | deterministic action share on Gap-7 |
|---|---|---|---|
| frozen S8-C / migrated | 0.93 / 0.83 | 0.85 / 0.77 | WAIT 15%, ADD 79% |
| trained 5,120 | 0.73 / **0.02** | 0.54 / **0.00** | WAIT 90%, ADD 6% |
| trained 10,240 | 0.68 / **0.02** | 0.36 / **0.00** | WAIT 88%, ADD 8% |

Mechanism: the multi-house reward (deliveries only, −10 at game over) never pays for construction directly and does pay for not wasting concrete, so the WAIT logit rose until WAIT became the greedy action almost everywhere; the stochastic policy still completes 68–73% of Gap-7 routes (it samples ADDs 42–49% of the time) but Gap-10 fell to 36%. This meets the brief's "severe catastrophic forgetting" stop condition: the run was already at its 10,240-decision cap, nothing further was launched, and the S14 preregistered eligibility rule (S7 validation ≥ 0.90 × step 0) would reject both checkpoints.

## 7. Recommendation
**Gates:** A pass (identity is accessible, stable, human-visible, appropriate) · B pass (9/9) · C pass (exact equivalence) · **D: feasible mechanically (stable, 24 min / 100k) but the continuation as configured fails the retention criterion**.

**Do not run the full 100k continuation with the Gate D recipe.** Two things must change first, both outside this phase's authorisation:
1. *Retention.* Options, in order of preference: (a) mixed training — interleave S7 long-route tasks (v2 observation, slot 0) with multi-house scenarios in the rollout stream, so route construction keeps being rewarded; (b) a KL/behaviour-cloning penalty toward the migrated step-0 policy; (c) freezing the conv trunk and training only the group-channel columns plus heads. Each is a training-recipe change that needs approval; none is a reward or observation change.
2. *Group usage.* The group channels were almost unused after 10k decisions; the receptive-field limitation (audit) means they can only act within 3 tiles. Expect slow uptake; a 100k run should preregister a "group-channel weight growth" diagnostic and the S7 eligibility rule, and compare against a v1-observation control run with the same recipe to separate representation from training effects (the step-0 comparison here shows the representation alone is neutral by construction).

What this phase established: the information bottleneck was real and was Python-side; v2 fixes it with an exact, label-free, leak-free encoding validated natively and synthetically; S8-C migrates with zero loss; the environment trains stably at ≈70 decisions/s. What it did not establish: that the policy can exploit group identity, or that multi-house training preserves long-route skill under the current recipe.

**STOPPED. No native evaluation, no further training.**
