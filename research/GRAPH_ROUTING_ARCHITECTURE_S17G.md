# S17-G — Graph neural network benchmark for colour-aware routing (supervised, isolated)

Scope kept: supervised only; no PPO, no native runs, no change to any existing file. Isolated code: `research/s17_graph/code/s17g.py` (single script; imports `experiments/s16_train.py` read-only for `load / set_loss / metrics / paired_bootstrap`). Artifacts: `research/s17_graph/` — `results.json`, `conversion_checks.json`, `model_G.pt`, `model_G0.pt`, `run_log.txt`. Data: the frozen S16 dataset (`research/s16_colour_routing/{train,val,test}.{npz,json}`), unchanged.
Compute actually used: **644 s total** (conversion of all 1,832 graphs 0.8 s; training G 309 s / G0 304 s; test evaluation + 3 ablations + latency probes ≈ 20 s) — **10.7 min of the 15-min cap**, CPU, 4 threads, one seed (17), no hyperparameter sweep. Another experiment (S17 CNN+attention) may have shared the CPU.

## 1. Graph schema (deterministic function of the v2 observation only — `graph_from_obs(obs_vec)`)

* **Nodes** = the 777 tiles (37×21). 22 features: normalised x, y; buildable, water, mountain, rail; the four raw road channels N/E/S/W (0 / 0.667 Active / 1.0 Mothballed); house flag (content = 0.5), destination flag (content = 1.0); has-group, demand, overcrowding, vehicles; the 6 one-hot group-slot channels (15..20). Carparks are plain road tiles (content 0, stub road edge) as in the observation.
* **Physical edges** = every in-bounds 4-neighbour pair, one representative per unordered pair exactly as the native action space enumerates it: `(x,y,'E')` joins (x,y)–(x+1,y), `(x,y,'S')` joins (x,y)–(x,y−1) (South = y−1). 1,496 candidate edges (36·21 + 37·20), each mapped 1:1 to the logit index `tile_index(x,y)·4 + {0 for E, 2 for S}` = native ADD action id − 1. Per-sample edge features (4): Active road present (max of the two endpoint channels), Mothballed road present, potential construction (both endpoints non-water, non-mountain, no road), both-non-water. Message passing uses both directions of every edge (2,992 directed) with a 4-way sender-direction one-hot; messages across edges with a water endpoint are gated to zero (so the message graph is the "non-water-both" adjacency), while every candidate edge is still scored (the native mask allows ADD on water edges = bridges; in this water-free dataset no label touches water).
* **Semantic edges** = every visible house tile ↔ every visible destination tile with the same group slot (both directions, mean-aggregated, separate weights, never scored). Mean 9.7–9.9 per graph (2–3 houses × 4 destination tiles of the matching 2×2 block). Nothing else enters: no meta, label, mask, cost, query or oracle field is read (the function signature takes only the observation vector; the feature list is enumerated in `conversion_checks.json → graph_inputs`).

## 2. Model (plain PyTorch, scatter-add on flattened batches; **86,209 parameters**; S16 CNN A: 61,764)

node projection Linear(22→64)+ReLU → 3 message-passing layers, each: physical message `ReLU(W_p[h_v ‖ edge feats ‖ direction one-hot])` summed over gated neighbours (/4), semantic message `ReLU(W_s h_v)` averaged over semantic neighbours, residual update `h += MLP([h ‖ agg_phys ‖ agg_sem])` (64→64→64) → edge head `MLP([h_u ‖ h_v ‖ edge feats ‖ E/S one-hot])` (134→64→1) written into a (N, 3108) logit tensor at the candidate ADD positions, −1e9 elsewhere (REMOVE/non-candidate positions never win). Loss, masking, prediction and metrics are S16's unchanged (`set_loss` = −log Σ_valid softmax over legal tile actions; the S16 "add_mask" is `mask[1:3109]`, i.e. it also contains legal REMOVE slots — the CNNs could pick those, the GNN cannot; labels are ADD-only and this had no measurable effect since no model's failures were REMOVE picks).
**G** (group-aware): slot features + semantic edges. **G0** (group-blind): identical architecture and seed, slot channels zeroed in node features and semantic edges removed (has-group flag stays, as in S16 model C).
Receptive field: 3 hops along physical edges plus the semantic shortcut (house → destination in one hop). The query frontier is 0–4 road steps from the house (S16 `partial`), so for the longest partial routes the frontier lies outside the hop range of the house's semantic edge — a known limitation of the "2–3 layers" spec, see §7.

## 3. Correctness checks (all automatic, `conversion_checks.json`)

| check | train | val | test |
|---|---|---|---|
| 1. semantic edges connect only visible house/destination tiles of the same slot (independent recount, exact edge count) | ✓ | ✓ | ✓ |
| 2. no oracle info: graph = pure function of the obs vector; deterministic re-conversion identical; features enumerated | ✓ | ✓ | ✓ |
| 3. candidate edges obey the spatial conventions (E: x+1, S: y−1, in bounds, unique, `decode_action` round-trip) | ✓ (1,496 edges) | | |
| 4. labels map to graph outputs: label ADD actions on candidate edges | 1,502 / 1,502 | 255 / 255 | 692 / 692 |
| 4b. legal tile actions on candidate edges (the remainder are legal REMOVE slots in S16's mask) | 328,131 / 328,399 | 54,658 / 54,703 | 145,509 / 145,630 |
| 5. counterfactual twins alter semantic relations (edge sets differ / slot features differ / physical graph identical) | 243/243/243 of 243 | 37/37/37 of 37 | 102/102/102 of 102 |
| 6. conversion does not reveal labels | by construction (check 2); twins share the identical physical graph and differ only in the colour relation | | |
| 7. test geometry unseen | 0 test observations identical to any train observation (splits by geometry seed in S16) | | |

Conversion time: **0.41–0.48 ms per sample** (numpy, single thread); 0.8 s for all three splits.

## 4. Training settings
Same as S16: Adam 1e-3, batch 64, seed fixed (17), best-val checkpoint on overall val accuracy, ≤30 epochs, **per-model time cap 300 s** (needed so that two arms + evaluation fit in 15 min). Train = the 1,068 S16 train samples with query slot ≠ 2 (75 slot-2 samples dropped = held-out slot). Achieved: G 20 epochs (309 s, 69 samples/s), G0 22 epochs (304 s, 77 samples/s). Val curves (`results.json → models.*.curve`): both arms reach ~0.68–0.72 val accuracy by epoch 6 and plateau (best val 0.717 for both); G's val CF-pair accuracy fluctuates 0–0.11 (37 pairs) with no upward trend; G0's is 0 throughout.

## 5. Results — test (502 samples; S16 CNN numbers from `COLOUR_ROUTING_ARCHITECTURE_S16.md` for reference)

| test subset (n) | **G** GNN group-aware | **G0** GNN group-blind | S16 A existing CNN | S16 C blind CNN |
|---|---|---|---|---|
| all (502) valid-set accuracy | 0.697 | 0.679 | 0.669 | 0.697 |
| within RF (150) | 0.707 | 0.687 | 0.693 | 0.673 |
| beyond RF (352) | 0.693 | 0.676 | 0.659 | 0.707 |
| decoy nearer (257) | 0.728 | 0.735 | 0.696 | 0.693 |
| **colour-dependent (204)** | **0.412** | **0.377** | **0.397** | **0.412** |
| originals of CF pairs (102) | 0.647 | 0.627 | 0.676 | 0.588 |
| CF twins (102) | 0.176 | 0.127 | 0.118 | 0.235 |
| **CF pairs both correct (102)** | **0.020** (2 pairs) | **0.000** | **0.020** | 0.000 |
| decoy chosen on colour-dependent | 0.38 | 0.38 | 0.40 | 0.41 |
| unseen slot 2 (40) | 0.275 | 0.250 | 0.150 | 0.350 |

Paired bootstrap **G − G0** (95 % CI, identical examples): all +0.018 [−0.010, +0.046]; colour-dependent +0.034 [−0.010, +0.078]; beyond-RF +0.017 [−0.014, +0.048]; within-RF +0.020 [−0.033, +0.073]; unseen slot +0.025 [−0.050, +0.100]. No interval excludes zero.

Counterfactual results: G solves 2 of 102 pairs; on the recoloured twins it keeps the original's geometric answer (CF-twin accuracy 0.176, decoy-target rate on twins 0.63 — the same signature as the S16 CNNs). G0, which cannot see colour, is forced to 0 pairs, exactly as S16 model C.

## 6. Inference-only ablations of trained G (test)

| ablation | all | colour-dep. | beyond RF | unseen slot | CF pairs | Δ colour-dep. vs G (paired CI) | predictions changed (of 502 / of 204 cd) |
|---|---|---|---|---|---|---|---|
| G (reference) | 0.697 | 0.412 | 0.693 | 0.275 | 0.020 | — | — |
| (a) semantic edges removed | 0.606 | 0.328 | 0.594 | 0.175 | 0.000 | −0.083 [−0.132, −0.034] | 139 / 70 |
| (b) group features zeroed (edges kept) | 0.713 | 0.426 | 0.710 | 0.300 | 0.020 | +0.015 [−0.015, +0.044] | 32 / 18 |
| (c) both | 0.665 | 0.358 | 0.656 | 0.175 | 0.000 | −0.054 [−0.108, 0.000] | 115 / 65 |

Reading: the trained G *does* use the semantic edges — removing them changes 28 % of predictions and costs 8–10 points everywhere, not only on colour-dependent scenes — but it uses them as a **generic "there is a house/destination here" structural signal** (the one-hot slot features themselves are irrelevant: zeroing them changes 32 predictions and nothing significant), not as the colour-matching rule: the ablation drop is the same on colour-independent scenes (−0.09 overall) as on colour-dependent ones (−0.08), and CF-pair success is 2 % with or without the colour information. Ablation (c) is interestingly *less* harmful than (a) alone, i.e. the network learned a joint statistic of the two signals rather than a matching rule.

## 7. Costs
| quantity | value |
|---|---|
| parameters | 86,209 (S16 CNN A 61,764) |
| graph conversion | 0.46 ms / sample (numpy) |
| inference latency, batch 1 | 7.4 ms / sample |
| inference latency, batch 64 | 383 ms / batch = 6.0 ms / sample |
| training throughput | G 69 samples/s, G0 77 samples/s (4 threads; ≈ 14.5 s / epoch) |
| S16 CNN A for comparison | 30 epochs in 73 s ≈ 440 samples/s |

The GNN is ~6× slower to train and roughly an order of magnitude slower at inference than the 7×7 CNN for the same data (191k directed messages × 64 hidden per sample); an RL rollout at ≥ 6 ms per observation on CPU would be a real cost.

## 8. Failure examples (test, colour-dependent)
* geom 3 (same as the S16 failure): query house (−2,−2), correct cost 3, decoy cost 1 — G builds `ADD −2,1,S` toward the decoy (identical to CNNs A/B/C); G0 picks an unrelated frontier edge `ADD 0,0,S` on both twins.
* geom 22 (costs equal 4 vs 4): G answers `ADD 1,3,S` (the decoy direction) on the original; G0 gives `ADD 3,4,E` on both twins — identical answers regardless of colour.
* geom 24 (CF twin, correct cost 2 vs decoy 9): G still prefers the far decoy direction `ADD 0,0,E` instead of `ADD 2,0,S` — the twin's answer is carried over from the original.
* geom 26 (CF twin, costs 6 vs 6): G predicts `ADD −6,−5,E`, one tile off the valid `ADD −7,−5,E`, toward the decoy.
More in `results.json → models.*.failure_examples`.

## 9. Decision — screening thresholds
| threshold | required | observed | pass |
|---|---|---|---|
| colour-dependent accuracy ≥ S16 CNN + 15 points | ≥ 0.547 | 0.412 (+1.5 points) | **no** |
| counterfactual pair success ≥ 25 % | ≥ 0.25 | 0.020 (2/102) | **no** |
| clear advantage over G0 | CI excludes 0 | +0.034 [−0.010, +0.078] on colour-dependent | **no** |

**Recommendation: NO-GO** for the GNN architecture as the answer to the colour-routing problem. Making the house↔destination colour relation an explicit graph edge is *not* sufficient: with the S16 data budget (486 colour-dependent training samples) and a strong nearest-destination prior, a message-passing network with separate semantic weights lands exactly where the CNNs did (colour-dependent ≈ 0.41, CF pairs 2 %, held-out slot 0.28). The semantic edges are used, but as an extra structural feature, not as a matching rule — the same learning failure as S16 Outcome C, now shown to be architecture-independent across CNN, dilated CNN and GNN.

Caveats (as in S16): one seed; 102 test pairs (CI half-width ≈ ±0.09 on pair accuracy); 20–22 epochs under the time cap (val had plateaued by epoch 6); 3-hop receptive field means the frontier of a 4-step partial route cannot "see" the house's semantic edge — a 4–5-layer or road-walk variant would fix that but the within-RF subset (where the frontier is near both the house and a destination) shows the same colour-blindness (0.707 overall, colour-dependent failures identical to the CNN), so hop range is not the explanation.

What would still be worth testing (not done here, not a sweep): (i) a *colour-only* training regime (S16 follow-up showed the first weak colour signal at 12 % pairs for the 7×7 CNN); (ii) a slot-equivariant derived observation (S16 §7 recommendation) which gives the matching rule as an input feature rather than asking the network to discover it from ~500 examples; (iii) more colour-dependent data. Since GO was not reached, no mapping of node/edge outputs to the 3,111 primitive actions is proposed beyond what this benchmark already implements (candidate physical edge ↔ ADD action id; REMOVE/WAIT/upgrades unmodelled).

**STOPPED. No PPO, no production integration.**
