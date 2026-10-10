# S17 — CNN + Global Self-Attention: Direct Architectural Benchmark

Scope kept: supervised only; no PPO, no native runs, no reward/environment change, no production policy replacement; one architecture configuration; no hyperparameter search; all S1–S16 checkpoints untouched. Compute: 521 s main benchmark + 229 s fix/re-verification = **750 s (12.5 min)** of the 15-minute cap.
Artifacts: `research/s17_global_attention/` — `results.json`, `s8c_migrated_v3_attention_step0.zip` (new file), `model_{A,B,C}.pt`, `model_B_zero_init.pt`, `fix_check.log`. Code: `python/synthetic_rl/models_v3.py` (new), `experiments/s17_benchmark.py`, `experiments/s17_fix_check.py`. S16 data/labels/splits reused unchanged.

## 1. Architecture and insertion point
```
v2 obs (777×21 + 11) ─► S8-C trunk: Conv3×3(21→32)·ReLU·Conv3×3(32→64)·ReLU·Conv3×3(64→64)·ReLU ─► F ∈ ℝ^{64×21×37}
      F ─► tokens (777×64) + learned positional embedding (777×64) ─► LayerNorm ─► 4-head self-attention (dim 64, head dim 16, torch SDPA)
        ─► output projection W_out (zero-initialised) ─► F' = F + W_out(attn)     [residual; F' keeps the 21×37 layout]
      F' ─► existing 1×1 tile head (4 logits/tile = 3,108 tile actions)  and  [mean-pool, max-pool, 11 globals] ─► existing WAIT/upgrade and value heads
```
Insertion: between the trunk's last ReLU and all heads, so per-tile logits and the pooled heads both see the attended features; nothing is globally pooled before action scoring. Masks and Discrete(3111) unchanged. Parameters: 75,144 (v2 policy) → 141,640 (+66,496: positional 49,728; qkv 12,480; out 4,160; LN 128).

## 2. Migration and equivalence
Procedure: S8-C → v2 (S14 zero-widening of conv1) → v3 by copying all 16 v2 tensors by name; the 7 new attention tensors keep their init with **W_out = 0**. First attempt failed (max |Δlogit| 2.16 on 348 observations) because SB3's `ortho_init` runs inside `_build` and overwrote the zero projection; fixed by re-zeroing after `super()._build` (`models_v3.py`). After the fix, on 348 saved observations (48 native capture states, 240 S7 validation states, 60 S13 multi-house states): **max |Δlogit| = 0.0, max |Δvalue| = 0.0, argmax identical 348/348**. Saved as `s8c_migrated_v3_attention_step0.zip`.
Consequence for the benchmark: the three supervised models in §4 were built before the fix, so B and C started with the *same* pretrained trunk/head but a small random (orthogonal) residual rather than zero; B was then re-trained from the true zero residual (`B_zero_residual_init`) — results are the same within noise (§4), so the conclusion does not depend on this.

## 3. Attention checks (random non-zero W_out, before training)
1. Information propagates: gradient of the logit at tile (2,2) reaches tiles in rows ≥18 / cols ≥30 (|∇| sum 0.22; zero for the pure CNN).
2. Attention changes spatial logits: max |Δ| 0.64 between residual on/off.
3. Gradients reach every attention parameter (|∇| sums: qkv 384, out 785, pos 7.0, LN 4.8/6.5).
4. Group information reaches the block: group channels enter conv1 which feeds the tokens — but ∇ w.r.t. the group channels at that probe was **0.0**: the *migrated* conv1 has zero weights on channels 15–20 (S14 widening), so at init no group information reaches attention until those columns are trained. (They are trainable; after training the group ablation changes predictions, §5.)
5. B and C share code, seed and init; C zeroes channels 15–20 before the trunk.
6. Counterfactual twins receive different predictions from trained B in **41/102** pairs (the twins differ only in the query house's slot channels).
7. Inputs: forward() receives the v2 observation only; labels/masks are used by the loss and the argmax.

## 4. Controlled supervised benchmark (S16 data: train 1,068 / val 187 / test 502; 102 CF pairs; slot 2 held out; same loss/optimiser as S16: Adam 1e-3, batch 64, ≤30 epochs, 210 s cap per model, best-val checkpoint; all trunks initialised from the migrated S8-C)

| test subset (n) | A existing CNN (pretrained) | B CNN + attention | C attention, group-blind | B zero-residual re-run | S16 A (scratch) |
|---|---|---|---|---|---|
| all (502) | 0.677 | 0.647 | 0.673 | 0.659 | 0.669 |
| **colour-dependent (204)** | **0.382** | **0.373** | **0.392** | **0.402** | 0.397 |
| **CF pairs both correct (102)** | **0.010** | **0.029** | **0.000** | **0.029** | 0.020 |
| CF twins (102) | 0.147 | 0.167 | 0.216 | 0.225 | 0.118 |
| beyond RF (352) | 0.682 | 0.645 | 0.690 | 0.668 | 0.659 |
| unseen slot (40) | 0.225 | 0.175 | 0.275 | 0.250 | 0.150 |
| decoy chosen, colour-dependent | 0.40 | 0.40 | 0.39 | 0.40 | 0.40 |
| epochs / train s | 30 / 67 | 23 / 220 (cap) | 22 / 215 (cap) | 20 / 221 (cap) | 30 / 73 |

Paired bootstrap (95 % CI, identical test examples): B − A colour-dependent −0.010 [−0.074, 0.054], all −0.030 [−0.066, 0.004], beyond-RF −0.037 [−0.080, 0.003]; B − C colour-dependent −0.020 [−0.069, 0.029], all −0.026 [−0.052, −0.002]; A − C colour-dependent −0.010 [−0.074, 0.059]. Training/validation: all three reach val ≈0.60–0.68 overall and ≈0.36–0.43 colour-dependent; B's train loss falls faster (0.31 vs 0.41) with no validation gain — it memorises geometry.

## 5. Functional ablations of trained B (inference only)

| configuration | all | colour-dependent | CF pairs | beyond RF | unseen slot |
|---|---|---|---|---|---|
| normal | 0.647 | 0.373 | 0.029 | 0.645 | 0.175 |
| attention residual disabled | 0.602 | 0.343 | 0.020 | 0.594 | 0.250 |
| group channels zeroed | 0.671 | 0.402 | 0.000 | 0.679 | 0.250 |
Paired B − B(no attention): all +0.046 [0.004, 0.086], colour-dependent +0.029 [−0.039, 0.093]; B − B(groups zeroed): all −0.024 [−0.046, −0.002], colour-dependent −0.029 [−0.074, 0.010]. Disabling attention costs ~4.6 points of *geometric* accuracy (the block was used — for geometry); zeroing the group channels makes B slightly **better**. Attention did not learn colour matching.

## 6. Cost
Inference (CPU, 4 threads): v2 CNN 1.10 ms/obs (batch 1), 0.56 ms (batch 64); hybrid 4.91 ms (batch 1), 2.60 ms (batch 64) → **4.5× slower**. Training throughput on the supervised task: 481 samples/s (CNN) vs 112 samples/s (hybrid), 4.3× slower; at the S15 rollout rate (~230 decisions/s) the hybrid would roughly halve PPO throughput. Memory: attention scores 4×777×777 per sample (≈2.4 M floats; SDPA avoids materialising the full batch matrix); RSS not measured (psutil unavailable).

## 7. Failure examples (B, colour-dependent test)
* geom 3: house (−2,−2), slots [1,0], correct cost 3 vs decoy 1 → predicts `ADD −2,1,S` toward the decoy (same as A and C).
* geom 22 and its twin: equal costs; B predicts the same unrelated edge `ADD 3,3,E` for both twins — the colour swap changes nothing.
More in `results.json → failure_examples_B`.

## 8. Decision
Screening targets: ≥ +15 points over the existing CNN on colour-dependent accuracy (needed ≥ 0.53; got 0.37–0.40), ≥ 25 % CF-pair success (got 3 %), above the group-blind control (no), measurable drop when groups are ablated (no — slight gain). **Outcome B — attention fails on this benchmark. NO-GO for an S18 attention migration.** The hybrid is mechanically sound (exact equivalence at init, global gradient flow, attention used for geometry) and costs 4.5× inference time, but with 486 colour-dependent training samples it learns no colour association, exactly like the plain and dilated CNNs in S16. Not a disproof of attention in general: the limiting factor across S16/S17 is that the slot one-hot signal is tiny and the geometric prior explains ~67 % of labels, so none of three architectures found the matching rule under this data/compute budget.

**One concrete alternative (not run):** explicit same-group destination features derived from visible information — per tile, "is a same-slot destination present / direction and normalised distance to the nearest same-slot destination from this tile" — which makes the matching rule an input rather than something to discover; re-run the S16 diagnostic with it before any PPO. (S17-G is testing the graph-with-semantic-edges variant of this idea in parallel.)

**STOPPED. No S18 work started.**
