"""Assembles research/SYNTHETIC_RL_LAB.md from stored artifacts (validation results, benchmark, S1 results section)."""
import glob
import json
import os

ROOT = r"C:\Users\TKA\Desktop\mini moto"
LAB = os.path.join(ROOT, "research", "synthetic_lab")
val = json.load(open(os.path.join(LAB, "validation_results.json")))
bench = json.load(open(os.path.join(LAB, "throughput_benchmark.json")))
s1 = open(os.path.join(LAB, "s1_results_section.md"), encoding="utf-8").read()
hand = open(os.path.join(LAB, "hand_example.txt"), encoding="utf-8").read()
hashes = open(os.path.join(LAB, "manifests", "gap1_SHA256.txt")).read().strip()
runs = sorted(glob.glob(os.path.join(ROOT, "runs", "*syn_s1_*")))

L = []
A = L.append
A("# Synthetic RL lab\n")
A("Workflow: **hypothesis -> minimal synthetic experiment -> controlled comparison -> native Mini Motorways validation.** The synthetic simulator is an experimental "
  "instrument, not a replacement or an approximation of the game. Nothing here was run against the native game; no native PPO experiment was launched for this study "
  "and all earlier native experiment artifacts are untouched.\n")
A("## 1. Minimal simulator (`python/synthetic_rl/`)")
A("""Models only: a 2D tile grid (the native 37 x 21 grid, x -18..18, y -10..10), one house with a driveway edge, one matching 2 x 2 destination with a carpark stub edge, road edges, legal ADD / REMOVE / WAIT, connectivity, episode termination, configurable reward and task difficulty. No Unity, IPC, traffic, spawning, animation or rendering. Pure Python + numpy.

| rule | implementation |
|---|---|
| edges | undirected, canonical `(x, y, 'E'|'S')`; `S` connects `(x,y)` and `(x,y-1)` (native: South = y-1) |
| ADD(x,y,d) legal | both endpoint tiles inside the task's reach window and free (not house / destination / carpark), edge absent, concrete > 0 |
| REMOVE(x,y,d) legal | edge exists, both endpoints free (stub edges can never be removed), and tile `(x,y)` has exactly ONE road edge (the native rule: ClearTileExplicit clears a whole tile, so a single-edge removal is only offered where it equals that) |
| WAIT | always legal, except while a delayed reward is pending (then WAIT-only, like the forced upgrade choice natively) |
| upgrade slots (3109, 3110) | always illegal (no upgrades modelled) |
| connected | house start tile reaches the goal tile (the carpark stub's tile) through road edges |
| construction window | 20 decisions; SUCCESS when connected; DESTRUCTIVE FAILURE if an edge of the original network is removed; TIMEOUT (truncated) after 20 decisions |
| concrete | 25 total; the pre-built edges consume it; ADD costs 1, REMOVE refunds 1 (native accounting is more involved) |

Tasks (`generator.py`): Gap-N = a path start -> goal with N edges left out; any order of adding them completes the connection. **Solution uniqueness** is verified at generation: the minimum number of new edges to connect house and destination equals N and there is exactly one shortest edge set (counted on the contracted component graph), so no alternative or accidental solution exists. Layouts vary in window size/position anywhere on the map, house position and driveway side, destination position, carpark side and stub direction, path shape and length (N + 2 .. 14 edges), gap positions. Splits are disjoint by base layout (house, destination, path): Gap-1 2,000 / 200 / 400 (train / validation / test), Gap-2 and Gap-3 1,000 / 100 / 200, frozen in `research/synthetic_lab/manifests/` with SHA-256:

```
""" + hashes + """
```
""")
A("## 2. Interface mapping (inspected from the real implementation, not hard-coded)")
A("""Sources: `python/mini_motorways_env.py` (`NUM_TILE_FEATURES = 15`, `NUM_GLOBAL_FEATURES = 11`, `_ROAD_STATE_ORDINAL`, `_CONTENT_TYPE_ORDINAL`, `encode_observation`) and `Entrypoint.cs` (`ActionIndexToDescriptor` / `DescriptorToActionIndex`, observation bounds = action bounds = xMin -18, yMin -10, 37 x 21). `layout.py` **derives** the sizes: observation = 37*21*15 + 11 = **11,666**, actions = 1 + 37*21*4 + 2 upgrade slots = **3,111**; both asserted in the tests.

* Observation: tiles row-major (y outer, x inner), 15 features each, then 11 globals - identical layout, so the same flat vector feeds MLPs and the CNN's tile-tensor reshape (15, 21, 37).
* Actions: 0 = WAIT; `1 + tile*4 + dir*2 + kind` with dir E=0 / S=1 and kind ADD=0 / REMOVE=1; the two upgrade slots follow. Gymnasium API, `action_masks()` (MaskablePPO convention), `reset(options={'task_index'})`, `info['appliedAction' / 'accepted' / 'rewardComponents']` as the native env.
* Feature table (verified, see check 9; fields the sandbox does not model carry the neutral value 0 - **no fake gameplay signals**):

| native feature | synthetic value |
|---|---|
| 0 buildable | 1 on every tile (neutral: no terrain) |
| 1-3 water / mountain / rail | 0 |
| 4-7 road N/E/S/W | Active/3 = 0.667 on live edges (both endpoint tiles), else 0 |
| 8-9 traffic light / roundabout | 0 |
| 10 content | House 0.5; Destination 1.0 (4 footprint tiles); the carpark tile is content 0 exactly as natively (the native encoder maps unknown content types, incl. Carpark, to 0) |
| 11 group flag | 1 on house and destination tiles |
| 12-14 demand, overcrowding, vehicles | 0 |
| globals | hour = 1.2 t mod 24 (/24), day, week 0, score/1000, concrete ratio; other resource ratios and pending upgrades 0 |
""")
A("## 3. Validation before any training\n")
A("| # | check | result |\n|---|---|---|")
for k, v in val.items():
    if k.startswith("_"):
        continue
    A(f"| {k.split('_')[0]} | {k} | {'PASS' if v['passed'] else 'FAIL'} - {v['detail'][:220]} |")
for k, v in val.items():
    if k.startswith("_info"):
        A(f"\nInformational: {v['solved']}/{v['of']} Gap-1 tasks are solvable from the observation alone when roads are allowed to pass beside the destination ({v['note']}).")
A("\nAll checks run via `python -m synthetic_rl.tests`; results in `research/synthetic_lab/validation_results.json`.\n")
A("**A finding about the native interface** (made while debugging check 8): the native observation encodes the carpark tile as content 0 (`_CONTENT_TYPE_ORDINAL.get(...)` default), so a road that merely passes beside a destination is indistinguishable from its real entry. With unconstrained layouts an observation-only solver solves only ~85% of Gap-1 tasks. The sandbox generator therefore keeps every other road tile off the destination's neighbourhood so the task is inferable; the native game does not provide that guarantee. This is a representation question worth testing natively.\n")
A("### Hand-inspectable example\n```\n" + hand + "\n```\n")
A("## 4. Hypotheses and falsification criteria")
A("""| id | hypothesis | experiment (one variable at a time) | falsified if |
|---|---|---|---|
| H1 | a flat MLP cannot learn location-invariant gap completion; a spatial CNN can | S1: MLP vs spatial CNN, Gap-1, completion reward, native-compatible 3,111 actions | the CNN does not beat the MLP / random on held-out layouts at equal budget and seeds, or the CNN's gain is explained by memorised action indices |
| H2 | the large global action space makes exploration inefficient | native-compatible vs compact task-local action space (diagnostic control only) | the compact space gives no improvement to the architecture that already learns |
| H3 | delayed / sparse reward blocks learning | immediate +1 vs +1 after a WAIT continuation vs a native-like sparse stream (implemented in the env: `reward_mode`) | learning curves are indistinguishable across reward timing |
| H4 | multi-step construction is harder than one edit | Gap-1 vs Gap-2 vs Gap-N (manifests generated) | completion does not fall with N / policies that solve N=1 solve N=2 zero-shot |

Only S1 was run (as instructed). Success criteria for S1: held-out completion clearly above RANDOM at equal budget in >= 3 of 3 seeds, first-action/rank evidence of targeting (not WAIT/ADD frequency), and success on tasks whose correct action index never appears in training.
""")
A("## 5. Experimental configuration (S1)")
A("""`experiments/configs/synthetic_s1_{mlp,cnn}_s{11,12,13}.yaml` run through `python -m experiments.synthetic_run --config ...` (harness: immutable run directory, resolved config, metadata with source and library hashes, checkpoint index with SHA-256, per-episode / training CSVs, validation per checkpoint, checkpoint selection on VALIDATION only, one TEST evaluation of the selected checkpoint).
* Task: synthetic Gap-1, reward = +1 on the connecting transition (episode terminates), 20-decision construction window, 3,111-action native-compatible space with native-style masks.
* Algorithm: MaskablePPO with the hyperparameters of the frozen native Baseline-0 (n_steps 512, batch 256, 4 epochs, target_kl 0.03, gamma 0.99, lr 3e-4, ent_coef 0.01, gae_lambda 0.95, clip 0.2). Budget: 50,000 decisions per run (a short pilot), checkpoints every 10k plus step 0; 3 training seeds (11, 12, 13) per architecture, identical across architectures.
* Flat MLP: SB3 `MlpPolicy` (64-64) on the 11,666-vector, as used natively. Spatial CNN: `synthetic_rl/models.py::SpatialMaskablePolicy` - the same 11,666 observation is reshaped to (15, 21, 37) + 11 globals; three 3x3 convolutions at full resolution, a 1x1 per-tile head producing the four native logits per tile (E-add, E-remove, S-add, S-remove) laid out exactly as action indices 1..3108, WAIT and the two upgrade slots from a pooled-features + globals branch; the native legal mask is applied by MaskablePPO. 73,416 parameters (the MLP has 1.7 M). The class depends only on the documented observation/action layout, so it can be plugged into the native training stack unchanged (architecture transfer); learned weights are not claimed to transfer.
* Runs: """ + ", ".join(f"`{os.path.basename(r)}`" for r in runs) + "\n")
A("## 6. Throughput\n")
A("| measurement (single process, otherwise idle CPU) | steps / second |\n|---|---|")
for k, v in bench.items():
    A(f"| {k} | {v:,.0f} |")
A("\nThe validation suite measured about 9,000-10,000 env steps/s for random-legal stepping on a quieter moment; the benchmark above (4,170) is the slower, later reading - the spread is CPU variance, not a different code path. Training throughput inside PPO is dominated by the policy network: the MLP trains at ~230 steps/s and the CNN at ~55 steps/s alone (96 and 30 steps/s per process with six processes sharing the CPU), versus ~4-5 steps/s for the native game. A 50k-decision pilot took 9 min (MLP) / 28 min (CNN) here versus roughly 3 hours natively.\n")
A("## 7. S1 results\n")
A(s1)
A("![S1](synthetic_lab/s1_mlp_vs_cnn.png)\n")
A("Top row: validation completion vs training decisions, first-action accuracy, median rank of a correct action at decision 1. Bottom row: training-episode success, TEST completion by gap orientation and by gap x-position (selected checkpoint).\n")
A("## 8. Reading the S1 result\n")
A("""* **H1 is supported in this sandbox.** At the same budget, seeds, tasks, reward and action space, the flat MLP stays at the random level (6.8% on TEST vs 9.3% for RANDOM) with no upward trend, while the spatial CNN reaches 89.8% +- 6.1 (per seed 83.5 / 90.2 / 95.8%) on 400 unseen layouts. The MLP failure reproduces the native-game failure in minutes instead of hours, and it happens with an immediate +1 reward, so reward delay is not needed to explain it.
* **Not memorisation:** the most common correct-action index covers 0.8% of test tasks; on the 70 TEST tasks whose correct action index never occurs in TRAIN the CNN still completes 81-94% (MLP 3-11%). CNN TEST completion by orientation: horizontal 92% (n=235), vertical 86% (n=165); by gap x-position 88-96% in all bins with n >= 46 (78% in the far-right bin, n=12); it falls mildly with path length (95% at 3-5 edges to 83% at 12-14). The MLP is flat at 6-8% in every group.
* **The CNN is targeting but not exact:** first-action accuracy is only about 40% (P(correct) at decision 1 about 0.25, median rank 2-3) - it narrows 3,000 actions to a handful of plausible gap candidates and, because wrong additions are free inside the 20-decision window, tries them in turn (median 2 decisions to completion, destructive failures <= 2%, ADD fraction 94%). A stricter protocol (penalised wrong roads or a one-shot budget) would score it differently; that was deliberately not changed here.
* **Variance:** seeds differ by about +-6 points on TEST; 3 seeds support the qualitative MLP-vs-CNN conclusion, not a precise accuracy.
* **Cost:** the CNN has 23 times fewer parameters but is about 4 times slower per training step.
* **Budget caveat:** 50k decisions is a short pilot. The MLP not learning within 50k does not prove it cannot learn with more steps, a different learning rate or width; it shows it is much less sample-efficient on this task. The CNN curve (65% at 20k, ~85% at 40k) was still rising at 50k for two seeds.
""")
A("## 9. Limitations\n")
A("""* The sandbox is a toy: one house-destination pair on a flat grid, no terrain, traffic, vehicles, demand, overcrowding, bridges or tunnels, no weekly upgrades, no time pressure, always-Active roads (no Pending / Mothballed states), a single connection colour.
* Legality is an approximation of the native rules: reach window per task, REMOVE offered only on single-edge tiles, simplified concrete accounting. Legal action counts differ (synthetic mean 202, native Gap-1 starts mean 151); the agreement between synthetic and native masks was **not** measured.
* The observation mapping was tested against the real `encode_observation` function fed with native-style field names, **not against live observations from the running game**; a live roundtrip comparison is a prerequisite for native transfer claims.
* Task generator constraint (carpark identifiable from the observation) makes the sandbox easier than the native game in one respect (see section 3).
* S1 is a 50k-decision pilot with 3 seeds per architecture; no hyperparameter tuning for either architecture; deterministic and stochastic policy results are both reported.
* Gap completion is scored within a free-retry construction window; first-action accuracy is reported separately.
* H2, H3, H4 were not run; the environment supports them (`reward_mode` delayed / native_like, Gap-2/3 manifests) but no results exist.
* Findings about architectures transfer as hypotheses, not weights.
""")
A("## 10. What justifies native validation\n")
A("""Supported by S1 and cheap to check natively: a **native Gap-1 run of the spatial CNN policy** (architecture transfer; the policy class is already compatible with the native observation/action layout) with the same frozen native manifests and the native completion-reward task, compared with the flat MLP at an equal budget. Before that, a one-off live roundtrip check of the synthetic adapter against real game observations and a measurement of mask agreement. Not justified yet: claims about delayed reward (H3), action-space size (H2) or multi-step construction (H4) - run those synthetically first, one variable at a time.
""")
A("## 11. Reproduction\n")
A("""```
python -m synthetic_rl.make_tasks --n-gaps 1            # frozen manifests (also --n-gaps 2/3)
python -m synthetic_rl.tests                              # 10 validation checks + hand example
python -m synthetic_rl.bench                              # throughput
python -m experiments.synthetic_run --config experiments/configs/synthetic_s1_cnn_s11.yaml
python experiments/synthetic_s1_report.py                 # tables + figure
```
Artifacts: `research/synthetic_lab/` (manifests, validation_results.json, hand_example.txt, throughput_benchmark.json, s1_results_section.md, s1_mlp_vs_cnn.png), run directories under `runs/`.
""")
open(os.path.join(ROOT, "research", "SYNTHETIC_RL_LAB.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
print("written research/SYNTHETIC_RL_LAB.md,", sum(len(x) for x in L), "chars")
