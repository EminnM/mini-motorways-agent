# S26 — Retrospective on the delivery cap (existing logs only) and the revised spawn-steering protocol (prepared, not executed)

## 1. Retrospective (no new episodes; `research/s26_retrospective/retrospective.json`)
Data with per-decision destination records: S13 replicate (S8-C, 12 episodes, 10,766 destination-decisions) and S22 V0/V0dup/V1/V2 (planner B, 48 episodes, 40,655 destination-decisions). **Unavailable:** S21 native-evaluation logs (no destination records → per-destination rate cannot be computed; totals only) and S24 (anchor snapshots at decision 200 only, no time series). These are measurement gaps, not negative findings.

**Deliveries per destination-decision vs connected houses per destination** (both policies agree):

| houses / destination | S8-C | planner B |
|---|---|---|
| 0 | 0.009 | 0.000 |
| (0, 0.75] | 0.029 | 0.031 |
| (0.75, 1.25] | 0.050 | 0.044 |
| (1.25, 2] | 0.064 | 0.049 |
| (2, 3] | 0.068 | 0.063 |

**Vs game progression:** S8-C 0.039 (decisions 0–99) → 0.047 → 0.064 → 0.054 → 0.061 → 0.058 (500–599); planner B 0.043 → 0.053 → 0.054 → 0.055 → 0.059 → 0.075. Late-game (t ≥ 300) by houses/destination: S8-C 0.053 / 0.060, planner B 0.039 / 0.043 / 0.054 / 0.065 — the same curve as early game. **No late-game departure** from the pattern; rates never exceed ≈0.065–0.075.
**Vs demand level:** max demand 1–3: 0.051 / 0.059; 4–6: 0.058 / 0.050; ≥ 7 (overcrowded): 0.049 / 0.053; overcrowded vs normal 0.049 vs 0.055 (S8-C), 0.053 vs 0.054 (B). **Deliveries do not rise when pins are waiting.**

**Conclusion.** The per-destination delivery rate is bounded above by ≈0.065 per decision — the S13-fitted pin-generation rate (0.055 per decision when demand > 0, 0.22 from zero) — and is reached only with ≥ 2 houses per destination. Below that it is **car-supply-limited**: each house owns 2 cars that must complete a full round trip (S25 §5–6), so with ≤ 1 house per destination a waiting pin cannot be served faster than the trip cycle, and overcrowded destinations (pins waiting) deliver no faster. Two consequences: (i) equal-cost road redundancy is correctly closed (S25) — it changes no trip time; (ii) *trip length* (travel-time cost of the one route the game uses) does bound deliveries when houses per destination ≤ 1 — a latency mechanism, not a redundancy one, and not tested here. The S21 cross-policy gap is fully explained by houses per destination (S25 §7).

## 2. Revised spawn-steering protocol (prepared for review; `research/spawn_harness/code/spawn_pilot.py`; DIAG_FORK validated in S22b)
**Seeds/anchors (fixed):** 10 fresh seeds 600000 + 7919·i (not the 12 evaluation seeds, not the S13 test seeds) × anchors 60 and 150 on a planner-B main line = 20 forks; horizon 90 decisions WAIT continuation; checkpoints 10/30/60/90 with observations.
**Branches per fork (all restored from one snapshot):** `base`, `base2` (negative control; any difference is reported), `block_far` = 3-edge disconnected segment on the empty in-window tiles farthest from every destination block, `block_near` = same-size segment nearest to a destination block (deterministic tie-breaks; a fork is skipped if fewer than 2 legal segments or far/near separation < 3 tiles — skips reported, never replaced).
**Why two opposite blocks instead of a sham:** both interventions perturb the placer's PRNG stream in the same way (one draw per feasible candidate), so stream perturbation is direction-neutral; **a directional effect — far-block moves spawns nearer, near-block moves them farther — isolates geometric blocking from stream perturbation** without needing an unobservable RNG-neutral sham. The harness's PRNG test still classifies each branch (stream shift before/after the first spawn difference).
**Per-spawn classification (predefined):** BLOCKED (base's spawn tile or a 4-neighbour is occupied by the intervention), SHIFTED (base's tile still free; a different tile won), UNCHANGED.
**Predefined strategic metric:** mean Manhattan distance of new houses to the nearest same-colour destination tile present at the anchor (lower = cheaper to connect); for new destinations, distance to the nearest anchor house. Primary comparison: `metric(block_far) − metric(base)` (expected negative if steering works) and `metric(block_near) − metric(base)` (expected positive), paired per fork; sign agreement across forks and bootstrap CIs; `base2 − base` must be 0.
**Opportunity cost:** intervention concrete (3 per branch) and the change in the planner's Dijkstra connection cost for houses present at the anchor and for new houses, computed on each branch's final (checkpoint-90) road graph vs base — benefit claimed only if the connection-cost saving on new houses exceeds 3 concrete plus any increase for anchor houses.
**Pre-specified success:** block_far reduces the metric in ≥ 14/20 forks (where a spawn occurred) and block_near increases it in ≥ 14/20, with base2 = base in all forks; otherwise "no directional steering detectable". Budget ≈ 20 × (main line ≤ 150 decisions + 4 × 90) ≈ 12 min native. Offline checks passed on 24 real anchor states: plans legal and deterministic, far/near separation (e.g. 8 vs 2 tiles).
**Not measured in phase 1:** deliveries under play (WAIT continuation); if steering is directional, a phase 2 with planner-B continuation (one fork per branch) would test whether nearer spawns raise deliveries — subject to the S26 cap: more deliveries require more houses per destination or shorter trips, which nearer spawns provide only indirectly.

## 3. One recommendation
Run the phase-1 spawn pilot exactly as specified above (20 forks, ≈12 min native) — it is the only remaining mechanism identified in this programme where cheap road placement changes the game's future state in a way a connectivity planner cannot, and the design now separates blocking from stream perturbation by direction and uses a predefined metric. If the directional test fails, close spawn steering as well and the honest residual lever is trip length (latency) when houses per destination ≤ 1, to be examined from existing logs before any new run. No RL, no architecture, no synthetic experiments.
