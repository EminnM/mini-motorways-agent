# S22 native throughput probe — preregistration (written before execution)

**Question.** Holding connectivity and demand comparable, does road topology (minimal tree vs parallel route vs denser mesh) change native delivery throughput?

**Design (no DIAG_FORK available).** For each of the 12 designated seeds (500000 + 7919·i) four episodes are run from `RESET(seed)`, `NORMAL_START`, `decision_delta` 1.0, 600-decision cap:
* **V0 minimal** — planner B throughout (reference).
* **V0dup** — identical to V0 (no-edit duplicate control): estimates native run-to-run variation and establishes starting-state equivalence.
* **V1 parallel** — planner B until T0 = 60 decisions, then a deterministic "parallel route" edit set computed from the state at T0 (offset copy of the longest trunk between the destination stub and the farthest connected house, with end connectors; up to K = 8 plain edges), one edit per decision, then planner B.
* **V2 dense** — same prefix; a deterministic "mesh" edit set: up to K = 8 legal plain edges attached to trunk tiles in order of distance from the stub, preferring edges that close loops between two existing road tiles; then planner B.
All four share the first 60 decisions' action sequence by construction (same deterministic planner, same seed). Edits are executed only if legal under the native mask at that decision (skipped otherwise; counts reported). Nothing is tuned after seeing results; the edit rules are fixed in `code/throughput_probe.py` before the first run.

**Equivalence requirements (checked per seed before any comparison):**
1. V0 and V0dup raw observations identical at every decision ≤ T0 (hash of the full tile JSON incl. vehicle counts + globals). If not identical → seed excluded from the matched analysis (reported as native non-determinism).
2. All four variants identical at every decision ≤ T0 (same check) — otherwise excluded.
3. Connectivity: the set of connected houses at T0 is identical across variants (follows from 1–2); after the edits, V1/V2 must not lose any connection (planner continuation + S11 model; destroyed-connection count must be 0 in the edit window or the seed is flagged).
4. Demand: destination demand at T0 identical (follows from 1–2); demand trajectories after T0 are an *outcome*, reported.
5. Infrastructure: concrete differs by construction (V1/V2 spend up to K); reported, and V1 vs V2 are compared at equal spend.

**Spawn control.** Every new house/destination (decision, type, group, tile) is logged. `t_div(seed, variant)` = first decision at which the variant's spawn sequence differs from V0's. The **matched window** is `[T0+1, min(end_V0, end_variant, t_div)]`. The primary outcome is computed inside the matched window, so spawn-steering by the added roads cannot drive it; whole-episode outcomes are secondary and flagged when `t_div` < end.

**Primary outcome.** Deliveries per connected-house-decision in the matched window: `R = Σ_t scoreDelta_t / Σ_t |connected_t|`. **Primary comparison:** V2 dense − V0 minimal, paired per seed. Noise reference: `|R_V0dup − R_V0|` per seed.
**Decision rule (preregistered):** a topology effect on throughput is claimed only if (a) ≥ 9 of the matched seeds have the same sign of `R_V2 − R_V0`, and (b) the median |R_V2 − R_V0| exceeds 2× the median |R_V0dup − R_V0|. Otherwise the result is "no detectable topology effect at this scale" (a negative result).
**Secondary outcomes** (same window and whole episode): V1 − V0; per-house deliveries; connected-house exposure; total visible vehicles per decision and per connected house; first overcrowding decision and overcrowded-decision fraction; concrete spent; survival decisions; new houses connected ≤ 60 after T0 (planner continuation); destroyed connections.

**Budget.** 48 episodes in 4 parallel instances, ≈ 7–8 min native wall-clock + a 1-seed smoke run (V2, 150 decisions) to verify the edit rules execute legally. No topology variant is modified after the smoke except to fix an execution bug (documented).

**Limitations known in advance.** One episode per (seed, variant); the native per-tile vehicle count is the only traffic observable; the edit rules are generic, not optimised; equivalence is established empirically (hash equality), not by snapshot restore.
