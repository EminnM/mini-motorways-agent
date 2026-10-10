# S12 feasibility gate — written BEFORE any S12 native run or graph analysis

## What the existing S9–S11 data establish (observed, 28 S11 episodes, 9,000+ ADDs)
| action type | concrete change |
|---|---|
| ADD, plain extension (no house endpoint, one new functional edge) | **−1** in 2,266/2,267 cases |
| ADD, house-edge replacement (functional edge count unchanged) | **0** in 6,402/6,402 cases (swaps are free) |
| REMOVE | +1 (426) or 0 (117) |
| WAIT | 0 |
So concrete is consumed only by non-house extension edges. A "wasteful concrete expenditure" can therefore only be a plain-extension ADD. The existing logs keep edge COUNTS and connected-house sets, not edge identities or graph components, so the share of such ADDs that are redundant cannot be computed from them → one measurement-only control capture with edge-level logging is needed (it doubles as the control arm if the gate passes).

## Candidate wasteful class (the only one I will test): CYCLE-CLOSING EXTENSION
A native-legal ADD of edge e = (a, b) such that
1. no endpoint is a House / Destination / Carpark tile and neither endpoint is over water (water / bridge-like edges may use other inventory → unknown),
2. e is not currently present in any non-None road state (a Mothballed / Pending edge would be a re-activation → unknown),
3. a and b already lie in the SAME connected component of the current functional (Active / VisiblyActive) road graph (components built from road edges only, not via destination adjacency → conservative),
→ the ADD consumes 1 concrete, creates no new reachable tile and no new connection possibility.
Why it is permanently redundant for connectivity: non-house edges can only be removed by REMOVE, which native allows only on a tile with exactly one road edge (a leaf); a cycle edge has both endpoints of degree ≥ 2 so it is never removable, and a leaf spur cannot lie on a cycle. House swaps only move leaf edges. Hence connectivity with and without the edge is identical for the rest of the episode.
Not claimed: that the edge is useless for traffic / travel time (unmeasured) — this is a *connectivity-redundancy* claim and is reported as such.
Explicitly NOT vetoed (unknown or allowed): spurs and dead ends, edges between components, edges to unconnected tiles, any edge on or toward a long route, edges near houses, house-edge replacements, REMOVE / WAIT / upgrades, water edges, re-activations, anything the transition model cannot predict.

## Gate criteria (fixed now)
Proceed to the 12-seed treatment run only if ALL hold on the control capture + offline checks:
1. Concrete-balance, Mothballed-state and legal-availability predictions of the transition model match held-out observed transitions (concrete accounting ≥ 99 %; plain extension never changes Mothballed edges).
2. **Redundancy replay:** deleting every edge that was cycle-closing when added leaves the set of functionally connected houses unchanged at every decision of every captured episode (0 differences).
3. The class is non-trivial: cycle-closing ADDs are ≥ 5 % of executed ADDs OR ≥ 5 % of the concrete consumed by plain extensions in the control capture (otherwise the guard would rarely intervene and a null result would be uninformative).
4. All 10 regression tests pass; unknown classifications are logged and never vetoed.
If criterion 3 fails, or no defensible class exists, I STOP and report a feasibility failure rather than widen the definition.
Native budget: 10 min in total, including the capture.
