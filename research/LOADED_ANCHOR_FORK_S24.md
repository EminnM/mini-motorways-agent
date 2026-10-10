# S24 — Loaded-anchor fork experiment: stopped at the preregistered eligibility gate (design failure)

Preregistered in `research/s24_loaded_anchor/PREREGISTRATION.md` before inspection. Pass 1 (anchor inspection, no fork, no outcomes seen): `code/inspect_anchors.py`, per-seed records with the anchor observation, mask, candidate edit sets, per-tile vehicle counts over decisions 170–200 and the structural checks in `raw/anchor_<seed>.json.gz`; summary `eligibility.json`. Native wall-clock 161 s. **No fork was run, no intervention was substituted, no RL, no spawn pilot.**

## 1. Eligibility results (rules E1–E4, decided from pass 1 only)

| seed | decisions to anchor | connected / houses | traffic (Σ visible vehicles, 170–200) | parallel: edges, distinct route, usable | dense: edges, Δcyclomatic, usable | eligible |
|---|---|---|---|---|---|---|
| 500000 | 200 | 4/6 | 25 | 7, **no** (spur: one connector only), yes | 8, **0**, no | no |
| 507919 | 151 (game over) | 2/4 | 0 | – | – | no (E1/E2) |
| 515838 | 145 (game over) | 2/4 | 0 | – | – | no |
| 523757 | 200 | 6/7 | 28 | 7, no (spur), no | 8, 0, no | no |
| 531676 | 200 | 4/5 | 18 | 8, no, yes | 8, <2, yes | no (E3) |
| 539595 | 200 | 5/5 | 39 | 5, no, no | 8, <2, no | no |
| 547514 | 154 (game over) | 2/4 | 0 | – | – | no |
| **555433** | 200 | 6/6 | 42 | 8, **yes** (cycle through 8 added edges; two houses' routes shortened 9→6, 12→9) | 8, **+3**, yes | **yes** (both) |
| 563352 | 200 | 6/6 | 44 | 6, no (attaches at one tile), no | 8, +1, yes | no (E3 dense needs ≥ 2) |
| **571271** | 200 | 4/6 | 15 | 6, yes, yes | 8, ≥2, yes | **yes** (both) |
| **579190** | 200 | 5/5 | 22 | 8, yes, yes | 8, <2, no | **yes** (parallel only) |
| 587109 | 200 | 5/5 | 26 | 6, no, no | 8, <2, no | no |

**3 eligible seeds (555433, 571271, 579190) < the preregistered minimum of 6 → stop.** Three further seeds never reach decision 200 (game over at 145–154: the colour-timing failure seen in S21 — two houses of a colour whose destination has not spawned).

## 2. Why the interventions fail the manipulation-validity rules at loaded anchors (inspection evidence, not outcomes)
* **Parallel rule (S22, ≤ 8 edges, connectors first then ≤ K−2 line edges):** at decision 200 the longest trunk is 9–14 hops, so the offset copy cannot be completed within 8 edges; the line ends short of the far connector and attaches at one end only (e.g. 523757: connector (−16,3,S) + line y=5 from x=−15 to −10, far end open; 563352: attaches at (−14,6) only). The result is a spur, not an alternative path — vehicles cannot use it. It is a valid loop only where the trunk happens to be ≤ 7 hops (555433, 571271, 579190).
* **Dense rule (loop-closing edges first, then free tiles):** at the anchor the trunk is a tree with few pairs of adjacent road tiles not already joined, so it finds 0–1 loop-closing edges and fills the rest with one-tile spurs (Δcyclomatic 0–1 on 9/12 seeds). On 500000 and 523757 it adds no cycle at all.
* Usability (E4: a connected house's shortest path to its own-colour stub shortened, or an equal-length alternative through added edges) holds for only 5 parallel and 4 dense sets; on the structurally valid seeds it holds (555433: hops 9→6 and 12→9 for two houses).
The checker itself was spot-checked on positive and negative cases (555433 valid loop with attachment tiles at both ends; 500000/523757/563352 single-attachment spurs).

## 3. What this means for the question
The S22/S23 interventions are **decorative at loaded anchors** on 9 of 12 seeds: they add road without creating a route traffic could take. Any delivery difference they produced would have been uninterpretable as a topology effect, which is exactly what the required manipulation check was meant to catch. S23's null result (anchor 60) is also partly explained: its trunks were short enough for loops to form, but the network carried ≤ 2 vehicles.
Bottleneck tiles (top-5 vehicle sums over 170–200) sit on the shared trunk next to the destination stub on every seed (e.g. 555433: (−13,6), (−7,6), (−7,7), (−8,7), (−14,6)); a meaningful bypass must duplicate exactly that segment.

## 4. Exactly one recommended next experiment (to be preregistered before any run)
Same fork protocol (12 seeds, planner-B main line to 200, base/base2/intervention, 90-decision WAIT, manipulation checks M1–M4, pre-divergence vs full-horizon analysis), with **one redesigned, mechanically guaranteed intervention instead of two heuristic ones**: *bottleneck bypass* — take the 3–5-hop trunk segment containing the top bottleneck tile, build a disjoint bypass of equal hop length between its two endpoint road tiles (both connectors included; concrete budget = segment length + 2 ≤ 8, verified legal from the mask), and require before forking that (a) the bypass closes a cycle (Δcyclomatic = 1 with both attachments), (b) at least one connected house has an equal-length alternative path through it (so the game's shortest-path routing can take it), (c) traffic Σ ≥ 10 at the anchor. Seeds failing (a)–(c) are ineligible; no substitution. If fewer than 6 seeds qualify under that rule as well, the honest conclusion is that this map/planner regime does not offer the topology choices the hypothesis needs, and the throughput question should be closed rather than pursued with further variants.
