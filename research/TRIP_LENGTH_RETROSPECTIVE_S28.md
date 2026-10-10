# S28 — Trip-length hypothesis: retrospective on existing native logs (no new runs)

Question: when fewer than ~1.5 houses are connected per destination (S26: car-supply / round-trip regime), do shorter trips give more deliveries per destination-decision?

## 1. What can be reconstructed (and what cannot)
* **Road graph per decision:** only the S13 replicate logs (S8-C stochastic, 12 episodes, 3,867 decisions) record the functional-edge changes per decision (`F0` + `fa`/`fr`); the planner logs (S21, S22, S24) record only edge counts. The functional graph was rebuilt exactly for every decision of the 12 episodes.
* **Route length:** hop count of the shortest functional-road path from each connected house to a stub tile of a same-colour destination — reconstructable. **Travel time is not**: the game routes on `LaneModel.PathfindingCost` (length / speed limit with turn, junction, roundabout and motorway factors, S25 §1), and lane speeds/turn geometry are not in any log; hop count is used explicitly as a proxy and nothing else was substituted.
* **Per-destination deliveries:** the score is global per decision. Deliveries can be attributed to a destination only in decisions where exactly one colour group has connected houses: 1,537 of the 3,442 decisions with any connected house (1,905 are multi-group and unusable). With the ≤ 1.5 houses-per-destination filter: **702 decisions, 11 seeds, all with 1 connected house** (the first house of the game) — the regime of interest exists in the data only as the opening phase.

## 2. Association (702 decisions)
| route hops (1 connected house) | deliveries / destination-decision | decisions |
|---|---|---|
| ≤ 6 | 0.059 | 17 |
| 7–10 | 0.050 | 343 |
| 11–14 | 0.085 | 248 |
Raw bins run **opposite** to the hypothesis (longer routes, higher rate), but each seed contributes essentially one route length (the first house's), so this is a between-seed comparison of 11 cities, confounded by destination identity, demand and timing. Per-seed rates at fixed hops range from 0.00 to 0.12. Within-seed variation in hops exists in 8/12 seeds but only over a handful of transitional decisions (e.g. 507919: 9 hops for 63 decisions, 10 hops for 20, 11 for 8).
Within-seed fixed-effects regression of deliveries/destination-decision on (hops, demand, stage): hops **+0.009 ± 0.017** per hop (sign opposite to the hypothesis, not distinguishable from zero); demand −0.047 ± 0.012; stage +0.17 ± 0.13. Effective sample: 11 seeds, not 702 independent decisions.

## 3. Answers
1. *Reconstructable reliably?* Hop-count route length: yes, for the 12 S8-C episodes only. Round-trip travel time: **no** (lane costs not logged). Per-destination deliveries: only in single-group decisions.
2. *Shorter trips → more deliveries?* No association detectable; the only estimate (+0.009 ± 0.017 per hop within seed) is consistent with zero and has the wrong sign; between-seed bins are confounded.
3. *Strategically large enough?* Cannot be assessed: the observable regime (one house, opening phase, 0–7 deliveries per episode segment) is too thin, and the mechanistic effect size (2 cars per house ÷ round-trip time) cannot be computed without travel times.
4. *Causal or correlational?* Correlational at best, and underpowered even for that; the within-seed variation comes from S8-C's road churn, which also changes connectivity and traffic at the same time.
5. *Would a controlled native intervention be necessary?* To decide causally, yes — but it would be premature: the measurement gap below means even a controlled fork could not attribute deliveries to the manipulated destination when several colours are connected.

## 4. Recommendation: **B — existing data is insufficient; one specific minimal measurement gap**
The gap is attribution and cost, not more episodes: (i) per-destination delivery counts (the score is credited in `DestinationModel.AcceptVehicleArrival`, S25 §5 — a per-destination counter exported in the PolicyObservation or by a diagnostic op would make every decision attributable), and (ii) the pathfinder's actual path cost per house→destination (`ExternPathfinding.GetPathCost`, available natively) so trip time rather than hop count is measured. Both are read-only exports of existing native quantities (an additive plugin change, like DIAG_FORK). Without them the trip-length hypothesis cannot be tested from logs; with them, a first pass is again retrospective (any future planner or S8-C episodes), and only then would a small fork experiment (same connectivity, strictly cheaper vs longer single route to one destination) be worth proposing. No native run, RL, architecture or synthetic work is proposed now.
