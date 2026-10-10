# Spawn-causality pilot (research only; nothing integrated into PPO)

Protocol: CONNECTED_START city, WAIT main line to anchor (30 s / 60 s), snapshot (restored branches share RNG/state),
branches A=WAIT, A2=independent duplicate WAIT, B=3-edge straight road segment (4 tiles) disconnected from every road /
building / carpark (not even 4-adjacent) at Manhattan distance 3-8 from the nearest building, C=same geometry as far as
legal (target >=14), then WAIT for 180 s; every new house/destination logged (time, type, group, tile).
Data: pilot.jsonl (19 snapshots + 1 skipped, 10 held-out EVAL_SEEDS x anchors 30,60). Analysis: pilot_analysis.txt.
Driver: python/spawn_probe.py, analysis: python/spawn_analyze.py, C# op DIAG_SPAWN (probe DLL only).

## Protocol problems found
1. The "far" control is not far. In every snapshot the legal reach area was too small for a 14-tile-far segment
   (farCandidates=0); C ended up 4-6 tiles from the nearest building vs 3-5 for B. C is a nearby disconnected road, not an
   irrelevant one.
2. RNG-stream confound: placing any road changes the number of legal placement candidates (tiles are blocked), and the
   placer draws one random number per candidate, so the stream shifts. Observed: PRNG state differs at some checkpoint in
   12/19 C branches. Spawn displacement can therefore come from stream shift, not geometry. Needs an RNG-neutral control.
3. Game-over: WAIT-only cities reach the game-over condition ~174 s after the anchor; the primary window stops one step
   before the first branch's terminal step. One smoke-test run (not reproduced in the 19-snapshot pilot) showed A vs A2
   differing by 0.1 s on a post-terminal spawn.
4. Two anchors per seed share a city (non-independent); one snapshot skipped (no valid local segment).
5. Index-matched displacement is approximate once sequences diverge; set-based counts reported alongside.

## Status: STOPPED (user decision) — results preserved, no further compute
Code-reading findings recorded before stopping (useful for a later behavioural analysis):
- A spawn event (BuildingPlacer.StartPlacing -> GeneratePlacements -> ChoosePlacement) enumerates every placeable tile
  row-major (y then x) x layouts; each *feasible* candidate draws one variate (`pseudorandomGenerator.Fix64(weight)`) and
  candidates are then sorted by weight (base weight x variate); the best candidate whose driveways are free is chosen.
- Road tiles (IsTileBuildable false) drop out of enumeration, which shifts the variate of every later candidate and the
  stream position for later events. Exact coupling would need a hook inside that loop (not feasible without detouring
  non-virtual game methods); an offline "is the control's chosen tile still feasible" analysis is possible instead.
- PRNG draw count between two states is recoverable exactly: n = (w1 - w0) * inverse(_s) mod 2^64, _s = 13091206342165455529.
Planned use: only if a trained agent places apparently non-connective roads — then fork around those actions and test
spawn geometry and long-term score. No code changes were left running; the probe game instance has been stopped.
