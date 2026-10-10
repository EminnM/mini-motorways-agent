# Offline counterfactual-action probe — results

Research only. Nothing here is wired into PPO. Tools: `python/cf_probe.py`, `python/cf_analyze.py`,
C# op `DIAG_CF` (game_plugin/DoorstopBootstrap/Entrypoint.cs, probe DLL only).

## Method
At an anchor decision, the native simulation is exported once (the game's own save format, ~90–350 KB,
80–110 ms), then for every candidate first action the snapshot is restored (50–290 ms), the action applied, the
game continued with WAIT (or oracle follow-ups, see below), and the native score gained recorded at
horizons 1, 3, 5, 10, 20, 40, 60, 90 decisions (1 decision = 1 game second).
`A_h(a) = dScore_h(a) − dScore_h(WAIT)`.

Restore fidelity (what was fixed to make branches exact): the game-behaviour model's private random generator
(upgrade offers), `DestinationModel.ActivationTime`, transient demand counters, lane speed/bookkeeping fields and
the order of each lane's inbound-lane list are not carried by the game's save format and are captured/re-applied
alongside the snapshot (`SnapshotExtras`).
Evidence: restoring the same snapshot twice gave identical trajectories in 15/15 trials (5 policies x 3 seeds,
crossing weekly upgrade claims); duplicate-WAIT branches identical in 22/22 + 15/15 + 15/15 + 15/15 anchors.
**Known residual:** a restore-of-a-restore diverged on seed 182150 (car-following positions at one interaction,
step 39) and original-vs-restored diverged on seed 229664 (step 46). Cause not found. The probe is unaffected
because every compared branch is restored from the same snapshot.

## Runs
| file | start | main policy | anchors | candidates |
|---|---|---|---|---|
| gap0_randommain | working city | random legal | 22 | 514 |
| gap1_waitmain | 1 edit short of connected | WAIT | 15 | 273 |
| gap3_waitmain_waitcont | 3 edits short | WAIT | 15 | 300 |
| gap3_waitmain_oraclefollowups | 3 edits short | WAIT | 15 | 300 |

## Findings
1. **Effects are rare and late.** Working city: 17% of candidate actions change the 90 s score at all; 0% of them
   are visible at 1–5 s; 18% by 20 s, 49% by 40 s, 56% by 60 s. Gap-1: the single connecting edit is the only
   consistent mover (100% effective, +1.7 at 20 s, +6.7 at 90 s, 0 at <=5 s); random and remove actions: 0 effect.
2. **A connecting edit is credited ~10–20 decisions late** and then keeps paying (~+0.07 score/s).
3. **Multi-edit connections have no gradient.** Gap-3 with WAIT afterwards: every single candidate, including each
   of the three missing edits, has exactly 0 effect at every horizon and 0 change in the connection potential Φ.
   Neither the native reward nor the Φ shaping rewards partial paths.
4. **Oracle follow-ups test (flawed as a credit measure, kept for the record):** WAIT-first and oracle-first both
   finish the connection one step apart (mean +6.6 at 90 s each), so the first edit carries no marginal credit —
   value belongs to the completed sequence. The informative part: with a working connection, off-path actions
   cost score (adjacent roads −0.8 mean at 90 s, 46 negative vs 5 positive; removals −2.9).
5. **Φ shaping points the right way but misses most value.** Working city: actions that raise Φ have mean
   A_90 = +1.81 (13 positive / 2 negative / 6 zero), yet ~70 of the 85 effective actions do not change Φ; the
   largest gains (+6..+10) came from roads next to existing buildings with no Φ change.

## Implications (not acted on)
- A gap-1 start is learnable in principle (1 correct edit, 10–20 decision credit delay, positive Φ signal) but
  exploration is ~1/100 per decision.
- Gap-3 cannot be learned from reward or Φ alone; either a curriculum (gap 1→2→3) or a partial-progress signal
  (e.g. graph distance from the house's component to the destination) is needed. Not implemented.
- Any marginal-reward baseline should be a restored WAIT branch from the same snapshot (exact), with a horizon
  of >= 40–60 decisions to capture most delayed effects.
