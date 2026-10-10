# S10: coverage of the S9 data, and replay seed selection (written before the replay)

| Measurement | Recoverable from S9? | New native capture required? |
|---|---|---|
| Concrete budget (available / consumed / returned) | **No.** Not logged. Reconstruction from edges added/removed is unreliable because spawning houses add free driveway edges during the simulation advance. | **Yes** (raw `globals.concreteAvailable`, read-only) |
| Legal ADD count / legal REMOVE count | Only the **total** legal count per step (`n_legal`); no split | Yes (split taken from the native mask) |
| Spawn timing | **Houses: yes** (houses per step). **Destinations: no.** | Yes, for destinations |
| Road churn | Partly: per-step edges added/removed, ADD-with-removal ("replacement") flag, episode swap counts, road-state cycle statistics | No (per-step connection-loss attribution is new) |
| Connected buildings | Yes: connected houses per step, houses per step; unconnected = houses − connected | No |
| Delivery timing | Yes (scoreDelta per step) | No |
| Terminal conditions | Partly: terminated/truncated, penalty-missing flag, final status; **which destination overcrowded / demand pressure: no** | Yes (per-destination demand and overcrowding flags are in the raw observation) |
| Traffic / vehicles | No | Yes (visible vehicle counts, read-only) |

Critical variables (concrete, legal ADD/REMOVE split, destination spawns, demand / overcrowding) are missing, so the minimal measurement-only replay is justified.

## Seed selection (S8-C deterministic, frozen checkpoint; chosen from S9 failure patterns, not for favourable outcomes)
* 563352 — long episode (463 decisions, 85 deliveries) that nevertheless ends with 14 houses / 7 connected: growth then collapse
* 523757 — never connects (initial k = 10): sequential failure + 142 swaps in 152 ADDs
* 539595 — first connection at decision 145 of 150: late initial connection
* 587109 — short mid-length episode (211 decisions), 5 houses, low replacement share (39/167): a different churn pattern

Budget: at most 4 episodes, at most 10 minutes of native wall-clock, plus a short instrumented-vs-plain paired rollout.
