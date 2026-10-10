# M1 — preregistered deterministic upgrade-choice rule (defined before any Stage A/B outcome was seen)

Native facts used (S21 audit D, S25, S18-H MECHANICS): weekly upgrade packages are granted by `UpgradeAwardingProcess`; each pending choice offers `GetNumberOfUpgradeOptionsPerWeek()` = 2 options drawn by the behaviour PRNG; `ChooseUpgrade(i)` selects `pendingUpgradeChoices[0].choices[i]`; while a choice is pending only ChooseUpgrade actions are legal. `UpgradeType` ∈ {Concrete, Bridge, Motorway, TrafficLight, Roundabout, Tunnel, House, Destination, DoubleDestination}. Our agent can USE only Concrete (road building) and Bridge (implicit in ADD over water); Motorway/TrafficLight/Roundabout have no action interface (S29); Tunnel is irrelevant on this map (no mountains); House/Destination/DoubleDestination are building packages whose in-play effect is not verified by us (no mechanics research is authorised).

**Rule R (applied to the two offered types a = slot 0, b = slot 1):** pick the slot whose type has the lower rank in this fixed order, with `concreteAvailable` read from the observation at the moment of choice:
1. `Concrete` if `concreteAvailable` < 10 (planner B spends ≈ 5–10 concrete per connection; S21 median 21–22 per episode; this is the only resource the planner consumes)
2. `Bridge` if `bridgesAvailable` == 0 (the S13 map has a river; a route across water is impossible without one)
3. `DoubleDestination`, then `Destination`, then `House` (building packages: the planner can connect whatever appears; preferred over infrastructure it cannot place)
4. `Concrete` (when ≥ 10: still usable later)
5. `Bridge` (when ≥ 1 available)
6. `Motorway`, `Roundabout`, `TrafficLight`, `Tunnel` (not usable by this agent)
Ties (same rank) → slot 0. Unknown type string → treated as rank 6.

**Arms (Stage B):** `slot0` (current blind behaviour), `random` (seeded `random.Random(seed)` per episode choosing slot 0/1 uniformly), `ruleR`. Road construction = planner B unchanged (`research/hierarchical_planner`). 12 designated seeds, 600 decisions, one episode per arm per seed (deterministic policy ⇒ exact pairing; random arm seeded).
**Logged per choice:** decision, offered types, chosen slot/type, `applyInfo` type (must match), inventory totals before/after (concrete, bridges, motorways, lights, roundabouts, tunnels), concrete available; per episode: deliveries, survival decisions, connected/houses, new-house latency.
**Stage A pass criteria:** export present in every observation; offered list length == 2 whenever `pendingUpgradeChoices` > 0 and empty otherwise; chosen type from `applyInfo` equals the exported type of the chosen slot on every choice; at least 2 choices per episode on average and the two offered types differ in ≥ 50 % of choices. Otherwise stop.
**Decision after Stage B:** keep rule R if it improves survival or deliveries vs slot0 on a paired majority of seeds with no seed worse by more than the gain elsewhere; reject if indistinguishable; the result is a policy-level comparison and says nothing causal about any single choice.
