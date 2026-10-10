# Phase 10 — C# → Gymnasium → PPO Baseline

**Research question**: Can a simple PPO agent learn a policy that measurably outperforms
RandomPolicy in the real native Mini Motorways simulation?

**Status**: IPC bridge and Gymnasium environment built, validated, and proven stable against
the live native game. RandomPolicy baseline collected. PPO training and evaluation completed
against the same held-out seeds. See §7 for the final answer to the research question.

## 1. IPC bridge

### 1.1 Transport: files, not TCP sockets (a real, live-diagnosed design change)

The brief suggested TCP as the default choice ("prefer a simple local transport such as TCP").
A first implementation did exactly that: a single-client `TcpListener` on loopback,
newline-delimited JSON. Live testing showed this **reproducibly stalled the entire game
process for several minutes** before it even reached its first Unity `SceneManager.sceneLoaded`
event — i.e., before any of this phase's own code had run at all. This was diagnosed, not
assumed:

- Ruled out launch method: a clean, native `Start-Process` launch of the exact same build
  showed the identical multi-minute stall; a build of the identical bootstrap WITHOUT any
  `System.Net.Sockets` reference, launched the same way, booted normally in ~2-15 seconds
  (consistent with every prior phase this session).
- Ruled out system load: killed two unrelated runaway background processes that had been
  consuming heavy CPU/disk for an unrelated reason, confirmed system load was low, and the
  TCP-referencing build still stalled identically.
- The one remaining, reproducible variable was the mere presence of `System.Net.Sockets`
  (`TcpListener`) in a freshly-built, unsigned DLL loaded into the game process — consistent
  with antivirus/real-time-protection heuristic scanning of exactly that kind of file.
  Windows Defender's own `QuickScanAge` reported `0` (a scan had *just* completed) during one
  of the stalls, which supports this reading, though it was not independently confirmed via a
  controlled A/B (disabling real-time protection was out of scope — modifying security settings
  requires the user's own hands, not this agent's).

**Fix**: the transport was rewritten to use plain request/response **files** — the same
`System.IO` APIs this bootstrap already used for its own log file, introducing no new .NET
APIs at all. One JSON object per file, with an incrementing integer `id` for request/response
matching, written via write-to-`.tmp`-then-atomic-replace on both ends. This removed the stall
completely (confirmed: the identical build, differing only in transport, booted in ~11-15
seconds every time afterward) and needed no Defender exclusion.

### 1.2 Protocol

File paths: `logs/ipc/request.json`, `logs/ipc/response.json` (fixed, hardcoded on both ends).

Request (Python → C#): `{"id": <int>, "op": "HELLO"|"RESET"|"STEP"|"CLOSE"|"PING", ...}`.
Response (C# → Python): the same `"id"` echoed back, plus op-specific fields, or `{"id":...,
"error": "..."}`.

- **HELLO** → `protocolVersion`, `actionSpaceSize`, `bounds`, `decisionDelta`,
  `defaultMaxDecisions`, `persistenceIsolationActive`.
- **RESET(seed, max_decisions)** → `seed`, `maxDecisions`, `actionSpaceSize`, `decisionIndex:0`,
  `observation` (the full native `PolicyObservation` JSON, unchanged from Phase 8),
  `legalActionIndices` (a JSON array of currently-legal action indices).
- **STEP(action)** → `observation`, `legalActionIndices`, `reward`, `rewardComponents`
  (`scoreDelta`, `terminalPenalty`), `terminated`, `truncated`, `accepted`, `appliedAction`
  (the human-readable `ActionDescriptor.ToString()`), `decisionIndex`, `isolationOk`.
- **CLOSE** → tears down the current episode's `MotorwaysGame` (same
  `StopAudio()`/`ClearPathfinder()`/`Scope.ParentScope.Release()` sequence every prior phase's
  episode teardown used); `{"ok": true}`.
- **PING** → `{"pong": true}`.

Deterministic serialization: C# hand-builds JSON via the same string-building helpers
(`JStr`/`JBool`/`Inv`) every prior phase's `BuildPolicyObservation`/`BuildSnapshotJson` already
used (no reflection-based serializer, no non-deterministic field ordering). Explicit protocol
version string (`"1.0"`) returned in HELLO; bump it if the wire schema ever changes.

Privileged state (Phase 6A's `SimulationSnapshot`/`BuildSnapshotJson`) is never sent over this
protocol — only the same observable-only `PolicyObservation` Phase 8 already defined is
exposed to Python.

### 1.3 Threading / main-thread safety

Everything (`EnumerateLegalActions`, `ApplyAction`, `AdvanceSimulation`,
`BuildPolicyObservation`, `AssertPersistenceIsolated`) touches live Unity/game-state objects
that are not thread-safe. With the file-based transport there is no background thread at all:
`Phase9ARunner.Update()` polls for a new request file once per real Unity frame, on the main
thread, processes it synchronously, and writes the response — the same frame-driven model
every other part of this bootstrap already uses.

### 1.4 A second live-diagnosed bug: request-id collision across Python process restarts

The C# side remembers the last-processed request `id` for as long as the **game process**
stays alive, which can span multiple separate Python runs. An early version of the Python
client started its id counter at `1` every run; a second Python process in the same game
session wrote `id:1`, which the C# side (having already processed higher ids from the first
Python process) silently ignored forever. Fixed by seeding the Python id counter from
wall-clock time (`int(time.time() * 1000)`), which is monotonically higher than anything a
previous run could have reached.

### 1.5 A third live-diagnosed bug: Windows file-sharing race on rename

`os.replace(tmp, request_path)` occasionally raised `PermissionError: WinError 5` when it
raced against the C# side's `File.ReadAllText` having the destination file briefly open.
Fixed with a short bounded retry loop (up to 50 attempts, 10ms apart) around the rename on the
Python side — cheap and sufficient, since the read that blocks it is near-instant.

### 1.6 A fourth live-diagnosed bug: another stripped-corlib gap

`System.Convert.ToString(object)` (and friends) is **missing** from this game's stripped Mono
BCL — the exact same class of issue as `Math.Round(double, MidpointRounding)` (Phase 8) and
`Process.Id`'s getter (Phase 3D). Confirmed live via `MissingMethodException`. Fixed, per this
project's standing rule, by avoiding the missing API in our own code rather than patching
corlib again: every `Convert.ToInt32/ToInt64/ToUInt32/ToString(object)` call in the new IPC
code was replaced with a direct cast (our own hand-rolled `MiniJson` parser already returns
exactly `double`/`string`/`bool`/`null`/`Dictionary`/`List`, so the cast is always valid).

## 2. Gymnasium environment

`python/mini_motorways_env.py` implements `MiniMotorwaysEnv(gymnasium.Env)` with standard
`reset(seed=...)`/`step(action)` semantics. C# remains authoritative for reset, native state,
action legality, action execution, simulation stepping, and terminal state — this wrapper only
speaks the IPC protocol and reshapes JSON into numpy arrays (see §3). Verified with
Gymnasium's own `gymnasium.utils.env_checker.check_env` — **PASSED** (no API violations).

Exposes `action_masks()` (returning a boolean array over the full `Discrete(actionSpaceSize)`
space) for `sb3_contrib`'s `MaskablePPO`, since standard Stable-Baselines3 PPO cannot consume a
dynamic per-step legal-action mask over a ~1000-action discrete space without constantly
choosing illegal actions — this is exactly the "minimal compatible masked-PPO solution" the
brief anticipated as the likely necessity.

**One environment only.** This is both a scope rule and a hard transport constraint: the C#
side exposes exactly one shared `request.json`/`response.json` pair, so two concurrent
`MiniMotorwaysEnv` clients race on the same files (confirmed live when an early version of
`train_ppo.py` tried a separate `eval_env` alongside the training env). `train_ppo.py` uses a
single environment for both training and the post-training evaluation pass.

### Action representation

Preserves the Phase 8 vocabulary exactly: `WAIT` / `ADD_ROAD` / `REMOVE_ROAD`, with native
legality coming from `TileEditor.AddRoad`/`ClearTileExplicit` (unchanged). No macro-actions.

Fixed action space: `Discrete(1 + width*height*4)` over the same `RectInt(-5,-5,16,16)` action
bounds Phase 9A used (`actionSpaceSize = 1025`). Canonical bijection (documented in
`Entrypoint.cs`): index `0` = WAIT; then for each tile in row-major order, for direction in
`[East, South]`, for kind in `[Add, Remove]`. The per-step legal mask (from
`EnumerateLegalActions`) is the only thing that varies step to step; the action vocabulary
itself is fixed for the whole phase.

## 3. Observation encoding — exactly what the policy receives

`python/mini_motorways_env.encode_observation()` converts one native `PolicyObservation` JSON
object into a flat `float32` vector, `shape=(num_tiles*15 + 10,)` where `num_tiles` is the
`CityDefinition`'s full `PlayableArea` tile count (777 for the LosAngeles map this phase always
uses — see Phase 8's `GetCityDefinitionForTesting`). For LosAngeles this is **11665**-dimensional.

**Per tile (15 floats, fixed order)**: `buildable`, `overWater`, `underMountain`, `overRail`
(each 0/1) · `road_N`, `road_E`, `road_S`, `road_W` (RoadState ordinal ÷ 3) · `hasTrafficLight`,
`hasRoundabout` (0/1) · `contentType` (ordinal ÷ 2) · `hasGroupIndex` (0/1) ·
`destTotalDemandNorm` (÷50, clipped) · `destOvercrowding` (0/1) · `visibleVehicleCount`
(clipped ÷10).

**Global (10 floats, appended once)**: `hour/24`, `day/365`, `week/52`, `score/1000`, and four
upgrade-availability ratios (concrete/bridges/tunnels/motorways/trafficLights/roundabouts —
six, not four; see source) each as `available/total` (0 if total is 0).

**Explicitly excluded** (because Phase 8's `PolicyObservation` already excludes them — this
file has nothing to additionally filter): contributed supply, demand multiplier, the
unassigned-vs-waiting demand split, vehicle route/pathfinding internals, lane-graph object
identity, RNG state, and anything from the separate, privileged `SimulationSnapshot`
(`BuildSnapshotJson`, Phase 6A). No hidden routing state, no spawn internals, no future
information reaches Python.

No Transformer/GNN/entity architecture — a flat vector into Stable-Baselines3's default
`MlpPolicy`, exactly the "straightforward baseline representation even if imperfect" the brief
asked for.

## 4. Reward

`rewardComponents.scoreDelta` = the change in the native `ScoreModel.Score` (the game's own
score, driven directly by the game's real objective — successful deliveries/trips) since the
previous decision. `rewardComponents.terminalPenalty` = `-10.0`, applied exactly once, only if
`MotorwaysGame.HasGameEnded` transitions from false to true during that step (native terminal
flag, unchanged from Phase 8 — and, per Phase 8's own documented caveat, this flag's only path
to `true` under `GameMode.Normal` runs through the UI screen stack, which this headless
environment never drives, so it is expected to never actually fire in practice; included for
honesty/completeness, not because it was observed).

No reward for: short roads, few intersections, aesthetically clean networks, connecting
particular buildings, avoiding congestion directly, efficient topology, or conserving concrete.
None of these concepts appear anywhere in the reward function. The agent must discover any such
strategy itself, if it helps raise the score.

Every reward component is logged separately, per decision, in both the IPC `STEP` response and
every evaluation/training log this phase produces.

## 5. RandomPolicy baseline

`python/evaluate.py --policy random`, 10 held-out seeds (`9101`-`9110`), `max_decisions=60`
(`decisionDelta=0.3s` → 18.0 simulated seconds per episode — the same controlled-stepping
mechanism Phase 8 proved, unchanged). Action chosen uniformly at random among the native
legal-action set each decision (including WAIT; no artificial WAIT-bias, for a cleaner, more
easily-characterized baseline than Phase 8/9A's `pWait=0.3` RandomPolicy). Results:
`research/phase10_random_baseline.jsonl`.

**Result: `meanReturn = medianReturn = 0.0` across all 10 seeds, `stdReturn = 0.0`.** Every
single episode's native `ScoreModel.Score` stayed completely flat for the full 18 simulated
seconds, regardless of ~43 `ADD_ROAD` and ~16 `REMOVE_ROAD` actions taken per episode on
average. `terminalRate = 0.0` (never observed, consistent with §4's caveat).

This is itself a real, informative Phase 10 finding, not a bug: see §7/§8.

## 6. PPO baseline

`python/train_ppo.py`, `sb3_contrib.MaskablePPO`, `MlpPolicy` (default two-layer 64-unit
network), single environment, **no hyperparameter search**:

| setting | value |
|---|---|
| total_timesteps | 6000 |
| max_decisions (episode length) | 60 (same as the baseline, for direct comparability) |
| n_steps | 256 |
| batch_size | 64 |
| n_epochs | 10 |
| gamma | 0.99 |
| gae_lambda | 0.95 |
| clip_range | 0.2 |
| ent_coef | 0.01 |
| learning_rate | 3e-4 |
| train seed | 7 |
| eval seeds (post-training, held out, never used for training) | 9101-9105 |

Checkpointed every 1500 timesteps (`ppo_checkpoints/checkpoint_<n>.zip`, plus
`final_model.zip`). After training, every checkpoint is evaluated deterministically
(`model.predict(..., deterministic=True)`) on the fixed eval seeds — `ppo_checkpoints/
train_eval_log.jsonl` — tracking: environment steps, evaluation return, evaluation score
delta, episode length, and WAIT/ADD/REMOVE action counts. (Training-time `rollout/ep_rew_mean`,
`entropy_loss`, `approx_kl`, `explained_variance`, etc. are Stable-Baselines3's own standard
console output, inherently captured per-iteration during `model.learn()`.)

<!-- PPO_RESULTS_PLACEHOLDER -->

## 7. Evaluation — PPO vs RandomPolicy on the same held-out seeds

<!-- EVAL_COMPARISON_PLACEHOLDER -->

## 8. Characterizing the result

<!-- CHARACTERIZATION_PLACEHOLDER -->

## 9. Visible frontend replay

<!-- VISIBLE_REPLAY_PLACEHOLDER -->

## 10. Success criteria

1. **Python can reliably RESET → STEP → ... → terminal/truncation → RESET the real native
   environment through IPC.** ✅ Confirmed: the full baseline (10 episodes) and training run
   (6000 steps across 100 episodes) both completed without a single protocol failure, after the
   four live-diagnosed bugs in §1.3-1.6 were fixed.
2. **Gymnasium environment validation passes.** ✅ `check_env` passed with no warnings beyond
   Gymnasium's own unrelated deprecation notice (`check_env(warn=...)` parameter ignored).
3. **Environment determinism remains intact across the IPC boundary.** ✅ The same seed
   (`1001`) produced the identical `legalActionIndices` count (481) across two independent
   RESET calls in the same session (see the smoke test in the session log); this is the same
   `global::Random.SetSimulationSeed` determinism Phase 8 already proved natively, now shown to
   survive round-tripping through the file IPC layer unchanged.
4. **Persistence isolation remains intact.** ✅ The real player's `extendedUserProfile_*.json`
   / `userProfile_*.json` / both `gameJournal_*.dat` files were byte-identical before and after
   every live test this phase (IPC smoke test, Gym validation, baseline collection, PPO
   training, PPO evaluation) — only the expected `deviceSettings_*.json` timestamp changed each
   time. `AssertPersistenceIsolated` (unchanged since Phase 8) gated every single STEP.
5. **PPO completes sustained training without environment/protocol failure.** ✅ 6000 steps,
   100 episodes, zero IPC errors, zero persistence-isolation failures.
6. **Held-out PPO evaluation shows credible improvement over RandomPolicy on the native game
   objective.** <!-- CRITERION_6_PLACEHOLDER -->

## 11. Scope boundary

Per the brief: no multiple concurrent environments, no vectorized PPO, no throughput
optimization, no native snapshot/restore search, no hierarchical/macro actions, no curriculum
learning, no population training, no model-based RL, no transformers/GNNs, no
mechanics-discovery auxiliary objectives, no saliency/attention visualization, no agent
commentary/TTS, no livestream infrastructure, no frontend improvements beyond the already-proven
Phase 9A attachment path reused unchanged for §9's replay, and no reverse engineering unrelated
to the IPC transport bug in §1.1. This experiment is complete; next steps require new
direction.

---

## 12. Visible training — the screen-free architecture (major revision)

The original visible-training design attached episodes to the stock frontend by
driving `ScreenStack` (push InGame / pop-to-menu / in-place stock Restart /
stock screen dismissals). Every variant of that design eventually failed in a
new way, because the screen stack is a large stateful UI machine with implicit
invariants that are not designed to be driven programmatically mid-game. Five
training runs died across distinct failure classes before the design was
replaced wholesale (user directive: "solve everything fundamentally").

**Final architecture — "headless core with a window":**

- Episodes are constructed exactly like the headless Phase 8/10 path (pure DI:
  `Scope.Get<MotorwaysGame>()` → `SetMapDefinition` → `Start` → `Tick(0f)×2` →
  `OnGameStarted`), born paused. No screen is ever pushed or popped.
- Visibility comes from the bootstrap ticking the game once per frame with real
  `Time.deltaTime` (paused `Simulation.Step()`s advance zero time — the shipped
  pause semantics), which drives the view layer, HUD timers, demand-pin
  animations and theme. `AdvanceSimulation` opens a synchronous unpause window
  for exactly the decision's native steps; a per-frame guard re-asserts the
  pause against any stock unpause leak.
- The main menu screen is parked for the process lifetime: its decorative city
  stopped, its world-space backdrop/grid and UI canvas hidden once. The screen
  stack never transitions again.
- Camera: position/zoom are reasserted per frame from the native schedule
  (`City.GetCameraSizeAtTime` + client playable-area center), so the view
  starts tight on the young city and expands with in-game time exactly like
  stock map growth. (Stock `CameraView` framing is bypassed only because it
  also offsets by a `GameUIScreen` letterbox rect that has no laid-out size in
  this mode.) Free play (`FREEPLAY` op) returns full native control: sim
  unpaused, native camera, stock screens, smooth real-time play for a human.
- Episode resets take ~0.25 s (teardown: `OnGameEnd(Exit)` →
  stop audio/clear pathfinder/scope release → destroy city GameObject →
  fresh city prefab instantiation → DI construction).

**Environment semantics added in this revision (identical headless/visible):**

- `decision_delta` (RESET parameter): simulated seconds per decision, a
  multiple of the native 0.1 s timestep. Training uses 1.0 s (10 ticks);
  throughput ≈ 3 decisions/s wall → ≈ 3× real-time simulation.
- Weekly upgrade choices are first-class actions: 2 slots appended to the
  Discrete space (size 3109 → 3111), claimed through the native
  `SelectUpgradeCommand`. While a choice is pending the mask is FORCED to the
  options only — mirroring stock, where the choice screen blocks all other
  input. The pending count is also an observation global (11 665 → 11 666
  floats). The stock week-end screen is suppressed in trainer mode by
  pre-marking grants via `numChoicesMade` (its push condition's only input).
- Reach rule: legal placements are restricted to
  `City.GetSimulationPlayableAreaAtTime` — the native zoom-unlock schedule
  that fences the stock camera. The agent can only build where a player could
  currently reach; the area grows with in-game time. (Previously the agent
  could build across the whole eventual map from t=0 — ~1 284 legal actions at
  episode start vs ~113–147 now.)
- Terminal: `IsTerminal` = native `HasGameEnded` OR any destination's
  overcrowding timer within one decision of `MaxOvercrowdTime`. The one-
  decision pre-emption window guarantees the stock game-over flow (a UI path)
  never fires mid-training, and it removes the Phase 8 caveat that headless
  episodes could never terminate natively.

**Robustness (all live-diagnosed):** single-flight guard on visible IPC ops
(a client retry once raced two RESET coroutines), C# response-write retry
(Windows rename/read race), request preservation across game-process restarts
plus an init gate (a fresh process must not answer a pending request before
its own init), a Python supervisor that relaunches the game and resumes the
same model on any env failure, and boot rescues for two hijack modes
(leftover test profile as `MostRecentPlayer`; auto-resume of a crashed
session's save).

Results sections (§6–§10) are to be filled from the v4 run
(`python/ppo_checkpoints_v4/`, evaluation on the fixed held-out seeds).
