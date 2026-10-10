# Gap-1 curriculum — experimental specification (as given) + implementation decisions

The specification text is the user's, reproduced in the session; the binding points are summarised here together with the
decisions that had to be made to implement it. Nothing below changes PolicyObservation, action indexing/semantics,
legal masking, native stepping, the decision interval (1.0 s as in Baseline-0), PPO architecture or hyperparameters.

## Task (spec §3-§8)
Every episode is a real native CONNECTED_START game (`python/mini_motorways_env.py` unchanged apart from two additive
attributes, see below) with exactly ONE edit of the native road path left out. The policy sees the ordinary
PolicyObservation: no demonstration, no forced action, no extra features. a* (the left-out edit), the house and the
destination are evaluation metadata inside the task wrapper only.

### Native generation (what was added, all optional / default-off)
* `Entrypoint.cs`: optional RESET field `start_gap_index`. `< 0` (default) = old behaviour; `0..N` = omit exactly the path
  edit at that index; `>= 1000` = omit the edit at per-mille position `(v-1000)/1000` along the path. initInfo now also
  returns `fullPathEdits`, `pathLengthEdits`, `gapIndex`. (The old `start_gap=N` always removed the LAST N edits, i.e.
  the gap always touched the destination; the new option varies the gap position along the path.)
* `python/mini_motorways_env.py`: `extra_reset_args` (dict merged into the RESET request, default empty) and
  `last_raw_observation` (read-only copy of the last native observation dict, for evaluators). No behavioural change.
* Layout variation comes from the real game: the seed selects the initial house/destination placement and the planner's
  road path. 38/40 random seeds gave distinct layouts in a pilot (1.4 s per reset).

### Validity checks per generated configuration (real game, real STEP)
* a* is in the native legal-action mask at reset;
* the target house is NOT connected to the target destination at reset;
* one real STEP of a* connects them (native live road graph; Pending/Active/VisiblyActive edges count, Mothballed do not);
* configuration signature unique; splits assigned by BASE LAYOUT (house, destination, full path), so no layout appears in
  two splits (also checked: no shared configuration, base layout or game seed).

### Known structural limitation (reported, not hidden)
At t = 0 only a small part of the map is reachable under the native reach rule, so houses/destinations/gaps lie in the
left-central region (gap x about -18..0, y about -10..7). Distributions are reported in
`manifests/distribution_report.txt` / `distributions.png`. Generalisation across the full map is NOT tested by this study.

## Episode protocol (spec §9-§11)
* Construction window K = 20 policy decisions.
* SUCCESS = target house connected to target destination (native live road graph). First action correct and
  solve-within-K are both reported. A connection achieved by another edit than a* would still count as success
  (reported via `a_star_added_at`); none were expected given exactly one missing edge.
* DESTRUCTIVE FAILURE = any pre-existing required path edge no longer live -> episode terminates (terminated=True).
* TIMEOUT = K decisions without success -> truncated. NATIVE TERMINAL reported separately (never counted as timeout).
* Outcome continuation: after SUCCESS the episode continues for 40 decisions in which the only legal action is WAIT
  (the wrapper restricts the mask; the policy is not asked to keep editing); native score earned meanwhile is the measured
  outcome (`native_score_after_fixed_continuation`). The oracle first-score delay after the correct edit is ~11-15
  decisions (pilot), hence 40. For VALIDATION (checkpoint selection) the continuation is skipped for speed.

## Rewards (spec §15-§17), PPO identical across variants
Hyperparameters = the frozen Baseline-0 native run (n_steps 512, batch 256, n_epochs 4, target_kl 0.03, lr 3e-4,
ent_coef 0.01, gamma 0.99, MLP). Budget 100k environment steps per run. Checkpoints every ~20k steps.
* A native: `r = native score delta (+ native terminal penalty if the game itself ends)`. Delivery credit arrives
  ~11-15 decisions after the edit, through the forced-WAIT continuation, exactly like the native stream.
* B completion: `r = 1` at the success step, else 0 (native reward ignored).
* C native + completion: `r = native + beta * completion`, beta = 5 (fixed here, before any run; one native delivery is +1).
* D connection shaping: the existing implementation unchanged (coefficient 1.0; `shaping_gamma` = PPO gamma, i.e. 0.99
  here whereas the original shaped Baseline-0 used 0.995 -- the only difference, needed to keep PPO identical across A-D).
No road penalty, REMOVE penalty, WAIT bonus, distance/partial-path shaping, counterfactual reward, CNN/GNN, BC,
demonstrations or Gap-N curriculum is used.

## Splits and checkpoint selection (spec §6, §18-§19)
TRAIN >= 600, VALIDATION >= 120, TEST >= 240 configurations (stable ids `gap1_<split>_<n>`), frozen in
`manifests/*.jsonl` with SHA-256 in `manifests/SHA256.txt`. Each record stores game seed, `start_gap_index`, a*, geometry,
full path, signatures, native-oracle metadata. Checkpoints are scored on VALIDATION only; the selection rule is fixed in
`experiments/gap1/select.py` (highest validation completion; tie -> lower median steps to success; tie -> earlier
checkpoint) and written to `evaluation/selection.json` before TEST is touched. TEST is evaluated once per frozen
selection (`python -m experiments.gap1.final_test`).

## Success criterion (spec §21)
Learned = selected policy >= 80% held-out Gap completion AND substantially above RANDOM LEGAL; strong = >= 90%. Also
required: not explained by one dominant action index or a narrow map region (reported in the breakdowns).

## Execution order (spec §25)
1 manifests + validation -> 2 WAIT / RANDOM / frozen Baseline-0 on TEST -> 3 Experiment A (3 seeds) -> 4 freeze + evaluate A
-> branch (A succeeds -> zero-shot Gap-2; A fails -> B; B succeeds -> C, D; B fails -> STOP).

## Tooling
`experiments/gap1/{generate,build_manifests,gap1_env,evaluation,evaluate_cli,select,final_test}.py`; configs
`experiments/configs/gap1_{native,completion,native_plus_completion,connection_shaped}.yaml`; runs via the experiment
harness (`python -m experiments.run --config ...`), every run directory carrying config, metadata (DLL/game/source
hashes), checkpoint index, metrics, plots.
