# Experiment harness

Infrastructure around the existing Mini Motorways RL stack. **No environment semantics changed**: the
PolicyObservation, action space/indexing, legal masks, native stepping, reset behaviour, reward, initialization
modes, PPO architecture/defaults, persistence isolation and frontend behaviour are untouched. The harness calls the
existing `python/train_ppo.py` `main()` unchanged (so an experiment launched through the harness trains exactly like
one launched by hand with the same flags) and attaches **passive** recording from the outside (see "How it is wired").
No file under `python/` or `game_plugin/` was modified for it.

## Commands

```
python -m experiments.run --config experiments/configs/baseline_native.yaml [--dry-run] [--no-eval]
python -m experiments.evaluate --run runs/<run_id>            --config experiments/configs/eval_heldout.yaml [--which final|all|step_xxxxxx.zip]
python -m experiments.evaluate --checkpoint runs/<id>/checkpoints/final.zip --config experiments/configs/eval_heldout.yaml
python -m experiments.evaluate --policy wait   --config experiments/configs/eval_heldout.yaml
python -m experiments.evaluate --policy random --config experiments/configs/eval_heldout.yaml
python -m experiments.compare runs/run_A runs/run_B [--eval evaluation/<eval_id>]
python -m experiments.reproduce runs/<run_id> [--check-only]
```

Run them from the project root (`mini moto/`). `--dry-run` validates and prints the fully resolved config and creates
nothing. `evaluate` never trains or updates a model. Each command launches its own headless game process (dedicated
IPC dir under `logs/ipc_exp...`, killed by its own PID afterwards) and never touches other instances.

## Directory structure

```
runs/2026-10-07_231202_smoke_test/
├── config.yaml            fully RESOLVED config actually used (defaults filled from the existing code)
├── metadata.json          see "Reproducibility metadata"
├── README.md              generated manifest: config, checkpoint, duration, evaluation status, measured headline numbers (no conclusions)
├── train_status.json
├── checkpoints/           step_000256.zip ... final.zip, index.json, index.csv (timestep, sha256, created_utc, run_id)
├── metrics/               training.csv, episodes.csv, actions.csv, action_indices.csv, sb3_progress.csv (raw SB3 log), raw/episodes_<pid>.jsonl
├── evaluation/            results.json, episodes.csv, actions.csv, action_indices.csv, episodes_full.jsonl, checkpoints.csv,
│                          checkpoints/step_xxxxxx/{...}   (per-checkpoint evaluations on the eval_every cadence)
│                          <eval_id>/{...}                  (each later `experiments.evaluate --run` invocation; never overwrites)
├── plots/                 return_vs_steps.png native_score_vs_steps.png episode_length_vs_steps.png action_distribution.png eval_score_vs_steps.png
└── logs/                  train_stdout.log (raw SB3/trainer output), config_as_given.yaml, source_snapshot.zip, git_diff.patch (only when a git repo is dirty)
```

* Run ids are `YYYY-MM-DD_HHMMSS_<name>`; directories are created exclusively (`os.makedirs`, never `exist_ok`), a
  numeric suffix is added on collision; config/checkpoints are written with exclusive modes. Nothing is overwritten.
* **Several training seeds**: with exactly one seed the layout above is flat; with several, each seed gets the same layout
  under `seeds/seed_<s>/` and the top-level `metadata.json` lists per-seed summaries.
* `evaluate --policy/--checkpoint` (no `--run`) creates its own run directory `runs/<ts>_eval_<name>_<label>/` (kind
  `evaluation`, with config, metadata, results).

## Config schema (YAML, strictly validated)

Unknown keys, wrong types, bad enums, duplicate seeds, a decision interval that is not a multiple of 0.1 s, and
`reward: connection_shaped` without an explicit `shaping_coefficient` all fail **before** anything is created.

| key | meaning / mapping to the existing code | default (read from the code) |
|---|---|---|
| `experiment.name` (req), `description`, `hypothesis`, `tags` | labels; `hypothesis` is echoed verbatim in README.md | – |
| `environment.initialization` | `normal`→`NORMAL_START`, `connected`→`CONNECTED_START` (`--start-mode`) | train_ppo default |
| `environment.start_gap` | `--start-gap` (CONNECTED_START with N edits left unbuilt) | train_ppo default (0) |
| `environment.reward` | `native_score` or `connection_shaped` (`--shaping <coef>`, uses `shaping_gamma = gamma`) | `native_score` |
| `environment.shaping_coefficient` | required iff `connection_shaped` | none invented |
| `environment.decision_interval_seconds` | `--decision-delta` | train_ppo default |
| `environment.episode_horizon_decisions` | `--max-decisions` | train_ppo default |
| `policy.algorithm` | only `maskable_ppo` | `maskable_ppo` |
| `policy.network` | `mlp` / `cnn` (`--net`) | train_ppo default |
| `policy.hyperparameters` | `n_steps batch_size learning_rate n_epochs target_kl ent_coef gamma` (the ones train_ppo exposes); `gae_lambda`=0.95 and `clip_range`=0.2 are fixed inside train_ppo.py and only reported | train_ppo defaults |
| `training.total_timesteps` (req), `seeds` (req) | PPO/SB3 seed per run (`--seed`); it also seeds the first game seed | – |
| `training.checkpoint_every`, `eval_every`, `n_envs`, `headless`, `torch_threads` | `--checkpoint-every-steps`, post-training evaluation cadence, `--n-envs`, `--headless`, `--torch-threads` | train_ppo defaults / headless `true` |
| `evaluation.seeds` (req), `episodes_per_seed`, `deterministic`, `random_seed` | held-out game seeds; `random_seed` seeds the RANDOM baseline | 1 / true / 0 |
| `logging.action_distribution`, `training_curve`, `episode_metrics` | accepted and recorded in the resolved config; the files are always produced (they are cheap and the raw data is authoritative) | true |

An evaluation config accepts `experiment`, `environment` (any subset), `evaluation`, `logging`. For `--run` /
`--checkpoint` of a harness run the run's own environment is the base and the eval config's `environment` overrides
it; for `--policy wait|random` the existing code defaults apply (normal start, 0.3 s, 30-decision horizon) unless the
eval config sets them — the environment used is echoed in `results.json`. `eval_heldout_connected_600.yaml` is an
explicit variant matching the frozen connected-start baselines.

## Output files

* `metrics/training.csv` — per PPO iteration: `timestep, wall_time, simulated_game_time (=timestep×decision interval),
  episodes_completed, episode_return (SB3 ep_rew_mean), episode_native_score (mean of the last 100 recorded episodes),
  episode_length, SPS, policy_loss, value_loss, entropy (=−entropy_loss), approx_kl, clip_fraction, explained_variance`.
  Derived from SB3's own logger (`sb3_progress.csv`, kept raw). SB3 dumps `train/*` values of iteration *k* in row
  *k+1*, so the first row's PPO columns are empty. Nothing was added to training to manufacture metrics.
* `metrics/episodes.csv` / `evaluation/episodes.csv` — one row per episode: `episode, training_timestep, game_seed,
  initialization, start_gap, reward_mode, return, native_score, length_decisions, length_sim_seconds, terminal,
  truncated, wait/add/remove/upgrade counts and fractions, wall_seconds`. `return` is the reward the agent received
  (native score delta + terminal penalty, plus the shaping term in shaped training); `native_score` is the sum of the
  environment's `rewardComponents.scoreDelta` (excludes the terminal penalty and any shaping).
* `actions.csv` — category totals and fractions (`WAIT, ADD_ROAD, REMOVE_ROAD, UPGRADE`); `action_indices.csv` — per
  action index: count, `legal_when_selected` (always equal to count under masking; kept to detect mask problems).
  Training and evaluation are written separately (`metrics/` vs `evaluation/`).
* `evaluation/results.json` — `checkpoint`, `checkpoint_sha256`, linked `checkpoint_run_id`/`training_timestep`,
  `num_episodes`, mean/median/std native score and 95% CI, mean return and CI, terminal rate with Wilson CI, mean
  episode length, action counts/fractions, evaluated environment, seeds, observed observation shape/action-space size.
  All episodes are preserved (`episodes.csv`, `episodes_full.jsonl`) so every statistic is recomputable. CIs are
  normal approximations over episodes (episodes sharing a seed are not independent).

Baselines use the identical evaluation loop (same seeds, initialization, horizon, environment, metrics) and mirror
`python/phase11a_runner.py`'s policy definitions so numbers are comparable with the already-frozen baselines:
WAIT (claims upgrade option 0 only when WAIT is illegal), RANDOM (uniform over the native legal actions,
`numpy.default_rng(random_seed)`), PPO (`MaskablePPO.predict` with the native mask, deterministic per config).

## How it is wired (no edits to the existing code)

`experiments.run` → writes config/metadata → spawns `python -m experiments.train_worker` per training seed. The worker
imports `train_ppo`, then **by reassigning module names from the outside**: wraps `train_ppo.MiniMotorwaysEnv` and
`mini_motorways_env.make_monitored_env` (multi-env path) with a pass-through `RecorderEnv` (reads reward/`info`/the
env's own mask; forwards everything else), subclasses `train_ppo.CheckpointCallback` (registers each checkpoint with
SHA-256, adds a CSV output to SB3's logger), sets `sys.argv` from the resolved config and calls `train_ppo.main()`.
The trainer's supervisor/relaunch logic, PPO construction, rollout handling and checkpoint timing are therefore the
unmodified code paths. After training the CSVs are derived from the raw records, evaluation runs in a fresh game
instance, plots and README are generated.

## Reproducibility metadata (`metadata.json`)

`run_id, start_time, end_time, status, wall_clock_seconds, mean_sps (+note), training/evaluation seeds, resolved
environment and PPO hyperparameters (incl. the fixed ones), observation shape and action-space size (expected from the
code and observed from the live environment), config_sha256, command line, python executable;`
code identity: `git_commit/git_dirty` (null when not a git repo) **and** `source_tree_sha256` + per-file SHA-256 of
`python/*.py`, `experiments/**`, configs, `Entrypoint.cs`, the csproj, plus `logs/source_snapshot.zip` (the exact
sources) — this project directory is **not** a git repository, so these replace the commit id;
binaries: SHA-256 of the *deployed* bootstrap DLL (the file the game actually loads, `Bootstrap\MiniMotorwaysBootstrap.dll`),
of the freshly *built* DLL and of `Entrypoint.cs`, game build id (Steam manifest), Unity version, game exe SHA-256;
machine: OS, CPU, core count, GPU/CUDA (null when unused), versions of Python, Gymnasium, SB3, sb3-contrib, PyTorch,
NumPy, PyYAML, Matplotlib. Anything undeterminable is `null`.
If a git repo is ever initialized and dirty, `git_dirty: true` and `logs/git_diff.patch` are written (experiments are
never blocked by a dirty tree).

Traceability chain: plot/table → CSV in the run → `checkpoints/index.json` (file, timestep, SHA-256, run id) →
`results.json.checkpoint_sha256` → `metadata.json` (config hash, source hashes, DLL hashes, game build, libraries) →
`logs/source_snapshot.zip` / `config.yaml`.

## Reproduction

`python -m experiments.reproduce runs/<id> [--check-only]` re-reads the resolved config, recomputes the current
environment snapshot and prints every difference (DLL hashes, game build, library versions, OS/CPU, each changed source
file) with a clear **"this will NOT be an exact reproduction"** warning; without `--check-only` it then launches a new
run whose metadata records `reproduces: <old run id>`. It never claims exactness.

## What is guaranteed / what is not

Guaranteed: config validated before launch; resolved config stored; no overwrite of runs/checkpoints; every checkpoint
hashed and linked to its run; the environment/trainer code path is the existing one; baselines and checkpoints are
evaluated by one shared loop; derived files are re-creatable from raw records.

Not guaranteed / not captured:
* **Bit-exact training reproduction.** The simulation is deterministic for a given seed and actions, but PPO training
  runs against a live game process with asynchronous IPC, multi-threaded PyTorch and (with `n_envs>1`) parallel game
  instances; expect statistically similar, not identical, curves. `reproduce` therefore verifies *inputs*, not outputs.
* The game's savegame/profile state is isolated by the existing persistence isolation; the harness does not snapshot it.
* Game window/frontend settings, other processes' CPU load (it changes SPS, not results), the Windows audio mixer.
* Wall-clock/SPS comparisons across runs when other jobs share the machine.

## Known limitations

1. `training.seeds` is the SB3/PPO seed per run (it also seeds the first game seed; later game seeds come from the
   environment's RNG chain). Multiple seeds are run sequentially, not in parallel.
2. `eval_every` is implemented as a **post-training** evaluation of the first checkpoint at/after each multiple (train_ppo
   has no in-training evaluation and may not be changed). Checkpoint file names use the actual timestep (SB3 rounds
   to whole rollouts), e.g. `step_010240.zip`, not the nominal multiple.
3. With `n_envs>1` the harness' `training_timestep` in `episodes.csv` is approximate (cumulative decisions ordered by
   completion time); the raw records and SB3 progress are exact. Multi-env mode wiring (picklable replacement factory)
   exists but was **not** exercised by the smoke test.
4. The smoke test (512 timesteps, 20-decision episodes, one seed) only proves the plumbing; its scores are meaningless.
5. Windows paths of the game install are those hard-coded in `train_ppo.py`; `logging.*` switches only record intent.
6. Evaluation of a shaped-reward checkpoint is native by default (shaping applied only if the eval config asks for
   `connection_shaped`), so its `return` is directly comparable with other policies' native returns.
7. `mean_sps` includes game start-up; use `metrics/training.csv` SPS for per-iteration throughput.
8. The deployed DLL hash and the freshly built DLL hash can legitimately differ (the build output is not always
   deployed); the deployed one is what ran.

## Smoke test (done at implementation time)

`experiments/configs/smoke_test.yaml` — 1 seed, 512 timesteps, 20-decision episodes: produced a complete run
(`runs/2026-10-07_231202_smoke_test`): 3 checkpoints with SHA-256 index, `training.csv`/`episodes.csv`/`actions.csv`,
primary + per-checkpoint evaluations, all plots, metadata and README. `evaluate --policy wait|random`, `evaluate --run
... --which step_000256.zip`, `compare` (3 runs) and `reproduce --check-only` (correctly warned about the 3 harness
files edited after that run) were also exercised. A bad config (unknown keys, bad enums, duplicate seeds, missing
shaping coefficient) is rejected before any directory is created. The running held-out PPO evaluation was not
touched (separate instances and IPC directories).
