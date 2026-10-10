# S34 — Native RL replication: preregistration (written before any S34 run started, 2026-10-09)

Environment: `postfix_v8` (DLL sha 2aa42367…, conservative stock-boundary mask as documented in `PLAYABLE_AREA_FIX_S33.md` §3; not widened).

## Training (2 independent replications)
* Code: `research/s31_hybrid_native_rl/code/{hybrid_env,hybrid_policy}.py` unchanged; trainer `research/s34_replication/code/train_rep.py` = S31 `train.py` with only the run directory, instance-name prefix and PPO seed parametrised.
* Identical to S31: HybridMaskablePolicy initialised from `s8c_migrated_v2_step0.zip` (PLAN bias 6.0), 4 SubprocVecEnv workers, n_steps 512/env, batch 256, 4 epochs, target_kl 0.03, lr 3e-4, ent_coef 0.01, γ 0.99, λ 0.95, max 600 decisions/episode, reward = native score delta − 10 terminal, training map seeds 700000 + 7919·i (i < 40) cycled per worker, budget 100,352 decisions, checkpoints at 0 / 24,576 / 49,152 / 73,728 / 100,352.
* Only difference: PPO/torch seed **131 (rep1)** and **231 (rep2)** (S31 used 31). Runs: `research/s34_replication/runs/rep1_seed131`, `rep2_seed231`. The two runs train concurrently (8 headless instances); wall-time may differ from S31, decisions do not.
* No hyper-parameter, reward, exploration or architecture change, whatever the S31 outcome suggested.

## Evaluation A — development benchmark (12 designated seeds 500000 + 7919·i)
* Checkpoints: **50k (49,152) and 100k (100,352)** of each replication; S31's same checkpoints already exist pre-fix and are re-run post-fix here for environment-version consistency (S31 50k sto post-fix exists from S33 and is reused).
* Modes: deterministic (1 rollout) and stochastic (1 rollout, torch seed 8008 + seed-index, exactly as S31). Baseline: planner B post-fix (S33 regression, identical to pre-fix).
* Status: development benchmark — these seeds informed the S31 go/no-go and the choice of 50k as the "best" checkpoint; results are reported but are not the robustness decision.

## Evaluation B — longer-horizon, fresh seeds (the decision benchmark)
* Seeds: **900000 + 7919·i, i = 0..7** (900000, 907919, 915838, 923757, 931676, 939595, 947514, 955433). Verified unused anywhere in `research/`, `python/`, `experiments/` (only substring matches inside floats).
* Horizon: **1,500 decisions** (fixed now; S31's 600 cap saturated on 4/12 seeds).
* Policies: planner B; S31 ck50k, ck100k; rep1 ck50k, ck100k; rep2 ck50k, ck100k.
* Modes: deterministic 1 rollout per seed; stochastic **3 rollouts per seed**, torch seed = 20000 + 1000·r + i (r = rollout 0..2, i = seed index), the same RNG seeds for every policy.
* All policies on the same seeds and horizon; one game instance per process; same `HybridNativeEnv` wrapper.

## Analysis (fixed)
* Metrics per episode: deliveries (score), survived-to-horizon, episode length, PLAN/EDIT/WAIT/UPG shares, destroyed connections, illegal actions, env version.
* Paired per-seed differences vs planner B; for stochastic, pair on the per-seed *mean over the 3 rollouts*.
* Variation reported separately: (i) **training-run variation** = spread of per-run totals across S31 / rep1 / rep2 at the same checkpoint (n = 3 runs); (ii) **rollout variation** = within-run SD across the 3 stochastic rollouts per seed. No pooling of rollouts across runs as independent evidence.
* Robustness decision rule (set now): the hybrid is "robust enough for further investment" if, on Evaluation B, **every** training run's 100k stochastic checkpoint (mean over rollouts) beats planner B in total deliveries over the 8 fresh seeds with ≥ 5 of 8 seeds higher, *and* the worst run's advantage is at least half the best run's; otherwise "not robust" (a single-run success is not replication). 50k results are reported alongside but the rule uses 100k (the preregistered endpoint).
* Stop conditions: any illegal action, any episode where Σ rewards ≠ score + penalty, or any run whose `globals.playableAreaAtTime` export is missing (environment-version mismatch) stops the phase.
