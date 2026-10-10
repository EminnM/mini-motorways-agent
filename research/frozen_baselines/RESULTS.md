# Frozen baselines: held-out evaluation of the two 49,152-step checkpoints

Checkpoints (SHA-256 in SHA256.txt): `native_reward_connected_49152.zip` (sparse native reward, CONNECTED_START training),
`connection_shaped_normal_49152.zip` (potential-based connection shaping coefficient 1.0, gamma 0.995, NORMAL_START training).
Both: MaskablePPO MLP, 1 s decisions, 600-decision horizon. Evaluated on the 10 held-out EVAL_SEEDS (python/eval_seeds.py),
reward = native score delta + terminal penalty -10, one episode per seed, deterministic (argmax) unless marked stoch.
Raw per-episode data: eval_*.jsonl. Table: eval_table.txt. Curves: training_curves.png. Evaluator: python/phase11a_runner.py.

(see eval_table.txt for the exact numbers; summary in the session report)
