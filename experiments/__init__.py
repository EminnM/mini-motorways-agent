"""Reproducible experiment harness around the existing Mini Motorways RL stack.

Infrastructure only: it drives the existing python/train_ppo.py, python/mini_motorways_env.py and the
bootstrap DLL without changing environment semantics, observations, actions, masks, rewards, reset
behaviour or PPO defaults. See research/EXPERIMENT_HARNESS.md."""
