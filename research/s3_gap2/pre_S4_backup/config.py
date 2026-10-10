"""YAML experiment config: schema, defaults, validation, resolution.

Defaults are READ from the existing code (python/train_ppo.py's argparse defaults and
MiniMotorwaysEnv's signature) instead of being restated here. Unknown keys fail loudly."""
import copy
import inspect
import os
import re
from typing import Any, Dict

import yaml

from . import paths  # noqa: F401  (puts python/ on sys.path)

INIT_TO_START_MODE = {"normal": "NORMAL_START", "connected": "CONNECTED_START"}
REWARD_MODES = ("native_score", "connection_shaped", "completion", "native_plus_completion")   # last two: Gap-1 tasks only
GAP1_KEYS = {"manifest_dir": str, "construction_window": int, "continuation_decisions": int, "beta": (int, float),
             "train_split": str, "validation_split": str, "success_terminates": bool}


class ConfigError(ValueError):
    pass


_DEFAULTS_CACHE = None


def existing_defaults() -> Dict[str, Any]:
    """argparse defaults of python/train_ppo.py, obtained by letting its own main() build its parser
    and aborting right after parse_args (no training, no game launch)."""
    global _DEFAULTS_CACHE
    if _DEFAULTS_CACHE is not None:
        return dict(_DEFAULTS_CACHE)
    import argparse
    import sys

    import train_ppo

    class _Stop(Exception):
        pass

    captured = {}
    original = argparse.ArgumentParser.parse_args

    def fake(self, args=None, namespace=None):
        ns = original(self, [], None)       # defaults only (all trainer args are optional)
        captured.update(vars(ns))
        raise _Stop()

    argparse.ArgumentParser.parse_args = fake
    saved_argv = sys.argv
    try:
        sys.argv = ["train_ppo"]
        train_ppo.main()
    except _Stop:
        pass
    finally:
        argparse.ArgumentParser.parse_args = original
        sys.argv = saved_argv
    if not captured:
        raise RuntimeError("could not read train_ppo defaults")
    _DEFAULTS_CACHE = captured
    return dict(captured)


def env_defaults() -> Dict[str, Any]:
    from mini_motorways_env import MiniMotorwaysEnv
    sig = inspect.signature(MiniMotorwaysEnv.__init__)
    return {k: v.default for k, v in sig.parameters.items() if v.default is not inspect._empty}


# --------------------------------------------------------------------------- schema
# Each section: {key: (type or tuple of types, required?)}.  None => value must be a mapping handled specially.
SCHEMA = {
    "experiment": {"name": (str, True), "description": (str, False), "hypothesis": (str, False), "tags": (list, False)},
    "environment": {"task": (str, False), "gap1": (dict, False),
                    "initialization": (str, False), "start_gap": (int, False), "reward": (str, False),
                    "shaping_coefficient": ((int, float), False), "decision_interval_seconds": ((int, float), False),
                    "episode_horizon_decisions": (int, False)},
    "policy": {"algorithm": (str, False), "network": (str, False), "hyperparameters": (dict, False)},
    "training": {"total_timesteps": (int, True), "seeds": (list, True), "checkpoint_every": (int, False),
                 "eval_every": (int, False), "n_envs": (int, False), "headless": (bool, False), "torch_threads": (int, False)},
    "evaluation": {"seeds": (list, True), "episodes_per_seed": (int, False), "deterministic": (bool, False),
                   "random_seed": (int, False)},
    "logging": {"action_distribution": (bool, False), "training_curve": (bool, False), "episode_metrics": (bool, False)},
}
HYPERPARAMS = {"n_steps": "n_steps", "batch_size": "batch_size", "learning_rate": "learning_rate", "n_epochs": "n_epochs",
               "target_kl": "target_kl", "ent_coef": "ent_coef", "gamma": "gamma"}
# Hyperparameters fixed inside train_ppo.py (not CLI-configurable) -- reported, not settable:
FIXED_PPO = {"gae_lambda": 0.95, "clip_range": 0.2}

EVAL_SCHEMA = {
    "experiment": {"name": (str, False), "description": (str, False), "hypothesis": (str, False), "tags": (list, False)},
    "environment": SCHEMA["environment"],
    "evaluation": SCHEMA["evaluation"],
    "logging": SCHEMA["logging"],
}


def _check_section(name: str, value: Any, spec: Dict[str, Any], problems: list):
    if value is None:
        value = {}
    if not isinstance(value, dict):
        problems.append(f"'{name}' must be a mapping")
        return
    for k, v in value.items():
        if k not in spec:
            problems.append(f"unknown key '{name}.{k}' (allowed: {sorted(spec)})")
            continue
        typ, _ = spec[k]
        if isinstance(v, bool) and typ is not bool and bool not in (typ if isinstance(typ, tuple) else (typ,)):
            problems.append(f"'{name}.{k}' must be {typ}, got bool")
        elif not isinstance(v, typ):
            problems.append(f"'{name}.{k}' must be {typ}, got {type(v).__name__} ({v!r})")
    for k, (typ, req) in spec.items():
        if req and k not in value:
            problems.append(f"missing required key '{name}.{k}'")


def validate(cfg: Dict[str, Any], schema: Dict[str, Dict[str, Any]], what: str) -> None:
    problems = []
    if not isinstance(cfg, dict):
        raise ConfigError(f"{what}: top level must be a mapping")
    for sec in cfg:
        if sec not in schema:
            problems.append(f"unknown top-level section '{sec}' (allowed: {sorted(schema)})")
    for sec, spec in schema.items():
        if sec in cfg:
            _check_section(sec, cfg[sec], spec, problems)
        elif any(req for _, req in spec.values()):
            problems.append(f"missing required section '{sec}'")
    env = cfg.get("environment") or {}
    if "initialization" in env and env["initialization"] not in INIT_TO_START_MODE:
        problems.append(f"environment.initialization must be one of {sorted(INIT_TO_START_MODE)}, got {env['initialization']!r}")
    if "reward" in env and env["reward"] not in REWARD_MODES:
        problems.append(f"environment.reward must be one of {REWARD_MODES}, got {env['reward']!r}")
    task = env.get("task")
    if task not in (None, "gap1"):
        problems.append("environment.task must be 'gap1' when given")
    if task == "gap1":
        g = env.get("gap1") or {}
        for k, v in g.items():
            if k not in GAP1_KEYS:
                problems.append(f"unknown key 'environment.gap1.{k}' (allowed: {sorted(GAP1_KEYS)})")
            elif not isinstance(v, GAP1_KEYS[k]) or (isinstance(v, bool) and GAP1_KEYS[k] is not bool):
                problems.append(f"'environment.gap1.{k}' has wrong type")
        if "manifest_dir" not in g:
            problems.append("environment.gap1.manifest_dir is required for task gap1")
        if env.get("initialization", "connected") != "connected":
            problems.append("task gap1 requires initialization: connected")
        if env.get("start_gap", 0) != 0:
            problems.append("task gap1 defines its own gap; do not set environment.start_gap")
        if (cfg.get("training") or {}).get("n_envs", 1) != 1:
            problems.append("task gap1 supports training.n_envs == 1 only")
        if env.get("reward") == "native_plus_completion" and "beta" not in g:
            problems.append("reward native_plus_completion requires environment.gap1.beta (no default is invented)")
    else:
        if "gap1" in env:
            problems.append("environment.gap1 is only valid with task: gap1")
        if env.get("reward") in ("completion", "native_plus_completion"):
            problems.append(f"environment.reward={env.get('reward')} is only valid with task: gap1")
    if env.get("reward") == "connection_shaped" and "shaping_coefficient" not in env:
        problems.append("environment.reward=connection_shaped requires environment.shaping_coefficient (no default is invented)")
    if env.get("reward", "native_score") != "connection_shaped" and "shaping_coefficient" in env:
        problems.append("environment.shaping_coefficient is only valid with reward=connection_shaped")
    di = env.get("decision_interval_seconds")
    if di is not None and (di < 0.1 or abs(round(di * 10) - di * 10) > 1e-6):
        problems.append("environment.decision_interval_seconds must be a multiple of 0.1 (simulation tick) and >= 0.1")
    pol = cfg.get("policy") or {}
    if "algorithm" in pol and pol["algorithm"] != "maskable_ppo":
        problems.append("policy.algorithm: only 'maskable_ppo' is supported")
    if "network" in pol and pol["network"] not in ("mlp", "cnn"):
        problems.append("policy.network must be 'mlp' or 'cnn'")
    for k in (pol.get("hyperparameters") or {}):
        if k not in HYPERPARAMS:
            problems.append(f"unknown key 'policy.hyperparameters.{k}' (allowed: {sorted(HYPERPARAMS)}; "
                            f"gae_lambda/clip_range are fixed inside train_ppo.py)")
    tr = cfg.get("training") or {}
    for k in ("total_timesteps", "checkpoint_every", "eval_every", "n_envs", "torch_threads"):
        if k in tr and tr[k] <= 0:
            problems.append(f"training.{k} must be positive")
    for sec, key in (("training", "seeds"), ("evaluation", "seeds")):
        seeds = (cfg.get(sec) or {}).get(key)
        if seeds is not None:
            if not seeds or not all(isinstance(s, int) and not isinstance(s, bool) for s in seeds):
                problems.append(f"{sec}.{key} must be a non-empty list of integers")
            elif len(set(seeds)) != len(seeds):
                problems.append(f"{sec}.{key} contains duplicates")
    if problems:
        raise ConfigError(f"{what} is invalid:\n  - " + "\n  - ".join(problems))


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if data is not None else {}


def resolve_training(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Fully resolved config: every value either user-supplied or read from the existing code defaults."""
    validate(cfg, SCHEMA, "training config")
    d = existing_defaults()
    r = copy.deepcopy(cfg)
    env = r.setdefault("environment", {})
    env.setdefault("initialization", "normal" if d["start_mode"] == "NORMAL_START" else "connected")
    env.setdefault("start_gap", d["start_gap"])
    env.setdefault("reward", "native_score")
    env.setdefault("decision_interval_seconds", d["decision_delta"])
    env.setdefault("episode_horizon_decisions", d["max_decisions"])
    if env.get("task") == "gap1":
        env["initialization"] = "connected"
        g = env["gap1"]
        g.setdefault("construction_window", 20)
        g.setdefault("continuation_decisions", 40)
        g.setdefault("train_split", "train")
        g.setdefault("validation_split", "validation")
        g.setdefault("success_terminates", False)
        env["episode_horizon_decisions"] = g["construction_window"] + g["continuation_decisions"] + 40   # native cap only; the task wrapper ends episodes itself
    env["start_mode"] = INIT_TO_START_MODE[env["initialization"]]      # derived, informational
    pol = r.setdefault("policy", {})
    pol.setdefault("algorithm", "maskable_ppo")
    pol.setdefault("network", d["net"])
    hp = pol.setdefault("hyperparameters", {})
    for k, arg in HYPERPARAMS.items():
        hp.setdefault(k, d[arg])
    pol["fixed_hyperparameters"] = dict(FIXED_PPO)
    tr = r["training"]
    tr.setdefault("checkpoint_every", d["checkpoint_every_steps"])
    tr.setdefault("eval_every", tr["checkpoint_every"])
    tr.setdefault("n_envs", d["n_envs"])
    tr.setdefault("headless", True)
    tr.setdefault("torch_threads", d["torch_threads"])
    ev = r["evaluation"]
    ev.setdefault("episodes_per_seed", 1)
    ev.setdefault("deterministic", True)
    lg = r.setdefault("logging", {})
    for k in ("action_distribution", "training_curve", "episode_metrics"):
        lg.setdefault(k, True)
    exp = r["experiment"]
    exp.setdefault("description", "")
    exp.setdefault("hypothesis", "")
    exp.setdefault("tags", [])
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", exp["name"]):
        raise ConfigError("experiment.name may only contain letters, digits, '_', '-', '.'")
    return r


def resolve_eval(cfg: Dict[str, Any], base_env: Dict[str, Any] = None) -> Dict[str, Any]:
    """Evaluation config; environment values fall back to base_env (a run's resolved environment) then to
    the existing code defaults."""
    validate(cfg, EVAL_SCHEMA, "evaluation config")
    d = existing_defaults()
    r = copy.deepcopy(cfg)
    env = r.setdefault("environment", {})
    base = base_env or {}
    env.setdefault("initialization", base.get("initialization", "normal" if d["start_mode"] == "NORMAL_START" else "connected"))
    env.setdefault("start_gap", base.get("start_gap", d["start_gap"]))
    env.setdefault("reward", base.get("reward", "native_score"))
    if env["reward"] == "connection_shaped" and "shaping_coefficient" not in env and "shaping_coefficient" in base:
        env["shaping_coefficient"] = base["shaping_coefficient"]
    env.setdefault("decision_interval_seconds", base.get("decision_interval_seconds", d["decision_delta"]))
    env.setdefault("episode_horizon_decisions", base.get("episode_horizon_decisions", d["max_decisions"]))
    env["start_mode"] = INIT_TO_START_MODE[env["initialization"]]
    ev = r["evaluation"]
    ev.setdefault("episodes_per_seed", 1)
    ev.setdefault("deterministic", True)
    lg = r.setdefault("logging", {})
    for k in ("action_distribution", "training_curve", "episode_metrics"):
        lg.setdefault(k, True)
    r.setdefault("experiment", {}).setdefault("name", "evaluation")
    return r


def load_training_config(path: str) -> Dict[str, Any]:
    return resolve_training(load_yaml(path))


def load_eval_config(path: str, base_env: Dict[str, Any] = None) -> Dict[str, Any]:
    return resolve_eval(load_yaml(path), base_env)


def trainer_argv(r: Dict[str, Any], seed: int, checkpoint_dir: str, ipc_dir: str, game_log: str, tag: str) -> list:
    """Translate a resolved config into train_ppo.py CLI arguments (the ONLY path into training)."""
    env, pol, tr = r["environment"], r["policy"], r["training"]
    hp = pol["hyperparameters"]
    argv = ["--total-timesteps", str(tr["total_timesteps"]),
            "--max-decisions", str(env["episode_horizon_decisions"]),
            "--decision-delta", repr(float(env["decision_interval_seconds"])),
            "--n-steps", str(hp["n_steps"]), "--batch-size", str(hp["batch_size"]),
            "--learning-rate", repr(float(hp["learning_rate"])), "--n-epochs", str(hp["n_epochs"]),
            "--ent-coef", repr(float(hp["ent_coef"])), "--gamma", repr(float(hp["gamma"])),
            "--checkpoint-dir", checkpoint_dir, "--checkpoint-every-steps", str(tr["checkpoint_every"]),
            "--skip-post-eval", "--torch-threads", str(tr["torch_threads"]),
            "--start-mode", env["start_mode"], "--start-gap", str(env["start_gap"]),
            "--net", pol["network"], "--seed", str(seed),
            "--ipc-dir", ipc_dir, "--game-log", game_log, "--launch-game",
            "--n-envs", str(tr["n_envs"]), "--tag", tag]
    if hp["target_kl"] is not None:
        argv += ["--target-kl", repr(float(hp["target_kl"]))]
    if tr["headless"]:
        argv += ["--headless"]
    if env["reward"] == "connection_shaped":
        argv += ["--shaping", repr(float(env["shaping_coefficient"]))]
    return argv
