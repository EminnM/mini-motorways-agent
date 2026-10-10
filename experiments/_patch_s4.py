"""One-off patch script for S4 (P2 wiring). Each replacement must match exactly once or the script aborts before writing."""
import sys


def patch(path, pairs):
    s = open(path, encoding="utf-8").read()
    for old, new in pairs:
        if s.count(old) != 1:
            sys.exit(f"{path}: pattern matched {s.count(old)}x: {old[:70]!r}")
        s = s.replace(old, new)
    open(path, "w", encoding="utf-8").write(s)
    print("patched", path)


patch("experiments/recording.py", [
    ("        self.meta = dict(meta)\n        self.reset_episode(None)",
     "        self.meta = dict(meta)\n        self.diag = None\n        if self.meta.get(\"diagnostics\"):\n"
     "            from .diagnostics import NativeDiagnostics        # measurement-only (S4); off unless the run asks for it\n"
     "            self.diag = NativeDiagnostics()\n        self.reset_episode(None)"),
    ("    def reset_episode(self, game_seed: Optional[int], init_info: Any = None):\n        self.game_seed = game_seed",
     "    def reset_episode(self, game_seed: Optional[int], init_info: Any = None, raw0: Any = None):\n"
     "        if self.diag is not None:\n            self.diag.reset(raw0)\n        self.game_seed = game_seed"),
    ("    def add_step(self, action: int, legal: bool, reward: float, info: Dict[str, Any], terminated: bool, truncated: bool):\n        applied",
     "    def add_step(self, action: int, legal: bool, reward: float, info: Dict[str, Any], terminated: bool, truncated: bool,\n"
     "                 raw_after: Any = None, mask_before: Any = None, policy_entropy: Optional[float] = None):\n"
     "        if self.diag is not None:\n"
     "            self.diag.add_step(action, info.get(\"accepted\"), reward, info, terminated, truncated, raw_after, mask_before, policy_entropy)\n        applied"),
    ("            \"wall_seconds\": round(time.time() - self.t0, 3),\n        }\n",
     "            \"wall_seconds\": round(time.time() - self.t0, 3),\n            **(self.diag.row() if self.diag is not None else {}),\n        }\n"),
    ("        self._acc.reset_episode(info.get(\"seed\"), self.env.get_wrapper_attr(\"last_init_info\"))",
     "        self._acc.reset_episode(info.get(\"seed\"), self.env.get_wrapper_attr(\"last_init_info\"),\n"
     "                                self.env.get_wrapper_attr(\"last_raw_observation\") if self._acc.diag is not None else None)"),
    ("        legal = bool(self.env.get_wrapper_attr(\"action_masks\")()[int(action)])\n        obs, reward, terminated, truncated, info = self.env.step(action)\n"
     "        self._steps_total += 1\n        self._acc.add_step(int(action), legal, reward, info, terminated, truncated)",
     "        mask = self.env.get_wrapper_attr(\"action_masks\")()\n        legal = bool(mask[int(action)])\n"
     "        mask_copy = mask.copy() if self._acc.diag is not None else None\n"
     "        obs, reward, terminated, truncated, info = self.env.step(action)\n        self._steps_total += 1\n"
     "        self._acc.add_step(int(action), legal, reward, info, terminated, truncated,\n"
     "                           raw_after=(self.env.get_wrapper_attr(\"last_raw_observation\") if self._acc.diag is not None else None), mask_before=mask_copy)"),
])
patch("experiments/train_worker.py", [
    ("\"reward_mode\": env_cfg[\"reward\"], \"decision_interval_seconds\": env_cfg[\"decision_interval_seconds\"]})",
     "\"reward_mode\": env_cfg[\"reward\"], \"decision_interval_seconds\": env_cfg[\"decision_interval_seconds\"],\n"
     "        \"diagnostics\": bool((r.get(\"logging\") or {}).get(\"diagnostics\", False))})"),
])
patch("experiments/config.py", [
    ("\"logging\": {\"action_distribution\": (bool, False), \"training_curve\": (bool, False), \"episode_metrics\": (bool, False)},",
     "\"logging\": {\"action_distribution\": (bool, False), \"training_curve\": (bool, False), \"episode_metrics\": (bool, False),\n                \"diagnostics\": (bool, False)},"),
    ("        lg.setdefault(k, True)\n    exp = r[\"experiment\"]",
     "        lg.setdefault(k, True)\n    lg.setdefault(\"diagnostics\", False)          # S4 measurement-only native diagnostics (experiments/diagnostics.py)\n    exp = r[\"experiment\"]"),
    ("        lg.setdefault(k, True)\n    r.setdefault(\"experiment\", {}).setdefault(\"name\", \"evaluation\")",
     "        lg.setdefault(k, True)\n    lg.setdefault(\"diagnostics\", False)\n    r.setdefault(\"experiment\", {}).setdefault(\"name\", \"evaluation\")"),
])
patch("experiments/evaluation.py", [
    ("            \"decision_interval_seconds\": env_cfg[\"decision_interval_seconds\"]}\n    acc = EpisodeAccumulator(meta)",
     "            \"decision_interval_seconds\": env_cfg[\"decision_interval_seconds\"],\n"
     "            \"diagnostics\": bool((eval_cfg.get(\"logging\") or {}).get(\"diagnostics\", False))}\n    acc = EpisodeAccumulator(meta)"),
    ("                acc.reset_episode(info.get(\"seed\", seed))",
     "                acc.reset_episode(info.get(\"seed\", seed), None, env.last_raw_observation if acc.diag is not None else None)"),
    ("                    legal_sel = bool(mask[action])\n                    obs, reward, terminated, truncated, info = env.step(action)\n"
     "                    acc.add_step(action, legal_sel, reward, info, terminated, truncated)",
     "                    legal_sel = bool(mask[action])\n                    ent = None\n"
     "                    if acc.diag is not None and model is not None:\n                        ent = policy_entropy(model, obs, mask)\n"
     "                    mask_before = mask.copy() if acc.diag is not None else None\n"
     "                    obs, reward, terminated, truncated, info = env.step(action)\n"
     "                    acc.add_step(action, legal_sel, reward, info, terminated, truncated,\n"
     "                                 raw_after=(env.last_raw_observation if acc.diag is not None else None), mask_before=mask_before, policy_entropy=ent)"),
    ("def run_policy(",
     "def policy_entropy(model, obs, mask):\n"
     "    \"\"\"Entropy (nats) of the masked policy distribution at this state. Read-only forward pass (no gradient, no RNG use).\"\"\"\n"
     "    import torch\n    with torch.no_grad():\n        t, _ = model.policy.obs_to_tensor(obs)\n"
     "        d = model.policy.get_distribution(t, action_masks=mask[None, :])\n        return float(d.entropy().mean().item())\n\n\ndef run_policy("),
])
patch("experiments/postprocess.py", [
    ("EXTRA_EPISODE_COLUMNS = [\"worker_pid\", \"unix_time\"] + [f\"gap1_{k}\" for k in GAP1_FIELDS]",
     "DIAG_COLUMNS = [\"diag_deliveries\", \"diag_first_delivery_step\", \"diag_connections_max\", \"diag_connections_final\", \"diag_first_connection_step\",\n"
     "                \"diag_connection_to_first_delivery_steps\", \"diag_n_add_accepted\", \"diag_n_remove_accepted\", \"diag_churn\", \"diag_repeated_edits\",\n"
     "                \"diag_n_nonzero_nonterminal_reward\", \"diag_road_edges_max\", \"diag_legal_mean\", \"diag_empirical_action_entropy\", \"diag_policy_entropy_mean\"]\n"
     "EXTRA_EPISODE_COLUMNS = [\"worker_pid\", \"unix_time\"] + [f\"gap1_{k}\" for k in GAP1_FIELDS] + DIAG_COLUMNS"),
])
