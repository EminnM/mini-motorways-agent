import sys
def patch(path, pairs):
    s = open(path, encoding="utf-8").read()
    for old, new in pairs:
        if s.count(old) != 1:
            sys.exit(f"{path}: pattern matched {s.count(old)}x: {old[:70]!r}")
        s = s.replace(old, new)
    open(path, "w", encoding="utf-8").write(s)
    print("patched", path)

patch("python/mini_motorways_env.py", [
    ('            "isolationOk": resp["isolationOk"],\n        }',
     '            "isolationOk": resp["isolationOk"],\n            "applyInfo": resp.get("applyInfo"),     # additive, read-only (S6): lets diagnostics distinguish terminal-on-entry steps\n        }'),
])
patch("experiments/diagnostics.py", [
    ("import math\nfrom collections", "import math\nimport time\nfrom collections"),
    ("        self.edges0 = self.edges_max = self.edges_last = None\n        try:",
     "        self.edges0 = self.edges_max = self.edges_last = None\n        self.final_status = None\n        self.terminal_on_entry = None\n        self.terminal_penalty_missing = None\n"
     "        self.sim_prev = None if raw0 is None else raw0.get(\"simulationTime\")\n        self.t_prev = time.time()\n        try:"),
    ("        i = self.step_i\n        self.step_i += 1\n        try:",
     "        i = self.step_i\n        self.step_i += 1\n        now = time.time()\n        wall_interval, self.t_prev = now - self.t_prev, now\n        try:\n            if terminated or truncated:\n                self._final_status(action, accepted, reward, info, terminated, truncated, raw_after, wall_interval)\n        except Exception:\n            self.errors += 1\n        try:"),
    ("            if raw_after is not None:\n                self.edges_last",
     "            if raw_after is not None:\n                self.sim_prev = raw_after.get(\"simulationTime\", self.sim_prev)\n                self.edges_last"),
    ("    def row(self) -> Dict[str, Any]:",
     "    def _final_status(self, action, accepted, reward, info, terminated, truncated, raw_after, wall_interval):\n"
     "        \"\"\"Measurement only (S6): classify how the episode ended. Never feeds back into the environment or the reward.\n"
     "        terminal_on_entry   = the final STEP started with the game already terminal (applyInfo 'terminal state (not applied)', accepted False)\n"
     "        terminal_penalty_missing = native game-over (terminated, not truncated) whose terminalPenalty component is 0\"\"\"\n"
     "        rc = info.get(\"rewardComponents\", {})\n        ai = info.get(\"applyInfo\")\n"
     "        on_entry = (ai == \"terminal state (not applied)\") if ai is not None else (accepted is False)\n"
     "        pen = float(rc.get(\"terminalPenalty\", 0.0))\n"
     "        sim_after = raw_after.get(\"simulationTime\") if raw_after is not None else None\n"
     "        self.terminal_on_entry = bool(terminated and on_entry)\n"
     "        self.terminal_penalty_missing = bool(terminated and not truncated and pen == 0.0)\n"
     "        self.final_status = {\"reward\": float(reward), \"terminated\": bool(terminated), \"truncated\": bool(truncated), \"accepted\": accepted,\n"
     "                             \"appliedAction\": info.get(\"appliedAction\"), \"applyInfo\": ai, \"rewardComponents\": rc, \"decisionIndex\": info.get(\"decisionIndex\"),\n"
     "                             \"sim_time_before\": self.sim_prev, \"sim_time_after\": sim_after,\n"
     "                             \"sim_time_advance\": (None if sim_after is None or self.sim_prev is None else round(sim_after - self.sim_prev, 6)),\n"
     "                             \"wall_interval_s\": round(wall_interval, 4)}\n\n"
     "    def row(self) -> Dict[str, Any]:"),
    ("            \"diag_errors\": self.errors,\n        }",
     "            \"diag_terminal_penalty_missing\": self.terminal_penalty_missing, \"diag_terminal_on_entry\": self.terminal_on_entry,\n"
     "            \"diag_final_status\": self.final_status,\n            \"diag_errors\": self.errors,\n        }"),
])
patch("experiments/postprocess.py", [
    ('"diag_policy_entropy_mean"]', '"diag_policy_entropy_mean",\n                "diag_terminal_penalty_missing", "diag_terminal_on_entry"]'),
])
