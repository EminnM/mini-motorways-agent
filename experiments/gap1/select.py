"""Checkpoint selection rule, fixed BEFORE any training run (SPEC section 19):
highest VALIDATION gap completion rate; ties -> lower median steps-to-success; then earlier checkpoint.
TEST is never consulted."""
import json
import os


def select(rows):
    def key(r):
        med = r.get("median_steps_to_success")
        return (-r["gap_completion_rate"], float("inf") if med is None else med, r["training_timestep"])
    return sorted(rows, key=key)[0]


def write_selection(sd: str, rows):
    best = select(rows)
    sel = {"rule": "max validation gap_completion_rate; tie -> lower median_steps_to_success; tie -> earlier training_timestep",
           "selected_file": best["file"], "training_timestep": best["training_timestep"], "sha256": best["sha256"],
           "validation_gap_completion_rate": best["gap_completion_rate"],
           "validation_median_steps_to_success": best.get("median_steps_to_success"),
           "candidates": [{k: r[k] for k in ("file", "training_timestep", "gap_completion_rate", "median_steps_to_success")} for r in rows]}
    with open(os.path.join(sd, "evaluation", "selection.json"), "x", encoding="utf-8") as f:
        json.dump(sel, f, indent=2)
    return sel
