# Held-out evaluation seeds (Phase 11A): 10 widely spaced seeds, each a DISTINCT starting house.
# Chosen from research/phase11a_seed_probe.json. The old 9101-9110 set all produced ONE identical city.
EVAL_SEEDS = [7932, 31689, 71284, 102960, 134636, 150474, 166312, 190069, 213826, 221745]

# VALIDATION seeds: used ONLY to pick the best checkpoint (best_tracker.py / the stream).
# Disjoint from EVAL_SEEDS (the TEST set used for the final report) -- no selection bias.
VAL_SEEDS = [182150, 205907, 229664, 253421, 261340, 269259]
