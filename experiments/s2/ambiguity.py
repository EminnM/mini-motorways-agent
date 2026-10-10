"""Carpark-entry ambiguity of the live native PolicyObservation over the frozen native Gap-1 tasks."""
import glob
import json
import os
from collections import Counter

ROOT = r"C:\Users\TKA\Desktop\mini moto"
OUT = os.path.join(ROOT, "research", "synthetic_to_native")


def load(split):
    rows = []
    for p in sorted(glob.glob(os.path.join(OUT, f"capture_{split}_*.jsonl"))):
        for l in open(p):
            if l.strip():
                c = json.loads(l)
                c.pop("action_tests", None)
                rows.append(c)
    return rows


def main():
    rep = {"definition": {
        "affected": "more than one road tile touches the destination footprint, so the (encoded) observation does not mark which one is the carpark entry",
        "resolvable_by_leaf_rule": "affected, but exactly one touching road tile is a dead end (degree 1) and it is the true entry (a rule a policy could learn from the observation)",
        "unresolvable_by_leaf_rule": "affected and the leaf rule does not single out the true entry",
        "where_the_information_is": "the raw native PolicyObservation JSON carries contentType 'Carpark' on the carpark tiles; python/mini_motorways_env.py::_CONTENT_TYPE_ORDINAL has only None/House/Destination, so Carpark (like Tree) is encoded as 0 by .get(..., 0)"}}
    for split in ("validation", "test"):
        rows = load(split)
        cat = Counter(r["ambiguity"]["category"] for r in rows)
        n = len(rows)
        rep[split] = {"n": n, "categories": dict(cat), "affected_fraction": 1 - cat.get("unique", 0) / n,
                      "unresolvable_fraction": cat.get("unresolvable_by_leaf_rule", 0) / n,
                      "raw_json_has_carpark_tile_in_all": all(r["ambiguity"]["carpark_visible_in_raw_native_json"] for r in rows),
                      "encoded_obs_marks_carpark_in_any": any(r["ambiguity"]["carpark_visible_in_encoded_observation"] for r in rows),
                      "mean_adjacent_road_tiles": sum(r["ambiguity"]["n_adjacent_road_tiles"] for r in rows) / n,
                      "n_carpark_tiles_in_raw_mean": sum(r["ambiguity"]["n_carpark_tiles_in_raw"] for r in rows) / n,
                      "true_entry_tiles_counts": dict(Counter(len(r["ambiguity"]["entry_tiles_true"]) for r in rows)),
                      "by_orientation": {o: dict(Counter(r["ambiguity"]["category"] for r in rows if r["orientation"] == o)) for o in ("horizontal", "vertical")},
                      "a_star_legal_native_all": all(r["a_star_legal_native"] for r in rows), "house_connected_at_reset_any": any(r["house_connected_at_reset"] for r in rows)}
    rep["verdict"] = ("The ambiguity exists in the live native observations as encoded for the policy. It is an ENCODING omission, not missing game information: "
                      "the native observation JSON identifies carpark tiles, the python encoder discards that. Whether the encoded observation still suffices depends on the leaf rule (see categories).")
    json.dump(rep, open(os.path.join(OUT, "ambiguity_report.json"), "w"), indent=2)
    print(json.dumps({k: rep[k] for k in ("validation", "test")}, indent=1))


if __name__ == "__main__":
    main()
