"""PR-14 canaries A1–A5 on a fit-only stand-in.

Default mode is fixtures. `--live` without `--owner-live` exits 2. Both flags
exit 3 and do not load weights: this package has no live encoder. A weights
path is never opened.
"""

import argparse
import copy
import sys

from rlfw.arena.game_spec import canonical, digest

from .crosswalk import classify_family, unseen_word
from .encoding import NOT_DECONTAMINATION, VECTOR_DIM, fit_chain
from .seal import fit_judge_split

CANARY_SEED = 20261009


def fixture_table(seed=CANARY_SEED):
    """Ten rows. The 80/20 split is 8 fit and 2 judge; fit then splits 6/2."""
    row_ids = [f"r{index:02d}" for index in range(10)]
    table = {}
    for index, row_id in enumerate(row_ids):
        table[row_id] = {
            "features": [float(index), 0.0],
            "label": f"lab-{index}",
            "target": float(index),
        }
    fit_ids, judge_ids = fit_judge_split(row_ids, seed)
    for judge_id in judge_ids:
        table[judge_id]["features"][1] = 1.0
    return table, fit_ids, judge_ids


def _chain(table, fit_ids, judge_ids, *, mask=False):
    return fit_chain(
        table,
        fit_ids,
        judge_row_ids=judge_ids,
        proposals=["prog-a", "prog-b"],
        mask_diagnostics=mask,
    )


def _hidden_and_spread(vector):
    return vector[:2048], vector[2049], vector[2050]


def run_canaries(seed=CANARY_SEED):
    table, fit_ids, judge_ids = fixture_table(seed)
    base = _chain(table, fit_ids, judge_ids)
    masked = _chain(table, fit_ids, judge_ids, mask=True)
    query_ids = base["query_row_ids"]
    changed = copy.deepcopy(table)
    changed[query_ids[0]]["target"] = -100.0
    changed[query_ids[1]]["target"] = 100.0
    query_edit = _chain(changed, fit_ids, judge_ids)
    poisoned_targets = copy.deepcopy(table)
    for judge_id in judge_ids:
        poisoned_targets[judge_id]["target"] = 10_000.0
    poisoned = _chain(poisoned_targets, fit_ids, judge_ids)
    poisoned_labels = copy.deepcopy(table)
    for judge_id in judge_ids:
        poisoned_labels[judge_id]["label"] = "poison"
    labelled = _chain(poisoned_labels, fit_ids, judge_ids)
    disclosure = {"sha256": digest("kumo-pretraining-fixture"), "source_ids": ["kumo-src"]}
    overlap = classify_family(
        {"pretraining_disclosure": disclosure, "source_ids": ["kumo-src", "family-src"]}
    )
    missing = classify_family({"source_ids": ["family-src"]})
    clean = classify_family(
        {"pretraining_disclosure": disclosure, "source_ids": ["family-src"]}
    )
    refused = False
    try:
        unseen_word(overlap)
    except ValueError:
        refused = True
    hidden_same = _hidden_and_spread(base["vector"]) == _hidden_and_spread(query_edit["vector"])
    results = {
        "A1": {
            "diagnostic_label": masked["diagnostic_label"],
            "not_decontamination": masked["not_decontamination"],
            "passed": (
                masked["diagnostic_label"] == NOT_DECONTAMINATION
                and masked["vector"][2048:2051] == [0.0, 0.0, 0.0]
                and masked["diagnostic_l2"] > 0.0
                and masked["selected_indices"] == base["selected_indices"]
                and masked["selection_implementation"] == base["selection_implementation"]
            ),
        },
        "A2": {
            "passed": hidden_same and base["vector"][2048] != query_edit["vector"][2048],
            "query_row_ids": query_ids,
        },
        "A3": {
            "passed": (
                poisoned["selected_indices"] == base["selected_indices"]
                and poisoned["vector"] == base["vector"]
            ),
        },
        "A4": {
            "passed": (
                labelled["vector"] == base["vector"]
                and labelled["proposals"] == base["proposals"]
                and labelled["weights"] == base["weights"]
                and labelled["cache_key"] == base["cache_key"]
                and labelled["optimization_score"] == base["optimization_score"]
            ),
        },
        "A5": {
            "clean_status": clean,
            "missing_disclosure": missing,
            "overlap": overlap,
            "passed": overlap == "quarantine" and missing == "overlap_unknown" and refused and clean == "no_overlap",
            "unseen_refused_for_overlap": refused,
        },
    }
    return {
        "canaries": results,
        "fit_row_ids": fit_ids,
        "judge_row_ids": judge_ids,
        "passed": all(item["passed"] for item in results.values()),
        "selection_implementation": base["selection_implementation"],
        "vector_dim": VECTOR_DIM,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Lane B canaries. The default is fixtures.")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--owner-live", action="store_true")
    parser.add_argument("--weights")
    args = parser.parse_args(argv)
    if args.live and not args.owner_live:
        print("refusing live canaries without --owner-live", file=sys.stderr)
        return 2
    if args.live and args.owner_live:
        print(
            "owner flag accepted; no live encoder is bundled; Kumo weights were not loaded",
            file=sys.stderr,
        )
        return 3
    report = run_canaries()
    sys.stdout.write(canonical(report) + "\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
