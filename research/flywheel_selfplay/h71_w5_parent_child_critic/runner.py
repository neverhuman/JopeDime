#!/usr/bin/env python3
"""Offline W5 design-plan validator; no data access or study execution."""

import argparse
import json
import sys


PLAN = {
    "schema": "jope.rd-breakthrough.h71-w5-plan/v1",
    "wave": "W5",
    "claim": "H71",
    "status": "GATED_CPU_PREP_ONLY",
    "proposal_commit": "35f49f70ce7011adcf800876972c677efa3b9a36",
    "proposal_sha256": "f0526f20699dea3fe5259d10eee30483ac3c52655249906fd1ce79dd8e193e50",
    "arms": [
        "r882-absolute-cost-ranking",
        "program-feature-gbm-transition-baseline",
        "neural-parent-child-head",
        "parent-operator-shuffled-placebo",
    ],
    "primary_contrast": "neural-parent-child-head-vs-gbm-transition-same-parent-proposals",
    "screening_contrast_requires_fixed_selector": True,
    "seeds": [701, 702, 703],
    "family_folds": 5,
    "fixed_updates_per_pair": 40,
    "prospective_full_fit_opportunities_per_context": 4,
    "planned_pair_count": None,
    "planned_fit_count": None,
    "fit_ceiling": 4800,
    "gpu_hours_ceiling": 2,
    "cpu_hours_ceiling": 4,
    "common_advance_gate": {
        "adjusted_family_mean_lcb_strictly_above": 0.001,
        "median_family_delta_at_least": 0.0,
        "maximum_fraction_families_loss_over_0_01": 0.10,
    },
    "required_pair_identity": [
        "same_context",
        "same_evaluator",
        "verified_parent_id",
        "verified_operator_id",
        "original_unrounded_outcome",
    ],
    "provenance_status": "not_checked",
    "execution_admitted": False,
    "data_access": "none",
    "model_fits_executed": 0,
    "numerical_results_claimed": False,
}


def build_plan():
    """Validate the frozen design fields without inferring unavailable counts."""
    if len(PLAN["seeds"]) != 3 or PLAN["family_folds"] != 5:
        raise ValueError("W5 seed or family-fold contract changed")
    if PLAN["planned_pair_count"] is not None or PLAN["planned_fit_count"] is not None:
        raise ValueError("pair and fit counts require an admitted roster")
    if PLAN["execution_admitted"] or PLAN["provenance_status"] != "not_checked":
        raise ValueError("CPU preparation must remain fail-closed")
    if PLAN["model_fits_executed"] != 0 or PLAN["data_access"] != "none":
        raise ValueError("design-plan validation cannot access data or fit models")
    return dict(PLAN)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="validate and print the plan only")
    parser.add_argument("--json", action="store_true", help="emit the plan as JSON")
    args = parser.parse_args(argv)
    if not args.dry_run:
        parser.error("only --dry-run is implemented; W5 execution is provenance-gated")
    plan = build_plan()
    if args.json:
        print(json.dumps(plan, indent=2, sort_keys=True))
    else:
        print("W5 H71 design plan validated; provenance is unchecked and execution is not admitted.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
