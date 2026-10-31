#!/usr/bin/env python3
"""Offline W7 design-plan skeleton; it never reads data or executes a study."""

import argparse
import json


PLAN = {
    "schema": "jope.rd-breakthrough.h69-w7-plan/v1",
    "wave": "W7",
    "claim": "H69",
    "status": "CPU_DESIGN_PREP_ONLY",
    "proposal_commit": "1d249852f4080b512910a7303cd97c20d384f4f1",
    "proposal_sha256": "f0526f20699dea3fe5259d10eee30483ac3c52655249906fd1ce79dd8e193e50",
    "arms": [
        "r882/current-two-greedy-two-random",
        "matched-all-completed-uniform-replay",
        "two-exploitation-one-disagreement-diversity-one-uniform",
    ],
    "seeds": [701, 702, 703],
    "family_folds": 5,
    "acquisition_rounds": 20,
    "terminal_updates": 40,
    "training_context_rounds_per_arm_seed": 100,
    "fits_per_context_round": 4,
    "planned_fits": 3600,
    "fit_ceiling": 4800,
    "gpu_hours_ceiling": 2,
    "cpu_hours_ceiling": 4,
    "common_advance_gate": {
        "adjusted_family_mean_lcb_strictly_above": 0.001,
        "median_family_delta_at_least": 0.0,
        "maximum_fraction_families_loss_over_0_01": 0.10,
    },
    "execution_admitted": False,
    "data_access": "none",
    "model_fits_executed": 0,
    "numerical_results_claimed": False,
}


def build_plan():
    """Validate the proposal's arithmetic and return a fresh plan object."""
    expected = (
        len(PLAN["arms"])
        * len(PLAN["seeds"])
        * PLAN["fits_per_context_round"]
        * PLAN["training_context_rounds_per_arm_seed"]
    )
    if expected != PLAN["planned_fits"]:
        raise ValueError(f"W7 fit arithmetic mismatch: expected {expected}")
    if expected > PLAN["fit_ceiling"]:
        raise ValueError("planned fits exceed the W7 ceiling")
    return dict(PLAN)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true", required=True,
        help="print the offline plan only; execution is not implemented",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON")
    args = parser.parse_args()
    plan = build_plan()
    if args.json:
        print(json.dumps(plan, indent=2, sort_keys=True))
    else:
        print("W7 H69 design plan validated; no data or study execution.")
        print(f"Planned fits: {plan['planned_fits']} / {plan['fit_ceiling']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
