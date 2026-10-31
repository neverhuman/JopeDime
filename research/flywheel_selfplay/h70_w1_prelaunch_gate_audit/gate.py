#!/usr/bin/env python3
"""Calculate the preregistered W1 advancement gates from a complete summary."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any


SCHEMA = "jope.w1-gate-evidence/v1"
CONTRASTS = (
    "u4_vs_program_gbm",
    "u4_vs_program_context_gbm",
    "legacy_vs_program_gbm",
    "legacy_vs_program_context_gbm",
)
COMMON_LCB = 0.001
LEGACY_NO_HARM_LCB = -0.005
MAX_HARM_FRACTION = 0.10
EXPECTED_FAMILIES = 205


class Refusal(ValueError):
    """Evidence is malformed or does not identify the registered W1 gate."""


def _reject_constant(value: str) -> None:
    raise Refusal(f"non-finite JSON constant: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise Refusal(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Refusal(f"{label} must be numeric")
    try:
        number = float(value)
    except OverflowError as error:
        raise Refusal(f"{label} must be finite") from error
    if not math.isfinite(number):
        raise Refusal(f"{label} must be finite")
    return number


def _exact_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise Refusal(f"{label} must contain exactly: {', '.join(sorted(expected))}")
    return value


def validate_evidence(evidence: Any) -> dict[str, dict[str, float]]:
    root = _exact_keys(
        evidence,
        {"schema", "bootstrap", "contrasts"},
        "evidence",
    )
    if root["schema"] != SCHEMA:
        raise Refusal("unsupported evidence schema")

    bootstrap = _exact_keys(
        root["bootstrap"],
        {
            "unit",
            "draws",
            "seed",
            "paired_seeds_together",
            "all_registered_contrasts_adjusted",
        },
        "bootstrap",
    )
    if (
        bootstrap["unit"] != "family"
        or type(bootstrap["draws"]) is not int
        or bootstrap["draws"] != 10000
        or type(bootstrap["seed"]) is not int
        or bootstrap["seed"] != 20261031
        or type(bootstrap["paired_seeds_together"]) is not bool
        or bootstrap["paired_seeds_together"] is not True
        or type(bootstrap["all_registered_contrasts_adjusted"]) is not bool
        or bootstrap["all_registered_contrasts_adjusted"] is not True
    ):
        raise Refusal(
            "bootstrap identity or joint adjustment differs from the W1 contract"
        )

    contrasts = _exact_keys(root["contrasts"], set(CONTRASTS), "contrasts")
    validated: dict[str, dict[str, float]] = {}
    for name, raw in contrasts.items():
        row = _exact_keys(
            raw,
            {
                "adjusted_family_mean_lcb",
                "median_family_delta",
                "families_loss_gt_0_01",
                "family_count",
            },
            name,
        )
        lcb = _number(
            row["adjusted_family_mean_lcb"], f"{name}.adjusted_family_mean_lcb"
        )
        median = _number(row["median_family_delta"], f"{name}.median_family_delta")
        count = row["families_loss_gt_0_01"]
        total = row["family_count"]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise Refusal(f"{name}.families_loss_gt_0_01 must be a nonnegative integer")
        if isinstance(total, bool) or not isinstance(total, int) or total <= 0:
            raise Refusal(f"{name}.family_count must be a positive integer")
        if count > total:
            raise Refusal(f"{name} harm count exceeds family count")
        if total != EXPECTED_FAMILIES:
            raise Refusal(
                f"{name} must cover all {EXPECTED_FAMILIES} training families"
            )
        validated[name] = {
            "lcb": lcb,
            "median": median,
            "harm_fraction": count / total,
        }
    return validated


def common_gate(row: dict[str, float]) -> bool:
    return (
        row["lcb"] > COMMON_LCB
        and row["median"] >= 0.0
        and row["harm_fraction"] <= MAX_HARM_FRACTION
    )


def verdict(evidence: Any) -> dict[str, Any]:
    rows = validate_evidence(evidence)
    u4_names = CONTRASTS[:2]
    legacy_names = CONTRASTS[2:]
    search_pass = all(common_gate(rows[name]) for name in u4_names)
    legacy_no_harm_lcb = min(rows[name]["lcb"] for name in legacy_names)
    legacy_pass = all(common_gate(rows[name]) for name in legacy_names)

    if search_pass and legacy_no_harm_lcb > LEGACY_NO_HARM_LCB:
        decision = "ADVANCE_SEARCH_ONLY"
    elif legacy_pass:
        decision = "ADVANCE_LEGACY"
    else:
        decision = "NO_ADVANCE"
    return {
        "schema": "jope.w1-gate-verdict/v1",
        "decision": decision,
        "scientific_result": False,
        "u4_common_gate_pass": search_pass,
        "legacy_common_gate_pass": legacy_pass,
        "legacy_no_harm_lcb": legacy_no_harm_lcb,
        "contrast_count": len(rows),
        "gpu_used": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        raw = json.loads(
            args.evidence.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
        print(json.dumps(verdict(raw), sort_keys=True))
    except (OSError, json.JSONDecodeError, Refusal) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
