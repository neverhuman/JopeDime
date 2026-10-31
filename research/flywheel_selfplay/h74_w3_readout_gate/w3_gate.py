#!/usr/bin/env python3
"""Evaluate the preregistered W3 paired-family gates from an admitted summary."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import re
import sys
from typing import Any


SCHEMA = "jope.h74.w3-family-evidence/v1"
EXPECTED_DOPE_SHA256 = (
    "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814"
)
EXPECTED_CONTEXTS = 1219
EXPECTED_FAMILIES = 205
EXPECTED_FOLD_SEED = 4900
EXPERIMENT_SEEDS = (701, 702, 703)
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20261031
PRIMARY_CONTRASTS = ("top_resolved_vs_h58_d", "decomposed_vs_smoothl1_kl")
LCB_THRESHOLD = 0.001
HARM_THRESHOLD = -0.01
MAX_HARM_FRACTION = 0.10
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class Refusal(ValueError):
    """Input is malformed or does not match the registered W3 gate."""


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise Refusal(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise Refusal(f"non-finite JSON constant: {value}")


def _exact(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise Refusal(f"{label} must contain exactly: {', '.join(sorted(keys))}")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Refusal(f"{label} must be numeric")
    try:
        number = float(value)
    except OverflowError as error:
        raise Refusal(f"{label} must be finite") from error
    if not math.isfinite(number):
        raise Refusal(f"{label} must be finite")
    return number


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise Refusal(f"{label} must be a lowercase SHA256 digest")
    return value


def validate(evidence: Any) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    root = _exact(
        evidence,
        {"schema", "closure", "bootstrap", "kill_checks", "contrasts"},
        "evidence",
    )
    if root["schema"] != SCHEMA:
        raise Refusal("unsupported W3 evidence schema")

    closure = _exact(
        root["closure"],
        {
            "training_role",
            "context_count",
            "family_count",
            "fold_seed",
            "fold_count",
            "seeds",
            "dl_val_rows",
            "outer_rows",
            "sealed_rows",
            "dope_sha256",
            "admitted_export_sha256",
            "fold_manifest_sha256",
            "winner_config_sha256",
            "h59_disposition_sha256",
        },
        "closure",
    )
    expected_closure = {
        "training_role": "DL-Train",
        "context_count": EXPECTED_CONTEXTS,
        "family_count": EXPECTED_FAMILIES,
        "fold_seed": EXPECTED_FOLD_SEED,
        "fold_count": 5,
        "seeds": list(EXPERIMENT_SEEDS),
        "dl_val_rows": 0,
        "outer_rows": 0,
        "sealed_rows": 0,
        "dope_sha256": EXPECTED_DOPE_SHA256,
    }
    for key, expected in expected_closure.items():
        if closure[key] != expected or type(closure[key]) is not type(expected):
            raise Refusal(f"closure.{key} does not match the W3 training contract")
    for key in (
        "admitted_export_sha256",
        "fold_manifest_sha256",
        "winner_config_sha256",
        "h59_disposition_sha256",
    ):
        _digest(closure[key], f"closure.{key}")

    bootstrap = _exact(
        root["bootstrap"],
        {
            "unit",
            "draws",
            "seed",
            "seed_average_within_family",
            "paired_contrasts",
            "adjustment",
        },
        "bootstrap",
    )
    expected_bootstrap = {
        "unit": "family",
        "draws": BOOTSTRAP_DRAWS,
        "seed": BOOTSTRAP_SEED,
        "seed_average_within_family": True,
        "paired_contrasts": True,
        "adjustment": "bonferroni_two_primary_contrasts",
    }
    for key, expected in expected_bootstrap.items():
        if bootstrap[key] != expected or type(bootstrap[key]) is not type(expected):
            raise Refusal(f"bootstrap.{key} does not match the registered contract")

    checks = _exact(
        root["kill_checks"],
        {
            "auxiliary_loss_only_gain",
            "shuffled_auxiliary_matches_gain",
            "near_pass_semantics_changed",
            "zero_update_choices_preserved",
        },
        "kill_checks",
    )
    if any(type(value) is not bool for value in checks.values()):
        raise Refusal("kill_checks values must be booleans")

    raw_contrasts = _exact(root["contrasts"], set(PRIMARY_CONTRASTS), "contrasts")
    parsed: dict[str, list[dict[str, Any]]] = {}
    canonical_families: list[str] | None = None
    canonical_context_counts: dict[str, int] | None = None
    for name in PRIMARY_CONTRASTS:
        rows = raw_contrasts[name]
        if not isinstance(rows, list) or len(rows) != EXPECTED_FAMILIES:
            raise Refusal(
                f"{name} must include exactly {EXPECTED_FAMILIES} family rows"
            )
        out: list[dict[str, Any]] = []
        ids: set[str] = set()
        context_counts: dict[str, int] = {}
        for index, raw in enumerate(rows):
            row = _exact(
                raw,
                {
                    "family_id",
                    "context_count",
                    "seed_deltas",
                    "gate_failures_arm",
                    "gate_failures_control",
                },
                f"{name}[{index}]",
            )
            family_id = row["family_id"]
            if (
                not isinstance(family_id, str)
                or not family_id.strip()
                or family_id in ids
            ):
                raise Refusal(f"{name} has an empty or duplicate family_id")
            ids.add(family_id)
            context_count = row["context_count"]
            if type(context_count) is not int or context_count <= 0:
                raise Refusal(
                    f"{name}[{family_id}].context_count must be a positive integer"
                )
            context_counts[family_id] = context_count

            deltas = row["seed_deltas"]
            if not isinstance(deltas, list) or len(deltas) != len(EXPERIMENT_SEEDS):
                raise Refusal(
                    f"{name}[{family_id}].seed_deltas must cover seeds 701, 702, and 703"
                )
            numeric_deltas = [
                _finite(value, f"{name}[{family_id}].seed_deltas") for value in deltas
            ]
            failures: dict[str, list[int]] = {}
            for key in ("gate_failures_arm", "gate_failures_control"):
                values = row[key]
                if not isinstance(values, list) or len(values) != len(EXPERIMENT_SEEDS):
                    raise Refusal(
                        f"{name}[{family_id}].{key} must cover all three seeds"
                    )
                if any(
                    type(value) is not int or value < 0 or value > context_count
                    for value in values
                ):
                    raise Refusal(
                        f"{name}[{family_id}].{key} contains an invalid count"
                    )
                failures[key] = values
            out.append(
                {
                    "family_id": family_id,
                    "context_count": context_count,
                    "deltas": numeric_deltas,
                    **failures,
                }
            )

        if sum(context_counts.values()) != EXPECTED_CONTEXTS:
            raise Refusal(
                f"{name} family context counts must sum to {EXPECTED_CONTEXTS}"
            )
        if canonical_families is None:
            canonical_families = sorted(ids)
            canonical_context_counts = context_counts
        elif (
            sorted(ids) != canonical_families
            or context_counts != canonical_context_counts
        ):
            raise Refusal(
                "the two primary contrasts must use the same family and context roster"
            )
        out.sort(key=lambda item: item["family_id"])
        parsed[name] = out
    return checks, parsed


def _paired_family_bootstrap(
    contrasts: dict[str, list[dict[str, Any]]],
) -> dict[str, float]:
    families = len(next(iter(contrasts.values())))
    names = list(PRIMARY_CONTRASTS)
    family_deltas = {
        name: [
            math.fsum(row["deltas"]) / len(EXPERIMENT_SEEDS) for row in contrasts[name]
        ]
        for name in names
    }
    rng = random.Random(BOOTSTRAP_SEED)
    draws = {name: [] for name in names}
    for _ in range(BOOTSTRAP_DRAWS):
        sampled = [rng.randrange(families) for _ in range(families)]
        for name in names:
            draws[name].append(
                math.fsum(family_deltas[name][index] for index in sampled) / families
            )

    # Two one-sided primary bounds: alpha=0.05/2, nearest-rank percentile.
    lower_index = math.ceil((0.05 / len(names)) * BOOTSTRAP_DRAWS) - 1
    return {name: sorted(draws[name])[lower_index] for name in names}


def evaluate(evidence: Any) -> dict[str, Any]:
    kill_checks, contrasts = validate(evidence)
    lower_bounds = _paired_family_bootstrap(contrasts)
    results: dict[str, Any] = {}
    for name, rows in contrasts.items():
        family_deltas = [
            math.fsum(row["deltas"]) / len(EXPERIMENT_SEEDS) for row in rows
        ]
        ordered = sorted(family_deltas)
        median = ordered[EXPECTED_FAMILIES // 2]
        harm_count = sum(value < HARM_THRESHOLD for value in family_deltas)
        arm_failures = sum(sum(row["gate_failures_arm"]) for row in rows)
        control_failures = sum(sum(row["gate_failures_control"]) for row in rows)
        pass_gate = (
            lower_bounds[name] > LCB_THRESHOLD
            and median >= 0.0
            and harm_count / EXPECTED_FAMILIES <= MAX_HARM_FRACTION
            and arm_failures <= control_failures
        )
        results[name] = {
            "adjusted_family_mean_lcb": lower_bounds[name],
            "median_family_delta": median,
            "families_loss_gt_0_01": harm_count,
            "family_count": EXPECTED_FAMILIES,
            "gate_failures_arm": arm_failures,
            "gate_failures_control": control_failures,
            "primary_gate_pass": pass_gate,
        }

    kill_pass = (
        not kill_checks["auxiliary_loss_only_gain"]
        and not kill_checks["shuffled_auxiliary_matches_gain"]
        and not kill_checks["near_pass_semantics_changed"]
        and kill_checks["zero_update_choices_preserved"]
    )
    advance = kill_pass and all(
        result["primary_gate_pass"] for result in results.values()
    )
    return {
        "schema": "jope.h74.w3-gate-decision/v1",
        "decision": "ADVANCE_BOTH_PRIMARY_CONTRASTS" if advance else "NO_ADVANCE",
        "bootstrap": {
            "unit": "family",
            "draws": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED,
            "adjustment": "bonferroni_two_primary_contrasts",
            "method": "paired_family_percentile_nearest_rank",
        },
        "launch_authorized": False,
        "source_provenance_verified": False,
        "interpretation": "gate calculation only; validate the source evidence independently",
        "kill_checks_pass": kill_pass,
        "contrasts": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        evidence = json.loads(
            args.evidence.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
        print(json.dumps(evaluate(evidence), sort_keys=True))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, Refusal) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
