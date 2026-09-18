"""Gate-controlled best-of-ten acquisition and exact preference replay."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from typing import Any

from .contracts import content_sha256, validate_broad_receipt

SELF_PLAY_ROLE_COUNTS = {
    "exploitation": 4,
    "uncertainty": 2,
    "diversity": 2,
    "transfer": 1,
    "random_calibration": 1,
}


def _finite(candidate: Mapping[str, Any], field: str) -> float:
    value = candidate.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"candidate {field} must be finite")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"candidate {field} must be finite")
    return number


def select_self_play_round(
    candidates: Sequence[Mapping[str, Any]],
    *,
    dataset_hash: str,
    seed: int,
) -> list[dict[str, Any]]:
    """Select 4/2/2/1/1 unique candidates with deterministic refill."""

    unique: dict[str, Mapping[str, Any]] = {}
    for candidate in candidates:
        semantic = str(candidate.get("semantic_sha256") or "")
        if len(semantic) != 64:
            raise ValueError("self-play candidate lacks semantic SHA-256")
        _finite(candidate, "ensemble_mean")
        _finite(candidate, "ensemble_uncertainty")
        unique.setdefault(semantic, candidate)
    if len(unique) < 10:
        raise ValueError("self-play round needs at least ten unique candidates")
    remaining = dict(unique)
    selected: list[dict[str, Any]] = []

    def take(role: str, ordered: Sequence[Mapping[str, Any]], count: int) -> None:
        for candidate in ordered:
            semantic = str(candidate["semantic_sha256"])
            if semantic not in remaining:
                continue
            selected.append({"role": role, **dict(candidate)})
            remaining.pop(semantic)
            if sum(row["role"] == role for row in selected) == count:
                return

    values = list(unique.values())
    take(
        "exploitation",
        sorted(
            values,
            key=lambda row: (
                -_finite(row, "ensemble_mean"),
                _finite(row, "ensemble_uncertainty"),
                str(row["semantic_sha256"]),
            ),
        ),
        4,
    )
    take(
        "uncertainty",
        sorted(
            values,
            key=lambda row: (
                -_finite(row, "ensemble_uncertainty"),
                -_finite(row, "ensemble_mean"),
                str(row["semantic_sha256"]),
            ),
        ),
        2,
    )
    diversity_seen: set[str] = set()
    diverse = []
    for row in sorted(
        values,
        key=lambda value: (
            str(value.get("structural_diversity_key") or ""),
            hashlib.sha256(
                f"{seed}\0{dataset_hash}\0{value['semantic_sha256']}".encode()
            ).digest(),
        ),
    ):
        key = str(row.get("structural_diversity_key") or "")
        if key and key not in diversity_seen:
            diversity_seen.add(key)
            diverse.append(row)
    diverse.extend(values)
    take("diversity", diverse, 2)
    take(
        "transfer",
        sorted(
            [row for row in values if row.get("source") == "transfer"],
            key=lambda row: (
                -_finite(row, "ensemble_mean"),
                str(row["semantic_sha256"]),
            ),
        ),
        1,
    )
    random_order = sorted(
        values,
        key=lambda row: hashlib.sha256(
            f"{seed}\0{dataset_hash}\0random\0{row['semantic_sha256']}".encode()
        ).digest(),
    )
    take("random_calibration", random_order, 1)

    # A missing source-specific bucket refills from unselected candidates while
    # preserving the slot's scientific role.
    for role, count in SELF_PLAY_ROLE_COUNTS.items():
        missing = count - sum(row["role"] == role for row in selected)
        if missing:
            take(
                role,
                sorted(
                    remaining.values(),
                    key=lambda row: (
                        -_finite(row, "ensemble_mean"),
                        str(row["semantic_sha256"]),
                    ),
                ),
                count,
            )
    if len(selected) != 10 or len(
        {row["semantic_sha256"] for row in selected}
    ) != 10:
        raise AssertionError("self-play best-of-ten allocation invariant failed")
    for slot, row in enumerate(selected):
        row["slot"] = slot
    return selected


def exact_preference_pairs(
    receipts: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Create replay preferences only from confirmed exact within-dataset labels."""

    by_dataset: dict[str, list[Mapping[str, Any]]] = {}
    for receipt in receipts:
        validate_broad_receipt(receipt)
        if receipt["fidelity"] != "exact" or not receipt["valid"]:
            continue
        by_dataset.setdefault(str(receipt["dataset_hash"]), []).append(receipt)
    pairs: list[dict[str, Any]] = []
    for dataset, rows in sorted(by_dataset.items()):
        ordered = sorted(
            rows,
            key=lambda row: (
                -float(row["robust_score"]),
                str(row["semantic_sha256"]),
            ),
        )
        for left_index, winner in enumerate(ordered):
            for loser in ordered[left_index + 1 :]:
                if winner["robust_score"] == loser["robust_score"]:
                    continue
                pair = {
                    "schema": "BroadExactPreference.v1",
                    "dataset_hash": dataset,
                    "winner_semantic_sha256": winner["semantic_sha256"],
                    "loser_semantic_sha256": loser["semantic_sha256"],
                    "winner_receipt_sha256": winner["receipt_sha256"],
                    "loser_receipt_sha256": loser["receipt_sha256"],
                    "winner_robust_score": winner["robust_score"],
                    "loser_robust_score": loser["robust_score"],
                    "fidelity": "exact",
                    "critic_prediction_used_as_label": False,
                }
                pair["preference_sha256"] = content_sha256(pair)
                pairs.append(pair)
    return pairs


def self_play_authorized(prospective_gate: Mapping[str, Any]) -> bool:
    return (
        prospective_gate.get("schema") == "BroadProspectiveGate.v1"
        and prospective_gate.get("passed") is True
        and prospective_gate.get("gate_sha256")
        == content_sha256(
            {
                key: value
                for key, value in prospective_gate.items()
                if key != "gate_sha256"
            }
        )
    )


def snapshot_due(
    *,
    labels_since_snapshot: int,
    seconds_since_snapshot: float,
) -> bool:
    """The later-of rule requires both thresholds to be satisfied."""

    return labels_since_snapshot >= 25_000 and seconds_since_snapshot >= 30 * 60
