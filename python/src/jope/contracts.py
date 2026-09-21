"""Versioned, deterministic contracts for the shadow Prime/Lime generation.

These contracts are additive. They do not authorize a v1 checkpoint, a legacy
split, or historical full-fit evidence to become live league authority.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum


def _canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(name: str, value: str) -> None:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")


def _f32(value: float) -> float:
    """Round once to the f32 representation required by the Rust contract."""
    return struct.unpack("!f", struct.pack("!f", value))[0]


class TaskType(str, Enum):
    REGRESSION = "regression"
    BINARY = "binary"
    # Four-task program: wire values are append-only (jope-dime TASK_CODE
    # {regression: 0, binary: 1, timeseries: 2, multiclass: 3}).
    TIMESERIES = "timeseries"
    MULTICLASS = "multiclass"


class OuterSplit(str, Enum):
    TRAIN = "train"
    HOLDOUT = "holdout"


def freeze_source_family_split(
    dataset_families: Mapping[str, str],
    *,
    train_count: int,
    holdout_count: int,
    seed: int = 0,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Choose exact family-disjoint counts or fail instead of leaking a family."""
    if train_count < 0 or holdout_count < 0:
        raise ValueError("split counts must be nonnegative")
    if train_count + holdout_count != len(dataset_families):
        raise ValueError("split counts must cover every dataset exactly once")
    families: dict[str, list[str]] = {}
    for dataset, family in dataset_families.items():
        if not dataset or not family:
            raise ValueError("dataset and source-family identities must be non-empty")
        families.setdefault(family, []).append(dataset)
    ordered = sorted(
        families,
        key=lambda family: hashlib.sha256(f"{seed}|{family}".encode()).digest(),
    )
    # Subset sum over family sizes. Keeping the first deterministic witness makes
    # the frozen result stable even when input mappings have different iteration order.
    reachable: dict[int, tuple[str, ...]] = {0: ()}
    for family in ordered:
        size = len(families[family])
        for total, chosen in sorted(reachable.items(), reverse=True):
            candidate = total + size
            if candidate <= holdout_count and candidate not in reachable:
                reachable[candidate] = (*chosen, family)
    if holdout_count not in reachable:
        raise ValueError("source families cannot realize the requested exact holdout count")
    held_families = set(reachable[holdout_count])
    holdout = tuple(sorted(dataset for family in held_families for dataset in families[family]))
    train = tuple(sorted(set(dataset_families).difference(holdout)))
    if len(train) != train_count:
        raise AssertionError("family split count invariant failed")
    return train, holdout


def _row_hash(base_dataset: str, row_id: str) -> str:
    return hashlib.sha256(f"{base_dataset}|{row_id}".encode()).hexdigest()


def _row_set_hash(row_hashes: Sequence[str]) -> str:
    return _canonical_hash(sorted(row_hashes))


@dataclass(frozen=True)
class SplitViewV1:
    view_index: int
    train_rows: int
    test_rows: int
    row_assignment_hash: str
    fold_hash: str
    train_hash: str
    test_hash: str
    encoder_hash: str
    _train_row_ids: tuple[str, ...] = field(default=(), repr=False, compare=False)
    _test_row_ids: tuple[str, ...] = field(default=(), repr=False, compare=False)

    def to_json(self) -> dict[str, object]:
        return {
            "view_index": self.view_index,
            "train_rows": self.train_rows,
            "test_rows": self.test_rows,
            "row_assignment_hash": self.row_assignment_hash,
            "fold_hash": self.fold_hash,
            "train_hash": self.train_hash,
            "test_hash": self.test_hash,
            "encoder_hash": self.encoder_hash,
        }


@dataclass(frozen=True)
class SplitViewManifestV1:
    base_dataset: str
    source_family: str
    task: TaskType
    outer_split: OuterSplit
    outer_split_hash: str
    fold_plan_hash: str
    views: tuple[SplitViewV1, ...]
    schema: str = "SplitViewManifest.v1"

    def to_json(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "base_dataset": self.base_dataset,
            "source_family": self.source_family,
            "task": self.task.value,
            "outer_split": self.outer_split.value,
            "outer_split_hash": self.outer_split_hash,
            "fold_plan_hash": self.fold_plan_hash,
            "views": [view.to_json() for view in self.views],
        }

    @property
    def sha256(self) -> str:
        return _canonical_hash(self.to_json())


def build_split_view_manifest(
    *,
    dataset: str,
    source_family: str,
    task: TaskType | str,
    outer_split: OuterSplit | str,
    row_ids: Sequence[str],
    train_sha256: str,
    test_sha256: str | None,
) -> SplitViewManifestV1:
    """Build five complementary, balanced views with train-only encoder bindings."""
    task = TaskType(task)
    outer_split = OuterSplit(outer_split)
    if not dataset or not source_family or len(row_ids) < 5:
        raise ValueError("dataset/family and at least five row identities are required")
    if any(not row_id for row_id in row_ids) or len(set(row_ids)) != len(row_ids):
        raise ValueError("row identities must be non-empty and unique")
    _require_sha256("train_sha256", train_sha256)
    if test_sha256 is not None:
        _require_sha256("test_sha256", test_sha256)

    row_hashes = {row: _row_hash(dataset, row) for row in row_ids}
    ordered = sorted(row_ids, key=lambda row: bytes.fromhex(row_hashes[row]))
    buckets: list[list[str]] = [[] for _ in range(5)]
    for index, row in enumerate(ordered):
        buckets[index % 5].append(row)

    views: list[SplitViewV1] = []
    for view_index, test_row_ids in enumerate(buckets):
        test_set = set(test_row_ids)
        train_row_ids = tuple(row for row in ordered if row not in test_set)
        test_row_ids_tuple = tuple(test_row_ids)
        train_row_hashes = tuple(row_hashes[row] for row in train_row_ids)
        test_row_hashes = tuple(row_hashes[row] for row in test_row_ids_tuple)
        train_hash = _row_set_hash(train_row_hashes)
        test_hash = _row_set_hash(test_row_hashes)
        assignment = {row_hashes[row]: "test" if row in test_set else "train" for row in ordered}
        row_assignment_hash = _canonical_hash(
            {
                "schema": "SplitViewRowAssignment.v1",
                "view_index": view_index,
                "assignment": assignment,
            }
        )
        fold_hash = _canonical_hash(
            {
                "schema": "SplitViewFold.v1",
                "view_index": view_index,
                "row_assignment_hash": row_assignment_hash,
            }
        )
        encoder_kinds = ["dataset", "target"]
        if task is TaskType.BINARY:
            encoder_kinds.append("class_stats")
        encoder_hash = _canonical_hash(
            {
                "schema": "TrainOnlyEncoderFits.v1",
                "task": task.value,
                "train_hash": train_hash,
                "encoders": encoder_kinds,
            }
        )
        views.append(
            SplitViewV1(
                view_index=view_index,
                train_rows=len(train_row_ids),
                test_rows=len(test_row_ids_tuple),
                row_assignment_hash=row_assignment_hash,
                fold_hash=fold_hash,
                train_hash=train_hash,
                test_hash=test_hash,
                encoder_hash=encoder_hash,
                _train_row_ids=train_row_ids,
                _test_row_ids=test_row_ids_tuple,
            )
        )

    outer_split_hash = _canonical_hash(
        {
            "schema": "OuterSplitBinding.v1",
            "base_dataset": dataset,
            "source_family": source_family,
            "task": task.value,
            "outer_split": outer_split.value,
            "train_sha256": train_sha256,
            "test_sha256": test_sha256,
        }
    )
    fold_plan_hash = _canonical_hash(
        {
            "schema": "FiveComplementaryViews.v1",
            "outer_split_hash": outer_split_hash,
            "views": [view.to_json() for view in views],
        }
    )
    return SplitViewManifestV1(
        base_dataset=dataset,
        source_family=source_family,
        task=task,
        outer_split=outer_split,
        outer_split_hash=outer_split_hash,
        fold_plan_hash=fold_plan_hash,
        views=tuple(views),
    )


@dataclass(frozen=True)
class BinaryTargetSummaryV1:
    split_view_manifest_hash: str
    view_index: int
    training_rows_hash: str
    training_row_count: int
    positive_row_count: int
    values: tuple[float, ...]
    task: TaskType = TaskType.BINARY
    schema: str = "BinaryTargetSummary.v1"

    def __post_init__(self) -> None:
        _require_sha256("split_view_manifest_hash", self.split_view_manifest_hash)
        _require_sha256("training_rows_hash", self.training_rows_hash)
        if self.task is not TaskType.BINARY:
            raise ValueError("binary target summary task must be binary")
        if self.view_index not in range(5):
            raise ValueError("binary target summary view_index must be in 0..5")
        if self.training_row_count < 2:
            raise ValueError("binary target summary requires at least two training rows")
        if not 0 < self.positive_row_count < self.training_row_count:
            raise ValueError("binary target summary requires both classes")
        if len(self.values) != 512 or any(not math.isfinite(value) for value in self.values):
            raise ValueError("binary target summary must be a finite 512-vector")
        if any(value != 0.0 for value in self.values[7:]):
            raise ValueError("BinaryTargetSummary.v1 reserved values must be zero")
        rate = self.positive_row_count / self.training_row_count
        expected = (
            self.training_row_count / (self.training_row_count + 1.0),
            rate,
            min(rate, 1.0 - rate),
            -(rate * math.log(rate) + (1.0 - rate) * math.log(1.0 - rate)),
            max(-12.0, min(12.0, math.log(rate / (1.0 - rate)))),
            0.5 / rate,
            0.5 / (1.0 - rate),
        )
        if any(
            not math.isclose(actual, wanted, rel_tol=1e-6, abs_tol=1e-6)
            for actual, wanted in zip(self.values[:7], expected, strict=True)
        ):
            raise ValueError("binary target summary values do not match its row counts")

    @property
    def vector(self) -> tuple[float, ...]:
        """Compatibility alias for the model's target-embedding input."""
        return self.values

    def to_json(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "task": self.task.value,
            "split_view_manifest_hash": self.split_view_manifest_hash,
            "view_index": self.view_index,
            "training_rows_hash": self.training_rows_hash,
            "training_row_count": self.training_row_count,
            "positive_row_count": self.positive_row_count,
            "values": list(self.values),
        }


def binary_target_summary(
    training_targets: Sequence[int | float],
    *,
    training_row_ids: Sequence[str],
    split_view_manifest: SplitViewManifestV1,
    view_index: int,
) -> BinaryTargetSummaryV1:
    """Summarize one selected view's training rows and bind their exact row hash."""
    if split_view_manifest.task is not TaskType.BINARY:
        raise ValueError("binary target summary requires a binary split view manifest")
    if view_index not in range(len(split_view_manifest.views)):
        raise ValueError("binary target summary view_index must be in 0..5")
    if len(training_targets) != len(training_row_ids) or not training_targets:
        raise ValueError("binary targets and row identities must be non-empty and aligned")
    if any(isinstance(value, bool) or value not in (0, 1, 0.0, 1.0) for value in training_targets):
        raise ValueError("binary targets must be a non-empty strict 0/1 sequence")
    view = split_view_manifest.views[view_index]
    if tuple(training_row_ids) != view._train_row_ids:
        raise ValueError("binary target rows do not match the selected split view")
    actual_train_hash = _row_set_hash(
        [_row_hash(split_view_manifest.base_dataset, row) for row in training_row_ids]
    )
    if actual_train_hash != view.train_hash:
        raise ValueError("binary target rows do not match the selected training hash")

    n = len(training_targets)
    positive = sum(int(value) for value in training_targets)
    if positive in {0, n}:
        raise ValueError("binary target summary requires both classes")
    rate = positive / n
    entropy = -(rate * math.log(rate) + (1.0 - rate) * math.log(1.0 - rate))
    log_odds = max(-12.0, min(12.0, math.log(rate / (1.0 - rate))))
    defined_values = (
        n / (n + 1.0),
        rate,
        min(rate, 1.0 - rate),
        entropy,
        log_odds,
        0.5 / rate,
        0.5 / (1.0 - rate),
    )
    values = tuple(_f32(value) for value in defined_values) + (0.0,) * (512 - 7)
    return BinaryTargetSummaryV1(
        split_view_manifest_hash=split_view_manifest.sha256,
        view_index=view_index,
        training_rows_hash=view.train_hash,
        training_row_count=n,
        positive_row_count=positive,
        values=values,
    )


def build_binary_target_summaries(
    training_targets: Sequence[int | float],
    *,
    row_ids: Sequence[str],
    split_view_manifest: SplitViewManifestV1,
) -> tuple[BinaryTargetSummaryV1, ...]:
    """Build all five train-only summaries for one binary base dataset."""
    if len(training_targets) != len(row_ids) or len(set(row_ids)) != len(row_ids):
        raise ValueError("binary targets and unique row identities must be aligned")
    target_by_row = dict(zip(row_ids, training_targets, strict=True))
    return tuple(
        binary_target_summary(
            [target_by_row[row] for row in view._train_row_ids],
            training_row_ids=view._train_row_ids,
            split_view_manifest=split_view_manifest,
            view_index=view.view_index,
        )
        for view in split_view_manifest.views
    )
