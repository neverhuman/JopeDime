"""Canonical identity for an ordered Q1 FIT cut and its deterministic folds."""

import hashlib
import json
from pathlib import Path

SCHEMA = "jope.q1-cut/v1"
FIELDS = {
    "schema",
    "dataset_sha256",
    "train_blob_sha256",
    "train_rows",
    "target",
    "feature_columns",
    "folds",
}


def canonical_bytes(definition):
    if not isinstance(definition, dict) or set(definition) != FIELDS:
        raise ValueError("Q1 cut definition has missing or unknown fields")
    if definition["schema"] != SCHEMA:
        raise ValueError("unsupported Q1 cut schema")
    for name in ("dataset_sha256", "train_blob_sha256"):
        value = definition[name]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)
        ):
            raise ValueError(f"invalid {name}")
    if type(definition["train_rows"]) is not int or definition["train_rows"] < 2:
        raise ValueError("train_rows must be an integer >= 2")
    target = definition["target"]
    columns = definition["feature_columns"]
    if (
        not isinstance(target, str)
        or not target
        or not isinstance(columns, list)
        or not columns
    ):
        raise ValueError("target and ordered feature_columns are required")
    if (
        any(not isinstance(c, str) or not c for c in columns)
        or len(set(columns)) != len(columns)
        or target in columns
    ):
        raise ValueError(
            "feature_columns must be unique nonempty names excluding target"
        )
    folds = definition["folds"]
    if not isinstance(folds, list) or len(folds) < 2:
        raise ValueError("at least two ordered folds are required")
    for fold in folds:
        if not isinstance(fold, dict) or set(fold) != {"repeat", "train", "validation"}:
            raise ValueError("fold shape mismatch")
        if type(fold["repeat"]) is not int or not 0 <= fold["repeat"] <= 255:
            raise ValueError("invalid fold repeat")
        for name in ("train", "validation"):
            rows = fold[name]
            if (
                not isinstance(rows, list)
                or not rows
                or any(
                    type(i) is not int or not 0 <= i < definition["train_rows"]
                    for i in rows
                )
            ):
                raise ValueError(f"invalid fold {name} rows")
            if len(set(rows)) != len(rows):
                raise ValueError(f"duplicate fold {name} row")
        if set(fold["train"]) & set(fold["validation"]):
            raise ValueError("fold train and validation overlap")
    return json.dumps(
        definition,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def cut_sha256(definition):
    return hashlib.sha256(canonical_bytes(definition)).hexdigest()


def load_content_addressed(
    path,
    expected_sha256,
    *,
    expected_dataset_sha256=None,
    expected_train_blob_sha256=None,
):
    """Load a cut definition and verify its address and optional roster binding."""
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError("cut definition must be a regular non-symlink file")
    if source.stem != expected_sha256:
        raise ValueError("cut definition filename differs from its content address")

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate cut definition key")
            result[key] = value
        return result

    def reject_constant(_value):
        raise ValueError("non-finite cut definition value")

    definition = json.loads(
        source.read_text(encoding="utf-8"),
        object_pairs_hook=unique_object,
        parse_constant=reject_constant,
    )
    actual_sha256 = cut_sha256(definition)
    if actual_sha256 != expected_sha256:
        raise ValueError("cut definition differs from its content address")
    if (
        expected_dataset_sha256 is not None
        and definition["dataset_sha256"] != expected_dataset_sha256
    ):
        raise ValueError("cut definition dataset differs from its roster entry")
    if (
        expected_train_blob_sha256 is not None
        and definition["train_blob_sha256"] != expected_train_blob_sha256
    ):
        raise ValueError("cut definition train blob differs from its roster entry")
    return definition


def require_three_fold_fit_partition(definition):
    """Enforce R2's three-way disjoint FIT validation partition and ascending complements."""
    cut_sha256(definition)
    folds = definition["folds"]
    if len(folds) != 3:
        raise ValueError("T0610 roster cuts require exactly three FIT folds")
    train_rows = definition["train_rows"]
    all_rows = set(range(train_rows))
    validation_union = []
    for fold in folds:
        if fold["repeat"] != 0:
            raise ValueError("T0610 roster cuts require repeat 0")
        validation = fold["validation"]
        train = fold["train"]
        if validation != sorted(validation):
            raise ValueError("T0610 validation rows must be ascending")
        if train != sorted(all_rows - set(validation)):
            raise ValueError(
                "T0610 training rows must be the ascending validation complement"
            )
        validation_union.extend(validation)
    if len(validation_union) != train_rows or set(validation_union) != all_rows:
        raise ValueError(
            "T0610 validation folds must form a disjoint three-way FIT partition"
        )
    return definition
