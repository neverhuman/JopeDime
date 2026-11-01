"""Canonical identity for an ordered Q1 FIT cut and its deterministic folds."""

import hashlib
import json

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
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"invalid {name}")
    if type(definition["train_rows"]) is not int or definition["train_rows"] < 2:
        raise ValueError("train_rows must be an integer >= 2")
    target = definition["target"]
    columns = definition["feature_columns"]
    if not isinstance(target, str) or not target or not isinstance(columns, list) or not columns:
        raise ValueError("target and ordered feature_columns are required")
    if any(not isinstance(c, str) or not c for c in columns) or len(set(columns)) != len(columns) or target in columns:
        raise ValueError("feature_columns must be unique nonempty names excluding target")
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
            if not isinstance(rows, list) or not rows or any(type(i) is not int or not 0 <= i < definition["train_rows"] for i in rows):
                raise ValueError(f"invalid fold {name} rows")
            if len(set(rows)) != len(rows):
                raise ValueError(f"duplicate fold {name} row")
        if set(fold["train"]) & set(fold["validation"]):
            raise ValueError("fold train and validation overlap")
    return json.dumps(definition, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def cut_sha256(definition):
    return hashlib.sha256(canonical_bytes(definition)).hexdigest()
