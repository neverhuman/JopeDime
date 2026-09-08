"""Prime grammar loaded from the Rust registry snapshot.

This module deliberately contains no Prime operator table. The vendored Rust
``battle-runtime`` registry is the authority; this JSON is a generated,
reviewable snapshot used by Python policy code and checkpoint metadata.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from ..prime_config import load_config
from .spec import ExprSpec, GrammarSpec, OpSpec, ParamSpec

ENGINE = "prime"
_HEAD_NAMES = {"ridge"}


def registry_snapshot_path() -> Path:
    if configured := os.environ.get("JOPE_OPERATOR_REGISTRY"):
        return Path(configured).expanduser().resolve(strict=False)
    return load_config().registry_snapshot


def registry_document() -> dict[str, Any]:
    with registry_snapshot_path().open(encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, dict):
        raise ValueError("Prime registry snapshot must be a metadata object")
    if not isinstance(document.get("operators"), list):
        raise ValueError("Prime registry snapshot must contain an operators array")
    if not isinstance(document.get("registry_version"), str):
        raise ValueError("Prime registry snapshot must contain registry_version")
    if not isinstance(document.get("plan_version"), int):
        raise ValueError("Prime registry snapshot must contain plan_version")
    if not isinstance(document.get("maximum_output_columns"), int):
        raise ValueError("Prime registry snapshot must contain maximum_output_columns")
    if int(document["maximum_output_columns"]) != load_config().max_width:
        raise ValueError("prime.toml max_width disagrees with the canonical Rust registry")
    digest = document.get("registry_hash")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("Prime registry snapshot must contain a SHA-256 registry_hash")
    return document


def registry_snapshot() -> list[dict[str, Any]]:
    return list(registry_document()["operators"])


def registry_hash() -> str:
    """Canonical semantic hash produced by Rust, shared with the evaluator."""
    return str(registry_document()["registry_hash"])


def registry_snapshot_hash() -> str:
    """Byte hash of the generated, reviewable JSON snapshot file."""
    return hashlib.sha256(registry_snapshot_path().read_bytes()).hexdigest()


def plan_version() -> int:
    return int(registry_document()["plan_version"])


def maximum_output_columns() -> int:
    return int(registry_document()["maximum_output_columns"])


def _default_value(value: Any) -> Any:
    if value is None:
        return None
    if not isinstance(value, dict) or len(value) != 1:
        raise ValueError(f"malformed Rust registry value: {value!r}")
    return next(iter(value.values()))


def _param(entry: dict[str, Any]) -> ParamSpec:
    bins = tuple(_default_value(value) for value in entry.get("bins", []))
    kind = str(entry["kind"])
    py_kind = {"usize": "int", "u64": "int", "f64": "float"}.get(kind, kind)
    return ParamSpec(
        name=str(entry["name"]),
        kind=py_kind,
        bins=bins,
        default=_default_value(entry.get("default")),
        required=bool(entry.get("required")),
    )


def _cost_class(cost: int) -> str:
    return {0: "cheap", 1: "cheap", 2: "medium", 3: "medium", 4: "expensive"}.get(
        int(cost), "extreme"
    )


def _op(entry: dict[str, Any]) -> OpSpec:
    # Task aliasing: the Rust registry only carries the two engine-native booleans
    # (`regression`, `binary_classification`). Policy-side, timeseries rides the
    # regression lane (reduction-v2 tables are scored by regression-family heads) and
    # multiclass rides the binary lane (K-class decode arrives with the ovr_ridge
    # engine head), so each registry boolean grants its alias task as well. Tuple
    # order (regression, binary, timeseries, multiclass) is load-bearing: tokens.py
    # derives TASK token ids in first-seen order.
    regression = bool(entry.get("regression"))
    binary = bool(entry.get("binary_classification"))
    tasks = []
    if regression:
        tasks.append("regression")
    if binary:
        tasks.append("binary")
    if regression:
        tasks.append("timeseries")
    if binary:
        tasks.append("multiclass")
    return OpSpec(
        name=str(entry["name"]),
        role=str(entry["role"]),
        params=tuple(_param(item) for item in entry.get("params", [])),
        families=tuple(entry.get("tags", [])),
        cost_class=_cost_class(int(entry.get("estimated_cost", 0))),
        allow_append=bool(entry.get("allow_append")),
        generate=bool(entry.get("lifecycle", {}).get("generatable", False)),
        tasks=tuple(tasks) or ("regression", "timeseries"),
    )


def _prime_transform_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select Prime transforms by canonical registry metadata, in wire-ID order."""
    selected = [
        entry
        for entry in entries
        if entry.get("role") == "transform" and "prime" in entry.get("tags", [])
    ]
    if not selected:
        raise ValueError("Prime registry exposes no transforms tagged 'prime'")
    names = [str(entry.get("name")) for entry in selected]
    if len(names) != len(set(names)):
        raise ValueError("Prime registry contains duplicate transform names")
    return sorted(selected, key=lambda entry: int(entry["id"]))


def load_registry_grammar() -> GrammarSpec:
    entries = registry_snapshot()
    transform_entries = _prime_transform_entries(entries)
    head_entries = [entry for entry in entries if entry.get("name") in _HEAD_NAMES]
    if {str(entry.get("name")) for entry in head_entries} != _HEAD_NAMES:
        raise ValueError("Prime registry must expose the ridge compatibility head")
    transforms = {str(entry["name"]): _op(entry) for entry in transform_entries}
    heads = {str(entry["name"]): _op(entry) for entry in head_entries}
    version = f"{entry_version()}+sha256:{registry_hash()}"
    return GrammarSpec(
        engine=ENGINE,
        version=version,
        evaluator_version="jope-prime-evald.exact-f64.v1",
        max_transforms=4,
        transforms=transforms,
        heads=heads,
        expr=ExprSpec(),
        fallback_genome="(pipe X (ridge))",
    )


def entry_version() -> str:
    return str(registry_document()["registry_version"])


GRAMMAR = load_registry_grammar()
