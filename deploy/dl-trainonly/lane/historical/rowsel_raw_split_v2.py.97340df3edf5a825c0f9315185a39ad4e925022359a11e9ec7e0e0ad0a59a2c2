"""ROWSEL-SPLIT outer split, owner-literal ordering (v2). PREPARED, NOT YET USED BY ANY BUILD: rowsel_raw.outer_split
(v1) is unchanged, so every ROWSEL-RAW-v1 / -coverage / -nonnumeric receipt still verifies against the code that built
it. A rebuild opts in explicitly.

Owner ruling ROWSEL-SPLIT (owner-decisions-20261005T210659Z.json): "value-free 5-fold outer split (sha256 of dataset
hash, partition, index; exact-duplicate rows kept together; no target stratification)".

v1 (rowsel_raw.outer_split) ordered the duplicate groups by sha256(h NUL domain NUL group key), the group key being
sha256 of the rows' parsed FEATURE bytes. Grouping by feature equality is part of the ruling; ORDERING by feature content
is not: a duplicate-structure-preserving change of outer-test feature or text values re-ranks the groups and, through the
greedy fill, moves most rows to other folds (red-team receipt redteam-nonnumeric-publish/receipt.json, 76-82 % of rows on
24/24 sampled contexts).

v2:
  row rank     sha256(h NUL 'trainonly-rowsel-split-v1' NUL partition NUL decimal(index))      (positions only)
  groups       exact-duplicate parsed feature bytes (rowsel_raw.group_keys; the target is never part of it). The group
               key is a dict key for grouping only; it never orders anything.
  group rank   the smallest row rank over the group's members (ties, impossible for sha256, to the smaller position)
  fill         groups in ascending group rank, each whole into the currently smallest fold, ties to the lower fold
So the fold assignment is a function of (dataset hash, the positions, the partition of the positions into duplicate
groups). Any change of feature, text or target values that leaves that partition unchanged leaves every fold unchanged.
A change that alters the duplicate partition can still move rows: that is the admitted grouping, not the ordering.

value_check() is the regression gate for this property, with its own validity check (the perturbation must preserve the
duplicate partition of the WHOLE universe, else the case is reported invalid, never passed). The v1 ordering must fail it.
"""
from __future__ import annotations

import hashlib
from typing import Callable, Mapping, Sequence

import numpy as np

from dltrain import rowsel_raw, t1raw

SPLIT_SCHEMA = "rowsel-raw-split-v2"
SPLIT_DOMAIN = "trainonly-rowsel-split-v1"           # the domain string the coordinator's restatement of the rule names
PARTITION = rowsel_raw.PARTITION
OUTER_FOLDS = rowsel_raw.OUTER_FOLDS
CF_SALT = "rowsel-raw-split-v2-cf"
CF_KINDS = ("features_and_text", "targets", "features_text_and_targets")
NUMERIC_REMAP_BASE = -1.0e15                         # remapped outer-test numerics: -1e15 - rank (exact in float64)
RULE = ("groups of exact-duplicate parsed feature bytes (target excluded); a group is ranked by the smallest "
        "sha256(h NUL domain NUL partition NUL decimal(index)) over its rows; groups in ascending rank, each whole into "
        "the currently smallest fold, ties to the lower fold; no target stratification")


def row_rank(dataset_hash: str, index: int, partition: str = PARTITION) -> bytes:
    return hashlib.sha256(dataset_hash.encode() + b"\0" + SPLIT_DOMAIN.encode() + b"\0" + partition.encode() + b"\0"
                          + str(int(index)).encode()).digest()


def duplicate_groups(positions, values, specs) -> dict[str, list[int]]:
    """Positions grouped by exact parsed feature bytes (rowsel_raw.group_keys)."""
    pos = [int(p) for p in np.asarray(positions, dtype=np.int64).reshape(-1)]
    groups: dict[str, list[int]] = {}
    for p, g in zip(pos, rowsel_raw.group_keys(values, specs)):
        groups.setdefault(g, []).append(p)
    return groups


def duplicate_partition(positions, values, specs) -> list[tuple]:
    """Canonical duplicate partition: sorted tuple of sorted position groups (no feature bytes in it)."""
    return sorted(tuple(sorted(v)) for v in duplicate_groups(positions, values, specs).values())


def outer_split(dataset_hash: str, positions, values, specs, folds: int = OUTER_FOLDS,
                partition: str = PARTITION) -> dict[int, list[int]]:
    """ROWSEL-SPLIT v2 (owner-literal ordering). Same signature as rowsel_raw.outer_split."""
    ranked = sorted((min(row_rank(dataset_hash, p, partition) for p in members), min(members), members)
                    for members in duplicate_groups(positions, values, specs).values())
    sizes = [0] * folds
    out: dict[int, list[int]] = {f: [] for f in range(folds)}
    for _, _, members in ranked:
        f = min(range(folds), key=lambda i: (sizes[i], i))
        out[f].extend(members)
        sizes[f] += len(members)
    return {f: sorted(v) for f, v in out.items()}


def split_sha256(dataset_hash: str, folds: Mapping[int, Sequence[int]]) -> str:
    return rowsel_raw.sha(rowsel_raw.canon({"schema": SPLIT_SCHEMA, "dataset_hash": dataset_hash, "partition": PARTITION,
                                            "folds": [[int(p) for p in folds[f]] for f in sorted(folds)]}))


def split_report(prep, folds: Mapping[int, Sequence[int]]) -> dict:
    """rowsel_raw.split_report's structural checks (exact partition, no group straddling folds, sizes) under v2 identity."""
    rep = rowsel_raw.split_report(prep, folds)
    rep.update(schema=SPLIT_SCHEMA, sha256=split_sha256(prep.dataset_hash, folds), domain=SPLIT_DOMAIN, rule=RULE)
    return rep


# ------------------------------------------------------------------------------------------------- value gate

def perturb_fold_values(values, specs, rows, kind: str, salt: str = CF_SALT):
    """(values, specs) with the rows at `rows` (row numbers) changed and every other row untouched. Feature/text changes
    are injective per column, so they preserve the duplicate structure among the changed rows, and map to values no
    other row holds: numeric cells -> NUMERIC_REMAP_BASE - rank of the cell's value among the changed rows (NaN stays
    NaN); categorical/text cells -> a NEW key (old key + NUL salt) under a NEW code appended to a copy of the spec's
    inverse (missing stays missing). Targets -> fresh values (never part of the grouping)."""
    V = np.array(values, dtype=np.float64, copy=True)
    out_specs = [dict(s) for s in specs]
    rows = np.asarray(sorted(int(r) for r in rows), dtype=np.int64)
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(f"{salt}|{kind}".encode()).digest()[:8], "little"))
    if kind in ("features_and_text", "features_text_and_targets"):
        for s in out_specs[:-1]:
            j = s["index"]
            col = V[rows, j]
            if s["kind"] == "categorical_codes":
                inv = dict(s["inverse"])
                nxt = max(inv) + 1
                remap = {}
                for c in sorted({int(c) for c in col.tolist() if c != t1raw.MISSING_CODE}):
                    remap[c] = nxt
                    inv[nxt] = str(inv[c]) + "\0" + salt
                    nxt += 1
                V[rows, j] = [float(remap[int(c)]) if c != t1raw.MISSING_CODE else c for c in col.tolist()]
                s["inverse"] = inv
            else:
                fin = np.isfinite(col)
                distinct = np.unique(col[fin])
                rank = {float(v): k for k, v in enumerate(distinct.tolist())}
                V[rows, j] = [NUMERIC_REMAP_BASE - rank[float(v)] if np.isfinite(v) else v for v in col.tolist()]
    if kind in ("targets", "features_text_and_targets"):
        V[rows, -1] = rng.normal(1e7, 1e6, size=rows.size)
    if kind not in CF_KINDS:
        raise rowsel_raw.RowselRawError("unknown_perturbation", kind)
    return V, out_specs


def value_check(split_fn: Callable, prep, fold: int = 0, salt: str = CF_SALT) -> dict:
    """The fold assignment must not move when the values of one outer-test fold change in a duplicate-structure-
    preserving way (features, text, targets). Each case first verifies that the duplicate partition of the WHOLE
    universe is unchanged (else 'valid': False, which counts against invariance). split_fn(h, positions, values, specs)."""
    h, pos = prep.dataset_hash, prep.positions
    base = split_fn(h, pos, prep.values, prep.specs)
    rows = prep.rows_of(base[fold])
    part0 = duplicate_partition(pos, prep.values, prep.specs)
    fold_of = {p: f for f, v in base.items() for p in v}
    cases = {}
    for kind in CF_KINDS:
        V2, s2 = perturb_fold_values(prep.values, prep.specs, rows, kind, salt)
        changed = int(np.sum(~((V2 == prep.values) | (np.isnan(V2) & np.isnan(prep.values)))))
        valid = duplicate_partition(pos, V2, s2) == part0 and changed > 0
        got = split_fn(h, pos, V2, s2)
        moved = sum(1 for f, v in got.items() for p in v if fold_of[p] != f)
        cases[kind] = {"valid": bool(valid), "cells_changed": changed, "split_identical": moved == 0 and
                       {f: list(map(int, got[f])) for f in sorted(got)} == {f: list(map(int, base[f])) for f in sorted(base)},
                       "rows_moved": int(moved)}
    return {"fold": int(fold), "outer_test_rows": int(rows.size), "cases": cases,
            "valid": all(c["valid"] for c in cases.values()),
            "invariant": all(c["valid"] and c["split_identical"] for c in cases.values())}
