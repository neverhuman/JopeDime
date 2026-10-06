"""ROWSEL-SPLIT v2 with annotated groups (owner ruling GROUP-UNION-B): the target-free split v2, amended so that rows
sharing an annotated group value are kept together, unioned with the exact-duplicate feature groups. Shared by the T1
builder (tools/v2_rowsel_build.py) and the clean-route builder. dltrain/rowsel_raw_split_v2.py (97340df3) is NOT
modified: the raw route's split identity stays fixed.

  components  union-find over rows: two rows are joined when their parsed FEATURE bytes are exactly equal (as v2) OR
              their annotated group values are equal. A row whose group value is None is in no annotation group. The
              target is never an input: callers pass feature bytes that exclude it and group values from a feature column.
  ranking     a component's rank is the smallest rowsel_raw_split_v2.row_rank(h, index, partition) over its rows, i.e.
              sha256(h NUL 'trainonly-rowsel-split-v1' NUL partition NUL decimal(index)); ties (impossible) to the smaller
              position
  placement   components in ascending rank, each whole into the currently smallest fold, ties to the lower fold
  identity    split_sha256 = sha256 of canonical JSON {schema 'rowsel-split-v2-groups', dataset_hash, partition (the
              universe label, e.g. 'raw' or 't1'), group_column (the spec: index and name; null without groups), folds}
              cut_id = 'trainonly-split-v2g:' + split_sha256
With group_values=None, split() returns exactly rowsel_raw_split_v2's assignment for the same rows and partition.
Numpy-free; the standard library and rowsel_raw_split_v2.row_rank only.
"""
from __future__ import annotations

import hashlib
import json
from typing import Hashable, Mapping, Sequence

from dltrain import rowsel_raw_split_v2 as _v2

SPLIT_SCHEMA = "rowsel-split-v2-groups"
CUT_PREFIX = "trainonly-split-v2g:"
SPLIT_DOMAIN = _v2.SPLIT_DOMAIN
OUTER_FOLDS = 5
RULE = ("union-find over rows joined by exactly equal parsed feature bytes (target excluded) OR an equal annotated group "
        "value; a component is ranked by the smallest sha256(h NUL domain NUL partition NUL decimal(index)) over its rows; "
        "components in ascending rank, each whole into the currently smallest fold, ties to the lower fold; no target")


def _canon(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def components(positions: Sequence[int], feature_bytes: Sequence[bytes],
               group_values: Sequence[Hashable | None] | None = None) -> list[list[int]]:
    """Components (lists of positions) of the union of exact-feature equality and annotated-group equality."""
    pos = [int(p) for p in positions]
    n = len(pos)
    if len(feature_bytes) != n or (group_values is not None and len(group_values) != n):
        raise ValueError("positions, feature_bytes and group_values must align")
    if len(set(pos)) != n:
        raise ValueError("positions are not unique")
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    first_feature: dict[bytes, int] = {}
    first_group: dict = {}
    for i in range(n):
        k = bytes(feature_bytes[i])
        if k in first_feature:
            union(first_feature[k], i)
        else:
            first_feature[k] = i
        if group_values is not None and group_values[i] is not None:
            g = group_values[i]
            if g in first_group:
                union(first_group[g], i)
            else:
                first_group[g] = i
    out: dict[int, list[int]] = {}
    for i in range(n):
        out.setdefault(find(i), []).append(pos[i])
    return [sorted(v) for v in out.values()]


def split(dataset_hash: str, positions: Sequence[int], feature_bytes: Sequence[bytes],
          group_values: Sequence[Hashable | None] | None = None, partition: str = "raw",
          folds: int = OUTER_FOLDS) -> dict[int, list[int]]:
    """The fold assignment {fold: sorted positions}."""
    ranked = sorted((min(_v2.row_rank(dataset_hash, p, partition) for p in comp), min(comp), comp)
                    for comp in components(positions, feature_bytes, group_values))
    sizes = [0] * folds
    out: dict[int, list[int]] = {f: [] for f in range(folds)}
    for _, _, comp in ranked:
        f = min(range(folds), key=lambda i: (sizes[i], i))
        out[f].extend(comp)
        sizes[f] += len(comp)
    return {f: sorted(v) for f, v in out.items()}


def group_spec(index: int | None, name: str | None) -> dict | None:
    return None if index is None else {"index": int(index), "name": str(name)}


def split_sha256(dataset_hash: str, partition: str, group_column: Mapping | None, folds: Mapping[int, Sequence[int]]) -> str:
    body = {"schema": SPLIT_SCHEMA, "dataset_hash": dataset_hash, "partition": partition,
            "group_column": None if group_column is None else {"index": int(group_column["index"]), "name": str(group_column["name"])},
            "folds": [[int(p) for p in folds[f]] for f in sorted(folds)]}
    return hashlib.sha256(_canon(body)).hexdigest()


def cut_id(split_sha: str) -> str:
    return CUT_PREFIX + split_sha


def straddling_groups(folds: Mapping[int, Sequence[int]], positions: Sequence[int],
                      group_values: Sequence[Hashable | None]) -> int:
    """How many annotated group values occur in more than one fold (0 for any split that keeps groups together)."""
    fold_of = {int(p): f for f, v in folds.items() for p in v}
    seen: dict = {}
    bad = set()
    for p, g in zip(positions, group_values):
        if g is None:
            continue
        f = fold_of[int(p)]
        if seen.setdefault(g, f) != f:
            bad.add(g)
    return len(bad)


def component_partition(positions, feature_bytes, group_values=None) -> list[tuple]:
    """Canonical component structure (sorted tuple of sorted position tuples): what a group- and duplicate-preserving
    change must leave unchanged."""
    return sorted(tuple(c) for c in components(positions, feature_bytes, group_values))
