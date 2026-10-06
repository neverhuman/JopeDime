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

  admissibility  target_tied_group(group_values, y) refuses a group column that copies or determines the target (PR
              #218 finding 8). It is a DATASET-LEVEL refusal decided before any split: when it returns a reason no context
              of the dataset is built; when it returns None the dataset is admitted and y never reaches the split
              (split() and components() take no target). It does not change split(), components() or any split identity.
Numpy-free; the standard library and rowsel_raw_split_v2.row_rank only.
"""
from __future__ import annotations

import hashlib
import json
import math
import struct
from typing import Hashable, Mapping, Sequence

from dltrain import rowsel_raw_split_v2 as _v2

SPLIT_SCHEMA = "rowsel-split-v2-groups"
CUT_PREFIX = "trainonly-split-v2g:"
SPLIT_DOMAIN = _v2.SPLIT_DOMAIN
OUTER_FOLDS = 5
RULE = ("union-find over rows joined by exactly equal parsed feature bytes (target excluded) OR an equal annotated group "
        "value; a component is ranked by the smallest sha256(h NUL domain NUL partition NUL decimal(index)) over its rows; "
        "components in ascending rank, each whole into the currently smallest fold, ties to the lower fold; no target")


TARGET_TIE_REASONS = ("group_bytes_equal_target_bytes", "group_affine_copy_of_target", "target_function_of_group",
                      "group_is_binned_target", "target_nearly_determined_by_group")
AFFINE_REL_RMSE = 1e-10                              # as dltrain/t1prep.target_copy_columns (audit_target_leakage.py)
BINNED_MIN_GROUPS = 2                                # multi-row groups with pairwise disjoint y ranges that make a binning
NEAR_DETERMINED = (99, 100)                          # share of ALL rows in multi-row single-target groups: >= 99/100
_F64 = struct.Struct("<d")


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
          group_values: Sequence[Hashable | None] | None, partition: str,
          folds: int = OUTER_FOLDS) -> dict[int, list[int]]:
    """The fold assignment {fold: sorted positions}. group_values (None: no annotation) and partition (the universe
    label, e.g. 'raw' or 't1') are required: no default selects a universe (PR #218 finding 10 / F5)."""
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


def _group_number(g) -> float | None:
    """The numeric value of a group value, for the affine test: an int/float, or 8 bytes of little-endian float64 (the
    builders' group encoding: canonical '<f8' bytes of a T1 column). Anything else (None, a missing-cell marker, text)
    is not numeric. Non-finite values are not numeric."""
    if isinstance(g, bool):
        return None
    if isinstance(g, (int, float)):
        v = float(g)
    elif isinstance(g, (bytes, bytearray)) and len(g) == 8:
        v = _F64.unpack(bytes(g))[0]
    else:
        return None
    return v if math.isfinite(v) else None


def target_tied_group(group_values: Sequence[Hashable | None], y: Sequence[float], rel_tol: float = AFFINE_REL_RMSE) -> str | None:
    """Dataset-level admissibility of a group column: None admits it, else the refusal reason (TARGET_TIE_REASONS).

    A group column must neither copy nor determine the target, because group values of every row (outer-test rows
    included) enter the union-find. Refused when, over the dataset's rows (row i: group_values[i], y[i]):
      group_bytes_equal_target_bytes  every row's group value equals the '<f8' bytes of its target (canonical +0.0);
      group_affine_copy_of_target     over the rows whose group value is numeric (_group_number; at least 3), group
                                      = a*y + b with a != 0: the least-squares fit of y on the group value has relative
                                      RMSE <= rel_tol (1e-10, as t1prep.target_copy_columns);
      target_function_of_group        at least one group (rows sharing a non-None group value) has >= 2 rows, and every
                                      such group has a single distinct target value. All-singleton groups are trivially
                                      functional and admitted.
      group_is_binned_target          the group is a function of y (each distinct y value of the grouped rows lies in
                                      exactly one group) and at least BINNED_MIN_GROUPS (2) multi-row groups exist whose
                                      [min y, max y] ranges are pairwise disjoint (all of them): a monotone step function
                                      of y such as floor(y/k) or quantile bins, though y varies inside each group;
      target_nearly_determined_by_group  at least 99% of ALL rows (NEAR_DETERMINED) sit in multi-row groups that each
                                      have a single distinct target value.
    This reads y of every row, but only to decide whether the dataset is admitted at all: a refusal means no context of
    the dataset is built; an admission changes nothing in split()/components(), which never receive y. Pure; no state."""
    n = len(group_values)
    ys = [float(v) + 0.0 for v in y]
    if len(ys) != n:
        raise ValueError("group_values and y must align")
    if n == 0:
        return None
    if all(isinstance(g, (bytes, bytearray)) and bytes(g) == _F64.pack(t) for g, t in zip(group_values, ys)):
        return "group_bytes_equal_target_bytes"
    pairs = [(v, t) for v, t in ((_group_number(g), t) for g, t in zip(group_values, ys)) if v is not None]
    m = len(pairs)
    if m >= 3:
        gm = math.fsum(v for v, _ in pairs) / m
        tm = math.fsum(t for _, t in pairs) / m
        sgg = math.fsum((v - gm) ** 2 for v, _ in pairs)
        stt = math.fsum((t - tm) ** 2 for _, t in pairs)
        if sgg > 0 and stt > 0:
            slope = math.fsum((v - gm) * (t - tm) for v, t in pairs) / sgg
            icpt = tm - slope * gm
            rmse = math.sqrt(math.fsum((t - (slope * v + icpt)) ** 2 for v, t in pairs) / m)
            if slope != 0 and rmse / math.sqrt(stt / m) <= rel_tol:
                return "group_affine_copy_of_target"
    stats: dict = {}                                     # group -> [rows, min y, max y]
    y_group: dict = {}                                   # y -> its group while y lies in one group only
    y_function = True
    for g, t in zip(group_values, ys):
        if g is None:
            continue
        st = stats.get(g)
        if st is None:
            stats[g] = [1, t, t]
        else:
            st[0] += 1
            if t < st[1]:
                st[1] = t
            if t > st[2]:
                st[2] = t
        if y_group.setdefault(t, g) != g:
            y_function = False
    multi = [st for st in stats.values() if st[0] >= 2]
    if multi and all(lo == hi for _, lo, hi in multi):
        return "target_function_of_group"
    if y_function and len(multi) >= BINNED_MIN_GROUPS:
        ranges = sorted((lo, hi) for _, lo, hi in multi)
        if all(b[0] > a[1] for a, b in zip(ranges, ranges[1:])):
            return "group_is_binned_target"
    covered = sum(c for c, lo, hi in multi if lo == hi)
    if covered * NEAR_DETERMINED[1] >= NEAR_DETERMINED[0] * n:
        return "target_nearly_determined_by_group"
    return None
