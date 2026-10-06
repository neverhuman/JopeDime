"""Test helper: the registry split grouped-stratified-kfold/v1 (regression path), for must-fail tests only.

Copied verbatim (functions csv_fields .. generate_folds) from the read-only split-scope port
  /home/ubuntu/scratch/jopedime-dl-training-v1/tmp/split-scope/gsk.py  sha256 83183db43a42740c3929c6de302e9633ac2c2731fdad143a9dac5e020027c599
which reproduces every census split object byte for byte (receipt split-scope-nonraw-20261006T034028Z.json, ff65a65d) and
ports crates/jope-datasets/src/virtual_split.rs (xbabe2-recovery worktree, sha256 bac73aca...). Its duplicate key hashes
EVERY parsed field of a row, the target included (virtual_split.rs:639-665): the reason for NONRAW-SPLIT-A.
"""
from __future__ import annotations

import hashlib
import struct

MASK64 = (1 << 64) - 1
REPEATS = 5
REQUESTED_FOLDS = 5
BASE_SEED = 20260930


def csv_fields(line: bytes) -> list[bytes]:
    if line.endswith(b"\n"):
        line = line[:-1]
    if line.endswith(b"\r"):
        line = line[:-1]
    if b'"' not in line:
        return line.split(b",")
    fields, field = [], bytearray()
    i, quoted, closed = 0, False, False
    while i < len(line):
        c = line[i:i + 1]
        if quoted:
            if c == b'"':
                if line[i + 1:i + 2] == b'"':
                    field += b'"'; i += 1
                else:
                    quoted, closed = False, True
            else:
                field += c
        elif c == b",":
            fields.append(bytes(field)); field = bytearray(); closed = False
        elif c == b'"' and not field and not closed:
            quoted = True
        else:
            if closed or c == b'"':
                raise ValueError("malformed quoted CSV field")
            field += c
        i += 1
    if quoted:
        raise ValueError("unterminated quoted field")
    fields.append(bytes(field))
    return fields


def csv_lines(body: bytes) -> list[bytes]:
    """Rust read_until(b'\\n') semantics: every chunk ending in \\n, plus a final unterminated chunk."""
    parts = body.split(b"\n")
    if parts and parts[-1] == b"":
        parts.pop()
    return parts


def dup_key(fields: list[bytes]) -> bytes:
    h = hashlib.sha256()
    for f in fields:
        h.update(struct.pack("<Q", len(f)))
        h.update(f)
    return h.digest()


def components(dup_keys: list, groups: list | None = None) -> list[list[int]]:
    n = len(dup_keys)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(l, r):
        l, r = find(l), find(r)
        if l != r:
            parent[r] = l

    gfirst, dfirst = {}, {}
    for i in range(n):
        if groups is not None and groups[i] is not None:
            g = groups[i]
            if g in gfirst:
                union(gfirst[g], i)  # HashMap::insert returns the OLD value; the map now holds i
            gfirst[g] = i
        k = dup_keys[i]
        if k in dfirst:
            union(dfirst[k], i)
        dfirst[k] = i
    out: dict[int, list[int]] = {}
    for i in range(n):
        out.setdefault(find(i), []).append(i)
    return [out[r] for r in sorted(out)]


def salted(seed: int, first: int) -> int:
    d = hashlib.sha256(struct.pack("<QI", seed & MASK64, first)).digest()
    return int.from_bytes(d[:8], "little")


def assign(groups: list[list[int]], folds: int, seed: int) -> list[list[int]]:
    order = sorted(range(len(groups)), key=lambda g: (-len(groups[g]), salted(seed, groups[g][0])))
    assigned = [[] for _ in range(folds)]
    sizes = [0] * folds
    for pos, gi in enumerate(order):
        if pos < folds:
            chosen = pos
        else:
            m = min(sizes)
            tied = [f for f in range(folds) if sizes[f] == m]
            chosen = tied[0] if len(tied) == 1 else min(tied, key=lambda f: salted(seed ^ f, groups[gi][0]))
        assigned[chosen].extend(groups[gi])
        sizes[chosen] += len(groups[gi])
    return [sorted(a) for a in assigned]


def generate_folds(dup_keys: list, base_seed: int, repeats=REPEATS, groups=None):
    comps = components(dup_keys, groups)
    k = min(len(comps), REQUESTED_FOLDS)
    if k < 2:
        return 0, [], comps
    out = []
    for r in range(repeats):
        seed = (base_seed + r) & MASK64
        for f, idx in enumerate(assign(comps, k, seed)):  # regression: attempt 0 always succeeds
            out.append({"repeat": r, "fold": f, "seed": seed, "test_indices": idx})
    return k, out, comps
