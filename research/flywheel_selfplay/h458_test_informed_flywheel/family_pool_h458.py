#!/usr/bin/env python3
"""H458 family pools: the de-duplicated union of each dev91 family's sibling tables.

A dev91 family's contexts are overlapping row samples of one source table. The training bank (`mmap_v1`, cohort
`D91-COMBINED-SHUFFLED`) stores each context as its search + audit rows merged and shuffled. This builder stacks a
family's tables, drops exact duplicate rows and orders the survivors by a keyed hash of their bytes, so the pool
carries no row order, no table membership per row and no benchmark split. It never opens the evaluation bank.

In-corpus by construction: `canonical_promotion_forbidden = true`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

import paths_h458 as P

SCHEMA = "jope.h458.family_pool/v1"
HASH_KEY = b"h458-family-pool-v1"
COHORT = "D91-COMBINED-SHUFFLED"
META_KEYS = (
    "schema",
    "cohort",
    "families",
    "shapes",
    "offsets",
    "member_rows",
    "member_cids",
    "total_rows",
    "total_cells",
    "data_file",
    "data_sha256",
    "source_bank_data_sha256",
    "built_utc",
    "in_corpus_dev91_disclosure",
    "canonical_promotion_forbidden",
)


def row_keys(mat: np.ndarray) -> list[bytes]:
    """Keyed 128-bit digest of every row's exact float64 bytes."""
    c = np.ascontiguousarray(mat, dtype=np.float64)
    return [hashlib.blake2b(c[i].tobytes(), key=HASH_KEY, digest_size=16).digest() for i in range(c.shape[0])]


STRATA_KEY = b"h458-k5-strata-v1"
_CANON_NAN = np.frombuffer((0x7FF8000000000000).to_bytes(8, "little"), dtype="<f8")[0]


def canon_feature_rows(x_rows: np.ndarray) -> np.ndarray:
    """Canonical feature bytes shared by Tracks D, S and G: float64 little-endian, -0.0 -> +0.0, one NaN pattern."""
    c = np.array(x_rows, dtype="<f8", order="C", copy=True)
    c[c == 0.0] = 0.0
    c[np.isnan(c)] = _CANON_NAN
    return c


def feature_stratum(x_rows: np.ndarray, k: int = 5) -> np.ndarray:
    """Cross-fitting stratum of every row from its canonical FEATURE bytes only.

    `k = blake2b(canon(X_row), digest_size=8, key=b"h458-k5-strata-v1") mod 5`. Rows that share X share a stratum
    (so a held-out label cannot leak through a twin row), and a row can be routed to the model that never trained
    on it without reading its label.
    """
    c = canon_feature_rows(x_rows)
    return np.fromiter(
        (int.from_bytes(hashlib.blake2b(c[i].tobytes(), digest_size=8, key=STRATA_KEY).digest(), "little") % k for i in range(c.shape[0])),
        dtype=np.int64,
        count=c.shape[0],
    )


def dedup_and_order(mat: np.ndarray) -> np.ndarray:
    """Unique rows in ascending key order (so the result does not depend on the input order)."""
    keys = row_keys(mat)
    first: dict[bytes, int] = {}
    for i, k in enumerate(keys):
        first.setdefault(k, i)
    order = [first[k] for k in sorted(first)]
    return np.ascontiguousarray(mat[np.asarray(order, dtype=np.int64)], dtype=np.float64)


def build_pool(mmap_dir: Path, out_dir: Path) -> dict:
    out_dir = P.guard_write_path(out_dir)
    if (out_dir / "READY").exists():
        raise SystemExit(f"{out_dir} already built (create-once)")
    src = json.loads((mmap_dir / "meta.json").read_text())
    if any(tok in str(c) for c in src["cohorts"] for tok in P.SEALED_TOKENS) or int(src.get("sealed_cohorts_in_bank", 0)):
        raise SystemExit("sealed cohort token in source bank")
    mm = np.memmap(mmap_dir / src["data_file"], dtype=np.float64, mode="r")
    by_fam: dict[str, list[int]] = {}
    for i, (fam, coh) in enumerate(zip(src["families"], src["cohorts"])):
        if coh == COHORT:
            by_fam.setdefault(str(fam), []).append(i)

    fams, shapes, offsets, member_rows, member_cids, blocks = [], [], [], [], [], []
    off = 0
    for fam in sorted(by_fam):
        idx = by_fam[fam]
        widths = {int(src["shapes"][i][1]) for i in idx}
        if len(widths) != 1:
            raise SystemExit(f"{fam}: sibling tables differ in width {sorted(widths)}")
        tabs = []
        for i in idx:
            r, c = (int(v) for v in src["shapes"][i])
            o = int(src["offsets"][i])
            tabs.append(np.asarray(mm[o : o + r * c], dtype=np.float64).reshape(r, c))
        pool = dedup_and_order(np.vstack(tabs))
        if not np.all(np.isfinite(pool[:, -1])):
            raise SystemExit(f"{fam}: non-finite target in pool")
        fams.append(fam)
        shapes.append([int(pool.shape[0]), int(pool.shape[1])])
        offsets.append(off)
        member_rows.append(sorted(int(src["shapes"][i][0]) for i in idx))
        member_cids.append(sorted(str(src["cids"][i]) for i in idx))
        blocks.append(pool)
        off += int(pool.size)

    out_dir.mkdir(parents=True, exist_ok=True)
    data_p = out_dir / "data.f64"
    with open(data_p, "xb") as fh:
        for b in blocks:
            fh.write(b.tobytes())
    meta = {
        "schema": SCHEMA,
        "cohort": COHORT,
        "families": fams,
        "shapes": shapes,
        "offsets": offsets,
        "member_rows": member_rows,
        "member_cids": member_cids,
        "total_rows": int(sum(s[0] for s in shapes)),
        "total_cells": int(off),
        "data_file": "data.f64",
        "data_sha256": P.digest_file(data_p),
        "source_bank_data_sha256": str(src.get("data_sha256", "")),
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **P.DISCLOSURE,
    }
    assert tuple(meta) == META_KEYS
    with open(out_dir / "meta.json", "x") as fh:
        fh.write(json.dumps(meta, indent=1) + "\n")
    with open(out_dir / "READY", "x") as fh:
        fh.write(meta["data_sha256"] + "\n")
    return meta


class FamilyPool:
    """Read-only view of a built pool. `table(fam)` is `[rows, features + 1]` float64 with the target last."""

    def __init__(self, pool_dir: Path, expect_sha256: str = "", verify_data: bool = False):
        self.dir = Path(pool_dir)
        self.meta = json.loads((self.dir / "meta.json").read_text())
        if self.meta["schema"] != SCHEMA or tuple(self.meta) != META_KEYS:
            raise SystemExit("unexpected family-pool schema or meta keys")
        if not (self.meta["in_corpus_dev91_disclosure"] and self.meta["canonical_promotion_forbidden"]):
            raise SystemExit("family pool without disclosure flags")
        if expect_sha256 and self.meta["data_sha256"] != expect_sha256:
            raise SystemExit("family pool sha256 mismatch")
        if verify_data and P.digest_file(self.dir / "data.f64") != self.meta["data_sha256"]:
            raise SystemExit("family pool data bytes do not match meta")
        self.families: list[str] = list(self.meta["families"])
        self._mm = np.memmap(self.dir / "data.f64", dtype=np.float64, mode="r")
        self.sha256: str = self.meta["data_sha256"]

    def resolve(self, fam: str) -> int:
        """Accepts a full family id or the 8-hex short form used in reports (`60b608db`)."""
        hits = [i for i, f in enumerate(self.families) if f == fam or f.startswith(fam) or f[4:12] == fam]
        if len(hits) != 1:
            raise KeyError(f"family {fam!r}: {len(hits)} matches")
        return hits[0]

    def short(self, i: int) -> str:
        f = self.families[i]
        return f[4:12] if f.startswith("fam-") else f[:8]

    def table(self, fam: str | int) -> np.ndarray:
        i = fam if isinstance(fam, int) else self.resolve(fam)
        r, c = self.meta["shapes"][i]
        o = self.meta["offsets"][i]
        return self._mm[o : o + r * c].reshape(r, c)

    def member_rows(self, fam: str | int) -> list[int]:
        i = fam if isinstance(fam, int) else self.resolve(fam)
        return list(self.meta["member_rows"][i])

    def table_without_stratum(self, fam: str | int, k_out: int, k: int = 5) -> np.ndarray:
        """The pool without one feature-hash stratum: the training table of a cross-fitted ("skill") specialist."""
        full = np.asarray(self.table(fam), dtype=np.float64)
        return np.ascontiguousarray(full[feature_stratum(full[:, :-1], k) != int(k_out)])

    def table_excluding(self, fam: str | int, mmap_dir: Path, cids: list[str]) -> np.ndarray:
        """The pool without every row of the named member tables (matched by row bytes, so sibling copies go too).

        Used for leave-context-out specialists: weights trained on the result never saw any row of those contexts.
        """
        i = fam if isinstance(fam, int) else self.resolve(fam)
        unknown = [c for c in cids if c not in self.meta["member_cids"][i]]
        if unknown:
            raise KeyError(f"not member tables of {self.families[i]}: {unknown}")
        src = json.loads((Path(mmap_dir) / "meta.json").read_text())
        mm = np.memmap(Path(mmap_dir) / src["data_file"], dtype=np.float64, mode="r")
        drop: set[bytes] = set()
        for t, cid in enumerate(src["cids"]):
            if cid in cids and src["cohorts"][t] == COHORT:
                r, c = (int(v) for v in src["shapes"][t])
                o = int(src["offsets"][t])
                drop.update(row_keys(np.asarray(mm[o : o + r * c], dtype=np.float64).reshape(r, c)))
        full = np.asarray(self.table(i), dtype=np.float64)
        keep = np.fromiter((k not in drop for k in row_keys(full)), dtype=bool, count=full.shape[0])
        return np.ascontiguousarray(full[keep])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=("build", "info"))
    ap.add_argument("--mmap-bank", default=str(P.MMAP_V1))
    ap.add_argument("--out", default=str(P.POOL_DIR))
    a = ap.parse_args()
    if a.cmd == "build":
        meta = build_pool(Path(a.mmap_bank), Path(a.out))
    else:
        meta = FamilyPool(Path(a.out), verify_data=True).meta
    rows = {f[4:12]: s for f, s in zip(meta["families"], meta["shapes"])}
    print(json.dumps({"families": len(rows), "total_rows": meta["total_rows"], "data_sha256": meta["data_sha256"], "shapes": rows}))


if __name__ == "__main__":
    main()
