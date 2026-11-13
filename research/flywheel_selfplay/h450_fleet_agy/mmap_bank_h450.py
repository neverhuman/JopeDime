#!/usr/bin/env python3
"""Zero-Copy `np.memmap` Drop-In Replacement for `CombinedF64Bank` (`mmap_bank_h450.py`).

Provides:
  1. `MmapF64Bank(mmap_dir, expect_source_sha256=...)`: 100% API-compatible drop-in replacement for
     `train_aggressive_random_h450.CombinedF64Bank`. Maps `data.f64` read-only (`mode='r'`), exposing `tabs_f64` as
     zero-copy `np.memmap` views (`< 60 MiB` private RSS across parent + forked workers instead of unpacking 3,525
     arrays into private heap RAM).
  2. `--cmd convert`: Converts `everything_combined_f64_h450_bank.npz` (`9dfc60ed...`) into `mmap_v1/` (`data.f64` +
     `meta.json` + `READY`, all `0444` create-once).
  3. `--cmd verify-parity`: Verifies bit-for-bit IEEE-754 (`uint64` view, exact NaN payload & sign bit) equality across
     all 3,525 tables, metadata arrays, structural strata, and smart-stratified sampling weights against
     `CombinedF64Bank`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import sys
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

SEALED_TOKENS = ("CONFIRM", "HOLDOUT", "SEALED", "DL-VAL", "DLVAL")
EXPECTED_COHORTS = {
    "APEX-DEV-FULL", "APEX-FIT-ICL-FULL", "APEX-FIT-SEL-FULL", "D91-COMBINED-SHUFFLED", "D91-S3-COUSIN",
    "DL-TRAIN-FULL", "S3-DEV-FULL", "S3-FIT-ICL-FULL", "S3-FIT-SEL-FULL",
}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class MmapF64Bank:
    """Zero-copy read-only `np.memmap` float64 table bank, 100% API-compatible with `CombinedF64Bank`."""

    def __init__(
        self,
        mmap_dir: Path,
        expect_sha256: str | None = None,
        verify_data_sha: bool = False,
        sampling_mode: str = "smart_stratified",
    ):
        self.path = Path(mmap_dir)
        self.sampling_mode = str(sampling_mode)
        meta_p = self.path / "meta.json"
        data_p = self.path / "data.f64"
        ready_p = self.path / "READY"
        if not (meta_p.is_file() and data_p.is_file() and ready_p.is_file()):
            raise SystemExit(f"incomplete mmap bank directory: {self.path}")
        meta = json.loads(meta_p.read_text())
        self.source_npz_sha256 = str(meta["source_npz_sha256"])
        self.data_sha256 = str(meta["data_sha256"])
        # Keep `.sha256` equal to `source_npz_sha256` so `ckpt_last.pt` / `epoch_NN.json` checks match seamlessly
        self.sha256 = self.source_npz_sha256
        if expect_sha256 and expect_sha256 not in (self.source_npz_sha256, self.data_sha256):
            raise SystemExit(f"mmap bank sha256 ({self.source_npz_sha256} / {self.data_sha256}) != expected {expect_sha256}")
        if verify_data_sha:
            got_data_sha = sha256_file(data_p)
            if got_data_sha != self.data_sha256:
                raise SystemExit(f"data.f64 sha256 {got_data_sha} != meta {self.data_sha256}")

        n_elem = int(meta["data_elements"])
        if data_p.stat().st_size != n_elem * 8:
            raise SystemExit(f"data.f64 byte size {data_p.stat().st_size} != {n_elem * 8}")

        self._mm = np.memmap(data_p, dtype=np.float64, mode="r", shape=(n_elem,))
        shapes = np.asarray(meta["shapes"], dtype=np.int64)
        offs = np.asarray(meta["offsets"], dtype=np.int64)
        self.cids = [str(x) for x in meta["cids"]]
        self.cohorts = [str(x) for x in meta["cohorts"]]
        self.families = [str(x) for x in meta["families"]]

        names = sorted(set(self.cohorts))
        bad = [c for c in names if any(t in c.upper() for t in SEALED_TOKENS)]
        if bad or set(names) != EXPECTED_COHORTS:
            raise SystemExit(f"REFUSING mmap bank: sealed {bad}, unexpected {sorted(set(names) ^ EXPECTED_COHORTS)}")

        self.tabs_f64: list[np.ndarray] = []
        for i in range(len(shapes)):
            r, w = int(shapes[i, 0]), int(shapes[i, 1])
            o = int(offs[i])
            view = self._mm[o : o + r * w].reshape(r, w)
            view.flags.writeable = False
            self.tabs_f64.append(view)

        self.cohort_names = names
        self.cohort_idx = np.array([names.index(c) for c in self.cohorts], dtype=np.int64)
        fam_names = sorted(set(self.families))
        self.family_idx = np.array([fam_names.index(f) for f in self.families], dtype=np.int64)
        self.n_tables = len(self.tabs_f64)
        self.rows_arr = shapes[:, 0].copy()
        self.cols_arr = (shapes[:, 1] - 1).copy()
        self.strata = [str(x) for x in meta["strata"]] if "strata" in meta else self._compute_structural_strata()
        self.is_d91 = np.array([c.startswith("D91") for c in self.cohorts], dtype=bool)
        self.all_indices = np.arange(self.n_tables, dtype=np.int64)
        if self.sampling_mode in ("uniform", "uniform_dataset"):
            self.all_weights = np.full(self.n_tables, 1.0 / float(self.n_tables), dtype=np.float64)
        elif self.sampling_mode == "smart_stratified":
            self.all_weights = self._smart_stratified_weights(self.all_indices)
        else:
            raise SystemExit(f"unknown MmapF64Bank sampling_mode: {self.sampling_mode}")

    def sample_table_idx(self, rng: np.random.Generator) -> int:
        if self.sampling_mode in ("uniform", "uniform_dataset"):
            return int(rng.integers(0, self.n_tables))
        return int(rng.choice(self.all_indices, p=self.all_weights))

    def sample_eligible(self, rng: np.random.Generator, elig: np.ndarray) -> int | None:
        if self.sampling_mode in ("uniform", "uniform_dataset"):
            idx_ok = np.flatnonzero(elig > 0.0)
            if idx_ok.size == 0:
                return None
            return int(idx_ok[int(rng.integers(0, idx_ok.size))])
        w = self.all_weights * elig
        s = float(w.sum())
        if s <= 0.0:
            return None
        return int(rng.choice(self.all_indices, p=w / s))

    def _compute_structural_strata(self) -> list[str]:
        strata: list[str] = []
        for i, tab in enumerate(self.tabs_f64):
            r, p = int(self.rows_arr[i]), int(self.cols_arr[i])
            r_b = "S" if r < 250 else ("M" if r < 1200 else "L")
            p_b = "N" if p <= 8 else ("W" if p <= 30 else "XW")
            yf = tab[:, -1][np.isfinite(tab[:, -1])]
            tail_b = "norm"
            if yf.size >= 16:
                q10, q50, q90 = np.percentile(yf, [10.0, 50.0, 90.0])
                if float(np.max(np.abs(yf - q50))) / max(q90 - q10, 1e-12) > 8.0:
                    tail_b = "heavy"
            strata.append(f"{r_b}:{p_b}:{tail_b}")
        return strata

    def _smart_stratified_weights(self, idxs: np.ndarray) -> np.ndarray:
        fam_counts: dict[str, int] = {}
        strat_counts: dict[str, int] = {}
        for i in idxs.tolist():
            fam_counts[self.families[i]] = fam_counts.get(self.families[i], 0) + 1
            strat_counts[self.strata[i]] = strat_counts.get(self.strata[i], 0) + 1
        mean_sc = float(len(idxs)) / max(float(len(strat_counts)), 1.0)
        w = np.array(
            [
                ((min(self.rows_arr[i], 2048) ** 0.30) / math.sqrt(float(fam_counts[self.families[i]])))
                * ((mean_sc / float(strat_counts[self.strata[i]])) ** 0.25)
                for i in idxs.tolist()
            ],
            dtype=np.float64,
        )
        return w / float(np.sum(w))


def convert_npz_to_mmap(npz_path: Path, expect_sha256: str, out_dir: Path) -> dict:
    if (out_dir / "READY").is_file():
        return json.loads((out_dir / "meta.json").read_text())
    t0 = time.time()
    src_sha = sha256_file(npz_path)
    if expect_sha256 and src_sha != expect_sha256:
        raise SystemExit(f"npz sha256 {src_sha} != expected {expect_sha256}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with np.load(npz_path, allow_pickle=False) as z:
        data = np.ascontiguousarray(z["data"], dtype=np.float64)
        shapes = z["shape"].astype(np.int64)
        offs = z["offset"].astype(np.int64)
        cids = [str(x) for x in z["cids"].tolist()]
        cohorts = [str(x) for x in z["cohort"].tolist()]
        families = [str(x) for x in z["family"].tolist()]

    names = sorted(set(cohorts))
    bad = [c for c in names if any(t in c.upper() for t in SEALED_TOKENS)]
    if bad or set(names) != EXPECTED_COHORTS:
        raise SystemExit(f"REFUSING source bank: sealed {bad}, unexpected {sorted(set(names) ^ EXPECTED_COHORTS)}")

    # Precompute exact structural strata once so MmapF64Bank.__init__ finishes in < 25 ms without paging all 1.9 GB!
    strata: list[str] = []
    for i in range(len(shapes)):
        r, w = int(shapes[i, 0]), int(shapes[i, 1])
        p = w - 1
        o = int(offs[i])
        tab = data[o : o + r * w].reshape(r, w)
        r_b = "S" if r < 250 else ("M" if r < 1200 else "L")
        p_b = "N" if p <= 8 else ("W" if p <= 30 else "XW")
        yf = tab[:, -1][np.isfinite(tab[:, -1])]
        tail_b = "norm"
        if yf.size >= 16:
            q10, q50, q90 = np.percentile(yf, [10.0, 50.0, 90.0])
            if float(np.max(np.abs(yf - q50))) / max(q90 - q10, 1e-12) > 8.0:
                tail_b = "heavy"
        strata.append(f"{r_b}:{p_b}:{tail_b}")

    data_p = out_dir / "data.f64"
    tmp_p = out_dir / f".data.f64.{os.getpid()}.tmp"
    with open(tmp_p, "wb") as f:
        data.tofile(f)
    os.replace(tmp_p, data_p)
    os.chmod(data_p, 0o444)
    data_sha = sha256_file(data_p)

    meta = {
        "schema": "jope.h450.mmap_f64_bank/v1",
        "dtype": "float64",
        "source_npz": str(npz_path),
        "source_npz_sha256": src_sha,
        "data_file": "data.f64",
        "data_bytes": int(data_p.stat().st_size),
        "data_elements": int(data.size),
        "data_sha256": data_sha,
        "n_tables": len(cids),
        "total_rows": int(shapes[:, 0].sum()),
        "total_cells": int((shapes[:, 0] * shapes[:, 1]).sum()),
        "cohort_counts": dict(Counter(cohorts)),
        "confirm_table_opens": 0,
        "dl_holdout_opens": 0,
        "sealed_cohorts_in_bank": 0,
        "convert_wall_s": round(time.time() - t0, 2),
        "shapes": shapes.tolist(),
        "offsets": offs.tolist(),
        "cids": cids,
        "cohorts": cohorts,
        "families": families,
        "strata": strata,
    }
    meta_p = out_dir / "meta.json"
    meta_p.write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
    os.chmod(meta_p, 0o444)
    ready_p = out_dir / "READY"
    ready_p.write_text(data_sha + "\n")
    os.chmod(ready_p, 0o444)
    return meta


def verify_parity(npz_path: Path, expect_sha256: str, mmap_dir: Path, ref_code_dir: Path) -> dict:
    sys.path.insert(0, str(ref_code_dir))
    import train_aggressive_random_h450 as T  # noqa: E402

    t0 = time.time()
    b_mm = MmapF64Bank(mmap_dir, expect_sha256=expect_sha256, verify_data_sha=True)
    t_mm_init_ms = round((time.time() - t0) * 1000.0, 2)
    rss_after_mm_mib = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 2)

    t1 = time.time()
    b_np = T.CombinedF64Bank(npz_path, expect_sha256=expect_sha256)
    t_np_init_s = round(time.time() - t1, 2)
    rss_after_np_mib = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 2)

    assert b_mm.n_tables == b_np.n_tables == 3525, f"n_tables {b_mm.n_tables} vs {b_np.n_tables}"
    assert b_mm.sha256 == b_np.sha256 == expect_sha256
    assert b_mm.cids == b_np.cids, "cids mismatch"
    assert b_mm.cohorts == b_np.cohorts, "cohorts mismatch"
    assert b_mm.families == b_np.families, "families mismatch"
    assert b_mm.strata == b_np.strata, "strata mismatch"
    assert np.array_equal(b_mm.rows_arr, b_np.rows_arr), "rows_arr mismatch"
    assert np.array_equal(b_mm.cols_arr, b_np.cols_arr), "cols_arr mismatch"
    assert np.array_equal(b_mm.cohort_idx, b_np.cohort_idx), "cohort_idx mismatch"
    assert np.array_equal(b_mm.family_idx, b_np.family_idx), "family_idx mismatch"
    assert np.array_equal(b_mm.is_d91, b_np.is_d91), "is_d91 mismatch"
    max_w_diff = float(np.max(np.abs(b_mm.all_weights - b_np.all_weights)))
    assert max_w_diff == 0.0, f"all_weights diff {max_w_diff}"

    # Bit-for-bit IEEE-754 uint64 comparison across all 3,525 tables (catches sign of 0.0 and NaN payload differences!)
    total_cells_checked = 0
    for i in range(b_np.n_tables):
        a_mm = np.ascontiguousarray(b_mm.tabs_f64[i]).view(np.uint64)
        a_np = np.ascontiguousarray(b_np.tabs_f64[i]).view(np.uint64)
        if a_mm.shape != a_np.shape or not np.array_equal(a_mm, a_np):
            raise AssertionError(f"bit mismatch at table {i} ({b_np.cids[i]})")
        total_cells_checked += int(a_np.size)

    # Also verify identical episode draws under same RNG seed!
    for seed in (101, 202, 303, 404, 505):
        r1 = np.random.default_rng(seed)
        r2 = np.random.default_rng(seed)
        ep1 = T.real_episode(b_mm, b_mm.sample_table_idx(r1), r1)
        ep2 = T.real_episode(b_np, b_np.sample_table_idx(r2), r2)
        assert (ep1 is None) == (ep2 is None)
        if ep1 is not None:
            assert ep1[2] == ep2[2] and ep1[3] == ep2[3]
            assert np.array_equal(np.ascontiguousarray(ep1[0]).view(np.uint64), np.ascontiguousarray(ep2[0]).view(np.uint64))
            assert np.array_equal(np.ascontiguousarray(ep1[1]).view(np.uint64), np.ascontiguousarray(ep2[1]).view(np.uint64))

    return {
        "parity_status": "PASS_BIT_IDENTICAL",
        "n_tables": b_mm.n_tables,
        "total_cells_uint64_verified": total_cells_checked,
        "source_npz_sha256": b_mm.source_npz_sha256,
        "mmap_data_sha256": b_mm.data_sha256,
        "meta_sha256": sha256_file(mmap_dir / "meta.json"),
        "max_weight_abs_diff": max_w_diff,
        "mmap_init_ms": t_mm_init_ms,
        "npz_init_s": t_np_init_s,
        "rss_after_mmap_init_mib": rss_after_mm_mib,
        "rss_after_npz_init_mib": rss_after_np_mib,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cmd", choices=("convert", "verify-parity"), required=True)
    ap.add_argument("--npz", default="/home/ubuntu/jope-data/h450-quarantine/banks/everything_combined_f64_h450_bank.npz")
    ap.add_argument("--expect-sha256", default="9dfc60ed29a72266bf69228addcfce5e946064905edde79115914dee6eb0c507")
    ap.add_argument("--mmap-dir", default="/home/ubuntu/jope-data/h450-fleet-agy/banks/mmap_v1")
    ap.add_argument("--ref-code-dir", default="/home/ubuntu/h450-stage/pub-r5/code")
    ap.add_argument("--receipt-out", default="")
    a = ap.parse_args()

    if a.cmd == "convert":
        meta = convert_npz_to_mmap(Path(a.npz), a.expect_sha256, Path(a.mmap_dir))
        out = {k: meta[k] for k in ("schema", "n_tables", "total_rows", "total_cells", "data_bytes",
                                    "source_npz_sha256", "data_sha256", "convert_wall_s", "cohort_counts")}
    else:
        out = verify_parity(Path(a.npz), a.expect_sha256, Path(a.mmap_dir), Path(a.ref_code_dir))
    if a.receipt_out:
        rp = Path(a.receipt_out)
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
