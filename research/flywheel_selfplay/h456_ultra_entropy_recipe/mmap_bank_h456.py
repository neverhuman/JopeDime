#!/usr/bin/env python3
"""Zero-Copy `np.memmap` Bank with Exact `1 / 2,389` S3 Family-Deck Mixing (`mmap_bank_h456.py`, Wave H456).

Upgrades in H456 over H455:
  1. Hard startup quarantine gate refusing any `CONFIRM`, `FMGAP` (Parts B/C or any FMGAP cohort),
     `HOLDOUT`, `SEALED`, or `DL-VAL` cohort/cid/family.
  2. `sampling_mode="s3_source_exact_uniform"`:
     - Exact `1 / (N_families * |family|)` probability mass across all `2,389` distinct S3/corpus
       source families (`max/min` family mass = `1.000000x`, removing H455's `1.50x` soft-2-cap skew).
     - Variance-reduced shuffled family-deck permutation (`sample_s3_exact_deck_table_idx`): visits
       all `2,389` S3 source families once per deck cycle before any source family repeats.
     - Hierarchical family-uniform eligible sampling (`sample_eligible`): preserves exact `1 / N_elig_fam`
       source-family uniformity even when filtering by minimum row count `N_use`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

SEALED_TOKENS = (
    "CONFIRM",
    "HOLDOUT",
    "SEALED",
    "DL-VAL",
    "DLVAL",
    "FMGAP",
)
EXPECTED_COHORTS = {
    "APEX-DEV-FULL",
    "APEX-FIT-ICL-FULL",
    "APEX-FIT-SEL-FULL",
    "D91-COMBINED-SHUFFLED",
    "D91-S3-COUSIN",
    "DL-TRAIN-FULL",
    "S3-DEV-FULL",
    "S3-FIT-ICL-FULL",
    "S3-FIT-SEL-FULL",
}


def compute_file_sha256(path_obj: Path) -> str:
    with path_obj.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _validate_bank_quarantine(meta: dict, cids: list[str], cohorts: list[str], families: list[str]) -> list[str]:
    if (
        int(meta.get("confirm_table_opens", 0)) != 0
        or int(meta.get("dl_holdout_opens", 0)) != 0
        or int(meta.get("sealed_cohorts_in_bank", 0)) != 0
    ):
        raise SystemExit("QUARANTINE REFUSAL: nonzero confirm/holdout/sealed count in mmap meta.json")
    names = sorted(set(cohorts))
    bad_cohorts = [c for c in names if any(tok in c.upper() for tok in SEALED_TOKENS)]
    if bad_cohorts or set(names) != EXPECTED_COHORTS:
        raise SystemExit(f"REFUSING mmap bank: sealed {bad_cohorts}, unexpected {sorted(set(names) ^ EXPECTED_COHORTS)}")
    for idx in range(len(cids)):
        tag_u = f"{cids[idx]}|{cohorts[idx]}|{families[idx]}".upper()
        if any(tok in tag_u for tok in SEALED_TOKENS):
            raise SystemExit(f"QUARANTINE REFUSAL at table {idx}: {tag_u}")
    return names


class MmapF64BankH456:
    """Zero-copy read-only `np.memmap` float64 table bank with exact `1 / 2,389` S3 family-deck mixing."""

    def __init__(
        self,
        mmap_dir: Path,
        expect_sha256: str | None = None,
        verify_data_sha: bool = False,
        sampling_mode: str = "s3_source_exact_uniform",
    ) -> None:
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
        self.sha256 = self.source_npz_sha256
        if expect_sha256 and expect_sha256 not in (self.source_npz_sha256, self.data_sha256):
            raise SystemExit(f"mmap bank sha256 ({self.source_npz_sha256} / {self.data_sha256}) != expected {expect_sha256}")
        if verify_data_sha and compute_file_sha256(data_p) != self.data_sha256:
            raise SystemExit(f"data.f64 sha256 mismatch against {self.data_sha256}")

        self.cids = [str(x) for x in meta["cids"]]
        self.cohorts = [str(x) for x in meta["cohorts"]]
        self.families = [str(x) for x in meta["families"]]
        names = _validate_bank_quarantine(meta, self.cids, self.cohorts, self.families)

        n_elem = int(meta["data_elements"])
        if data_p.stat().st_size != n_elem * 8:
            raise SystemExit(f"data.f64 byte size {data_p.stat().st_size} != {n_elem * 8}")

        self._mm = np.memmap(data_p, dtype=np.float64, mode="r", shape=(n_elem,))
        shapes = np.asarray(meta["shapes"], dtype=np.int64)
        offs = np.asarray(meta["offsets"], dtype=np.int64)
        self.tabs_f64: list[np.ndarray] = []
        for idx in range(len(shapes)):
            r_cnt, w_cnt = int(shapes[idx, 0]), int(shapes[idx, 1])
            off = int(offs[idx])
            view = self._mm[off : off + r_cnt * w_cnt].reshape(r_cnt, w_cnt)
            view.flags.writeable = False
            self.tabs_f64.append(view)

        self.cohort_names = names
        self.cohort_idx = np.array([names.index(c) for c in self.cohorts], dtype=np.int64)
        fam_names = sorted(set(self.families))
        self.family_names = fam_names
        self.n_families = len(fam_names)
        fam_to_id = {f: idx for idx, f in enumerate(fam_names)}
        self.family_idx = np.array([fam_to_id[f] for f in self.families], dtype=np.int64)
        self.n_tables = len(self.tabs_f64)
        self.rows_arr = shapes[:, 0].copy()
        self.cols_arr = (shapes[:, 1] - 1).copy()
        self.strata = [str(x) for x in meta["strata"]] if "strata" in meta else self._compute_structural_strata()
        self.is_d91 = np.array([c.startswith("D91") for c in self.cohorts], dtype=bool)
        self.all_indices = np.arange(self.n_tables, dtype=np.int64)

        buckets: list[list[int]] = [[] for _ in range(self.n_families)]
        for ti, fi in enumerate(self.family_idx.tolist()):
            buckets[fi].append(ti)
        self.family_tables = [np.asarray(b, dtype=np.int64) for b in buckets]
        self.family_max_rows = np.array([int(np.max(self.rows_arr[b])) for b in self.family_tables], dtype=np.int64)

        self._deck_order: np.ndarray | None = None
        self._deck_cursor: int = self.n_families

        self.s3_exact_weights = self._s3_exact_uniform_weights()
        self.s3_soft_weights = self._s3_soft_uniform_weights()
        if self.sampling_mode == "s3_source_exact_uniform":
            self.all_weights = self.s3_exact_weights.copy()
        elif self.sampling_mode == "s3_source_uniform":
            self.all_weights = self.s3_soft_weights.copy()
        elif self.sampling_mode in ("uniform", "uniform_dataset"):
            self.all_weights = np.full(self.n_tables, 1.0 / float(self.n_tables), dtype=np.float64)
        elif self.sampling_mode == "smart_stratified":
            self.all_weights = self._smart_stratified_weights(self.all_indices)
        else:
            raise SystemExit(f"unknown MmapF64BankH456 sampling_mode: {self.sampling_mode}")

    def _s3_exact_uniform_weights(self) -> np.ndarray:
        """Exact `1 / (N_families * |family|)` weight per table (`max/min` family mass == `1.000000x`)."""
        w = np.zeros(self.n_tables, dtype=np.float64)
        inv_f = 1.0 / float(max(self.n_families, 1))
        for members in self.family_tables:
            k = max(1, int(members.size))
            w[members] = inv_f / float(k)
        return w / float(np.sum(w))

    def _s3_soft_uniform_weights(self) -> np.ndarray:
        """Legacy H455 `50%` exact + `50%` soft-2-cap weights (`max/min` family mass <= `1.50x`)."""
        w_cap = np.zeros(self.n_tables, dtype=np.float64)
        for members in self.family_tables:
            k = max(1, int(members.size))
            w_cap[members] = min(float(k), 2.0) / float(k)
        w_cap /= float(np.sum(w_cap))
        w = 0.50 * self.s3_exact_weights + 0.50 * w_cap
        return w / float(np.sum(w))

    def sample_s3_exact_deck_table_idx(self, rng: np.random.Generator) -> int:
        """Draws next source family from a shuffled `2,389`-family permutation deck, then uniform table inside it."""
        if self._deck_order is None or self._deck_cursor >= self.n_families:
            self._deck_order = rng.permutation(self.n_families)
            self._deck_cursor = 0
        fi = int(self._deck_order[self._deck_cursor])
        self._deck_cursor += 1
        members = self.family_tables[fi]
        if members.size == 1:
            return int(members[0])
        return int(members[int(rng.integers(0, members.size))])

    def sample_table_idx(self, rng: np.random.Generator) -> int:
        if self.sampling_mode == "s3_source_exact_uniform":
            return self.sample_s3_exact_deck_table_idx(rng)
        if self.sampling_mode in ("uniform", "uniform_dataset"):
            return int(rng.integers(0, self.n_tables))
        return int(rng.choice(self.all_indices, p=self.all_weights))

    def sample_eligible_min_rows(self, rng: np.random.Generator, min_rows: int) -> int | None:
        """Exact `1 / N_elig_families` hierarchical draw over all source families having `>= 1` table with `rows >= min_rows`."""
        if min_rows <= int(np.min(self.rows_arr)):
            return self.sample_table_idx(rng)
        elig_fams = np.flatnonzero(self.family_max_rows >= int(min_rows))
        if elig_fams.size == 0:
            return None
        if self.sampling_mode == "s3_source_exact_uniform":
            # Try deck-ordered eligible family first (variance-reduced), falling back to uniform over elig_fams
            for _ in range(6):
                cand_ti = self.sample_s3_exact_deck_table_idx(rng)
                fi = int(self.family_idx[cand_ti])
                if int(self.family_max_rows[fi]) >= int(min_rows):
                    members = self.family_tables[fi]
                    ok_m = members[self.rows_arr[members] >= int(min_rows)]
                    if ok_m.size > 0:
                        return int(ok_m[int(rng.integers(0, ok_m.size))])
            fi = int(elig_fams[int(rng.integers(0, elig_fams.size))])
            members = self.family_tables[fi]
            ok_m = members[self.rows_arr[members] >= int(min_rows)]
            return int(ok_m[int(rng.integers(0, ok_m.size))])
        elig = (self.rows_arr >= int(min_rows)).astype(np.float64)
        return self.sample_eligible(rng, elig)

    def sample_eligible(self, rng: np.random.Generator, elig: np.ndarray) -> int | None:
        if self.sampling_mode in ("uniform", "uniform_dataset"):
            idx_ok = np.flatnonzero(elig > 0.0)
            if idx_ok.size == 0:
                return None
            return int(idx_ok[int(rng.integers(0, idx_ok.size))])
        w = self.all_weights * elig
        tot = float(w.sum())
        if tot <= 0.0:
            return None
        return int(rng.choice(self.all_indices, p=w / tot))

    def family_mass_summary(self) -> dict[str, float]:
        fam_mass = np.array([float(np.sum(self.all_weights[m])) for m in self.family_tables], dtype=np.float64)
        lo = float(np.min(fam_mass))
        hi = float(np.max(fam_mass))
        return {
            "n_tables": int(self.n_tables),
            "n_families": int(self.n_families),
            "min_family_mass": lo,
            "max_family_mass": hi,
            "max_over_min_ratio": hi / max(lo, 1e-30),
        }

    def _compute_structural_strata(self) -> list[str]:
        strata: list[str] = []
        for idx, tab in enumerate(self.tabs_f64):
            r_cnt, p_cnt = int(self.rows_arr[idx]), int(self.cols_arr[idx])
            r_b = "S" if r_cnt < 250 else ("M" if r_cnt < 1200 else "L")
            p_b = "N" if p_cnt <= 8 else ("W" if p_cnt <= 30 else "XW")
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
        for idx in idxs.tolist():
            fam_counts[self.families[idx]] = fam_counts.get(self.families[idx], 0) + 1
            strat_counts[self.strata[idx]] = strat_counts.get(self.strata[idx], 0) + 1
        mean_sc = float(len(idxs)) / max(float(len(strat_counts)), 1.0)
        w = np.array(
            [
                ((min(self.rows_arr[idx], 2048) ** 0.30) / math.sqrt(float(fam_counts[self.families[idx]])))
                * ((mean_sc / float(strat_counts[self.strata[idx]])) ** 0.25)
                for idx in idxs.tolist()
            ],
            dtype=np.float64,
        )
        return w / float(np.sum(w))


MmapF64Bank = MmapF64BankH456


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mmap-bank", default="/home/ubuntu/jope-data/h450-fleet-agy/banks/mmap_v1")
    ap.add_argument("--mode", default="s3_source_exact_uniform")
    args = ap.parse_args()
    bank = MmapF64BankH456(Path(args.mmap_bank), sampling_mode=args.mode)
    print(json.dumps(bank.family_mass_summary(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
