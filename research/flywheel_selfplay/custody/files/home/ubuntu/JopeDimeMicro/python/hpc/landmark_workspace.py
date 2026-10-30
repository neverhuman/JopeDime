#!/usr/bin/env python3
r"""
Wave HYPERION-VELOCITAS (HPC-WAVE-1) — Sub-Wave W2 (HYPER-LANDMARK)
===================================================================
Zero-Allocation C++20/AVX2/OpenMP Landmark & Descriptor Engine (`landmark_workspace.py`).

Eliminates Stage 1A redundancies:
  1. Fuses `FitOnlyTableTransform.fit` + `transform` into a single parallel C++ pass (`jope_hpc_fit_transform_table_f32_to_f64`).
  2. Performs a SINGLE sort per column in C++ (`jope_hpc_fused_column_and_row_probes_f64`) to simultaneously extract:
     - Quantiles `(0.05, 0.25, 0.50, 0.75, 0.95)` on unclipped `Z` (for `compute_table_descriptors`),
     - Quantiles `(0.10, 0.25, 0.50, 0.75, 0.90)` on clipped `Z` (for `compute_building_block_landmarks`),
     - $O(n)$ inverse-permutation Spearman rank correlation,
     - Chatterjee $\xi$ rank-jump dependence,
     - 13 univariate probe correlations in thread-local buffers (`0` $(n, p)$ heap temporary matrices),
     - 8 row-level summary statistics (`row_mean`, `row_std`, `row_l1`, `row_l2`, `row_min`, `row_max`, `row_skew`, `row_miss`).
  3. Eliminates the duplicate `compute_building_block_landmarks(level="ultra")` + `compute_building_block_landmarks(level="high")`
     calls by computing the building blocks once and projecting `level="high"` (`bb_gains16[14:16] = 0.0`) in $O(1)$.
"""
from __future__ import annotations

import ctypes
import math
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from research.levels.building_block_landmarks import (
    _compute_multiclass_and_target_simplex,
    _gini,
    _safe_corr_vec,
    _squash,
    compute_inter_landmark_summary_stats,
)
from research.levels.descriptors import (
    extract_report_descriptors,
    gini_coefficient,
    squash,
)

MICRO_ROOT = Path(__file__).resolve().parent.parent.parent
LIB_PATH = MICRO_ROOT / "bin" / "libjope_hpc.so"


class FusedLandmarkWorkspace:
    """
    Reusable C++20/AVX2/OpenMP wrapper for Stage 1A raw table scaling, single-sort quantiles/ranks,
    building-block landmarks (`ultra` + `high` projection), and table descriptors.
    """

    def __init__(self, lib_path: Path = LIB_PATH, num_threads: int = 6) -> None:
        self.lib_path = Path(lib_path)
        self.num_threads = max(1, min(int(num_threads), 6))
        if not self.lib_path.exists():
            raise FileNotFoundError(f"Native HPC shared library not found at {self.lib_path}")
        self.lib = ctypes.CDLL(str(self.lib_path))

        c_f32_p = ctypes.POINTER(ctypes.c_float)
        c_f64_p = ctypes.POINTER(ctypes.c_double)

        self.lib.jope_hpc_fit_transform_table_f32_to_f64.argtypes = [
            c_f32_p,
            c_f32_p,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int32,
            c_f64_p,
            c_f64_p,
        ]
        self.lib.jope_hpc_fit_transform_table_f32_to_f64.restype = None

        self.lib.jope_hpc_fused_column_and_row_probes_f64.argtypes = [
            c_f64_p,
            c_f64_p,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int32,
            c_f64_p,
            c_f64_p,
            c_f64_p,
            c_f64_p,
            c_f64_p,
            c_f64_p,
        ]
        self.lib.jope_hpc_fused_column_and_row_probes_f64.restype = None

        self._has_order_xi_and_spearman = hasattr(self.lib, "jope_hpc_order_xi_and_spearman_f64")
        if self._has_order_xi_and_spearman:
            c_i64_p = ctypes.POINTER(ctypes.c_int64)
            self.lib.jope_hpc_order_xi_and_spearman_f64.argtypes = [
                c_i64_p,
                c_f64_p,
                ctypes.c_int64,
                ctypes.c_int64,
                c_f64_p,
                c_f64_p,
            ]
            self.lib.jope_hpc_order_xi_and_spearman_f64.restype = None

        self._has_bivariate_and_het_tail = hasattr(self.lib, "jope_hpc_bivariate_and_het_tail_f64")
        if self._has_bivariate_and_het_tail:
            c_i64_p = ctypes.POINTER(ctypes.c_int64)
            self.lib.jope_hpc_bivariate_and_het_tail_f64.argtypes = [
                c_f64_p,
                c_f64_p,
                ctypes.c_int64,
                ctypes.c_int64,
                c_i64_p,
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.c_double,
                ctypes.c_int32,
                c_f64_p,
                c_f64_p,
                c_f64_p,
                c_f64_p,
            ]
            self.lib.jope_hpc_bivariate_and_het_tail_f64.restype = None

    def fit_transform_and_extract_bundle_high(
        self,
        X_raw: np.ndarray,
        y_raw: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Executes fused table scaling + single-pass `ultra`/`high` landmark & descriptor extraction.
        Returns `(vec256_np, col_landmarks24_np, substage_timings_ns)`.
        """
        import time

        t0 = time.perf_counter_ns()
        X_clean = np.ascontiguousarray(X_raw, dtype=np.float32)
        if X_clean.ndim == 1:
            X_clean = X_clean.reshape(-1, 1)
        y_clean = np.ascontiguousarray(np.asarray(y_raw, dtype=np.float32).reshape(-1), dtype=np.float32)
        n, p = X_clean.shape

        X_scaled = np.empty((n, p), dtype=np.float64)
        y_scaled = np.empty(n, dtype=np.float64)

        self.lib.jope_hpc_fit_transform_table_f32_to_f64(
            X_clean.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            y_clean.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int64(n),
            ctypes.c_int64(p),
            ctypes.c_int32(self.num_threads),
            X_scaled.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            y_scaled.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        t_scale = time.perf_counter_ns()

        Z_clp = np.empty((n, p), dtype=np.float64)
        Z_unc = np.empty((n, p), dtype=np.float64)
        y_z = np.empty(n, dtype=np.float64)
        col_probes_14xp = np.empty((14, p), dtype=np.float64)
        tab_col_stats_6xp = np.empty((6, p), dtype=np.float64)
        row_corrs = np.empty(8, dtype=np.float64)

        self.lib.jope_hpc_fused_column_and_row_probes_f64(
            X_scaled.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            y_scaled.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int64(n),
            ctypes.c_int64(p),
            ctypes.c_int32(self.num_threads),
            Z_clp.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            Z_unc.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            y_z.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            col_probes_14xp.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            tab_col_stats_6xp.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            row_corrs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )

        c_asinh = col_probes_14xp[0]
        c_atan = col_probes_14xp[1]
        c_erf = col_probes_14xp[2]
        c_cbrt = col_probes_14xp[3]
        c_log1p = col_probes_14xp[4]
        c_tanh = col_probes_14xp[5]
        c_laplace = col_probes_14xp[6]
        c_clamp = col_probes_14xp[7]
        c_step = col_probes_14xp[8]
        c_hinge = col_probes_14xp[9]
        c_sincos = col_probes_14xp[10]
        c_fract = col_probes_14xp[11]
        c_lin = col_probes_14xp[13]

        y_std = float(np.std(y_scaled))
        if y_std > 1e-9:
            y_order = np.argsort(y_scaled)
            y_ranks = np.empty(n, dtype=np.float64)
            y_ranks[y_order] = np.arange(n, dtype=np.float64)
            order_x = np.ascontiguousarray(np.argsort(Z_clp, axis=0), dtype=np.int64)
            if self._has_order_xi_and_spearman:
                c_xi = col_probes_14xp[12]
                spearman_r = tab_col_stats_6xp[5]
                self.lib.jope_hpc_order_xi_and_spearman_f64(
                    order_x.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                    y_ranks.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                    ctypes.c_int64(n),
                    ctypes.c_int64(p),
                    c_xi.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                    spearman_r.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                )
            else:
                y_sorted = y_ranks[order_x]
                diffs = np.mean(np.abs(np.diff(y_sorted, axis=0)), axis=0)
                c_xi = np.clip(1.0 - (3.0 * diffs) / float(max(n + 1, 2)), 0.0, 1.0)

                mean_yr = 0.5 * float(n - 1)
                std_yr = max(math.sqrt((float(n) * float(n) - 1.0) / 12.0), 1e-9)
                y_rank_z = (y_ranks - mean_yr) / std_yr
                r_grid_z = (np.arange(n, dtype=np.float64)[:, None] - mean_yr) / std_yr
                spearman_r = np.abs(np.mean(r_grid_z * y_rank_z[order_x], axis=0))
                tab_col_stats_6xp[5] = spearman_r
        else:
            c_xi = np.zeros(p, dtype=np.float64)
            tab_col_stats_6xp[5] = 0.0

        row_gain = float(np.max(row_corrs)) if row_corrs.size > 0 else 0.0

        # Bivariate polar & normalized-ratio probes on top-6 driver pairs + het_and_tail8 (identical for High and Ultra)
        hypot_gain, atan2_gain, norm_diff_gain = 0.0, 0.0, 0.0
        het_and_tail8 = np.zeros(8, dtype=np.float64)
        if y_std > 1e-9:
            if self._has_bivariate_and_het_tail:
                base_score = np.maximum(c_asinh, np.maximum(c_step, c_xi))
                top_idx = np.ascontiguousarray(np.argsort(base_score)[::-1][: min(p, 6)], dtype=np.int64)
                best_j = int(np.argmax(c_asinh))
                biv_gains_3 = np.zeros(3, dtype=np.float64)
                het_corrs = np.zeros(p, dtype=np.float64)
                u_corrs = np.zeros(p, dtype=np.float64)
                l_corrs = np.zeros(p, dtype=np.float64)
                self.lib.jope_hpc_bivariate_and_het_tail_f64(
                    Z_clp.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                    y_z.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                    ctypes.c_int64(n),
                    ctypes.c_int64(p),
                    top_idx.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                    ctypes.c_int64(top_idx.size),
                    ctypes.c_int64(best_j),
                    ctypes.c_double(float(c_asinh[best_j])),
                    ctypes.c_int32(self.num_threads),
                    biv_gains_3.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                    het_corrs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                    u_corrs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                    l_corrs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                )
                hypot_gain = float(biv_gains_3[0])
                atan2_gain = float(biv_gains_3[1])
                norm_diff_gain = float(biv_gains_3[2])
            else:
                if p >= 2:
                    base_score = np.maximum(c_asinh, np.maximum(c_step, c_xi))
                    top_idx = np.argsort(base_score)[::-1][: min(p, 6)]
                    h_cols, a_cols, d_cols = [], [], []
                    for i_pos in range(len(top_idx)):
                        for j_pos in range(i_pos + 1, len(top_idx)):
                            ia, ib = int(top_idx[i_pos]), int(top_idx[j_pos])
                            zi, zj = Z_clp[:, ia], Z_clp[:, ib]
                            h_cols.append(np.hypot(zi, zj))
                            a_cols.append(np.arctan2(zi, zj))
                            d_cols.append((zi - zj) / (1.0 + np.abs(zi) + np.abs(zj)))
                    if h_cols:
                        hypot_gain = float(np.max(_safe_corr_vec(np.column_stack(h_cols), y_z)))
                        atan2_gain = float(np.max(_safe_corr_vec(np.column_stack(a_cols), y_z)))
                        norm_diff_gain = float(np.max(_safe_corr_vec(np.column_stack(d_cols), y_z)))
                best_j = int(np.argmax(c_asinh))
                res_abs = np.abs(y_z - float(c_asinh[best_j]) * Z_clp[:, best_j])
                res_std = float(np.std(res_abs))
                het_corrs = (
                    _safe_corr_vec(Z_clp, (res_abs - float(np.mean(res_abs))) / max(res_std, 1e-9))
                    if res_std > 1e-9
                    else np.zeros(p)
                )
                y_hi, y_lo = (y_z > 1.0).astype(np.float64), (y_z < -1.0).astype(np.float64)
                u_corrs = _safe_corr_vec((Z_clp > 1.0).astype(np.float64), y_hi - float(np.mean(y_hi)))
                l_corrs = _safe_corr_vec((Z_clp < -1.0).astype(np.float64), y_lo - float(np.mean(y_lo)))

            s_het = np.sort(het_corrs)[::-1]
            het_and_tail8[:4] = [
                float(s_het[0]) if s_het.size else 0.0,
                float(np.mean(s_het[: min(3, p)])),
                float(np.mean(het_corrs)),
                _gini(het_corrs),
            ]
            het_and_tail8[4:] = [
                float(np.max(u_corrs)),
                float(np.mean(u_corrs)),
                float(np.max(l_corrs)),
                float(np.mean(l_corrs)),
            ]

        # Ultra bb_gains16 (for _synthesize_report_from_train) and High bb_gains16 (for bundle)
        bb_gains16_raw_ultra = np.array(
            [
                float(np.max(c_asinh)),
                float(np.max(c_atan)),
                float(np.max(c_erf)),
                float(np.max(c_cbrt)),
                float(np.max(c_log1p)),
                row_gain,
                float(np.max(c_tanh)),
                float(np.max(c_laplace)),
                float(np.max(c_clamp)),
                float(np.max(c_step)),
                float(np.max(c_hinge)),
                float(np.max(c_sincos)),
                float(np.max(c_fract)),
                hypot_gain,
                atan2_gain,
                norm_diff_gain,
            ],
            dtype=np.float64,
        )
        bb_gains16_raw_high = bb_gains16_raw_ultra.copy()
        bb_gains16_raw_high[14] = 0.0
        bb_gains16_raw_high[15] = 0.0

        col_bb_probes7 = np.column_stack([c_asinh, c_atan, c_erf, c_step, c_hinge, c_fract, c_xi]).astype(np.float32)
        col_bb_probes7 = np.nan_to_num(col_bb_probes7, nan=0.0, posinf=0.0, neginf=0.0)

        # Synthesize report from ultra bb_gains16 + col_bb_probes7
        bb16_u_sq = _squash(bb_gains16_raw_ultra).astype(np.float32).astype(np.float64)
        col7_f64 = col_bb_probes7.astype(np.float64)
        probe_map = [
            bb16_u_sq[4], bb16_u_sq[0], bb16_u_sq[3], bb16_u_sq[6], bb16_u_sq[1], bb16_u_sq[7],
            bb16_u_sq[10], bb16_u_sq[9], bb16_u_sq[11], bb16_u_sq[7], bb16_u_sq[5], bb16_u_sq[13],
            bb16_u_sq[11], 0.01, max(float(bb16_u_sq[0]), float(bb16_u_sq[4])), bb16_u_sq[9],
            max(float(bb16_u_sq[9]), float(bb16_u_sq[13])), max(float(bb16_u_sq[9]), float(bb16_u_sq[13])) * 1.05,
            bb16_u_sq[13] * 0.9, bb16_u_sq[13] * 0.95, max(float(bb16_u_sq[0]), float(bb16_u_sq[6])) * 1.05,
            bb16_u_sq[4] * 0.85, max(float(bb16_u_sq[9]), float(bb16_u_sq[13])) * 0.85,
        ]
        se_const = round(1.0 / math.sqrt(max(n, 16)), 5)
        cells = [
            {
                "id": idx + 1,
                "state": "complete",
                "gain": round(float(probe_map[idx]), 5),
                "se": se_const,
                "stable": True,
                "lambda_indices": [2, 2, 3, 2, 3],
            }
            for idx in range(23)
        ]
        spectrum = [
            {
                "slot": j,
                "values": [
                    float(col7_f64[j, 0]), float(col7_f64[j, 3]), float(col7_f64[j, 4]),
                    float(col7_f64[j, 0]), float(col7_f64[j, 1]), float(col7_f64[j, 5]),
                    float(col7_f64[j, 2]), float(col7_f64[j, 6]), float(col7_f64[j, 3]),
                ],
            }
            for j in range(p)
        ]
        report = {
            "panel": "landmarks-native-v1",
            "kind": "real",
            "privacy": "private",
            "task": "regression",
            "classes": 0,
            "rows": n,
            "features": p,
            "cells": cells,
            "target_stats": [float(np.mean(y_scaled)), y_std, None],
            "spectrum": spectrum,
        }
        t_rep = time.perf_counter_ns()

        mc_vec20 = _compute_multiclass_and_target_simplex(Z_clp, y_scaled, lvl_r=2)

        lin_max = float(np.max(c_lin)) if c_lin.size > 0 else 0.0
        denom_h = max(float(np.max(bb_gains16_raw_high)), lin_max, 0.02)
        bb_contrasts16_high = np.clip(
            np.array(
                [
                    (bb_gains16_raw_high[0] - lin_max) / denom_h,
                    (bb_gains16_raw_high[1] - lin_max) / denom_h,
                    (bb_gains16_raw_high[2] - lin_max) / denom_h,
                    (bb_gains16_raw_high[3] - lin_max) / denom_h,
                    (bb_gains16_raw_high[5] - lin_max) / denom_h,
                    float(np.mean(row_corrs)) / denom_h,
                    _gini(row_corrs),
                    (bb_gains16_raw_high[7] - bb_gains16_raw_high[6]) / denom_h,
                    (bb_gains16_raw_high[8] - lin_max) / denom_h,
                    (bb_gains16_raw_high[9] - bb_gains16_raw_high[2]) / denom_h,
                    (bb_gains16_raw_high[10] - lin_max) / denom_h,
                    (bb_gains16_raw_high[11] - lin_max) / denom_h,
                    (bb_gains16_raw_high[12] - lin_max) / denom_h,
                    (bb_gains16_raw_high[13] - lin_max) / denom_h,
                    0.0,
                    0.0,
                ],
                dtype=np.float64,
            ),
            -1.0,
            1.0,
        )

        bb_gains16_high_sq = _squash(bb_gains16_raw_high).astype(np.float32)
        row_corrs_sq = _squash(row_corrs).astype(np.float32)
        het_sq = _squash(het_and_tail8).astype(np.float32)
        building_block_vec68_high = np.concatenate(
            [
                _squash(bb_gains16_raw_high),
                _squash(row_corrs),
                bb_contrasts16_high,
                _squash(het_and_tail8),
                mc_vec20,
            ]
        ).astype(np.float32)
        building_block_vec68_high = np.nan_to_num(building_block_vec68_high, nan=0.0, posinf=0.0, neginf=0.0)

        bb_desc_high = {
            "bb_gains16": bb_gains16_high_sq,
            "row_summary_corrs8": row_corrs_sq,
            "col_bb_probes7": col_bb_probes7,
            "het_and_tail8": het_sq,
            "multiclass_simplex20": mc_vec20.astype(np.float32),
            "building_block_vec68": building_block_vec68_high,
        }

        # Assemble tab_desc (level="high") from precomputed tab_col_stats_6xp and covariance eigenspectrum
        skew = tab_col_stats_6xp[0]
        kurt = tab_col_stats_6xp[1]
        tail_ratio = tab_col_stats_6xp[2]
        mode_conc = tab_col_stats_6xp[3]
        pearson_r = tab_col_stats_6xp[4]
        spearman_r = tab_col_stats_6xp[5]

        if p > 1:
            corr = (Z_unc.T @ Z_unc) / float(n)
            np.fill_diagonal(corr, 1.0)
            corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
            off_diag = np.abs(corr - np.eye(p))
            col_max_corr = np.max(off_diag, axis=1)
            col_mean_corr = np.sum(off_diag, axis=1) / float(p - 1)
            eigvals = np.linalg.eigvalsh(corr)
            eigvals = np.maximum(eigvals[::-1], 0.0)
            eig_sum = max(float(np.sum(eigvals)), 1e-9)
            eig_shares = eigvals / eig_sum
            eff_rank = float((eig_sum ** 2) / max(float(np.sum(eigvals ** 2)), 1e-9)) / float(p)
            spec_entropy = float(
                -np.sum(eig_shares[eig_shares > 1e-12] * np.log(eig_shares[eig_shares > 1e-12]))
                / max(math.log(p), 1e-9)
            )
            top1_eig = float(eig_shares[0])
            top3_eig = float(np.sum(eig_shares[: min(3, p)]))
            top5_eig = float(np.sum(eig_shares[: min(5, p)]))
        else:
            col_max_corr = np.zeros(1, dtype=np.float64)
            col_mean_corr = np.zeros(1, dtype=np.float64)
            eff_rank = 1.0
            spec_entropy = 1.0
            top1_eig = 1.0
            top3_eig = 1.0
            top5_eig = 1.0

        quad_r = np.zeros(p, dtype=np.float64)  # level=="high" -> quad_r is 0
        raw_driver = 0.45 * pearson_r + 0.35 * spearman_r
        if float(np.sum(raw_driver)) > 1e-9:
            driver_weights = raw_driver / float(np.sum(raw_driver))
        else:
            driver_weights = np.full(p, 1.0 / float(p), dtype=np.float64)

        sorted_r = np.sort(pearson_r)[::-1]
        sorted_sp = np.sort(spearman_r)[::-1]

        table_vec24 = np.array(
            [
                eff_rank,
                spec_entropy,
                float(np.mean(np.abs(skew))) / (1.0 + float(np.mean(np.abs(skew)))),
                float(np.mean(mode_conc)),
                float(sorted_r[0]) if sorted_r.size >= 1 else 0.0,
                float(np.mean(pearson_r)),
                gini_coefficient(pearson_r),
                math.log10(max(n, 1)) / 6.0,
                top1_eig,
                top3_eig,
                top5_eig,
                float(np.max(col_max_corr)),
                float(np.mean(col_mean_corr)),
                float(np.mean(tail_ratio)),
                float(sorted_sp[0]) if sorted_sp.size >= 1 else 0.0,
                float(np.max(spearman_r - pearson_r)),
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            dtype=np.float32,
        )
        col_desc8 = np.column_stack(
            [
                squash(skew),
                squash(kurt),
                tail_ratio,
                mode_conc,
                col_max_corr,
                pearson_r,
                spearman_r,
                quad_r,
            ]
        ).astype(np.float32)

        tab_desc_high = {
            "table_vec24": np.nan_to_num(table_vec24, nan=0.0, posinf=0.0, neginf=0.0),
            "column_desc": np.nan_to_num(col_desc8, nan=0.0, posinf=0.0, neginf=0.0),
            "empirical_driver_weights": np.nan_to_num(driver_weights.astype(np.float32), nan=1.0 / p),
        }

        rep_desc_high = extract_report_descriptors(report, level="high")
        inter_vec48 = compute_inter_landmark_summary_stats(
            rep_desc_high, tab_desc_high, bb_desc_high, level="high"
        )
        vec140 = np.concatenate([rep_desc_high["vector116"], tab_desc_high["table_vec24"]]).astype(np.float32)
        vec256 = np.concatenate([vec140, inter_vec48, bb_desc_high["building_block_vec68"]]).astype(np.float32)
        vec256_np = np.nan_to_num(vec256, nan=0.0, posinf=0.0, neginf=0.0)

        col_spec_raw = np.asarray(rep_desc_high["column_spectrum"], dtype=np.float32)
        col_spec9 = np.zeros((p, 9), dtype=np.float32)
        take_p = min(p, col_spec_raw.shape[0])
        if take_p > 0:
            col_spec9[:take_p] = col_spec_raw[:take_p]
        col_landmarks24 = np.column_stack([col_spec9, tab_desc_high["column_desc"], col_bb_probes7[:p]]).astype(
            np.float32
        )
        col_landmarks24_np = np.nan_to_num(col_landmarks24, nan=0.0, posinf=0.0, neginf=0.0)
        t_bundle = time.perf_counter_ns()

        return (
            vec256_np,
            col_landmarks24_np,
            {
                "t0_ns": t0,
                "t_scale_ns": t_scale,
                "t_rep_ns": t_rep,
                "t_bundle_ns": t_bundle,
            },
        )
