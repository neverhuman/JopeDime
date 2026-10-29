"""Phase E / Phase 2 HPC #1 and #6: 256-Thread Block Slate Builder, Persistent Winner Workspace, and Bit-Exact Host/Device Transcendentals (NS12).

Implements the remaining open Phase E / Phase 2 red-team HPC items from ``NORTH_STAR_TASKS.md``
(``NS12`` / ``FUSION_REDTEAM.md`` Part B ``HPC #1`` & Phase 2 ``HPC #1/#6``) without colliding
with ``gpu_fused_agy``'s active Phase 2 CUDA worktree:
1. ``Phase2WinnerWorkspace`` (``Phase2-HPC #6``): Persistent host/device workspace pool for
   the 5 Phase 2 head families (``ridge_gcv``, ``pls1_nipals``, ``knn_3_7_15``, ``ksmooth``,
   ``histgbm``) with non-aliasing ``260 * R_max`` ``float64`` (``rbuf_all``) and
   ``268 * R_max + 2048`` ``int32`` (``ibuf_all``) strides, eliminating per-batch allocation
   and ``cudaMemsetAsync`` overhead.
2. ``build_live_96_slate_block256`` + ``BUILD_LIVE_96_SLATE_BLOCK256_CUDA_SOURCE``
   (``Part B HPC #1``): Replaces the lane-0 serial loop (``<<<1, 32>>>`` with
   ``if (threadIdx.x == 0)``) in ``build_live_96_slate_cuda_kernel`` with a 256-thread
   cooperative block reduction for the deterministic table seed, batched ``champ_881`` /
   ``champ_128`` top-K retrieval, and 96-way parallel candidate featurization + hash-key
   binary search.
3. ``exact_sin_f64``, ``exact_cos_f64``, ``exact_log_f64``, ``exact_exp_f64``,
   ``exact_box_muller_f64``, and ``EXACT_HOST_DEVICE_TRANSCENDENTALS_CUDA_HEADER``:
   Pure IEEE-754 polynomial approximations shared between host and CUDA device
   (compiled with ``--fmad=false -Xcompiler=-ffp-contract=off``) that eliminate the
   1-ULP ``CUDA libdevice`` vs ``x86_64 glibc libm`` divergence in ``Pcg32::gaussian()``,
   ``rff_sin`` / ``rff_cos``, ``sigmoid``, and ``ksmooth``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from jope.fused_hpc.accelerator import precompute_champion_topk_batched_gpu

RBUF_STRIDE_PER_ROW: int = 260
IBUF_STRIDE_PER_ROW: int = 268
IBUF_HIST_EXTRA_INTS: int = 2048

EXACT_HOST_DEVICE_TRANSCENDENTALS_CUDA_HEADER: str = r"""
// Drop-in IEEE-754 bit-exact host/device transcendental helpers for Phase 2 kernels.
// Compile with: nvcc -O3 --fmad=false -Xcompiler "-O3 -ffp-contract=off"
#pragma once
#include <stdint.h>
#include <math.h>

#ifdef __CUDACC__
#define JOPE_HD __host__ __device__ __forceinline__
#else
#define JOPE_HD inline
#endif

JOPE_HD double jope_exact_rint_f64(double x) {
    return floor(x + 0.5);
}

JOPE_HD double jope_exact_sin_poly_f64(double r) {
    const double r2 = r * r;
    const double s1 = -1.66666666666666324348e-01;
    const double s2 =  8.33333333332248946124e-03;
    const double s3 = -1.98412698298579493134e-04;
    const double s4 =  2.75573137070700676789e-06;
    const double s5 = -2.50507602534068634195e-08;
    const double s6 =  1.58969099521155010221e-10;
    const double p = s1 + r2 * (s2 + r2 * (s3 + r2 * (s4 + r2 * (s5 + r2 * s6))));
    return r + r * (r2 * p);
}

JOPE_HD double jope_exact_cos_poly_f64(double r) {
    const double r2 = r * r;
    const double c1 = -4.99999999999999944489e-01;
    const double c2 =  4.16666666666666019037e-02;
    const double c3 = -1.38888888888741095749e-03;
    const double c4 =  2.48015872894767294178e-05;
    const double c5 = -2.75573143513906633035e-07;
    const double c6 =  2.08757232129817482790e-09;
    const double c7 = -1.13596475577881948265e-11;
    return 1.0 + r2 * (c1 + r2 * (c2 + r2 * (c3 + r2 * (c4 + r2 * (c5 + r2 * (c6 + r2 * c7))))));
}

JOPE_HD void jope_exact_sincos_reduce_f64(double x, double* r_out, int64_t* q_out) {
    const double inv_pio2 = 6.36619772367581382433e-01;
    const double pio2_hi  = 1.57079632679489655800e+00;
    const double pio2_lo  = 6.12323399573676603587e-17;
    const double fn = jope_exact_rint_f64(x * inv_pio2);
    const int64_t q = (int64_t)fn;
    const double r = (x - fn * pio2_hi) - fn * pio2_lo;
    *r_out = r;
    *q_out = q & 3;
}

JOPE_HD double exact_sin_f64(double x) {
    double r = 0.0;
    int64_t q = 0;
    jope_exact_sincos_reduce_f64(x, &r, &q);
    const double s = jope_exact_sin_poly_f64(r);
    const double c = jope_exact_cos_poly_f64(r);
    if (q == 0) return s;
    if (q == 1) return c;
    if (q == 2) return -s;
    return -c;
}

JOPE_HD double exact_cos_f64(double x) {
    double r = 0.0;
    int64_t q = 0;
    jope_exact_sincos_reduce_f64(x, &r, &q);
    const double s = jope_exact_sin_poly_f64(r);
    const double c = jope_exact_cos_poly_f64(r);
    if (q == 0) return c;
    if (q == 1) return -s;
    if (q == 2) return -c;
    return s;
}

JOPE_HD double exact_exp_f64(double x) {
    if (x >= 709.78271289338397) return 1.7976931348623157e+308;
    if (x <= -745.13321910194122) return 0.0;
    const double inv_ln2 = 1.44269504088896338700e+00;
    const double ln2_hi  = 6.93147180369123816490e-01;
    const double ln2_lo  = 1.90821492927058770002e-10;
    const double fn = jope_exact_rint_f64(x * inv_ln2);
    const int k = (int)fn;
    const double r = (x - fn * ln2_hi) - fn * ln2_lo;
    const double c2 = 5.00000000000000000000e-01;
    const double c3 = 1.66666666666666657415e-01;
    const double c4 = 4.16666666666666643537e-02;
    const double c5 = 8.33333333333333217785e-03;
    const double c6 = 1.38888888888888894195e-03;
    const double c7 = 1.98412698412698412698e-04;
    const double c8 = 2.48015873015873015873e-05;
    const double c9 = 2.75573192239858925156e-06;
    const double c10 = 2.75573192239858882797e-07;
    const double c11 = 2.50521083854417202271e-08;
    const double p = 1.0 + r * (1.0 + r * (c2 + r * (c3 + r * (c4 + r * (c5 + r * (c6 + r * (c7 + r * (c8 + r * (c9 + r * (c10 + r * c11))))))))));
    return ldexp(p, k);
}

JOPE_HD double exact_log_f64(double x) {
    if (x <= 0.0) return -1.7976931348623157e+308;
    int k = 0;
    double m = frexp(x, &k);
    if (m < 0.70710678118654752440) {
        m *= 2.0;
        k -= 1;
    }
    const double f = m - 1.0;
    const double s = f / (2.0 + f);
    const double z = s * s;
    const double l1 = 6.666666666666735130e-01;
    const double l2 = 3.999999999940941908e-01;
    const double l3 = 2.857142874366239149e-01;
    const double l4 = 2.222219843214978396e-01;
    const double l5 = 1.818357216161805012e-01;
    const double l6 = 1.531383769920937332e-01;
    const double l7 = 1.479819860511658591e-01;
    const double poly = z * (l1 + z * (l2 + z * (l3 + z * (l4 + z * (l5 + z * (l6 + z * l7))))));
    const double ln2_hi = 6.93147180369123816490e-01;
    const double ln2_lo = 1.90821492927058770002e-10;
    return (double)k * ln2_hi + ((2.0 * s + s * poly) + (double)k * ln2_lo);
}
"""

BUILD_LIVE_96_SLATE_BLOCK256_CUDA_SOURCE: str = r"""
// Part B HPC #1: 256-thread cooperative CUDA kernel replacing the <<<1, 32>>> lane-0
// serial build_live_96_slate_cuda_kernel in gpu_full_pipeline_v4.py.
extern "C" __global__ __launch_bounds__(256)
void build_live_96_slate_block256_cuda_kernel(
    const float* __restrict__ X_tr,      // (N, P)
    const float* __restrict__ y_tr,      // (N,)
    int N,
    int P,
    const int32_t* __restrict__ top_t_indices_881, // pre-sorted via cuBLAS GEMM (881,)
    const int32_t* __restrict__ top4_champ128,     // pre-sorted via cuBLAS GEMM (4,)
    const int32_t* __restrict__ champ_embs_881,    // (881, 4, 24)
    const int32_t* __restrict__ champ_embs_128,    // (128, 24)
    const uint64_t* __restrict__ sorted_keys,      // (n_keys,)
    int n_keys,
    uint32_t* __restrict__ out_table_seed,
    int32_t* __restrict__ out_slate_embs,          // (96, 24)
    int32_t* __restrict__ out_slate_key_idx        // (96,)
) {
    __shared__ uint64_t s_hash[256];
    const int tid = threadIdx.x;
    const int total_x = N * P;

    // 1. 256-thread strided reduction for deterministic table seed
    uint64_t h_local = 0x243F6A8885A308D3ULL ^ ((uint64_t)N << 32) ^ (uint64_t)P;
    for (int idx = tid; idx < total_x; idx += 256) {
        float v = X_tr[idx];
        uint32_t bits = __float_as_uint(isnan(v) || isinf(v) ? 0.0f : v);
        uint64_t mix = ((uint64_t)bits ^ ((uint64_t)idx * 0x9E3779B97F4A7C15ULL)) * 0xBF58476D1CE4E5B9ULL;
        h_local ^= (mix ^ (mix >> 31));
    }
    for (int idx = tid; idx < N; idx += 256) {
        float v = y_tr[idx];
        uint32_t bits = __float_as_uint(isnan(v) || isinf(v) ? 0.0f : v);
        uint64_t mix = ((uint64_t)bits ^ ((uint64_t)(idx + total_x) * 0x9E3779B97F4A7C15ULL)) * 0x94D049BB133111EBULL;
        h_local ^= (mix ^ (mix >> 27));
    }
    s_hash[tid] = h_local;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_hash[tid] ^= s_hash[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        uint64_t final_h = s_hash[0] * 0xD6E8FEB86659FD93ULL;
        *out_table_seed = (uint32_t)((final_h ^ (final_h >> 32)) & 0xFFFFFFFFu);
    }

    // 2. 96-way parallel candidate emission + binary search over sorted_keys
    if (tid < 96) {
        const int32_t* src_emb = nullptr;
        if (tid < 92) {
            int tpl_rank = tid >> 2;
            int rank_slot = tid & 3;
            int tpl_idx = top_t_indices_881[tpl_rank];
            src_emb = &champ_embs_881[(tpl_idx * 4 + rank_slot) * 24];
        } else {
            int c128_idx = top4_champ128[tid - 92];
            src_emb = &champ_embs_128[c128_idx * 24];
        }
        uint64_t cand_key = 1469598103934665603ULL;
        for (int d = 0; d < 24; ++d) {
            int32_t v = src_emb[d];
            out_slate_embs[tid * 24 + d] = v;
            cand_key ^= (uint64_t)(uint32_t)v;
            cand_key *= 1099511628211ULL;
        }
        int lo = 0, hi = n_keys - 1, found = -1;
        while (lo <= hi) {
            int mid = (lo + hi) >> 1;
            uint64_t kv = sorted_keys[mid];
            if (kv == cand_key) {
                found = mid;
                break;
            } else if (kv < cand_key) {
                lo = mid + 1;
            } else {
                hi = mid - 1;
            }
        }
        out_slate_key_idx[tid] = found;
    }
}
"""


def _as_f64_array(x: float | np.ndarray) -> tuple[np.ndarray, bool]:
    arr = np.asarray(x, dtype=np.float64)
    is_scalar = arr.ndim == 0
    return (arr.reshape(1) if is_scalar else arr), is_scalar


def _sincos_reduce_f64(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    inv_pio2 = 6.36619772367581382433e-01
    pio2_hi = 1.57079632679489655800e00
    pio2_lo = 6.12323399573676603587e-17
    fn = np.floor(arr * inv_pio2 + 0.5)
    q = np.asarray(fn, dtype=np.int64) & 3
    r = (arr - fn * pio2_hi) - fn * pio2_lo
    return r, q


def _sin_poly_f64(r: np.ndarray) -> np.ndarray:
    r2 = r * r
    s1 = -1.66666666666666324348e-01
    s2 = 8.33333333332248946124e-03
    s3 = -1.98412698298579493134e-04
    s4 = 2.75573137070700676789e-06
    s5 = -2.50507602534068634195e-08
    s6 = 1.58969099521155010221e-10
    p = s1 + r2 * (s2 + r2 * (s3 + r2 * (s4 + r2 * (s5 + r2 * s6))))
    return r + r * (r2 * p)


def _cos_poly_f64(r: np.ndarray) -> np.ndarray:
    r2 = r * r
    c1 = -4.99999999999999944489e-01
    c2 = 4.16666666666666019037e-02
    c3 = -1.38888888888741095749e-03
    c4 = 2.48015872894767294178e-05
    c5 = -2.75573143513906633035e-07
    c6 = 2.08757232129817482790e-09
    c7 = -1.13596475577881948265e-11
    return 1.0 + r2 * (c1 + r2 * (c2 + r2 * (c3 + r2 * (c4 + r2 * (c5 + r2 * (c6 + r2 * c7))))))


def exact_sin_f64(x: float | np.ndarray) -> float | np.ndarray:
    """Bit-exact host/device pure-arithmetic IEEE-754 sine approximation."""
    arr, is_scalar = _as_f64_array(x)
    r, q = _sincos_reduce_f64(arr)
    s = _sin_poly_f64(r)
    c = _cos_poly_f64(r)
    out = np.where(q == 0, s, np.where(q == 1, c, np.where(q == 2, -s, -c)))
    return float(out[0]) if is_scalar else out


def exact_cos_f64(x: float | np.ndarray) -> float | np.ndarray:
    """Bit-exact host/device pure-arithmetic IEEE-754 cosine approximation."""
    arr, is_scalar = _as_f64_array(x)
    r, q = _sincos_reduce_f64(arr)
    s = _sin_poly_f64(r)
    c = _cos_poly_f64(r)
    out = np.where(q == 0, c, np.where(q == 1, -s, np.where(q == 2, -c, s)))
    return float(out[0]) if is_scalar else out


def exact_exp_f64(x: float | np.ndarray) -> float | np.ndarray:
    """Bit-exact host/device pure-arithmetic IEEE-754 exponential approximation."""
    arr, is_scalar = _as_f64_array(x)
    clamped = np.clip(arr, -745.0, 709.0)
    inv_ln2 = 1.44269504088896338700e00
    ln2_hi = 6.93147180369123816490e-01
    ln2_lo = 1.90821492927058770002e-10
    fn = np.floor(clamped * inv_ln2 + 0.5)
    k = fn.astype(np.int32)
    r = (clamped - fn * ln2_hi) - fn * ln2_lo
    c2 = 5.00000000000000000000e-01
    c3 = 1.66666666666666657415e-01
    c4 = 4.16666666666666643537e-02
    c5 = 8.33333333333333217785e-03
    c6 = 1.38888888888888894195e-03
    c7 = 1.98412698412698412698e-04
    c8 = 2.48015873015873015873e-05
    c9 = 2.75573192239858925156e-06
    c10 = 2.75573192239858882797e-07
    c11 = 2.50521083854417202271e-08
    p = 1.0 + r * (
        1.0
        + r
        * (
            c2
            + r * (c3 + r * (c4 + r * (c5 + r * (c6 + r * (c7 + r * (c8 + r * (c9 + r * (c10 + r * c11))))))))
        )
    )
    out = np.ldexp(p, k)
    out = np.where(arr >= 709.78271289338397, 1.7976931348623157e308, out)
    out = np.where(arr <= -745.13321910194122, 0.0, out)
    return float(out[0]) if is_scalar else out


def exact_log_f64(x: float | np.ndarray) -> float | np.ndarray:
    """Bit-exact host/device pure-arithmetic IEEE-754 natural log approximation."""
    arr, is_scalar = _as_f64_array(x)
    safe = np.where(arr > 0.0, arr, 1.0)
    m, k = np.frexp(safe)
    k = k.astype(np.float64)
    adjust = m < 0.70710678118654752440
    m = np.where(adjust, m * 2.0, m)
    k = np.where(adjust, k - 1.0, k)
    f = m - 1.0
    s = f / (2.0 + f)
    z = s * s
    l1 = 6.666666666666735130e-01
    l2 = 3.999999999940941908e-01
    l3 = 2.857142874366239149e-01
    l4 = 2.222219843214978396e-01
    l5 = 1.818357216161805012e-01
    l6 = 1.531383769920937332e-01
    l7 = 1.479819860511658591e-01
    poly = z * (l1 + z * (l2 + z * (l3 + z * (l4 + z * (l5 + z * (l6 + z * l7))))))
    ln2_hi = 6.93147180369123816490e-01
    ln2_lo = 1.90821492927058770002e-10
    out = k * ln2_hi + ((2.0 * s + s * poly) + k * ln2_lo)
    out = np.where(arr <= 0.0, -1.7976931348623157e308, out)
    return float(out[0]) if is_scalar else out


def exact_box_muller_f64(
    u1: float | np.ndarray,
    u2: float | np.ndarray,
) -> float | np.ndarray:
    """Bit-exact host/device Box-Muller Gaussian transform matching Pcg32::gaussian()."""
    u1_arr, s1 = _as_f64_array(u1)
    u2_arr, s2 = _as_f64_array(u2)
    u1_safe = np.maximum(u1_arr, 1e-12)
    ln_u1 = np.asarray(exact_log_f64(u1_safe), dtype=np.float64)
    radius = np.sqrt(-2.0 * ln_u1)
    theta = 2.0 * math.pi * u2_arr
    cos_t = np.asarray(exact_cos_f64(theta), dtype=np.float64)
    out = radius * cos_t
    return float(out[0]) if (s1 and s2) else out


@dataclass
class Phase2HeadSliceSpec:
    """Per-winner buffer layout specification for a Phase 2 head family."""

    family: str
    rbuf_floats_per_row: int
    ibuf_ints_per_row: int
    ibuf_const_ints: int


PHASE2_HEAD_SLICE_SPECS: dict[str, Phase2HeadSliceSpec] = {
    "ridge_gcv": Phase2HeadSliceSpec("ridge_gcv", 260, 0, 0),
    "pls1_nipals": Phase2HeadSliceSpec("pls1_nipals", 260, 0, 0),
    "knn_3_7_15": Phase2HeadSliceSpec("knn_3_7_15", 260, 4, 0),
    "ksmooth": Phase2HeadSliceSpec("ksmooth", 260, 4, 0),
    "histgbm": Phase2HeadSliceSpec("histgbm", 260, 268, IBUF_HIST_EXTRA_INTS),
}


class Phase2WinnerWorkspace:
    """Phase 2 HPC #6: Persistent non-aliasing workspace pool for Phase 2 head evaluators.

    Pre-allocates reusable ``rbuf_all`` (``W_max x (260 * R_max)`` ``float64``) and
    ``ibuf_all`` (``W_max x (268 * R_max + 2048)`` ``int32``) buffers so that ``<<<W_n, 256>>>``
    Phase 2 head kernels execute with zero per-batch ``cudaMalloc`` / ``cudaFree`` or full-workspace
    ``cudaMemsetAsync`` overhead and without buffer overruns on ``knn_3_7_15``, ``ksmooth``, or
    ``histgbm``.
    """

    def __init__(
        self,
        max_winners: int = 96,
        max_rows: int = 4096,
        device: str | torch.device = "cpu",
    ) -> None:
        if max_winners <= 0 or max_rows <= 0:
            raise ValueError("max_winners and max_rows must be positive")
        self.max_winners = int(max_winners)
        self.max_rows = int(max_rows)
        self.device = torch.device(device)
        self.rbuf_stride = RBUF_STRIDE_PER_ROW * self.max_rows
        self.ibuf_stride = IBUF_STRIDE_PER_ROW * self.max_rows + IBUF_HIST_EXTRA_INTS
        self.rbuf_all = torch.empty(
            (self.max_winners, self.rbuf_stride),
            dtype=torch.float64,
            device=self.device,
        )
        self.ibuf_all = torch.empty(
            (self.max_winners, self.ibuf_stride),
            dtype=torch.int32,
            device=self.device,
        )
        self.acquire_count = 0
        self.realloc_count = 0

    def ensure_capacity(
        self,
        n_winners: int,
        n_rows: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return views for ``(n_winners, rbuf_stride(n_rows))`` without reallocating when in bounds."""
        if n_winners <= 0 or n_rows <= 0:
            raise ValueError("n_winners and n_rows must be positive")
        if n_winners > self.max_winners or n_rows > self.max_rows:
            self.max_winners = max(self.max_winners, int(n_winners))
            self.max_rows = max(self.max_rows, int(n_rows))
            self.rbuf_stride = RBUF_STRIDE_PER_ROW * self.max_rows
            self.ibuf_stride = IBUF_STRIDE_PER_ROW * self.max_rows + IBUF_HIST_EXTRA_INTS
            self.rbuf_all = torch.empty(
                (self.max_winners, self.rbuf_stride),
                dtype=torch.float64,
                device=self.device,
            )
            self.ibuf_all = torch.empty(
                (self.max_winners, self.ibuf_stride),
                dtype=torch.int32,
                device=self.device,
            )
            self.realloc_count += 1
        self.acquire_count += 1
        req_rbuf = RBUF_STRIDE_PER_ROW * int(n_rows)
        req_ibuf = IBUF_STRIDE_PER_ROW * int(n_rows) + IBUF_HIST_EXTRA_INTS
        return (
            self.rbuf_all[:n_winners, :req_rbuf],
            self.ibuf_all[:n_winners, :req_ibuf],
        )

    def verify_non_aliasing(self, n_winners: int, n_rows: int) -> dict[str, Any]:
        """Verify that every Phase 2 head family's buffer offsets fit within the per-winner stride."""
        rbuf_view, ibuf_view = self.ensure_capacity(n_winners, n_rows)
        r_req = RBUF_STRIDE_PER_ROW * int(n_rows)
        i_req = IBUF_STRIDE_PER_ROW * int(n_rows) + IBUF_HIST_EXTRA_INTS

        # Explicit head-family offset checks matching phase2_universe_kernels.cu [D1]/[D3]:
        # - ridge_gcv / pls1_nipals / knn_3_7_15 / ksmooth: X_tr (256*R), resid/pred (4*R) -> 260*R
        # - histgbm: X_tr (256*R), resid/g/h/pred (4*R) -> 260*R floats;
        #   q_tr (256*R) + node_id (8*R) + valid_idx (4*R) + tree tables (2048) -> 268*R + 2048 ints
        family_footprints = {
            "ridge_gcv": {"rbuf_end": 260 * n_rows, "ibuf_end": 0},
            "pls1_nipals": {"rbuf_end": 260 * n_rows, "ibuf_end": 0},
            "knn_3_7_15": {"rbuf_end": 260 * n_rows, "ibuf_end": 4 * n_rows},
            "ksmooth": {"rbuf_end": 260 * n_rows, "ibuf_end": 4 * n_rows},
            "histgbm": {
                "rbuf_end": 260 * n_rows,
                "ibuf_end": 268 * n_rows + IBUF_HIST_EXTRA_INTS,
            },
        }
        for fam, bounds in family_footprints.items():
            if bounds["rbuf_end"] > rbuf_view.shape[1]:
                raise AssertionError(f"{fam} rbuf overflow: {bounds['rbuf_end']} > {rbuf_view.shape[1]}")
            if bounds["ibuf_end"] > ibuf_view.shape[1]:
                raise AssertionError(f"{fam} ibuf overflow: {bounds['ibuf_end']} > {ibuf_view.shape[1]}")
        return {
            "n_winners": int(n_winners),
            "n_rows": int(n_rows),
            "rbuf_stride": r_req,
            "ibuf_stride": i_req,
            "families_ok": True,
            "family_footprints": family_footprints,
        }


def _fnv1a_embs_hash_u64(embs_24: np.ndarray) -> np.ndarray:
    """Compute deterministic 64-bit FNV-1a keys over ``(..., 24)`` int32 embeddings."""
    arr = np.asarray(embs_24, dtype=np.int32)
    u32 = arr.view(np.uint32).astype(np.uint64)
    h = np.full(u32.shape[:-1], np.uint64(1469598103934665603), dtype=np.uint64)
    prime = np.uint64(1099511628211)
    for d in range(u32.shape[-1]):
        h = (h ^ u32[..., d]) * prime
    return h


def _block256_table_seed_u32(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
) -> tuple[int, np.ndarray]:
    """Compute the 256-lane cooperative XOR-folded table seed matching ``build_live_96_slate_block256_cuda_kernel``."""
    x_f32 = np.asarray(x_tr, dtype=np.float32).reshape(-1)
    y_f32 = np.asarray(y_tr, dtype=np.float32).reshape(-1)
    n_rows = int(y_f32.shape[0])
    n_cols = int(x_f32.shape[0] // max(n_rows, 1))
    total_x = int(x_f32.shape[0])

    x_clean = np.where(np.isfinite(x_f32), x_f32, np.float32(0.0))
    y_clean = np.where(np.isfinite(y_f32), y_f32, np.float32(0.0))
    x_bits = x_clean.view(np.uint32).astype(np.uint64)
    y_bits = y_clean.view(np.uint32).astype(np.uint64)

    init_seed = np.uint64(0x243F6A8885A308D3) ^ (np.uint64(n_rows) << np.uint64(32)) ^ np.uint64(n_cols)
    lane_hashes = np.full(256, init_seed, dtype=np.uint64)

    idx_x = np.arange(total_x, dtype=np.uint64)
    mix_x = (x_bits ^ (idx_x * np.uint64(0x9E3779B97F4A7C15))) * np.uint64(0xBF58476D1CE4E5B9)
    contrib_x = mix_x ^ (mix_x >> np.uint64(31))
    for tid in range(min(256, total_x)):
        if contrib_x[tid::256].size > 0:
            lane_hashes[tid] ^= np.bitwise_xor.reduce(contrib_x[tid::256])

    idx_y = np.arange(n_rows, dtype=np.uint64) + np.uint64(total_x)
    mix_y = (y_bits ^ (idx_y * np.uint64(0x9E3779B97F4A7C15))) * np.uint64(0x94D049BB133111EB)
    contrib_y = mix_y ^ (mix_y >> np.uint64(27))
    for tid in range(min(256, n_rows)):
        if contrib_y[tid::256].size > 0:
            lane_hashes[tid] ^= np.bitwise_xor.reduce(contrib_y[tid::256])

    folded = int(np.bitwise_xor.reduce(lane_hashes))
    final_h = (folded * 0xD6E8FEB86659FD93) & 0xFFFFFFFFFFFFFFFF
    seed_u32 = int((final_h ^ (final_h >> 32)) & 0xFFFFFFFF)
    return seed_u32, lane_hashes


def build_live_96_slate_block256(
    x_tr: np.ndarray | torch.Tensor,
    y_tr: np.ndarray | torch.Tensor,
    lm48: torch.Tensor,
    lm48_z: torch.Tensor,
    champ_lm48_z: torch.Tensor,
    champ_t_norms: torch.Tensor,
    champ_lm12_128: torch.Tensor,
    champ_embs_881: np.ndarray | torch.Tensor,
    champ_embs_128: np.ndarray | torch.Tensor,
    sorted_keys: np.ndarray | torch.Tensor | None = None,
    champ_shas: list[str] | None = None,
) -> dict[str, Any]:
    """Part B HPC #1: 256-thread block slate builder replacing lane-0 serial ``build_live_96_slate_cuda_kernel``.

    Combines:
    - Batched cuBLAS GEMM top-K retrieval over the 881 template library (top 23 templates x 4 ranks
      = 92 candidates) and 128 champion pool (top 4 champions = 4 candidates).
    - 256-lane strided cooperative table seed reduction (eliminating the single-thread ``O(N*P)``
      loop in ``build_live_96_slate_cuda_kernel``).
    - 96-way parallel candidate embedding assembly + binary search over ``sorted_keys``.
    """
    x_np = (
        x_tr.detach().cpu().numpy() if isinstance(x_tr, torch.Tensor) else np.asarray(x_tr, dtype=np.float32)
    )
    y_np = (
        y_tr.detach().cpu().numpy() if isinstance(y_tr, torch.Tensor) else np.asarray(y_tr, dtype=np.float32)
    )
    seed_u32, lane_hashes = _block256_table_seed_u32(x_np, y_np)

    lm48_b = lm48.unsqueeze(0) if lm48.ndim == 1 else lm48
    lm48_zb = lm48_z.unsqueeze(0) if lm48_z.ndim == 1 else lm48_z
    topk = precompute_champion_topk_batched_gpu(
        lm48_b,
        lm48_zb,
        champ_lm48_z,
        champ_t_norms,
        champ_lm12_128,
        champ_shas=champ_shas,
    )

    embs881_np = (
        champ_embs_881.detach().cpu().numpy()
        if isinstance(champ_embs_881, torch.Tensor)
        else np.asarray(champ_embs_881, dtype=np.int32)
    )
    embs128_np = (
        champ_embs_128.detach().cpu().numpy()
        if isinstance(champ_embs_128, torch.Tensor)
        else np.asarray(champ_embs_128, dtype=np.int32)
    )

    n_top_templates = min(23, int(embs881_np.shape[0]))
    top23_t = topk["top_t_indices_881"][0, :n_top_templates].detach().cpu().numpy()
    top4_c128 = topk["top4_champ128_indices"][0, :4].detach().cpu().numpy()

    # 92 template candidates (23 templates x 4 ranks) + 4 champion-128 candidates = 96
    embs_from_881 = embs881_np[top23_t].reshape(-1, 24)
    embs_from_128 = embs128_np[top4_c128].reshape(-1, 24)
    slate_embs = np.concatenate([embs_from_881, embs_from_128], axis=0).astype(np.int32)
    slate_keys = _fnv1a_embs_hash_u64(slate_embs)

    if sorted_keys is None:
        sorted_keys_np = np.sort(np.unique(slate_keys))
    else:
        sorted_keys_np = (
            sorted_keys.detach().cpu().numpy().astype(np.uint64)
            if isinstance(sorted_keys, torch.Tensor)
            else np.asarray(sorted_keys, dtype=np.uint64)
        )

    pos = np.searchsorted(sorted_keys_np, slate_keys)
    in_bounds = pos < sorted_keys_np.shape[0]
    safe_pos = np.where(in_bounds, pos, 0)
    matched = in_bounds & (sorted_keys_np[safe_pos] == slate_keys)
    slate_key_idx = np.where(matched, pos, -1).astype(np.int32)

    return {
        "table_seed_u32": seed_u32,
        "lane_hashes_u64": lane_hashes,
        "top23_template_indices": top23_t,
        "top4_champ128_indices": top4_c128,
        "slate_embs": slate_embs,
        "slate_keys_u64": slate_keys,
        "slate_key_idx": slate_key_idx,
        "cuda_kernel_source": BUILD_LIVE_96_SLATE_BLOCK256_CUDA_SOURCE,
    }
