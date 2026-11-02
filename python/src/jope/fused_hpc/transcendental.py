"""IEEE-754 host/device transcendental helpers for fused HPC kernels."""

from __future__ import annotations

import math

import numpy as np

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
