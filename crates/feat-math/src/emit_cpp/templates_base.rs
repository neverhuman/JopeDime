pub(crate) const EXPRESSION_SHAPE: &str = r#"static std::size_t expression_width(const std::vector<std::vector<double>>& rows) {
    if(rows.empty() || rows.front().empty()) throw std::invalid_argument("empty numeric table");
    const std::size_t width=rows.front().size();
    for(const auto& row:rows) if(row.size()!=width) throw std::invalid_argument("ragged numeric table");
    return width;
}"#;

pub(crate) const PREAMBLE: &str = r#"// Standalone regression model — self-contained, depends only on the C++
// standard library, and accepts all data through its C ABI.
//   Build: g++ -O2 -std=c++17 -ffp-contract=off -fno-fast-math -c model.cpp
//
// All arithmetic uses IEEE-754 double precision in a fixed evaluation order, so
// repeated runs on the same input produce identical results. Build without
// fused-multiply-add contraction (-ffp-contract=off) to keep each multiply and
// add separately rounded.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

// Clamp to [lo, hi] with a fixed branch order; a NaN input passes through.
static double clamp_range(double x, double lo, double hi) {
    if (x < lo) x = lo;
    if (x > hi) x = hi;
    return x;
}

// Larger of two values, ignoring NaN.
static double max_num(double a, double b) {
    if (std::isnan(a)) return b;
    if (std::isnan(b)) return a;
    if (a < b) return b;
    return a;
}

// IEEE-754 total order, so sorts and score-ranking are deterministic (matches the
// reference tie/sign handling). Unused by pipelines with no ordering stage.
[[maybe_unused]] static bool total_less(double a, double b) {
    std::uint64_t ua;
    std::uint64_t ub;
    std::memcpy(&ua, &a, sizeof(double));
    std::memcpy(&ub, &b, sizeof(double));
    std::int64_t la = static_cast<std::int64_t>(ua);
    std::int64_t lb = static_cast<std::int64_t>(ub);
    la ^= static_cast<std::int64_t>(static_cast<std::uint64_t>(la >> 63) >> 1);
    lb ^= static_cast<std::int64_t>(static_cast<std::uint64_t>(lb >> 63) >> 1);
    return la < lb;
}"#;

pub(crate) const H_SANITIZE: &str = r#"// Replace non-finite values with 0.0 in every row.
static std::vector<std::vector<double>> sanitize(const std::vector<std::vector<double>>& x) {
    std::vector<std::vector<double>> out;
    out.reserve(x.size());
    for (const auto& row : x) {
        std::vector<double> r;
        r.reserve(row.size());
        for (double v : row) r.push_back(std::isfinite(v) ? v : 0.0);
        out.push_back(std::move(r));
    }
    return out;
}"#;

pub(crate) const H_FINISH_STAGE: &str = r#"// Stage epilogue: optionally keep the inputs, cap width at 512, zero non-finite cells.
static std::vector<std::vector<double>> finish_stage(const std::vector<std::vector<double>>& x, std::vector<std::vector<double>> out, bool append) {
    if (append) {
        for (std::size_t i = 0; i < out.size(); ++i) {
            std::vector<double> merged = x[i];
            merged.insert(merged.end(), out[i].begin(), out[i].end());
            out[i] = std::move(merged);
        }
    }
    for (auto& row : out) {
        if (row.size() > 512) row.resize(512);
        for (double& v : row) {
            if (!std::isfinite(v)) v = 0.0;
        }
        if (row.empty()) row.push_back(0.0);
    }
    return out;
}"#;

pub(crate) const H_STD_FIT: &str = r#"// Per-column mean and standard deviation (1/n convention), floored at 1e-9.
static void std_fit(const std::vector<std::vector<double>>& x, std::vector<double>& means, std::vector<double>& scales) {
    double n = static_cast<double>(std::max<std::size_t>(x.size(), 1));
    std::size_t p = x[0].size();
    means.assign(p, 0.0);
    scales.assign(p, 1.0);
    for (std::size_t j = 0; j < p; ++j) {
        double s = 0.0;
        for (const auto& row : x) s += row[j];
        means[j] = s / n;
        double v = 0.0;
        for (const auto& row : x) {
            double d = row[j] - means[j];
            v += d * d;
        }
        scales[j] = max_num(std::sqrt(v / n), 1e-9);
    }
}"#;

pub(crate) const H_CHOLESKY: &str = r#"// Solve the symmetric positive-definite system A w = b (A row-major, overwritten
// in place as L L^T, then forward and backward triangular solves).
static std::vector<double> cholesky_solve(std::vector<double>& a, const std::vector<double>& b, std::size_t p) {
    for (std::size_t i = 0; i < p; ++i) {
        for (std::size_t j = 0; j < i + 1; ++j) {
            double s = a[i * p + j];
            for (std::size_t k = 0; k < j; ++k) s -= a[i * p + k] * a[j * p + k];
            if (i == j) {
                a[i * p + j] = (s > 1e-12) ? std::sqrt(s) : 1e-6;
            } else {
                a[i * p + j] = s / a[j * p + j];
            }
        }
    }
    std::vector<double> z(p, 0.0);
    for (std::size_t i = 0; i < p; ++i) {
        double s = b[i];
        for (std::size_t k = 0; k < i; ++k) s -= a[i * p + k] * z[k];
        z[i] = s / a[i * p + i];
    }
    std::vector<double> w(p, 0.0);
    for (std::size_t ii = 0; ii < p; ++ii) {
        std::size_t i = p - 1 - ii;
        double s = z[i];
        for (std::size_t k = i + 1; k < p; ++k) s -= a[k * p + i] * w[k];
        w[i] = s / a[i * p + i];
    }
    return w;
}"#;

pub(crate) const H_SORTED_COLUMN: &str = r#"// Column j of x in ascending order (IEEE total order via total_less).
static std::vector<double> sorted_column(const std::vector<std::vector<double>>& x, std::size_t j) {
    std::vector<double> v;
    v.reserve(x.size());
    for (const auto& row : x) v.push_back(row[j]);
    std::sort(v.begin(), v.end(), total_less);
    return v;
}"#;

pub(crate) const H_QUANTILE: &str = r#"// Linearly interpolated quantile of an ascending slice (the numpy 'linear' rule).
static double quantile(const std::vector<double>& sorted_values, double q) {
    if (sorted_values.empty()) return 0.0;
    double pos = q * static_cast<double>(sorted_values.size() - 1);
    std::size_t lo = static_cast<std::size_t>(pos);
    std::size_t hi = std::min(lo + 1, sorted_values.size() - 1);
    double frac = pos - static_cast<double>(lo);
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac;
}"#;

pub(crate) const H_LOWER_BOUND: &str = r#"// Count of elements strictly less than v (hand-rolled binary search).
static std::size_t lower_bound(const std::vector<double>& sorted_values, double v) {
    std::size_t lo = 0;
    std::size_t hi = sorted_values.size();
    while (lo < hi) {
        std::size_t mid = (lo + hi) / 2;
        if (sorted_values[mid] < v) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}"#;

pub(crate) const H_UPPER_BOUND: &str = r#"// Count of elements less than or equal to v (hand-rolled binary search).
static std::size_t upper_bound(const std::vector<double>& sorted_values, double v) {
    std::size_t lo = 0;
    std::size_t hi = sorted_values.size();
    while (lo < hi) {
        std::size_t mid = (lo + hi) / 2;
        if (sorted_values[mid] <= v) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}"#;

pub(crate) const H_LOWER_BOUND_PARTITION_POINT: &str = r#"// Count of elements strictly less than v (hand-rolled binary search).
static std::size_t lower_bound(const std::vector<double>& sorted_values, double v) {
    std::size_t size = sorted_values.size();
    if (size == 0) {
        return 0;
    }
    std::size_t base = 0;
    while (size > 1) {
        std::size_t half = size / 2;
        std::size_t mid = base + half;
        if (sorted_values[mid] < v) {
            base = mid;
        }
        size -= half;
    }
    return base + static_cast<std::size_t>(sorted_values[base] < v);
}"#;

pub(crate) const H_UPPER_BOUND_PARTITION_POINT: &str = r#"// Count of elements less than or equal to v (hand-rolled binary search).
static std::size_t upper_bound(const std::vector<double>& sorted_values, double v) {
    std::size_t size = sorted_values.size();
    if (size == 0) {
        return 0;
    }
    std::size_t base = 0;
    while (size > 1) {
        std::size_t half = size / 2;
        std::size_t mid = base + half;
        if (sorted_values[mid] <= v) {
            base = mid;
        }
        size -= half;
    }
    return base + static_cast<std::size_t>(sorted_values[base] <= v);
}"#;

pub(crate) const H_MID_RANK: &str = r#"// Average of the strict and non-strict ranks of v, scaled to [0, 1].
static double mid_rank(const std::vector<double>& sorted_values, double v) {
    std::size_t lb = lower_bound(sorted_values, v);
    std::size_t ub = upper_bound(sorted_values, v);
    return static_cast<double>(lb + ub) / (2.0 * static_cast<double>(sorted_values.size()));
}"#;

pub(crate) const H_PROBIT: &str = r#"// Standard normal quantile function via Acklam's rational approximation.
static double probit(double p) {
    const double a[6] = {-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
        1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00};
    const double b[5] = {-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
        6.680131188771972e+01, -1.328068155288572e+01};
    const double c[6] = {-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
        -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00};
    const double d[4] = {7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
        3.754408661907416e+00};
    p = clamp_range(p, 1e-12, 1.0 - 1e-12);
    if (p < 0.02425) {
        double q = std::sqrt(-2.0 * std::log(p));
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    } else if (p <= 1.0 - 0.02425) {
        double q = p - 0.5;
        double r = q * q;
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0);
    } else {
        double q = std::sqrt(-2.0 * std::log(1.0 - p));
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }
}"#;

pub(crate) const H_MATH_SIGMOID: &str = r#"// Logistic function 1/(1+exp(-v)) with the argument clipped to [-30, 30].
static double m_sigmoid(double value) {
    return 1.0 / (1.0 + std::exp(-clamp_range(value, -30.0, 30.0)));
}"#;

pub(crate) const H_PCG32: &str = r#"// 32-bit rotate right (rotate by 0 is identity).
static std::uint32_t rotr32(std::uint32_t x, std::uint32_t r) {
    return (x >> r) | (x << ((32 - r) & 31));
}

// Permuted congruential generator, XSH-RR 64/32 (O'Neill, 2014), with the
// uniform/range/normal draws layered on top. Mirrors the reference RNG bit-for-bit.
struct Pcg32 {
    std::uint64_t state;
    std::uint64_t inc;

    std::uint32_t next_u32() {
        std::uint64_t old = state;
        state = old * 6364136223846793005ULL + inc;
        std::uint32_t xorshifted = static_cast<std::uint32_t>((((old >> 18) ^ old) >> 27));
        std::uint32_t rot = static_cast<std::uint32_t>(old >> 59);
        return rotr32(xorshifted, rot);
    }

    // Uniform in [0, 1) from 53 random bits: ((hi<<32|lo) >> 11) * 2^-53.
    double next_f64() {
        std::uint64_t hi = static_cast<std::uint64_t>(next_u32()) << 32;
        std::uint64_t lo = static_cast<std::uint64_t>(next_u32());
        return static_cast<double>((hi | lo) >> 11) * (1.0 / 9007199254740992.0);
    }

    double range_f64(double a, double b) {
        return a + (b - a) * next_f64();
    }

    // Standard normal via the Box-Muller cosine branch (u1 floored at 1e-12).
    double gaussian() {
        double u1 = max_num(next_f64(), 1e-12);
        double u2 = next_f64();
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * 3.141592653589793 * u2);
    }
};

static Pcg32 pcg32_new(std::uint64_t seed, std::uint64_t stream) {
    Pcg32 rng;
    rng.state = 0;
    rng.inc = (stream << 1) | 1ULL;
    rng.next_u32();
    rng.state = rng.state + seed;
    rng.next_u32();
    return rng;
}"#;

pub(crate) const H_STAGE_RNG: &str = r#"// Independent random stream keyed by a stage's input and output widths.
static Pcg32 stage_rng(std::uint64_t seed, std::size_t in_cols, std::size_t n_out) {
    return pcg32_new(seed, (static_cast<std::uint64_t>(in_cols) << 32) ^ static_cast<std::uint64_t>(n_out));
}"#;

pub(crate) const H_DOT: &str = r#"// Inner product over the shorter of the two vectors.
static double dot(const std::vector<double>& a, const std::vector<double>& b) {
    double total = 0.0;
    std::size_t m = std::min(a.size(), b.size());
    for (std::size_t i = 0; i < m; ++i) total += a[i] * b[i];
    return total;
}"#;
