// Standalone regression model — self-contained, depends only on the C++
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
}

// Replace non-finite values with 0.0 in every row.
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
}

// Stage epilogue: optionally keep the inputs, cap width at 512, zero non-finite cells.
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
}

// Per-column mean and standard deviation (1/n convention), floored at 1e-9.
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
}

// Solve the symmetric positive-definite system A w = b (A row-major, overwritten
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
}

struct StandardScaler {
    std::vector<double> means;
    std::vector<double> scales;
};

// Centers each column by its training mean and divides by its standard deviation.
static StandardScaler standard_scaler_fit(const std::vector<std::vector<double>>& x) {
    StandardScaler state;
    std_fit(x, state.means, state.scales);
    return state;
}

static std::vector<std::vector<double>> standard_scaler_apply(const StandardScaler& state, const std::vector<std::vector<double>>& x, bool append) {
    std::vector<std::vector<double>> out;
    out.reserve(x.size());
    for (const auto& row : x) {
        std::vector<double> r;
        r.reserve(row.size());
        for (std::size_t j = 0; j < row.size(); ++j) {
            r.push_back(clamp_range((row[j] - state.means[j]) / state.scales[j], -20.0, 20.0));
        }
        out.push_back(std::move(r));
    }
    return finish_stage(x, std::move(out), append);
}

struct RidgeRegression {
    std::vector<double> means;
    std::vector<double> scales;
    std::vector<double> w;
};

// Ridge regression on standardized features via the Cholesky normal equations.
static RidgeRegression ridge_regression_fit(const std::vector<std::vector<double>>& x, const std::vector<double>& y, double lambda_frac) {
    std::size_t n = std::min(x.size(), y.size());
    std::size_t p = x[0].size();
    std::vector<double> means;
    std::vector<double> scales;
    std_fit(x, means, scales);
    double lam = max_num(lambda_frac * static_cast<double>(n), 1e-3);
    double y_mean = 0.0;
    for (std::size_t i = 0; i < n; ++i) y_mean += y[i];
    y_mean /= static_cast<double>(std::max<std::size_t>(n, 1));
    std::vector<double> ata(p * p, 0.0);
    std::vector<double> atb(p, 0.0);
    std::vector<double> xs_row(p, 0.0);
    for (std::size_t r = 0; r < n; ++r) {
        const auto& row = x[r];
        for (std::size_t j = 0; j < p; ++j) {
            xs_row[j] = clamp_range((row[j] - means[j]) / scales[j], -20.0, 20.0);
        }
        double yr = y[r] - y_mean;
        for (std::size_t i = 0; i < p; ++i) {
            double xi = xs_row[i];
            atb[i] += xi * yr;
            for (std::size_t j = i; j < p; ++j) {
                ata[i * p + j] += xi * xs_row[j];
            }
        }
    }
    for (std::size_t i = 0; i < p; ++i) {
        ata[i * p + i] += lam;
        for (std::size_t j = 0; j < i; ++j) {
            ata[i * p + j] = ata[j * p + i];
        }
    }
    std::vector<double> coeffs = cholesky_solve(ata, atb, p);
    std::vector<double> w;
    w.reserve(p + 1);
    w.push_back(y_mean);
    for (double c : coeffs) w.push_back(c);
    RidgeRegression state;
    state.means = std::move(means);
    state.scales = std::move(scales);
    state.w = std::move(w);
    return state;
}

static std::vector<double> ridge_regression_predict(const RidgeRegression& state, const std::vector<std::vector<double>>& x) {
    std::size_t p = state.means.size();
    std::vector<double> predictions;
    predictions.reserve(x.size());
    for (const auto& row : x) {
        double z = state.w[0];
        std::size_t lim = std::min(p, row.size());
        for (std::size_t j = 0; j < lim; ++j) {
            double xs = clamp_range((row[j] - state.means[j]) / state.scales[j], -20.0, 20.0);
            z += state.w[j + 1] * xs;
        }
        predictions.push_back(clamp_range(z, -60.0, 60.0));
    }
    return predictions;
}

// Feature stages applied in order, then the ridge regression head.
struct Model {
    StandardScaler stage1;
    RidgeRegression head;

    static Model fit(const std::vector<std::vector<double>>& x_in, const std::vector<double>& y);
    std::vector<double> predict(const std::vector<std::vector<double>>& x_in) const;
};

Model Model::fit(const std::vector<std::vector<double>>& x_in, const std::vector<double>& y) {
    if (x_in.empty() || y.size() < x_in.size()) {
        throw std::invalid_argument("training data is empty or target is short");
    }
    std::vector<std::vector<double>> x = sanitize(x_in);
    auto stage1 = standard_scaler_fit(x);
    x = standard_scaler_apply(stage1, x, false);
    RidgeRegression head = ridge_regression_fit(x, y, 0.02);
    Model m;
    m.stage1 = std::move(stage1);
    m.head = std::move(head);
    return m;
}

std::vector<double> Model::predict(const std::vector<std::vector<double>>& x_in) const {
    std::vector<std::vector<double>> x = sanitize(x_in);
    x = standard_scaler_apply(stage1, x, false);
    return ridge_regression_predict(head, x);
}

// ---- caller-owned C ABI ----------------------------------------------------

extern "C" {

struct model_handle {
    Model model;
    std::size_t width;

    model_handle(Model&& fitted, std::size_t fitted_width)
        : model(std::move(fitted)), width(fitted_width) {}
};

std::uint32_t model_abi_version(void) noexcept {
    return 1U;
}

std::size_t model_workspace_bytes(const model_handle*, std::size_t) noexcept {
    return 0U;
}

int model_fit_numeric(
    const double* values,
    std::size_t values_len,
    std::size_t rows,
    std::size_t columns,
    const double* targets,
    std::size_t target_len,
    model_handle** out
) noexcept {
    if (out == nullptr) return 1;
    *out = nullptr;
    if (values == nullptr || targets == nullptr || rows == 0U || columns == 0U) return 1;
    if (target_len != rows) return 1;
    if (rows > std::numeric_limits<std::size_t>::max() / columns) return 1;
    if (values_len < rows * columns) return 1;
    try {
        std::vector<std::vector<double>> x;
        x.reserve(rows);
        std::vector<double> y;
        y.reserve(rows);
        for (std::size_t row = 0; row < rows; ++row) {
            std::vector<double> cells;
            cells.reserve(columns);
            for (std::size_t column = 0; column < columns; ++column) {
                cells.push_back(values[row * columns + column]);
            }
            if (!std::isfinite(targets[row])) return 1;
            x.push_back(std::move(cells));
            y.push_back(targets[row]);
        }
        Model fitted = Model::fit(x, y);
        *out = new model_handle(std::move(fitted), columns);
        return 0;
    } catch (...) {
        return 2;
    }
}

int model_predict_numeric_batch(
    const model_handle* handle,
    const double* values,
    std::size_t values_len,
    std::size_t rows,
    std::size_t columns,
    void* workspace,
    std::size_t workspace_bytes,
    double* out,
    std::size_t out_len
) noexcept {
    (void)workspace;
    (void)workspace_bytes;
    if (handle == nullptr || columns == 0U || columns != handle->width) return 1;
    if (rows == 0U) return values_len == 0U ? 0 : 1;
    if (rows > std::numeric_limits<std::size_t>::max() / columns) return 1;
    if (values == nullptr || values_len < rows * columns || out == nullptr || out_len < rows) return 1;
    try {
        std::vector<std::vector<double>> x;
        x.reserve(rows);
        for (std::size_t row = 0; row < rows; ++row) {
            std::vector<double> cells;
            cells.reserve(columns);
            for (std::size_t column = 0; column < columns; ++column) {
                cells.push_back(values[row * columns + column]);
            }
            x.push_back(std::move(cells));
        }
        std::vector<double> predictions = handle->model.predict(x);
        if (predictions.size() != rows) return 2;
        for (std::size_t row = 0; row < rows; ++row) out[row] = predictions[row];
        return 0;
    } catch (...) {
        return 2;
    }
}

void model_unload(model_handle* handle) noexcept {
    delete handle;
}

}  // extern "C"

