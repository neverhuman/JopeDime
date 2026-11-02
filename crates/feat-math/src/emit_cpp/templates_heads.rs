use super::*;

pub(super) const H_SOFT_THRESHOLD: &str = r#"// Shrink v toward zero by t (the lasso proximal step).
static double soft_threshold(double v, double t) {
    if (v > t) return v - t;
    if (v < -t) return v + t;
    return 0.0;
}"#;

pub(super) const HEAD_ELASTICNET: &str = r#"struct ElasticNet {
    std::vector<double> means;
    std::vector<double> scales;
    double w0;
    std::vector<double> w;
};

// Elastic net regression (Zou & Hastie, 2005) via cyclic coordinate descent with
// soft-thresholding (Friedman, Hastie & Tibshirani, 2010); predictions clipped to [-60, 60].
static ElasticNet elastic_net_fit(const std::vector<std::vector<double>>& x, const std::vector<double>& y, double alpha, double l1_ratio, std::size_t sweeps) {
    std::size_t n = std::min(x.size(), y.size());
    std::size_t p = x[0].size();
    std::vector<double> means;
    std::vector<double> scales;
    std_fit(x, means, scales);
    std::vector<std::vector<double>> xs;
    xs.reserve(n);
    for (std::size_t r = 0; r < n; ++r) {
        const auto& row = x[r];
        std::vector<double> sr;
        sr.reserve(p);
        for (std::size_t j = 0; j < p; ++j) {
            sr.push_back(clamp_range((row[j] - means[j]) / scales[j], -20.0, 20.0));
        }
        xs.push_back(std::move(sr));
    }
    double y_mean = 0.0;
    for (std::size_t i = 0; i < n; ++i) y_mean += y[i];
    y_mean /= static_cast<double>(std::max<std::size_t>(n, 1));
    double l1 = alpha * l1_ratio;
    double l2 = alpha * (1.0 - l1_ratio);
    std::vector<double> col_sq(p, 0.0);
    for (const auto& row : xs) {
        for (std::size_t j = 0; j < p; ++j) col_sq[j] += row[j] * row[j];
    }
    std::vector<double> w(p, 0.0);
    std::vector<double> resid(n, 0.0);
    for (std::size_t r = 0; r < n; ++r) resid[r] = y[r] - y_mean;
    std::size_t max_sweeps = std::max<std::size_t>(sweeps, 1);
    for (std::size_t sweep = 0; sweep < max_sweeps; ++sweep) {
        double max_delta = 0.0;
        for (std::size_t j = 0; j < p; ++j) {
            if (col_sq[j] <= 1e-12) continue;
            double rho = 0.0;
            for (std::size_t r = 0; r < n; ++r) rho += xs[r][j] * resid[r];
            rho = rho / static_cast<double>(n) + w[j] * col_sq[j] / static_cast<double>(n);
            double denom = col_sq[j] / static_cast<double>(n) + l2;
            double new_w = soft_threshold(rho, l1) / denom;
            double delta = new_w - w[j];
            if (delta != 0.0) {
                for (std::size_t r = 0; r < n; ++r) resid[r] -= delta * xs[r][j];
                w[j] = new_w;
                double ad = std::fabs(delta);
                if (ad > max_delta) max_delta = ad;
            }
        }
        if (max_delta < 1e-9) break;
    }
    ElasticNet state;
    state.means = std::move(means);
    state.scales = std::move(scales);
    state.w0 = y_mean;
    state.w = std::move(w);
    return state;
}

static std::vector<double> elastic_net_predict(const ElasticNet& state, const std::vector<std::vector<double>>& x) {
    std::size_t p = state.means.size();
    std::vector<double> predictions;
    predictions.reserve(x.size());
    for (const auto& row : x) {
        double z = state.w0;
        std::size_t lim = std::min(p, row.size());
        for (std::size_t j = 0; j < lim; ++j) {
            double xs = clamp_range((row[j] - state.means[j]) / state.scales[j], -20.0, 20.0);
            z += state.w[j] * xs;
        }
        predictions.push_back(clamp_range(z, -60.0, 60.0));
    }
    return predictions;
}"#;

pub(super) const HEAD_RIDGE: &str = r#"struct RidgeRegression {
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
}"#;

// Verified portable-scalar-kernel-v1 C++ implementation. This is the same
// defined-arithmetic body exercised by tests/portable/full_ref.cpp; the
// generated source contains no fixture harness or external access.
