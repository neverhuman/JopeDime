use super::*;

pub(super) const OP_STANDARDIZE: &str = r#"struct StandardScaler {
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
}"#;

pub(super) const OP_ROBUST_SCALE: &str = r#"struct RobustScaler {
    std::vector<double> medians;
    std::vector<double> scales;
};

// Centers each column by its median and divides by its interquartile range.
static RobustScaler robust_scaler_fit(const std::vector<std::vector<double>>& x) {
    RobustScaler state;
    for (std::size_t j = 0; j < x[0].size(); ++j) {
        std::vector<double> s = sorted_column(x, j);
        state.medians.push_back(quantile(s, 0.5));
        double iqr = quantile(s, 0.75) - quantile(s, 0.25);
        state.scales.push_back(max_num(iqr, 1e-9));
    }
    return state;
}

static std::vector<std::vector<double>> robust_scaler_apply(const RobustScaler& state, const std::vector<std::vector<double>>& x, bool append) {
    std::vector<std::vector<double>> out;
    out.reserve(x.size());
    for (const auto& row : x) {
        std::vector<double> r;
        r.reserve(row.size());
        for (std::size_t j = 0; j < row.size(); ++j) {
            r.push_back((row[j] - state.medians[j]) / state.scales[j]);
        }
        out.push_back(std::move(r));
    }
    return finish_stage(x, std::move(out), append);
}"#;

pub(super) const OP_RANK: &str = r#"struct MidRankTransform {
    std::vector<std::vector<double>> sorted_cols;
};

// Maps each value to its mid-rank within its training column, in [0, 1].
static MidRankTransform mid_rank_transform_fit(const std::vector<std::vector<double>>& x) {
    MidRankTransform state;
    for (std::size_t j = 0; j < x[0].size(); ++j) state.sorted_cols.push_back(sorted_column(x, j));
    return state;
}

static std::vector<std::vector<double>> mid_rank_transform_apply(const MidRankTransform& state, const std::vector<std::vector<double>>& x, bool append) {
    std::vector<std::vector<double>> out;
    out.reserve(x.size());
    for (const auto& row : x) {
        std::vector<double> r;
        r.reserve(row.size());
        for (std::size_t j = 0; j < row.size(); ++j) r.push_back(mid_rank(state.sorted_cols[j], row[j]));
        out.push_back(std::move(r));
    }
    return finish_stage(x, std::move(out), append);
}"#;

pub(super) const OP_BLOM: &str = r#"struct BlomNormalScores {
    std::vector<std::vector<double>> sorted_cols;
};

// Maps each value to a normal quantile of its mid-rank (Blom, 1958).
static BlomNormalScores blom_normal_scores_fit(const std::vector<std::vector<double>>& x) {
    BlomNormalScores state;
    for (std::size_t j = 0; j < x[0].size(); ++j) state.sorted_cols.push_back(sorted_column(x, j));
    return state;
}

static std::vector<std::vector<double>> blom_normal_scores_apply(const BlomNormalScores& state, const std::vector<std::vector<double>>& x, bool append) {
    std::vector<std::vector<double>> out;
    out.reserve(x.size());
    for (const auto& row : x) {
        std::vector<double> values;
        values.reserve(row.size());
        for (std::size_t j = 0; j < row.size(); ++j) {
            double n = static_cast<double>(state.sorted_cols[j].size());
            double mid = mid_rank(state.sorted_cols[j], row[j]) * n;
            values.push_back(probit((mid + 0.125) / (n + 0.25)));
        }
        out.push_back(std::move(values));
    }
    return finish_stage(x, std::move(out), append);
}"#;

pub(super) const OP_SIGMOID: &str = r#"// Applies the logistic squashing function to every value.
static std::vector<std::vector<double>> logistic_squash_apply(const std::vector<std::vector<double>>& x, bool append) {
    std::vector<std::vector<double>> out;
    out.reserve(x.size());
    for (const auto& row : x) {
        std::vector<double> r;
        r.reserve(row.size());
        for (double v : row) r.push_back(m_sigmoid(v));
        out.push_back(std::move(r));
    }
    return finish_stage(x, std::move(out), append);
}"#;

pub(super) const OP_STEP: &str = r#"// Per-column hard indicator: 1.0 where the value exceeds the threshold, else 0.0.
static std::vector<std::vector<double>> step_indicator_apply(const std::vector<std::vector<double>>& x, double t, bool append) {
    std::vector<std::vector<double>> out;
    out.reserve(x.size());
    for (const auto& row : x) {
        std::vector<double> r;
        r.reserve(row.size());
        for (double v : row) r.push_back(v > t ? 1.0 : 0.0);
        out.push_back(std::move(r));
    }
    return finish_stage(x, std::move(out), append);
}"#;

pub(super) const OP_WINSOR: &str = r#"struct WinsorLimiter {
    std::vector<double> lo;
    std::vector<double> hi;
};

// Clamps each column to its lower/upper training quantiles.
static WinsorLimiter winsor_limiter_fit(const std::vector<std::vector<double>>& x, double q) {
    WinsorLimiter state;
    for (std::size_t j = 0; j < x[0].size(); ++j) {
        std::vector<double> s = sorted_column(x, j);
        double low = quantile(s, q);
        double high = max_num(quantile(s, 1.0 - q), low);
        state.lo.push_back(low);
        state.hi.push_back(high);
    }
    return state;
}

static std::vector<std::vector<double>> winsor_limiter_apply(const WinsorLimiter& state, const std::vector<std::vector<double>>& x, bool append) {
    std::vector<std::vector<double>> out;
    out.reserve(x.size());
    for (const auto& row : x) {
        std::vector<double> r;
        r.reserve(row.size());
        for (std::size_t j = 0; j < row.size(); ++j) r.push_back(clamp_range(row[j], state.lo[j], state.hi[j]));
        out.push_back(std::move(r));
    }
    return finish_stage(x, std::move(out), append);
}"#;

pub(super) const OP_NORM_ROW: &str = r#"// Scales each row to unit L2 norm.
static std::vector<std::vector<double>> unit_norm_rows_apply(const std::vector<std::vector<double>>& x, bool append) {
    std::vector<std::vector<double>> out;
    out.reserve(x.size());
    for (const auto& row : x) {
        double s = 0.0;
        for (double v : row) s += v * v;
        double inv = 1.0 / max_num(std::sqrt(s), 1e-9);
        std::vector<double> r;
        r.reserve(row.size());
        for (double v : row) r.push_back(v * inv);
        out.push_back(std::move(r));
    }
    return finish_stage(x, std::move(out), append);
}"#;

pub(super) const OP_RFF_COS: &str = r#"struct RandomFourierCosineFeatures {
    std::vector<std::vector<double>> w;
    std::vector<double> b;
};

// Random Fourier cosine features (Rahimi & Recht, 2007): cosines of random projections.
static RandomFourierCosineFeatures random_fourier_cosine_features_fit(const std::vector<std::vector<double>>& x, std::size_t n_features, double gamma, std::uint64_t seed) {
    std::size_t cols = x[0].size();
    Pcg32 rng = stage_rng(seed, cols, n_features);
    double scale = std::sqrt(2.0 * gamma);
    std::vector<std::vector<double>> w;
    w.reserve(n_features);
    for (std::size_t i = 0; i < n_features; ++i) {
        std::vector<double> row;
        row.reserve(cols);
        for (std::size_t c = 0; c < cols; ++c) {
            row.push_back(rng.gaussian() * scale);
        }
        w.push_back(std::move(row));
    }
    std::vector<double> b;
    b.reserve(n_features);
    for (std::size_t i = 0; i < n_features; ++i) {
        b.push_back(rng.range_f64(0.0, 2.0 * 3.141592653589793));
    }
    RandomFourierCosineFeatures state;
    state.w = std::move(w);
    state.b = std::move(b);
    return state;
}

static std::vector<std::vector<double>> random_fourier_cosine_features_apply(const RandomFourierCosineFeatures& state, const std::vector<std::vector<double>>& x, bool append) {
    double scale = std::sqrt(2.0 / static_cast<double>(state.w.size()));
    std::vector<std::vector<double>> out;
    out.reserve(x.size());
    for (const auto& row : x) {
        std::vector<double> values;
        values.reserve(state.w.size());
        for (std::size_t i = 0; i < state.w.size(); ++i) {
            values.push_back(scale * std::cos(dot(state.w[i], row) + state.b[i]));
        }
        out.push_back(std::move(values));
    }
    return finish_stage(x, std::move(out), append);
}"#;

pub(super) const OP_RFF_SIN: &str = r#"struct RandomFourierSineFeatures {
    std::vector<std::vector<double>> w;
    std::vector<double> b;
};

// Random Fourier sine features (Rahimi & Recht, 2007): sines of random projections.
static RandomFourierSineFeatures random_fourier_sine_features_fit(const std::vector<std::vector<double>>& x, std::size_t n_features, double gamma, std::uint64_t seed) {
    std::size_t cols = x[0].size();
    Pcg32 rng = stage_rng(seed, cols, n_features);
    double scale = std::sqrt(2.0 * gamma);
    std::vector<std::vector<double>> w;
    w.reserve(n_features);
    for (std::size_t i = 0; i < n_features; ++i) {
        std::vector<double> row;
        row.reserve(cols);
        for (std::size_t c = 0; c < cols; ++c) {
            row.push_back(rng.gaussian() * scale);
        }
        w.push_back(std::move(row));
    }
    std::vector<double> b;
    b.reserve(n_features);
    for (std::size_t i = 0; i < n_features; ++i) {
        b.push_back(rng.range_f64(0.0, 2.0 * 3.141592653589793));
    }
    RandomFourierSineFeatures state;
    state.w = std::move(w);
    state.b = std::move(b);
    return state;
}

static std::vector<std::vector<double>> random_fourier_sine_features_apply(const RandomFourierSineFeatures& state, const std::vector<std::vector<double>>& x, bool append) {
    double scale = std::sqrt(2.0 / static_cast<double>(state.w.size()));
    std::vector<std::vector<double>> out;
    out.reserve(x.size());
    for (const auto& row : x) {
        std::vector<double> values;
        values.reserve(state.w.size());
        for (std::size_t i = 0; i < state.w.size(); ++i) {
            values.push_back(scale * std::sin(dot(state.w[i], row) + state.b[i]));
        }
        out.push_back(std::move(values));
    }
    return finish_stage(x, std::move(out), append);
}"#;
