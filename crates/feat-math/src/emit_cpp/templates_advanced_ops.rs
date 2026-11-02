pub(crate) const H_QUANTILE_EDGES: &str = r#"// Interior quantile cut points (1/bins .. (bins-1)/bins) for every column.
static std::vector<std::vector<double>> quantile_edges(const std::vector<std::vector<double>>& x, std::size_t bins) {
    std::vector<std::vector<double>> edges;
    for (std::size_t j = 0; j < x[0].size(); ++j) {
        std::vector<double> s = sorted_column(x, j);
        std::vector<double> col;
        for (std::size_t b = 1; b < bins; ++b) {
            col.push_back(quantile(s, static_cast<double>(b) / static_cast<double>(bins)));
        }
        edges.push_back(std::move(col));
    }
    return edges;
}"#;

pub(crate) const H_PEARSON: &str = r#"// Pearson correlation, 0.0 for degenerate inputs.
static double pearson(const std::vector<double>& a, const std::vector<double>& b) {
    std::size_t n = std::min(a.size(), b.size());
    if (n < 2) return 0.0;
    double ma = 0.0;
    for (std::size_t i = 0; i < n; ++i) ma += a[i];
    ma /= static_cast<double>(n);
    double mb = 0.0;
    for (std::size_t i = 0; i < n; ++i) mb += b[i];
    mb /= static_cast<double>(n);
    double sab = 0.0;
    double saa = 0.0;
    double sbb = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double da = a[i] - ma;
        double db = b[i] - mb;
        sab += da * db;
        saa += da * da;
        sbb += db * db;
    }
    if (saa <= 0.0 || sbb <= 0.0) return 0.0;
    return sab / std::sqrt(saa * sbb);
}"#;

pub(crate) const OP_CHEB: &str = r#"// Chebyshev polynomial basis (T_1..T_degree) of each column, mapped to [-1, 1].
static std::vector<std::vector<double>> chebyshev_basis_apply(const std::vector<std::vector<double>>& x, std::size_t degree, bool append) {
    std::vector<std::vector<double>> out;
    out.reserve(x.size());
    for (const auto& row : x) {
        std::vector<double> values;
        for (double v : row) {
            double u = clamp_range(2.0 * v - 1.0, -1.0, 1.0);
            double t_prev = 1.0;
            double t_cur = u;
            for (std::size_t d = 0; d < degree; ++d) {
                values.push_back(t_cur);
                double t_next = 2.0 * u * t_cur - t_prev;
                t_prev = t_cur;
                t_cur = t_next;
            }
        }
        out.push_back(std::move(values));
    }
    return finish_stage(x, std::move(out), append);
}"#;

pub(crate) const OP_HINGE: &str = r#"struct HingeBasis {
    std::vector<std::vector<double>> knots;
};

// Piecewise-linear hinge features at per-column interior quantile knots.
static HingeBasis hinge_basis_fit(const std::vector<std::vector<double>>& x, std::size_t knots) {
    HingeBasis state;
    for (std::size_t j = 0; j < x[0].size(); ++j) {
        std::vector<double> s = sorted_column(x, j);
        std::vector<double> fitted;
        for (std::size_t i = 1; i < knots + 1; ++i) {
            fitted.push_back(quantile(s, static_cast<double>(i) / static_cast<double>(knots + 1)));
        }
        state.knots.push_back(std::move(fitted));
    }
    return state;
}

static std::vector<std::vector<double>> hinge_basis_apply(const HingeBasis& state, const std::vector<std::vector<double>>& x, bool append) {
    std::vector<std::vector<double>> out;
    out.reserve(x.size());
    for (const auto& row : x) {
        std::vector<double> values;
        for (std::size_t j = 0; j < row.size(); ++j) {
            for (double t : state.knots[j]) {
                values.push_back(max_num(row[j] - t, 0.0));
            }
        }
        out.push_back(std::move(values));
    }
    return finish_stage(x, std::move(out), append);
}"#;

pub(crate) const OP_POLY_INTERACT: &str = r#"struct PairwiseProducts {
    std::vector<std::pair<std::size_t, std::size_t>> pairs;
};

// Products of every column with itself and each other column, capped at 512.
static PairwiseProducts pairwise_products_fit(const std::vector<std::vector<double>>& x) {
    std::size_t cols = x[0].size();
    PairwiseProducts state;
    for (std::size_t i = 0; i < cols; ++i) state.pairs.push_back(std::make_pair(i, i));
    bool full = false;
    for (std::size_t i = 0; i < cols; ++i) {
        if (full) break;
        for (std::size_t j = i + 1; j < cols; ++j) {
            if (state.pairs.size() >= 512) {
                full = true;
                break;
            }
            state.pairs.push_back(std::make_pair(i, j));
        }
    }
    return state;
}

static std::vector<std::vector<double>> pairwise_products_apply(const PairwiseProducts& state, const std::vector<std::vector<double>>& x, bool append) {
    std::vector<std::vector<double>> out;
    out.reserve(x.size());
    for (const auto& row : x) {
        std::vector<double> values;
        values.reserve(state.pairs.size());
        for (const auto& pr : state.pairs) values.push_back(row[pr.first] * row[pr.second]);
        out.push_back(std::move(values));
    }
    return finish_stage(x, std::move(out), append);
}"#;

pub(crate) const OP_QUANTILE_BINS: &str = r#"struct QuantileBinIndexer {
    std::vector<std::vector<double>> edges;
};

// Replaces each value with its quantile bin index, scaled to [0, 1].
static QuantileBinIndexer quantile_bin_indexer_fit(const std::vector<std::vector<double>>& x, std::size_t bins) {
    QuantileBinIndexer state;
    state.edges = quantile_edges(x, bins);
    return state;
}

static std::vector<std::vector<double>> quantile_bin_indexer_apply(const QuantileBinIndexer& state, const std::vector<std::vector<double>>& x, std::size_t bins, bool append) {
    std::vector<std::vector<double>> out;
    out.reserve(x.size());
    for (const auto& row : x) {
        std::vector<double> r;
        r.reserve(row.size());
        for (std::size_t j = 0; j < row.size(); ++j) {
            r.push_back(static_cast<double>(lower_bound(state.edges[j], row[j])) / (static_cast<double>(bins) - 1.0));
        }
        out.push_back(std::move(r));
    }
    return finish_stage(x, std::move(out), append);
}"#;

pub(crate) const OP_BIN_ONEHOT: &str = r#"struct QuantileOnehotEncoder {
    std::vector<std::vector<double>> edges;
};

// One-hot encodes each value's quantile bin (bins columns per input column).
static QuantileOnehotEncoder quantile_onehot_encoder_fit(const std::vector<std::vector<double>>& x, std::size_t bins) {
    QuantileOnehotEncoder state;
    state.edges = quantile_edges(x, bins);
    return state;
}

static std::vector<std::vector<double>> quantile_onehot_encoder_apply(const QuantileOnehotEncoder& state, const std::vector<std::vector<double>>& x, std::size_t bins, bool append) {
    std::vector<std::vector<double>> out;
    out.reserve(x.size());
    for (const auto& row : x) {
        std::vector<double> encoded(row.size() * bins, 0.0);
        for (std::size_t j = 0; j < row.size(); ++j) {
            std::size_t b = std::min(lower_bound(state.edges[j], row[j]), bins - 1);
            encoded[j * bins + b] = 1.0;
        }
        out.push_back(std::move(encoded));
    }
    return finish_stage(x, std::move(out), append);
}"#;

pub(crate) const OP_PAIR_PROD: &str = r#"struct CorrelationScreenedProducts {
    std::vector<std::pair<std::size_t, std::size_t>> pairs;
};

// Products of the top-k columns most correlated with the target (by |Pearson|).
static CorrelationScreenedProducts correlation_screened_products_fit(const std::vector<std::vector<double>>& x, const std::vector<double>& y, std::size_t top_k) {
    std::size_t cols = x[0].size();
    top_k = std::min(top_k, cols);
    std::vector<std::pair<std::size_t, double>> scored;
    scored.reserve(cols);
    for (std::size_t j = 0; j < cols; ++j) {
        std::vector<double> col;
        col.reserve(x.size());
        for (const auto& row : x) col.push_back(row[j]);
        scored.push_back(std::make_pair(j, std::fabs(pearson(col, y))));
    }
    std::sort(scored.begin(), scored.end(),
        [](const std::pair<std::size_t, double>& a, const std::pair<std::size_t, double>& b) {
            if (total_less(b.second, a.second)) return true;
            if (total_less(a.second, b.second)) return false;
            return a.first < b.first;
        });
    std::vector<std::size_t> selected;
    selected.reserve(top_k);
    for (std::size_t i = 0; i < top_k; ++i) selected.push_back(scored[i].first);
    std::sort(selected.begin(), selected.end());
    CorrelationScreenedProducts state;
    for (std::size_t a = 0; a < selected.size(); ++a) {
        for (std::size_t b = a + 1; b < selected.size(); ++b) {
            state.pairs.push_back(std::make_pair(selected[a], selected[b]));
        }
    }
    return state;
}

static std::vector<std::vector<double>> correlation_screened_products_apply(const CorrelationScreenedProducts& state, const std::vector<std::vector<double>>& x, bool append) {
    std::vector<std::vector<double>> out;
    out.reserve(x.size());
    for (const auto& row : x) {
        std::vector<double> values;
        values.reserve(state.pairs.size());
        for (const auto& pr : state.pairs) values.push_back(row[pr.first] * row[pr.second]);
        out.push_back(std::move(values));
    }
    return finish_stage(x, std::move(out), append);
}"#;

pub(crate) const H_RBF_ROW: &str = r#"// Gaussian kernel similarities exp(-gamma * ||row - l||^2) to each landmark.
static std::vector<double> rbf_row(const std::vector<std::vector<double>>& landmarks, const std::vector<double>& row, double gamma) {
    std::vector<double> values;
    values.reserve(landmarks.size());
    for (const auto& landmark : landmarks) {
        double d2 = 0.0;
        std::size_t lim = std::min(landmark.size(), row.size());
        for (std::size_t j = 0; j < lim; ++j) {
            double d = row[j] - landmark[j];
            d2 += d * d;
        }
        values.push_back(std::exp(-gamma * d2));
    }
    return values;
}"#;

pub(crate) const OP_RAND_PROJ: &str = r#"struct GaussianRandomProjection {
    std::vector<std::vector<double>> w;
};

// Projects each row onto k random Gaussian directions, scaled by 1/sqrt(k) (Johnson & Lindenstrauss, 1984).
static GaussianRandomProjection gaussian_random_projection_fit(const std::vector<std::vector<double>>& x, std::size_t k, std::uint64_t seed) {
    std::size_t cols = x[0].size();
    Pcg32 rng = stage_rng(seed, cols, k);
    GaussianRandomProjection state;
    state.w.reserve(k);
    for (std::size_t i = 0; i < k; ++i) {
        std::vector<double> row;
        row.reserve(cols);
        for (std::size_t j = 0; j < cols; ++j) {
            row.push_back(rng.gaussian());
        }
        state.w.push_back(std::move(row));
    }
    return state;
}

static std::vector<std::vector<double>> gaussian_random_projection_apply(const GaussianRandomProjection& state, const std::vector<std::vector<double>>& x, bool append) {
    double inv = 1.0 / std::sqrt(static_cast<double>(state.w.size()));
    std::vector<std::vector<double>> out;
    out.reserve(x.size());
    for (const auto& row : x) {
        std::vector<double> values;
        values.reserve(state.w.size());
        for (std::size_t i = 0; i < state.w.size(); ++i) {
            values.push_back(dot(state.w[i], row) * inv);
        }
        out.push_back(std::move(values));
    }
    return finish_stage(x, std::move(out), append);
}"#;

pub(crate) const OP_ROW_STATS: &str = r#"// Per-row summary [mean, std, min, max, median, q25, q75, skewness, excess kurtosis, mean energy].
static std::vector<std::vector<double>> row_summary_statistics_apply(const std::vector<std::vector<double>>& x, bool append) {
    std::vector<std::vector<double>> out;
    out.reserve(x.size());
    for (const auto& row : x) {
        double n = static_cast<double>(row.size());
        double mean = 0.0;
        for (double v : row) mean += v;
        mean /= n;
        double m2 = 0.0;
        double m3 = 0.0;
        double m4 = 0.0;
        double energy = 0.0;
        double mn = std::numeric_limits<double>::infinity();
        double mx = -std::numeric_limits<double>::infinity();
        for (double v : row) {
            double d = v - mean;
            m2 += d * d;
            m3 += d * d * d;
            m4 += d * d * d * d;
            energy += v * v;
            if (v < mn) mn = v;
            if (v > mx) mx = v;
        }
        m2 /= n;
        m3 /= n;
        m4 /= n;
        energy /= n;
        double sd = std::sqrt(m2);
        double skew = (m2 > 1e-24) ? m3 / (m2 * sd) : 0.0;
        double kurt = (m2 > 1e-24) ? m4 / (m2 * m2) - 3.0 : 0.0;
        std::vector<double> s = row;
        std::sort(s.begin(), s.end(), total_less);
        out.push_back({mean, sd, mn, mx, quantile(s, 0.5), quantile(s, 0.25), quantile(s, 0.75), skew, kurt, energy});
    }
    return finish_stage(x, std::move(out), append);
}"#;

pub(crate) const OP_ROW_DIFF_STATS: &str = r#"// Per-row first-difference summary [mean, std, min, max, mean |diff|, sign-change rate, lag-1 autocorrelation, mean energy].
static std::vector<std::vector<double>> row_difference_statistics_apply(const std::vector<std::vector<double>>& x, bool append) {
    std::vector<std::vector<double>> out;
    out.reserve(x.size());
    for (const auto& row : x) {
        std::size_t d = row.size();
        if (d < 2) {
            out.push_back(std::vector<double>(8, 0.0));
            continue;
        }
        std::size_t nd = d - 1;
        std::vector<double> diffs;
        diffs.reserve(nd);
        for (std::size_t j = 0; j < nd; ++j) diffs.push_back(row[j + 1] - row[j]);
        double mean = 0.0;
        for (double v : diffs) mean += v;
        mean /= static_cast<double>(nd);
        double m2 = 0.0;
        double mean_abs = 0.0;
        double energy = 0.0;
        double mn = std::numeric_limits<double>::infinity();
        double mx = -std::numeric_limits<double>::infinity();
        std::size_t signs = 0;
        for (std::size_t j = 0; j < nd; ++j) {
            double v = diffs[j];
            double dd = v - mean;
            m2 += dd * dd;
            mean_abs += std::fabs(v);
            energy += v * v;
            if (v < mn) mn = v;
            if (v > mx) mx = v;
            if (j > 0 && diffs[j - 1] * v < 0.0) signs += 1;
        }
        double sd = std::sqrt(m2 / static_cast<double>(nd));
        std::vector<double> a(row.begin(), row.begin() + nd);
        std::vector<double> b(row.begin() + 1, row.end());
        double autoc = pearson(a, b);
        out.push_back({mean, sd, mn, mx, mean_abs / static_cast<double>(nd),
            static_cast<double>(signs) / static_cast<double>(nd), autoc, energy / static_cast<double>(nd)});
    }
    return finish_stage(x, std::move(out), append);
}"#;

pub(crate) const OP_OBLIQUE_HINGE: &str = r#"struct ObliqueHingeFeatures {
    std::vector<std::vector<double>> w;
    std::vector<double> t;
};

// Rectified random projections thresholded at each direction's median training projection.
static ObliqueHingeFeatures oblique_hinge_features_fit(const std::vector<std::vector<double>>& x, std::size_t n_features, std::uint64_t seed) {
    std::size_t cols = x[0].size();
    Pcg32 rng = stage_rng(seed, cols, n_features);
    double inv = 1.0 / std::sqrt(static_cast<double>(cols));
    ObliqueHingeFeatures state;
    state.w.reserve(n_features);
    for (std::size_t i = 0; i < n_features; ++i) {
        std::vector<double> row;
        row.reserve(cols);
        for (std::size_t j = 0; j < cols; ++j) {
            row.push_back(rng.gaussian() * inv);
        }
        state.w.push_back(std::move(row));
    }
    for (const auto& wi : state.w) {
        std::vector<double> proj;
        proj.reserve(x.size());
        for (const auto& row : x) proj.push_back(dot(wi, row));
        std::sort(proj.begin(), proj.end(), total_less);
        state.t.push_back(quantile(proj, 0.5));
    }
    return state;
}

static std::vector<std::vector<double>> oblique_hinge_features_apply(const ObliqueHingeFeatures& state, const std::vector<std::vector<double>>& x, bool append) {
    std::vector<std::vector<double>> out;
    out.reserve(x.size());
    for (const auto& row : x) {
        std::vector<double> values;
        values.reserve(state.w.size());
        for (std::size_t i = 0; i < state.w.size(); ++i) {
            values.push_back(max_num(dot(state.w[i], row) - state.t[i], 0.0));
        }
        out.push_back(std::move(values));
    }
    return finish_stage(x, std::move(out), append);
}"#;

pub(crate) const OP_LANDMARK_DIST: &str = r#"struct GaussianLandmarkFeatures {
    std::vector<std::vector<double>> landmarks;
};

// Gaussian kernel similarity to k fixed random landmarks (Williams & Seeger, 2001).
static GaussianLandmarkFeatures gaussian_landmark_features_fit(const std::vector<std::vector<double>>& x, std::size_t k, std::uint64_t seed) {
    std::size_t cols = x[0].size();
    Pcg32 rng = stage_rng(seed, cols, k);
    GaussianLandmarkFeatures state;
    state.landmarks.reserve(k);
    for (std::size_t i = 0; i < k; ++i) {
        std::vector<double> row;
        row.reserve(cols);
        for (std::size_t j = 0; j < cols; ++j) {
            row.push_back(rng.range_f64(0.0, 1.0));
        }
        state.landmarks.push_back(std::move(row));
    }
    return state;
}

static std::vector<std::vector<double>> gaussian_landmark_features_apply(const GaussianLandmarkFeatures& state, const std::vector<std::vector<double>>& x, double gamma, bool append) {
    std::vector<std::vector<double>> out;
    out.reserve(x.size());
    for (const auto& row : x) out.push_back(rbf_row(state.landmarks, row, gamma));
    return finish_stage(x, std::move(out), append);
}"#;

pub(crate) const OP_TARGET_BIN_ENCODE: &str = r#"struct SmoothedTargetBinEncoder {
    std::vector<std::vector<double>> edges;
    std::vector<std::vector<double>> enc;
};

// Smoothed target (mean-of-y) encoding per quantile bin (Micci-Barreca, 2001).
static SmoothedTargetBinEncoder smoothed_target_bin_encoder_fit(const std::vector<std::vector<double>>& x, const std::vector<double>& y, std::size_t bins, double prior) {
    SmoothedTargetBinEncoder state;
    state.edges = quantile_edges(x, bins);
    double y_mean = 0.0;
    for (double v : y) y_mean += v;
    y_mean /= static_cast<double>(std::max<std::size_t>(y.size(), 1));
    for (std::size_t j = 0; j < x[0].size(); ++j) {
        std::vector<double> sums(bins, 0.0);
        std::vector<double> counts(bins, 0.0);
        for (std::size_t r = 0; r < x.size(); ++r) {
            std::size_t b = std::min(lower_bound(state.edges[j], x[r][j]), bins - 1);
            sums[b] += y[r];
            counts[b] += 1.0;
        }
        std::vector<double> col;
        col.reserve(bins);
        for (std::size_t b = 0; b < bins; ++b) {
            col.push_back((sums[b] + prior * y_mean) / max_num(counts[b] + prior, 1e-9));
        }
        state.enc.push_back(std::move(col));
    }
    return state;
}

static std::vector<std::vector<double>> smoothed_target_bin_encoder_apply(const SmoothedTargetBinEncoder& state, const std::vector<std::vector<double>>& x, bool append) {
    std::vector<std::vector<double>> out;
    out.reserve(x.size());
    for (const auto& row : x) {
        std::vector<double> values;
        values.reserve(row.size());
        for (std::size_t j = 0; j < row.size(); ++j) {
            std::size_t bins = state.enc[j].size();
            values.push_back(state.enc[j][std::min(lower_bound(state.edges[j], row[j]), bins - 1)]);
        }
        out.push_back(std::move(values));
    }
    return finish_stage(x, std::move(out), append);
}"#;
