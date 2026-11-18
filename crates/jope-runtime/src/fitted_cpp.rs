//! Standalone fitted C++17 predictor (`predict_rows` / `predict_normalized_batch`) over the
//! certified `feat_math::transpile::emit_cpp17` stage kernels plus fitted regression heads
//! (`ridge`, `ridge_gcv`, `elasticnet`, `gbm`/`histgbm`, `histgbm2`, `ensemble`/`blend`, `krr`,
//! `gpr`, and `ksmooth`). No prediction call fits a model.

use crate::error::Failure;
use crate::fitted_js::stage_only_ridge_genome;
use crate::math_pipeline::MathPipeline;
use anyhow::Result;
use feat_math::genome::GenomeIr;
use feat_math::heads::{FittedHead, GbmModel, GbmNode, RidgeModel};
use feat_math::model::Model;
use feat_math::ops::FittedOp;
use feat_math::spec::{build_plan, Plan, PlannedHead};
use feat_math::transpile::{emit_cpp17, lint_shipped_text, EmitOptions};
use std::fmt::Write as _;

fn cpp_f64(v: f64) -> Result<String> {
    if !v.is_finite() {
        return Err(Failure::integrity("non-finite f64 in fitted C++17 state"));
    }
    let mut s = format!("{v:?}");
    if !s.contains('.') && !s.contains('e') && !s.contains('E') {
        s.push_str(".0");
    }
    Ok(s)
}

fn cpp_vec_f64(vals: &[f64]) -> Result<String> {
    let mut parts = Vec::with_capacity(vals.len());
    for &v in vals {
        parts.push(cpp_f64(v)?);
    }
    Ok(format!("{{{}}}", parts.join(",")))
}

fn cpp_vec2_f64(rows: &[Vec<f64>]) -> Result<String> {
    let mut parts = Vec::with_capacity(rows.len());
    for r in rows {
        parts.push(cpp_vec_f64(r)?);
    }
    Ok(format!("{{{}}}", parts.join(",")))
}

fn cpp_vec_usize(vals: &[usize]) -> String {
    let parts: Vec<String> = vals.iter().map(|v| format!("{v}ULL")).collect();
    format!("{{{}}}", parts.join(","))
}

fn cpp_pairs(pairs: &[(usize, usize)]) -> String {
    let parts: Vec<String> = pairs
        .iter()
        .map(|(a, b)| format!("{{{a}ULL,{b}ULL}}"))
        .collect();
    format!("{{{}}}", parts.join(","))
}

fn emit_stage_init(index_1: usize, op: &FittedOp, out: &mut String) -> Result<()> {
    use FittedOp as F;
    let fld = format!("m.stage{index_1}");
    match op {
        F::Standardize { means, scales, .. } => {
            let _ = writeln!(out, "    {fld}.means = {};", cpp_vec_f64(means)?);
            let _ = writeln!(out, "    {fld}.scales = {};", cpp_vec_f64(scales)?);
        }
        F::RobustScale {
            medians, scales, ..
        } => {
            let _ = writeln!(out, "    {fld}.medians = {};", cpp_vec_f64(medians)?);
            let _ = writeln!(out, "    {fld}.scales = {};", cpp_vec_f64(scales)?);
        }
        F::Rank { sorted_cols, .. } | F::Blom { sorted_cols, .. } => {
            let _ = writeln!(
                out,
                "    {fld}.sorted_cols = {};",
                cpp_vec2_f64(sorted_cols)?
            );
        }
        F::WinsorClip { lo, hi, .. } => {
            let _ = writeln!(out, "    {fld}.lo = {};", cpp_vec_f64(lo)?);
            let _ = writeln!(out, "    {fld}.hi = {};", cpp_vec_f64(hi)?);
        }
        F::RffCos { w, b, .. } | F::RffSin { w, b, .. } => {
            let _ = writeln!(out, "    {fld}.w = {};", cpp_vec2_f64(w)?);
            let _ = writeln!(out, "    {fld}.b = {};", cpp_vec_f64(b)?);
        }
        F::Hinge { knots, .. } => {
            let _ = writeln!(out, "    {fld}.knots = {};", cpp_vec2_f64(knots)?);
        }
        F::PolyInteract { pairs, .. } | F::PairProd { pairs, .. } => {
            let _ = writeln!(out, "    {fld}.pairs = {};", cpp_pairs(pairs));
        }
        F::QuantileBins { edges, .. } | F::BinOnehot { edges, .. } => {
            let _ = writeln!(out, "    {fld}.edges = {};", cpp_vec2_f64(edges)?);
        }
        F::RandProj { w, .. } => {
            let _ = writeln!(out, "    {fld}.w = {};", cpp_vec2_f64(w)?);
        }
        F::ObliqueHinge { w, t, .. } => {
            let _ = writeln!(out, "    {fld}.w = {};", cpp_vec2_f64(w)?);
            let _ = writeln!(out, "    {fld}.t = {};", cpp_vec_f64(t)?);
        }
        F::LandmarkDist { landmarks, .. } => {
            let _ = writeln!(out, "    {fld}.landmarks = {};", cpp_vec2_f64(landmarks)?);
        }
        F::TargetBinEncode { edges, enc, .. } => {
            let _ = writeln!(out, "    {fld}.edges = {};", cpp_vec2_f64(edges)?);
            let _ = writeln!(out, "    {fld}.enc = {};", cpp_vec2_f64(enc)?);
        }
        F::Sigmoid { .. }
        | F::Step { .. }
        | F::NormRow { .. }
        | F::Cheb { .. }
        | F::RowStats { .. }
        | F::RowDiffStats { .. }
        | F::MapExpr { .. }
        | F::DeriveExpr { .. } => {}
        _ => {
            return Err(Failure::backend(
                "unsupported fitted C++17 stage state variant",
            ));
        }
    }
    Ok(())
}

fn emit_gbm_cpp_fn(fn_name: &str, g: &GbmModel) -> Result<String> {
    let mut tree_inits = Vec::with_capacity(g.trees.len());
    for tree in &g.trees {
        let mut nodes = Vec::with_capacity(tree.len());
        for node in tree {
            match node {
                GbmNode::Leaf(v) => {
                    nodes.push(format!("FittedGbmNode{{-1,0U,0ULL,0ULL,{}}}", cpp_f64(*v)?));
                }
                GbmNode::Split {
                    feature,
                    thr_bin,
                    left,
                    right,
                } => {
                    nodes.push(format!(
                        "FittedGbmNode{{static_cast<int>({feature}),static_cast<unsigned>({thr_bin}),{left}ULL,{right}ULL,0.0}}"
                    ));
                }
            }
        }
        tree_inits.push(format!("{{{}}}", nodes.join(",")));
    }
    Ok(format!(
        r#"static std::vector<double> {fn_name}(const std::vector<std::vector<double>>& x) {{
    const std::size_t n_feat = {n_feat}ULL;
    if (n_feat == 0ULL) return std::vector<double>(x.size(), {const_val});
    static const std::vector<std::vector<double>> edges = {edges};
    static const std::vector<std::vector<FittedGbmNode>> trees = {{{trees}}};
    std::vector<double> out;
    out.reserve(x.size());
    std::vector<unsigned> bins(n_feat, 0U);
    for (const auto& row : x) {{
        for (std::size_t f = 0; f < n_feat; ++f) {{
            double v = f < row.size() ? row[f] : 0.0;
            const auto& ed = edges[f];
            std::size_t lo = 0, hi = ed.size();
            while (lo < hi) {{
                std::size_t mid = lo + (hi - lo) / 2;
                if (ed[mid] < v) lo = mid + 1; else hi = mid;
            }}
            bins[f] = static_cast<unsigned>(lo);
        }}
        double acc = {init};
        for (const auto& tree : trees) {{
            std::size_t idx = 0;
            while (true) {{
                const auto& node = tree[idx];
                if (node.feature < 0) {{ acc += {lr} * node.value; break; }}
                idx = (bins[static_cast<std::size_t>(node.feature)] <= node.thr_bin) ? node.left : node.right;
            }}
        }}
        out.push_back(acc);
    }}
    return out;
}}"#,
        fn_name = fn_name,
        n_feat = g.n_feat,
        const_val = cpp_f64(g.constant_value)?,
        edges = cpp_vec2_f64(&g.edges)?,
        trees = tree_inits.join(","),
        init = cpp_f64(g.init)?,
        lr = cpp_f64(g.lr)?,
    ))
}

fn emit_ridge_cpp_fn(fn_name: &str, r: &RidgeModel) -> Result<String> {
    Ok(format!(
        r#"static std::vector<double> {fn_name}(const std::vector<std::vector<double>>& x) {{
    static const std::vector<double> means = {means};
    static const std::vector<double> scales = {scales};
    static const std::vector<double> w = {w};
    const std::size_t p = means.size();
    std::vector<double> out;
    out.reserve(x.size());
    for (const auto& row : x) {{
        double z = w[0];
        const std::size_t lim = std::min(p, row.size());
        for (std::size_t j = 0; j < lim; ++j) {{
            double xs = clamp_range((row[j] - means[j]) / scales[j], -20.0, 20.0);
            z += w[j + 1] * xs;
        }}
        out.push_back(clamp_range(z, -60.0, 60.0));
    }}
    return out;
}}"#,
        fn_name = fn_name,
        means = cpp_vec_f64(&r.means)?,
        scales = cpp_vec_f64(&r.scales)?,
        w = cpp_vec_f64(&r.w)?,
    ))
}

fn emit_head_cpp(genome: &GenomeIr, head: &FittedHead) -> Result<String> {
    let plan = build_plan(genome).map_err(|m| Failure::backend(m.join("; ")))?;
    let Plan::Pipeline {
        head: planned_head, ..
    } = &plan
    else {
        return Err(Failure::backend("expected pipeline plan"));
    };
    let node_structs = r#"struct FittedGbmNode { int feature; unsigned thr_bin; std::size_t left; std::size_t right; double value; };
struct FittedBoost2Node { int feature; double threshold; std::size_t left; std::size_t right; double value; };
"#;
    match (planned_head, head) {
        (PlannedHead::Ridge { .. }, FittedHead::Ridge(r)) => Ok(format!(
            "{node_structs}\n{}",
            emit_ridge_cpp_fn("fitted_predict_head", r)?
        )),
        (PlannedHead::RidgeGcv, FittedHead::RidgeGcv(r)) => Ok(format!(
            r#"{node_structs}
static std::vector<double> fitted_predict_head(const std::vector<std::vector<double>>& x) {{
    static const std::vector<std::size_t> cols = {cols};
    static const std::vector<double> means = {means};
    static const std::vector<double> scales = {scales};
    static const std::vector<double> w = {w};
    std::vector<double> out;
    out.reserve(x.size());
    for (const auto& row : x) {{
        double z = w[0];
        for (std::size_t k = 0; k < cols.size(); ++k) {{
            std::size_t j = cols[k];
            if (j < row.size()) z += w[k + 1] * ((row[j] - means[k]) / scales[k]);
        }}
        out.push_back(clamp_range(z, -60.0, 60.0));
    }}
    return out;
}}"#,
            cols = cpp_vec_usize(&r.cols),
            means = cpp_vec_f64(&r.means)?,
            scales = cpp_vec_f64(&r.scales)?,
            w = cpp_vec_f64(&r.w)?,
        )),
        (PlannedHead::ElasticNet { .. }, FittedHead::ElasticNet(e)) => Ok(format!(
            r#"{node_structs}
static std::vector<double> fitted_predict_head(const std::vector<std::vector<double>>& x) {{
    static const std::vector<double> means = {means};
    static const std::vector<double> scales = {scales};
    static const std::vector<double> w = {w};
    const std::size_t p = means.size();
    std::vector<double> out;
    out.reserve(x.size());
    for (const auto& row : x) {{
        double z = {w0};
        const std::size_t lim = std::min(p, row.size());
        for (std::size_t j = 0; j < lim; ++j) {{
            double xs = clamp_range((row[j] - means[j]) / scales[j], -20.0, 20.0);
            z += w[j] * xs;
        }}
        out.push_back(clamp_range(z, -60.0, 60.0));
    }}
    return out;
}}"#,
            means = cpp_vec_f64(&e.means)?,
            scales = cpp_vec_f64(&e.scales)?,
            w0 = cpp_f64(e.w0)?,
            w = cpp_vec_f64(&e.w)?,
        )),
        (PlannedHead::Gbm(_), FittedHead::Gbm(g)) => Ok(format!(
            "{node_structs}\n{}",
            emit_gbm_cpp_fn("fitted_predict_head", g)?
        )),
        (PlannedHead::Blend { .. }, FittedHead::Ensemble { ridge, gbm, w }) => {
            let r_fn = emit_ridge_cpp_fn("fitted_predict_blend_ridge", ridge)?;
            let g_fn = emit_gbm_cpp_fn("fitted_predict_blend_gbm", gbm)?;
            let wf = cpp_f64(*w)?;
            Ok(format!(
                r#"{node_structs}
{r_fn}
{g_fn}
static std::vector<double> fitted_predict_head(const std::vector<std::vector<double>>& x) {{
    auto r = fitted_predict_blend_ridge(x);
    auto g = fitted_predict_blend_gbm(x);
    std::vector<double> out(r.size(), 0.0);
    for (std::size_t i = 0; i < r.size(); ++i) out[i] = {wf} * r[i] + (1.0 - {wf}) * g[i];
    return out;
}}"#
            ))
        }
        (PlannedHead::HistGbm2 { .. }, FittedHead::HistGbm2(h)) => {
            use feat_math::boost2::BoostNode;
            let mut bag_inits = Vec::with_capacity(h.bags.len());
            for bag in &h.bags {
                let mut tree_inits = Vec::with_capacity(bag.len());
                for tree in bag {
                    let mut nodes = Vec::with_capacity(tree.len());
                    for node in tree {
                        match node {
                            BoostNode::Leaf(v) => nodes.push(format!(
                                "FittedBoost2Node{{-1,0.0,0ULL,0ULL,{}}}",
                                cpp_f64(*v)?
                            )),
                            BoostNode::Split {
                                feature,
                                threshold,
                                left,
                                right,
                            } => nodes.push(format!(
                                "FittedBoost2Node{{static_cast<int>({feature}),{thr},{left}ULL,{right}ULL,0.0}}",
                                thr = cpp_f64(*threshold)?
                            )),
                        }
                    }
                    tree_inits.push(format!("{{{}}}", nodes.join(",")));
                }
                bag_inits.push(format!("{{{}}}", tree_inits.join(",")));
            }
            Ok(format!(
                r#"{node_structs}
static std::vector<double> fitted_predict_head(const std::vector<std::vector<double>>& x) {{
    static const std::vector<std::vector<std::vector<FittedBoost2Node>>> bags = {{{bags}}};
    std::vector<double> out;
    out.reserve(x.size());
    for (const auto& row : x) {{
        if (bags.empty()) {{ out.push_back({init}); continue; }}
        double total = 0.0;
        for (const auto& bag : bags) {{
            double s = 0.0;
            for (const auto& tree : bag) {{
                std::size_t idx = 0;
                while (true) {{
                    const auto& node = tree[idx];
                    if (node.feature < 0) {{ s += node.value; break; }}
                    std::size_t f = static_cast<std::size_t>(node.feature);
                    double v = f < row.size() ? row[f] : 0.0;
                    idx = (v <= node.threshold) ? node.left : node.right;
                }}
            }}
            total += s;
        }}
        out.push_back({init} + {lr} * total / static_cast<double>(bags.size()));
    }}
    return out;
}}"#,
                bags = bag_inits.join(","),
                init = cpp_f64(h.init)?,
                lr = cpp_f64(h.lr)?,
            ))
        }
        (PlannedHead::Krr { .. }, FittedHead::Krr(k)) => Ok(format!(
            r#"{node_structs}
static std::vector<double> fitted_predict_head(const std::vector<std::vector<double>>& x) {{
    static const std::vector<double> means = {means};
    static const std::vector<double> scales = {scales};
    static const std::vector<std::vector<double>> basis = {basis};
    static const std::vector<double> alpha = {alpha};
    std::vector<double> out;
    out.reserve(x.size());
    for (const auto& row : x) {{
        if (basis.empty()) {{ out.push_back(clamp_range({ym}, -60.0, 60.0)); continue; }}
        std::size_t p = std::min(means.size(), row.size());
        std::vector<double> zr(p, 0.0);
        for (std::size_t j = 0; j < p; ++j) zr[j] = (row[j] - means[j]) / scales[j];
        double s = {ym};
        for (std::size_t b = 0; b < basis.size(); ++b) {{
            double d2 = 0.0;
            std::size_t lim = std::min(zr.size(), basis[b].size());
            for (std::size_t j = 0; j < lim; ++j) {{ double d = zr[j] - basis[b][j]; d2 += d * d; }}
            s += alpha[b] * std::exp(-{gamma} * d2);
        }}
        out.push_back(clamp_range(s, -60.0, 60.0));
    }}
    return out;
}}"#,
            means = cpp_vec_f64(&k.means)?,
            scales = cpp_vec_f64(&k.scales)?,
            basis = cpp_vec2_f64(&k.basis)?,
            alpha = cpp_vec_f64(&k.alpha)?,
            gamma = cpp_f64(k.gamma)?,
            ym = cpp_f64(k.y_mean)?,
        )),
        (PlannedHead::Gpr { .. }, FittedHead::Gpr(g)) => Ok(format!(
            r#"{node_structs}
static std::vector<double> fitted_predict_head(const std::vector<std::vector<double>>& x) {{
    static const std::vector<double> means = {means};
    static const std::vector<double> scales = {scales};
    static const std::vector<std::vector<double>> basis = {basis};
    static const std::vector<double> alpha = {alpha};
    static const std::vector<double> ls = {ls};
    std::vector<double> out;
    out.reserve(x.size());
    for (const auto& row : x) {{
        if (basis.empty()) {{ out.push_back(clamp_range({ym}, {clo}, {chi})); continue; }}
        std::size_t p = std::min(means.size(), row.size());
        std::vector<double> zr(p, 0.0);
        for (std::size_t j = 0; j < p; ++j) zr[j] = (row[j] - means[j]) / scales[j];
        double s = {ym};
        for (std::size_t b = 0; b < basis.size(); ++b) {{
            double r2 = 0.0;
            std::size_t lim = std::min(zr.size(), basis[b].size());
            for (std::size_t j = 0; j < lim; ++j) {{
                double d = zr[j] - basis[b][j];
                r2 += (d * d) / (ls[j] * ls[j]);
            }}
            s += alpha[b] * {sf2} * std::exp(-0.5 * r2);
        }}
        out.push_back(clamp_range(s, {clo}, {chi}));
    }}
    return out;
}}"#,
            means = cpp_vec_f64(&g.means)?,
            scales = cpp_vec_f64(&g.scales)?,
            basis = cpp_vec2_f64(&g.basis)?,
            alpha = cpp_vec_f64(&g.alpha)?,
            ls = cpp_vec_f64(&g.ls)?,
            sf2 = cpp_f64(g.sf2)?,
            ym = cpp_f64(g.y_mean)?,
            clo = cpp_f64(g.clamp_lo)?,
            chi = cpp_f64(g.clamp_hi)?,
        )),
        (PlannedHead::GprKernel { .. }, FittedHead::GprKernel(g)) => {
            use feat_math::gpr::KernelPrim;
            let width = g.means.len();
            let mut offset = 0usize;
            let mut kernel_cpp = String::from("double k_sum = 0.0;\n");
            for term in &g.spec {
                let amp = *g
                    .theta
                    .get(offset)
                    .ok_or_else(|| Failure::integrity("gpr_kernel theta truncated at amplitude"))?;
                let sig_lit = cpp_f64((2.0 * amp).exp())?;
                offset += 1;
                let _ = writeln!(kernel_cpp, "            {{ double term = {sig_lit};");
                for prim in term {
                    match prim {
                        KernelPrim::Rbf => {
                            let mut inv = Vec::with_capacity(width);
                            for d in 0..width {
                                let l = g
                                    .theta
                                    .get(offset + d)
                                    .ok_or_else(|| {
                                        Failure::integrity("gpr_kernel theta truncated at rbf")
                                    })?
                                    .exp();
                                inv.push(1.0 / (l * l));
                            }
                            offset += width;
                            let inv_cpp = cpp_vec_f64(&inv)?;
                            let _ = writeln!(
                                kernel_cpp,
                                "              {{ static const std::vector<double> inv = {inv_cpp}; double dist = 0.0; for (std::size_t d = 0; d < lim; ++d) {{ double q = zr[d] - basis[b][d]; dist += q * q * inv[d]; }} term *= std::exp(-0.5 * dist); }}"
                            );
                        }
                        KernelPrim::Rq => {
                            let mut inv = Vec::with_capacity(width);
                            for d in 0..width {
                                let l = g
                                    .theta
                                    .get(offset + d)
                                    .ok_or_else(|| {
                                        Failure::integrity("gpr_kernel theta truncated at rq")
                                    })?
                                    .exp();
                                inv.push(1.0 / (l * l));
                            }
                            let shape = g
                                .theta
                                .get(offset + width)
                                .ok_or_else(|| {
                                    Failure::integrity("gpr_kernel theta truncated at rq alpha")
                                })?
                                .exp();
                            offset += width + 1;
                            let inv_cpp = cpp_vec_f64(&inv)?;
                            let shape_cpp = cpp_f64(shape)?;
                            let _ = writeln!(
                                kernel_cpp,
                                "              {{ static const std::vector<double> inv = {inv_cpp}; double dist = 0.0; for (std::size_t d = 0; d < lim; ++d) {{ double q = zr[d] - basis[b][d]; dist += q * q * inv[d]; }} term *= std::exp(-{shape_cpp} * std::log(1.0 + dist / (2.0 * {shape_cpp}))); }}"
                            );
                        }
                        KernelPrim::Periodic => {
                            let period = g
                                .theta
                                .get(offset)
                                .ok_or_else(|| {
                                    Failure::integrity("gpr_kernel theta truncated at period")
                                })?
                                .exp();
                            let l = g
                                .theta
                                .get(offset + 1)
                                .ok_or_else(|| {
                                    Failure::integrity("gpr_kernel theta truncated at periodic ls")
                                })?
                                .exp();
                            offset += 2;
                            let per_cpp = cpp_f64(period)?;
                            let inv_cpp = cpp_f64(1.0 / (l * l))?;
                            let pi_cpp = cpp_f64(std::f64::consts::PI)?;
                            let _ = writeln!(
                                kernel_cpp,
                                "              {{ double s2 = 0.0; for (std::size_t d = 0; d < lim; ++d) {{ double sn = std::sin({pi_cpp} * (zr[d] - basis[b][d]) / {per_cpp}); s2 += sn * sn; }} term *= std::exp(-2.0 * s2 * {inv_cpp}); }}"
                            );
                        }
                        KernelPrim::Linear => {
                            let wf = width.max(1);
                            let _ = writeln!(
                                kernel_cpp,
                                "              {{ double prod = 0.0; for (std::size_t d = 0; d < lim; ++d) prod += zr[d] * basis[b][d]; term *= prod / {wf}.0; }}"
                            );
                        }
                    }
                }
                let _ = writeln!(kernel_cpp, "              k_sum += term; }}");
            }
            if g.theta.get(offset).is_none() {
                return Err(Failure::integrity("gpr_kernel theta truncated at noise"));
            }
            Ok(format!(
                r#"{node_structs}
static std::vector<double> fitted_predict_head(const std::vector<std::vector<double>>& x) {{
    static const std::vector<double> means = {means};
    static const std::vector<double> scales = {scales};
    static const std::vector<std::vector<double>> basis = {basis};
    static const std::vector<double> alpha = {alpha};
    std::vector<double> out;
    out.reserve(x.size());
    for (const auto& row : x) {{
        if (basis.empty()) {{ out.push_back(clamp_range({ym}, {clo}, {chi})); continue; }}
        std::size_t p = std::min(means.size(), row.size());
        std::vector<double> zr(p, 0.0);
        for (std::size_t j = 0; j < p; ++j) zr[j] = (row[j] - means[j]) / scales[j];
        double s = {ym};
        for (std::size_t b = 0; b < basis.size(); ++b) {{
            std::size_t lim = std::min(zr.size(), basis[b].size());
            {kernel_cpp}
            s += alpha[b] * k_sum;
        }}
        out.push_back(clamp_range(s, {clo}, {chi}));
    }}
    return out;
}}"#,
                means = cpp_vec_f64(&g.means)?,
                scales = cpp_vec_f64(&g.scales)?,
                basis = cpp_vec2_f64(&g.basis)?,
                alpha = cpp_vec_f64(&g.alpha)?,
                ym = cpp_f64(g.y_mean)?,
                clo = cpp_f64(g.clamp_lo)?,
                chi = cpp_f64(g.clamp_hi)?,
            ))
        }
        (PlannedHead::Stack, FittedHead::Stack(s)) => {
            let mut helpers = String::new();
            let mut calls = Vec::new();
            if let Some(ref r) = s.ridge_a {
                let _ = writeln!(helpers, "{}", emit_ridge_cpp_fn("fitted_stack_m0", r)?);
                calls.push(("fitted_stack_m0", s.weights.first().copied().unwrap_or(0.0)));
            }
            if let Some(ref r) = s.ridge_b {
                let _ = writeln!(helpers, "{}", emit_ridge_cpp_fn("fitted_stack_m1", r)?);
                calls.push(("fitted_stack_m1", s.weights.get(1).copied().unwrap_or(0.0)));
            }
            if let Some(ref e) = s.net {
                let _ = writeln!(
                    helpers,
                    r#"static std::vector<double> fitted_stack_m2(const std::vector<std::vector<double>>& x) {{
    static const std::vector<double> means = {};
    static const std::vector<double> scales = {};
    static const std::vector<double> w = {};
    std::vector<double> out; out.reserve(x.size());
    for (const auto& row : x) {{
        double z = {}; std::size_t lim = std::min(means.size(), row.size());
        for (std::size_t j = 0; j < lim; ++j) z += w[j] * clamp_range((row[j] - means[j]) / scales[j], -20.0, 20.0);
        out.push_back(clamp_range(z, -60.0, 60.0));
    }}
    return out;
}}"#,
                    cpp_vec_f64(&e.means)?,
                    cpp_vec_f64(&e.scales)?,
                    cpp_vec_f64(&e.w)?,
                    cpp_f64(e.w0)?,
                );
                calls.push(("fitted_stack_m2", s.weights.get(2).copied().unwrap_or(0.0)));
            }
            if let Some(ref h) = s.trees {
                let sub_plan = GenomeIr::Pipeline(feat_math::dsl::DslExpr {
                    stages: vec![feat_math::dsl::Stage {
                        op: "histgbm2".to_owned(),
                        params: std::collections::BTreeMap::new(),
                    }],
                });
                let tree_cpp = emit_head_cpp(&sub_plan, &FittedHead::HistGbm2(h.clone()))?
                    .replace(node_structs, "")
                    .replace("fitted_predict_head(", "fitted_stack_m3(");
                let _ = writeln!(helpers, "{tree_cpp}");
                calls.push(("fitted_stack_m3", s.weights.get(3).copied().unwrap_or(0.0)));
            }
            if let Some(ref k) = s.kernel {
                let _ = writeln!(
                    helpers,
                    r#"static std::vector<double> fitted_stack_m4(const std::vector<std::vector<double>>& x) {{
    static const std::vector<double> means = {};
    static const std::vector<double> scales = {};
    static const std::vector<std::vector<double>> basis = {};
    static const std::vector<double> alpha = {};
    std::vector<double> out; out.reserve(x.size());
    for (const auto& row : x) {{
        if (basis.empty()) {{ out.push_back(clamp_range({}, -60.0, 60.0)); continue; }}
        std::size_t p = std::min(means.size(), row.size());
        std::vector<double> zr(p, 0.0);
        for (std::size_t j = 0; j < p; ++j) zr[j] = (row[j] - means[j]) / scales[j];
        double sv = {};
        for (std::size_t b = 0; b < basis.size(); ++b) {{
            double d2 = 0.0; std::size_t lim = std::min(zr.size(), basis[b].size());
            for (std::size_t j = 0; j < lim; ++j) {{ double d = zr[j] - basis[b][j]; d2 += d * d; }}
            sv += alpha[b] * std::exp(-{} * d2);
        }}
        out.push_back(clamp_range(sv, -60.0, 60.0));
    }}
    return out;
}}"#,
                    cpp_vec_f64(&k.means)?,
                    cpp_vec_f64(&k.scales)?,
                    cpp_vec2_f64(&k.basis)?,
                    cpp_vec_f64(&k.alpha)?,
                    cpp_f64(k.y_mean)?,
                    cpp_f64(k.y_mean)?,
                    cpp_f64(k.gamma)?,
                );
                calls.push(("fitted_stack_m4", s.weights.get(4).copied().unwrap_or(0.0)));
            }
            let mut sum_body = String::new();
            for (fn_name, w) in calls {
                let wl = cpp_f64(w)?;
                let _ = writeln!(
                    sum_body,
                    "    {{ auto pv = {fn_name}(x); for (std::size_t i = 0; i < x.size(); ++i) out[i] += {wl} * pv[i]; }}"
                );
            }
            Ok(format!(
                r#"{node_structs}
{helpers}
static std::vector<double> fitted_predict_head(const std::vector<std::vector<double>>& x) {{
    std::vector<double> out(x.size(), 0.0);
{sum_body}    return out;
}}"#
            ))
        }
        (PlannedHead::Ksmooth { k, gamma }, FittedHead::Ksmooth(m)) => {
            if *k != m.k || !(m.gamma.is_finite() && m.gamma.to_bits() == gamma.to_bits()) {
                return Err(Failure::backend(
                    "ksmooth fitted parameters do not match planned head",
                ));
            }
            Ok(format!(
                r#"{node_structs}
static std::vector<double> fitted_predict_head(const std::vector<std::vector<double>>& x) {{
    static const std::vector<double> means = {means};
    static const std::vector<double> scales = {scales};
    static const std::vector<std::vector<double>> z = {z};
    static const std::vector<double> ys = {ys};
    const std::size_t p = means.size();
    const double pf = static_cast<double>(p);
    std::vector<double> out;
    out.reserve(x.size());
    for (const auto& row : x) {{
        if (z.empty()) {{ out.push_back(clamp_range({ym}, -60.0, 60.0)); continue; }}
        std::size_t lim_p = std::min(p, row.size());
        std::vector<double> zr(lim_p, 0.0);
        for (std::size_t j = 0; j < lim_p; ++j) zr[j] = (row[j] - means[j]) / scales[j];
        std::vector<std::pair<double, std::size_t>> dists(z.size());
        for (std::size_t i = 0; i < z.size(); ++i) {{
            double d2 = 0.0;
            std::size_t lim = std::min(zr.size(), z[i].size());
            for (std::size_t j = 0; j < lim; ++j) {{ double d = zr[j] - z[i][j]; d2 += d * d; }}
            dists[i] = {{d2, i}};
        }}
        std::sort(dists.begin(), dists.end());
        std::size_t kk = std::min<std::size_t>({k}ULL, dists.size());
        double base = dists[0].first, num = 0.0, den = 0.0;
        for (std::size_t i = 0; i < kk; ++i) {{
            double w = std::exp(-{gamma} * (dists[i].first - base) / pf);
            num += w * ys[dists[i].second];
            den += w;
        }}
        out.push_back(clamp_range(num / den, -60.0, 60.0));
    }}
    return out;
}}"#,
                means = cpp_vec_f64(&m.means)?,
                scales = cpp_vec_f64(&m.scales)?,
                z = cpp_vec2_f64(&m.z)?,
                ys = cpp_vec_f64(&m.ys)?,
                k = m.k,
                gamma = cpp_f64(m.gamma)?,
                ym = cpp_f64(m.y_mean)?,
            ))
        }
        _ => Err(Failure::backend(
            "C++17 fitted exporter does not yet support this head variant",
        )),
    }
}

/// Emit standalone fitted C++17 source for `pipeline` with optional per-column `missing_fills`
/// and optional post-hoc `lower_clamp`.
pub fn emit_math_pipeline_cpp17(
    pipeline: &MathPipeline,
    genome: &GenomeIr,
    missing_fills: Option<&[f64]>,
    lower_clamp: Option<f64>,
) -> Result<String> {
    let Model::Pipeline {
        target: None,
        ops,
        head,
    } = &pipeline.state.model
    else {
        return Err(Failure::backend(
            "fitted C++17 requires a pipeline without a target transform",
        ));
    };
    let width = pipeline.input.selected.len();
    feat_math::kernel::width::check_input_width(&pipeline.state.model, width)
        .map_err(|e| Failure::integrity(format!("fitted C++17 input width: {e}")))?;

    let stage_genome = stage_only_ridge_genome(genome)
        .ok_or_else(|| Failure::backend("fitted C++17 requires a pipeline genome"))?;
    let base_cpp = emit_cpp17(&stage_genome, &EmitOptions::default())
        .map_err(|e| Failure::backend(format!("{e:#}")))?;

    // Extract the stage-application lines emitted inside `Model::predict` prior to `return ridge_regression_predict`.
    let (_, after_pred_sig) = base_cpp
        .split_once(
            "std::vector<double> Model::predict(const std::vector<std::vector<double>>& x_in) const {\n",
        )
        .ok_or_else(|| Failure::integrity("C++17 recipe missing Model::predict signature"))?;
    let (stage_apply_block, _) = after_pred_sig
        .split_once("    return ridge_regression_predict(")
        .ok_or_else(|| Failure::integrity("C++17 recipe missing ridge_regression_predict tail"))?;
    let stage_apply_bound = rewrite_stage_refs_to_member(stage_apply_block, ops.len());

    let mut stage_inits = String::new();
    if stage_apply_bound.contains("input_width") {
        let _ = writeln!(stage_inits, "    m.input_width = {width}ULL;");
    }
    for (idx, op) in ops.iter().enumerate() {
        emit_stage_init(idx + 1, op, &mut stage_inits)?;
    }

    let head_cpp = emit_head_cpp(genome, head)?;
    let cols_cpp: Vec<String> = pipeline
        .input
        .columns
        .iter()
        .map(|c| serde_json::to_string(&c.name).map_err(|e| Failure::integrity(e.to_string())))
        .collect::<Result<_>>()?;
    let mins: Vec<f64> = pipeline.input.columns.iter().map(|c| c.min).collect();
    let maxs: Vec<f64> = pipeline.input.columns.iter().map(|c| c.max).collect();
    let consts: Vec<String> = pipeline
        .input
        .columns
        .iter()
        .map(|c| if c.constant { "true" } else { "false" }.to_owned())
        .collect();
    let has_fills = missing_fills.is_some();
    let fills_vec: Vec<f64> = match missing_fills {
        Some(f) => {
            if f.len() != pipeline.input.columns.len() {
                return Err(Failure::integrity("missing_fills width mismatch"));
            }
            f.to_vec()
        }
        None => vec![0.0; pipeline.input.columns.len()],
    };
    let lower_expr = match lower_clamp {
        Some(lb) => format!("std::max(value, {})", cpp_f64(lb)?),
        None => "value".to_owned(),
    };

    let mut full = base_cpp;
    let _ = write!(
        full,
        r#"
// Standalone fitted predictor state (inference only; zero external dependencies).
{head_cpp}

static const Model& fitted_stage_model() {{
    static const Model instance = []() {{
        Model m;
{stage_inits}        return m;
    }}();
    return instance;
}}

static std::vector<double> predict_normalized_batch(const std::vector<std::vector<double>>& x_in) {{
    if (x_in.empty()) return {{}};
    const Model& m = fitted_stage_model();
    (void)m;
{stage_apply_bound}    return fitted_predict_head(x);
}}

inline const std::vector<std::string>& model_columns() {{
    static const std::vector<std::string> cols = {{{cols_joined}}};
    return cols;
}}

inline std::vector<double> predict_rows(const std::vector<std::vector<double>>& rows) {{
    static const std::vector<double> col_min = {col_min};
    static const std::vector<double> col_max = {col_max};
    static const std::vector<double> col_fill = {col_fill};
    static const std::vector<bool> col_const = {{{col_const}}};
    static const std::vector<std::size_t> selected = {selected};
    const std::size_t expected_cols = col_min.size();
    if (rows.empty()) return {{}};
    std::vector<std::vector<double>> x_norm;
    x_norm.reserve(rows.size());
    for (std::size_t r = 0; r < rows.size(); ++r) {{
        if (rows[r].size() != expected_cols) {{
            throw std::invalid_argument("row width does not match model_columns()");
        }}
        std::vector<double> enc;
        enc.reserve(selected.size());
        for (std::size_t k = 0; k < selected.size(); ++k) {{
            std::size_t j = selected[k];
            double raw = rows[r][j];
            if (!std::isfinite(raw)) {{
                if ({has_fills}) {{
                    raw = col_fill[j];
                }} else {{
                    enc.push_back(0.0);
                    continue;
                }}
            }}
            if (col_const[j] || col_max[j] == col_min[j]) {{
                enc.push_back(0.0);
            }} else {{
                double scaled = clamp_range((raw - col_min[j]) / (col_max[j] - col_min[j]), 0.0, 1.0);
                if (!std::isfinite(scaled)) throw std::runtime_error("non-finite normalized input");
                enc.push_back(scaled);
            }}
        }}
        x_norm.push_back(std::move(enc));
    }}
    std::vector<double> raw_preds = predict_normalized_batch(x_norm);
    const double t_min = {t_min};
    const double t_scale = {t_scale};
    std::vector<double> out;
    out.reserve(raw_preds.size());
    for (double p : raw_preds) {{
        if (!std::isfinite(p)) throw std::runtime_error("non-finite fitted prediction");
        double value = (t_scale <= 1e-12) ? t_min : (t_min + p * t_scale);
        value = {lower_expr};
        if (!std::isfinite(value)) throw std::runtime_error("non-finite target prediction");
        out.push_back(value);
    }}
    return out;
}}

inline double predict_row(const std::vector<double>& row) {{
    auto out = predict_rows({{row}});
    return out[0];
}}
"#,
        cols_joined = cols_cpp.join(","),
        col_min = cpp_vec_f64(&mins)?,
        col_max = cpp_vec_f64(&maxs)?,
        col_fill = cpp_vec_f64(&fills_vec)?,
        col_const = consts.join(","),
        selected = cpp_vec_usize(&pipeline.input.selected),
        has_fills = if has_fills { "true" } else { "false" },
        t_min = cpp_f64(pipeline.input.target.min)?,
        t_scale = cpp_f64(pipeline.input.target.scale)?,
    );

    let hits = lint_shipped_text(&full);
    if !hits.is_empty() {
        return Err(Failure::integrity(format!(
            "banned vocabulary in fitted C++17 artifact: {}",
            hits.join(", ")
        )));
    }
    Ok(full)
}

fn rewrite_stage_refs_to_member(block: &str, stage_count: usize) -> String {
    let mut out = block.replace("!= input_width)", "!= m.input_width)");
    for idx in (1..=stage_count).rev() {
        let plain = format!("(stage{idx},");
        let member = format!("(m.stage{idx},");
        out = out.replace(&plain, &member);
    }
    out
}
