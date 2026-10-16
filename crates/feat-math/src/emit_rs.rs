//! Rust emitter: a single std-only file that compiles with plain `rustc -O
//! model.rs` (edition 2015 defaults — no `use` statements, no external
//! crates) and mirrors the Python artifact line for line in structure.

use crate::mathexpr::fmt_f64;
// B05–B07: scalar-v2 recipe assembly.
use crate::kernel::scalar_program::emit::{self as scalar_emit, ScalarLanguage};
use crate::spec::{
    doc_paragraphs, head_fn_name, helper_set, op_fn_name, op_struct_name, target_fn_name,
    target_struct_name, wrap, Arena, Helper, Plan, PlannedHead, PlannedOp, PlannedStage,
    PlannedTarget, TargetKind,
};
use crate::transpile::EmitOptions;

pub fn emit_rust(plan: &Plan, opts: &EmitOptions) -> String {
    // B05–B07: retain the explicit old-operation numerical contract.
    assert!(
        !scalar_emit::raw_input(plan)
            || opts.math_kernel == crate::transpile::MathKernel::PlatformLibm,
        "scalar-v2 retained operations require PlatformLibm"
    );
    // One blank line between top-level items; single pre-reserved buffer.
    let mut out = Arena::new("\n\n", 24 * 1024);
    out.push(&module_doc(plan, opts));
    let has_formula = match plan {
        Plan::Symbolic { .. } => true,
        Plan::Pipeline { stages, .. } => stages.iter().any(|s| {
            matches!(
                s.op,
                PlannedOp::MapExpr { .. }
                    | PlannedOp::DeriveExpr { .. }
                    | PlannedOp::ScalarV2 { .. }
            )
        }),
    };
    // `score` is a public convenience method that `main` never calls; allow the
    // dead-code path so the artifact compiles warning-free.
    if has_formula {
        // Generated formulas are fully parenthesized for unambiguous
        // precedence; keep rustc quiet about the redundant pairs.
        out.push("#![forbid(unsafe_code)]\n#![allow(dead_code)]\n#![allow(unused_parens)]");
    } else {
        out.push("#![forbid(unsafe_code)]\n#![allow(dead_code)]");
    }
    for h in helper_set(plan) {
        let src = helper_src(h, opts.partition_point_quantile_bounds);
        if !src.is_empty() {
            out.push(src);
        }
    }
    let scalar_programs = scalar_emit::programs(plan);
    if !scalar_programs.is_empty() {
        out.push(
            &scalar_emit::emit_v2_helpers(&scalar_programs, ScalarLanguage::Rust)
                .expect("admitted scalar helpers"),
        );
    }
    match plan {
        Plan::Pipeline {
            target,
            stages,
            head,
        } => {
            if let Some(target) = target {
                out.push(target_src(target));
            }
            let mut seen: Vec<&'static str> = Vec::new();
            for (i, stage) in stages.iter().enumerate() {
                match &stage.op {
                    PlannedOp::ScalarV2 {
                        program,
                        carrier,
                        width,
                    } => {
                        out.push(
                            &scalar_emit::emit_stage(
                                program,
                                *carrier,
                                *width,
                                i + 1,
                                ScalarLanguage::Rust,
                            )
                            .expect("admitted scalar stage"),
                        );
                    }
                    // Formula stages carry their own expression: one snippet
                    // per stage, named by position (never deduped).
                    PlannedOp::MapExpr { source, expr } => {
                        out.push(&map_src(i + 1, source, expr));
                    }
                    PlannedOp::DeriveExpr { source, expr } => {
                        out.push(&derive_src(i + 1, source, expr));
                    }
                    op => {
                        let name = op_fn_name(op);
                        if !seen.contains(&name) {
                            seen.push(name);
                            out.push(&op_src(op));
                        }
                    }
                }
            }
            if let PlannedHead::Portfolio(pp) = head {
                for src in portfolio_srcs(pp, &mut seen) {
                    out.push(&src);
                }
            } else {
                for src in head_srcs(head) {
                    out.push(&src);
                }
            }
            out.push(&model_pipeline(
                target.as_ref(),
                stages,
                head,
                opts.fast_runtime,
                opts.guard_zero_width_roll
                    && stages
                        .iter()
                        .any(|stage| matches!(stage.op, PlannedOp::Roll { .. })),
            ));
        }
        Plan::Symbolic {
            source,
            expr,
            affine,
        } => {
            out.push(&expression_fn(source, expr));
            out.push(&model_symbolic(*affine, opts.fast_runtime));
        }
    }
    if scalar_emit::raw_input(plan) {
        out.push(
            &crate::kernel::scalar_program::io::io_source(ScalarLanguage::Rust, IO_BLOCK)
                .expect("scalar IO template"),
        );
    } else {
        out.push(IO_BLOCK);
    }
    out.finish()
}

fn module_doc(plan: &Plan, opts: &EmitOptions) -> String {
    let (title, paragraphs) = doc_paragraphs(plan, opts);
    let mut lines: Vec<String> = Vec::new();
    lines.push(format!("//! {title}."));
    for p in &paragraphs {
        lines.push("//!".to_string());
        for l in wrap(p, 94) {
            lines.push(format!("//! {l}"));
        }
    }
    lines.push("//!".to_string());
    if opts.fast_runtime {
        // LLVM performs no floating-point reassociation at these levels, so
        // the recommendation is bit-safe. (-C target-cpu=native is usually
        // safe too, but verify parity first: a toolchain forming FMA would
        // change rounding.)
        lines.push(
            "//! Build: rustc -C opt-level=3 -C lto=fat -C codegen-units=1 model.rs".to_string(),
        );
        lines.push(
            "//! Runtime: predict spreads rows across threads; each row's arithmetic is unchanged."
                .to_string(),
        );
    } else {
        lines.push("//! Build: rustc -O model.rs".to_string());
    }
    lines.push("//! Usage: ./model train.csv test_features.csv predictions.csv".to_string());
    lines.join("\n")
}

/// The row-parallel `predict` wrapper emitted by the fast-runtime profile.
/// Contiguous chunks in input order, each running the exact sequential
/// per-row pipeline, so results are bit-identical to the sequential path.
fn rs_fast_predict_wrapper() -> Vec<String> {
    [
        "    /// Row-parallel predict: contiguous chunks in input order, each running",
        "    /// the exact sequential per-row pipeline, so results are bit-identical.",
        "    fn predict(&self, x: &[Vec<f64>]) -> Vec<f64> {",
        "        let workers = std::thread::available_parallelism().map_or(1, |n| n.get());",
        "        if workers <= 1 || x.len() <= 64 {",
        "            return self.predict_rows(x);",
        "        }",
        "        let chunk = (x.len() + workers - 1) / workers;",
        "        let mut out: Vec<f64> = Vec::with_capacity(x.len());",
        "        std::thread::scope(|scope| {",
        "            let mut parts = Vec::new();",
        "            for rows in x.chunks(chunk) {",
        "                parts.push(scope.spawn(move || self.predict_rows(rows)));",
        "            }",
        "            for part in parts {",
        "                out.extend(part.join().expect(\"prediction thread\"));",
        "            }",
        "        });",
        "        out",
        "    }",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

// -------------------------------- helpers ---------------------------------

fn helper_src(h: Helper, partition_point_bounds: bool) -> &'static str {
    match h {
        // Python-only guards; Rust uses the native f64 methods, which already
        // carry the reference semantics.
        Helper::Clamp
        | Helper::Fmin
        | Helper::Fmax
        | Helper::SafeSin
        | Helper::SafeCos
        | Helper::PowGuard
        | Helper::MathFloor => "",
        Helper::MathDiv => {
            r#"/// Guarded division: a denominator with magnitude under 1e-12 yields 0.0.
fn m_div(a: f64, b: f64) -> f64 {
    if b.abs() < 1e-12 {
        0.0
    } else {
        a / b
    }
}"#
        }
        Helper::MathLog1p => {
            r#"/// log(1 + value), defined as 0.0 for value <= -1 where the logarithm diverges.
fn m_log1p(value: f64) -> f64 {
    if value <= -1.0 {
        0.0
    } else {
        value.ln_1p()
    }
}"#
        }
        Helper::MathSigmoid => {
            r#"/// Logistic function 1/(1+exp(-v)) with the argument clipped to [-30, 30].
fn m_sigmoid(value: f64) -> f64 {
    1.0 / (1.0 + (-value.clamp(-30.0, 30.0)).exp())
}"#
        }
        Helper::MathClip => {
            r#"/// Clip value to the interval spanned by a and b.
fn m_clip(value: f64, a: f64, b: f64, failed: &mut bool) -> f64 {
    let lo = a.min(b);
    let hi = a.max(b);
    if lo.is_nan() || hi.is_nan() { *failed = true; return 0.0; }
    value.clamp(lo, hi)
}"#
        }
        Helper::Asinh => {
            r#"/// Inverse hyperbolic sine via the stable identity sign(v)*ln(|v| + sqrt(v*v + 1)).
fn m_asinh(v: f64) -> f64 {
    let a = v.abs();
    let s = if v > 0.0 {
        1.0
    } else if v < 0.0 {
        -1.0
    } else {
        0.0
    };
    s * (a + (v * v + 1.0).sqrt()).ln()
}"#
        }
        Helper::MathExp => {
            r#"/// exp() with the argument clipped to [-30, 30], keeping the result finite.
fn m_exp(value: f64) -> f64 {
    value.clamp(-30.0, 30.0).exp()
}"#
        }
        Helper::MathLog => {
            r#"/// Natural logarithm, defined as 0.0 for arguments no greater than 1e-300.
fn m_log(value: f64) -> f64 {
    if value <= 1e-300 {
        0.0
    } else {
        value.ln()
    }
}"#
        }
        Helper::Erf => {
            r#"/// Error function via the Abramowitz-Stegun 7.1.26 rational approximation.
fn m_erf(v: f64) -> f64 {
    let a = v.abs();
    let s = if v > 0.0 {
        1.0
    } else if v < 0.0 {
        -1.0
    } else {
        0.0
    };
    let t = 1.0 / (1.0 + 0.3275911 * a);
    let poly = t
        * (0.254829592
            + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    s * (1.0 - poly * (-(a * a)).exp())
}"#
        }
        Helper::Sign => {
            r#"/// Sign function: 1.0 for positive, -1.0 for negative, 0.0 at zero (and for NaN).
fn m_sign(v: f64) -> f64 {
    if v > 0.0 {
        1.0
    } else if v < 0.0 {
        -1.0
    } else {
        0.0
    }
}"#
        }
        Helper::RoundHalfAway => {
            r#"/// Round half away from zero via the explicit branch formula shared with the
/// Python artifact (whose builtin round is banker's).
fn m_round(v: f64) -> f64 {
    if v >= 0.0 {
        (v + 0.5).floor()
    } else {
        (v - 0.5).ceil()
    }
}"#
        }
        Helper::RowMean => {
            r#"/// Arithmetic mean of a row (0.0 for an empty row).
fn row_mean(row: &[f64]) -> f64 {
    let mut total = 0.0;
    for v in row {
        total += *v;
    }
    total / row.len().max(1) as f64
}"#
        }
        Helper::RowSum => {
            r#"fn row_sum(row: &[f64]) -> f64 {
    let mut total = 0.0;
    for v in row {
        total += *v;
    }
    total
}"#
        }
        Helper::RowL1 => {
            r#"/// Sum of absolute values (L1 norm) of a row.
fn row_l1(row: &[f64]) -> f64 {
    let mut total = 0.0;
    for v in row {
        total += v.abs();
    }
    total
}"#
        }
        Helper::RowL2 => {
            r#"/// Euclidean (L2) norm of a row.
fn row_l2(row: &[f64]) -> f64 {
    let mut total = 0.0;
    for v in row {
        total += *v * *v;
    }
    total.sqrt()
}"#
        }
        Helper::RowMin => {
            r#"fn row_min(row: &[f64]) -> f64 {
    let mut m = f64::INFINITY;
    for &v in row {
        m = m.min(v);
    }
    m
}"#
        }
        Helper::RowMax => {
            r#"fn row_max(row: &[f64]) -> f64 {
    let mut m = f64::NEG_INFINITY;
    for &v in row {
        m = m.max(v);
    }
    m
}"#
        }
        Helper::RowStd => {
            r#"/// Standard deviation of a row with the 1/n convention.
fn row_std(row: &[f64]) -> f64 {
    let mu = row_mean(row);
    let mut total = 0.0;
    for v in row {
        let d = *v - mu;
        total += d * d;
    }
    (total / row.len().max(1) as f64).sqrt()
}"#
        }
        Helper::Pcg32 => {
            r#"/// Permuted congruential generator, XSH-RR 64/32 variant (O'Neill, 2014).
struct Pcg32 {
    state: u64,
    inc: u64,
}

impl Pcg32 {
    fn new(seed: u64, stream: u64) -> Pcg32 {
        let mut rng = Pcg32 { state: 0, inc: (stream << 1) | 1 };
        rng.next_u32();
        rng.state = rng.state.wrapping_add(seed);
        rng.next_u32();
        rng
    }

    fn next_u32(&mut self) -> u32 {
        let old = self.state;
        self.state = old.wrapping_mul(6364136223846793005).wrapping_add(self.inc);
        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        let rot = (old >> 59) as u32;
        xorshifted.rotate_right(rot)
    }
}"#
        }
        Helper::RngUniform => {
            r#"impl Pcg32 {
    /// Uniform in [0, 1) from 53 random bits: ((hi<<32|lo) >> 11) * 2^-53.
    fn next_f64(&mut self) -> f64 {
        let hi = (self.next_u32() as u64) << 32;
        let lo = self.next_u32() as u64;
        (((hi | lo) >> 11) as f64) * (1.0 / 9007199254740992.0)
    }
}"#
        }
        Helper::RngRange => {
            r#"impl Pcg32 {
    fn range_f64(&mut self, a: f64, b: f64) -> f64 {
        a + (b - a) * self.next_f64()
    }
}"#
        }
        Helper::RngGaussian => {
            r#"impl Pcg32 {
    /// Standard normal via the Box-Muller cosine branch (u1 floored at 1e-12).
    fn gaussian(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-12);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}"#
        }
        Helper::StageRng => {
            r#"/// Independent random stream keyed by a stage's input and output widths.
fn stage_rng(seed: u64, in_cols: usize, n_out: usize) -> Pcg32 {
    Pcg32::new(seed, ((in_cols as u64) << 32) ^ (n_out as u64))
}"#
        }
        Helper::Sanitize => {
            r#"/// Replace non-finite values with 0.0 in every row.
fn sanitize(x: &[Vec<f64>]) -> Vec<Vec<f64>> {
    x.iter()
        .map(|row| row.iter().map(|v| if v.is_finite() { *v } else { 0.0 }).collect())
        .collect()
}"#
        }
        Helper::FinishStage => {
            r#"/// Stage epilogue: optionally keep the inputs, cap width at 512, zero non-finite cells.
fn finish_stage(x: &[Vec<f64>], mut out: Vec<Vec<f64>>, append: bool) -> Vec<Vec<f64>> {
    if append {
        for (i, row) in out.iter_mut().enumerate() {
            let mut merged = x[i].clone();
            merged.append(row);
            *row = merged;
        }
    }
    for row in out.iter_mut() {
        if row.len() > 512 {
            row.truncate(512);
        }
        for v in row.iter_mut() {
            if !v.is_finite() {
                *v = 0.0;
            }
        }
        if row.is_empty() {
            row.push(0.0);
        }
    }
    out
}"#
        }
        Helper::SortedColumn => {
            r#"/// Column j of x in ascending order.
fn sorted_column(x: &[Vec<f64>], j: usize) -> Vec<f64> {
    let mut v: Vec<f64> = x.iter().map(|row| row[j]).collect();
    v.sort_by(|a, b| a.total_cmp(b));
    v
}"#
        }
        Helper::Quantile => {
            r#"/// Linearly interpolated quantile of an ascending slice (the numpy 'linear' rule).
fn quantile(sorted_values: &[f64], q: f64) -> f64 {
    if sorted_values.is_empty() {
        return 0.0;
    }
    let pos = q * (sorted_values.len() - 1) as f64;
    let lo = pos as usize;
    let hi = (lo + 1).min(sorted_values.len() - 1);
    let frac = pos - lo as f64;
    sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac
}"#
        }
        Helper::LowerBound => {
            if partition_point_bounds {
                r#"/// Count of elements strictly less than v (hand-rolled binary search).
fn lower_bound(sorted_values: &[f64], v: f64) -> usize {
    let mut size = sorted_values.len();
    if size == 0 {
        return 0;
    }
    let mut base = 0usize;
    while size > 1 {
        let half = size / 2;
        let mid = base + half;
        if sorted_values[mid] < v {
            base = mid;
        }
        size -= half;
    }
    base + usize::from(sorted_values[base] < v)
}"#
            } else {
                r#"/// Count of elements strictly less than v (hand-rolled binary search).
fn lower_bound(sorted_values: &[f64], v: f64) -> usize {
    let mut lo = 0usize;
    let mut hi = sorted_values.len();
    while lo < hi {
        let mid = (lo + hi) / 2;
        if sorted_values[mid] < v {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}"#
            }
        }
        Helper::UpperBound => {
            if partition_point_bounds {
                r#"/// Count of elements less than or equal to v (hand-rolled binary search).
fn upper_bound(sorted_values: &[f64], v: f64) -> usize {
    let mut size = sorted_values.len();
    if size == 0 {
        return 0;
    }
    let mut base = 0usize;
    while size > 1 {
        let half = size / 2;
        let mid = base + half;
        if sorted_values[mid] <= v {
            base = mid;
        }
        size -= half;
    }
    base + usize::from(sorted_values[base] <= v)
}"#
            } else {
                r#"/// Count of elements less than or equal to v (hand-rolled binary search).
fn upper_bound(sorted_values: &[f64], v: f64) -> usize {
    let mut lo = 0usize;
    let mut hi = sorted_values.len();
    while lo < hi {
        let mid = (lo + hi) / 2;
        if sorted_values[mid] <= v {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}"#
            }
        }
        Helper::MidRank => {
            r#"/// Average of the strict and non-strict ranks of v, scaled to [0, 1].
fn mid_rank(sorted_values: &[f64], v: f64) -> f64 {
    let lb = lower_bound(sorted_values, v);
    let ub = upper_bound(sorted_values, v);
    (lb + ub) as f64 / (2.0 * sorted_values.len() as f64)
}"#
        }
        Helper::Probit => {
            r#"/// Standard normal quantile function via Acklam's rational approximation.
fn probit(p: f64) -> f64 {
    let a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
        1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00];
    let b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
        6.680131188771972e+01, -1.328068155288572e+01];
    let c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
        -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00];
    let d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
        3.754408661907416e+00];
    let p = p.clamp(1e-12, 1.0 - 1e-12);
    if p < 0.02425 {
        let q = (-2.0 * p.ln()).sqrt();
        (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    } else if p <= 1.0 - 0.02425 {
        let q = p - 0.5;
        let r = q * q;
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    }
}"#
        }
        Helper::QuantileEdges => {
            r#"/// Interior quantile cut points (1/bins .. (bins-1)/bins) for every column.
fn quantile_edges(x: &[Vec<f64>], bins: usize) -> Vec<Vec<f64>> {
    let mut edges = Vec::new();
    for j in 0..x[0].len() {
        let s = sorted_column(x, j);
        edges.push((1..bins).map(|b| quantile(&s, b as f64 / bins as f64)).collect());
    }
    edges
}"#
        }
        Helper::Dot => {
            r#"/// Inner product over the shorter of the two vectors.
fn dot(a: &[f64], b: &[f64]) -> f64 {
    let mut total = 0.0;
    for i in 0..a.len().min(b.len()) {
        total += a[i] * b[i];
    }
    total
}"#
        }
        Helper::Normalize => {
            r#"/// Scale a vector to unit length in place; a 1e-12 floor keeps zeros finite.
fn normalize(v: &mut [f64]) {
    let mut s = 0.0;
    for e in v.iter() {
        s += e * e;
    }
    let inv = 1.0 / s.sqrt().max(1e-12);
    for e in v.iter_mut() {
        *e *= inv;
    }
}"#
        }
        Helper::RbfRow => {
            r#"/// Gaussian kernel similarities exp(-gamma * ||row - l||^2) to each landmark.
fn rbf_row(landmarks: &[Vec<f64>], row: &[f64], gamma: f64) -> Vec<f64> {
    let mut values = Vec::new();
    for landmark in landmarks {
        let mut d2 = 0.0;
        for j in 0..landmark.len().min(row.len()) {
            let d = row[j] - landmark[j];
            d2 += d * d;
        }
        values.push((-gamma * d2).exp());
    }
    values
}"#
        }
        Helper::DftMags => {
            r#"/// Magnitudes of the direct discrete Fourier transform at frequencies 1..=k, scaled by 1/d.
fn dft_magnitudes(row: &[f64], k: usize) -> Vec<f64> {
    let d = row.len() as f64;
    let mut out = Vec::with_capacity(k);
    for f in 1..(k + 1) {
        let mut re = 0.0;
        let mut im = 0.0;
        for (j, v) in row.iter().enumerate() {
            let ang = 2.0 * std::f64::consts::PI * f as f64 * j as f64 / d;
            re += v * ang.cos();
            im += v * ang.sin();
        }
        out.push((re * re + im * im).sqrt() / d);
    }
    out
}"#
        }
        Helper::MeanOf => {
            r#"fn mean_of(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut total = 0.0;
    for v in values {
        total += *v;
    }
    total / values.len() as f64
}"#
        }
        Helper::Pearson => {
            r#"/// Pearson correlation, 0.0 for degenerate inputs.
fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n < 2 {
        return 0.0;
    }
    let mut ma = 0.0;
    for i in 0..n {
        ma += a[i];
    }
    ma /= n as f64;
    let mut mb = 0.0;
    for i in 0..n {
        mb += b[i];
    }
    mb /= n as f64;
    let mut sab = 0.0;
    let mut saa = 0.0;
    let mut sbb = 0.0;
    for i in 0..n {
        let da = a[i] - ma;
        let db = b[i] - mb;
        sab += da * db;
        saa += da * da;
        sbb += db * db;
    }
    if saa <= 0.0 || sbb <= 0.0 {
        return 0.0;
    }
    sab / (saa * sbb).sqrt()
}"#
        }
        Helper::Rmse => {
            r#"/// Root mean squared error over the common prefix.
fn rmse(y: &[f64], p: &[f64]) -> f64 {
    let n = y.len().min(p.len());
    if n == 0 {
        return 0.0;
    }
    let mut ss = 0.0;
    for i in 0..n {
        let d = y[i] - p[i];
        ss += d * d;
    }
    (ss / n as f64).sqrt()
}"#
        }
        Helper::RSquared => {
            r#"/// Coefficient of determination R^2 = 1 - SS_res/SS_tot (0.0 when SS_tot <= 0).
fn r2_score(y: &[f64], p: &[f64]) -> f64 {
    let n = y.len().min(p.len());
    if n == 0 {
        return 0.0;
    }
    let mut mean = 0.0;
    for i in 0..n {
        mean += y[i];
    }
    mean /= n as f64;
    let mut ss_res = 0.0;
    let mut ss_tot = 0.0;
    for i in 0..n {
        let d = y[i] - p[i];
        ss_res += d * d;
        let t = y[i] - mean;
        ss_tot += t * t;
    }
    if ss_tot <= 0.0 {
        return 0.0;
    }
    1.0 - ss_res / ss_tot
}"#
        }
        Helper::StdFit => {
            r#"/// Per-column mean and standard deviation (1/n convention), floored at 1e-9.
fn std_fit(x: &[Vec<f64>]) -> (Vec<f64>, Vec<f64>) {
    let n = x.len().max(1) as f64;
    let p = x[0].len();
    let mut means = vec![0.0; p];
    let mut scales = vec![1.0; p];
    for j in 0..p {
        let mut s = 0.0;
        for row in x {
            s += row[j];
        }
        means[j] = s / n;
        let mut v = 0.0;
        for row in x {
            let d = row[j] - means[j];
            v += d * d;
        }
        scales[j] = (v / n).sqrt().max(1e-9);
    }
    (means, scales)
}"#
        }
        Helper::CholeskySolve => {
            r#"/// Solve the symmetric positive-definite system A w = b.
///
/// A (flattened row-major, overwritten) is factored in place as L L^T, then a
/// forward and a backward triangular solve produce w. A tiny 1e-6 pivot stands
/// in whenever a diagonal entry falls under 1e-12.
fn cholesky_solve(a: &mut [f64], b: &[f64], p: usize) -> Vec<f64> {
    for i in 0..p {
        for j in 0..(i + 1) {
            let mut s = a[i * p + j];
            for k in 0..j {
                s -= a[i * p + k] * a[j * p + k];
            }
            if i == j {
                a[i * p + j] = if s > 1e-12 { s.sqrt() } else { 1e-6 };
            } else {
                a[i * p + j] = s / a[j * p + j];
            }
        }
    }
    let mut z = vec![0.0; p];
    for i in 0..p {
        let mut s = b[i];
        for k in 0..i {
            s -= a[i * p + k] * z[k];
        }
        z[i] = s / a[i * p + i];
    }
    let mut w = vec![0.0; p];
    for i in (0..p).rev() {
        let mut s = z[i];
        for k in (i + 1)..p {
            s -= a[k * p + i] * w[k];
        }
        w[i] = s / a[i * p + i];
    }
    w
}"#
        }
        Helper::JacobiEigen => {
            r#"/// Eigendecomposition of the symmetric p x p matrix `a` (flattened row-major,
/// overwritten) by cyclic Jacobi rotations (Jacobi, 1846): returns
/// (eigenvalues, eigenvectors) with the eigenvectors flattened row-major so
/// column i holds the unit vector of eigenvalue i. Deterministic: fixed sweep
/// order, at most 20 sweeps, stopping as soon as a whole-matrix check finds
/// the off-diagonal mass under a 1e-12-of-trace threshold.
fn jacobi_eigen(a: &mut [f64], p: usize) -> (Vec<f64>, Vec<f64>) {
    let mut v = vec![0.0; p * p];
    for i in 0..p {
        v[i * p + i] = 1.0;
    }
    let mut scale = 0.0;
    for i in 0..p {
        scale += a[i * p + i];
    }
    let limit = (scale * 1e-12) * (scale * 1e-12);
    for _ in 0..20 {
        let mut off = 0.0;
        for i in 0..p {
            for j in (i + 1)..p {
                off += a[i * p + j] * a[i * p + j];
            }
        }
        if off <= limit {
            break;
        }
        for i in 0..p {
            for j in (i + 1)..p {
                let apq = a[i * p + j];
                if apq == 0.0 {
                    continue;
                }
                let theta = (a[j * p + j] - a[i * p + i]) / (2.0 * apq);
                let t = if theta >= 0.0 {
                    1.0 / (theta + (theta * theta + 1.0).sqrt())
                } else {
                    -1.0 / ((theta * theta + 1.0).sqrt() - theta)
                };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..p {
                    if k != i && k != j {
                        let aki = a[k * p + i];
                        let akj = a[k * p + j];
                        a[k * p + i] = c * aki - s * akj;
                        a[i * p + k] = a[k * p + i];
                        a[k * p + j] = s * aki + c * akj;
                        a[j * p + k] = a[k * p + j];
                    }
                }
                let aii = a[i * p + i];
                let ajj = a[j * p + j];
                a[i * p + i] = c * c * aii - 2.0 * s * c * apq + s * s * ajj;
                a[j * p + j] = s * s * aii + 2.0 * s * c * apq + c * c * ajj;
                a[i * p + j] = 0.0;
                a[j * p + i] = 0.0;
                for k in 0..p {
                    let vki = v[k * p + i];
                    let vkj = v[k * p + j];
                    v[k * p + i] = c * vki - s * vkj;
                    v[k * p + j] = s * vki + c * vkj;
                }
            }
        }
    }
    let mut e = vec![0.0; p];
    for i in 0..p {
        e[i] = a[i * p + i];
    }
    (e, v)
}"#
        }
        Helper::SoftThreshold => {
            r#"/// Shrink v toward zero by t (the lasso proximal step).
fn soft_threshold(v: f64, t: f64) -> f64 {
    if v > t {
        v - t
    } else if v < -t {
        v + t
    } else {
        0.0
    }
}"#
        }
        Helper::TreeCore => TREE_CORE,
    }
}

const TREE_CORE: &str = r#"/// One node of a regression tree; feature -1 marks a leaf holding `value`.
struct TreeNode {
    feature: i64,
    threshold: i64,
    left: usize,
    right: usize,
    value: f64,
}

/// Quantile cut points for one feature, consecutive duplicates collapsed.
fn tree_bin_edges(x: &[Vec<f64>], f: usize, n: usize, bins: usize) -> Vec<f64> {
    let mut vals: Vec<f64> = (0..n).map(|r| x[r][f]).collect();
    vals.sort_by(|a, b| a.total_cmp(b));
    let mut edges = Vec::new();
    for q in 1..bins {
        let idx = ((vals.len() as f64) * (q as f64 / bins as f64)) as usize;
        edges.push(vals[idx.min(vals.len() - 1)]);
    }
    edges.dedup();
    edges
}

/// Flat row-major matrix of per-feature bin indices.
fn tree_bin_rows(x: &[Vec<f64>], rows: usize, n_feat: usize, edges: &[Vec<f64>]) -> Vec<i64> {
    let mut out = vec![0i64; rows * n_feat];
    for r in 0..rows {
        for f in 0..n_feat {
            let v = if f < x[r].len() { x[r][f] } else { 0.0 };
            out[r * n_feat + f] = lower_bound(&edges[f], v) as i64;
        }
    }
    out
}

/// Greedy histogram split search; returns the index of the new node.
/// Leaf values are the second-order step -g/(h + lam).
fn grow_tree(bins: &[i64], grad: &[f64], hess: &[f64], rows: Vec<usize>, n_feat: usize,
             nbins: usize, max_depth: usize, min_leaf: usize, lam: f64, depth: usize,
             tree: &mut Vec<TreeNode>, leaf_of: &mut [usize]) -> usize {
    let mut g = 0.0;
    for &i in &rows {
        g += grad[i];
    }
    let mut h = 0.0;
    for &i in &rows {
        h += hess[i];
    }
    let leaf_value = -g / (h + lam);
    if depth >= max_depth || rows.len() < 2 * min_leaf {
        let idx = tree.len();
        tree.push(TreeNode { feature: -1, threshold: 0, left: 0, right: 0, value: leaf_value });
        for &i in &rows {
            leaf_of[i] = idx;
        }
        return idx;
    }
    let mut best_gain = 0.0;
    let mut best_feat: i64 = -1;
    let mut best_thr: i64 = 0;
    let parent = g * g / (h + lam);
    for f in 0..n_feat {
        let mut gh = vec![0.0; nbins];
        let mut hh = vec![0.0; nbins];
        for &i in &rows {
            let b = bins[i * n_feat + f] as usize;
            gh[b] += grad[i];
            hh[b] += hess[i];
        }
        let mut gl = 0.0;
        let mut hl = 0.0;
        for b in 0..(nbins - 1) {
            gl += gh[b];
            hl += hh[b];
            let gr = g - gl;
            let hr = h - hl;
            if hl < 1.0 || hr < 1.0 {
                continue;
            }
            let gain = gl * gl / (hl + lam) + gr * gr / (hr + lam) - parent;
            if gain > best_gain {
                best_gain = gain;
                best_feat = f as i64;
                best_thr = b as i64;
            }
        }
    }
    if best_feat < 0 || best_gain <= 1e-9 {
        let idx = tree.len();
        tree.push(TreeNode { feature: -1, threshold: 0, left: 0, right: 0, value: leaf_value });
        for &i in &rows {
            leaf_of[i] = idx;
        }
        return idx;
    }
    let mut left_rows = Vec::new();
    let mut right_rows = Vec::new();
    for &i in &rows {
        if bins[i * n_feat + best_feat as usize] <= best_thr {
            left_rows.push(i);
        } else {
            right_rows.push(i);
        }
    }
    if left_rows.len() < min_leaf || right_rows.len() < min_leaf {
        let idx = tree.len();
        tree.push(TreeNode { feature: -1, threshold: 0, left: 0, right: 0, value: leaf_value });
        for &i in &rows {
            leaf_of[i] = idx;
        }
        return idx;
    }
    let idx = tree.len();
    tree.push(TreeNode { feature: -1, threshold: 0, left: 0, right: 0, value: 0.0 });
    let left = grow_tree(bins, grad, hess, left_rows, n_feat, nbins, max_depth, min_leaf, lam,
                         depth + 1, tree, leaf_of);
    let right = grow_tree(bins, grad, hess, right_rows, n_feat, nbins, max_depth, min_leaf, lam,
                          depth + 1, tree, leaf_of);
    tree[idx] = TreeNode { feature: best_feat, threshold: best_thr, left, right, value: 0.0 };
    idx
}

/// Follow split nodes to a leaf and return its value.
fn route_tree(tree: &[TreeNode], bins: &[i64], r: usize, n_feat: usize) -> f64 {
    let mut idx = 0usize;
    loop {
        let node = &tree[idx];
        if node.feature < 0 {
            return node.value;
        }
        idx = if bins[r * n_feat + node.feature as usize] <= node.threshold {
            node.left
        } else {
            node.right
        };
    }
}"#;

// ----------------------------- target transform ----------------------------

/// The target-transform snippet for one kind. Every arithmetic step mirrors
/// the reference target transform exactly: total-order sort, the shared
/// mid-rank + Acklam probit helpers, binary search, linear interpolation.
fn target_src(target: &PlannedTarget) -> &'static str {
    match target.kind {
        TargetKind::RankGauss => {
            r#"/// Fitted normal-scores target transform: the sorted training target plus one
/// (value, score) pair per distinct value, where the score is the standard
/// normal quantile of the value's mid-rank.
struct NormalScoresTarget {
    sorted_values: Vec<f64>,
    table_values: Vec<f64>,
    table_scores: Vec<f64>,
}

fn normal_scores_target_fit(y: &[f64]) -> NormalScoresTarget {
    let mut sorted_values = y.to_vec();
    sorted_values.sort_by(|a, b| a.total_cmp(b));
    let mut table_values = Vec::new();
    let mut table_scores = Vec::new();
    for i in 0..sorted_values.len() {
        if i == 0 || sorted_values[i] > sorted_values[i - 1] {
            table_values.push(sorted_values[i]);
            table_scores.push(probit(mid_rank(&sorted_values, sorted_values[i])));
        }
    }
    NormalScoresTarget { sorted_values, table_values, table_scores }
}

/// Map every target value to the standard normal quantile of its mid-rank.
fn normal_scores_target_apply(state: &NormalScoresTarget, y: &[f64]) -> Vec<f64> {
    y.iter().map(|v| probit(mid_rank(&state.sorted_values, *v))).collect()
}

/// Map one prediction back to the original target scale by piecewise-linear
/// interpolation through the fitted (score, value) table, clamped to the
/// extreme training values.
fn normal_scores_target_invert(state: &NormalScoresTarget, z: f64) -> f64 {
    let values = &state.table_values;
    let scores = &state.table_scores;
    let m = values.len();
    // NaN also routes to the low end.
    if !(z > scores[0]) {
        return values[0];
    }
    if z >= scores[m - 1] {
        return values[m - 1];
    }
    let hi = lower_bound(scores, z);
    let lo = hi - 1;
    let ds = scores[hi] - scores[lo];
    if ds > 0.0 {
        values[lo] + (z - scores[lo]) / ds * (values[hi] - values[lo])
    } else {
        values[lo]
    }
}"#
        }
        TargetKind::Log1p => {
            r#"/// Map every target value through the shifted logarithm ln(1 + y), with the
/// argument floored at -0.999999 so the logarithm stays finite.
fn shifted_log_target_apply(y: &[f64]) -> Vec<f64> {
    y.iter().map(|v| (1.0 + v.max(-0.999999)).ln()).collect()
}

/// Exact inverse exp(z) - 1, the exponent clipped to [-745, 700] so exp()
/// stays within double range.
fn shifted_log_target_invert(z: f64) -> f64 {
    z.clamp(-745.0, 700.0).exp() - 1.0
}"#
        }
        TargetKind::Winsor => {
            r#"/// Fitted winsorized target transform: the [q, 1-q] quantiles of the
/// training target.
struct WinsorizedTarget {
    lo: f64,
    hi: f64,
}

fn winsorized_target_fit(y: &[f64], q: f64) -> WinsorizedTarget {
    let mut s = y.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    let lo = quantile(&s, q);
    // interpolated quantiles on a near-constant target can cross by one ulp
    let hi = quantile(&s, 1.0 - q).max(lo);
    WinsorizedTarget { lo, hi }
}

/// Clip every target value to the fitted quantiles; the inverse is the
/// identity, so winsorized predictions stay winsorized.
fn winsorized_target_apply(state: &WinsorizedTarget, y: &[f64]) -> Vec<f64> {
    y.iter().map(|v| v.clamp(state.lo, state.hi)).collect()
}"#
        }
        TargetKind::ZScore => {
            r#"/// Fitted standardized target transform: training mean and standard deviation
/// (floored at 1e-9), plus the training-range prediction clamp bounds (min/max
/// widened by 5% of the range).
struct StandardizedTarget {
    mu: f64,
    sigma: f64,
    lo: f64,
    hi: f64,
}

fn standardized_target_fit(y: &[f64]) -> StandardizedTarget {
    let n = y.len() as f64;
    let mut mu = 0.0;
    for v in y {
        mu += v;
    }
    mu /= n;
    let mut var = 0.0;
    for v in y {
        let d = v - mu;
        var += d * d;
    }
    var /= n;
    let sigma = var.sqrt().max(1e-9);
    let mut y_min = f64::INFINITY;
    let mut y_max = f64::NEG_INFINITY;
    for v in y {
        y_min = y_min.min(*v);
        y_max = y_max.max(*v);
    }
    let spread = (y_max - y_min).max(1e-9);
    StandardizedTarget {
        mu,
        sigma,
        lo: (y_min - 0.05 * spread).min(-60.0),
        hi: (y_max + 0.05 * spread).max(60.0),
    }
}

/// Standardize every target value to (v - mu) / sigma.
fn standardized_target_apply(state: &StandardizedTarget, y: &[f64]) -> Vec<f64> {
    y.iter().map(|v| (v - state.mu) / state.sigma).collect()
}

/// Map one prediction back to the original target scale (z * sigma + mu), then
/// clamp to the training-target range widened by 5%.
fn standardized_target_invert(state: &StandardizedTarget, z: f64) -> f64 {
    (z * state.sigma + state.mu).clamp(state.lo, state.hi)
}"#
        }
    }
}

/// (fit-call expression or None for the stateless kind, forward-apply call on
/// the training target, inverse expression over one prediction `z`).
fn target_calls(target: &PlannedTarget) -> (Option<String>, String, String) {
    let name = target_fn_name(target.kind);
    match target.kind {
        TargetKind::RankGauss => (
            Some(format!("{name}_fit(y)")),
            format!("{name}_apply(&target, y)"),
            format!("{name}_invert(&self.target, *z)"),
        ),
        TargetKind::Log1p => (
            None,
            format!("{name}_apply(y)"),
            format!("{name}_invert(*z)"),
        ),
        TargetKind::Winsor => (
            Some(format!("{name}_fit(y, {})", fmt_f64(target.q))),
            format!("{name}_apply(&target, y)"),
            "z".to_string(),
        ),
        TargetKind::ZScore => (
            Some(format!("{name}_fit(y)")),
            format!("{name}_apply(&target, y)"),
            format!("{name}_invert(&self.target, *z)"),
        ),
    }
}

// ------------------------------- transforms --------------------------------

fn doc_comment(text: &str) -> String {
    wrap(text, 92)
        .iter()
        .map(|l| format!("/// {l}"))
        .collect::<Vec<_>>()
        .join("\n")
}

fn op_src(op: &PlannedOp) -> String {
    let doc = doc_comment(&format!("This transform {}", crate::spec::op_doc(op)));
    let body: &str = match op {
        PlannedOp::ScalarV2 { .. } => "",
        // B13: the goertzel_bank helper, only for a bank-bearing program.
        PlannedOp::RowVectorV2 {
            state: crate::kernel::row_vector::FittedStage::Goertzel(_),
        } => crate::kernel::row_vector::emit::goertzel_helper(
            crate::kernel::row_vector::emit::RowRecipeLanguage::Rust,
        ),
        // B12: helper bodies remain feature-specific.
        PlannedOp::RowVectorV2 { .. } | PlannedOp::RowVectorInputV2 { .. } => {
            crate::kernel::row_vector::emit::helper(
                crate::kernel::row_vector::emit::RowRecipeLanguage::Rust,
            )
        }
        PlannedOp::Standardize => {
            r#"struct StandardScaler {
    means: Vec<f64>,
    scales: Vec<f64>,
}

{DOC}
fn standard_scaler_fit(x: &[Vec<f64>]) -> StandardScaler {
    let (means, scales) = std_fit(x);
    StandardScaler { means, scales }
}

fn standard_scaler_apply(state: &StandardScaler, x: &[Vec<f64>], append: bool) -> Vec<Vec<f64>> {
    let out: Vec<Vec<f64>> = x
        .iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .map(|(j, v)| ((v - state.means[j]) / state.scales[j]).clamp(-20.0, 20.0))
                .collect()
        })
        .collect();
    finish_stage(x, out, append)
}"#
        }
        PlannedOp::RobustScale => {
            r#"struct RobustScaler {
    medians: Vec<f64>,
    scales: Vec<f64>,
}

{DOC}
fn robust_scaler_fit(x: &[Vec<f64>]) -> RobustScaler {
    let mut medians = Vec::new();
    let mut scales = Vec::new();
    for j in 0..x[0].len() {
        let s = sorted_column(x, j);
        medians.push(quantile(&s, 0.5));
        let iqr = quantile(&s, 0.75) - quantile(&s, 0.25);
        scales.push(iqr.max(1e-9));
    }
    RobustScaler { medians, scales }
}

fn robust_scaler_apply(state: &RobustScaler, x: &[Vec<f64>], append: bool) -> Vec<Vec<f64>> {
    let out: Vec<Vec<f64>> = x
        .iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .map(|(j, v)| (v - state.medians[j]) / state.scales[j])
                .collect()
        })
        .collect();
    finish_stage(x, out, append)
}"#
        }
        PlannedOp::Rank => {
            r#"struct MidRankTransform {
    sorted_cols: Vec<Vec<f64>>,
}

{DOC}
fn mid_rank_transform_fit(x: &[Vec<f64>]) -> MidRankTransform {
    MidRankTransform { sorted_cols: (0..x[0].len()).map(|j| sorted_column(x, j)).collect() }
}

fn mid_rank_transform_apply(state: &MidRankTransform, x: &[Vec<f64>], append: bool) -> Vec<Vec<f64>> {
    let out: Vec<Vec<f64>> = x
        .iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .map(|(j, v)| mid_rank(&state.sorted_cols[j], *v))
                .collect()
        })
        .collect();
    finish_stage(x, out, append)
}"#
        }
        PlannedOp::Blom => {
            r#"struct BlomNormalScores {
    sorted_cols: Vec<Vec<f64>>,
}

{DOC}
fn blom_normal_scores_fit(x: &[Vec<f64>]) -> BlomNormalScores {
    BlomNormalScores { sorted_cols: (0..x[0].len()).map(|j| sorted_column(x, j)).collect() }
}

fn blom_normal_scores_apply(state: &BlomNormalScores, x: &[Vec<f64>], append: bool) -> Vec<Vec<f64>> {
    let mut out = Vec::new();
    for row in x {
        let mut values = Vec::new();
        for (j, v) in row.iter().enumerate() {
            let n = state.sorted_cols[j].len() as f64;
            let mid = mid_rank(&state.sorted_cols[j], *v) * n;
            values.push(probit((mid + 0.125) / (n + 0.25)));
        }
        out.push(values);
    }
    finish_stage(x, out, append)
}"#
        }
        PlannedOp::QuantileBins { .. } => {
            r#"struct QuantileBinIndexer {
    edges: Vec<Vec<f64>>,
}

{DOC}
fn quantile_bin_indexer_fit(x: &[Vec<f64>], bins: usize) -> QuantileBinIndexer {
    QuantileBinIndexer { edges: quantile_edges(x, bins) }
}

fn quantile_bin_indexer_apply(state: &QuantileBinIndexer, x: &[Vec<f64>], bins: usize, append: bool) -> Vec<Vec<f64>> {
    let out: Vec<Vec<f64>> = x
        .iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .map(|(j, v)| lower_bound(&state.edges[j], *v) as f64 / (bins as f64 - 1.0))
                .collect()
        })
        .collect();
    finish_stage(x, out, append)
}"#
        }
        PlannedOp::BinOnehot { .. } => {
            r#"struct QuantileOnehotEncoder {
    edges: Vec<Vec<f64>>,
}

{DOC}
fn quantile_onehot_encoder_fit(x: &[Vec<f64>], bins: usize) -> QuantileOnehotEncoder {
    QuantileOnehotEncoder { edges: quantile_edges(x, bins) }
}

fn quantile_onehot_encoder_apply(state: &QuantileOnehotEncoder, x: &[Vec<f64>], bins: usize, append: bool) -> Vec<Vec<f64>> {
    let mut out = Vec::new();
    for row in x {
        let mut encoded = vec![0.0; row.len() * bins];
        for (j, v) in row.iter().enumerate() {
            let b = lower_bound(&state.edges[j], *v).min(bins - 1);
            encoded[j * bins + b] = 1.0;
        }
        out.push(encoded);
    }
    finish_stage(x, out, append)
}"#
        }
        PlannedOp::Hinge { .. } => {
            r#"struct HingeBasis {
    knots: Vec<Vec<f64>>,
}

{DOC}
fn hinge_basis_fit(x: &[Vec<f64>], knots: usize) -> HingeBasis {
    let mut fitted = Vec::new();
    for j in 0..x[0].len() {
        let s = sorted_column(x, j);
        fitted.push((1..(knots + 1)).map(|i| quantile(&s, i as f64 / (knots + 1) as f64)).collect());
    }
    HingeBasis { knots: fitted }
}

fn hinge_basis_apply(state: &HingeBasis, x: &[Vec<f64>], append: bool) -> Vec<Vec<f64>> {
    let mut out = Vec::new();
    for row in x {
        let mut values = Vec::new();
        for (j, v) in row.iter().enumerate() {
            for t in &state.knots[j] {
                values.push((v - t).max(0.0));
            }
        }
        out.push(values);
    }
    finish_stage(x, out, append)
}"#
        }
        PlannedOp::WinsorClip { .. } => {
            r#"struct WinsorLimiter {
    lo: Vec<f64>,
    hi: Vec<f64>,
}

{DOC}
fn winsor_limiter_fit(x: &[Vec<f64>], q: f64) -> WinsorLimiter {
    let mut lo = Vec::new();
    let mut hi = Vec::new();
    for j in 0..x[0].len() {
        let s = sorted_column(x, j);
        let low = quantile(&s, q);
        // interpolated quantiles on a near-constant column can cross by one ulp
        let high = quantile(&s, 1.0 - q).max(low);
        lo.push(low);
        hi.push(high);
    }
    WinsorLimiter { lo, hi }
}

fn winsor_limiter_apply(state: &WinsorLimiter, x: &[Vec<f64>], append: bool) -> Vec<Vec<f64>> {
    let out: Vec<Vec<f64>> = x
        .iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .map(|(j, v)| v.clamp(state.lo[j], state.hi[j]))
                .collect()
        })
        .collect();
    finish_stage(x, out, append)
}"#
        }
        PlannedOp::Cheb { .. } => {
            r#"{DOC}
fn chebyshev_basis_apply(x: &[Vec<f64>], degree: usize, append: bool) -> Vec<Vec<f64>> {
    let mut out = Vec::new();
    for row in x {
        let mut values = Vec::new();
        for v in row {
            let u = (2.0 * v - 1.0).clamp(-1.0, 1.0);
            let mut t_prev = 1.0;
            let mut t_cur = u;
            for _ in 0..degree {
                values.push(t_cur);
                let t_next = 2.0 * u * t_cur - t_prev;
                t_prev = t_cur;
                t_cur = t_next;
            }
        }
        out.push(values);
    }
    finish_stage(x, out, append)
}"#
        }
        PlannedOp::PolyInteract => {
            r#"struct PairwiseProducts {
    pairs: Vec<(usize, usize)>,
}

{DOC}
fn pairwise_products_fit(x: &[Vec<f64>]) -> PairwiseProducts {
    let cols = x[0].len();
    let mut pairs: Vec<(usize, usize)> = (0..cols).map(|i| (i, i)).collect();
    let mut full = false;
    for i in 0..cols {
        if full {
            break;
        }
        for j in (i + 1)..cols {
            if pairs.len() >= 512 {
                full = true;
                break;
            }
            pairs.push((i, j));
        }
    }
    PairwiseProducts { pairs }
}

fn pairwise_products_apply(state: &PairwiseProducts, x: &[Vec<f64>], append: bool) -> Vec<Vec<f64>> {
    let out: Vec<Vec<f64>> = x
        .iter()
        .map(|row| state.pairs.iter().map(|&(a, b)| row[a] * row[b]).collect())
        .collect();
    finish_stage(x, out, append)
}"#
        }
        PlannedOp::RffCos { .. } => {
            r#"struct RandomFourierCosineFeatures {
    w: Vec<Vec<f64>>,
    b: Vec<f64>,
}

{DOC}
fn random_fourier_cosine_features_fit(x: &[Vec<f64>], n_features: usize, gamma: f64, seed: u64) -> RandomFourierCosineFeatures {
    let cols = x[0].len();
    let mut rng = stage_rng(seed, cols, n_features);
    let scale = (2.0 * gamma).sqrt();
    let mut w = Vec::new();
    for _ in 0..n_features {
        let mut row = Vec::new();
        for _ in 0..cols {
            row.push(rng.gaussian() * scale);
        }
        w.push(row);
    }
    let b: Vec<f64> = (0..n_features).map(|_| rng.range_f64(0.0, 2.0 * std::f64::consts::PI)).collect();
    RandomFourierCosineFeatures { w, b }
}

fn random_fourier_cosine_features_apply(state: &RandomFourierCosineFeatures, x: &[Vec<f64>], append: bool) -> Vec<Vec<f64>> {
    let scale = (2.0 / state.w.len() as f64).sqrt();
    let out: Vec<Vec<f64>> = x
        .iter()
        .map(|row| {
            (0..state.w.len())
                .map(|i| scale * (dot(&state.w[i], row) + state.b[i]).cos())
                .collect()
        })
        .collect();
    finish_stage(x, out, append)
}"#
        }
        PlannedOp::RffSin { .. } => {
            r#"struct RandomFourierSineFeatures {
    w: Vec<Vec<f64>>,
    b: Vec<f64>,
}

{DOC}
fn random_fourier_sine_features_fit(x: &[Vec<f64>], n_features: usize, gamma: f64, seed: u64) -> RandomFourierSineFeatures {
    let cols = x[0].len();
    let mut rng = stage_rng(seed, cols, n_features);
    let scale = (2.0 * gamma).sqrt();
    let mut w = Vec::new();
    for _ in 0..n_features {
        let mut row = Vec::new();
        for _ in 0..cols {
            row.push(rng.gaussian() * scale);
        }
        w.push(row);
    }
    let b: Vec<f64> = (0..n_features).map(|_| rng.range_f64(0.0, 2.0 * std::f64::consts::PI)).collect();
    RandomFourierSineFeatures { w, b }
}

fn random_fourier_sine_features_apply(state: &RandomFourierSineFeatures, x: &[Vec<f64>], append: bool) -> Vec<Vec<f64>> {
    let scale = (2.0 / state.w.len() as f64).sqrt();
    let out: Vec<Vec<f64>> = x
        .iter()
        .map(|row| {
            (0..state.w.len())
                .map(|i| scale * (dot(&state.w[i], row) + state.b[i]).sin())
                .collect()
        })
        .collect();
    finish_stage(x, out, append)
}"#
        }
        PlannedOp::RandProj { .. } => {
            r#"struct GaussianRandomProjection {
    w: Vec<Vec<f64>>,
}

{DOC}
fn gaussian_random_projection_fit(x: &[Vec<f64>], k: usize, seed: u64) -> GaussianRandomProjection {
    let cols = x[0].len();
    let mut rng = stage_rng(seed, cols, k);
    let mut w = Vec::new();
    for _ in 0..k {
        let mut row = Vec::new();
        for _ in 0..cols {
            row.push(rng.gaussian());
        }
        w.push(row);
    }
    GaussianRandomProjection { w }
}

fn gaussian_random_projection_apply(state: &GaussianRandomProjection, x: &[Vec<f64>], append: bool) -> Vec<Vec<f64>> {
    let inv = 1.0 / (state.w.len() as f64).sqrt();
    let out: Vec<Vec<f64>> = x
        .iter()
        .map(|row| (0..state.w.len()).map(|i| dot(&state.w[i], row) * inv).collect())
        .collect();
    finish_stage(x, out, append)
}"#
        }
        PlannedOp::LandmarkDist { .. } => {
            r#"struct GaussianLandmarkFeatures {
    landmarks: Vec<Vec<f64>>,
}

{DOC}
fn gaussian_landmark_features_fit(x: &[Vec<f64>], k: usize, seed: u64) -> GaussianLandmarkFeatures {
    let cols = x[0].len();
    let mut rng = stage_rng(seed, cols, k);
    let mut landmarks = Vec::new();
    for _ in 0..k {
        let mut row = Vec::new();
        for _ in 0..cols {
            row.push(rng.range_f64(0.0, 1.0));
        }
        landmarks.push(row);
    }
    GaussianLandmarkFeatures { landmarks }
}

fn gaussian_landmark_features_apply(state: &GaussianLandmarkFeatures, x: &[Vec<f64>], gamma: f64, append: bool) -> Vec<Vec<f64>> {
    let out: Vec<Vec<f64>> = x.iter().map(|row| rbf_row(&state.landmarks, row, gamma)).collect();
    finish_stage(x, out, append)
}"#
        }
        PlannedOp::ObliqueHinge { .. } => {
            r#"struct ObliqueHingeFeatures {
    w: Vec<Vec<f64>>,
    t: Vec<f64>,
}

{DOC}
fn oblique_hinge_features_fit(x: &[Vec<f64>], n_features: usize, seed: u64) -> ObliqueHingeFeatures {
    let cols = x[0].len();
    let mut rng = stage_rng(seed, cols, n_features);
    let inv = 1.0 / (cols as f64).sqrt();
    let mut w = Vec::new();
    for _ in 0..n_features {
        let mut row = Vec::new();
        for _ in 0..cols {
            row.push(rng.gaussian() * inv);
        }
        w.push(row);
    }
    let mut t = Vec::new();
    for wi in &w {
        let mut proj: Vec<f64> = x.iter().map(|row| dot(wi, row)).collect();
        proj.sort_by(|a, b| a.total_cmp(b));
        t.push(quantile(&proj, 0.5));
    }
    ObliqueHingeFeatures { w, t }
}

fn oblique_hinge_features_apply(state: &ObliqueHingeFeatures, x: &[Vec<f64>], append: bool) -> Vec<Vec<f64>> {
    let out: Vec<Vec<f64>> = x
        .iter()
        .map(|row| {
            (0..state.w.len())
                .map(|i| (dot(&state.w[i], row) - state.t[i]).max(0.0))
                .collect()
        })
        .collect();
    finish_stage(x, out, append)
}"#
        }
        PlannedOp::RowStats => {
            r#"{DOC}
fn row_summary_statistics_apply(x: &[Vec<f64>], append: bool) -> Vec<Vec<f64>> {
    let mut out = Vec::new();
    for row in x {
        let n = row.len() as f64;
        let mut mean = 0.0;
        for v in row {
            mean += *v;
        }
        mean /= n;
        let mut m2 = 0.0;
        let mut m3 = 0.0;
        let mut m4 = 0.0;
        let mut energy = 0.0;
        let mut mn = f64::INFINITY;
        let mut mx = f64::NEG_INFINITY;
        for v in row {
            let d = *v - mean;
            m2 += d * d;
            m3 += d * d * d;
            m4 += d * d * d * d;
            energy += *v * *v;
            if *v < mn {
                mn = *v;
            }
            if *v > mx {
                mx = *v;
            }
        }
        m2 /= n;
        m3 /= n;
        m4 /= n;
        energy /= n;
        let std = m2.sqrt();
        let skew = if m2 > 1e-24 { m3 / (m2 * std) } else { 0.0 };
        let kurt = if m2 > 1e-24 { m4 / (m2 * m2) - 3.0 } else { 0.0 };
        let mut s = row.clone();
        s.sort_by(|a, b| a.total_cmp(b));
        out.push(vec![mean, std, mn, mx, quantile(&s, 0.5), quantile(&s, 0.25),
                      quantile(&s, 0.75), skew, kurt, energy]);
    }
    finish_stage(x, out, append)
}"#
        }
        PlannedOp::PairProd { .. } => {
            r#"struct CorrelationScreenedProducts {
    pairs: Vec<(usize, usize)>,
}

{DOC}
fn correlation_screened_products_fit(x: &[Vec<f64>], y: &[f64], top_k: usize) -> CorrelationScreenedProducts {
    let cols = x[0].len();
    let top_k = top_k.min(cols);
    let mut scored: Vec<(usize, f64)> = Vec::new();
    for j in 0..cols {
        let col: Vec<f64> = x.iter().map(|row| row[j]).collect();
        scored.push((j, pearson(&col, y).abs()));
    }
    scored.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    let mut selected: Vec<usize> = scored[..top_k].iter().map(|&(j, _)| j).collect();
    selected.sort();
    let mut pairs = Vec::new();
    for a in 0..selected.len() {
        for b in (a + 1)..selected.len() {
            pairs.push((selected[a], selected[b]));
        }
    }
    CorrelationScreenedProducts { pairs }
}

fn correlation_screened_products_apply(state: &CorrelationScreenedProducts, x: &[Vec<f64>], append: bool) -> Vec<Vec<f64>> {
    let out: Vec<Vec<f64>> = x
        .iter()
        .map(|row| state.pairs.iter().map(|&(a, b)| row[a] * row[b]).collect())
        .collect();
    finish_stage(x, out, append)
}"#
        }
        PlannedOp::Symlog { .. } => {
            r#"{DOC}
fn signed_log_apply(x: &[Vec<f64>], c: f64, append: bool) -> Vec<Vec<f64>> {
    let out: Vec<Vec<f64>> = x
        .iter()
        .map(|row| row.iter().map(|v| m_asinh(v / c)).collect())
        .collect();
    finish_stage(x, out, append)
}"#
        }
        PlannedOp::Impute { .. } => {
            r#"struct ZeroImputer {
    fill: Vec<f64>,
    indicator_cols: Vec<usize>,
}

{DOC}
fn zero_imputer_fit(x: &[Vec<f64>], strategy: usize, value: f64, add_indicator: bool) -> ZeroImputer {
    let cols = x[0].len();
    let mut fill = vec![0.0; cols];
    let mut indicator_cols = Vec::new();
    for j in 0..cols {
        let mut vals: Vec<f64> = Vec::new();
        let mut missing = false;
        for row in x {
            if row[j] == 0.0 {
                missing = true;
            } else {
                vals.push(row[j]);
            }
        }
        fill[j] = if vals.is_empty() {
            0.0
        } else if strategy == 0 {
            let mut s = 0.0;
            for v in &vals {
                s += *v;
            }
            s / vals.len() as f64
        } else if strategy == 2 {
            value
        } else {
            vals.sort_by(|a, b| a.total_cmp(b));
            quantile(&vals, 0.5)
        };
        if missing && add_indicator && cols + indicator_cols.len() < 512 {
            indicator_cols.push(j);
        }
    }
    ZeroImputer { fill, indicator_cols }
}

fn zero_imputer_apply(state: &ZeroImputer, x: &[Vec<f64>], append: bool) -> Vec<Vec<f64>> {
    let mut out: Vec<Vec<f64>> = Vec::new();
    for row in x {
        let mut vals: Vec<f64> = (0..row.len())
            .map(|j| if row[j] == 0.0 { state.fill[j] } else { row[j] })
            .collect();
        for &j in state.indicator_cols.iter() {
            vals.push(if row[j] == 0.0 { 1.0 } else { 0.0 });
        }
        out.push(vals);
    }
    finish_stage(x, out, append)
}"#
        }
        PlannedOp::Spline { .. } => {
            r#"struct NaturalCubicSpline {
    knots: Vec<Vec<f64>>,
}

{DOC}
fn natural_cubic_spline_fit(x: &[Vec<f64>], knots: usize) -> NaturalCubicSpline {
    let mut fitted: Vec<Vec<f64>> = Vec::new();
    for j in 0..x[0].len() {
        let s = sorted_column(x, j);
        fitted.push(
            (1..=knots)
                .map(|i| quantile(&s, i as f64 / (knots + 1) as f64))
                .collect(),
        );
    }
    NaturalCubicSpline { knots: fitted }
}

fn natural_cubic_spline_apply(state: &NaturalCubicSpline, x: &[Vec<f64>], append: bool) -> Vec<Vec<f64>> {
    let mut out: Vec<Vec<f64>> = Vec::new();
    for row in x {
        let mut values: Vec<f64> = Vec::new();
        for (j, v) in row.iter().enumerate() {
            let xi = &state.knots[j];
            let k = xi.len();
            if xi[k - 1] - xi[0] < 1e-9 {
                values.push(*v);
            } else {
                values.push(*v);
                let pl = (v - xi[k - 1]).max(0.0);
                let pk = (v - xi[k - 2]).max(0.0);
                let dlast = (pk * pk * pk - pl * pl * pl) / (xi[k - 1] - xi[k - 2]).max(1e-9);
                for t in 0..(k - 2) {
                    let pt = (v - xi[t]).max(0.0);
                    let d = (pt * pt * pt - pl * pl * pl) / (xi[k - 1] - xi[t]).max(1e-9);
                    values.push(d - dlast);
                }
            }
        }
        out.push(values);
    }
    finish_stage(x, out, append)
}"#
        }
        PlannedOp::PairRatio { .. } => {
            r#"struct RatioScreenedPairs {
    pairs: Vec<(usize, usize)>,
    eps: Vec<f64>,
}

{DOC}
fn ratio_screened_pairs_fit(x: &[Vec<f64>], y: &[f64], top_k: usize) -> RatioScreenedPairs {
    let cols = x[0].len();
    let m = cols.min(32);
    let mut scored: Vec<(usize, f64)> = Vec::new();
    for j in 0..cols {
        let col: Vec<f64> = x.iter().map(|row| row[j]).collect();
        scored.push((j, pearson(&col, y).abs()));
    }
    scored.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    let mut screened: Vec<usize> = scored[..m].iter().map(|&(j, _)| j).collect();
    screened.sort();
    let mut guards: Vec<f64> = Vec::new();
    for &j in &screened {
        let mut s = 0.0;
        for row in x {
            s += row[j];
        }
        let mean = s / x.len() as f64;
        let mut v = 0.0;
        for row in x {
            let d = row[j] - mean;
            v += d * d;
        }
        let e = 0.01 * (v / x.len() as f64).sqrt();
        guards.push((e * e).max(1e-12));
    }
    let mut ranked_pairs: Vec<(usize, usize, f64, f64)> = Vec::new();
    for ai in 0..screened.len() {
        for bi in 0..screened.len() {
            if ai == bi {
                continue;
            }
            let a = screened[ai];
            let b = screened[bi];
            let eps = guards[bi];
            let ratio: Vec<f64> = x
                .iter()
                .map(|row| row[a] * row[b] / (row[b] * row[b] + eps))
                .collect();
            ranked_pairs.push((a, b, eps, pearson(&ratio, y).abs()));
        }
    }
    ranked_pairs.sort_by(|l, r| r.3.total_cmp(&l.3).then(l.0.cmp(&r.0)).then(l.1.cmp(&r.1)));
    ranked_pairs.truncate(top_k.min(ranked_pairs.len()));
    RatioScreenedPairs {
        pairs: ranked_pairs.iter().map(|&(a, b, _, _)| (a, b)).collect(),
        eps: ranked_pairs.iter().map(|&(_, _, e, _)| e).collect(),
    }
}

fn ratio_screened_pairs_apply(state: &RatioScreenedPairs, x: &[Vec<f64>], append: bool) -> Vec<Vec<f64>> {
    let mut out: Vec<Vec<f64>> = Vec::new();
    for row in x {
        let mut vals: Vec<f64> = Vec::new();
        for i in 0..state.pairs.len() {
            let (a, b) = state.pairs[i];
            vals.push(row[a] * row[b] / (row[b] * row[b] + state.eps[i]));
        }
        out.push(vals);
    }
    finish_stage(x, out, append)
}"#
        }
        PlannedOp::SelectCorr { .. } => {
            r#"struct ColumnSelect {
    indices: Vec<usize>,
}

{DOC}
fn column_select_fit(x: &[Vec<f64>], y: &[f64], k: usize, by_var: bool) -> ColumnSelect {
    let cols = x[0].len();
    let k = k.min(cols);
    let mut scored: Vec<(usize, f64)> = Vec::new();
    for j in 0..cols {
        let score = if by_var {
            let mut s = 0.0;
            for row in x {
                s += row[j];
            }
            let m = s / x.len() as f64;
            let mut v = 0.0;
            for row in x {
                let d = row[j] - m;
                v += d * d;
            }
            v / x.len() as f64
        } else {
            let col: Vec<f64> = x.iter().map(|row| row[j]).collect();
            pearson(&col, y).abs()
        };
        scored.push((j, score));
    }
    scored.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    let mut indices: Vec<usize> = scored[..k].iter().map(|&(j, _)| j).collect();
    indices.sort();
    ColumnSelect { indices }
}

fn column_select_apply(state: &ColumnSelect, x: &[Vec<f64>], append: bool) -> Vec<Vec<f64>> {
    let out: Vec<Vec<f64>> = x
        .iter()
        .map(|row| state.indices.iter().map(|&j| row[j]).collect())
        .collect();
    finish_stage(x, out, append)
}"#
        }
        PlannedOp::Sigmoid => {
            r#"{DOC}
fn logistic_squash_apply(x: &[Vec<f64>], append: bool) -> Vec<Vec<f64>> {
    let out: Vec<Vec<f64>> = x
        .iter()
        .map(|row| row.iter().map(|v| m_sigmoid(*v)).collect())
        .collect();
    finish_stage(x, out, append)
}"#
        }
        PlannedOp::Step { .. } => {
            r#"{DOC}
fn step_indicator_apply(x: &[Vec<f64>], t: f64, append: bool) -> Vec<Vec<f64>> {
    let out: Vec<Vec<f64>> = x
        .iter()
        .map(|row| row.iter().map(|v| if *v > t { 1.0 } else { 0.0 }).collect())
        .collect();
    finish_stage(x, out, append)
}"#
        }
        PlannedOp::NormRow => {
            r#"{DOC}
fn unit_norm_rows_apply(x: &[Vec<f64>], append: bool) -> Vec<Vec<f64>> {
    let mut out = Vec::new();
    for row in x {
        let mut s = 0.0;
        for v in row {
            s += *v * *v;
        }
        let inv = 1.0 / s.sqrt().max(1e-9);
        out.push(row.iter().map(|v| v * inv).collect());
    }
    finish_stage(x, out, append)
}"#
        }
        PlannedOp::Polar => {
            r#"{DOC}
fn polar_coordinates_apply(x: &[Vec<f64>], append: bool) -> Vec<Vec<f64>> {
    let mut out = Vec::new();
    for row in x {
        let d = row.len();
        let mut values = Vec::with_capacity(d);
        let mut j = 0;
        while j + 1 < d {
            let a = row[j];
            let b = row[j + 1];
            values.push((a * a + b * b).sqrt());
            values.push(b.atan2(a));
            j += 2;
        }
        if d % 2 == 1 {
            values.push(row[d - 1]);
        }
        out.push(values);
    }
    finish_stage(x, out, append)
}"#
        }
        PlannedOp::Lag { .. } => {
            r#"{DOC}
fn shift_columns_apply(x: &[Vec<f64>], k: usize, append: bool) -> Vec<Vec<f64>> {
    let out: Vec<Vec<f64>> = x
        .iter()
        .map(|row| (0..row.len()).map(|j| if j >= k { row[j - k] } else { 0.0 }).collect())
        .collect();
    finish_stage(x, out, append)
}"#
        }
        PlannedOp::Roll { .. } => {
            r#"{DOC}
fn rotate_columns_apply(x: &[Vec<f64>], k: usize, append: bool) -> Vec<Vec<f64>> {
    let mut out = Vec::new();
    for row in x {
        let d = row.len();
        let s = k % d;
        out.push((0..d).map(|j| row[(j + d - s) % d]).collect());
    }
    finish_stage(x, out, append)
}"#
        }
        PlannedOp::FftMag { .. } => {
            r#"{DOC}
fn fourier_magnitudes_fit(x: &[Vec<f64>], k: usize) -> usize {
    let cols = x[0].len();
    k.clamp(1, (cols / 2).max(1))
}

fn fourier_magnitudes_apply(k: usize, x: &[Vec<f64>], append: bool) -> Vec<Vec<f64>> {
    let out: Vec<Vec<f64>> = x.iter().map(|row| dft_magnitudes(row, k)).collect();
    finish_stage(x, out, append)
}"#
        }
        PlannedOp::RowSpectral => {
            r#"{DOC}
fn row_spectral_summary_apply(x: &[Vec<f64>], append: bool) -> Vec<Vec<f64>> {
    let mut out = Vec::new();
    for row in x {
        let nf = row.len() / 2;
        if nf == 0 {
            out.push(vec![0.0, 0.0, 0.0, 0.0, 0.0]);
            continue;
        }
        let mags = dft_magnitudes(row, nf);
        let mut total = 0.0;
        for m in &mags {
            total += *m;
        }
        let denom = total.max(1e-12);
        let mut ent = 0.0;
        let mut m1 = 0.0;
        let mut m2 = 0.0;
        let mut peak = 0usize;
        for (i, m) in mags.iter().enumerate() {
            let p = m / denom;
            if p > 1e-12 {
                ent -= p * p.ln();
            }
            let f = (i + 1) as f64;
            m1 += f * p;
            m2 += f * f * p;
            if *m > mags[peak] {
                peak = i;
            }
        }
        out.push(vec![ent, m1, m2, (peak + 1) as f64, mags[peak]]);
    }
    finish_stage(x, out, append)
}"#
        }
        PlannedOp::RowDiffStats => {
            r#"{DOC}
fn row_difference_statistics_apply(x: &[Vec<f64>], append: bool) -> Vec<Vec<f64>> {
    let mut out = Vec::new();
    for row in x {
        let d = row.len();
        if d < 2 {
            out.push(vec![0.0; 8]);
            continue;
        }
        let nd = d - 1;
        let diffs: Vec<f64> = (0..nd).map(|j| row[j + 1] - row[j]).collect();
        let mut mean = 0.0;
        for v in &diffs {
            mean += *v;
        }
        mean /= nd as f64;
        let mut m2 = 0.0;
        let mut mean_abs = 0.0;
        let mut energy = 0.0;
        let mut mn = f64::INFINITY;
        let mut mx = f64::NEG_INFINITY;
        let mut signs = 0usize;
        for (j, v) in diffs.iter().enumerate() {
            let dd = *v - mean;
            m2 += dd * dd;
            mean_abs += v.abs();
            energy += *v * *v;
            if *v < mn {
                mn = *v;
            }
            if *v > mx {
                mx = *v;
            }
            if j > 0 && diffs[j - 1] * *v < 0.0 {
                signs += 1;
            }
        }
        let std = (m2 / nd as f64).sqrt();
        let auto = pearson(&row[..nd], &row[1..]);
        out.push(vec![mean, std, mn, mx, mean_abs / nd as f64, signs as f64 / nd as f64,
                      auto, energy / nd as f64]);
    }
    finish_stage(x, out, append)
}"#
        }
        PlannedOp::SvdProject { .. } => {
            r#"struct PrincipalComponentProjection {
    means: Vec<f64>,
    components: Vec<Vec<f64>>,
}

{DOC}
fn principal_component_projection_fit(x: &[Vec<f64>], k: usize, iters: usize, seed: u64) -> PrincipalComponentProjection {
    let cols = x[0].len();
    let n = x.len();
    let k = k.min(cols);
    let mut means = vec![0.0; cols];
    for j in 0..cols {
        let mut s = 0.0;
        for row in x {
            s += row[j];
        }
        means[j] = s / n as f64;
    }
    let mut rng = stage_rng(seed, cols, k);
    let mut components: Vec<Vec<f64>> = Vec::with_capacity(k);
    for _ in 0..k {
        let mut v: Vec<f64> = (0..cols).map(|_| rng.gaussian()).collect();
        normalize(&mut v);
        for _ in 0..iters {
            let mut w = vec![0.0; cols];
            for row in x {
                let mut s = 0.0;
                for j in 0..cols {
                    s += (row[j] - means[j]) * v[j];
                }
                for j in 0..cols {
                    w[j] += s * (row[j] - means[j]);
                }
            }
            for u in &components {
                let dd = dot(&w, u);
                for j in 0..cols {
                    w[j] -= dd * u[j];
                }
            }
            normalize(&mut w);
            v = w;
        }
        let mut best = 0;
        for j in 1..cols {
            if v[j].abs() > v[best].abs() {
                best = j;
            }
        }
        if v[best] < 0.0 {
            for e in v.iter_mut() {
                *e = -*e;
            }
        }
        components.push(v);
    }
    PrincipalComponentProjection { means, components }
}

fn principal_component_projection_apply(state: &PrincipalComponentProjection, x: &[Vec<f64>], append: bool) -> Vec<Vec<f64>> {
    let mut out = Vec::new();
    for row in x {
        let mut values = Vec::new();
        for c in &state.components {
            let mut s = 0.0;
            for j in 0..state.means.len().min(row.len()) {
                s += (row[j] - state.means[j]) * c[j];
            }
            values.push(s);
        }
        out.push(values);
    }
    finish_stage(x, out, append)
}"#
        }
        PlannedOp::Nystrom { .. } => {
            r#"struct NystromLandmarkFeatures {
    landmarks: Vec<Vec<f64>>,
}

{DOC}
fn nystrom_landmark_features_fit(x: &[Vec<f64>], k: usize, seed: u64) -> NystromLandmarkFeatures {
    let cols = x[0].len();
    let n = x.len();
    let k = k.min(n);
    let mut rng = stage_rng(seed, cols, k);
    let mut idx: Vec<usize> = (0..n).collect();
    for i in 0..k {
        let j = i + (rng.next_u32() as usize) % (n - i);
        idx.swap(i, j);
    }
    let landmarks: Vec<Vec<f64>> = idx[..k].iter().map(|&i| x[i].clone()).collect();
    NystromLandmarkFeatures { landmarks }
}

fn nystrom_landmark_features_apply(state: &NystromLandmarkFeatures, x: &[Vec<f64>], gamma: f64, append: bool) -> Vec<Vec<f64>> {
    let out: Vec<Vec<f64>> = x.iter().map(|row| rbf_row(&state.landmarks, row, gamma)).collect();
    finish_stage(x, out, append)
}"#
        }
        PlannedOp::MapExpr { .. } | PlannedOp::DeriveExpr { .. } => {
            unreachable!("formula stages are emitted per stage index")
        }
        PlannedOp::TargetBinEncode { .. } => {
            r#"struct SmoothedTargetBinEncoder {
    edges: Vec<Vec<f64>>,
    enc: Vec<Vec<f64>>,
}

{DOC}
fn smoothed_target_bin_encoder_fit(x: &[Vec<f64>], y: &[f64], bins: usize, prior: f64) -> SmoothedTargetBinEncoder {
    let edges = quantile_edges(x, bins);
    let mut y_mean = 0.0;
    for v in y {
        y_mean += *v;
    }
    y_mean /= y.len().max(1) as f64;
    let mut enc = Vec::new();
    for j in 0..x[0].len() {
        let mut sums = vec![0.0; bins];
        let mut counts = vec![0.0; bins];
        for r in 0..x.len() {
            let b = lower_bound(&edges[j], x[r][j]).min(bins - 1);
            sums[b] += y[r];
            counts[b] += 1.0;
        }
        enc.push((0..bins).map(|b| (sums[b] + prior * y_mean) / (counts[b] + prior).max(1e-9)).collect());
    }
    SmoothedTargetBinEncoder { edges, enc }
}

fn smoothed_target_bin_encoder_apply(state: &SmoothedTargetBinEncoder, x: &[Vec<f64>], append: bool) -> Vec<Vec<f64>> {
    let mut out = Vec::new();
    for row in x {
        let mut values = Vec::new();
        for (j, v) in row.iter().enumerate() {
            let bins = state.enc[j].len();
            values.push(state.enc[j][lower_bound(&state.edges[j], *v).min(bins - 1)]);
        }
        out.push(values);
    }
    finish_stage(x, out, append)
}"#
        }
    };
    body.replace("{DOC}", &doc)
}

/// Elementwise `map` stage: one pair of functions per stage index (the
/// expression is baked in, so stages cannot share code).
fn map_src(n: usize, source: &str, expr: &crate::mathexpr::MathAst) -> String {
    let features = expr.features();
    let mut lines: Vec<String> = Vec::new();
    lines.push(format!(
        "/// f(x, row) = {source}, with guarded arithmetic; x is one cell of row."
    ));
    lines.push(format!(
        "fn elementwise_formula{n}(x: f64, row: &[f64]) -> f64 {{"
    ));
    if let Some(maxc) = features.max_col {
        lines.push(format!("    if row.len() <= {maxc} {{"));
        lines.push("        return 0.0;".to_string());
        lines.push("    }".to_string());
    } else if features.vars.is_empty()
        || features
            .vars
            .iter()
            .all(|v| matches!(v, crate::mathexpr::MathVar::X))
    {
        lines.push("    let _ = row;".to_string());
    }
    if !features.uses_x {
        lines.push("    let _ = x;".to_string());
    }
    if features.funcs.contains(&crate::mathexpr::MathFunc::Clip) {
        lines.push("    let mut math_failed = false;".to_string());
        lines.push(format!("    let value = {};", expr.to_rust()));
        lines.push("    if math_failed { return 0.0; }".to_string());
    } else {
        lines.push(format!("    let value = {};", expr.to_rust()));
    }
    lines.push("    if value.is_finite() {".to_string());
    lines.push("        value.clamp(-1e12, 1e12)".to_string());
    lines.push("    } else {".to_string());
    lines.push("        0.0".to_string());
    lines.push("    }".to_string());
    lines.push("}".to_string());
    lines.push(String::new());
    lines.push(format!(
        "/// Stage {n}: map every cell through elementwise_formula{n}."
    ));
    lines.push(format!(
        "fn elementwise_formula{n}_apply(x: &[Vec<f64>], append: bool) -> Vec<Vec<f64>> {{"
    ));
    lines.push("    let mut out = Vec::new();".to_string());
    lines.push("    for row in x {".to_string());
    lines.push(format!(
        "        out.push(row.iter().map(|v| elementwise_formula{n}(*v, row)).collect());"
    ));
    lines.push("    }".to_string());
    lines.push("    finish_stage(x, out, append)".to_string());
    lines.push("}".to_string());
    lines.join("\n")
}

/// `derive` stage: reduce each row to one derived column.
fn derive_src(n: usize, source: &str, expr: &crate::mathexpr::MathAst) -> String {
    let features = expr.features();
    let mut lines: Vec<String> = Vec::new();
    lines.push(format!("/// g(row) = {source}, with guarded arithmetic."));
    lines.push(format!("fn derived_column{n}(row: &[f64]) -> f64 {{"));
    if features.uses_x {
        lines.push(
            "    // The formula names the element variable x, which has no meaning for a whole"
                .to_string(),
        );
        lines.push("    // row, so every row maps to 0.0 (mirroring the reference).".to_string());
        lines.push("    let _ = row;".to_string());
        lines.push("    0.0".to_string());
    } else {
        if let Some(maxc) = features.max_col {
            lines.push(format!("    if row.len() <= {maxc} {{"));
            lines.push("        return 0.0;".to_string());
            lines.push("    }".to_string());
        } else if features.vars.is_empty() {
            lines.push("    let _ = row;".to_string());
        }
        if features.funcs.contains(&crate::mathexpr::MathFunc::Clip) {
            lines.push("    let mut math_failed = false;".to_string());
            lines.push(format!("    let value = {};", expr.to_rust()));
            lines.push("    if math_failed { return 0.0; }".to_string());
        } else {
            lines.push(format!("    let value = {};", expr.to_rust()));
        }
        lines.push("    if value.is_finite() {".to_string());
        lines.push("        value.clamp(-1e12, 1e12)".to_string());
        lines.push("    } else {".to_string());
        lines.push("        0.0".to_string());
        lines.push("    }".to_string());
    }
    lines.push("}".to_string());
    lines.push(String::new());
    lines.push(format!(
        "/// Stage {n}: reduce every row to the single column derived_column{n}(row)."
    ));
    lines.push(format!(
        "fn derived_column{n}_apply(x: &[Vec<f64>], append: bool) -> Vec<Vec<f64>> {{"
    ));
    lines.push(format!(
        "    let out: Vec<Vec<f64>> = x.iter().map(|row| vec![derived_column{n}(row)]).collect();"
    ));
    lines.push("    finish_stage(x, out, append)".to_string());
    lines.push("}".to_string());
    lines.join("\n")
}

// --------------------------------- heads -----------------------------------

fn head_srcs(head: &PlannedHead) -> Vec<String> {
    let doc = doc_comment(&format!("This head is {}", crate::spec::head_doc(head)));
    head_srcs_with_doc(head, &doc)
}

fn head_srcs_with_doc(head: &PlannedHead, doc: &str) -> Vec<String> {
    match head {
        PlannedHead::Ridge { .. } => vec![RIDGE_SRC.replace("{DOC}", doc)],
        PlannedHead::RidgeGcv => vec![GCV_RIDGE_SRC.replace("{DOC}", doc)],
        PlannedHead::ElasticNet { .. } => vec![ELASTIC_SRC.replace("{DOC}", doc)],
        PlannedHead::Gbm(_) => vec![GBM_SRC.replace("{DOC}", doc)],
        PlannedHead::Blend { .. } => vec![
            RIDGE_SRC.replace(
                "{DOC}",
                "/// Ridge regression member of the blend (see blended_regression_fit).",
            ),
            GBM_SRC.replace(
                "{DOC}",
                "/// Boosted-trees member of the blend (see blended_regression_fit).",
            ),
            BLEND_SRC.replace("{DOC}", doc),
        ],
        PlannedHead::HistGbm2 { .. } => vec![HISTGBM2_SRC.replace("{DOC}", doc)],
        PlannedHead::Krr { .. } => vec![KRR_SRC.replace("{DOC}", doc)],
        PlannedHead::Gpr { .. } => vec![GPR_SRC.replace("{DOC}", doc)],
        PlannedHead::GprKernel { .. } => vec![GPRK_SRC.replace("{DOC}", doc)],
        PlannedHead::Ksmooth { .. } => vec![KSMOOTH_SRC.replace("{DOC}", doc)],
        PlannedHead::Stack => vec![
            RIDGE_SRC.replace(
                "{DOC}",
                "/// Ridge regression member of the library (see greedy_ensemble_selection_fit).",
            ),
            ELASTIC_SRC.replace(
                "{DOC}",
                "/// Elastic net member of the library (see greedy_ensemble_selection_fit).",
            ),
            HISTGBM2_SRC.replace(
                "{DOC}",
                "/// Boosted-trees member of the library (see greedy_ensemble_selection_fit).",
            ),
            KRR_SRC.replace(
                "{DOC}",
                "/// Kernel ridge member of the library (see greedy_ensemble_selection_fit).",
            ),
            stack_src().replace("{DOC}", doc),
        ],
        // Portfolio sections depend on the resolved component roster and are
        // assembled by portfolio_srcs.
        PlannedHead::Portfolio(_) => unreachable!("portfolio sections come from portfolio_srcs"),
    }
}

/// The stack head bakes the library defaults in as literals; they are read from
/// battle-runtime so the three implementations cannot drift apart.
fn stack_src() -> String {
    STACK_SRC
        .replace("{LEAVES}", &crate::boost2::DEFAULT_LEAVES.to_string())
        .replace("{LR}", &fmt_f64(crate::boost2::DEFAULT_LR))
        .replace("{BAGS}", &crate::boost2::DEFAULT_BAGS.to_string())
        .replace("{M}", &crate::krr::DEFAULT_M.to_string())
}

// ---------------------- internally validated ensemble ----------------------

/// Every section the compact portfolio head needs, in deterministic order
/// (the exact twin of emit_py::portfolio_srcs; both mirror portfolio.rs
/// statement by statement).
fn portfolio_srcs(pp: &crate::spec::PortfolioPlan, seen: &mut Vec<&'static str>) -> Vec<String> {
    use crate::spec::PlannedMemberKind;
    let mut out: Vec<String> = Vec::new();
    for member in &pp.members {
        if let PlannedMemberKind::Pipeline { stages, .. } = &member.kind {
            for stage in stages {
                let name = op_fn_name(&stage.op);
                if !seen.contains(&name) {
                    seen.push(name);
                    out.push(op_src(&stage.op));
                }
            }
        }
    }
    // The wide-data projection view (an svd_project twin, k=256 at fit time).
    let proj_op = PlannedOp::SvdProject {
        k: 256,
        iters: 20,
        seed: 7,
    };
    let proj_name = op_fn_name(&proj_op);
    if !seen.contains(&proj_name) {
        seen.push(proj_name);
        out.push(op_src(&proj_op));
    }
    let mut seen_heads: Vec<&'static str> = Vec::new();
    let mut seen_aux: Vec<&'static str> = Vec::new();
    for member in &pp.members {
        if let PlannedMemberKind::Pipeline { head, .. } = &member.kind {
            let hname = head_fn_name(head);
            if !seen_heads.contains(&hname) {
                seen_heads.push(hname);
                let doc = "/// Component model of the internally validated ensemble (see \
                           internally_validated_ensemble_fit)."
                    .to_string();
                for src in head_srcs_with_doc(head, &doc) {
                    out.push(rs_strip_dup_aux(src, &mut seen_aux));
                }
            }
        }
    }
    out.push(PF_ROW_SIGNATURE_SRC.to_string());
    out.push(PF_BLEND_SOLVER_SRC.to_string());
    if pp
        .members
        .iter()
        .any(|m| matches!(m.kind, PlannedMemberKind::GroupMean))
    {
        out.push(PF_DUP_AVERAGE_SRC.to_string());
    }
    for (i, member) in pp.members.iter().enumerate() {
        out.push(rs_component_src(i + 1, member));
    }
    out.push(rs_component_dispatch(pp));
    out.push(rs_portfolio_driver(pp));
    out
}

/// Head snippets self-carry two small shared helpers (seeded_rows,
/// chol_inverse) because a single-pipe artifact only ever holds one of these
/// heads. A portfolio can hold several, so snippets after the first drop the
/// duplicate definitions (their bodies are byte-identical across snippets —
/// Rust would reject the duplicates outright).
fn rs_strip_dup_aux(src: String, seen_aux: &mut Vec<&'static str>) -> String {
    let mut out = src;
    for marker in ["fn seeded_rows(", "fn chol_inverse("] {
        if let Some(pos) = out.find(marker) {
            if seen_aux.contains(&marker) {
                // Extend back over the doc-comment lines directly above.
                let mut start = out[..pos].rfind('\n').map(|i| i + 1).unwrap_or(0);
                while start > 0 {
                    let prev_start = out[..start - 1].rfind('\n').map(|i| i + 1).unwrap_or(0);
                    if out[prev_start..start].starts_with("///") {
                        start = prev_start;
                    } else {
                        break;
                    }
                }
                // Cut through the closing brace at column 0 and the blank
                // separator that followed.
                let close = out[pos..]
                    .find("\n}")
                    .map(|i| pos + i + 2)
                    .unwrap_or(out.len());
                let mut end = close;
                while out[end..].starts_with('\n') {
                    end += 1;
                }
                out.replace_range(start..end, "");
            } else {
                seen_aux.push(marker);
            }
        }
    }
    out
}

/// Short natural-language title of one component (op/head shorts joined).
fn rs_member_title(member: &crate::spec::PlannedMember) -> String {
    match &member.kind {
        crate::spec::PlannedMemberKind::GroupMean => {
            "shrunk per-duplicate-row averages".to_string()
        }
        crate::spec::PlannedMemberKind::Pipeline { stages, head } => {
            let mut parts: Vec<&str> = Vec::new();
            for s in stages {
                parts.push(crate::spec::op_short(&s.op));
            }
            parts.push(crate::spec::head_short(head));
            parts.join(", then ")
        }
    }
}

fn rs_component_src(cid: usize, member: &crate::spec::PlannedMember) -> String {
    use crate::spec::PlannedMemberKind;
    let title = rs_member_title(member);
    let mut lines: Vec<String> = Vec::new();
    match &member.kind {
        PlannedMemberKind::GroupMean => {
            lines.push(format!("/// Fit component {cid}: {title}."));
            lines.push(format!(
                "fn component{cid}_fit(x: &[Vec<f64>], y: &[f64]) -> DuplicateRowAverage {{"
            ));
            lines.push("    duplicate_row_average_fit(x, y)".to_string());
            lines.push("}".to_string());
            lines.push(String::new());
            lines.push(format!(
                "fn component{cid}_predict(state: &DuplicateRowAverage, x: &[Vec<f64>]) -> Vec<f64> {{"
            ));
            lines.push("    duplicate_row_average_predict(state, x)".to_string());
            lines.push("}".to_string());
        }
        PlannedMemberKind::Pipeline { stages, head } => {
            let (head_fit, head_predict, head_ty) = head_calls(head);
            lines.push(format!("struct Component{cid} {{"));
            let mut field_names: Vec<String> = Vec::new();
            for (i, stage) in stages.iter().enumerate() {
                if let Some(ty) = op_struct_name(&stage.op) {
                    lines.push(format!("    stage{}: {ty},", i + 1));
                    field_names.push(format!("stage{}", i + 1));
                }
            }
            lines.push(format!("    head: {head_ty},"));
            lines.push("}".to_string());
            lines.push(String::new());
            lines.push(format!("/// Fit component {cid}: {title}."));
            lines.push(format!(
                "fn component{cid}_fit(x: &[Vec<f64>], y: &[f64]) -> Component{cid} {{"
            ));
            lines.push("    let x = sanitize(x);".to_string());
            for (i, stage) in stages.iter().enumerate() {
                let n = i + 1;
                let (fit_call, apply_call) = op_calls(&stage.op, n);
                let state = format!("stage{n}");
                if let Some(fit_call) = fit_call {
                    lines.push(format!("    let {state} = {fit_call};"));
                }
                let apply = apply_call
                    .replace("{state_val}", &state)
                    .replace("{state}", &format!("&{state}"))
                    .replace("{append}", rs_bool(stage.append));
                lines.push(format!("    let x = {apply};"));
            }
            lines.push(format!("    let head = {head_fit};"));
            let mut init = field_names.join(", ");
            if !init.is_empty() {
                init.push_str(", ");
            }
            lines.push(format!("    Component{cid} {{ {init}head }}"));
            lines.push("}".to_string());
            lines.push(String::new());
            lines.push(format!(
                "fn component{cid}_predict(state: &Component{cid}, x: &[Vec<f64>]) -> Vec<f64> {{"
            ));
            lines.push("    let x = sanitize(x);".to_string());
            for (i, stage) in stages.iter().enumerate() {
                let n = i + 1;
                let (_, apply_call) = op_calls(&stage.op, n);
                let apply = apply_call
                    .replace("{state_val}", &format!("state.stage{n}"))
                    .replace("{state}", &format!("&state.stage{n}"))
                    .replace("{append}", rs_bool(stage.append));
                lines.push(format!("    let x = {apply};"));
            }
            let predict = head_predict.replace("self.head", "state.head");
            lines.push(format!("    {predict}"));
            lines.push("}".to_string());
        }
    }
    lines.join("\n")
}

fn rs_component_state_variant(cid: usize, member: &crate::spec::PlannedMember) -> String {
    match member.kind {
        crate::spec::PlannedMemberKind::GroupMean => format!("C{cid}(DuplicateRowAverage)"),
        crate::spec::PlannedMemberKind::Pipeline { .. } => format!("C{cid}(Component{cid})"),
    }
}

fn rs_component_dispatch(pp: &crate::spec::PortfolioPlan) -> String {
    let mut lines: Vec<String> = Vec::new();
    lines.push("/// The fitted state of one ensemble component.".to_string());
    lines.push("enum ComponentState {".to_string());
    for (i, member) in pp.members.iter().enumerate() {
        lines.push(format!(
            "    {},",
            rs_component_state_variant(i + 1, member)
        ));
    }
    lines.push("}".to_string());
    lines.push(String::new());
    lines.push("/// Fit one component pipeline by its roster index.".to_string());
    lines.push(
        "fn component_fit(index: usize, x: &[Vec<f64>], y: &[f64]) -> ComponentState {".to_string(),
    );
    lines.push("    match index {".to_string());
    for i in 1..pp.members.len() {
        lines.push(format!(
            "        {i} => ComponentState::C{i}(component{i}_fit(x, y)),"
        ));
    }
    let last = pp.members.len();
    lines.push(format!(
        "        _ => ComponentState::C{last}(component{last}_fit(x, y)),"
    ));
    lines.push("    }".to_string());
    lines.push("}".to_string());
    lines.push(String::new());
    lines.push("/// Predict with a fitted component.".to_string());
    lines.push(
        "fn component_predict(state: &ComponentState, x: &[Vec<f64>]) -> Vec<f64> {".to_string(),
    );
    lines.push("    match state {".to_string());
    for i in 1..=pp.members.len() {
        lines.push(format!(
            "        ComponentState::C{i}(s) => component{i}_predict(s, x),"
        ));
    }
    lines.push("    }".to_string());
    lines.push("}".to_string());
    lines.join("\n")
}

/// One roster line (with its activation condition) per visible component.
fn rs_roster_lines(pp: &crate::spec::PortfolioPlan) -> String {
    use crate::members::Gate;
    let mut lines: Vec<String> = Vec::new();
    for (i, member) in pp.members.iter().enumerate() {
        if member.hidden {
            continue;
        }
        let cid = i + 1;
        let append = format!("roster.push(({cid}, {}));", rs_bool(member.proj_view));
        match member.gate {
            Gate::Always => lines.push(format!("        {append}")),
            Gate::NAtLeast(t) => {
                lines.push(format!("        if n >= {t} {{"));
                lines.push(format!("            {append}"));
                lines.push("        }".to_string());
            }
            Gate::NBelow(t) => {
                lines.push(format!("        if n < {t} {{"));
                lines.push(format!("            {append}"));
                lines.push("        }".to_string());
            }
            Gate::PAtMost(t) => {
                lines.push(format!("        if p <= {t} {{"));
                lines.push(format!("            {append}"));
                lines.push("        }".to_string());
            }
            Gate::DupFracAbove(t) => {
                lines.push(format!("        if dup > {} {{", fmt_f64(t)));
                lines.push(format!("            {append}"));
                lines.push("        }".to_string());
            }
        }
    }
    lines.join("\n")
}

fn rs_portfolio_driver(pp: &crate::spec::PortfolioPlan) -> String {
    let doc = doc_comment(&format!(
        "This head is {}",
        crate::spec::head_doc(&PlannedHead::Portfolio(pp.clone()))
    ));
    let repeats_expr = if pp.repeats_auto {
        "if n < 120 { 3 } else if n < 300 { 2 } else { 1 }"
    } else {
        "1"
    };
    PF_DRIVER_SRC
        .replace("{DOC}", &doc)
        .replace("{ROSTER}", &rs_roster_lines(pp))
        .replace("{FB_CID}", &(pp.fallback + 1).to_string())
        .replace("{FB_PROJ}", rs_bool(pp.members[pp.fallback].proj_view))
        .replace("{REPEATS}", repeats_expr)
        .replace("{FOLDS}", &pp.folds.to_string())
}

const PF_ROW_SIGNATURE_SRC: &str = r#"/// Row key for duplicate detection: every value rounded to nine decimals
/// (half away from zero; negative zero normalized), stored as raw bits.
fn row_signature(row: &[f64]) -> Vec<u64> {
    row.iter()
        .map(|v| ((v * 1e9 + 0.5).floor() / 1e9 + 0.0).to_bits())
        .collect()
}

/// Share of rows that duplicate an earlier row under row_signature.
fn duplicate_row_share(x: &[Vec<f64>]) -> f64 {
    if x.is_empty() {
        return 0.0;
    }
    let mut distinct: std::collections::BTreeSet<Vec<u64>> = std::collections::BTreeSet::new();
    for row in x {
        distinct.insert(row_signature(row));
    }
    1.0 - distinct.len() as f64 / x.len() as f64
}"#;

const PF_BLEND_SOLVER_SRC: &str = r#"/// min 0.5 w'Gw - c'w subject to w >= 0 by exact cyclic coordinate descent
/// (at most 500 sweeps; stops when the largest coordinate move is under 1e-10).
fn non_negative_blend_weights(g: &[f64], c: &[f64], k: usize) -> Vec<f64> {
    let mut w = vec![0.0; k];
    for _ in 0..500 {
        let mut max_d = 0.0;
        for j in 0..k {
            let gjj = g[j * k + j];
            if gjj <= 1e-12 {
                continue;
            }
            let mut acc = 0.0;
            for l in 0..k {
                acc += g[j * k + l] * w[l];
            }
            let mut nj = w[j] + (c[j] - acc) / gjj;
            if nj < 0.0 {
                nj = 0.0;
            }
            let d = (nj - w[j]).abs();
            if d > 0.0 {
                w[j] = nj;
                if d > max_d {
                    max_d = d;
                }
            }
        }
        if max_d < 1e-10 {
            break;
        }
    }
    w
}"#;

const PF_DUP_AVERAGE_SRC: &str = r#"/// Per-duplicate-row target averages with m-estimate shrinkage (strength 5)
/// toward the global mean; unknown rows fall back to the global mean.
struct DuplicateRowAverage {
    keys: Vec<Vec<u64>>,
    values: Vec<f64>,
    fallback: f64,
}

fn duplicate_row_average_fit(x: &[Vec<f64>], y: &[f64]) -> DuplicateRowAverage {
    let n = x.len().min(y.len());
    let mut g_mean = 0.0;
    for value in y.iter().take(n) {
        g_mean += value;
    }
    g_mean /= n.max(1) as f64;
    let mut sums: std::collections::BTreeMap<Vec<u64>, (f64, f64)> =
        std::collections::BTreeMap::new();
    for i in 0..n {
        let e = sums.entry(row_signature(&x[i])).or_insert((0.0, 0.0));
        e.0 += y[i];
        e.1 += 1.0;
    }
    let mut keys = Vec::with_capacity(sums.len());
    let mut values = Vec::with_capacity(sums.len());
    for (key, (s, count)) in sums {
        keys.push(key);
        values.push((s + 5.0 * g_mean) / (count + 5.0));
    }
    DuplicateRowAverage { keys, values, fallback: g_mean }
}

fn duplicate_row_average_predict(state: &DuplicateRowAverage, x: &[Vec<f64>]) -> Vec<f64> {
    x.iter()
        .map(|row| {
            let key = row_signature(row);
            match state.keys.binary_search(&key) {
                Ok(i) => state.values[i],
                Err(_) => state.fallback,
            }
        })
        .collect()
}"#;

const PF_DRIVER_SRC: &str = r#"/// The fitted ensemble: the constant-column mask, the optional projection
/// view for wide data, the fitted components with their blend weights, and
/// the padded training target range every prediction is clipped to.
struct InternallyValidatedEnsemble {
    mean_only: bool,
    y_mean: f64,
    width: usize,
    keep: Vec<usize>,
    proj: Option<PrincipalComponentProjection>,
    entries: Vec<EnsembleEntry>,
    clip_lo: f64,
    clip_hi: f64,
}

struct EnsembleEntry {
    projected: bool,
    weight: f64,
    state: Option<ComponentState>,
}

fn mean_only_blend(y_mean: f64, width: usize) -> InternallyValidatedEnsemble {
    InternallyValidatedEnsemble {
        mean_only: true,
        y_mean,
        width,
        keep: Vec::new(),
        proj: None,
        entries: Vec::new(),
        clip_lo: y_mean,
        clip_hi: y_mean,
    }
}

{DOC}
fn internally_validated_ensemble_fit(x: &[Vec<f64>], y: &[f64]) -> InternallyValidatedEnsemble {
    let n = x.len().min(y.len());
    if n == 0 {
        return mean_only_blend(0.0, 0);
    }
    let width = x[0].len();
    let mut rows: Vec<Vec<f64>> = Vec::with_capacity(n);
    for row in x.iter().take(n) {
        let mut r = row.clone();
        r.resize(width, 0.0);
        rows.push(r);
    }
    let x = sanitize(&rows);
    let y: Vec<f64> = y[..n]
        .iter()
        .map(|v| if v.is_finite() { *v } else { 0.0 })
        .collect();
    let mut y_mean = 0.0;
    for v in &y {
        y_mean += v;
    }
    y_mean /= n as f64;
    let mut y_var = 0.0;
    for v in &y {
        let d = v - y_mean;
        y_var += d * d;
    }
    y_var /= n as f64;
    let y_std = y_var.sqrt();
    if n < 5 || y_std < 1e-12 {
        return mean_only_blend(y_mean, width);
    }
    let mut keep: Vec<usize> = Vec::new();
    for j in 0..width {
        let mut m = 0.0;
        for row in &x {
            m += row[j];
        }
        m /= n as f64;
        let mut v = 0.0;
        for row in &x {
            let d = row[j] - m;
            v += d * d;
        }
        if (v / n as f64).sqrt() > 1e-12 {
            keep.push(j);
        }
    }
    if keep.is_empty() {
        return mean_only_blend(y_mean, width);
    }
    let x_raw: Vec<Vec<f64>> = x
        .iter()
        .map(|row| keep.iter().map(|&j| row[j]).collect())
        .collect();
    let p = keep.len();
    let dup = duplicate_row_share(&x_raw);
    let mut roster: Vec<(usize, bool)> = Vec::new();
    if n < 20 {
        roster.push(({FB_CID}, {FB_PROJ}));
    } else {
{ROSTER}
    }
    if roster.is_empty() {
        roster.push(({FB_CID}, {FB_PROJ}));
    }
    let m_count = roster.len();
    if m_count > 12 {
        eprintln!("too many active components");
        std::process::exit(2);
    }
    let mut use_projection = false;
    if p > 300 {
        for entry in &roster {
            if entry.1 {
                use_projection = true;
            }
        }
    }
    let mut proj: Option<PrincipalComponentProjection> = None;
    let mut x_proj: Vec<Vec<f64>> = Vec::new();
    if use_projection {
        let fitted = principal_component_projection_fit(&x_raw, 256, 20, 7);
        x_proj = principal_component_projection_apply(&fitted, &x_raw, false);
        proj = Some(fitted);
    }
    let k_folds = if n >= 50 {
        {FOLDS}
    } else {
        (n / 8).clamp(2, 5)
    };
    let repeats = {REPEATS};
    let rows_total = repeats * n;
    let mut held_out = vec![vec![0.0f64; m_count]; rows_total];
    for rep in 0..repeats {
        let mut idx: Vec<usize> = (0..n).collect();
        let mut rng = Pcg32::new(17, 1000 + rep as u64);
        for i in (1..n).rev() {
            let j = (rng.next_u32() as usize) % (i + 1);
            idx.swap(i, j);
        }
        for f in 0..k_folds {
            let lo = f * n / k_folds;
            let hi = (f + 1) * n / k_folds;
            if lo >= hi {
                continue;
            }
            let va: Vec<usize> = idx[lo..hi].to_vec();
            let mut in_va = vec![false; n];
            for &r in &va {
                in_va[r] = true;
            }
            let tr: Vec<usize> = (0..n).filter(|r| !in_va[*r]).collect();
            let ytr: Vec<f64> = tr.iter().map(|&r| y[r]).collect();
            let mut tr_mean = 0.0;
            for v in &ytr {
                tr_mean += v;
            }
            tr_mean /= ytr.len() as f64;
            let mut f_lo = ytr[0];
            let mut f_hi = ytr[0];
            for v in &ytr {
                if *v < f_lo {
                    f_lo = *v;
                }
                if *v > f_hi {
                    f_hi = *v;
                }
            }
            let f_pad = 0.05 * (f_hi - f_lo).max(1e-12);
            let xtr_raw: Vec<Vec<f64>> = tr.iter().map(|&r| x_raw[r].clone()).collect();
            let xva_raw: Vec<Vec<f64>> = va.iter().map(|&r| x_raw[r].clone()).collect();
            let mut xtr_proj: Vec<Vec<f64>> = Vec::new();
            let mut xva_proj: Vec<Vec<f64>> = Vec::new();
            if use_projection {
                xtr_proj = tr.iter().map(|&r| x_proj[r].clone()).collect();
                xva_proj = va.iter().map(|&r| x_proj[r].clone()).collect();
            }
            for (mi, entry) in roster.iter().enumerate() {
                let (cid, projected) = *entry;
                let (xtr_v, xva_v) = if use_projection && projected {
                    (&xtr_proj, &xva_proj)
                } else {
                    (&xtr_raw, &xva_raw)
                };
                let state = component_fit(cid, xtr_v, &ytr);
                let pm = component_predict(&state, xva_v);
                for (t, &r) in va.iter().enumerate() {
                    let mut v = pm[t];
                    if !v.is_finite() {
                        v = tr_mean;
                    }
                    if v < f_lo - f_pad {
                        v = f_lo - f_pad;
                    }
                    if v > f_hi + f_pad {
                        v = f_hi + f_pad;
                    }
                    held_out[rep * n + r][mi] = v;
                }
            }
        }
    }
    let lam = 1e-3 + 4.0 * y_var * m_count as f64 / rows_total.max(1) as f64;
    let mut g = vec![0.0f64; m_count * m_count];
    let mut c = vec![0.0f64; m_count];
    for i in 0..rows_total {
        let ti = y[i % n] - y_mean;
        for a in 0..m_count {
            let pa = held_out[i][a] - y_mean;
            c[a] += pa * ti;
            for b in a..m_count {
                g[a * m_count + b] += pa * (held_out[i][b] - y_mean);
            }
        }
    }
    for a in 0..m_count {
        g[a * m_count + a] += lam;
        for b in 0..a {
            g[a * m_count + b] = g[b * m_count + a];
        }
    }
    let mut w = non_negative_blend_weights(&g, &c, m_count);
    let mut wsum = 0.0;
    for v in &w {
        wsum += v;
    }
    if wsum <= 1e-12 {
        w = vec![1.0 / m_count as f64; m_count];
    }
    let mut blend_sse = 0.0;
    let mut part_sse = vec![0.0f64; m_count];
    for i in 0..rows_total {
        let ti = y[i % n];
        let mut z = y_mean;
        for a in 0..m_count {
            z += (held_out[i][a] - y_mean) * w[a];
        }
        let d = z - ti;
        blend_sse += d * d;
        for a in 0..m_count {
            let d = held_out[i][a] - ti;
            part_sse[a] += d * d;
        }
    }
    let blend_rmse = (blend_sse / rows_total as f64).sqrt();
    let part_rmse: Vec<f64> = part_sse
        .iter()
        .map(|s| (s / rows_total as f64).sqrt())
        .collect();
    let mut best = 0;
    for a in 1..m_count {
        if part_rmse[a] < part_rmse[best] {
            best = a;
        }
    }
    if part_rmse[best] < blend_rmse - 1e-9 {
        for (a, wa) in w.iter_mut().enumerate() {
            *wa = if a == best { 1.0 } else { 0.0 };
        }
    }
    let mut t_lo = y[0];
    let mut t_hi = y[0];
    for v in &y {
        if *v < t_lo {
            t_lo = *v;
        }
        if *v > t_hi {
            t_hi = *v;
        }
    }
    let pad = 0.05 * (t_hi - t_lo).max(1e-12);
    let clip_lo = t_lo - pad;
    let clip_hi = t_hi + pad;
    let mut entries: Vec<EnsembleEntry> = Vec::with_capacity(m_count);
    for (mi, entry) in roster.iter().enumerate() {
        let (cid, projected) = *entry;
        if w[mi] <= 1e-4 {
            entries.push(EnsembleEntry {
                projected,
                weight: 0.0,
                state: None,
            });
            continue;
        }
        let xv = if use_projection && projected {
            &x_proj
        } else {
            &x_raw
        };
        let state = component_fit(cid, xv, &y);
        let ptr = component_predict(&state, xv);
        let mut sse = 0.0;
        for i in 0..n {
            let d = ptr[i] - y[i];
            sse += d * d;
        }
        let tr_rmse = (sse / n as f64).sqrt();
        if !tr_rmse.is_finite() || tr_rmse > 1.05 * y_std {
            entries.push(EnsembleEntry {
                projected,
                weight: w[mi],
                state: None,
            });
        } else {
            entries.push(EnsembleEntry {
                projected,
                weight: w[mi],
                state: Some(state),
            });
        }
    }
    InternallyValidatedEnsemble {
        mean_only: false,
        y_mean,
        width,
        keep,
        proj,
        entries,
        clip_lo,
        clip_hi,
    }
}

/// Weighted blend of the fitted components' predictions around the training
/// mean; every component output and the blend are clipped to the padded
/// training target range and non-finite values fall back to the mean.
fn internally_validated_ensemble_predict(
    state: &InternallyValidatedEnsemble,
    x: &[Vec<f64>],
) -> Vec<f64> {
    if state.mean_only {
        return vec![state.y_mean; x.len()];
    }
    let mut rows: Vec<Vec<f64>> = Vec::with_capacity(x.len());
    for row in x {
        let mut r = row.clone();
        r.resize(state.width, 0.0);
        rows.push(r);
    }
    let x = sanitize(&rows);
    let x_raw: Vec<Vec<f64>> = x
        .iter()
        .map(|row| state.keep.iter().map(|&j| row[j]).collect())
        .collect();
    let mut x_proj: Vec<Vec<f64>> = Vec::new();
    if let Some(proj) = &state.proj {
        x_proj = principal_component_projection_apply(proj, &x_raw, false);
    }
    let y_mean = state.y_mean;
    let mut acc = vec![y_mean; x.len()];
    for entry in &state.entries {
        if entry.weight <= 0.0 {
            continue;
        }
        let fitted = match &entry.state {
            Some(s) => s,
            None => continue,
        };
        let xv = if entry.projected && state.proj.is_some() {
            &x_proj
        } else {
            &x_raw
        };
        let pm = component_predict(fitted, xv);
        for (i, a) in acc.iter_mut().enumerate() {
            let mut v = pm[i];
            if !v.is_finite() {
                v = y_mean;
            }
            if v < state.clip_lo {
                v = state.clip_lo;
            }
            if v > state.clip_hi {
                v = state.clip_hi;
            }
            *a += entry.weight * (v - y_mean);
        }
    }
    acc.iter()
        .map(|z| {
            let mut v = *z;
            if !v.is_finite() {
                v = y_mean;
            }
            if v < state.clip_lo {
                v = state.clip_lo;
            }
            if v > state.clip_hi {
                v = state.clip_hi;
            }
            v
        })
        .collect()
}"#;

const RIDGE_SRC: &str = r#"struct RidgeRegression {
    means: Vec<f64>,
    scales: Vec<f64>,
    w: Vec<f64>,
}

{DOC}
fn ridge_regression_fit(x: &[Vec<f64>], y: &[f64], lambda_frac: f64) -> RidgeRegression {
    let n = x.len().min(y.len());
    let p = x[0].len();
    let (means, scales) = std_fit(x);
    let lam = (lambda_frac * n as f64).max(1e-3);
    let mut y_mean = 0.0;
    for i in 0..n {
        y_mean += y[i];
    }
    y_mean /= n.max(1) as f64;
    let mut ata = vec![0.0; p * p];
    let mut atb = vec![0.0; p];
    let mut xs_row = vec![0.0; p];
    for r in 0..n {
        let row = &x[r];
        for j in 0..p {
            xs_row[j] = ((row[j] - means[j]) / scales[j]).clamp(-20.0, 20.0);
        }
        let yr = y[r] - y_mean;
        for i in 0..p {
            let xi = xs_row[i];
            atb[i] += xi * yr;
            for j in i..p {
                ata[i * p + j] += xi * xs_row[j];
            }
        }
    }
    for i in 0..p {
        ata[i * p + i] += lam;
        for j in 0..i {
            ata[i * p + j] = ata[j * p + i];
        }
    }
    let coeffs = cholesky_solve(&mut ata, &atb, p);
    let mut w = Vec::with_capacity(p + 1);
    w.push(y_mean);
    w.extend(coeffs);
    RidgeRegression { means, scales, w }
}

fn ridge_regression_predict(state: &RidgeRegression, x: &[Vec<f64>]) -> Vec<f64> {
    let p = state.means.len();
    let mut predictions = Vec::new();
    for row in x {
        let mut z = state.w[0];
        for j in 0..p.min(row.len()) {
            let xs = ((row[j] - state.means[j]) / state.scales[j]).clamp(-20.0, 20.0);
            z += state.w[j + 1] * xs;
        }
        predictions.push(z.clamp(-60.0, 60.0));
    }
    predictions
}"#;

const GCV_RIDGE_SRC: &str = r#"struct GcvRidgeRegression {
    cols: Vec<usize>,
    means: Vec<f64>,
    scales: Vec<f64>,
    w: Vec<f64>,
}

{DOC}
fn gcv_ridge_regression_fit(x: &[Vec<f64>], y: &[f64]) -> GcvRidgeRegression {
    let n = x.len().min(y.len());
    let mut p = x[0].len();
    let (mut means, mut scales) = std_fit(x);
    for s in scales.iter_mut() {
        if *s <= 1e-9 {
            *s = 1.0;
        }
    }
    let mut y_mean = 0.0;
    for i in 0..n {
        y_mean += y[i];
    }
    y_mean /= n.max(1) as f64;
    let mut y_var = 0.0;
    for i in 0..n {
        let d = y[i] - y_mean;
        y_var += d * d;
    }
    let mut cols: Vec<usize> = (0..p).collect();
    if n < 2 || p == 0 || (y_var / n.max(1) as f64).sqrt() < 1e-12 {
        let mut w = vec![0.0; p + 1];
        w[0] = y_mean;
        return GcvRidgeRegression { cols, means, scales, w };
    }
    if n.min(p) > 384 {
        let mut score = vec![0.0; p];
        for r in 0..n {
            let row = &x[r];
            let yr = y[r] - y_mean;
            for j in 0..p {
                score[j] += ((row[j] - means[j]) / scales[j]) * yr;
            }
        }
        for s in score.iter_mut() {
            if *s < 0.0 {
                *s = -*s;
            }
        }
        let mut used = vec![false; p];
        let mut sel = Vec::with_capacity(384);
        for _ in 0..384 {
            let mut best_j = 0;
            let mut best_s = -1.0;
            for j in 0..p {
                if !used[j] && score[j] > best_s {
                    best_s = score[j];
                    best_j = j;
                }
            }
            used[best_j] = true;
            sel.push(best_j);
        }
        sel.sort_unstable();
        cols = sel;
        p = cols.len();
        let kept_means: Vec<f64> = cols.iter().map(|&j| means[j]).collect();
        let kept_scales: Vec<f64> = cols.iter().map(|&j| scales[j]).collect();
        means = kept_means;
        scales = kept_scales;
    }
    let lambdas = [
        1e-05,
        2.1544346900318823e-05,
        4.641588833612782e-05,
        0.0001,
        0.00021544346900318823,
        0.00046415888336127773,
        0.001,
        0.002154434690031882,
        0.004641588833612777,
        0.01,
        0.021544346900318822,
        0.046415888336127774,
        0.1,
        0.21544346900318823,
        0.46415888336127725,
        1.0,
        2.154434690031882,
        4.641588833612772,
        10.0,
        21.54434690031882,
        46.41588833612773,
        100.0,
        215.44346900318823,
        464.1588833612773,
        1000.0,
    ];
    let nf = n as f64;
    if n < p {
        let mut xs = vec![0.0; n * p];
        for r in 0..n {
            let row = &x[r];
            for j in 0..p {
                xs[r * p + j] = (row[cols[j]] - means[j]) / scales[j];
            }
        }
        let mut yc = vec![0.0; n];
        let mut yty = 0.0;
        for r in 0..n {
            yc[r] = y[r] - y_mean;
            yty += yc[r] * yc[r];
        }
        let mut km = vec![0.0; n * n];
        for r in 0..n {
            for s in r..n {
                let mut acc = 0.0;
                for j in 0..p {
                    acc += xs[r * p + j] * xs[s * p + j];
                }
                km[r * n + s] = acc;
            }
        }
        for i in 0..n {
            for j in 0..i {
                km[i * n + j] = km[j * n + i];
            }
        }
        let (mut e, q) = jacobi_eigen(&mut km, n);
        let mut e_max = 0.0;
        for i in 0..n {
            if e[i] > e_max {
                e_max = e[i];
            }
        }
        for i in 0..n {
            if e[i] < e_max * 1e-12 {
                e[i] = 0.0;
            }
        }
        let mut u = vec![0.0; n];
        for i in 0..n {
            let mut s = 0.0;
            for r in 0..n {
                s += q[r * n + i] * yc[r];
            }
            u[i] = s;
        }
        let mut best_lambda = lambdas[0];
        let mut best_gcv = f64::INFINITY;
        for &lam in lambdas.iter() {
            let mut rss = yty;
            let mut tr = 0.0;
            for i in 0..n {
                if e[i] > 0.0 {
                    let c = u[i] / (e[i] + lam);
                    let f = e[i] * c;
                    rss += f * f - 2.0 * f * u[i];
                    tr += e[i] / (e[i] + lam);
                }
            }
            if rss < 0.0 {
                rss = 0.0;
            }
            let mut denom = 1.0 - tr / nf;
            if denom < 1e-3 {
                denom = 1e-3;
            }
            let gcv = (rss / nf) / (denom * denom);
            if gcv < best_gcv {
                best_gcv = gcv;
                best_lambda = lam;
            }
        }
        let mut t = vec![0.0; n];
        for r in 0..n {
            let mut s = 0.0;
            for i in 0..n {
                if e[i] > 0.0 {
                    s += q[r * n + i] * (u[i] / (e[i] + best_lambda));
                }
            }
            t[r] = s;
        }
        let mut w = Vec::with_capacity(p + 1);
        w.push(y_mean);
        for j in 0..p {
            let mut s = 0.0;
            for r in 0..n {
                s += xs[r * p + j] * t[r];
            }
            w.push(s);
        }
        return GcvRidgeRegression { cols, means, scales, w };
    }
    let mut g = vec![0.0; p * p];
    let mut b = vec![0.0; p];
    let mut yty = 0.0;
    let mut xs_row = vec![0.0; p];
    for r in 0..n {
        let row = &x[r];
        for j in 0..p {
            xs_row[j] = (row[cols[j]] - means[j]) / scales[j];
        }
        let yr = y[r] - y_mean;
        yty += yr * yr;
        for i in 0..p {
            let xi = xs_row[i];
            b[i] += xi * yr;
            for j in i..p {
                g[i * p + j] += xi * xs_row[j];
            }
        }
    }
    for i in 0..p {
        for j in 0..i {
            g[i * p + j] = g[j * p + i];
        }
    }
    let (mut e, v) = jacobi_eigen(&mut g, p);
    let mut e_max = 0.0;
    for i in 0..p {
        if e[i] > e_max {
            e_max = e[i];
        }
    }
    for i in 0..p {
        if e[i] < e_max * 1e-12 {
            e[i] = 0.0;
        }
    }
    let mut bt = vec![0.0; p];
    for i in 0..p {
        let mut s = 0.0;
        for j in 0..p {
            s += v[j * p + i] * b[j];
        }
        bt[i] = s;
    }
    let mut best_lambda = lambdas[0];
    let mut best_gcv = f64::INFINITY;
    for &lam in lambdas.iter() {
        let mut rss = yty;
        let mut tr = 0.0;
        for i in 0..p {
            if e[i] > 0.0 {
                let c = bt[i] / (e[i] + lam);
                rss += c * c * e[i] - 2.0 * c * bt[i];
                tr += e[i] / (e[i] + lam);
            }
        }
        if rss < 0.0 {
            rss = 0.0;
        }
        let mut denom = 1.0 - tr / nf;
        if denom < 1e-3 {
            denom = 1e-3;
        }
        let gcv = (rss / nf) / (denom * denom);
        if gcv < best_gcv {
            best_gcv = gcv;
            best_lambda = lam;
        }
    }
    let mut w = Vec::with_capacity(p + 1);
    w.push(y_mean);
    for j in 0..p {
        let mut s = 0.0;
        for i in 0..p {
            if e[i] > 0.0 {
                s += v[j * p + i] * (bt[i] / (e[i] + best_lambda));
            }
        }
        w.push(s);
    }
    GcvRidgeRegression { cols, means, scales, w }
}

fn gcv_ridge_regression_predict(state: &GcvRidgeRegression, x: &[Vec<f64>]) -> Vec<f64> {
    let mut predictions = Vec::new();
    for row in x {
        let mut z = state.w[0];
        for (k, &j) in state.cols.iter().enumerate() {
            if j < row.len() {
                z += state.w[k + 1] * ((row[j] - state.means[k]) / state.scales[k]);
            }
        }
        predictions.push(z.clamp(-60.0, 60.0));
    }
    predictions
}"#;

const ELASTIC_SRC: &str = r#"struct ElasticNet {
    means: Vec<f64>,
    scales: Vec<f64>,
    w0: f64,
    w: Vec<f64>,
}

{DOC}
fn elastic_net_fit(x: &[Vec<f64>], y: &[f64], alpha: f64, l1_ratio: f64, sweeps: usize) -> ElasticNet {
    let n = x.len().min(y.len());
    let p = x[0].len();
    let (means, scales) = std_fit(x);
    let mut xs: Vec<Vec<f64>> = Vec::new();
    for r in 0..n {
        let row = &x[r];
        xs.push((0..p).map(|j| ((row[j] - means[j]) / scales[j]).clamp(-20.0, 20.0)).collect());
    }
    let mut y_mean = 0.0;
    for i in 0..n {
        y_mean += y[i];
    }
    y_mean /= n.max(1) as f64;
    let l1 = alpha * l1_ratio;
    let l2 = alpha * (1.0 - l1_ratio);
    let mut col_sq = vec![0.0; p];
    for row in &xs {
        for j in 0..p {
            col_sq[j] += row[j] * row[j];
        }
    }
    let mut w = vec![0.0; p];
    let mut resid: Vec<f64> = (0..n).map(|r| y[r] - y_mean).collect();
    for _ in 0..sweeps.max(1) {
        let mut max_delta = 0.0;
        for j in 0..p {
            if col_sq[j] <= 1e-12 {
                continue;
            }
            let mut rho = 0.0;
            for r in 0..n {
                rho += xs[r][j] * resid[r];
            }
            rho = rho / n as f64 + w[j] * col_sq[j] / n as f64;
            let denom = col_sq[j] / n as f64 + l2;
            let new_w = soft_threshold(rho, l1) / denom;
            let delta = new_w - w[j];
            if delta != 0.0 {
                for r in 0..n {
                    resid[r] -= delta * xs[r][j];
                }
                w[j] = new_w;
                let ad = delta.abs();
                if ad > max_delta {
                    max_delta = ad;
                }
            }
        }
        if max_delta < 1e-9 {
            break;
        }
    }
    ElasticNet { means, scales, w0: y_mean, w }
}

fn elastic_net_predict(state: &ElasticNet, x: &[Vec<f64>]) -> Vec<f64> {
    let p = state.means.len();
    let mut predictions = Vec::new();
    for row in x {
        let mut z = state.w0;
        for j in 0..p.min(row.len()) {
            let xs = ((row[j] - state.means[j]) / state.scales[j]).clamp(-20.0, 20.0);
            z += state.w[j] * xs;
        }
        predictions.push(z.clamp(-60.0, 60.0));
    }
    predictions
}"#;

const GBM_SRC: &str = r#"struct GradientBoostedTrees {
    edges: Vec<Vec<f64>>,
    trees: Vec<Vec<TreeNode>>,
    init: f64,
    lr: f64,
    n_feat: usize,
    fallback: f64,
}

{DOC}
fn gradient_boosted_trees_fit(x: &[Vec<f64>], y: &[f64], rounds: usize, max_depth: usize, lr: f64,
                              lam: f64, bins: usize, min_leaf: usize, max_features: usize) -> GradientBoostedTrees {
    let n = x.len().min(y.len());
    let cols = x[0].len();
    let n_feat = cols.min(max_features);
    if n < 4 || n_feat == 0 {
        let m = mean_of(&y[..n.max(1).min(y.len())]);
        return GradientBoostedTrees {
            edges: Vec::new(),
            trees: Vec::new(),
            init: m,
            lr,
            n_feat: 0,
            fallback: m,
        };
    }
    let edges: Vec<Vec<f64>> = (0..n_feat).map(|f| tree_bin_edges(x, f, n, bins)).collect();
    let train_bins = tree_bin_rows(x, n, n_feat, &edges);
    let init = mean_of(&y[..n]);
    let mut f_train = vec![init; n];
    let mut trees = Vec::new();
    for _ in 0..rounds {
        let grad: Vec<f64> = (0..n).map(|i| f_train[i] - y[i]).collect();
        let hess = vec![1.0; n];
        let mut tree: Vec<TreeNode> = Vec::new();
        let mut train_leaf = vec![0usize; n];
        grow_tree(&train_bins, &grad, &hess, (0..n).collect(), n_feat, bins, max_depth, min_leaf,
                  lam, 0, &mut tree, &mut train_leaf);
        for i in 0..n {
            f_train[i] += lr * tree[train_leaf[i]].value;
        }
        trees.push(tree);
    }
    GradientBoostedTrees { edges, trees, init, lr, n_feat, fallback: init }
}

fn gradient_boosted_trees_predict(state: &GradientBoostedTrees, x: &[Vec<f64>]) -> Vec<f64> {
    if state.n_feat == 0 {
        return vec![state.fallback; x.len()];
    }
    let bins = tree_bin_rows(x, x.len(), state.n_feat, &state.edges);
    let mut predictions = Vec::new();
    for r in 0..x.len() {
        let mut value = state.init;
        for tree in &state.trees {
            value += state.lr * route_tree(tree, &bins, r, state.n_feat);
        }
        predictions.push(value);
    }
    predictions
}"#;

const BLEND_SRC: &str = r#"struct BlendedRegression {
    linear: RidgeRegression,
    boosted: GradientBoostedTrees,
    weight: f64,
}

{DOC}
fn blended_regression_fit(x: &[Vec<f64>], y: &[f64], lambda_frac: f64, rounds: usize,
                          max_depth: usize, lr: f64, lam: f64, bins: usize, min_leaf: usize,
                          max_features: usize) -> BlendedRegression {
    let n = x.len();
    if n < 40 {
        let boosted = gradient_boosted_trees_fit(x, y, rounds, max_depth, lr, lam, bins, min_leaf, max_features);
        let linear = ridge_regression_fit(x, y, lambda_frac);
        return BlendedRegression { linear, boosted, weight: 0.0 };
    }
    let split = (n * 4) / 5;
    let linear_hold = ridge_regression_predict(
        &ridge_regression_fit(&x[..split], &y[..split], lambda_frac), &x[split..]);
    let boosted_hold = gradient_boosted_trees_predict(
        &gradient_boosted_trees_fit(&x[..split], &y[..split], rounds, max_depth, lr, lam, bins, min_leaf, max_features),
        &x[split..]);
    let mut best_w = 0.5;
    let mut best = f64::NEG_INFINITY;
    for &w in &[0.0, 0.25, 0.5, 0.75, 1.0] {
        let blend: Vec<f64> = (0..linear_hold.len())
            .map(|i| w * linear_hold[i] + (1.0 - w) * boosted_hold[i])
            .collect();
        let score = -rmse(&y[split..], &blend);
        if score > best {
            best = score;
            best_w = w;
        }
    }
    let linear = ridge_regression_fit(x, y, lambda_frac);
    let boosted = gradient_boosted_trees_fit(x, y, rounds, max_depth, lr, lam, bins, min_leaf, max_features);
    BlendedRegression { linear, boosted, weight: best_w }
}

fn blended_regression_predict(state: &BlendedRegression, x: &[Vec<f64>]) -> Vec<f64> {
    let a = ridge_regression_predict(&state.linear, x);
    let b = gradient_boosted_trees_predict(&state.boosted, x);
    (0..a.len()).map(|i| state.weight * a[i] + (1.0 - state.weight) * b[i]).collect()
}"#;

const HISTGBM2_SRC: &str = r#"/// One node of a leaf-wise regression tree; feature -1 marks a leaf.
struct BoostNode {
    feature: i64,
    threshold: f64,
    left: usize,
    right: usize,
    value: f64,
}

/// A grown-but-unsplit leaf: its rows per feature (in pre-sorted order) and
/// its best prospective split (gain 0 when none qualifies).
struct LeafGroup {
    node: usize,
    lists: Vec<Vec<usize>>,
    gain: f64,
    feature: usize,
    threshold: f64,
}

/// Early-stopping criterion: mean absolute error when `absolute`, else root
/// mean squared error.
fn holdout_error(y: &[f64], p: &[f64], absolute: bool) -> f64 {
    if absolute {
        let n = y.len().min(p.len());
        if n == 0 {
            return 0.0;
        }
        let mut s = 0.0;
        for i in 0..n {
            s += (y[i] - p[i]).abs();
        }
        return s / n as f64;
    }
    rmse(y, p)
}

/// Exact split search over the pre-sorted rows of one leaf.
///
/// Only boundaries between distinct values qualify; the strict > comparison
/// with an ascending (feature, position) scan keeps the lowest feature and
/// threshold on ties. Features with col_ok[f] false are skipped (per-tree
/// column subsampling). Returns (gain, feature, threshold); gain 0 means no
/// split clears the 1e-9 floor.
fn leafwise_best_split(x: &[Vec<f64>], grad: &[f64], lists: &[Vec<usize>], g: f64, h: f64,
                       col_ok: &[bool], n_feat: usize, min_leaf: usize, lam: f64) -> (f64, usize, f64) {
    let m = lists[0].len();
    if m < 2 * min_leaf {
        return (0.0, 0, 0.0);
    }
    let parent = g * g / (h + lam);
    let mut best_gain = 1e-9;
    let mut out = (0.0, 0usize, 0.0);
    for f in 0..n_feat {
        if !col_ok[f] {
            continue;
        }
        let lst = &lists[f];
        let mut gl = 0.0;
        for t in 0..(m - 1) {
            let i = lst[t];
            gl += grad[i];
            let cnt = t + 1;
            if cnt < min_leaf {
                continue;
            }
            if m - cnt < min_leaf {
                break;
            }
            let v = x[i][f];
            if v < x[lst[t + 1]][f] {
                let hl = cnt as f64;
                let hr = (m - cnt) as f64;
                let gr = g - gl;
                let gain = gl * gl / (hl + lam) + gr * gr / (hr + lam) - parent;
                if gain > best_gain {
                    best_gain = gain;
                    out = (gain, f, v);
                }
            }
        }
    }
    out
}

/// Grow one regression tree leaf-wise (best-first): the leaf with the highest
/// gain splits next. Leaf values are the second-order step -g/(h + lam).
fn leafwise_tree(x: &[Vec<f64>], order: &[Vec<usize>], grad: &[f64], in_sample: &[bool],
                 col_ok: &[bool], n_feat: usize, max_leaves: usize, min_leaf: usize,
                 lam: f64) -> Vec<BoostNode> {
    let mut lists: Vec<Vec<usize>> = Vec::new();
    for f in 0..n_feat {
        let mut kept = Vec::new();
        for &i in &order[f] {
            if in_sample[i] {
                kept.push(i);
            }
        }
        lists.push(kept);
    }
    let mut g = 0.0;
    for &i in &lists[0] {
        g += grad[i];
    }
    let h = lists[0].len() as f64;
    let mut tree = vec![BoostNode { feature: -1, threshold: 0.0, left: 0, right: 0, value: -g / (h + lam) }];
    let (gain, feat, thr) = leafwise_best_split(x, grad, &lists, g, h, col_ok, n_feat, min_leaf, lam);
    let mut pool = vec![LeafGroup { node: 0, lists, gain, feature: feat, threshold: thr }];
    let mut leaf_count = 1usize;
    while leaf_count < max_leaves {
        let mut pick = usize::MAX;
        let mut pick_gain = 0.0;
        for k in 0..pool.len() {
            if pool[k].gain > pick_gain {
                pick_gain = pool[k].gain;
                pick = k;
            }
        }
        if pick == usize::MAX {
            break;
        }
        let group = pool.remove(pick);
        let feat = group.feature;
        let thr = group.threshold;
        let mut left_lists: Vec<Vec<usize>> = Vec::new();
        let mut right_lists: Vec<Vec<usize>> = Vec::new();
        for f in 0..n_feat {
            let mut left = Vec::new();
            let mut right = Vec::new();
            for &i in &group.lists[f] {
                if x[i][feat] <= thr {
                    left.push(i);
                } else {
                    right.push(i);
                }
            }
            left_lists.push(left);
            right_lists.push(right);
        }
        let mut gl = 0.0;
        for &i in &left_lists[0] {
            gl += grad[i];
        }
        let hl = left_lists[0].len() as f64;
        let mut gr = 0.0;
        for &i in &right_lists[0] {
            gr += grad[i];
        }
        let hr = right_lists[0].len() as f64;
        let li = tree.len();
        tree.push(BoostNode { feature: -1, threshold: 0.0, left: 0, right: 0, value: -gl / (hl + lam) });
        let ri = tree.len();
        tree.push(BoostNode { feature: -1, threshold: 0.0, left: 0, right: 0, value: -gr / (hr + lam) });
        tree[group.node] = BoostNode { feature: feat as i64, threshold: thr, left: li, right: ri, value: 0.0 };
        let (bg, bf, bt) = leafwise_best_split(x, grad, &left_lists, gl, hl, col_ok, n_feat, min_leaf, lam);
        pool.push(LeafGroup { node: li, lists: left_lists, gain: bg, feature: bf, threshold: bt });
        let (bg, bf, bt) = leafwise_best_split(x, grad, &right_lists, gr, hr, col_ok, n_feat, min_leaf, lam);
        pool.push(LeafGroup { node: ri, lists: right_lists, gain: bg, feature: bf, threshold: bt });
        leaf_count += 1;
    }
    tree
}

/// Follow split nodes (go left when value <= threshold) to a leaf value.
fn leafwise_route(tree: &[BoostNode], row: &[f64]) -> f64 {
    let mut idx = 0usize;
    loop {
        let node = &tree[idx];
        if node.feature < 0 {
            return node.value;
        }
        let f = node.feature as usize;
        let v = if f < row.len() { row[f] } else { 0.0 };
        idx = if v <= node.threshold { node.left } else { node.right };
    }
}

struct LeafwiseBoostedTrees {
    init: f64,
    lr: f64,
    bags: Vec<Vec<Vec<BoostNode>>>,
}

{DOC}
fn leafwise_boosted_trees_fit(x: &[Vec<f64>], y: &[f64], max_leaves: usize, lr: f64,
                              n_bags: usize, rounds: i64, row_share: f64, col_share: f64,
                              min_leaf: usize, holdout_pct: usize, shuffled_split: bool,
                              robust_clip: f64, absolute_error: bool) -> LeafwiseBoostedTrees {
    let n = x.len().min(y.len());
    let cols = x[0].len();
    let n_feat = cols.min(256);
    if n < 12 || n_feat == 0 {
        let m = mean_of(&y[..n.max(1).min(y.len())]);
        return LeafwiseBoostedTrees { init: m, lr, bags: Vec::new() };
    }
    // Internal holdout split: the contiguous tail by default, or rows chosen
    // by a seeded shuffle.
    let n_tr = (n * (100 - holdout_pct)) / 100;
    let n_hold = n - n_tr;
    let mut split_idx: Vec<usize> = (0..n).collect();
    if shuffled_split {
        let mut rng = Pcg32::new(4127, n as u64);
        for i in (1..n).rev() {
            let j = (rng.next_u32() as usize) % (i + 1);
            split_idx.swap(i, j);
        }
    }
    let mut tr_idx = split_idx[..n_tr].to_vec();
    tr_idx.sort();
    let mut hold_idx = split_idx[n_tr..].to_vec();
    hold_idx.sort();
    let xtr: Vec<Vec<f64>> = tr_idx.iter().map(|&i| x[i].clone()).collect();
    let ytr: Vec<f64> = tr_idx.iter().map(|&i| y[i]).collect();
    let xh: Vec<Vec<f64>> = hold_idx.iter().map(|&i| x[i].clone()).collect();
    let y_hold: Vec<f64> = hold_idx.iter().map(|&i| y[i]).collect();
    let init = mean_of(&ytr);
    // Pre-sort feature indices once; ties break on the row index so the order
    // matches a stable value sort exactly.
    let mut order: Vec<Vec<usize>> = Vec::new();
    for f in 0..n_feat {
        let mut idx: Vec<usize> = (0..n_tr).collect();
        idx.sort_by(|&a, &b| {
            if xtr[a][f] < xtr[b][f] {
                std::cmp::Ordering::Less
            } else if xtr[a][f] > xtr[b][f] {
                std::cmp::Ordering::Greater
            } else {
                a.cmp(&b)
            }
        });
        order.push(idx);
    }
    let mut bags: Vec<Vec<Vec<BoostNode>>> = Vec::new();
    for bag in 1..(n_bags + 1) {
        let mut rng = Pcg32::new(bag as u64, 1442);
        let mut f_tr = vec![init; n_tr];
        let mut f_hold = vec![init; n_hold];
        let mut best_err = holdout_error(&y_hold, &f_hold, absolute_error);
        let mut best_round: i64 = -1;
        let mut trees: Vec<Vec<BoostNode>> = Vec::new();
        for round_index in 0i64..rounds {
            let mut in_sample = Vec::new();
            for _ in 0..n_tr {
                in_sample.push(rng.next_f64() < row_share);
            }
            // Column masks are drawn AFTER the row mask and only when active,
            // so a share of 1.0 consumes no randomness at all.
            let col_ok: Vec<bool> = if col_share < 1.0 {
                let mut ok = Vec::new();
                for _ in 0..n_feat {
                    ok.push(rng.next_f64() < col_share);
                }
                if !ok.iter().any(|&b| b) {
                    for v in ok.iter_mut() {
                        *v = true;
                    }
                }
                ok
            } else {
                vec![true; n_feat]
            };
            let mut grad = Vec::new();
            if robust_clip > 0.0 {
                for i in 0..n_tr {
                    grad.push((f_tr[i] - ytr[i]).clamp(-robust_clip, robust_clip));
                }
            } else {
                for i in 0..n_tr {
                    grad.push(f_tr[i] - ytr[i]);
                }
            }
            let tree = leafwise_tree(&xtr, &order, &grad, &in_sample, &col_ok, n_feat,
                                     max_leaves, min_leaf, 1.0);
            for i in 0..n_tr {
                f_tr[i] += lr * leafwise_route(&tree, &xtr[i]);
            }
            for hh in 0..n_hold {
                f_hold[hh] += lr * leafwise_route(&tree, &xh[hh]);
            }
            trees.push(tree);
            let r = holdout_error(&y_hold, &f_hold, absolute_error);
            if r < best_err {
                best_err = r;
                best_round = round_index;
            }
            if round_index - best_round >= 50 {
                break;
            }
        }
        trees.truncate((best_round + 1) as usize);
        bags.push(trees);
    }
    LeafwiseBoostedTrees { init, lr, bags }
}

fn leafwise_boosted_trees_predict(state: &LeafwiseBoostedTrees, x: &[Vec<f64>]) -> Vec<f64> {
    let mut predictions = Vec::new();
    for row in x {
        if state.bags.is_empty() {
            predictions.push(state.init);
            continue;
        }
        let mut acc = 0.0;
        for trees in &state.bags {
            let mut s = 0.0;
            for tree in trees {
                s += leafwise_route(tree, row);
            }
            acc += s;
        }
        predictions.push(state.init + state.lr * acc / state.bags.len() as f64);
    }
    predictions
}"#;

const KRR_SRC: &str = r#"struct KernelRidgeRegression {
    means: Vec<f64>,
    scales: Vec<f64>,
    basis: Vec<Vec<f64>>,
    alpha: Vec<f64>,
    gamma: f64,
    y_mean: f64,
}

/// Squared Euclidean distance over the shorter of the two vectors.
fn squared_distance(a: &[f64], b: &[f64]) -> f64 {
    let mut d = 0.0;
    for j in 0..a.len().min(b.len()) {
        let t = a[j] - b[j];
        d += t * t;
    }
    d
}

/// k distinct row indices out of n via a seeded partial Fisher-Yates shuffle,
/// returned in ascending order.
fn seeded_rows(n: usize, k: usize, seed: u64) -> Vec<usize> {
    let mut rng = Pcg32::new(seed, 54);
    let mut idx: Vec<usize> = (0..n).collect();
    for i in 0..k {
        let j = i + (rng.next_u32() as usize) % (n - i);
        let tmp = idx[i];
        idx[i] = idx[j];
        idx[j] = tmp;
    }
    let mut out = idx[..k].to_vec();
    out.sort();
    out
}

{DOC}
fn kernel_ridge_regression_fit(x: &[Vec<f64>], y: &[f64], m: usize) -> KernelRidgeRegression {
    let n = x.len().min(y.len());
    let p = x[0].len();
    let (means, scales) = std_fit(x);
    let mut y_mean = 0.0;
    for i in 0..n {
        y_mean += y[i];
    }
    y_mean /= n.max(1) as f64;
    if n < 12 || p == 0 {
        return KernelRidgeRegression {
            means,
            scales,
            basis: Vec::new(),
            alpha: Vec::new(),
            gamma: 0.0,
            y_mean,
        };
    }
    let mut z: Vec<Vec<f64>> = Vec::new();
    for i in 0..n {
        let row = &x[i];
        z.push((0..p).map(|j| (row[j] - means[j]) / scales[j]).collect());
    }
    let n_tr = (n * 4) / 5;
    let n_hold = n - n_tr;
    let k_sel = n_tr.min(256);
    let sel: Vec<usize> = if n_tr > 256 {
        seeded_rows(n_tr, 256, 401)
    } else {
        (0..n_tr).collect()
    };
    let ys: Vec<f64> = sel.iter().map(|&i| y[i] - y_mean).collect();
    let mut d2ss = vec![0.0; k_sel * k_sel];
    for a in 0..k_sel {
        for b in (a + 1)..k_sel {
            let d = squared_distance(&z[sel[a]], &z[sel[b]]);
            d2ss[a * k_sel + b] = d;
            d2ss[b * k_sel + a] = d;
        }
    }
    let mut d2hs = vec![0.0; n_hold * k_sel];
    for hh in 0..n_hold {
        for b in 0..k_sel {
            d2hs[hh * k_sel + b] = squared_distance(&z[n_tr + hh], &z[sel[b]]);
        }
    }
    let y_hold = &y[n_tr..n];
    let mut best_score = f64::INFINITY;
    let mut best_gamma = 0.1 / p as f64;
    let mut best_lambda = 0.001;
    for &coef in &[0.1, 0.3, 1.0, 3.0, 10.0] {
        let gamma = coef / p as f64;
        let k0: Vec<f64> = d2ss.iter().map(|d| (-gamma * d).exp()).collect();
        let kh: Vec<f64> = d2hs.iter().map(|d| (-gamma * d).exp()).collect();
        for &lam in &[0.001, 0.01, 0.1, 1.0, 10.0] {
            let mut a = k0.clone();
            for q in 0..k_sel {
                a[q * k_sel + q] += lam;
            }
            let alpha = cholesky_solve(&mut a, &ys, k_sel);
            let mut preds = Vec::new();
            for hh in 0..n_hold {
                let mut s = y_mean;
                let base = hh * k_sel;
                for q in 0..k_sel {
                    s += kh[base + q] * alpha[q];
                }
                preds.push(s);
            }
            let score = rmse(y_hold, &preds);
            if score < best_score {
                best_score = score;
                best_gamma = gamma;
                best_lambda = lam;
            }
        }
    }
    let fin: Vec<usize> = if n > m { seeded_rows(n, m, 402) } else { (0..n).collect() };
    let k_fin = fin.len();
    let basis: Vec<Vec<f64>> = fin.iter().map(|&i| z[i].clone()).collect();
    let yf: Vec<f64> = fin.iter().map(|&i| y[i] - y_mean).collect();
    let mut a = vec![0.0; k_fin * k_fin];
    for i2 in 0..k_fin {
        for j2 in (i2 + 1)..k_fin {
            let v = (-best_gamma * squared_distance(&basis[i2], &basis[j2])).exp();
            a[i2 * k_fin + j2] = v;
            a[j2 * k_fin + i2] = v;
        }
    }
    for i2 in 0..k_fin {
        a[i2 * k_fin + i2] = 1.0 + best_lambda;
    }
    let alpha = cholesky_solve(&mut a, &yf, k_fin);
    KernelRidgeRegression { means, scales, basis, alpha, gamma: best_gamma, y_mean }
}

fn kernel_ridge_regression_predict(state: &KernelRidgeRegression, x: &[Vec<f64>]) -> Vec<f64> {
    let p = state.means.len();
    let mut predictions = Vec::new();
    for row in x {
        if state.basis.is_empty() {
            predictions.push(state.y_mean.clamp(-60.0, 60.0));
            continue;
        }
        let zr: Vec<f64> = (0..p.min(row.len()))
            .map(|j| (row[j] - state.means[j]) / state.scales[j])
            .collect();
        let mut s = state.y_mean;
        for b in 0..state.basis.len() {
            s += state.alpha[b] * (-state.gamma * squared_distance(&zr, &state.basis[b])).exp();
        }
        predictions.push(s.clamp(-60.0, 60.0));
    }
    predictions
}"#;

const GPR_SRC: &str = r#"struct GaussianProcessArdRegression {
    means: Vec<f64>,
    scales: Vec<f64>,
    basis: Vec<Vec<f64>>,
    alpha: Vec<f64>,
    ls: Vec<f64>,
    sf2: f64,
    y_mean: f64,
}

/// k distinct row indices out of n via a seeded partial Fisher-Yates shuffle,
/// returned in ascending order.
fn seeded_rows(n: usize, k: usize, seed: u64) -> Vec<usize> {
    let mut rng = Pcg32::new(seed, 54);
    let mut idx: Vec<usize> = (0..n).collect();
    for i in 0..k {
        let j = i + (rng.next_u32() as usize) % (n - i);
        let tmp = idx[i];
        idx[i] = idx[j];
        idx[j] = tmp;
    }
    let mut out = idx[..k].to_vec();
    out.sort();
    out
}

/// Symmetric-positive-definite inverse of a (flattened m*m, consumed): factor
/// A = L L^T in place, invert the lower-triangular L, then form A^-1 = L^-T L^-1.
/// A tiny 1e-6 pivot stands in whenever a diagonal entry falls under 1e-12.
fn chol_inverse(mut a: Vec<f64>, m: usize) -> Vec<f64> {
    for i in 0..m {
        for j in 0..(i + 1) {
            let mut s = a[i * m + j];
            for k in 0..j {
                s -= a[i * m + k] * a[j * m + k];
            }
            if i == j {
                a[i * m + j] = if s > 1e-12 { s.sqrt() } else { 1e-6 };
            } else {
                a[i * m + j] = s / a[j * m + j];
            }
        }
    }
    let mut linv = vec![0.0; m * m];
    for i in 0..m {
        linv[i * m + i] = 1.0 / a[i * m + i];
        for j in 0..i {
            let mut s = 0.0;
            for k in j..i {
                s -= a[i * m + k] * linv[k * m + j];
            }
            linv[i * m + j] = s / a[i * m + i];
        }
    }
    let mut inv = vec![0.0; m * m];
    for i in 0..m {
        for j in 0..(i + 1) {
            let mut s = 0.0;
            for k in i..m {
                s += linv[k * m + i] * linv[k * m + j];
            }
            inv[i * m + j] = s;
            inv[j * m + i] = s;
        }
    }
    inv
}

{DOC}
fn gaussian_process_ard_regression_fit(x: &[Vec<f64>], y: &[f64], m: usize, steps: usize) -> GaussianProcessArdRegression {
    let n = x.len().min(y.len());
    let p = x[0].len();
    let (means, scales) = std_fit(x);
    let mut y_mean = 0.0;
    for i in 0..n {
        y_mean += y[i];
    }
    y_mean /= n.max(1) as f64;
    if n < 12 || p == 0 {
        return GaussianProcessArdRegression {
            means,
            scales,
            basis: Vec::new(),
            alpha: Vec::new(),
            ls: Vec::new(),
            sf2: 0.0,
            y_mean,
        };
    }
    let basis_size = n.min(m);
    let idx: Vec<usize> = if n > m { seeded_rows(n, m, 909) } else { (0..n).collect() };
    let z: Vec<Vec<f64>> = idx
        .iter()
        .map(|&i| (0..p).map(|j| (x[i][j] - means[j]) / scales[j]).collect())
        .collect();
    let yc: Vec<f64> = idx.iter().map(|&i| y[i] - y_mean).collect();
    let opt_size = basis_size.min(256);
    // Seed each lengthscale from the median pairwise per-feature distance, scaled
    // by sqrt(p) so the summed kernel exponent stays O(1) as p grows.
    let n_init = opt_size.min(96);
    let dim_scale = (p as f64).sqrt();
    let mut log_ls = vec![0.0; p];
    for d in 0..p {
        let mut dists: Vec<f64> = Vec::new();
        for i in 0..n_init {
            for j in (i + 1)..n_init {
                dists.push((z[i][d] - z[j][d]).abs());
            }
        }
        dists.sort_by(|a, b| a.total_cmp(b));
        let med = if dists.is_empty() { 1.0 } else { quantile(&dists, 0.5) };
        log_ls[d] = (med.max(1e-3) * dim_scale).ln();
    }
    let mut var = 0.0;
    for i in 0..opt_size {
        var += yc[i] * yc[i];
    }
    let sy = (var / opt_size.max(1) as f64).sqrt();
    let mut log_sf = sy.max(1e-6).ln();
    let mut log_sn = (0.1 * sy).max(1e-6).ln();
    // Gradient ascent on the log marginal likelihood over the first opt_size rows.
    for _ in 0..steps {
        let mo = opt_size;
        let ls_cur: Vec<f64> = log_ls.iter().map(|u| u.exp()).collect();
        let inv_ls2: Vec<f64> = ls_cur.iter().map(|v| 1.0 / (v * v)).collect();
        let sf2 = (2.0 * log_sf).exp();
        let sn2 = (2.0 * log_sn).exp();
        let mut kmat = vec![0.0; mo * mo];
        for i in 0..mo {
            kmat[i * mo + i] = sf2;
            for j in (i + 1)..mo {
                let mut r2 = 0.0;
                for d in 0..p {
                    let diff = z[i][d] - z[j][d];
                    r2 += diff * diff * inv_ls2[d];
                }
                let v = sf2 * (-0.5 * r2).exp();
                kmat[i * mo + j] = v;
                kmat[j * mo + i] = v;
            }
        }
        let mut a = kmat.clone();
        for i in 0..mo {
            a[i * mo + i] += sn2;
        }
        let kinv = chol_inverse(a, mo);
        let mut alpha = vec![0.0; mo];
        for i in 0..mo {
            let mut s = 0.0;
            let base = i * mo;
            for j in 0..mo {
                s += kinv[base + j] * yc[j];
            }
            alpha[i] = s;
        }
        let mut g_sf = 0.0;
        let mut g_sn = 0.0;
        let mut g_ls = vec![0.0; p];
        for i in 0..mo {
            let wii = alpha[i] * alpha[i] - kinv[i * mo + i];
            g_sf += wii * kmat[i * mo + i];
            g_sn += wii;
            let ai = alpha[i];
            for j in (i + 1)..mo {
                let wij = ai * alpha[j] - kinv[i * mo + j];
                let kij = kmat[i * mo + j];
                g_sf += 2.0 * wij * kij;
                let c = wij * kij;
                for d in 0..p {
                    let diff = z[i][d] - z[j][d];
                    g_ls[d] += c * diff * diff * inv_ls2[d];
                }
            }
        }
        g_sn *= sn2;
        let scale = 1.0 / mo as f64;
        for d in 0..p {
            let step = (scale * g_ls[d]).clamp(-0.5, 0.5);
            log_ls[d] = (log_ls[d] + step).clamp(-6.0, 6.0);
        }
        let step_sf = (scale * g_sf).clamp(-0.5, 0.5);
        log_sf = (log_sf + step_sf).clamp(-15.0, 15.0);
        let step_sn = (scale * g_sn).clamp(-0.5, 0.5);
        // Floor the noise at exp(-6) of the signal std: this jitter keeps Ky
        // well-conditioned, so the gradient cannot overflow even under an
        // aggressive step.
        log_sn = (log_sn + step_sn).clamp(log_sf - 6.0, log_sf);
    }
    let ls_final: Vec<f64> = log_ls.iter().map(|u| u.exp()).collect();
    let inv_ls2: Vec<f64> = ls_final.iter().map(|v| 1.0 / (v * v)).collect();
    let sf2 = (2.0 * log_sf).exp();
    let sn2 = (2.0 * log_sn).exp();
    let mb = basis_size;
    let mut a = vec![0.0; mb * mb];
    for i in 0..mb {
        a[i * mb + i] = sf2 + sn2;
        for j in (i + 1)..mb {
            let mut r2 = 0.0;
            for d in 0..p {
                let diff = z[i][d] - z[j][d];
                r2 += diff * diff * inv_ls2[d];
            }
            let v = sf2 * (-0.5 * r2).exp();
            a[i * mb + j] = v;
            a[j * mb + i] = v;
        }
    }
    let alpha = cholesky_solve(&mut a, &yc, mb);
    GaussianProcessArdRegression { means, scales, basis: z, alpha, ls: ls_final, sf2, y_mean }
}

fn gaussian_process_ard_regression_predict(state: &GaussianProcessArdRegression, x: &[Vec<f64>]) -> Vec<f64> {
    let p = state.means.len();
    if state.basis.is_empty() {
        return x.iter().map(|_| state.y_mean.clamp(-60.0, 60.0)).collect();
    }
    let inv_ls2: Vec<f64> = state.ls.iter().map(|v| 1.0 / (v * v)).collect();
    let mut predictions = Vec::new();
    for row in x {
        let zr: Vec<f64> = (0..p.min(row.len()))
            .map(|j| (row[j] - state.means[j]) / state.scales[j])
            .collect();
        let mut s = state.y_mean;
        for b in 0..state.basis.len() {
            let zb = &state.basis[b];
            let mut r2 = 0.0;
            for d in 0..zr.len().min(zb.len()) {
                let diff = zr[d] - zb[d];
                r2 += diff * diff * inv_ls2[d];
            }
            s += state.alpha[b] * state.sf2 * (-0.5 * r2).exp();
        }
        predictions.push(s.clamp(-60.0, 60.0));
    }
    predictions
}"#;

const GPRK_SRC: &str = r#"struct GaussianProcessCompositeKernelRegression {
    means: Vec<f64>,
    scales: Vec<f64>,
    basis: Vec<Vec<f64>>,
    alpha: Vec<f64>,
    theta: Vec<f64>,
    y_mean: f64,
}

/// Decoded per-evaluation kernel parameters (see kernel_decode).
struct KernelParams {
    /// Per-term signal variance exp(2 * log amplitude).
    sf2: Vec<f64>,
    /// Per (term, factor) inverse squared lengthscales: p entries for the
    /// squared-exponential / rational-quadratic factors, one shared entry for
    /// periodic factors, empty for linear factors.
    inv_len2: Vec<Vec<Vec<f64>>>,
    /// Per (term, factor) rational-quadratic mixing exponent (0.0 elsewhere).
    alpha: Vec<Vec<f64>>,
    /// Per (term, factor) period (0.0 elsewhere).
    period: Vec<Vec<f64>>,
    /// Observation noise variance exp(2 * log noise std).
    noise2: f64,
}

/// k distinct row indices out of n via a seeded partial Fisher-Yates shuffle,
/// returned in ascending order.
fn seeded_rows(n: usize, k: usize, seed: u64) -> Vec<usize> {
    let mut rng = Pcg32::new(seed, 54);
    let mut idx: Vec<usize> = (0..n).collect();
    for i in 0..k {
        let j = i + (rng.next_u32() as usize) % (n - i);
        let tmp = idx[i];
        idx[i] = idx[j];
        idx[j] = tmp;
    }
    let mut out = idx[..k].to_vec();
    out.sort();
    out
}

/// Symmetric-positive-definite inverse of a (flattened m*m, consumed): factor
/// A = L L^T in place, invert the lower-triangular L, then form A^-1 = L^-T L^-1.
/// A tiny 1e-6 pivot stands in whenever a diagonal entry falls under 1e-12.
fn chol_inverse(mut a: Vec<f64>, m: usize) -> Vec<f64> {
    for i in 0..m {
        for j in 0..(i + 1) {
            let mut s = a[i * m + j];
            for k in 0..j {
                s -= a[i * m + k] * a[j * m + k];
            }
            if i == j {
                a[i * m + j] = if s > 1e-12 { s.sqrt() } else { 1e-6 };
            } else {
                a[i * m + j] = s / a[j * m + j];
            }
        }
    }
    let mut linv = vec![0.0; m * m];
    for i in 0..m {
        linv[i * m + i] = 1.0 / a[i * m + i];
        for j in 0..i {
            let mut s = 0.0;
            for k in j..i {
                s -= a[i * m + k] * linv[k * m + j];
            }
            linv[i * m + j] = s / a[i * m + i];
        }
    }
    let mut inv = vec![0.0; m * m];
    for i in 0..m {
        for j in 0..(i + 1) {
            let mut s = 0.0;
            for k in i..m {
                s += linv[k * m + i] * linv[k * m + j];
            }
            inv[i * m + j] = s;
            inv[j * m + i] = s;
        }
    }
    inv
}

/// Decode the flat hyperparameter vector: per sum term [log amplitude, factor
/// shape parameters...], then the global log noise std last.
fn kernel_decode(kernel_terms: &[Vec<&str>], theta: &[f64], p: usize) -> KernelParams {
    let mut sf2: Vec<f64> = Vec::new();
    let mut inv_len2: Vec<Vec<Vec<f64>>> = Vec::new();
    let mut alpha: Vec<Vec<f64>> = Vec::new();
    let mut period: Vec<Vec<f64>> = Vec::new();
    let mut off = 0usize;
    for term in kernel_terms {
        sf2.push((2.0 * theta[off]).exp());
        off += 1;
        let mut inv_t: Vec<Vec<f64>> = Vec::new();
        let mut alpha_t: Vec<f64> = Vec::new();
        let mut period_t: Vec<f64> = Vec::new();
        for kind in term {
            match *kind {
                "squared_exponential" => {
                    let mut inv_f = Vec::new();
                    for d in 0..p {
                        let length = theta[off + d].exp();
                        inv_f.push(1.0 / (length * length));
                    }
                    inv_t.push(inv_f);
                    alpha_t.push(0.0);
                    period_t.push(0.0);
                    off += p;
                }
                "rational_quadratic" => {
                    let mut inv_f = Vec::new();
                    for d in 0..p {
                        let length = theta[off + d].exp();
                        inv_f.push(1.0 / (length * length));
                    }
                    inv_t.push(inv_f);
                    alpha_t.push(theta[off + p].exp());
                    period_t.push(0.0);
                    off += p + 1;
                }
                "periodic" => {
                    period_t.push(theta[off].exp());
                    let length = theta[off + 1].exp();
                    inv_t.push(vec![1.0 / (length * length)]);
                    alpha_t.push(0.0);
                    off += 2;
                }
                _ => {
                    inv_t.push(Vec::new());
                    alpha_t.push(0.0);
                    period_t.push(0.0);
                }
            }
        }
        inv_len2.push(inv_t);
        alpha.push(alpha_t);
        period.push(period_t);
    }
    let noise2 = (2.0 * theta[off]).exp();
    KernelParams { sf2, inv_len2, alpha, period, noise2 }
}

/// Covariance k(za, zb): the sum over terms of the term amplitude times the
/// product of its factor values, in declaration order.
fn kernel_value(kernel_terms: &[Vec<&str>], dec: &KernelParams, pf: f64, za: &[f64], zb: &[f64]) -> f64 {
    let dm = za.len().min(zb.len());
    let mut total = 0.0;
    for t in 0..kernel_terms.len() {
        let term = &kernel_terms[t];
        let mut val = dec.sf2[t];
        for f in 0..term.len() {
            match term[f] {
                "squared_exponential" => {
                    let inv = &dec.inv_len2[t][f];
                    let mut r2 = 0.0;
                    for d in 0..dm {
                        let diff = za[d] - zb[d];
                        r2 += diff * diff * inv[d];
                    }
                    val *= (-0.5 * r2).exp();
                }
                "rational_quadratic" => {
                    let inv = &dec.inv_len2[t][f];
                    let mut r2 = 0.0;
                    for d in 0..dm {
                        let diff = za[d] - zb[d];
                        r2 += diff * diff * inv[d];
                    }
                    let al = dec.alpha[t][f];
                    let s = r2 / (2.0 * al);
                    val *= (-al * (1.0 + s).ln()).exp();
                }
                "periodic" => {
                    let per = dec.period[t][f];
                    let inv0 = dec.inv_len2[t][f][0];
                    let mut s2 = 0.0;
                    for d in 0..dm {
                        let ang = std::f64::consts::PI * (za[d] - zb[d]) / per;
                        let sa = ang.sin();
                        s2 += sa * sa;
                    }
                    val *= (-2.0 * s2 * inv0).exp();
                }
                _ => {
                    let mut dot = 0.0;
                    for d in 0..dm {
                        dot += za[d] * zb[d];
                    }
                    val *= dot / pf;
                }
            }
        }
        total += val;
    }
    total
}

{DOC}
fn gaussian_process_composite_kernel_regression_fit(x: &[Vec<f64>], y: &[f64], m: usize,
                                                    steps: usize, kernel_terms: &[Vec<&str>])
                                                    -> GaussianProcessCompositeKernelRegression {
    let n = x.len().min(y.len());
    let p = x[0].len();
    let (means, scales) = std_fit(x);
    let mut y_mean = 0.0;
    for i in 0..n {
        y_mean += y[i];
    }
    y_mean /= n.max(1) as f64;
    if n < 12 || p == 0 {
        return GaussianProcessCompositeKernelRegression {
            means,
            scales,
            basis: Vec::new(),
            alpha: Vec::new(),
            theta: Vec::new(),
            y_mean,
        };
    }
    let pf = p as f64;
    let basis_size = n.min(m);
    let idx: Vec<usize> = if n > m { seeded_rows(n, m, 909) } else { (0..n).collect() };
    let z: Vec<Vec<f64>> = idx
        .iter()
        .map(|&i| (0..p).map(|j| (x[i][j] - means[j]) / scales[j]).collect())
        .collect();
    let yc: Vec<f64> = idx.iter().map(|&i| y[i] - y_mean).collect();
    let opt_size = basis_size.min(256);
    // Lengthscale seed for every distance factor: the median pairwise
    // per-feature distance scaled by sqrt(p), so the summed kernel exponent
    // stays O(1) as the feature count grows.
    let n_init = opt_size.min(96);
    let dim_scale = pf.sqrt();
    let mut ard_init = vec![0.0; p];
    for d in 0..p {
        let mut dists: Vec<f64> = Vec::new();
        for i in 0..n_init {
            for j in (i + 1)..n_init {
                dists.push((z[i][d] - z[j][d]).abs());
            }
        }
        dists.sort_by(|a, b| a.total_cmp(b));
        let med = if dists.is_empty() { 1.0 } else { quantile(&dists, 0.5) };
        ard_init[d] = (med.max(1e-3) * dim_scale).ln();
    }
    let mut var = 0.0;
    for i in 0..opt_size {
        var += yc[i] * yc[i];
    }
    let sy = (var / opt_size.max(1) as f64).sqrt();
    // Flat hyperparameter vector with per-slot box clamps; term amplitudes
    // split the target variance so the summed prior variance stays sy^2.
    let amp_init = sy.max(1e-6).ln() - 0.5 * (kernel_terms.len() as f64).ln();
    let mut theta: Vec<f64> = Vec::new();
    let mut lo: Vec<f64> = Vec::new();
    let mut hi: Vec<f64> = Vec::new();
    let mut amp_idx: Vec<usize> = Vec::new();
    for term in kernel_terms {
        amp_idx.push(theta.len());
        theta.push(amp_init);
        lo.push(-15.0);
        hi.push(15.0);
        for kind in term {
            match *kind {
                "squared_exponential" => {
                    for d in 0..p {
                        theta.push(ard_init[d]);
                        lo.push(-6.0);
                        hi.push(6.0);
                    }
                }
                "rational_quadratic" => {
                    for d in 0..p {
                        theta.push(ard_init[d]);
                        lo.push(-6.0);
                        hi.push(6.0);
                    }
                    theta.push(0.0);
                    lo.push(-3.0);
                    hi.push(5.0);
                }
                "periodic" => {
                    theta.push(2.0f64.ln());
                    lo.push(-3.0);
                    hi.push(3.0);
                    theta.push(0.5 * (2.0 * pf).ln());
                    lo.push(-6.0);
                    hi.push(6.0);
                }
                _ => {}
            }
        }
    }
    let noise_idx = theta.len();
    theta.push((0.1 * sy).max(1e-6).ln());
    // Gradient ascent on the log marginal likelihood over the first opt_size
    // rows; dK/dtheta uses per-term / per-factor chain rules.
    for _ in 0..steps {
        let mo = opt_size;
        let dec = kernel_decode(kernel_terms, &theta, p);
        let mut a = vec![0.0; mo * mo];
        for i in 0..mo {
            a[i * mo + i] = kernel_value(kernel_terms, &dec, pf, &z[i], &z[i]) + dec.noise2;
            for j in (i + 1)..mo {
                let v = kernel_value(kernel_terms, &dec, pf, &z[i], &z[j]);
                a[i * mo + j] = v;
                a[j * mo + i] = v;
            }
        }
        let kinv = chol_inverse(a, mo);
        let mut alpha = vec![0.0; mo];
        for i in 0..mo {
            let mut s = 0.0;
            let base = i * mo;
            for j in 0..mo {
                s += kinv[base + j] * yc[j];
            }
            alpha[i] = s;
        }
        let mut g = vec![0.0; theta.len()];
        for i in 0..mo {
            let wii = alpha[i] * alpha[i] - kinv[i * mo + i];
            g[noise_idx] += wii;
            // Diagonal: distance-based factor values are exactly 1 there, so
            // only the term amplitudes (and linear factors) contribute.
            let mut off = 0usize;
            for t in 0..kernel_terms.len() {
                let term = &kernel_terms[t];
                let amp = off;
                off += 1;
                let mut val = dec.sf2[t];
                for kind in term {
                    match *kind {
                        "squared_exponential" => off += p,
                        "rational_quadratic" => off += p + 1,
                        "periodic" => off += 2,
                        _ => {
                            let mut dot = 0.0;
                            for d in 0..p {
                                dot += z[i][d] * z[i][d];
                            }
                            val *= dot / pf;
                        }
                    }
                }
                g[amp] += wii * val;
            }
            for j in (i + 1)..mo {
                let wij = alpha[i] * alpha[j] - kinv[i * mo + j];
                let mut off = 0usize;
                for t in 0..kernel_terms.len() {
                    let term = &kernel_terms[t];
                    let amp = off;
                    let nf = term.len();
                    let mut r2s = vec![0.0; nf];
                    let mut s2s = vec![0.0; nf];
                    let mut psums = vec![0.0; nf];
                    let mut val = dec.sf2[t];
                    for f in 0..nf {
                        match term[f] {
                            "squared_exponential" => {
                                let inv = &dec.inv_len2[t][f];
                                let mut r2 = 0.0;
                                for d in 0..p {
                                    let diff = z[i][d] - z[j][d];
                                    r2 += diff * diff * inv[d];
                                }
                                r2s[f] = r2;
                                val *= (-0.5 * r2).exp();
                            }
                            "rational_quadratic" => {
                                let inv = &dec.inv_len2[t][f];
                                let mut r2 = 0.0;
                                for d in 0..p {
                                    let diff = z[i][d] - z[j][d];
                                    r2 += diff * diff * inv[d];
                                }
                                r2s[f] = r2;
                                let al = dec.alpha[t][f];
                                let s = r2 / (2.0 * al);
                                val *= (-al * (1.0 + s).ln()).exp();
                            }
                            "periodic" => {
                                let per = dec.period[t][f];
                                let inv0 = dec.inv_len2[t][f][0];
                                let mut s2 = 0.0;
                                let mut psum = 0.0;
                                for d in 0..p {
                                    let ang = std::f64::consts::PI * (z[i][d] - z[j][d]) / per;
                                    let sa = ang.sin();
                                    let ca = ang.cos();
                                    s2 += sa * sa;
                                    psum += ang * sa * ca;
                                }
                                s2s[f] = s2;
                                psums[f] = psum;
                                val *= (-2.0 * s2 * inv0).exp();
                            }
                            _ => {
                                let mut dot = 0.0;
                                for d in 0..p {
                                    dot += z[i][d] * z[j][d];
                                }
                                val *= dot / pf;
                            }
                        }
                    }
                    g[amp] += 2.0 * wij * val;
                    let c = wij * val;
                    let mut o2 = amp + 1;
                    for f in 0..nf {
                        match term[f] {
                            "squared_exponential" => {
                                let inv = &dec.inv_len2[t][f];
                                for d in 0..p {
                                    let diff = z[i][d] - z[j][d];
                                    g[o2 + d] += c * diff * diff * inv[d];
                                }
                                o2 += p;
                            }
                            "rational_quadratic" => {
                                let inv = &dec.inv_len2[t][f];
                                let al = dec.alpha[t][f];
                                let s = r2s[f] / (2.0 * al);
                                let denom = 1.0 + s;
                                for d in 0..p {
                                    let diff = z[i][d] - z[j][d];
                                    g[o2 + d] += c * diff * diff * inv[d] / denom;
                                }
                                g[o2 + p] += c * al * (s / denom - denom.ln());
                                o2 += p + 1;
                            }
                            "periodic" => {
                                let inv0 = dec.inv_len2[t][f][0];
                                g[o2] += c * 4.0 * inv0 * psums[f];
                                g[o2 + 1] += c * 4.0 * s2s[f] * inv0;
                                o2 += 2;
                            }
                            _ => {}
                        }
                    }
                    off = o2;
                }
            }
        }
        let scale = 1.0 / mo as f64;
        for q in 0..noise_idx {
            let step = (scale * g[q]).clamp(-0.5, 0.5);
            theta[q] = (theta[q] + step).clamp(lo[q], hi[q]);
        }
        // Noise floored at exp(-6) of the LARGEST term amplitude (the same
        // conditioning jitter as a single-kernel Gaussian process).
        let g_sn = g[noise_idx] * dec.noise2;
        let mut max_amp = f64::NEG_INFINITY;
        for &ai in &amp_idx {
            if theta[ai] > max_amp {
                max_amp = theta[ai];
            }
        }
        let step_sn = (scale * g_sn).clamp(-0.5, 0.5);
        theta[noise_idx] = (theta[noise_idx] + step_sn).clamp(max_amp - 6.0, max_amp);
    }
    let dec = kernel_decode(kernel_terms, &theta, p);
    let mb = basis_size;
    let mut a = vec![0.0; mb * mb];
    for i in 0..mb {
        a[i * mb + i] = kernel_value(kernel_terms, &dec, pf, &z[i], &z[i]) + dec.noise2;
        for j in (i + 1)..mb {
            let v = kernel_value(kernel_terms, &dec, pf, &z[i], &z[j]);
            a[i * mb + j] = v;
            a[j * mb + i] = v;
        }
    }
    let alpha = cholesky_solve(&mut a, &yc, mb);
    GaussianProcessCompositeKernelRegression { means, scales, basis: z, alpha, theta, y_mean }
}

fn gaussian_process_composite_kernel_regression_predict(state: &GaussianProcessCompositeKernelRegression,
                                                        x: &[Vec<f64>], kernel_terms: &[Vec<&str>])
                                                        -> Vec<f64> {
    let p = state.means.len();
    if state.basis.is_empty() {
        return x.iter().map(|_| state.y_mean.clamp(-60.0, 60.0)).collect();
    }
    let pf = p as f64;
    let dec = kernel_decode(kernel_terms, &state.theta, p);
    let mut predictions = Vec::new();
    for row in x {
        let zr: Vec<f64> = (0..p.min(row.len()))
            .map(|j| (row[j] - state.means[j]) / state.scales[j])
            .collect();
        let mut s = state.y_mean;
        for b in 0..state.basis.len() {
            s += state.alpha[b] * kernel_value(kernel_terms, &dec, pf, &zr, &state.basis[b]);
        }
        predictions.push(s.clamp(-60.0, 60.0));
    }
    predictions
}"#;

const KSMOOTH_SRC: &str = r#"struct NadarayaWatsonRegression {
    means: Vec<f64>,
    scales: Vec<f64>,
    z: Vec<Vec<f64>>,
    ys: Vec<f64>,
    y_mean: f64,
}

{DOC}
fn nadaraya_watson_regression_fit(x: &[Vec<f64>], y: &[f64]) -> NadarayaWatsonRegression {
    let n = x.len().min(y.len());
    let p = x[0].len();
    let (means, scales) = std_fit(x);
    let mut y_mean = 0.0;
    for i in 0..n {
        y_mean += y[i];
    }
    y_mean /= n.max(1) as f64;
    if n < 8 || p == 0 {
        return NadarayaWatsonRegression {
            means,
            scales,
            z: Vec::new(),
            ys: Vec::new(),
            y_mean,
        };
    }
    let z: Vec<Vec<f64>> = x[..n]
        .iter()
        .map(|row| (0..p).map(|j| (row[j] - means[j]) / scales[j]).collect())
        .collect();
    let ys: Vec<f64> = y[..n].to_vec();
    NadarayaWatsonRegression { means, scales, z, ys, y_mean }
}

fn nadaraya_watson_regression_predict(state: &NadarayaWatsonRegression, x: &[Vec<f64>],
                                      k: usize, gamma: f64) -> Vec<f64> {
    let p = state.means.len();
    if state.z.is_empty() {
        return x.iter().map(|_| state.y_mean.clamp(-60.0, 60.0)).collect();
    }
    let pf = p as f64;
    let kk = k.min(state.z.len());
    let mut predictions = Vec::new();
    for row in x {
        let zr: Vec<f64> = (0..p.min(row.len()))
            .map(|j| (row[j] - state.means[j]) / state.scales[j])
            .collect();
        let mut d2s = Vec::new();
        for zb in &state.z {
            let mut d2 = 0.0;
            for d in 0..zr.len().min(zb.len()) {
                let t = zr[d] - zb[d];
                d2 += t * t;
            }
            d2s.push(d2);
        }
        let mut order: Vec<usize> = (0..state.z.len()).collect();
        order.sort_by(|&a, &b| d2s[a].total_cmp(&d2s[b]).then(a.cmp(&b)));
        let base = d2s[order[0]];
        let mut num = 0.0;
        let mut den = 0.0;
        for q in 0..kk {
            let i = order[q];
            let w = (-gamma * (d2s[i] - base) / pf).exp();
            num += w * state.ys[i];
            den += w;
        }
        predictions.push((num / den).clamp(-60.0, 60.0));
    }
    predictions
}"#;

const STACK_SRC: &str = r#"struct GreedyEnsembleSelection {
    weights: Vec<f64>,
    ridge_a: Option<RidgeRegression>,
    ridge_b: Option<RidgeRegression>,
    net: Option<ElasticNet>,
    trees: Option<LeafwiseBoostedTrees>,
    kernel: Option<KernelRidgeRegression>,
}

/// Fit library member b on the fold-train rows and predict the held fold.
fn ensemble_member_holdout(b: usize, xtr: &[Vec<f64>], ytr: &[f64], xva: &[Vec<f64>]) -> Vec<f64> {
    if b == 0 {
        return ridge_regression_predict(&ridge_regression_fit(xtr, ytr, 0.02), xva);
    }
    if b == 1 {
        return ridge_regression_predict(&ridge_regression_fit(xtr, ytr, 0.002), xva);
    }
    if b == 2 {
        return elastic_net_predict(&elastic_net_fit(xtr, ytr, 0.001, 0.5, 60), xva);
    }
    if b == 3 {
        return leafwise_boosted_trees_predict(
            &leafwise_boosted_trees_fit(xtr, ytr, {LEAVES}, {LR}, {BAGS}, 500, 0.7, 1.0, 3, 20, false, 0.0, false),
            xva,
        );
    }
    kernel_ridge_regression_predict(&kernel_ridge_regression_fit(xtr, ytr, {M}), xva)
}

{DOC}
fn greedy_ensemble_selection_fit(x: &[Vec<f64>], y: &[f64]) -> GreedyEnsembleSelection {
    let n = x.len().min(y.len());
    if n < 20 {
        return GreedyEnsembleSelection {
            weights: vec![1.0, 0.0, 0.0, 0.0, 0.0],
            ridge_a: Some(ridge_regression_fit(x, y, 0.02)),
            ridge_b: None,
            net: None,
            trees: None,
            kernel: None,
        };
    }
    let mut oof: Vec<Vec<f64>> = vec![vec![0.0; n]; 5];
    for f in 0..5 {
        let lo = f * n / 5;
        let hi = (f + 1) * n / 5;
        if lo >= hi {
            continue;
        }
        let mut xtr: Vec<Vec<f64>> = Vec::new();
        let mut ytr: Vec<f64> = Vec::new();
        for r in 0..lo {
            xtr.push(x[r].clone());
            ytr.push(y[r]);
        }
        for r in hi..n {
            xtr.push(x[r].clone());
            ytr.push(y[r]);
        }
        let xva: Vec<Vec<f64>> = x[lo..hi].to_vec();
        for b in 0..5 {
            let preds = ensemble_member_holdout(b, &xtr, &ytr, &xva);
            for k in 0..(hi - lo) {
                oof[b][lo + k] = preds[k];
            }
        }
    }
    let yn = &y[..n];
    let mut counts = [0usize; 5];
    let mut total = 0usize;
    let mut sum_vec = vec![0.0; n];
    let mut current = f64::NEG_INFINITY;
    for _ in 0..20 {
        let mut best_b = usize::MAX;
        let mut best_score = current;
        for b in 0..5 {
            let denom = (total + 1) as f64;
            let blend: Vec<f64> = (0..n).map(|i| (sum_vec[i] + oof[b][i]) / denom).collect();
            let s = r2_score(yn, &blend);
            if s > best_score {
                best_score = s;
                best_b = b;
            }
        }
        if best_b == usize::MAX {
            break;
        }
        for i in 0..n {
            sum_vec[i] += oof[best_b][i];
        }
        counts[best_b] += 1;
        total += 1;
        current = best_score;
    }
    if total == 0 {
        counts[0] = 1;
        total = 1;
    }
    let weights: Vec<f64> = counts.iter().map(|&c| c as f64 / total as f64).collect();
    GreedyEnsembleSelection {
        weights,
        ridge_a: if counts[0] > 0 { Some(ridge_regression_fit(x, y, 0.02)) } else { None },
        ridge_b: if counts[1] > 0 { Some(ridge_regression_fit(x, y, 0.002)) } else { None },
        net: if counts[2] > 0 { Some(elastic_net_fit(x, y, 0.001, 0.5, 60)) } else { None },
        trees: if counts[3] > 0 {
            Some(leafwise_boosted_trees_fit(x, y, {LEAVES}, {LR}, {BAGS}, 500, 0.7, 1.0, 3, 20, false, 0.0, false))
        } else {
            None
        },
        kernel: if counts[4] > 0 { Some(kernel_ridge_regression_fit(x, y, {M})) } else { None },
    }
}

fn greedy_ensemble_selection_predict(state: &GreedyEnsembleSelection, x: &[Vec<f64>]) -> Vec<f64> {
    let mut acc = vec![0.0; x.len()];
    if let Some(member) = &state.ridge_a {
        let p = ridge_regression_predict(member, x);
        for i in 0..x.len() {
            acc[i] += state.weights[0] * p[i];
        }
    }
    if let Some(member) = &state.ridge_b {
        let p = ridge_regression_predict(member, x);
        for i in 0..x.len() {
            acc[i] += state.weights[1] * p[i];
        }
    }
    if let Some(member) = &state.net {
        let p = elastic_net_predict(member, x);
        for i in 0..x.len() {
            acc[i] += state.weights[2] * p[i];
        }
    }
    if let Some(member) = &state.trees {
        let p = leafwise_boosted_trees_predict(member, x);
        for i in 0..x.len() {
            acc[i] += state.weights[3] * p[i];
        }
    }
    if let Some(member) = &state.kernel {
        let p = kernel_ridge_regression_predict(member, x);
        for i in 0..x.len() {
            acc[i] += state.weights[4] * p[i];
        }
    }
    acc
}"#;

// ------------------------------- model struct ------------------------------

fn rs_bool(b: bool) -> &'static str {
    if b {
        "true"
    } else {
        "false"
    }
}

/// Rust literal for the composite-GP kernel structure, e.g.
/// `[vec!["squared_exponential", "periodic"], vec!["linear"]]`.
fn rs_kernel_terms(spec: &crate::gpr::KernelSpec) -> String {
    let terms: Vec<String> = spec
        .iter()
        .map(|term| {
            let factors: Vec<String> = term
                .iter()
                .map(|p| format!("\"{}\"", crate::spec::kernel_prim_tag(*p)))
                .collect();
            format!("vec![{}]", factors.join(", "))
        })
        .collect();
    format!("[{}]", terms.join(", "))
}

/// (fit-call expression or None for stateless, apply-call template). The
/// `{state}` placeholder becomes `&stageN` / `&self.stageN`; `{state_val}`
/// becomes the bare Copy value (used for the usize-state fourier stage).
/// `n` is the 1-based stage index (used by the per-stage formula functions).
fn op_calls(op: &PlannedOp, n: usize) -> (Option<String>, String) {
    match op {
        PlannedOp::ScalarV2 { .. } => (None, format!("scalar_stage{n}_apply(&x, {{append}})")),
        PlannedOp::RowVectorInputV2 { width } => (
            None,
            format!("row_vector_v2_input(&x, {width}, {{append}})"),
        ),
        PlannedOp::RowVectorV2 { state } => (
            None,
            crate::kernel::row_vector::emit::apply_call(
                state,
                crate::kernel::row_vector::emit::RowRecipeLanguage::Rust,
            ),
        ),
        PlannedOp::Standardize => (
            Some("standard_scaler_fit(&x)".into()),
            "standard_scaler_apply({state}, &x, {append})".into(),
        ),
        PlannedOp::RobustScale => (
            Some("robust_scaler_fit(&x)".into()),
            "robust_scaler_apply({state}, &x, {append})".into(),
        ),
        PlannedOp::Rank => (
            Some("mid_rank_transform_fit(&x)".into()),
            "mid_rank_transform_apply({state}, &x, {append})".into(),
        ),
        PlannedOp::Blom => (
            Some("blom_normal_scores_fit(&x)".into()),
            "blom_normal_scores_apply({state}, &x, {append})".into(),
        ),
        PlannedOp::QuantileBins { bins } => (
            Some(format!("quantile_bin_indexer_fit(&x, {bins})")),
            format!("quantile_bin_indexer_apply({{state}}, &x, {bins}, {{append}})"),
        ),
        PlannedOp::BinOnehot { bins } => (
            Some(format!("quantile_onehot_encoder_fit(&x, {bins})")),
            format!("quantile_onehot_encoder_apply({{state}}, &x, {bins}, {{append}})"),
        ),
        PlannedOp::Hinge { knots } => (
            Some(format!("hinge_basis_fit(&x, {knots})")),
            "hinge_basis_apply({state}, &x, {append})".into(),
        ),
        PlannedOp::WinsorClip { q } => (
            Some(format!("winsor_limiter_fit(&x, {})", fmt_f64(*q))),
            "winsor_limiter_apply({state}, &x, {append})".into(),
        ),
        PlannedOp::Cheb { degree } => (
            None,
            format!("chebyshev_basis_apply(&x, {degree}, {{append}})"),
        ),
        PlannedOp::PolyInteract => (
            Some("pairwise_products_fit(&x)".into()),
            "pairwise_products_apply({state}, &x, {append})".into(),
        ),
        PlannedOp::RffCos { n, gamma, seed } => (
            Some(format!(
                "random_fourier_cosine_features_fit(&x, {n}, {}, {seed})",
                fmt_f64(*gamma)
            )),
            "random_fourier_cosine_features_apply({state}, &x, {append})".into(),
        ),
        PlannedOp::RffSin { n, gamma, seed } => (
            Some(format!(
                "random_fourier_sine_features_fit(&x, {n}, {}, {seed})",
                fmt_f64(*gamma)
            )),
            "random_fourier_sine_features_apply({state}, &x, {append})".into(),
        ),
        PlannedOp::RandProj { k, seed } => (
            Some(format!("gaussian_random_projection_fit(&x, {k}, {seed})")),
            "gaussian_random_projection_apply({state}, &x, {append})".into(),
        ),
        PlannedOp::LandmarkDist { k, gamma, seed } => (
            Some(format!("gaussian_landmark_features_fit(&x, {k}, {seed})")),
            format!(
                "gaussian_landmark_features_apply({{state}}, &x, {}, {{append}})",
                fmt_f64(*gamma)
            ),
        ),
        PlannedOp::ObliqueHinge { n, seed } => (
            Some(format!("oblique_hinge_features_fit(&x, {n}, {seed})")),
            "oblique_hinge_features_apply({state}, &x, {append})".into(),
        ),
        PlannedOp::RowStats => (None, "row_summary_statistics_apply(&x, {append})".into()),
        PlannedOp::PairProd { topk } => (
            Some(format!("correlation_screened_products_fit(&x, y, {topk})")),
            "correlation_screened_products_apply({state}, &x, {append})".into(),
        ),
        PlannedOp::TargetBinEncode { bins, prior } => (
            Some(format!(
                "smoothed_target_bin_encoder_fit(&x, y, {bins}, {})",
                fmt_f64(*prior)
            )),
            "smoothed_target_bin_encoder_apply({state}, &x, {append})".into(),
        ),
        PlannedOp::Sigmoid => (None, "logistic_squash_apply(&x, {append})".into()),
        PlannedOp::Symlog { c } => (
            None,
            format!("signed_log_apply(&x, {}, {{append}})", fmt_f64(*c)),
        ),
        PlannedOp::SelectCorr { k, by_var } => (
            Some(format!("column_select_fit(&x, y, {k}, {by_var})")),
            "column_select_apply({state}, &x, {append})".into(),
        ),
        PlannedOp::Impute {
            strategy,
            value,
            add_indicator,
        } => (
            Some(format!(
                "zero_imputer_fit(&x, {strategy}, {}, {add_indicator})",
                fmt_f64(*value)
            )),
            "zero_imputer_apply({state}, &x, {append})".into(),
        ),
        PlannedOp::PairRatio { topk } => (
            Some(format!("ratio_screened_pairs_fit(&x, y, {topk})")),
            "ratio_screened_pairs_apply({state}, &x, {append})".into(),
        ),
        PlannedOp::Spline { knots } => (
            Some(format!("natural_cubic_spline_fit(&x, {knots})")),
            "natural_cubic_spline_apply({state}, &x, {append})".into(),
        ),
        PlannedOp::Step { t } => (
            None,
            format!("step_indicator_apply(&x, {}, {{append}})", fmt_f64(*t)),
        ),
        PlannedOp::NormRow => (None, "unit_norm_rows_apply(&x, {append})".into()),
        PlannedOp::Polar => (None, "polar_coordinates_apply(&x, {append})".into()),
        PlannedOp::Lag { k } => (None, format!("shift_columns_apply(&x, {k}, {{append}})")),
        PlannedOp::Roll { k } => (None, format!("rotate_columns_apply(&x, {k}, {{append}})")),
        PlannedOp::FftMag { k } => (
            Some(format!("fourier_magnitudes_fit(&x, {k})")),
            "fourier_magnitudes_apply({state_val}, &x, {append})".into(),
        ),
        PlannedOp::RowSpectral => (None, "row_spectral_summary_apply(&x, {append})".into()),
        PlannedOp::RowDiffStats => (None, "row_difference_statistics_apply(&x, {append})".into()),
        PlannedOp::SvdProject { k, iters, seed } => (
            Some(format!(
                "principal_component_projection_fit(&x, {k}, {iters}, {seed})"
            )),
            "principal_component_projection_apply({state}, &x, {append})".into(),
        ),
        PlannedOp::Nystrom { k, gamma, seed } => (
            Some(format!("nystrom_landmark_features_fit(&x, {k}, {seed})")),
            format!(
                "nystrom_landmark_features_apply({{state}}, &x, {}, {{append}})",
                fmt_f64(*gamma)
            ),
        ),
        PlannedOp::MapExpr { .. } => (
            None,
            format!("elementwise_formula{n}_apply(&x, {{append}})"),
        ),
        PlannedOp::DeriveExpr { .. } => (None, format!("derived_column{n}_apply(&x, {{append}})")),
    }
}

fn head_calls(head: &PlannedHead) -> (String, String, &'static str) {
    match head {
        PlannedHead::Ridge { lambda_frac } => (
            format!("ridge_regression_fit(&x, y, {})", fmt_f64(*lambda_frac)),
            "ridge_regression_predict(&self.head, &x)".into(),
            "RidgeRegression",
        ),
        PlannedHead::RidgeGcv => (
            "gcv_ridge_regression_fit(&x, y)".into(),
            "gcv_ridge_regression_predict(&self.head, &x)".into(),
            "GcvRidgeRegression",
        ),
        PlannedHead::ElasticNet {
            alpha,
            l1_ratio,
            sweeps,
        } => (
            format!(
                "elastic_net_fit(&x, y, {}, {}, {sweeps})",
                fmt_f64(*alpha),
                fmt_f64(*l1_ratio)
            ),
            "elastic_net_predict(&self.head, &x)".into(),
            "ElasticNet",
        ),
        PlannedHead::Gbm(g) => (
            format!(
                "gradient_boosted_trees_fit(&x, y, {}, {}, {}, {}, {}, {}, {})",
                g.rounds,
                g.depth,
                fmt_f64(g.lr),
                fmt_f64(g.lambda),
                g.bins,
                g.min_leaf,
                g.max_features
            ),
            "gradient_boosted_trees_predict(&self.head, &x)".into(),
            "GradientBoostedTrees",
        ),
        PlannedHead::Blend { lambda_frac, gbm } => (
            format!(
                "blended_regression_fit(&x, y, {}, {}, {}, {}, {}, {}, {}, {})",
                fmt_f64(*lambda_frac),
                gbm.rounds,
                gbm.depth,
                fmt_f64(gbm.lr),
                fmt_f64(gbm.lambda),
                gbm.bins,
                gbm.min_leaf,
                gbm.max_features
            ),
            "blended_regression_predict(&self.head, &x)".into(),
            "BlendedRegression",
        ),
        PlannedHead::HistGbm2 {
            leaves,
            lr,
            bags,
            rounds,
            row_subsample,
            col_subsample,
            min_leaf,
            val_pct,
            val_shuffle,
            huber_delta,
            stop_mae,
        } => (
            format!(
                "leafwise_boosted_trees_fit(&x, y, {leaves}, {}, {bags}, {rounds}, {}, {}, \
                 {min_leaf}, {val_pct}, {}, {}, {})",
                fmt_f64(*lr),
                fmt_f64(*row_subsample),
                fmt_f64(*col_subsample),
                rs_bool(*val_shuffle),
                fmt_f64(*huber_delta),
                rs_bool(*stop_mae)
            ),
            "leafwise_boosted_trees_predict(&self.head, &x)".into(),
            "LeafwiseBoostedTrees",
        ),
        PlannedHead::Krr { m } => (
            format!("kernel_ridge_regression_fit(&x, y, {m})"),
            "kernel_ridge_regression_predict(&self.head, &x)".into(),
            "KernelRidgeRegression",
        ),
        PlannedHead::Gpr { m, steps } => (
            format!("gaussian_process_ard_regression_fit(&x, y, {m}, {steps})"),
            "gaussian_process_ard_regression_predict(&self.head, &x)".into(),
            "GaussianProcessArdRegression",
        ),
        PlannedHead::GprKernel { m, steps, spec, .. } => {
            let terms = rs_kernel_terms(spec);
            (
                format!(
                    "gaussian_process_composite_kernel_regression_fit(&x, y, {m}, {steps}, &{terms})"
                ),
                format!(
                    "gaussian_process_composite_kernel_regression_predict(&self.head, &x, &{terms})"
                ),
                "GaussianProcessCompositeKernelRegression",
            )
        }
        PlannedHead::Ksmooth { k, gamma } => (
            "nadaraya_watson_regression_fit(&x, y)".into(),
            format!(
                "nadaraya_watson_regression_predict(&self.head, &x, {k}, {})",
                fmt_f64(*gamma)
            ),
            "NadarayaWatsonRegression",
        ),
        PlannedHead::Stack => (
            "greedy_ensemble_selection_fit(&x, y)".into(),
            "greedy_ensemble_selection_predict(&self.head, &x)".into(),
            "GreedyEnsembleSelection",
        ),
        PlannedHead::Portfolio(_) => (
            "internally_validated_ensemble_fit(&x, y)".into(),
            "internally_validated_ensemble_predict(&self.head, &x)".into(),
            "InternallyValidatedEnsemble",
        ),
    }
}

fn model_pipeline(
    target: Option<&PlannedTarget>,
    stages: &[PlannedStage],
    head: &PlannedHead,
    fast: bool,
    guard_zero_width_roll: bool,
) -> String {
    let (head_fit, head_predict, head_ty) = head_calls(head);
    let mut lines: Vec<String> = Vec::new();
    if target.is_some() {
        lines.push(format!(
            "/// A fitted target transform, feature stages applied in order, then the {} head;",
            head_fn_name(head)
        ));
        lines.push("/// predictions are mapped back through the transform's inverse.".to_string());
    } else {
        lines.push(format!(
            "/// Feature stages applied in order, then the {} head.",
            head_fn_name(head)
        ));
    }
    lines.push("struct Model {".to_string());
    if let Some(target) = target {
        if let Some(ty) = target_struct_name(target.kind) {
            lines.push(format!("    target: {ty},"));
        }
    }
    for (i, stage) in stages.iter().enumerate() {
        if let Some(ty) = op_struct_name(&stage.op) {
            lines.push(format!("    stage{}: {ty},", i + 1));
        }
    }
    lines.push(format!("    head: {head_ty},"));
    lines.push("}".to_string());
    lines.push(String::new());
    lines.push("impl Model {".to_string());
    lines.push("    /// Fit every stage in order on the training rows.".to_string());
    lines.push("    fn fit(x: &[Vec<f64>], y: &[f64]) -> Model {".to_string());
    let raw_scalar = matches!(
        stages.first().map(|s| &s.op),
        Some(PlannedOp::ScalarV2 { .. })
    );
    if raw_scalar {
        lines.push("        assert!(!x.is_empty() && y.len() == x.len() && y.iter().all(|v| v.is_finite() && v.abs() <= 1e12), \"scalar training labels\");".to_string());
    }
    // B12: preserve the explicit v2 fit contract across native recipes.
    if stages
        .iter()
        .any(|s| matches!(s.op, PlannedOp::RowVectorInputV2 { .. }))
    {
        lines.push("        assert!(y.len() == x.len() && y.iter().all(|v| v.is_finite() && v.abs() <= 1e12), \"row-vector training labels\");".to_string());
    }

    lines.push("        if x.is_empty() || y.len() < x.len() {".to_string());
    lines.push("            eprintln!(\"training data is empty or target is short\");".to_string());
    lines.push("            std::process::exit(2);".to_string());
    lines.push("        }".to_string());
    if guard_zero_width_roll {
        lines.push("        if x.iter().any(Vec::is_empty) {".to_string());
        lines.push(
            "            eprintln!(\"stage 0 needs at least 1 input columns, got 0\");".to_string(),
        );
        lines.push("            std::process::exit(2);".to_string());
        lines.push("        }".to_string());
    }
    lines.push(
        if raw_scalar {
            "        let x = x.to_vec();"
        } else {
            "        let x = sanitize(x);"
        }
        .to_string(),
    );
    let mut field_names: Vec<String> = Vec::new();
    if let Some(target) = target {
        // The transform is fitted on the raw train target FIRST; every stage
        // and the head see only the transformed target.
        let (fit_call, apply_call, _) = target_calls(target);
        if let Some(fit_call) = fit_call {
            lines.push(format!("        let target = {fit_call};"));
            field_names.push("target".to_string());
        }
        lines.push(format!("        let y = {apply_call};"));
        lines.push("        let y = &y[..];".to_string());
    }
    for (i, stage) in stages.iter().enumerate() {
        let n = i + 1;
        let (fit_call, apply_call) = op_calls(&stage.op, n);
        let state = format!("stage{n}");
        if let Some(fit_call) = fit_call {
            lines.push(format!("        let {state} = {fit_call};"));
            field_names.push(state.clone());
        }
        let apply = apply_call
            .replace("{state_val}", &state)
            .replace("{state}", &format!("&{state}"))
            .replace("{append}", rs_bool(stage.append));
        lines.push(format!("        let x = {apply};"));
    }
    lines.push(format!("        let head = {head_fit};"));
    if raw_scalar && matches!(head, PlannedHead::Ridge { .. }) {
        lines.push("        let mut head = head; if y.iter().all(|v| v.to_bits() == (-0.0f64).to_bits()) {head.w[0]=-0.0;}".to_string());
    }

    let mut init = field_names.join(", ");
    if !init.is_empty() {
        init.push_str(", ");
    }
    lines.push(format!("        Model {{ {init}head }}"));
    lines.push("    }".to_string());
    lines.push(String::new());
    lines.push("    /// Apply the fitted stages, then the regression head.".to_string());
    if fast {
        lines.push("    fn predict_rows(&self, x: &[Vec<f64>]) -> Vec<f64> {".to_string());
    } else {
        lines.push("    fn predict(&self, x: &[Vec<f64>]) -> Vec<f64> {".to_string());
    }
    if guard_zero_width_roll {
        lines.push("        if x.iter().any(Vec::is_empty) {".to_string());
        lines.push(
            "            eprintln!(\"stage 0 needs at least 1 input columns, got 0\");".to_string(),
        );
        lines.push("            std::process::exit(2);".to_string());
        lines.push("        }".to_string());
    }
    lines.push(
        if raw_scalar {
            "        let x = x.to_vec();"
        } else {
            "        let x = sanitize(x);"
        }
        .to_string(),
    );
    for (i, stage) in stages.iter().enumerate() {
        let n = i + 1;
        let (_, apply_call) = op_calls(&stage.op, n);
        let apply = apply_call
            .replace("{state_val}", &format!("self.stage{n}"))
            .replace("{state}", &format!("&self.stage{n}"))
            .replace("{append}", rs_bool(stage.append));
        lines.push(format!("        let x = {apply};"));
    }
    match target {
        None => lines.push(format!("        {head_predict}")),
        // Exact inverse back to the original target scale, then the kind's
        // final prediction clamp — the same order as the reference model.
        // Range-aware kinds carry their train-range clamp inside the inverse.
        Some(target) => {
            let (_, _, invert) = target_calls(target);
            lines.push(format!("        let predictions = {head_predict};"));
            lines.push("        predictions".to_string());
            lines.push("            .iter()".to_string());
            if crate::spec::target_range_aware(target.kind) {
                lines.push(format!("            .map(|z| {invert})"));
            } else {
                lines.push(format!("            .map(|z| {invert}.clamp(-60.0, 60.0))"));
            }
            lines.push("            .collect()".to_string());
        }
    }
    lines.push("    }".to_string());
    if guard_zero_width_roll {
        lines.push(String::new());
        lines.push(
            "    fn try_predict(&self, x: &[Vec<f64>]) -> Result<Vec<f64>, &'static str> {"
                .to_string(),
        );
        lines.push("        if x.iter().any(Vec::is_empty) {".to_string());
        lines.push(
            "            return Err(\"stage 0 needs at least 1 input columns, got 0\");"
                .to_string(),
        );
        lines.push("        }".to_string());
        lines.push("        Ok(self.predict(x))".to_string());
        lines.push("    }".to_string());
    }
    if fast {
        lines.push(String::new());
        lines.extend(rs_fast_predict_wrapper());
    }
    // B05–B07: public prediction faults on the typed one-stage recipe.
    if raw_scalar && stages.len() == 1 && target.is_none() {
        lines.push("    fn predict_with_faults(&self, x: &[Vec<f64>]) -> (Vec<f64>,Vec<u32>) {let (_,faults)=scalar_stage1_values(x);(self.predict(x),faults)}".to_string());
    }
    lines.push(String::new());
    lines.extend(rs_score_method());
    lines.push("}".to_string());
    lines.join("\n")
}

/// The `score` method (coefficient of determination) shared by both Model
/// impls; self-contained so it needs no helper.
fn rs_score_method() -> Vec<String> {
    [
        "    /// Coefficient of determination R^2 of predict against y.",
        "    fn score(&self, x: &[Vec<f64>], y: &[f64]) -> f64 {",
        "        let predictions = self.predict(x);",
        "        let n = y.len().min(predictions.len());",
        "        if n == 0 {",
        "            return 0.0;",
        "        }",
        "        let mut mean = 0.0;",
        "        for i in 0..n {",
        "            mean += y[i];",
        "        }",
        "        mean /= n as f64;",
        "        let mut ss_res = 0.0;",
        "        let mut ss_tot = 0.0;",
        "        for i in 0..n {",
        "            let d = y[i] - predictions[i];",
        "            ss_res += d * d;",
        "            let t = y[i] - mean;",
        "            ss_tot += t * t;",
        "        }",
        "        if ss_tot <= 0.0 {",
        "            return 0.0;",
        "        }",
        "        1.0 - ss_res / ss_tot",
        "    }",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

// -------------------------------- symbolic ---------------------------------

fn expression_fn(source: &str, expr: &crate::mathexpr::MathAst) -> String {
    let features = expr.features();
    let mut lines: Vec<String> = Vec::new();
    lines.push(format!(
        "/// Evaluate f(row) = {source} with guarded arithmetic."
    ));
    lines.push("fn expression_value(row: &[f64]) -> f64 {".to_string());
    if features.uses_x {
        lines.push(
            "    // The formula names the element variable x, which has no meaning for a whole"
                .to_string(),
        );
        lines.push("    // row, so every row maps to 0.0 (mirroring the reference).".to_string());
        lines.push("    let _ = row;".to_string());
        lines.push("    0.0".to_string());
        lines.push("}".to_string());
        return lines.join("\n");
    }
    if let Some(maxc) = features.max_col {
        lines.push(format!("    if row.len() <= {maxc} {{"));
        lines.push("        return 0.0;".to_string());
        lines.push("    }".to_string());
    }
    if features.max_col.is_none() && features.vars.is_empty() {
        lines.push("    let _ = row;".to_string());
    }
    if features.funcs.contains(&crate::mathexpr::MathFunc::Clip) {
        lines.push("    let mut math_failed = false;".to_string());
        lines.push(format!("    let value = {};", expr.to_rust()));
        lines.push("    if math_failed { return 0.0; }".to_string());
    } else {
        lines.push(format!("    let value = {};", expr.to_rust()));
    }
    lines.push("    if value.is_finite() {".to_string());
    lines.push("        value.clamp(-1e12, 1e12)".to_string());
    lines.push("    } else {".to_string());
    lines.push("        0.0".to_string());
    lines.push("    }".to_string());
    lines.push("}".to_string());
    lines.join("\n")
}

fn model_symbolic(affine: bool, fast: bool) -> String {
    let mut lines: Vec<String> = Vec::new();
    if affine {
        lines.push(
            "/// Closed-form expression with a least-squares affine calibration.".to_string(),
        );
    } else {
        lines.push("/// Closed-form expression applied directly.".to_string());
    }
    lines.push("struct Model {".to_string());
    lines.push("    slope: f64,".to_string());
    lines.push("    intercept: f64,".to_string());
    lines.push("}".to_string());
    lines.push(String::new());
    lines.push("impl Model {".to_string());
    lines.push("    fn fit(x: &[Vec<f64>], y: &[f64]) -> Model {".to_string());
    lines.push("        if x.is_empty() || y.len() < x.len() {".to_string());
    lines.push("            eprintln!(\"training data is empty or target is short\");".to_string());
    lines.push("            std::process::exit(2);".to_string());
    lines.push("        }".to_string());
    if affine {
        lines.push(
            "        let f: Vec<f64> = x.iter().map(|row| expression_value(row)).collect();"
                .to_string(),
        );
        lines.push("        let n = f.len() as f64;".to_string());
        lines.push("        let mut fm = 0.0;".to_string());
        lines.push("        for v in &f {".to_string());
        lines.push("            fm += *v;".to_string());
        lines.push("        }".to_string());
        lines.push("        fm /= n;".to_string());
        lines.push("        let mut ym = 0.0;".to_string());
        lines.push("        for i in 0..f.len() {".to_string());
        lines.push("            ym += y[i];".to_string());
        lines.push("        }".to_string());
        lines.push("        ym /= n;".to_string());
        lines.push("        let mut cov = 0.0;".to_string());
        lines.push("        let mut var = 0.0;".to_string());
        lines.push("        for i in 0..f.len() {".to_string());
        lines.push("            let d = f[i] - fm;".to_string());
        lines.push("            cov += d * (y[i] - ym);".to_string());
        lines.push("            var += d * d;".to_string());
        lines.push("        }".to_string());
        lines.push("        if var > 1e-24 {".to_string());
        lines.push("            let slope = cov / var;".to_string());
        lines.push("            Model { slope, intercept: ym - slope * fm }".to_string());
        lines.push("        } else {".to_string());
        lines.push("            Model { slope: 0.0, intercept: ym }".to_string());
        lines.push("        }".to_string());
    } else {
        lines.push("        Model { slope: 1.0, intercept: 0.0 }".to_string());
    }
    lines.push("    }".to_string());
    lines.push(String::new());
    lines.push(
        "    /// slope * f(row) + intercept, clipped to [-60, 60]; non-finite maps to 0."
            .to_string(),
    );
    if fast {
        lines.push("    fn predict_rows(&self, x: &[Vec<f64>]) -> Vec<f64> {".to_string());
    } else {
        lines.push("    fn predict(&self, x: &[Vec<f64>]) -> Vec<f64> {".to_string());
    }
    lines.push("        let mut predictions = Vec::new();".to_string());
    lines.push("        for row in x {".to_string());
    lines.push("            let value = expression_value(row);".to_string());
    lines.push("            let z = self.slope * value + self.intercept;".to_string());
    lines.push("            if z.is_finite() {".to_string());
    lines.push("                predictions.push(z.clamp(-60.0, 60.0));".to_string());
    lines.push("            } else {".to_string());
    lines.push("                predictions.push(0.0);".to_string());
    lines.push("            }".to_string());
    lines.push("        }".to_string());
    lines.push("        predictions".to_string());
    lines.push("    }".to_string());
    if fast {
        lines.push(String::new());
        lines.extend(rs_fast_predict_wrapper());
    }
    lines.push(String::new());
    lines.extend(rs_score_method());
    lines.push("}".to_string());
    lines.join("\n")
}

// ----------------------------------- I/O -----------------------------------

const IO_BLOCK: &str = r#"/// Parse one CSV cell; blank, malformed, or non-finite cells become 0.0.
fn parse_cell(cell: &str) -> f64 {
    let text = cell.trim();
    if text.is_empty() || text.contains('_') {
        return 0.0;
    }
    match text.parse::<f64>() {
        Ok(value) => {
            if value.is_finite() {
                value
            } else {
                0.0
            }
        }
        Err(_) => 0.0,
    }
}

fn looks_numeric(cell: &str) -> bool {
    let text = cell.trim();
    text.is_empty() || (!text.contains('_') && text.parse::<f64>().is_ok())
}

/// Read CSV rows of floats, skipping blank lines and one header line if present.
fn read_csv(path: &str) -> Vec<Vec<f64>> {
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(err) => {
            eprintln!("cannot read {}: {}", path, err);
            std::process::exit(2);
        }
    };
    let mut records: Vec<Vec<&str>> = Vec::new();
    for line in text.lines() {
        let cells: Vec<&str> = line.split(',').collect();
        if cells.iter().all(|cell| cell.trim().is_empty()) {
            continue;
        }
        records.push(cells);
    }
    if !records.is_empty() && !records[0].iter().all(|cell| looks_numeric(cell)) {
        records.remove(0);
    }
    records
        .iter()
        .map(|cells| cells.iter().map(|cell| parse_cell(cell)).collect())
        .collect()
}

/// Pad short rows with 0.0 (and trim long ones) to the training feature width.
fn pad(mut row: Vec<f64>, width: usize) -> Vec<f64> {
    while row.len() < width {
        row.push(0.0);
    }
    row.truncate(width);
    row
}

/// One shortest round-trip decimal per line (matches Python's repr()).
fn write_predictions(path: &str, predictions: &[f64]) {
    let mut out = String::new();
    for value in predictions {
        out.push_str(&format!("{:?}\n", value));
    }
    if let Err(err) = std::fs::write(path, out) {
        eprintln!("cannot write {}: {}", path, err);
        std::process::exit(2);
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        eprintln!("usage: model train.csv test_features.csv predictions.csv");
        std::process::exit(2);
    }
    let train_rows = read_csv(&args[1]);
    if train_rows.is_empty() {
        eprintln!("training data is empty");
        std::process::exit(2);
    }
    let y_train: Vec<f64> = train_rows.iter().map(|row| row[row.len() - 1]).collect();
    let features: Vec<Vec<f64>> = train_rows.iter().map(|row| row[..row.len() - 1].to_vec()).collect();
    let mut width = 0usize;
    for row in &features {
        if row.len() > width {
            width = row.len();
        }
    }
    let mut x_train: Vec<Vec<f64>> = Vec::new();
    for row in features {
        x_train.push(pad(row, width));
    }
    let mut x_test: Vec<Vec<f64>> = Vec::new();
    for row in read_csv(&args[2]) {
        x_test.push(pad(row, width));
    }
    let model = Model::fit(&x_train, &y_train);
    let predictions = model.predict(&x_test);
    write_predictions(&args[3], &predictions);
}"#;
