//! Per-stage fault telemetry (kernel2 WP-1.4) and the zero-width refusal of the traced entry
//! points (WP-1.8), from `feat_math::kernel::telemetry`.
//!
//! - `apply_op_traced` returns `apply_op`'s rows, and `apply_op_train`'s, bit for bit: at every
//!   op of every fitted program of PR 01's corpus, on the training pass and on held-out rows; for
//!   every expression of PR 01's compatibility corpus as a `map` and a `derive` stage, with and
//!   without append, over every corpus row; for compat_freeze's stage set on its raw stage
//!   matrix; for every op, with append on and off, over raw boundary rows; and past the width cap.
//!   The counts of every expression stage are recounted independently, cell by cell, from the
//!   interpreter's traced calls.
//! - `fit_traced` returns a model whose `serde_json` bytes are `fit`'s, or `fit`'s error, and
//!   `predict_traced` returns `Model::predict`'s bits, for every program of PR 01's corpus (the
//!   portfolio program is skipped, as compat_freeze skips it), for every corpus expression as a
//!   raw and an affine symbolic program, and for the five stored models. The telemetry of the
//!   training pass equals `predict_traced`'s on the training rows, and two runs give identical
//!   telemetry.
//! - The workplan's cases: `log(col(0))` on negative and on positive data, `col(9)` at width 4,
//!   `step(col(0), 100)`; and the epilogue count, a compile failure, the symbolic path, an
//!   explicit portfolio, the empty-row identities and the zero-width refusal.
//!
//! PR 01's program corpus is defined in `tests/compat_freeze.rs`. It is copied here, and
//! `the_program_corpus_is_compat_freezes` proves the copy entry for entry against that file and
//! against the names `compat_freeze/model_serde.v1.json` records.

use feat_math::dsl::{parse_dsl, Stage};
use feat_math::kernel::telemetry::{
    apply_op_traced, fit_traced, predict_traced, PipelineTelemetry, StageFaultCounts,
    ZeroWidthInput, FAULT_SLOTS,
};
use feat_math::kernel::width::{MismatchKind, WidthMismatch, WidthReq};
use feat_math::math_expr::{FaultClass, FaultSet, FaultSeverity, MathExpr};
use feat_math::ops::{apply_op, apply_op_train, fit_op, FittedOp};
use feat_math::{fit, sanitize, GenomeIr, Model, Pcg32, CHAMPION_GENOME, MAX_WIDTH};
use std::collections::BTreeSet;

type Rows = Vec<Vec<f64>>;

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|value| value.to_bits()).collect()
}

fn row_bits(rows: &Rows) -> Vec<Vec<u64>> {
    rows.iter().map(|row| bits(row)).collect()
}

fn fixture_text(relative: &str) -> String {
    let path = format!("{}/tests/fixtures/{relative}", env!("CARGO_MANIFEST_DIR"));
    std::fs::read_to_string(&path).unwrap_or_else(|error| panic!("{path}: {error}"))
}

fn fixture_json(relative: &str) -> serde_json::Value {
    serde_json::from_str(&fixture_text(relative))
        .unwrap_or_else(|error| panic!("{relative}: {error}"))
}

fn parse_program(source: &str) -> GenomeIr {
    GenomeIr::parse(source).unwrap_or_else(|error| panic!("{source}: {error}"))
}

fn parse_stage(text: &str) -> Stage {
    parse_dsl(&format!("(pipe X {text})"))
        .unwrap_or_else(|error| panic!("{text}: {error}"))
        .stages
        .remove(0)
}

fn serialized(model: &Model) -> Vec<u8> {
    serde_json::to_vec(model).expect("serialize the model")
}

// ---------------------------------------------------------------------------------------------
// PR 01's program corpus, copied from tests/compat_freeze.rs
// ---------------------------------------------------------------------------------------------

/// Deterministic regression data, as in `tests/golden.rs` and `tests/compat_freeze.rs`.
fn toy_data(n: usize, cols: usize, seed: u64) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut rng = Pcg32::new(seed, 17);
    let mut x = Vec::new();
    let mut y = Vec::new();
    for _ in 0..n {
        let row: Vec<f64> = (0..cols).map(|_| rng.next_f64() * 4.0 - 2.0).collect();
        let target = 1.7 * row[0] - 0.9 * row[1]
            + 0.5 * (3.0 * row[2]).sin()
            + 0.3 * row[3] * row[4]
            + 0.1 * rng.gaussian();
        x.push(row);
        y.push(target);
    }
    (x, y)
}

/// Training rows (the first 64) and held-out rows (the last 16), as compat_freeze fits them.
fn fitted_surface_data() -> (Rows, Vec<f64>, Rows) {
    let (x, y) = toy_data(80, 5, 42);
    (x[..64].to_vec(), y[..64].to_vec(), x[64..].to_vec())
}

/// Programs with expressions and every expansion op, beside the champion and the ip_scrub set.
const NAMED_PROGRAMS: &[(&str, &str)] = &[
    ("golden_map_derive_ridge", "(pipe X (map expr=\"sigmoid(x) + 0.1*col(0)\" append=true) (derive expr=\"row_mean + row_std - c1/(c0 + 3)\" append=true) (ridge))"),
    ("golden_map_pow_elasticnet", "(pipe X (map expr=\"x^2 - min(x, c2) + tanh(c3)\") (elasticnet alpha=0.005 l1_ratio=0.7))"),
    ("golden_step_map_expr", "(pipe X (map expr=\"step(col(0), 0.0) + 0.5*step(row_mean, c1)\" append=true) (ridge))"),
    ("golden_scalar_batch_map_ridge", "(pipe X (map expr=\"exp(clip(col(0),-2,2)) + log(abs(col(1)) + 1.0) + asinh(col(2)) + atan(col(3))\") (ridge))"),
    ("golden_scalar_batch2_derive_ridge", "(pipe X (derive append=true expr=\"erf(col(0)) + sign(col(1)) * floor(col(2)) + round(col(3))\") (ridge))"),
    ("golden_symbolic_affine", "(sym affine \"sigmoid(col(0)) + 0.5*col(1) - log1p(abs(col(2))) + max(c3, c4^2)\")"),
    ("golden_symbolic_raw", "(sym raw \"clip(row_mean + 0.3*row_std - c0/(c1 + 5), -2, 2)\")"),
    ("golden_step_symbolic", "(sym affine \"step(c0, 0.0) + step(c1, c2) - 0.5\")"),
    ("golden_symbolic_scalar_batch", "(sym affine \"erf(c0) + asinh(c1) - exp(clip(c2, -1, 1)) + sign(c3) * round(c4)\")"),
    ("expressions_1", "(pipe X (map expr=\"abs(x) + sqrt(abs(c0)) - log1p(abs(x))\" append=true) (derive expr=\"sin(c0) + cos(c1) * tanh(c2)\" append=true) (ridge))"),
    ("expressions_2", "(pipe X (map expr=\"sigmoid(x) - min(x, c1) + max(x, c2)\") (derive expr=\"clip(c0, -1, 1) + step(c1, 0.5)\" append=true) (ridge))"),
    ("expressions_3", "(pipe X (map expr=\"exp(clip(x, -3, 3)) + log(abs(x) + 1)\" append=true) (derive expr=\"asinh(c0) + atan(c1) - erf(c2)\" append=true) (ridge))"),
    ("expressions_4", "(pipe X (map expr=\"sign(x) * floor(x) + round(x)\") (derive expr=\"row_sum - row_l1 + row_l2\" append=true) (ridge))"),
    ("expressions_5", "(pipe X (map expr=\"x - row_mean + row_std\" append=true) (derive expr=\"row_min * row_max + c0 / (c1 + 3)\" append=true) (ridge))"),
    ("expressions_6", "(pipe X (map expr=\"x^2 - (-x) * 0.5 + x / (c0 + 2)\") (derive expr=\"c3^3 - c4*c4*c4 + c2^0.5\" append=true) (elasticnet alpha=0.005 l1_ratio=0.7))"),
    ("expansion_symlog", "(pipe X (symlog c=0.7) (ridge))"),
    ("expansion_select_corr", "(pipe X (select_corr k=3 by=corr) (ridge))"),
    ("expansion_impute", "(pipe X (impute strategy=median add_indicator=true) (standardize) (ridge))"),
    ("expansion_pair_ratio", "(pipe X (pair_ratio topk=4 append=true) (ridge))"),
    ("expansion_spline", "(pipe X (spline knots=4) (ridge))"),
    ("recipe_random_features", "(pipe X (rff_cos n=8) (rff_sin n=8) (rand_proj k=4) (landmark_dist k=4) (ridge))"),
    ("recipe_row_features", "(pipe X (oblique_hinge n=4) (row_stats) (pair_prod) (target_bin_encode) (sigmoid) (ridge))"),
    ("recipe_indicator_features", "(pipe X (step t=0.5 append=true) (norm_row) (row_diff_stats) (elasticnet))"),
    ("recipe_logistic_alias", "(pipe X (standardize) (logistic))"),
    ("target_rank_gauss", "(pipe X (target kind=rank_gauss) (ridge))"),
    ("target_log1p", "(pipe X (target kind=log1p) (ridge lambda_frac=0.02))"),
    ("target_winsor", "(pipe X (target kind=winsor q=0.05) (standardize) (ridge))"),
    ("target_zscore", "(pipe X (target kind=zscore) (standardize) (ridge))"),
    ("step_impute_mean", "(pipe X (step t=0.0) (impute strategy=mean add_indicator=true) (ridge))"),
    ("step_impute_const", "(pipe X (step t=0.0 append=true) (impute strategy=const value=0.5) (ridge))"),
    ("select_corr_by_var", "(pipe X (select_corr k=3 by=var) (ridge))"),
    ("ridge_gcv_dual_route", "(pipe X (rff_cos n=96 seed=3) (ridge_gcv))"),
    ("empty_stage_output", "(pipe X (select_corr k=1) (pair_ratio topk=4) (ridge))"),
    ("derive_x_body", "(pipe X (derive expr=\"x\" append=true) (ridge))"),
    ("symbolic_x", "(sym raw \"x\")"),
];

/// `tests/ip_scrub.rs` PIPELINES and SYMBOLIC, verbatim.
const IP_SCRUB_PIPELINES: &[&str] = &[
    "(pipe X (portfolio preset=core))",
    "(pipe X (standardize) (rank) (blom) (quantile_bins) (bin_onehot bins=3) (ridge))",
    "(pipe X (hinge) (winsor_clip) (cheb) (poly_interact append=true) (elasticnet))",
    "(pipe X (rff_cos n=8) (rff_sin n=8) (rand_proj k=4) (landmark_dist k=4) (histgbm rounds=5))",
    "(pipe X (oblique_hinge n=4) (row_stats) (pair_prod) (target_bin_encode) (ensemble rounds=5))",
    "(pipe X (sigmoid) (norm_row) (polar) (lag k=1) (roll k=2) (fft_mag k=2) (ridge))",
    "(pipe X (row_spectral) (row_diff_stats) (svd_project k=2 iters=5) (nystrom k=4) (ridge))",
    "(pipe X (map expr=\"clip(x, -1, 1)\") (derive expr=\"row_l1 + row_l2 + row_min + row_max\") (ridge))",
    "(pipe X (histgbm2 leaves=15 lr=0.08 bags=2))",
    "(pipe X (standardize) (krr m=200))",
    "(pipe X (gpr m=200 steps=10))",
    "(pipe X (gpr kernel=rq+linear m=200 steps=10))",
    "(pipe X (gpr kernel=rbf*periodic m=200 steps=10))",
    "(pipe X (ksmooth k=16 gamma=2.0))",
    "(pipe X (standardize) (cheb degree=3 append=true) (ridge_gcv))",
    "(pipe X (stack))",
];
const IP_SCRUB_SYMBOLIC: &[&str] = &[
    "(sym affine \"tanh(c0) * sqrt(abs(c1)) + min(c2, 1) / max(c3, 1) + sin(c4) + cos(c0)\")",
    "(sym raw \"sigmoid(c0) + 0.5*c1\")",
];

/// compat_freeze's program set, named and in its order.
fn program_set() -> Vec<(String, String)> {
    let mut set = vec![("champion".to_string(), CHAMPION_GENOME.trim().to_string())];
    set.extend(
        NAMED_PROGRAMS
            .iter()
            .map(|(name, source)| (name.to_string(), source.to_string())),
    );
    set.extend(
        IP_SCRUB_PIPELINES
            .iter()
            .enumerate()
            .map(|(i, source)| (format!("ip_scrub_pipeline_{i:02}"), source.to_string())),
    );
    set.extend(
        IP_SCRUB_SYMBOLIC
            .iter()
            .enumerate()
            .map(|(i, source)| (format!("ip_scrub_symbolic_{i}"), source.to_string())),
    );
    let portable = fixture_text("emitted/portable-4lang.dsl");
    set.push(("portable_4lang".to_string(), portable.trim().to_string()));
    set
}

/// compat_freeze skips fitting portfolio programs: their diagnostics carry wall-clock `fit_ms`,
/// so their bytes differ from run to run, and a debug-build fit takes minutes.
fn fit_skipped(source: &str) -> bool {
    source.contains("(portfolio")
}

const COMPAT_FREEZE: &str = include_str!("compat_freeze.rs");

/// The entry lines of the array `const <name>` in compat_freeze.rs, without its comment lines.
fn compat_freeze_entries(name: &str) -> Vec<&'static str> {
    let start = COMPAT_FREEZE
        .find(&format!("const {name}: "))
        .unwrap_or_else(|| panic!("compat_freeze.rs no longer defines {name}"));
    let body = &COMPAT_FREEZE[start..];
    let open = body.find("= &[\n").expect("an array literal") + "= &[\n".len();
    let close = body[open..].find("\n];").expect("the end of the array") + open;
    body[open..close]
        .lines()
        .filter(|line| !line.trim_start().starts_with("//"))
        .collect()
}

// ---------------------------------------------------------------------------------------------
// Independent recount of an expression stage
// ---------------------------------------------------------------------------------------------

/// The fields of `StageFaultCounts`, comparable with an expectation built here.
#[derive(Debug, Default, PartialEq, Eq)]
struct Fields {
    evaluated_cells: u64,
    faulted_cells: [u64; FAULT_SLOTS],
    domain_cells: u64,
    saturation_cells: u64,
    hard_rows: u64,
    compile_failed: bool,
    constant_output: bool,
    epilogue_sanitized: u64,
}

fn fields(counts: &StageFaultCounts) -> Fields {
    Fields {
        evaluated_cells: counts.evaluated_cells,
        faulted_cells: counts.faulted_cells,
        domain_cells: counts.domain_cells,
        saturation_cells: counts.saturation_cells,
        hard_rows: counts.hard_rows,
        compile_failed: counts.compile_failed,
        constant_output: counts.constant_output,
        epilogue_sanitized: counts.epilogue_sanitized,
    }
}

/// The counts of a `map` or `derive` stage whose expression compiles, recounted cell by cell from
/// the interpreter's traced calls (PR 03 proved the compiled form's values and faults equal
/// theirs), with the epilogue applied to the generated values by hand.
fn recount(op: &FittedOp, x: &Rows) -> Fields {
    let (expr, elementwise, append) = match op {
        FittedOp::MapExpr { expr, append, .. } => (expr, true, *append),
        FittedOp::DeriveExpr { expr, append, .. } => (expr, false, *append),
        other => panic!("not an expression stage: {other:?}"),
    };
    assert!(
        expr.compile().is_ok(),
        "recount covers compiled expressions"
    );
    let hard = FaultSeverity::Hard.classes();
    let mut expected = Fields::default();
    let mut first: Option<u64> = None;
    let mut varied = false;
    for row in x {
        let cells: Vec<(f64, FaultSet)> = if elementwise {
            row.iter()
                .map(|value| expr.eval_element_traced(*value, row))
                .collect()
        } else {
            vec![expr.eval_row_traced(row)]
        };
        let mut row_is_hard = false;
        for (value, faults) in &cells {
            expected.evaluated_cells += 1;
            for class in faults.iter() {
                expected.faulted_cells[class.index()] += 1;
            }
            expected.domain_cells += u64::from(faults.intersects(FaultSeverity::Domain.classes()));
            expected.saturation_cells +=
                u64::from(faults.intersects(FaultSeverity::Saturation.classes()));
            row_is_hard |= faults.intersects(hard);
            match first {
                None => first = Some(value.to_bits()),
                Some(bits) => varied |= value.to_bits() != bits,
            }
        }
        if row_is_hard {
            expected.hard_rows += 1;
        }
        let mut merged: Vec<f64> = if append { row.clone() } else { Vec::new() };
        merged.extend(cells.iter().map(|(value, _)| *value));
        merged.truncate(MAX_WIDTH);
        expected.epilogue_sanitized +=
            merged.iter().filter(|value| !value.is_finite()).count() as u64;
    }
    expected.constant_output = first.is_some() && !varied;
    expected
}

/// `apply_op_traced` against `apply_op` and `apply_op_train` on `x`, and its counts against the
/// recount for an expression stage or against "epilogue only" for any other op. Returns the
/// counts.
fn check_op(at: &str, op: &FittedOp, x: &Rows) -> StageFaultCounts {
    let legacy = apply_op(op, x).unwrap();
    let (traced, counts) = apply_op_traced(op, x);
    assert_eq!(row_bits(&traced), row_bits(&legacy), "{at}: apply_op");
    assert_eq!(
        row_bits(&traced),
        row_bits(&apply_op_train(op, x).unwrap()),
        "{at}: apply_op_train"
    );
    match op {
        FittedOp::MapExpr { .. } | FittedOp::DeriveExpr { .. } => {
            assert_eq!(fields(&counts), recount(op, x), "{at}: counts");
        }
        _ => {
            let expression = Fields {
                epilogue_sanitized: counts.epilogue_sanitized,
                ..Fields::default()
            };
            assert_eq!(fields(&counts), expression, "{at}: a non-expression op");
        }
    }
    counts
}

/// Every op of a fitted pipeline, replayed on `rows` as `Model::predict` runs them.
fn replay_ops(at: &str, model: &Model, rows: &Rows) -> Vec<StageFaultCounts> {
    let Model::Pipeline { ops, .. } = model else {
        return Vec::new();
    };
    let mut current = rows.clone();
    sanitize(&mut current);
    let mut counts = Vec::with_capacity(ops.len());
    for (i, op) in ops.iter().enumerate() {
        counts.push(check_op(&format!("{at} op {i}"), op, &current));
        current = apply_op(op, &current).unwrap();
    }
    counts
}

/// The number of telemetry entries a model's pass has.
fn stage_count(model: &Model) -> usize {
    match model {
        Model::Pipeline { ops, .. } => ops.len(),
        Model::Symbolic { .. } => 1,
        Model::Portfolio(_) => 0,
    }
}

// ---------------------------------------------------------------------------------------------
// The corpus: models, predictions and every op
// ---------------------------------------------------------------------------------------------

// ---------------------------------------------------------------------------------------------
// Boundary rows
// ---------------------------------------------------------------------------------------------

// ---------------------------------------------------------------------------------------------
// The workplan's cases (WP-1.4)
// ---------------------------------------------------------------------------------------------

// ---------------------------------------------------------------------------------------------
// The zero-width refusal (WP-1.8)
// ---------------------------------------------------------------------------------------------

#[path = "stage_telemetry/corpus.rs"]
mod corpus;
#[path = "stage_telemetry/edge_cases.rs"]
mod edge_cases;
#[path = "stage_telemetry/entry_points.rs"]
mod entry_points;
#[path = "stage_telemetry/expressions.rs"]
mod expressions;
#[path = "stage_telemetry/fit.rs"]
mod fit;
#[path = "stage_telemetry/stages.rs"]
mod stages;
