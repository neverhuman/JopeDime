//! The static scope check (kernel2 WP-1.6): `kernel::scope::validate_scope`, and the search
//! profile's expression scope, `ExpressionScope::search_profile`.
//!
//! - The workplan's cases, each with its exact result, and the facts they rest on: `row_stats` is
//!   ten columns in the registry and at runtime, and the runtime meets an out-of-range column or a
//!   misplaced `x` only when it evaluates, with the class the static check reports.
//! - Where an error points (stage, portfolio member) and its class, and the programs that cannot
//!   be checked, each with its own reason.
//! - Agreement with `PipelineSlots::validate_legacy_widths` on PR 01's compatibility corpus
//!   (`tests/fixtures/compat_freeze/emitted_bytes.v1.json`, read only), on a grid of every width
//!   rule followed by an expression stage, and on random pipelines. The two intended differences
//!   are checked on their own, with the runtime behaviour they follow.
//! - A portfolio member is checked against the columns it can see, one at the fewest, since the
//!   portfolio drops the columns that are constant in training; pinned against such a fit.
//! - 5,000 chained `MathExpr::mutate_v1` edits under the search profile.
//! - Determinism, with the digest of every displayed result pinned, and the shipped-text lint on
//!   every message.
//!
//! The `AGREEMENT`, `PROPERTY` and `DETERMINISM` lines on stderr are the receipt's numbers:
//! `cargo nextest run --locked -P pr -p feat-math --test scope_validation --no-capture`.

use feat_math::dsl::{DslExpr, ParamValue, Stage};
use feat_math::kernel::pipeline::{stage_class, PipelineSlots, StageClass};
use feat_math::kernel::scope::{validate_scope, ScopeError, MAX_INPUT_WIDTH};
use feat_math::kernel::Error;
use feat_math::math_expr::{Expr, ExpressionScope, FaultClass, FaultSet, MathExpr, Var};
use feat_math::ops::{apply_op, fit_op};
use feat_math::GenomeIr;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fmt;

const COLUMN_OOB: Option<FaultClass> = Some(FaultClass::ColumnOob);
const ILLEGAL_SCOPE: Option<FaultClass> = Some(FaultClass::IllegalScope);

// ---------------------------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------------------------

fn program(source: &str) -> GenomeIr {
    GenomeIr::parse(source).unwrap_or_else(|error| panic!("{source}: {error}"))
}

fn check(source: &str, width: usize) -> Result<(), ScopeError> {
    validate_scope(&program(source), width)
}

#[track_caller]
fn accepts(source: &str, width: usize) {
    if let Err(error) = check(source, width) {
        panic!("{source} at width {width}: {error}");
    }
}

/// `source` at `width` fails with exactly this error, displayed exactly as `display`.
#[track_caller]
fn rejects(
    source: &str,
    width: usize,
    stage: Option<usize>,
    class: Option<FaultClass>,
    reason: &str,
    display: &str,
) {
    let Err(error) = check(source, width) else {
        panic!("{source} at width {width} is accepted");
    };
    assert_eq!(
        (
            error.member,
            error.stage,
            error.class,
            error.reason.as_str()
        ),
        (None, stage, class, reason),
        "{source} at width {width}"
    );
    assert_eq!(error.to_string(), display, "{source} at width {width}");
}

/// `(pipe X (standardize) (<op> expr=<source>) (ridge))`, built directly so that any expression
/// text reaches the check unchanged (the DSL printer leaves a numeric-looking one unquoted).
fn stage_program(op: &str, source: &str) -> GenomeIr {
    let mut stage = Stage::new(op);
    stage
        .params
        .insert("expr".into(), ParamValue::Str(source.into()));
    GenomeIr::Pipeline(DslExpr {
        stages: vec![Stage::new("standardize"), stage, Stage::new("ridge")],
    })
}

/// The one stage of `(pipe X <source> (ridge))`.
fn stage(source: &str) -> Stage {
    let GenomeIr::Pipeline(pipe) = program(&format!("(pipe X {source} (ridge))")) else {
        unreachable!("a pipe is a pipeline");
    };
    pipe.stages[0].clone()
}

fn slots(source: &str) -> PipelineSlots {
    let GenomeIr::Pipeline(pipe) = program(source) else {
        panic!("{source} is not a pipeline");
    };
    PipelineSlots::from_pipeline(&pipe).unwrap_or_else(|error| panic!("{source}: {error}"))
}

/// The message inside a kernel error, without the variant name its `Display` prints.
fn message(error: Error) -> String {
    match error {
        Error::Syntax(text)
        | Error::Domain(text)
        | Error::Bounds(text)
        | Error::Shape(text)
        | Error::Unsupported(text)
        | Error::Resource(text)
        | Error::NonFinite(text) => text,
        Error::IterationLimit => "iteration limit".into(),
        Error::TimeLimit => "time limit".into(),
    }
}

/// The legacy check's verdict, with any stage count allowed: `Ok`, or its error's message.
fn legacy(slots: &PipelineSlots, rows: usize, width: usize) -> Result<(), String> {
    slots
        .validate_legacy_widths(rows, width, usize::MAX)
        .map_err(message)
}

/// `rows` rows of `width` finite cells, and a target, for the runtime checks.
fn data(rows: usize, width: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
    let x = (0..rows)
        .map(|i| {
            (0..width)
                .map(|j| ((i * 7 + j * 3) % 11) as f64 - 5.0)
                .collect()
        })
        .collect();
    let y = (0..rows).map(|i| i as f64).collect();
    (x, y)
}

/// What a walk of the tree finds: the columns read, in reading order; whether `x` is read; the
/// node count; and the depth, the root at depth 1.
struct Shape {
    columns: Vec<usize>,
    reads_x: bool,
    nodes: usize,
    depth: usize,
}

fn shape(root: &Expr) -> Shape {
    fn walk(node: &Expr, level: usize, found: &mut Shape) {
        found.nodes += 1;
        found.depth = found.depth.max(level);
        match node {
            Expr::Var(Var::X) => found.reads_x = true,
            Expr::Var(Var::Col(index)) => found.columns.push(*index),
            Expr::Const(_) | Expr::Var(_) => {}
            Expr::Unary(_, child) => walk(child, level + 1, found),
            Expr::Binary(_, left, right) => {
                walk(left, level + 1, found);
                walk(right, level + 1, found);
            }
            Expr::Call(_, args) => {
                for arg in args {
                    walk(arg, level + 1, found);
                }
            }
        }
    }
    let mut found = Shape {
        columns: Vec::new(),
        reads_x: false,
        nodes: 0,
        depth: 0,
    };
    walk(root, 1, &mut found);
    found
}

/// The oracle for one expression at `width` columns, from the walk alone: the class and the
/// reason a check must report, or `None`.
fn expected_fault(
    expr: &MathExpr,
    width: usize,
    elementwise: bool,
) -> Option<(FaultClass, String)> {
    let found = shape(expr.ast());
    if found.reads_x && !elementwise {
        return Some((
            FaultClass::IllegalScope,
            "variable x is only valid in map expressions".into(),
        ));
    }
    found
        .columns
        .iter()
        .find(|index| **index >= width)
        .map(|index| {
            (
                FaultClass::ColumnOob,
                format!("column index {index} out of bounds for width {width}"),
            )
        })
}

/// Hand-written programs: the workplan's cases, the location and class cases, and programs that
/// cannot be checked.
const HAND_CASES: [&str; 23] = [
    r#"(pipe X (map expr="x") (ridge))"#,
    r#"(pipe X (derive expr="x") (ridge))"#,
    r#"(pipe X (derive expr="col(0)") (ridge))"#,
    r#"(pipe X (derive expr="col(7)") (ridge))"#,
    r#"(pipe X (select_corr k=2) (derive expr="col(3)") (ridge))"#,
    r#"(pipe X (row_stats) (derive expr="col(9)") (ridge))"#,
    r#"(pipe X (row_stats) (derive expr="col(10)") (ridge))"#,
    r#"(pipe X (derive expr="x + c9") (ridge))"#,
    r#"(pipe X (map expr="x + c9") (ridge))"#,
    r#"(pipe X (target kind=log1p) (standardize) (derive expr="c5") (ridge))"#,
    r#"(pipe X (nystrom k=4) (derive expr="c2") (ridge))"#,
    r#"(pipe X (standardize) (derive expr="c599") (ridge))"#,
    r#"(pipe X (derive expr=0.5) (ridge))"#,
    r#"(pipe X (derive expr="c0 +") (ridge))"#,
    "(pipe X (standardize) (map) (ridge))",
    "(pipe X (bogus_op) (ridge))",
    "(pipe X (ridge) (standardize) (ridge))",
    "(pipe X (target kind=log1p) (input) (ridge))",
    "(pipe X (standardize) (pls_project) (ridge))",
    r#"(sym raw "x")"#,
    r#"(sym affine "c4 + row_mean")"#,
    r#"(portfolio (member (pipe X (ridge))) (member (pipe X (row_stats) (derive expr="c10") (ridge))) (blend))"#,
    "(portfolio (member (pipe X (bogus_op))) (blend))",
];

/// Input widths for the determinism and lint sweeps, the out-of-bounds ones included.
const SWEEP_WIDTHS: [usize; 10] = [0, 1, 2, 3, 5, 8, 10, 512, 600, usize::MAX];

const WIDTH_STAGES: [&str; 40] = [
    "(standardize)",
    "(norm_features)",
    "(robust_scale)",
    "(rank)",
    "(blom)",
    "(quantile_bins)",
    "(winsor_clip)",
    "(winsor_moments)",
    "(target_bin_encode)",
    "(sigmoid)",
    "(step t=0.5)",
    "(norm_row)",
    "(polar)",
    "(lag k=1)",
    "(roll k=2)",
    "(symlog)",
    r#"(map expr="x * 2")"#,
    "(bin_onehot bins=3)",
    "(hinge knots=2)",
    "(cheb degree=3)",
    "(poly_interact)",
    "(rff_cos n=4)",
    "(rff_sin n=6)",
    "(oblique_hinge n=3)",
    "(rand_proj k=5)",
    "(landmark_dist k=3)",
    "(nystrom k=4)",
    "(row_stats)",
    "(row_spectral)",
    "(row_diff_stats)",
    r#"(derive expr="c0 - 1")"#,
    "(fft_mag k=3)",
    "(select_corr k=2)",
    "(svd_project k=3)",
    "(pair_prod topk=3)",
    "(pair_ratio topk=4)",
    "(spline knots=4)",
    "(impute)",
    "(impute add_indicator=true)",
    "(pls_project)",
];

mod agreement;
mod cases;
mod determinism;
mod portfolio;
mod search_profile;
