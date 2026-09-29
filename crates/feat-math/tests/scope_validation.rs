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

// ---------------------------------------------------------------------------------------------
// The workplan's cases
// ---------------------------------------------------------------------------------------------

#[test]
fn the_workplan_cases_have_their_exact_results() {
    // `x` is the element a map reads.
    accepts(r#"(pipe X (map expr="x") (ridge))"#, 3);
    // A derived column reads a whole row, where there is no element.
    for width in [1, 3, 8, 512] {
        rejects(
            r#"(pipe X (derive expr="x") (ridge))"#,
            width,
            Some(0),
            ILLEGAL_SCOPE,
            "variable x is only valid in map expressions",
            "stage 0: ILLEGAL_SCOPE: variable x is only valid in map expressions",
        );
    }
    accepts(r#"(pipe X (derive expr="col(0)") (ridge))"#, 1);
    accepts(r#"(pipe X (map expr="col(0)") (ridge))"#, 1);
    rejects(
        r#"(pipe X (derive expr="col(7)") (ridge))"#,
        1,
        Some(0),
        COLUMN_OOB,
        "column index 7 out of bounds for width 1",
        "stage 0: COLUMN_OOB: column index 7 out of bounds for width 1",
    );
    // At input width 8, `select_corr k=2` leaves a lower width of 2.
    rejects(
        r#"(pipe X (select_corr k=2) (derive expr="col(3)") (ridge))"#,
        8,
        Some(1),
        COLUMN_OOB,
        "column index 3 out of bounds for width 2",
        "stage 1: COLUMN_OOB: column index 3 out of bounds for width 2",
    );
    accepts(
        r#"(pipe X (select_corr k=2) (derive expr="col(1)") (ridge))"#,
        8,
    );
    // `row_stats` is ten columns, whatever the input width.
    for width in [1, 3, 20] {
        accepts(
            r#"(pipe X (row_stats) (derive expr="col(9)") (ridge))"#,
            width,
        );
        rejects(
            r#"(pipe X (row_stats) (derive expr="col(10)") (ridge))"#,
            width,
            Some(1),
            COLUMN_OOB,
            "column index 10 out of bounds for width 10",
            "stage 1: COLUMN_OOB: column index 10 out of bounds for width 10",
        );
    }
}

#[test]
fn the_cases_rest_on_the_registry_and_the_runtime() {
    // `row_stats` is `ShapeRule::Fixed(10)` in the registry, and the runtime emits ten values.
    let row_stats = feat_math::registry::operator("row_stats").expect("row_stats is registered");
    assert_eq!(row_stats.shape, feat_math::registry::ShapeRule::Fixed(10));
    let (x, y) = data(6, 3);
    let op = fit_op(&stage("(row_stats)"), &x, &y).expect("row_stats fits");
    assert!(apply_op(&op, &x).iter().all(|row| row.len() == 10));
    // `select_corr k=2` keeps two of eight columns.
    let (x, y) = data(12, 8);
    let op = fit_op(&stage("(select_corr k=2)"), &x, &y).expect("select_corr fits");
    assert!(apply_op(&op, &x).iter().all(|row| row.len() == 2));
    // The runtime meets an out-of-range column or a misplaced `x` only when it evaluates; the
    // stage then substitutes zero, and the trace carries the class the static check reports.
    let column = MathExpr::parse("col(7)").expect("col(7) parses");
    assert!(column.eval_row(&[0.5]).is_err());
    assert_eq!(column.eval_row_traced(&[0.5]), (0.0, FaultSet::COLUMN_OOB));
    assert_eq!(column.eval_row_traced(&[0.5; 8]).1, FaultSet::EMPTY);
    let element = MathExpr::parse("x").expect("x parses");
    assert!(element.eval_row(&[0.5]).is_err());
    assert_eq!(
        element.eval_row_traced(&[0.5]),
        (0.0, FaultSet::ILLEGAL_SCOPE)
    );
    assert_eq!(
        element.eval_element_traced(0.5, &[0.5]),
        (0.5, FaultSet::EMPTY)
    );
}

// ---------------------------------------------------------------------------------------------
// Where an error points, and its class
// ---------------------------------------------------------------------------------------------

#[test]
fn the_error_names_the_stage_the_member_and_the_class() {
    fn is_error<E: std::error::Error + Clone + Eq>() {}
    is_error::<ScopeError>();
    // `x` is reported before any column, as `validate_columns` checks it first.
    rejects(
        r#"(pipe X (derive expr="x + c9") (ridge))"#,
        2,
        Some(0),
        ILLEGAL_SCOPE,
        "variable x is only valid in map expressions",
        "stage 0: ILLEGAL_SCOPE: variable x is only valid in map expressions",
    );
    rejects(
        r#"(pipe X (map expr="x + c9") (ridge))"#,
        2,
        Some(0),
        COLUMN_OOB,
        "column index 9 out of bounds for width 2",
        "stage 0: COLUMN_OOB: column index 9 out of bounds for width 2",
    );
    // The first column out of range in reading order, not the largest.
    rejects(
        r#"(pipe X (derive expr="col(9) + col(12)") (ridge))"#,
        8,
        Some(0),
        COLUMN_OOB,
        "column index 9 out of bounds for width 8",
        "stage 0: COLUMN_OOB: column index 9 out of bounds for width 8",
    );
    rejects(
        r#"(pipe X (derive expr="col(12) + col(9)") (ridge))"#,
        8,
        Some(0),
        COLUMN_OOB,
        "column index 12 out of bounds for width 8",
        "stage 0: COLUMN_OOB: column index 12 out of bounds for width 8",
    );
    // A target stage counts in the index.
    rejects(
        r#"(pipe X (target kind=log1p) (standardize) (derive expr="c5") (ridge))"#,
        3,
        Some(2),
        COLUMN_OOB,
        "column index 5 out of bounds for width 3",
        "stage 2: COLUMN_OOB: column index 5 out of bounds for width 3",
    );
    // An expression reads the width that reaches its stage; an appending stage keeps its input.
    accepts(
        r#"(pipe X (derive append=true expr="c0") (map expr="c1 + 1") (ridge))"#,
        1,
    );
    rejects(
        r#"(pipe X (derive expr="c0") (map expr="c1 + 1") (ridge))"#,
        2,
        Some(1),
        COLUMN_OOB,
        "column index 1 out of bounds for width 1",
        "stage 1: COLUMN_OOB: column index 1 out of bounds for width 1",
    );
    // A closed-form program is one expression in row context, with no stage.
    rejects(
        r#"(sym raw "x")"#,
        4,
        None,
        ILLEGAL_SCOPE,
        "variable x is only valid in map expressions",
        "ILLEGAL_SCOPE: variable x is only valid in map expressions",
    );
    accepts(r#"(sym affine "c4 + row_mean")"#, 5);
    rejects(
        r#"(sym affine "c4 + row_mean")"#,
        4,
        None,
        COLUMN_OOB,
        "column index 4 out of bounds for width 4",
        "COLUMN_OOB: column index 4 out of bounds for width 4",
    );
    // An explicit portfolio names the member, and the stage inside it.
    let portfolio = |expr: &str| {
        format!(
            r#"(portfolio (member (pipe X (ridge))) (member (pipe X (row_stats) (derive expr="{expr}") (ridge))) (blend))"#
        )
    };
    accepts(&portfolio("c9"), 4);
    let Err(error) = check(&portfolio("c10"), 4) else {
        panic!("the portfolio is accepted");
    };
    assert_eq!(
        (error.member, error.stage, error.class),
        (Some(1), Some(1), COLUMN_OOB)
    );
    assert_eq!(
        error.to_string(),
        "member 1, stage 1: COLUMN_OOB: column index 10 out of bounds for width 10"
    );
    let Err(error) = check("(portfolio (member (pipe X (bogus_op))) (blend))", 4) else {
        panic!("the portfolio is accepted");
    };
    assert_eq!(
        (error.member, error.stage, error.class),
        (Some(0), None, None)
    );
    assert_eq!(error.to_string(), "member 0: unknown stage bogus_op");
}

#[test]
fn a_program_that_cannot_be_checked_fails_with_its_own_reason() {
    // Not a string: the printer leaves a numeric-looking expression unquoted, and it reads back
    // as a number.
    rejects(
        r#"(pipe X (derive expr=0.5) (ridge))"#,
        3,
        Some(0),
        None,
        "derive needs a string expression",
        "stage 0: derive needs a string expression",
    );
    rejects(
        "(pipe X (standardize) (map) (ridge))",
        3,
        Some(1),
        None,
        "map needs a string expression",
        "stage 1: map needs a string expression",
    );
    // An expression that does not parse fails with the parser's own message.
    let too_many_nodes = ["c0"; 40].join(" + ");
    for source in [
        "c0 +",
        "",
        "sin(c0",
        "bogus_fn(c0)",
        "c0 $ c1",
        "col(-1)",
        too_many_nodes.as_str(),
    ] {
        let reason = MathExpr::parse(source)
            .expect_err("the expression does not parse")
            .to_string();
        for op in ["map", "derive"] {
            let Err(error) = validate_scope(&stage_program(op, source), 3) else {
                panic!("{op} {source:?} is accepted");
            };
            assert_eq!(
                (
                    error.member,
                    error.stage,
                    error.class,
                    error.reason.as_str()
                ),
                (None, Some(1), None, reason.as_str()),
                "{op} {source:?}"
            );
        }
    }
    // A stage layout the pipeline slots refuse.
    rejects(
        "(pipe X (bogus_op) (ridge))",
        3,
        None,
        None,
        "unknown stage bogus_op",
        "unknown stage bogus_op",
    );
    rejects(
        "(pipe X (ridge) (standardize) (ridge))",
        3,
        None,
        None,
        "misplaced ridge stage at 0",
        "misplaced ridge stage at 0",
    );
    rejects(
        "(pipe X (standardize) (target kind=log1p) (ridge))",
        3,
        None,
        None,
        "misplaced target stage at 1",
        "misplaced target stage at 1",
    );
    // An input encoding stage needs a table schema.
    rejects(
        "(pipe X (target kind=log1p) (input) (ridge))",
        3,
        Some(1),
        None,
        "input encoding needs a table schema binding",
        "stage 1: input encoding needs a table schema binding",
    );
    // A stage whose width only the table-v3 path computes, even with nothing reading after it.
    rejects(
        "(pipe X (standardize) (pls_project) (ridge))",
        3,
        Some(1),
        None,
        "pls_project needs table-v3 width binding",
        "stage 1: pls_project needs table-v3 width binding",
    );
    // An input width out of bounds, for every program form.
    for width in [0, MAX_INPUT_WIDTH + 1, usize::MAX] {
        let reason = format!("input width {width} is outside 1..=16777216");
        for source in [
            "(pipe X (poly_interact) (ridge))",
            r#"(sym raw "c0")"#,
            "(portfolio (member (pipe X (poly_interact) (ridge))) (blend))",
        ] {
            rejects(source, width, None, None, &reason, &reason);
        }
    }
    // The widest accepted input keeps every width computation exact.
    accepts(
        r#"(pipe X (poly_interact) (bin_onehot bins=32) (derive expr="c511") (ridge))"#,
        MAX_INPUT_WIDTH,
    );
    accepts(
        r#"(pipe X (poly_interact append=true) (derive expr="c511") (ridge))"#,
        MAX_INPUT_WIDTH,
    );
}

/// Every op `kernel::pipeline::stage_class` classifies as a transform.
const TRANSFORMS: [&str; 43] = [
    "standardize",
    "norm_features",
    "robust_scale",
    "rank",
    "blom",
    "quantile_bins",
    "bin_onehot",
    "hinge",
    "winsor_clip",
    "winsor_moments",
    "cheb",
    "poly_interact",
    "rff_cos",
    "rff_sin",
    "rand_proj",
    "landmark_dist",
    "oblique_hinge",
    "row_stats",
    "pair_prod",
    "target_bin_encode",
    "sigmoid",
    "step",
    "norm_row",
    "polar",
    "lag",
    "roll",
    "fft_mag",
    "row_spectral",
    "row_diff_stats",
    "svd_project",
    "nystrom",
    "map",
    "derive",
    "symlog",
    "select_corr",
    "impute",
    "pair_ratio",
    "spline",
    "maxabs_scale",
    "power_transform",
    "piecewise_linear",
    "corr_screen",
    "pls_project",
];

#[test]
fn every_transform_at_every_edge_width_returns_an_error_or_ok_without_panicking() {
    let edges = [
        0,
        1,
        2,
        511,
        512,
        513,
        600,
        MAX_INPUT_WIDTH - 1,
        MAX_INPUT_WIDTH,
        MAX_INPUT_WIDTH + 1,
        usize::MAX,
    ];
    let mut outcomes = [0usize; 2];
    for op in TRANSFORMS {
        assert_eq!(stage_class(op), Some(StageClass::Transform), "{op}");
        let expr = if matches!(op, "map" | "derive") {
            r#" expr="c0""#
        } else {
            ""
        };
        for append in ["", " append=true"] {
            let first = format!("({op}{expr}{append})");
            for source in [
                format!("(pipe X {first} (ridge))"),
                format!("(pipe X {first} {first} (derive expr=\"c0\") (ridge))"),
            ] {
                let genome = program(&source);
                for width in edges {
                    outcomes[usize::from(validate_scope(&genome, width).is_ok())] += 1;
                }
            }
        }
    }
    assert!(outcomes[0] > 0 && outcomes[1] > 0, "{outcomes:?}");
}

// ---------------------------------------------------------------------------------------------
// Agreement with the legacy width check
// ---------------------------------------------------------------------------------------------

/// How the static check and the legacy check compare.
#[derive(Default)]
struct Agreement {
    comparisons: usize,
    both_accept: usize,
    both_reject: usize,
    scope_faults: usize,
    /// A column read after `nystrom`, compared with more than one training row: rejected here,
    /// and accepted by the legacy check or rejected against a wider width. The only difference
    /// for inputs up to `MAX_WIDTH`; with one row the two agree exactly.
    nystrom: usize,
}

impl Agreement {
    #[track_caller]
    fn compare(&mut self, name: &str, pipe: &DslExpr, rows: usize, width: usize) {
        let slots =
            PipelineSlots::from_pipeline(pipe).unwrap_or_else(|error| panic!("{name}: {error}"));
        let scope = validate_scope(&GenomeIr::Pipeline(pipe.clone()), width);
        let verdict = legacy(&slots, rows, width);
        self.comparisons += 1;
        match (&scope, &verdict) {
            (Ok(()), Ok(())) => self.both_accept += 1,
            (Err(error), Err(reason)) if &error.reason == reason => {
                let at = format!("{name} at width {width}, {rows} rows: {error}");
                assert_eq!(error.member, None, "{at}");
                assert!(
                    error.stage.is_some_and(|stage| stage < pipe.stages.len()),
                    "{at}"
                );
                // The class follows the reason: `x` is reported before any column.
                match error.class {
                    None => assert!(
                        !reason.starts_with("variable x ") && !reason.starts_with("column index "),
                        "{at}"
                    ),
                    Some(FaultClass::IllegalScope) => {
                        assert!(reason.starts_with("variable x "), "{at}")
                    }
                    Some(FaultClass::ColumnOob) => {
                        assert!(reason.starts_with("column index "), "{at}")
                    }
                    Some(other) => panic!("{at}: unexpected class {other}"),
                }
                self.both_reject += 1;
                self.scope_faults += usize::from(error.class.is_some());
            }
            // The legacy check accepts, or reports the column against a wider width.
            (Err(error), _)
                if rows > 1
                    && error.class == COLUMN_OOB
                    && error.stage.is_some_and(|stage| {
                        pipe.stages[..stage].iter().any(|s| s.op == "nystrom")
                    }) =>
            {
                // With one training row, the count the static check assumes, the legacy check
                // reports the same fault.
                assert_eq!(
                    legacy(&slots, 1, width),
                    Err(error.reason.clone()),
                    "{name} at width {width}"
                );
                self.nystrom += 1;
            }
            _ => panic!(
                "{name} at width {width}, {rows} rows: validate_scope {scope:?}, \
                 validate_legacy_widths {verdict:?}"
            ),
        }
    }
}

impl fmt::Display for Agreement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "comparisons={} both_accept={} both_reject={} scope_faults={} nystrom_differences={}",
            self.comparisons, self.both_accept, self.both_reject, self.scope_faults, self.nystrom
        )
    }
}

/// PR 01's compatibility corpus: every program's name and canonical text.
fn corpus() -> Vec<(String, String)> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/compat_freeze/emitted_bytes.v1.json"
    );
    let text = std::fs::read_to_string(path).unwrap_or_else(|error| panic!("{path}: {error}"));
    let file: serde_json::Value =
        serde_json::from_str(&text).expect("emitted_bytes.v1.json parses");
    let programs: Vec<(String, String)> = file["programs"]
        .as_array()
        .expect("programs")
        .iter()
        .map(|entry| {
            (
                entry["name"].as_str().expect("name").to_string(),
                entry["canonical"].as_str().expect("canonical").to_string(),
            )
        })
        .collect();
    assert!(!programs.is_empty(), "the corpus has programs");
    programs
}

/// Input widths for the corpus: every small width, and the legacy check's edge.
const CORPUS_WIDTHS: [usize; 14] = [1, 2, 3, 4, 5, 6, 7, 8, 10, 16, 20, 64, 511, 512];
const CORPUS_ROWS: [usize; 3] = [1, 32, 4096];

#[test]
fn the_check_agrees_with_the_legacy_check_on_the_compat_corpus() {
    let corpus = corpus();
    let mut agreement = Agreement::default();
    let (mut pipelines, mut slots_accept, mut closed_form, mut portfolio_members) = (0, 0, 0, 0);
    let mut rejected_at_every_width = Vec::new();
    for (name, canonical) in &corpus {
        let genome = program(canonical);
        match &genome {
            GenomeIr::Pipeline(pipe) => {
                pipelines += 1;
                // Only a layout the pipeline slots accept reaches the legacy check.
                if PipelineSlots::from_pipeline(pipe).is_err() {
                    continue;
                }
                slots_accept += 1;
                for rows in CORPUS_ROWS {
                    for width in CORPUS_WIDTHS {
                        agreement.compare(name, pipe, rows, width);
                    }
                }
                if CORPUS_WIDTHS
                    .iter()
                    .all(|width| validate_scope(&genome, *width).is_err())
                {
                    rejected_at_every_width.push(name.as_str());
                }
            }
            // A closed-form expression is read per row: the walk is its oracle.
            GenomeIr::Symbolic { expr, .. } => {
                closed_form += 1;
                for width in CORPUS_WIDTHS {
                    let actual = validate_scope(&genome, width).err().map(|error| {
                        assert_eq!((error.member, error.stage), (None, None), "{name}");
                        (error.class, error.reason)
                    });
                    let expected = expected_fault(expr, width, false)
                        .map(|(class, reason)| (Some(class), reason));
                    assert_eq!(actual, expected, "{name} at width {width}");
                }
            }
            // A member may see one column only (the portfolio drops the columns that are constant
            // in training), and is checked as its pipeline is at one column.
            GenomeIr::Portfolio(portfolio) => {
                for member in &portfolio.members {
                    portfolio_members += 1;
                    for rows in CORPUS_ROWS {
                        agreement.compare(name, member, rows, 1);
                    }
                }
            }
        }
    }
    eprintln!(
        "AGREEMENT corpus programs={} pipelines={pipelines} slots_accept={slots_accept} \
         closed_form={closed_form} portfolio_members={portfolio_members} \
         widths={CORPUS_WIDTHS:?} rows={CORPUS_ROWS:?} {agreement} \
         rejected_at_every_width={rejected_at_every_width:?}",
        corpus.len()
    );
    assert!(slots_accept > 0 && agreement.both_accept > 0 && agreement.scope_faults > 0);
    // The corpus never reads a column after `nystrom`, so the checks agree everywhere.
    assert_eq!(agreement.nystrom, 0);
}

/// A stage for every width rule of the pipeline slots, each followed by an expression stage in
/// the grid. The corpus never reads a column after a stage that changes the width.
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

#[test]
fn the_check_agrees_with_the_legacy_check_after_every_width_rule() {
    let mut expressions = vec![("derive", "x".to_string())];
    for column in 0..=16 {
        expressions.push(("derive", format!("c{column}")));
        expressions.push(("map", format!("x + c{column}")));
    }
    let mut agreement = Agreement::default();
    for template in WIDTH_STAGES {
        for append in [false, true] {
            let first = if append {
                format!("{} append=true)", &template[..template.len() - 1])
            } else {
                template.to_string()
            };
            for (op, expression) in &expressions {
                let source = format!(r#"(pipe X {first} ({op} expr="{expression}") (ridge))"#);
                let GenomeIr::Pipeline(pipe) = program(&source) else {
                    unreachable!("a pipe is a pipeline");
                };
                for rows in [1, 4096] {
                    for width in [1, 2, 3, 4, 5, 8, 16] {
                        agreement.compare(&source, &pipe, rows, width);
                    }
                }
            }
        }
    }
    eprintln!(
        "AGREEMENT grid stages={} expressions={} widths=[1, 2, 3, 4, 5, 8, 16] rows=[1, 4096] \
         {agreement}",
        WIDTH_STAGES.len() * 2,
        expressions.len()
    );
    assert!(agreement.both_accept > 0 && agreement.scope_faults > 0);
    assert!(agreement.nystrom > 0, "the nystrom difference is exercised");
}

#[test]
fn the_check_agrees_with_the_legacy_check_on_random_pipelines() {
    let mut agreement = Agreement::default();
    let (mut programs, mut with_expressions) = (0, 0);
    let mut not_generated = Vec::new();
    for seed in 0..300u64 {
        let feature_count = [1, 2, 3, 5, 8][(seed % 5) as usize];
        // The generator refuses a few seeds of its own accord.
        let random = match feat_math::full_rust::random_pipeline(seed, feature_count) {
            Ok(random) => random,
            Err(error) => {
                not_generated.push(format!("seed {seed}: {error}"));
                continue;
            }
        };
        let GenomeIr::Pipeline(pipe) = program(&random.canonical) else {
            panic!("seed {seed}: {} is not a pipeline", random.canonical);
        };
        programs += 1;
        with_expressions += usize::from(
            pipe.stages
                .iter()
                .any(|s| matches!(s.op.as_str(), "map" | "derive")),
        );
        let name = format!("seed {seed}: {}", random.canonical);
        for rows in [1, 4096] {
            for width in [1, 2, 3, 5, 8, 16] {
                agreement.compare(&name, &pipe, rows, width);
            }
        }
    }
    eprintln!(
        "AGREEMENT random seeds=0..300 programs={programs} with_expressions={with_expressions} \
         widths=[1, 2, 3, 5, 8, 16] rows=[1, 4096] {agreement} not_generated={not_generated:?}"
    );
    assert!(programs >= 250 && with_expressions > 0 && agreement.scope_faults > 0);
}

#[test]
fn the_two_intended_differences_from_the_legacy_check() {
    // 1. `nystrom` keeps min(k, rows) columns. The static check has no row count and takes one
    //    row, so a column after it must exist even then.
    let nystrom = r#"(pipe X (nystrom k=4) (derive expr="c2") (ridge))"#;
    rejects(
        nystrom,
        8,
        Some(1),
        COLUMN_OOB,
        "column index 2 out of bounds for width 1",
        "stage 1: COLUMN_OOB: column index 2 out of bounds for width 1",
    );
    let nystrom_slots = slots(nystrom);
    assert_eq!(
        legacy(&nystrom_slots, 1, 8),
        Err("column index 2 out of bounds for width 1".into())
    );
    assert_eq!(
        legacy(&nystrom_slots, 2, 8),
        Err("column index 2 out of bounds for width 2".into())
    );
    assert_eq!(legacy(&nystrom_slots, 3, 8), Ok(()));
    assert_eq!(legacy(&nystrom_slots, 4096, 8), Ok(()));
    accepts(r#"(pipe X (nystrom k=4) (derive expr="c0") (ridge))"#, 8);
    // The runtime width follows the training rows.
    for (rows, width) in [(2, 2), (3, 3), (8, 4)] {
        let (x, y) = data(rows, 8);
        let op = fit_op(&stage("(nystrom k=4)"), &x, &y).expect("nystrom fits");
        assert!(
            apply_op(&op, &x).iter().all(|row| row.len() == width),
            "{rows} rows"
        );
    }

    // 2. Inputs wider than MAX_WIDTH. The runtime truncates a row to MAX_WIDTH only after a
    //    stage, so the first stage reads every input column; the legacy check refuses them.
    let wide = r#"(pipe X (derive expr="c599") (ridge))"#;
    accepts(wide, 600);
    rejects(
        wide,
        599,
        Some(0),
        COLUMN_OOB,
        "column index 599 out of bounds for width 599",
        "stage 0: COLUMN_OOB: column index 599 out of bounds for width 599",
    );
    assert_eq!(
        legacy(&slots(wide), 32, 600),
        Err("legacy pipeline needs nonempty bounded matrix dimensions".into())
    );
    rejects(
        r#"(pipe X (standardize) (derive expr="c599") (ridge))"#,
        600,
        Some(1),
        COLUMN_OOB,
        "column index 599 out of bounds for width 512",
        "stage 1: COLUMN_OOB: column index 599 out of bounds for width 512",
    );
    accepts(
        r#"(pipe X (standardize) (derive expr="c511") (ridge))"#,
        600,
    );
    let (x, y) = data(8, 600);
    let derived = fit_op(&stage(r#"(derive expr="c599")"#), &x, &y).expect("derive fits");
    assert_eq!(
        apply_op(&derived, &x),
        x.iter().map(|row| vec![row[599]]).collect::<Vec<_>>()
    );
    let scaled = fit_op(&stage("(standardize)"), &x, &y).expect("standardize fits");
    assert!(apply_op(&scaled, &x).iter().all(|row| row.len() == 512));
}

// ---------------------------------------------------------------------------------------------
// Portfolio members
// ---------------------------------------------------------------------------------------------

/// Sixty rows of four columns, the first one constant when `constant_first`, and a target of three
/// times column 3 plus a tenth of column 1.
fn portfolio_data(constant_first: bool) -> (Vec<Vec<f64>>, Vec<f64>) {
    let x: Vec<Vec<f64>> = (0..60u32)
        .map(|i| {
            let t = f64::from(i);
            let first = if constant_first {
                1.0
            } else {
                (t * 0.23).sin()
            };
            vec![first, (t * 0.37).sin(), (t * 0.11).cos(), t / 10.0]
        })
        .collect();
    let y = x.iter().map(|row| 3.0 * row[3] + 0.1 * row[1]).collect();
    (x, y)
}

/// The share of the target's variance that `predictions` explain.
fn r2(y: &[f64], predictions: &[f64]) -> f64 {
    let mean = y.iter().sum::<f64>() / y.len() as f64;
    let total: f64 = y.iter().map(|v| (v - mean) * (v - mean)).sum();
    let residual: f64 = y
        .iter()
        .zip(predictions)
        .map(|(v, p)| (v - p) * (v - p))
        .sum();
    1.0 - residual / total
}

#[test]
fn a_portfolio_member_is_checked_against_the_columns_it_can_see() {
    let member = |expr: &str| {
        format!(r#"(portfolio (member (pipe X (derive expr="{expr}") (ridge))) (blend))"#)
    };
    // The portfolio drops the columns that are constant in training before its members see the
    // rows, so a member may see a single column at any input width: until one of its stages sets
    // a width, column 0 is the only column it can read.
    for width in [1, 4, 600] {
        accepts(&member("c0"), width);
        let Err(error) = check(&member("c3"), width) else {
            panic!("a member reading c3 is accepted at width {width}");
        };
        assert_eq!(
            (
                error.member,
                error.stage,
                error.class,
                error.reason.as_str()
            ),
            (
                Some(0),
                Some(0),
                COLUMN_OOB,
                "column index 3 out of bounds for width 1"
            ),
            "width {width}"
        );
        assert_eq!(
            error.to_string(),
            "member 0, stage 0: COLUMN_OOB: column index 3 out of bounds for width 1"
        );
    }
    // The same expression in a pipeline reads the input columns themselves.
    accepts(r#"(pipe X (derive expr="c3") (ridge))"#, 4);
    // A stage of fixed width, or an appended column, sets what the later stages may read.
    accepts(
        r#"(portfolio (member (pipe X (row_stats) (derive expr="c9") (ridge))) (blend))"#,
        4,
    );
    accepts(
        r#"(portfolio (member (pipe X (derive append=true expr="c0") (map expr="x + c1") (ridge))) (blend))"#,
        4,
    );
    // The error names the member that fails.
    let two = r#"(portfolio (member (pipe X (derive expr="c0") (ridge))) (member (pipe X (derive expr="c1") (ridge))) (blend))"#;
    let Err(error) = check(two, 4) else {
        panic!("{two} is accepted");
    };
    assert_eq!(
        error.to_string(),
        "member 1, stage 0: COLUMN_OOB: column index 1 out of bounds for width 1"
    );

    // Every member is checked as its pipeline is at one column, whatever the input width.
    let mut expressions = vec![("derive", "x".to_string())];
    for column in 0..=16 {
        expressions.push(("derive", format!("c{column}")));
        expressions.push(("map", format!("x + c{column}")));
    }
    let widths = [1, 2, 3, 4, 5, 8, 16, 600, MAX_INPUT_WIDTH];
    let (mut comparisons, mut rejected) = (0, 0);
    for template in WIDTH_STAGES {
        for append in [false, true] {
            let first = if append {
                format!("{} append=true)", &template[..template.len() - 1])
            } else {
                template.to_string()
            };
            for (op, expression) in &expressions {
                let pipe = format!(r#"(pipe X {first} ({op} expr="{expression}") (ridge))"#);
                let alone = check(&pipe, 1)
                    .map_err(|error| (Some(0), error.stage, error.class, error.reason));
                let portfolio = program(&format!("(portfolio (member {pipe}) (blend))"));
                for width in widths {
                    let within = validate_scope(&portfolio, width)
                        .map_err(|error| (error.member, error.stage, error.class, error.reason));
                    assert_eq!(within, alone, "{pipe} as a member at width {width}");
                    comparisons += 1;
                    rejected += usize::from(within.is_err());
                }
            }
        }
    }
    eprintln!(
        "AGREEMENT members stages={} expressions={} widths={widths:?} comparisons={comparisons} \
         rejected={rejected}",
        WIDTH_STAGES.len() * 2,
        expressions.len()
    );
    assert!(rejected > 0 && rejected < comparisons);

    // At runtime, with input column 0 constant in training, the portfolio keeps columns 1 to 3: a
    // member's rows are three wide, so its `c3` does not exist and its `c2` is input column 3.
    let fit = |source: &str, (x, y): &(Vec<Vec<f64>>, Vec<f64>)| {
        let model = feat_math::fit(&program(source), x, y)
            .unwrap_or_else(|error| panic!("{source}: {error}"));
        let predictions = model.predict(x);
        (model, predictions)
    };
    let constant_first = portfolio_data(true);
    let (model, predictions) = fit(&member("c3"), &constant_first);
    let feat_math::Model::Portfolio(fitted) = &model else {
        panic!("a portfolio program fits a portfolio model");
    };
    assert_eq!(fitted.keep, [1, 2, 3]);
    let row: Vec<f64> = fitted
        .keep
        .iter()
        .map(|&column| constant_first.0[5][column])
        .collect();
    assert_eq!(
        MathExpr::parse("c3")
            .expect("c3 parses")
            .eval_row_traced(&row),
        (0.0, FaultSet::COLUMN_OOB)
    );
    // The member reads nothing, so the portfolio predicts one value everywhere.
    assert!(
        predictions
            .iter()
            .all(|value| value.to_bits() == predictions[0].to_bits()),
        "{predictions:?}"
    );
    let y = &constant_first.1;
    assert!(r2(y, &fit(&member("c2"), &constant_first).1) > 0.99);
    let pipeline = r#"(pipe X (derive expr="c3") (ridge))"#;
    assert!(r2(y, &fit(pipeline, &constant_first).1) > 0.99);
    // With no constant column, the member's `c3` is input column 3.
    let varying = portfolio_data(false);
    assert!(r2(&varying.1, &fit(&member("c3"), &varying).1) > 0.99);
}

// ---------------------------------------------------------------------------------------------
// The search profile
// ---------------------------------------------------------------------------------------------

#[test]
fn the_search_profile_is_fifteen_nodes_and_depth_four() {
    for (columns, elementwise) in [(0, false), (3, true), (512, false)] {
        let scope = ExpressionScope::search_profile(columns, elementwise);
        assert_eq!(
            (
                scope.columns,
                scope.elementwise,
                scope.max_nodes,
                scope.max_depth
            ),
            (columns, elementwise, 15, 4)
        );
    }
    let scope = ExpressionScope::search_profile(3, true);
    let parse = |source: &str| MathExpr::parse(source).expect("the expression parses");
    // Depth 4 fits and depth 5 does not.
    scope.validate(&parse("sin(sin(sin(c0)))")).unwrap();
    assert!(scope.validate(&parse("sin(sin(sin(sin(c0))))")).is_err());
    // Fifteen nodes fit and sixteen do not, both at depth 4.
    let fifteen = parse("((c0 + c1) + (c2 + c0)) + ((c1 + c2) + (x + c0))");
    let found = shape(fifteen.ast());
    assert_eq!((found.nodes, found.depth), (15, 4));
    scope.validate(&fifteen).unwrap();
    let sixteen = parse("clip(c0 + c1, (c2 + c0) + c1, (c1 + c2) + (x + c0))");
    let found = shape(sixteen.ast());
    assert_eq!((found.nodes, found.depth), (16, 4));
    assert!(scope.validate(&sixteen).is_err());
    // The kernel pipeline operators' caps, 64 nodes and depth 16, still admit both.
    let kernel = ExpressionScope {
        columns: 3,
        elementwise: true,
        max_nodes: 64,
        max_depth: 16,
    };
    kernel.validate(&parse("sin(sin(sin(sin(c0))))")).unwrap();
    kernel.validate(&sixteen).unwrap();
}

/// Chained edits: each seed edits the parent with `mutate_v1` in a scope that changes every 50
/// seeds (map or row context) and every 250 seeds (0, 1, 2, 3 or 8 columns); a parent outside the
/// new scope restarts from `0.1`.
#[test]
fn five_thousand_chained_edits_under_the_search_profile_stay_in_scope_and_within_the_caps() {
    let restart = MathExpr::parse("0.1").expect("the restart parses");
    let mut parent = restart.clone();
    let (mut changed, mut restarts, mut max_nodes, mut max_depth) = (0, 0, 0, 0);
    let mut distinct = BTreeSet::new();
    for seed in 0..5_000u64 {
        let columns = [0, 1, 2, 3, 8][(seed / 250 % 5) as usize];
        let elementwise = seed / 50 % 2 == 0;
        let scope = ExpressionScope::search_profile(columns, elementwise);
        if scope.validate(&parent).is_err() {
            parent = restart.clone();
            restarts += 1;
        }
        let edit = parent
            .mutate_v1(seed, scope)
            .unwrap_or_else(|error| panic!("seed {seed}: {error}"));
        let child = &edit.expression;
        if let Err(error) = scope.validate(child) {
            panic!("seed {seed}: {error}");
        }
        // Counted again through the tree, independently of the scope's own check.
        let found = shape(child.ast());
        assert!(
            found.nodes <= 15 && found.depth <= 4,
            "seed {seed}: {} nodes, depth {}",
            found.nodes,
            found.depth
        );
        assert!(
            elementwise || !found.reads_x,
            "seed {seed}: x outside a map"
        );
        assert!(
            found.columns.iter().all(|column| *column < columns),
            "seed {seed}: columns {:?} of {columns}",
            found.columns
        );
        // The static check admits the same tree in a stage at that width.
        if columns > 0 {
            let op = if elementwise { "map" } else { "derive" };
            let source = child
                .to_source()
                .unwrap_or_else(|error| panic!("seed {seed}: {error}"));
            if let Err(error) = validate_scope(&stage_program(op, &source), columns) {
                panic!("seed {seed}: {op} {source}: {error}");
            }
        }
        max_nodes = max_nodes.max(found.nodes);
        max_depth = max_depth.max(found.depth);
        changed += usize::from(edit.changed);
        distinct.insert(format!("{:?}", child.ast()));
        parent = edit.expression;
    }
    eprintln!(
        "PROPERTY edits=5000 changed={changed} distinct={} restarts={restarts} \
         max_nodes={max_nodes} max_depth={max_depth}",
        distinct.len()
    );
    // The caps are reached, so they bound the edits rather than hold vacuously.
    assert_eq!((max_nodes, max_depth), (15, 4));
    assert!(changed > 2_500, "changed {changed} of 5000");
}

// ---------------------------------------------------------------------------------------------
// Determinism and shipped text
// ---------------------------------------------------------------------------------------------

/// Every hand-written case and every corpus program, at every sweep width.
fn sweep() -> Vec<(String, usize)> {
    let mut sources: Vec<String> = HAND_CASES.iter().map(|s| s.to_string()).collect();
    sources.extend(corpus().into_iter().map(|(_, canonical)| canonical));
    sources
        .into_iter()
        .flat_map(|source| SWEEP_WIDTHS.map(|width| (source.clone(), width)))
        .collect()
}

#[test]
fn the_same_input_gives_the_same_result_and_the_same_bytes() {
    let mut digest = Sha256::new();
    let cases = sweep();
    for (source, width) in &cases {
        let genome = program(source);
        let first = validate_scope(&genome, *width);
        for _ in 0..3 {
            assert_eq!(
                validate_scope(&genome, *width),
                first,
                "{source} at {width}"
            );
        }
        // The same program, read back from its canonical text.
        let again = program(&genome.canonical());
        assert_eq!(validate_scope(&again, *width), first, "{source} at {width}");
        let text = match &first {
            Ok(()) => "ok".to_string(),
            Err(error) => error.to_string(),
        };
        digest.update(format!("{source}\t{width}\t{text}\n"));
    }
    let hex: String = digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    eprintln!("DETERMINISM cases={} sha256={hex}", cases.len());
    // Every displayed result, reason bytes included, is pinned across runs and hosts: a change to
    // any of them fails here, and is made on purpose, with this digest and the WP-1.6 receipt.
    assert_eq!(
        (cases.len(), hex.as_str()),
        (
            780,
            "4579e907329496ced7ac5bfe8cabb519096eaa08b40a6eaba116c65a7317c63f"
        ),
        "the displayed results changed"
    );
}

#[test]
fn every_message_passes_the_shipped_text_lint() {
    let mut messages = BTreeSet::new();
    for (source, width) in sweep() {
        if let Err(error) = check(&source, width) {
            messages.insert(error.to_string());
            messages.insert(error.reason);
        }
    }
    assert!(messages.len() > 20, "{} messages", messages.len());
    for message in &messages {
        let hits = feat_math::transpile::lint_shipped_text(message);
        assert!(hits.is_empty(), "{message}: {hits:?}");
    }
}
