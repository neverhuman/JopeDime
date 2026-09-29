//! The fitted-width contract (kernel2 WP-1.7): `kernel::width` and `Model::try_predict`, checked
//! against the unchecked `Model::predict`.
//!
//! - Every `FittedOp` variant, with and without `append`: its width class, and `output_width`
//!   against the rows `apply_op` returns, the 512-column cap and the `[0.0]` padding included.
//! - The workplan cases and their siblings: an input that `predict` panics on or misreads is
//!   refused by `try_predict` before any stage runs, and at the fitted width `try_predict` returns
//!   `predict`'s bits. Where `predict` panics, the test shows that it still does.
//! - Heads, portfolios, symbolic models, ragged rows and empty inputs.
//! - Fitted state that disagrees with itself, built by hand because no fit produces it, and
//!   output widths that overflow: refused, never a panic.
//! - Over PR 01's compatibility corpus (`tests/fixtures/compat_freeze`), every program that fits:
//!   `try_predict` equals `predict` bit for bit on the held-out rows, and both equal the recorded
//!   predictions; the width walk matches a stage-by-stage replay; at other widths `try_predict`
//!   never panics, and whenever it returns a value that value is `predict`'s.
//! - The schema-bound (advanced-v3) boundary: a legacy stage, `corr_screen` or `pls_project`
//!   whose fitted state does not fit the feature block makes `predict_table` return an error
//!   instead of panicking, and a table with no rows keeps its result.

use feat_math::dsl::{parse_dsl, Stage};
use feat_math::heads::{fit_head, FittedHead};
use feat_math::kernel::width::{
    check_fitted_state, check_input_width, output_width, required_head_width, required_input_width,
    MismatchKind, WidthMismatch, WidthReq,
};
use feat_math::kernel::Error;
use feat_math::ops::{apply_op, fit_op, FittedOp};
use feat_math::{
    fit, fit_table, sanitize, BoundModel, ColumnSpec, DataSchema, GenomeIr, Model, Objective,
    Pcg32, RawTable, RowSemantics, Rows, TaskSpec,
};
use std::collections::BTreeSet;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::Path;

// ---------------------------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------------------------

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|value| value.to_bits()).collect()
}

fn panics<T>(call: impl FnOnce() -> T) -> bool {
    catch_unwind(AssertUnwindSafe(call)).is_err()
}

/// Rows of values in [-2, 2) with an exact 0.0 wherever `(row + column) % 5 == 0`, so every
/// column of five rows or more has a cell that `impute` treats as missing.
fn data(rows: usize, width: usize, seed: u64) -> Rows {
    let mut rng = Pcg32::new(seed, 29);
    (0..rows)
        .map(|i| {
            (0..width)
                .map(|j| {
                    let value = rng.next_f64() * 4.0 - 2.0;
                    if (i + j) % 5 == 0 {
                        0.0
                    } else {
                        value
                    }
                })
                .collect()
        })
        .collect()
}

/// A target that reads every column.
fn target(x: &Rows) -> Vec<f64> {
    x.iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .map(|(j, value)| (0.5 + 0.25 * j as f64) * value)
                .sum::<f64>()
                + 0.3 * (2.0 * row.first().copied().unwrap_or(0.0)).sin()
        })
        .collect()
}

fn program(source: &str) -> GenomeIr {
    GenomeIr::parse(source).unwrap_or_else(|error| panic!("{source}: {error}"))
}

fn stage(source: &str) -> Stage {
    parse_dsl(&format!("(pipe X {source})"))
        .unwrap_or_else(|error| panic!("{source}: {error}"))
        .stages
        .remove(0)
}

fn with_append(source: &str) -> String {
    let body = source.strip_suffix(')').expect("a stage ends with ')'");
    format!("{body} append=true)")
}

fn fit_program(source: &str, x: &Rows) -> Model {
    fit(&program(source), x, &target(x)).unwrap_or_else(|error| panic!("{source}: {error}"))
}

fn fitted_ops(model: &Model) -> &[FittedOp] {
    match model {
        Model::Pipeline { ops, .. } => ops,
        _ => panic!("a pipeline program fits a pipeline model"),
    }
}

/// The variant name of a fitted stage, from its `Debug` form.
fn variant(op: &FittedOp) -> String {
    format!("{op:?}")
        .split([' ', '{'])
        .next()
        .expect("a variant name")
        .to_string()
}

/// A model of one fitted stage ahead of a ridge head fitted on `head_width` columns.
fn one_stage_model(op: FittedOp, head_width: usize) -> Model {
    let x = data(24, head_width, 5);
    let head = fit_head(&Stage::new("ridge"), &x, &target(&x)).expect("ridge fits");
    Model::Pipeline {
        target: None,
        ops: vec![op],
        head: Box::new(head),
    }
}

fn claimed_width(op: &FittedOp) -> usize {
    match required_input_width(op) {
        WidthReq::Exactly(width) | WidthReq::AtLeast(width) => width,
        WidthReq::Any => 3,
    }
}

// ---------------------------------------------------------------------------------------------
// Every stage: its width class, and `output_width` against `apply_op`
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
enum Class {
    /// `Exactly(fitted width)`.
    Exact,
    /// `AtLeast(largest fitted column index + 1)`.
    Indices,
    /// `AtLeast(1)`.
    NonEmpty,
    Any,
}

/// One program per `FittedOp` variant (36), fitted on 8 columns.
const VARIANTS: &[(&str, Class)] = &[
    ("(standardize)", Class::Exact),
    ("(robust_scale)", Class::Exact),
    ("(rank)", Class::Exact),
    ("(blom)", Class::Exact),
    ("(quantile_bins bins=5)", Class::Exact),
    ("(bin_onehot bins=4)", Class::Exact),
    ("(hinge knots=3)", Class::Exact),
    ("(winsor_clip q=0.1)", Class::Exact),
    ("(cheb degree=3)", Class::Any),
    ("(poly_interact)", Class::Indices),
    ("(rff_cos n=5 gamma=0.7 seed=11)", Class::Exact),
    ("(rff_sin n=5 gamma=0.7 seed=11)", Class::Exact),
    ("(rand_proj k=5 seed=11)", Class::Exact),
    ("(landmark_dist k=5 gamma=0.7 seed=11)", Class::Exact),
    ("(oblique_hinge n=5 seed=11)", Class::Exact),
    ("(row_stats)", Class::Any),
    ("(pair_prod topk=4)", Class::Indices),
    ("(target_bin_encode bins=5 prior=3)", Class::Exact),
    ("(sigmoid)", Class::Any),
    ("(step t=0.2)", Class::Any),
    ("(norm_row)", Class::Any),
    ("(polar)", Class::Any),
    ("(lag k=2)", Class::Any),
    ("(roll k=3)", Class::NonEmpty),
    ("(fft_mag k=2)", Class::Any),
    ("(row_spectral)", Class::Any),
    ("(row_diff_stats)", Class::Any),
    ("(svd_project k=3 iters=8 seed=11)", Class::Exact),
    ("(nystrom k=5 gamma=0.7 seed=11)", Class::Exact),
    ("(map expr=\"asinh(x)+row_mean\")", Class::Any),
    ("(derive expr=\"row_l2+col(0)\")", Class::Any),
    ("(symlog c=0.7)", Class::Any),
    ("(select_corr k=3)", Class::Indices),
    ("(impute strategy=median add_indicator=true)", Class::Exact),
    ("(pair_ratio topk=5)", Class::Indices),
    ("(spline knots=4)", Class::Exact),
];

/// Widths each fitted stage is applied at, where its requirement admits them.
const WIDTHS: &[usize] = &[0, 1, 2, 3, 5, 7, 8, 9, 16, 600];

fn largest_index(op: &FittedOp) -> Option<usize> {
    match op {
        FittedOp::SelectCorr { indices, .. } => indices.iter().copied().max(),
        FittedOp::PolyInteract { pairs, .. }
        | FittedOp::PairProd { pairs, .. }
        | FittedOp::PairRatio { pairs, .. } => pairs.iter().map(|&(a, b)| a.max(b)).max(),
        _ => None,
    }
}

#[test]
fn every_stage_has_a_width_class_and_its_output_width_matches_apply_op() {
    let x = data(40, 8, 1);
    let y = target(&x);
    let mut variants = BTreeSet::new();
    let mut checked = 0;
    for &(source, class) in VARIANTS {
        for append in [false, true] {
            let source = if append {
                with_append(source)
            } else {
                source.to_string()
            };
            let op = fit_op(&stage(&source), &x, &y).unwrap_or_else(|e| panic!("{source}: {e}"));
            variants.insert(variant(&op));
            assert_eq!(check_fitted_state(&op), Ok(()), "{source}: fitted state");
            let required = required_input_width(&op);
            let expected = match class {
                Class::Exact => WidthReq::Exactly(8),
                Class::Indices => {
                    let largest = largest_index(&op).expect("fitted column indices");
                    WidthReq::AtLeast(largest + 1)
                }
                Class::NonEmpty => WidthReq::AtLeast(1),
                Class::Any => WidthReq::Any,
            };
            assert_eq!(required, expected, "{source}: width class");
            for (index, &width) in WIDTHS.iter().enumerate() {
                if !required.admits(width) {
                    assert!(
                        matches!(output_width(&op, width), Err(Error::Shape(_))),
                        "{source}: width {width} is refused"
                    );
                    continue;
                }
                let rows = data(3, width, 100 + index as u64);
                let produced = apply_op(&op, &rows);
                let width_out = output_width(&op, width)
                    .unwrap_or_else(|e| panic!("{source} at width {width}: {e}"));
                assert!(
                    produced.iter().all(|row| row.len() == width_out),
                    "{source} at width {width}: apply_op wrote {:?} columns, output_width says {width_out}",
                    produced.iter().map(Vec::len).collect::<Vec<_>>()
                );
                checked += 1;
            }
        }
    }
    assert_eq!(variants.len(), 36, "every FittedOp variant: {variants:?}");
    assert!(checked > 300, "{checked} (stage, append, width) cases");
    println!(
        "STAGES {} variants, {checked} (stage, append, width) cases",
        variants.len()
    );
}

#[test]
fn output_width_caps_at_max_width_and_pads_empty_rows() {
    // (stage, fitted width, applied width, expected output width)
    let cases: &[(&str, usize, usize, usize)] = &[
        // 100 + 800 columns, capped.
        ("(cheb degree=8 append=true)", 8, 100, 512),
        // 100 columns of 8 one-hot bins, capped.
        ("(bin_onehot bins=8)", 100, 100, 512),
        // 40 columns of 16 knots, capped.
        ("(hinge knots=16)", 40, 40, 512),
        // One column leaves no pair: the empty row is padded to [0.0].
        ("(pair_ratio topk=4)", 1, 1, 1),
        ("(pair_ratio topk=4)", 1, 7, 1),
        // Appending an empty output keeps the input.
        ("(pair_ratio topk=4 append=true)", 1, 7, 7),
        // An empty input row gives an empty output row, padded.
        ("(sigmoid)", 3, 0, 1),
        ("(cheb degree=4 append=true)", 3, 0, 1),
        ("(map expr=\"x+1\")", 3, 0, 1),
        // Fixed widths do not depend on the input width.
        ("(row_stats)", 3, 0, 10),
        ("(row_spectral append=true)", 3, 1, 6),
        ("(derive expr=\"col(0)\")", 3, 0, 1),
    ];
    for &(source, fitted, applied, expected) in cases {
        let x = data(40, fitted, 2);
        let op =
            fit_op(&stage(source), &x, &target(&x)).unwrap_or_else(|e| panic!("{source}: {e}"));
        assert_eq!(
            output_width(&op, applied),
            Ok(expected),
            "{source} at {applied}"
        );
        let produced = apply_op(&op, &data(3, applied, 3));
        assert!(
            produced.iter().all(|row| row.len() == expected),
            "{source} at {applied}"
        );
    }

    // A constant column gives its raw value alone; the others their raw value and k - 2 terms.
    let mut x = data(40, 3, 4);
    for row in &mut x {
        row[0] = 1.0;
    }
    let op = fit_op(&stage("(spline knots=5)"), &x, &target(&x)).unwrap();
    assert_eq!(output_width(&op, 3), Ok(1 + 4 + 4));
    assert!(apply_op(&op, &x).iter().all(|row| row.len() == 9));
}

// ---------------------------------------------------------------------------------------------
// The workplan cases and their siblings
// ---------------------------------------------------------------------------------------------

#[test]
fn select_corr_fitted_on_eight_columns_refuses_six() {
    // The target reads columns 1, 5 and 7 only, so k=3 keeps exactly those.
    let x = data(64, 8, 3);
    let y: Vec<f64> = x.iter().map(|row| row[1] - row[5] + row[7]).collect();
    let model = fit(&program("(pipe X (select_corr k=3) (ridge))"), &x, &y).unwrap();
    let ops = fitted_ops(&model);
    assert!(
        matches!(&ops[0], FittedOp::SelectCorr { indices, .. } if indices == &[1, 5, 7]),
        "{:?}",
        ops[0]
    );
    assert_eq!(required_input_width(&ops[0]), WidthReq::AtLeast(8));

    let valid = data(16, 8, 4);
    assert_eq!(
        bits(&model.try_predict(&valid).unwrap()),
        bits(&model.predict(&valid))
    );

    let narrow = data(16, 6, 5);
    assert_eq!(
        model.try_predict(&narrow),
        Err(WidthMismatch {
            stage: 0,
            expected: WidthReq::AtLeast(8),
            got: 6,
            kind: MismatchKind::StageInput,
        })
    );
    assert!(panics(|| model.predict(&narrow)), "predict still panics");

    // Wider rows are read at the same indices, which the contract admits.
    let wide = data(16, 9, 6);
    assert_eq!(
        bits(&model.try_predict(&wide).unwrap()),
        bits(&model.predict(&wide))
    );
}

#[test]
fn impute_fitted_on_four_columns_refuses_five() {
    let model = fit_program("(pipe X (impute) (ridge))", &data(64, 4, 7));
    assert_eq!(
        required_input_width(&fitted_ops(&model)[0]),
        WidthReq::Exactly(4)
    );
    let valid = data(16, 4, 8);
    assert_eq!(
        bits(&model.try_predict(&valid).unwrap()),
        bits(&model.predict(&valid))
    );
    let refusal = Err(WidthMismatch {
        stage: 0,
        expected: WidthReq::Exactly(4),
        got: 5,
        kind: MismatchKind::StageInput,
    });
    // A zero in column 4 makes predict read a fill value that was never fitted.
    let wide = data(16, 5, 9);
    assert!(wide.iter().any(|row| row[4] == 0.0));
    assert_eq!(model.try_predict(&wide), refusal);
    assert!(panics(|| model.predict(&wide)), "predict still panics");
    // Without one, predict passes column 4 through unfitted and does not panic.
    let unfilled: Rows = wide
        .iter()
        .map(|row| row.iter().map(|value| value + 0.125).collect())
        .collect();
    assert!(unfilled
        .iter()
        .all(|row| row.iter().all(|value| *value != 0.0)));
    assert_eq!(model.try_predict(&unfilled), refusal);
    assert!(!panics(|| model.predict(&unfilled)));
}

#[derive(Clone, Copy, Debug)]
enum Side {
    /// One column fewer than the stage requires.
    Narrower,
    /// One column more than an `Exactly` stage requires.
    Wider,
}

/// (program, fitted width, the side of the refused width, whether `predict` panics there)
const REFUSALS: &[(&str, usize, Side, bool)] = &[
    (
        "(pipe X (pair_ratio topk=4) (ridge))",
        6,
        Side::Narrower,
        true,
    ),
    (
        "(pipe X (pair_prod topk=4) (ridge))",
        6,
        Side::Narrower,
        true,
    ),
    ("(pipe X (poly_interact) (ridge))", 4, Side::Narrower, true),
    (
        "(pipe X (select_corr k=3) (ridge))",
        8,
        Side::Narrower,
        true,
    ),
    ("(pipe X (spline knots=4) (ridge))", 4, Side::Wider, true),
    ("(pipe X (standardize) (ridge))", 4, Side::Wider, true),
    ("(pipe X (standardize) (ridge))", 4, Side::Narrower, false),
    // Every column has a missing cell, so the fitted indicators include the last one.
    (
        "(pipe X (impute add_indicator=true) (ridge))",
        5,
        Side::Narrower,
        true,
    ),
    ("(pipe X (impute) (ridge))", 4, Side::Wider, true),
    ("(pipe X (winsor_clip) (ridge))", 4, Side::Wider, true),
    ("(pipe X (rank) (ridge))", 4, Side::Wider, true),
    ("(pipe X (bin_onehot bins=3) (ridge))", 4, Side::Wider, true),
    ("(pipe X (target_bin_encode) (ridge))", 4, Side::Wider, true),
    ("(pipe X (hinge) (ridge))", 4, Side::Wider, true),
    // Dot products truncate: predict misreads without panicking.
    ("(pipe X (rff_cos n=6) (ridge))", 4, Side::Wider, false),
    ("(pipe X (svd_project k=2) (ridge))", 4, Side::Wider, false),
    ("(pipe X (nystrom k=6) (ridge))", 4, Side::Narrower, false),
];

#[test]
fn inputs_predict_panics_on_or_misreads_are_refused_before_any_stage() {
    for (index, &(source, fitted, side, predict_panics)) in REFUSALS.iter().enumerate() {
        let seed = 10 * index as u64;
        let model = fit_program(source, &data(48, fitted, seed));
        let required = required_input_width(&fitted_ops(&model)[0]);
        let valid = data(16, fitted, seed + 1);
        assert_eq!(
            bits(&model.try_predict(&valid).unwrap()),
            bits(&model.predict(&valid)),
            "{source}: the fitted width"
        );
        let refused = match (side, required) {
            (Side::Narrower, WidthReq::Exactly(n) | WidthReq::AtLeast(n)) => n - 1,
            (Side::Wider, WidthReq::Exactly(n)) => n + 1,
            _ => panic!("{source}: {side:?} of {required:?}"),
        };
        let bad = data(16, refused, seed + 2);
        assert_eq!(
            model.try_predict(&bad),
            Err(WidthMismatch {
                stage: 0,
                expected: required,
                got: refused,
                kind: MismatchKind::StageInput,
            }),
            "{source} at width {refused}"
        );
        assert_eq!(
            panics(|| model.predict(&bad)),
            predict_panics,
            "{source}: predict at width {refused}"
        );
    }

    // `roll` divides by the width: an empty row is refused where predict divides by zero.
    let model = fit_program("(pipe X (roll k=2) (ridge))", &data(24, 3, 1));
    let empty_rows = vec![Vec::new(); 4];
    assert_eq!(
        model.try_predict(&empty_rows),
        Err(WidthMismatch {
            stage: 0,
            expected: WidthReq::AtLeast(1),
            got: 0,
            kind: MismatchKind::StageInput,
        })
    );
    assert!(panics(|| model.predict(&empty_rows)));
}

// ---------------------------------------------------------------------------------------------
// Heads, portfolios, symbolic models, ragged rows and empty inputs
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
enum HeadClass {
    Exact,
    LowerBound,
    Any,
}

#[test]
fn heads_require_the_width_their_fitted_state_reveals() {
    // (program, training rows, class of the head's requirement at 3 columns)
    let cases: &[(&str, usize, HeadClass)] = &[
        ("(pipe X (ridge))", 48, HeadClass::Exact),
        ("(pipe X (elasticnet))", 48, HeadClass::Exact),
        ("(pipe X (ridge_gcv))", 48, HeadClass::LowerBound),
        (
            "(pipe X (histgbm rounds=3 depth=2))",
            48,
            HeadClass::LowerBound,
        ),
        ("(pipe X (ensemble rounds=3 depth=2))", 48, HeadClass::Exact),
        (
            "(pipe X (histgbm2 rounds=3 leaves=4 bags=1))",
            48,
            HeadClass::LowerBound,
        ),
        ("(pipe X (krr m=50))", 48, HeadClass::Exact),
        ("(pipe X (gpr m=50 steps=2))", 48, HeadClass::Exact),
        (
            "(pipe X (gpr kernel=rq+linear m=50 steps=2))",
            48,
            HeadClass::Exact,
        ),
        ("(pipe X (ksmooth k=8))", 48, HeadClass::Exact),
        ("(pipe X (hurdle2))", 48, HeadClass::Exact),
        ("(pipe X (stack))", 48, HeadClass::Exact),
        // Constant fallbacks on too few rows keep no column.
        ("(pipe X (histgbm2 rounds=3))", 8, HeadClass::Any),
        ("(pipe X (histgbm rounds=3))", 3, HeadClass::Any),
        ("(pipe X (hurdle2))", 8, HeadClass::Any),
    ];
    for (index, &(source, rows, class)) in cases.iter().enumerate() {
        let seed = 7 * index as u64;
        let model = fit_program(source, &data(rows, 3, seed));
        let Model::Pipeline { head, .. } = &model else {
            panic!("{source}: a pipeline model")
        };
        let required = required_head_width(head);
        let expected = match class {
            HeadClass::Exact => WidthReq::Exactly(3),
            HeadClass::LowerBound => WidthReq::AtLeast(3),
            HeadClass::Any => WidthReq::Any,
        };
        assert_eq!(required, expected, "{source}");
        for width in [2, 3, 4] {
            let x = data(6, width, seed + 1);
            // Heads read columns defensively: predict never panics on width.
            let predicted = catch_unwind(AssertUnwindSafe(|| model.predict(&x)))
                .unwrap_or_else(|_| panic!("{source}: predict at width {width} panicked"));
            match model.try_predict(&x) {
                Ok(tried) => {
                    assert!(required.admits(width), "{source} at width {width}");
                    assert_eq!(bits(&tried), bits(&predicted), "{source} at width {width}");
                }
                Err(mismatch) => assert_eq!(
                    mismatch,
                    WidthMismatch {
                        stage: 0,
                        expected: required,
                        got: width,
                        kind: MismatchKind::HeadInput,
                    },
                    "{source} at width {width}"
                ),
            }
        }
    }
}

#[test]
fn a_tree_head_on_wide_rows_requires_only_its_feature_count() {
    // histgbm2 keeps at most 256 features; 300 columns stay admitted.
    let model = fit_program(
        "(pipe X (histgbm2 rounds=3 leaves=4 bags=1))",
        &data(40, 300, 12),
    );
    let Model::Pipeline { head, .. } = &model else {
        panic!("a pipeline model")
    };
    assert_eq!(required_head_width(head), WidthReq::AtLeast(256));
    let wide = data(8, 300, 13);
    assert_eq!(
        bits(&model.try_predict(&wide).unwrap()),
        bits(&model.predict(&wide))
    );
    assert_eq!(
        model.try_predict(&data(8, 255, 14)),
        Err(WidthMismatch {
            stage: 0,
            expected: WidthReq::AtLeast(256),
            got: 255,
            kind: MismatchKind::HeadInput,
        })
    );
}

#[test]
fn portfolios_require_their_fitted_width_unless_they_predict_the_mean() {
    // No training rows give a mean-only portfolio; the fields are public, so a padded one is
    // built from it.
    let mean_only = feat_math::portfolio::fit_portfolio(&[], &Vec::new(), &[], 5, false).unwrap();
    assert!(mean_only.mean_only);
    let mut padded = mean_only.clone();
    padded.mean_only = false;
    padded.width = 3;
    padded.keep = vec![0, 1, 2];

    let explicit = Model::Portfolio(mean_only.clone());
    for width in [0, 3, 7] {
        let x = data(4, width, 1);
        assert_eq!(
            bits(&explicit.try_predict(&x).unwrap()),
            bits(&explicit.predict(&x))
        );
    }

    for model in [
        Model::Portfolio(padded.clone()),
        Model::Pipeline {
            target: None,
            ops: Vec::new(),
            head: Box::new(FittedHead::Portfolio(Box::new(padded))),
        },
    ] {
        let x = data(4, 3, 2);
        assert_eq!(
            bits(&model.try_predict(&x).unwrap()),
            bits(&model.predict(&x))
        );
        let wide = data(4, 5, 3);
        assert_eq!(
            model.try_predict(&wide),
            Err(WidthMismatch {
                stage: 0,
                expected: WidthReq::Exactly(3),
                got: 5,
                kind: MismatchKind::HeadInput,
            })
        );
        // The portfolio pads or truncates rows itself, so predict does not panic.
        assert!(!panics(|| model.predict(&wide)));
    }
}

#[test]
fn symbolic_models_accept_any_width() {
    // An expression that reads a column beyond the row evaluates to 0.0: no width to require.
    let model = fit_program(
        "(sym affine \"sigmoid(c0) + 0.5*c2 - row_mean\")",
        &data(24, 3, 3),
    );
    for width in [0, 1, 3, 9] {
        let x = data(5, width, 4);
        assert_eq!(check_input_width(&model, width), Ok(()));
        assert_eq!(
            bits(&model.try_predict(&x).unwrap()),
            bits(&model.predict(&x))
        );
    }
}

#[test]
fn ragged_rows_are_refused_and_empty_inputs_pass() {
    let pipeline = fit_program("(pipe X (standardize) (ridge))", &data(24, 3, 1));
    let symbolic = fit_program("(sym raw \"c0 + c1\")", &data(24, 3, 1));
    let mut ragged = data(4, 3, 2);
    ragged[2].pop();
    for model in [&pipeline, &symbolic] {
        assert_eq!(
            model.try_predict(&ragged),
            Err(WidthMismatch {
                stage: 0,
                expected: WidthReq::Exactly(3),
                got: 2,
                kind: MismatchKind::RaggedInput { row: 2 },
            })
        );
        let empty: Rows = Vec::new();
        assert_eq!(model.try_predict(&empty), Ok(model.predict(&empty)));
        assert_eq!(model.try_predict(&empty), Ok(Vec::new()));
    }
}

// ---------------------------------------------------------------------------------------------
// Inconsistent fitted state and overflowing output widths
// ---------------------------------------------------------------------------------------------

type Edit = fn(&mut FittedOp);

/// One edit per rule of `check_fitted_state`, on a stage fitted on 4 columns: (stage, edit,
/// whether `predict` panics at the width the edited stage claims). No fit produces these.
#[rustfmt::skip]
const INCONSISTENT: &[(&str, Edit, bool)] = &[
    ("(standardize)", |op| if let FittedOp::Standardize { scales, .. } = op { scales.pop(); }, true),
    ("(robust_scale)", |op| if let FittedOp::RobustScale { scales, .. } = op { scales.pop(); }, true),
    ("(bin_onehot bins=4)", |op| if let FittedOp::BinOnehot { bins, .. } = op { *bins = 0; }, true),
    ("(hinge knots=3)", |op| if let FittedOp::Hinge { knots, .. } = op { knots.clear(); }, true),
    ("(winsor_clip q=0.1)", |op| if let FittedOp::WinsorClip { hi, .. } = op { hi.pop(); }, true),
    ("(winsor_clip q=0.1)", |op| if let FittedOp::WinsorClip { lo, hi, .. } = op { lo[0] = hi[0] + 1.0; }, true),
    ("(winsor_clip q=0.1)", |op| if let FittedOp::WinsorClip { lo, .. } = op { lo[1] = f64::NAN; }, true),
    ("(rff_cos n=5 seed=11)", |op| if let FittedOp::RffCos { b, .. } = op { b.pop(); }, true),
    ("(rff_sin n=5 seed=11)", |op| if let FittedOp::RffSin { w, .. } = op { w[2].pop(); }, false),
    ("(rand_proj k=5 seed=11)", |op| if let FittedOp::RandProj { w, .. } = op { w[1].push(0.5); }, false),
    ("(oblique_hinge n=5 seed=11)", |op| if let FittedOp::ObliqueHinge { t, .. } = op { t.pop(); }, true),
    ("(landmark_dist k=5 seed=11)", |op| if let FittedOp::LandmarkDist { landmarks, .. } = op { landmarks[3].pop(); }, false),
    ("(nystrom k=5 seed=11)", |op| if let FittedOp::Nystrom { landmarks, .. } = op { landmarks[2].push(0.0); }, false),
    ("(svd_project k=3 iters=8 seed=11)", |op| if let FittedOp::SvdProject { components, .. } = op { components[1].pop(); }, true),
    ("(target_bin_encode bins=5)", |op| if let FittedOp::TargetBinEncode { enc, .. } = op { enc.pop(); }, true),
    ("(target_bin_encode bins=5)", |op| if let FittedOp::TargetBinEncode { enc, .. } = op { enc[2].clear(); }, true),
    ("(impute add_indicator=true)", |op| if let FittedOp::Impute { indicator_cols, .. } = op { indicator_cols.push(4); }, true),
    ("(pair_ratio topk=4)", |op| if let FittedOp::PairRatio { eps, .. } = op { eps.pop(); }, false),
    ("(spline knots=4)", |op| if let FittedOp::Spline { knots, .. } = op { knots[1] = vec![f64::NAN]; }, true),
    ("(spline knots=4)", |op| if let FittedOp::Spline { knots, .. } = op { knots[3].clear(); }, true),
];

#[test]
fn inconsistent_fitted_state_is_refused_where_predict_panics() {
    let x = data(24, 4, 6);
    let y = target(&x);
    for (index, &(source, edit, predict_panics)) in INCONSISTENT.iter().enumerate() {
        let mut op = fit_op(&stage(source), &x, &y).unwrap();
        let before = format!("{op:?}");
        edit(&mut op);
        assert_ne!(
            format!("{op:?}"),
            before,
            "case {index} ({source}) edits the state"
        );
        let reason = check_fitted_state(&op).expect_err(source);
        let width = claimed_width(&op);
        assert!(
            matches!(output_width(&op, width), Err(Error::Shape(_))),
            "case {index} ({source})"
        );
        let model = one_stage_model(op, 3);
        let rows = data(3, width, 7);
        assert_eq!(
            model.try_predict(&rows),
            Err(WidthMismatch {
                stage: 0,
                expected: required_input_width(&fitted_ops(&model)[0]),
                got: width,
                kind: MismatchKind::InconsistentState { reason },
            }),
            "case {index} ({source})"
        );
        assert_eq!(
            panics(|| model.predict(&rows)),
            predict_panics,
            "case {index} ({source}): predict"
        );
    }
}

#[test]
fn output_widths_that_overflow_are_refused() {
    let x = data(24, 3, 8);
    let y = target(&x);
    let largest_row = isize::MAX as usize / std::mem::size_of::<f64>();

    let mut cheb = fit_op(&stage("(cheb degree=3)"), &x, &y).unwrap();
    if let FittedOp::Cheb { degree, .. } = &mut cheb {
        *degree = usize::MAX;
    }
    assert!(matches!(output_width(&cheb, 2), Err(Error::Resource(_))));
    // Nothing to multiply: an empty row stays empty and is padded.
    assert_eq!(output_width(&cheb, 0), Ok(1));
    let model = one_stage_model(cheb, 3);
    assert_eq!(
        check_input_width(&model, 2),
        Err(WidthMismatch {
            stage: 0,
            expected: WidthReq::Any,
            got: 2,
            kind: MismatchKind::OutputOverflow,
        })
    );
    assert_eq!(
        model.try_predict(&data(2, 2, 9)).unwrap_err().kind,
        MismatchKind::OutputOverflow
    );

    let mut onehot = fit_op(&stage("(bin_onehot bins=4)"), &x, &y).unwrap();
    if let FittedOp::BinOnehot { bins, .. } = &mut onehot {
        *bins = usize::MAX / 2;
    }
    assert!(matches!(output_width(&onehot, 3), Err(Error::Resource(_))));

    // `fft_mag` writes k columns: up to the largest allocatable row they are only capped (the
    // stage is never applied here), one more is refused, and so is appending to the largest.
    for (k, append, expected) in [
        (1usize << 40, false, Some(512)),
        (largest_row, false, Some(512)),
        (largest_row + 1, false, None),
        (largest_row, true, None),
        (usize::MAX, false, None),
    ] {
        let op = FittedOp::FftMag { k, append };
        match expected {
            Some(width) => assert_eq!(output_width(&op, 1), Ok(width), "k={k} append={append}"),
            None => assert!(
                matches!(output_width(&op, 1), Err(Error::Resource(_))),
                "k={k} append={append}"
            ),
        }
    }
}

// ---------------------------------------------------------------------------------------------
// PR 01's compatibility corpus
// ---------------------------------------------------------------------------------------------

fn fixture(name: &str) -> serde_json::Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/compat_freeze")
        .join(name);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path:?}: {e}"));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{path:?}: {e}"))
}

/// Deterministic regression data, copied from `tests/golden.rs` through `compat_freeze.rs`
/// (both hash-pinned, so copied rather than shared).
fn toy_data(n: usize, cols: usize, seed: u64) -> (Rows, Vec<f64>) {
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

/// The corpus's fitted surface: the first 64 rows train, the last 16 are predicted.
fn corpus_data() -> (Rows, Vec<f64>, Rows) {
    let (x, y) = toy_data(80, 5, 42);
    (x[..64].to_vec(), y[..64].to_vec(), x[64..].to_vec())
}

fn row_text(values: &[f64]) -> String {
    values
        .iter()
        .map(|value| format!("{:016x}", value.to_bits()))
        .collect::<Vec<_>>()
        .join(" ")
}

/// The widths a pipeline model's rows have, replayed stage by stage on `x`: the input width,
/// then after every stage. Each is checked against the static walk on the way.
fn replay_widths(name: &str, model: &Model, x: &Rows) -> Vec<usize> {
    let Model::Pipeline { ops, head, .. } = model else {
        return Vec::new();
    };
    let mut current = x.clone();
    sanitize(&mut current);
    let mut width = current[0].len();
    let mut widths = vec![width];
    for (index, op) in ops.iter().enumerate() {
        assert!(
            required_input_width(op).admits(width),
            "{name}: stage {index} at width {width}"
        );
        let expected = output_width(op, width).unwrap_or_else(|e| panic!("{name}: {e}"));
        current = apply_op(op, &current);
        assert!(
            current.iter().all(|row| row.len() == expected),
            "{name}: stage {index} output_width {expected}"
        );
        width = expected;
        widths.push(width);
    }
    assert!(
        required_head_width(head).admits(width),
        "{name}: the head at width {width}"
    );
    widths
}

#[test]
fn compat_corpus_try_predict_equals_predict_bit_for_bit() {
    let emitted = fixture("emitted_bytes.v1.json");
    let serde = fixture("model_serde.v1.json");
    let programs = emitted["programs"].as_array().expect("programs");
    let recorded = serde["models"].as_array().expect("models");
    assert_eq!(programs.len(), 55);
    let (train, target, held_out) = corpus_data();
    let (mut fitted, mut skipped) = (0, 0);
    let (mut accepted_elsewhere, mut refused_elsewhere) = (0, 0);
    for entry in programs {
        let name = entry["name"].as_str().expect("name");
        let canonical = entry["canonical"].as_str().expect("canonical");
        if canonical.contains("(portfolio") {
            // As in compat_freeze: a portfolio fit takes minutes in a debug build.
            skipped += 1;
            continue;
        }
        let model = fit(&program(canonical), &train, &target)
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        fitted += 1;

        let predicted = model.predict(&held_out);
        let tried = model
            .try_predict(&held_out)
            .unwrap_or_else(|mismatch| panic!("{name}: {mismatch}"));
        assert_eq!(bits(&tried), bits(&predicted), "{name}");
        let record = recorded
            .iter()
            .find(|record| record["name"] == name)
            .unwrap_or_else(|| panic!("{name}: no recorded predictions"));
        assert_eq!(
            row_text(&tried),
            record["predictions"].as_str().expect("predictions"),
            "{name}: the recorded predictions"
        );

        let widths = replay_widths(name, &model, &held_out);
        match name {
            // poly_interact appends 512 pairs to 70 columns; the row is capped at 512.
            "champion" => assert_eq!(widths, [5, 10, 70, 70, 70, 512, 512], "{name}"),
            // One column leaves pair_ratio no pair; the empty row is padded to one column.
            "empty_stage_output" => assert_eq!(widths, [5, 1, 1], "{name}"),
            _ => {}
        }

        // At any other width, try_predict never panics, and a value it returns is predict's.
        for (index, width) in [0usize, 1, 4, 6, 9].into_iter().enumerate() {
            let x = data(4, width, 50 + index as u64);
            let outcome = catch_unwind(AssertUnwindSafe(|| model.try_predict(&x)))
                .unwrap_or_else(|_| panic!("{name}: try_predict panicked at width {width}"));
            match outcome {
                Ok(values) => {
                    assert_eq!(bits(&values), bits(&model.predict(&x)), "{name}");
                    accepted_elsewhere += 1;
                }
                Err(_) => refused_elsewhere += 1,
            }
        }
    }
    assert_eq!((fitted, skipped), (54, 1));
    assert!(accepted_elsewhere > 0 && refused_elsewhere > 0);
    println!(
        "CORPUS fitted {fitted}, skipped {skipped}; at widths 0, 1, 4, 6 and 9: \
         {accepted_elsewhere} accepted, {refused_elsewhere} refused"
    );
}

#[test]
fn stored_compat_models_try_predict_equals_predict() {
    let serde = fixture("model_serde.v1.json");
    let stored = serde["stored"].as_array().expect("stored");
    assert_eq!(stored.len(), 5);
    let (_, _, held_out) = corpus_data();
    for entry in stored {
        let name = entry["name"].as_str().expect("name");
        // The same in-memory model on both paths: a reload may differ in the last bits from the
        // recording build (compat_freeze explains why), never between two calls.
        let model: Model = serde_json::from_str(entry["json"].as_str().expect("json"))
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(
            bits(&model.try_predict(&held_out).unwrap()),
            bits(&model.predict(&held_out)),
            "{name}"
        );
        replay_widths(name, &model, &held_out);
    }
}

// ---------------------------------------------------------------------------------------------
// The schema-bound (advanced-v3) boundary
// ---------------------------------------------------------------------------------------------

fn numeric_schema() -> DataSchema {
    DataSchema {
        columns: vec![ColumnSpec::numeric("a"), ColumnSpec::numeric("b")],
        row_semantics: RowSemantics::Iid,
    }
}

fn training_table() -> RawTable {
    RawTable::from_csv(
        "a,b\n0,-2\n0.25,-1\n0.5,0\n0.75,1\n1,2\n1.25,3\n1.5,4\n1.75,5\n2,6\n2.25,7\n",
    )
    .unwrap()
}

const TRAINING_TARGET: [f64; 10] = [0.1, 0.4, 0.2, 0.9, 1.1, 1.6, 1.4, 2.1, 2.4, 2.9];

fn bound(source: &str) -> BoundModel {
    fit_table(
        &program(source),
        &TaskSpec::regression(Objective::Rmse).unwrap(),
        &numeric_schema(),
        &training_table(),
        &TRAINING_TARGET,
    )
    .unwrap_or_else(|error| panic!("{source}: {error}"))
}

#[test]
fn schema_bound_stages_return_errors_instead_of_panicking() {
    let table = RawTable::from_csv("a,b\n0.125,-1.5\n0.5,2.5\n2.125,6.5\n").unwrap();
    // (program, path to the first stage's fitted state, edit, message fragment)
    type Tamper = fn(&mut serde_json::Value);
    let cases: &[(&str, &[&str], Tamper, &str)] = &[
        (
            "(pipe X (select_corr k=2) (ridge))",
            &["Legacy", "SelectCorr", "indices"],
            |indices| *indices = serde_json::json!([0, 99]),
            "legacy stage select_corr needs at least 100 input columns, but the feature block has 2",
        ),
        (
            "(pipe X (impute add_indicator=true) (ridge))",
            &["Legacy", "Impute", "indicator_cols"],
            |indicators| indicators.as_array_mut().unwrap().push(serde_json::json!(5)),
            "legacy stage impute has inconsistent fitted state: an indicator column is not a fitted column",
        ),
        (
            "(pipe X (standardize) (ridge))",
            &["Legacy", "Standardize", "scales"],
            |scales| *scales = serde_json::json!([1.0]),
            "legacy stage standardize has inconsistent fitted state: means and scales differ in length",
        ),
        (
            "(pipe X (corr_screen k=32) (ridge))",
            &["CorrScreen", "indices"],
            |indices| *indices = serde_json::json!([0, 7]),
            "corr_screen reads column 7, but the feature block has 2 columns",
        ),
        (
            "(pipe X (pls_project k=2) (ridge))",
            &["PlsProject", "weights"],
            |weights| weights[0].as_array_mut().unwrap().extend([serde_json::json!(0.5), serde_json::json!(0.5)]),
            "pls_project weight spans 4 columns, but the feature block has 2 and the fitted means 2",
        ),
    ];
    for &(source, path, tamper, message) in cases {
        let model = bound(source);
        assert!(
            model.predict_table(&table).is_ok(),
            "{source}: the fitted model"
        );
        let mut json = serde_json::to_value(&model).unwrap();
        let mut state = &mut json["predictor"]["Pipeline"]["ops"][0];
        for key in path {
            state = &mut state[*key];
        }
        assert!(!state.is_null(), "{source}: {path:?} is the fitted state");
        tamper(state);
        let tampered: BoundModel = serde_json::from_value(json).unwrap();
        let outcome = catch_unwind(AssertUnwindSafe(|| tampered.predict_table(&table)))
            .unwrap_or_else(|_| panic!("{source}: predict_table panicked"));
        let error = outcome.expect_err(source).to_string();
        assert_eq!(error, message, "{source}");
    }
}

#[test]
fn schema_bound_tables_with_no_rows_compare_no_width() {
    // With no rows nothing is read, and a stage that generates columns takes its width from the
    // first row, so the block after it is narrower than at fit time: no column, or only the
    // appended input. No width is compared then, and each of these keeps its empty result.
    let empty = RawTable::new(vec!["a".to_string(), "b".to_string()], Vec::new()).unwrap();
    for source in [
        "(pipe X (rank) (standardize) (ridge))",
        "(pipe X (quantile_bins) (winsor_clip) (ridge))",
        "(pipe X (rank append=true) (standardize) (ridge))",
        "(pipe X (cheb degree=3) (standardize) (ridge))",
        "(pipe X (pair_ratio topk=4) (standardize) (ridge))",
        "(pipe X (hinge knots=3) (impute) (ridge))",
        "(pipe X (rank append=true) (pls_project k=2) (ridge))",
        "(pipe X (select_corr k=2) (impute add_indicator=true) (ridge))",
    ] {
        let model = bound(source);
        let predicted = catch_unwind(AssertUnwindSafe(|| model.predict_table(&empty)))
            .unwrap_or_else(|_| panic!("{source}: predict_table panicked"))
            .unwrap_or_else(|error| panic!("{source}: {error}"));
        assert_eq!(predicted, Vec::<f64>::new(), "{source}");
    }

    // The multiclass entry point walks the same stages.
    let mut csv = String::from("a,b\n");
    let mut labels = Vec::new();
    for class in 0..3 {
        for repeat in 0..4 {
            let (class, repeat) = (f64::from(class), f64::from(repeat));
            csv.push_str(&format!(
                "{},{}\n",
                class + 0.1 * repeat,
                0.3 * repeat - class
            ));
            labels.push(class);
        }
    }
    let classes = RawTable::from_csv(&csv).unwrap();
    let task = TaskSpec::multiclass(Objective::MacroF1).unwrap();
    for source in [
        "(pipe X (rank) (standardize) (multiclass_ridge))",
        "(pipe X (rank append=true) (standardize) (multiclass_ridge))",
    ] {
        let model = fit_table(
            &program(source),
            &task,
            &numeric_schema(),
            &classes,
            &labels,
        )
        .unwrap_or_else(|error| panic!("{source}: {error}"));
        let predicted = catch_unwind(AssertUnwindSafe(|| model.predict_multiclass_table(&empty)))
            .unwrap_or_else(|_| panic!("{source}: predict_multiclass_table panicked"))
            .unwrap_or_else(|error| panic!("{source}: {error}"));
        assert!(
            predicted.labels.is_empty() && predicted.probabilities.is_empty(),
            "{source}"
        );
    }

    // Finishing a stage looks the block up at a fitted column that the narrower block lacks:
    // this panicked, and is an error now.
    for (source, stage) in [
        (
            "(pipe X (rank) (select_corr k=2) (ridge))",
            "legacy stage select_corr",
        ),
        (
            "(pipe X (hinge knots=3) (impute add_indicator=true) (ridge))",
            "legacy stage impute",
        ),
        ("(pipe X (hinge) (corr_screen k=32) (ridge))", "corr_screen"),
    ] {
        let model = bound(source);
        let error = catch_unwind(AssertUnwindSafe(|| model.predict_table(&empty)))
            .unwrap_or_else(|_| panic!("{source}: predict_table panicked"))
            .expect_err(source)
            .to_string();
        assert!(
            error.starts_with(&format!("{stage} reads column "))
                && error.ends_with(", but the feature block has 0 columns"),
            "{source}: {error}"
        );
    }
}

// ---------------------------------------------------------------------------------------------
// Messages
// ---------------------------------------------------------------------------------------------

#[test]
fn mismatch_messages_name_the_stage_and_the_widths() {
    let cases = [
        (
            MismatchKind::StageInput,
            WidthReq::AtLeast(8),
            "stage 2 needs at least 8 input columns, got 6",
        ),
        (
            MismatchKind::HeadInput,
            WidthReq::Exactly(8),
            "the head needs exactly 8 input columns, got 6",
        ),
        (
            MismatchKind::RaggedInput { row: 4 },
            WidthReq::Exactly(8),
            "ragged input: row 4 has 6 columns, every row needs exactly 8",
        ),
        (
            MismatchKind::InconsistentState {
                reason: "means and scales differ in length",
            },
            WidthReq::Exactly(8),
            "stage 2 has inconsistent fitted state: means and scales differ in length",
        ),
        (
            MismatchKind::OutputOverflow,
            WidthReq::Any,
            "stage 2 output width overflows at input width 6",
        ),
    ];
    for (kind, expected, message) in cases {
        let mismatch = WidthMismatch {
            stage: 2,
            expected,
            got: 6,
            kind,
        };
        assert_eq!(mismatch.to_string(), message);
    }
    assert_eq!(WidthReq::Any.to_string(), "any number of");
}
