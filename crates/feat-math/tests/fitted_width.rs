//! The fitted-width contract (kernel2 WP-1.7): `kernel::width` and `Model::try_predict`, checked
//! against `Model::predict`, which now refuses zero-width rows.
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
//!   instead of panicking, and a table with no rows keeps its result. The `piecewise_linear` arm,
//!   which the WP-1.7 receipt recorded as a gap, has been guarded the same way since kernel2 PR 07
//!   (X3), and its test asserts the error.

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
mod compat_corpus;
mod input_prediction;
mod schema_boundary;
mod width_contract;
