//! Current `MathExpr` semantics, pinned as explicit tests (workplan WP-0.5, OQ-7).
//!
//! Every value here is what the evaluator returns today, including the surprising ones (the
//! division and log guards, the round and asinh quirks, the sign of empty-row sums). Changing any
//! of them is a versioned decision, never a fix in place: the owner kept the v1 values (OQ-7), and
//! a new semantics would be a new version with its own fixtures.
//!
//! Inputs come from the shared boundary corpus `tests/fixtures/math_boundary/v1.json` (WP-0.3)
//! by name. Every row is checked through the interpreter (`MathExpr::eval_row`) and the compiled
//! form (`CompiledMathExpr::eval_row`), which must agree bit for bit.
//!
//! Two rows of the WP-0.5 table are recorded as the base actually behaves, not as the table
//! guessed: an empty row's `row_mean` (and `row_sum`, `row_l1`, `row_l2`, `row_std`) is -0.0,
//! because an empty `f64` sum is -0.0 in this toolchain; and `clip(col(0), nan, nan)` cannot be
//! written with literals (`nan` is not a literal), so the NaN bounds come from the row, from
//! `(-1)^0.5`, or from a constructed AST.

use feat_math::dsl::parse_dsl;
use feat_math::math_expr::{BinOp, Expr, Func, MathEvalError, MathExpr, Var};
use feat_math::ops::{apply_op, fit_op};
use std::collections::BTreeMap;

const POS_ZERO: u64 = 0x0000_0000_0000_0000;
const NEG_ZERO: u64 = 0x8000_0000_0000_0000;
const MAX_ABS: u64 = 0x426d_1a94_a200_0000; // 1e12, the evaluator's magnitude clamp
const ONE: u64 = 0x3ff0_0000_0000_0000;

/// The boundary corpus by entry name.
fn corpus() -> BTreeMap<String, f64> {
    let text = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/math_boundary/v1.json"
    ))
    .expect("tests/fixtures/math_boundary/v1.json");
    let file: serde_json::Value = serde_json::from_str(&text).expect("corpus JSON");
    file["entries"]
        .as_array()
        .expect("entries")
        .iter()
        .map(|entry| {
            let bits = entry["bits"]
                .as_str()
                .and_then(|bits| bits.strip_prefix("0x"))
                .expect("bits");
            (
                entry["name"].as_str().expect("name").to_string(),
                f64::from_bits(u64::from_str_radix(bits, 16).expect("hex bits")),
            )
        })
        .collect()
}

fn b(name: &str) -> f64 {
    *corpus()
        .get(name)
        .unwrap_or_else(|| panic!("boundary corpus entry {name} is missing"))
}

/// `eval_row` bits through the interpreter and the compiled form (which must agree).
fn row_bits(source: &str, row: &[f64]) -> u64 {
    let expr = MathExpr::parse(source).unwrap_or_else(|error| panic!("{source}: {error}"));
    let interpreted = expr
        .eval_row(row)
        .unwrap_or_else(|error| panic!("{source} on {row:?}: {error}"))
        .to_bits();
    let compiled = expr
        .compile()
        .expect("compile")
        .eval_row(row)
        .unwrap_or_else(|error| panic!("compiled {source} on {row:?}: {error}"))
        .to_bits();
    assert_eq!(
        interpreted, compiled,
        "{source} on {row:?}: interpreter and compiled form disagree"
    );
    interpreted
}

fn assert_row(source: &str, row: &[f64], expected: u64) {
    let actual = row_bits(source, row);
    assert_eq!(
        actual,
        expected,
        "{source} on {row:?}: expected 0x{expected:016x} ({:?}), got 0x{actual:016x} ({:?})",
        f64::from_bits(expected),
        f64::from_bits(actual)
    );
}

/// The `eval_row` error message through both forms.
fn row_error(source: &str, row: &[f64]) -> String {
    let expr = MathExpr::parse(source).unwrap_or_else(|error| panic!("{source}: {error}"));
    let interpreted = expr.eval_row(row).expect_err("an error").to_string();
    let compiled = expr
        .compile()
        .expect("compile")
        .eval_row(row)
        .expect_err("a compiled error")
        .to_string();
    assert_eq!(interpreted, compiled, "{source}: error messages differ");
    interpreted
}

/// One fitted `map`/`derive` stage applied to `rows`.
fn stage_output(stage: &str, rows: &[Vec<f64>]) -> Vec<Vec<u64>> {
    let stage = parse_dsl(&format!("(pipe X {stage})"))
        .expect("stage")
        .stages
        .remove(0);
    let rows = rows.to_vec();
    let op = fit_op(&stage, &rows, &vec![0.0; rows.len()]).expect("fit");
    apply_op(&op, &rows)
        .iter()
        .map(|row| row.iter().map(|v| v.to_bits()).collect())
        .collect()
}

fn assert_invalid_clip_bounds(expr: &MathExpr) {
    let error = expr.eval_row(&[1.0]).expect_err("an error");
    assert_eq!(
        error.downcast_ref::<MathEvalError>(),
        Some(&MathEvalError::InvalidClipBounds)
    );
    let error = expr
        .compile()
        .expect("compile")
        .eval_row(&[1.0])
        .expect_err("a compiled error");
    assert_eq!(
        error.downcast_ref::<MathEvalError>(),
        Some(&MathEvalError::InvalidClipBounds)
    );
}

#[test]
fn division_by_zero_returns_zero() {
    // CURRENT SEMANTICS -- changing this is a versioned decision (workplan WP-0.5 / OQ-7)
    assert_row("1/col(0)", &[b("pos_zero")], POS_ZERO);
    assert_row("1/col(0)", &[b("neg_zero")], POS_ZERO);
}

#[test]
fn division_guard_is_strictly_below_1e_minus_12() {
    // CURRENT SEMANTICS -- changing this is a versioned decision (workplan WP-0.5 / OQ-7)
    // The guard is `|b| < 1e-12`, not `<=`: 1/1e-12 is exactly 1e12 (the magnitude clamp).
    assert_row("1/col(0)", &[b("div_guard_pos")], MAX_ABS);
    assert_row("1/col(0)", &[b("div_guard_neg")], MAX_ABS | NEG_ZERO);
    // One ulp below the guard divides to zero (Appendix B).
    assert_row("1/col(0)", &[b("div_guard_pos_toward_zero")], POS_ZERO);
}

#[test]
fn log_returns_zero_at_or_below_1e_minus_300() {
    // CURRENT SEMANTICS -- changing this is a versioned decision (workplan WP-0.5 / OQ-7)
    assert_row("log(col(0))", &[b("neg_one")], POS_ZERO);
    assert_row("log(col(0))", &[b("log_guard")], POS_ZERO);
    // log(1) is a real zero, indistinguishable from the guard's zero.
    assert_row("log(col(0))", &[b("pos_one")], POS_ZERO);
}

#[test]
fn log1p_returns_zero_at_or_below_minus_one() {
    // CURRENT SEMANTICS -- changing this is a versioned decision (workplan WP-0.5 / OQ-7)
    assert_row("log1p(col(0))", &[b("neg_one")], POS_ZERO);
    assert_row("log1p(col(0))", &[b("neg_one_away_from_zero")], POS_ZERO);
    // Two ulps above -1 is a finite log (Appendix B: -36.04365338911715; recorded at base).
    assert_row(
        "log1p(col(0))",
        &[b("neg_one_two_ulps_toward_zero")],
        0xc042_0596_6f2b_4f12,
    );
}

#[test]
fn sqrt_of_a_negative_is_zero() {
    // CURRENT SEMANTICS -- changing this is a versioned decision (workplan WP-0.5 / OQ-7)
    assert_row("sqrt(col(0))", &[-4.0], POS_ZERO);
}

#[test]
fn exp_saturates_through_the_magnitude_clamp() {
    // CURRENT SEMANTICS -- changing this is a versioned decision (workplan WP-0.5 / OQ-7)
    // Below saturation: 532048240601.79865 (Appendix B; the bits are recorded at base).
    assert_row("exp(col(0))", &[b("pos_27")], 0x425e_f822_f7f6_731d);
    // ln(1e12) itself stays just below the clamp (Appendix B; recorded at base).
    assert_row("exp(col(0))", &[b("ln_1e12")], 0x426d_1a94_a1ff_fff8);
    for name in ["pos_27_64", "pos_28", "pos_30", "pos_31"] {
        assert_row("exp(col(0))", &[b(name)], MAX_ABS);
    }
    // The argument is clamped to [-30, 30] before exp (Appendix B; recorded at base).
    assert_row("exp(col(0))", &[b("neg_30")], 0x3d3a_56e0_c2ac_7f75);
    assert_row("exp(col(0))", &[b("neg_31")], 0x3d3a_56e0_c2ac_7f75);
}

#[test]
fn sigmoid_clamps_its_argument_to_plus_minus_30() {
    // CURRENT SEMANTICS -- changing this is a versioned decision (workplan WP-0.5 / OQ-7)
    // Appendix B values; the bits are recorded at base.
    assert_row("sigmoid(col(0))", &[b("pos_30")], 0x3fef_ffff_ffff_fcb6);
    assert_row("sigmoid(col(0))", &[b("pos_31")], 0x3fef_ffff_ffff_fcb6);
    assert_row("sigmoid(col(0))", &[b("neg_30")], 0x3d3a_56e0_c2ac_7cbf);
    assert_row("sigmoid(col(0))", &[b("neg_31")], 0x3d3a_56e0_c2ac_7cbf);
}

#[test]
fn round_is_half_away_from_zero_by_the_shared_branch_formula() {
    // CURRENT SEMANTICS -- changing this is a versioned decision (workplan WP-0.5 / OQ-7)
    assert_row("round(col(0))", &[b("pos_half")], ONE);
    assert_row("round(col(0))", &[b("neg_half")], ONE | NEG_ZERO);
    assert_row("round(col(0))", &[b("pos_one_and_half")], 2.0f64.to_bits());
    assert_row(
        "round(col(0))",
        &[b("neg_one_and_half")],
        (-2.0f64).to_bits(),
    );
    assert_row("round(col(0))", &[b("pos_two_and_half")], 3.0f64.to_bits());
    // V-11: floor(x + 0.5) rounds 0.49999999999999994 up, because x + 0.5 rounds to 1.0.
    assert_row("round(col(0))", &[b("below_half")], ONE);
    // The formula would move 2^52 + 1 to 2^52 + 2 (x + 0.5 is a tie that rounds to even), but
    // every value above 1e12 leaves the evaluator as the magnitude clamp (recorded at base).
    assert_row("round(col(0))", &[b("two_pow_52_plus_one")], MAX_ABS);
}

#[test]
fn asinh_overflow_becomes_zero_through_the_output_filter() {
    // CURRENT SEMANTICS -- changing this is a versioned decision (workplan WP-0.5 / OQ-7)
    // V-10: v*v overflows to infinity, ln(inf) is inf, and the output filter maps it to 0.0.
    assert_row("asinh(col(0))", &[b("pos_1e155")], POS_ZERO);
    assert_row("asinh(col(0))", &[b("pos_1_35e154")], POS_ZERO);
    // Just below the overflow (Appendix B values; the bits are recorded at base).
    assert_row("asinh(col(0))", &[b("pos_1e154")], 0x4076_34a8_f755_a77b);
    assert_row("asinh(col(0))", &[b("pos_1_34e154")], 0x4076_3957_bdaa_ef88);
    assert_row("asinh(col(0))", &[b("pos_one")], 0x3fec_3436_6179_d426);
}

#[test]
fn erf_of_negative_zero_is_positive_zero() {
    // CURRENT SEMANTICS -- changing this is a versioned decision (workplan WP-0.5 / OQ-7)
    // V-12: the sign factor of the rational approximation is 0.0 at +-0.0, so erf(-0.0) is +0.0.
    assert_row("erf(col(0))", &[b("neg_zero")], POS_ZERO);
}

#[test]
fn empty_row_min_and_max_are_positive_zero() {
    // CURRENT SEMANTICS -- changing this is a versioned decision (workplan WP-0.5 / OQ-7)
    // The folds start at +-inf, which the output filter maps to +0.0.
    assert_row("row_min", &[], POS_ZERO);
    assert_row("row_max", &[], POS_ZERO);
}

#[test]
fn empty_row_sums_are_negative_zero() {
    // CURRENT SEMANTICS -- changing this is a versioned decision (workplan WP-0.5 / OQ-7)
    // Recorded at base, correcting the WP-0.5 table's "0.0": an empty f64 sum is -0.0 in this
    // toolchain, the mean divides it by max(len, 1), the root keeps the sign, and the output
    // filter keeps -0.0.
    for source in ["row_mean", "row_sum", "row_l1", "row_l2", "row_std"] {
        assert_row(source, &[], NEG_ZERO);
    }
}

#[test]
fn out_of_range_column_is_an_error_and_a_zero_stage_output() {
    // CURRENT SEMANTICS -- changing this is a versioned decision (workplan WP-0.5 / OQ-7)
    assert_eq!(row_error("col(7)", &[1.0]), "column index 7 out of bounds");
    assert_eq!(
        stage_output("(derive expr=\"col(7)\")", &[vec![1.0]]),
        [[POS_ZERO]]
    );
    assert_eq!(
        stage_output("(map expr=\"col(7)\")", &[vec![1.0]]),
        [[POS_ZERO]]
    );
}

#[test]
fn x_in_row_context_is_an_error_and_a_zero_derive_output() {
    // CURRENT SEMANTICS -- changing this is a versioned decision (workplan WP-0.5 / OQ-7)
    assert_eq!(
        row_error("x", &[1.0]),
        "variable x is only valid in map expressions"
    );
    assert_eq!(
        stage_output("(derive expr=\"x\")", &[vec![1.0]]),
        [[POS_ZERO]]
    );
    // In a map stage x is each element.
    assert_eq!(
        stage_output("(map expr=\"x\")", &[vec![2.0]]),
        [[2.0f64.to_bits()]]
    );
}

#[test]
fn clip_with_two_nan_bounds_is_a_typed_error() {
    // CURRENT SEMANTICS -- changing this is a versioned decision (workplan WP-0.5 / OQ-7)
    // `nan` is not a literal, so the workplan's `clip(col(0), nan, nan)` is written three ways.
    assert!(MathExpr::parse("clip(col(0), nan, nan)").is_err());
    let from_row = MathExpr::parse("clip(col(0), col(1), col(2))").unwrap();
    let error = from_row
        .eval_row(&[1.0, b("nan_quiet"), b("nan_payload")])
        .expect_err("an error");
    assert_eq!(
        error.downcast_ref::<MathEvalError>(),
        Some(&MathEvalError::InvalidClipBounds)
    );
    assert_invalid_clip_bounds(&MathExpr::parse("clip(1,(-1)^0.5,(-1)^0.5)").unwrap());
    let nan = Expr::Const(b("nan_quiet"));
    let constructed = MathExpr::from_ast(Expr::Call(
        Func::Clip,
        vec![Expr::Var(Var::Col(0)), nan.clone(), nan],
    ))
    .unwrap();
    assert_invalid_clip_bounds(&constructed);
    // The whole stage output is 0.0.
    assert_eq!(
        stage_output("(derive expr=\"clip(1,(-1)^0.5,(-1)^0.5)\")", &[vec![1.0]]),
        [[POS_ZERO]]
    );
    // One NaN bound keeps the legacy minNum/maxNum reading: clip(1, NaN, 3) is 3.
    assert_row("clip(1,(-1)^0.5,3)", &[], 3.0f64.to_bits());
}

#[test]
fn the_output_filter_zeroes_non_finite_values_and_clamps_magnitude() {
    // CURRENT SEMANTICS -- changing this is a versioned decision (workplan WP-0.5 / OQ-7)
    assert_row("col(0)", &[b("pos_inf")], POS_ZERO);
    assert_row("col(0)", &[b("neg_inf")], POS_ZERO);
    assert_row("col(0)", &[b("nan_quiet")], POS_ZERO);
    assert_row("col(0)", &[b("pos_1e155")], MAX_ABS);
    assert_row("col(0)", &[b("neg_1e155")], MAX_ABS | NEG_ZERO);
    assert_row("col(0)", &[b("neg_zero")], NEG_ZERO);
}

#[test]
fn literal_parsing_quirks() {
    // CURRENT SEMANTICS -- changing this is a versioned decision (workplan WP-0.5 / OQ-7)
    // Unary minus binds tighter than ^, so -2^2 is (-2)^2.
    assert_row("-2^2", &[], 4.0f64.to_bits());
    assert!(matches!(
        MathExpr::parse("-2^2").unwrap().ast(),
        Expr::Binary(BinOp::Pow, _, _)
    ));
    // The exponent is clamped to [-8, 8].
    assert_row("2^100", &[], 256.0f64.to_bits());
    assert_row("col(0)^-9", &[2.0], 0.003_906_25f64.to_bits());
    // Non-finite literals are refused.
    assert!(MathExpr::parse("1e309").is_err());
}

#[test]
fn step_is_strict_and_nan_compares_false() {
    // CURRENT SEMANTICS -- changing this is a versioned decision (workplan WP-0.5 / OQ-7)
    assert_row("step(col(0), col(1))", &[1.0, 1.0], POS_ZERO);
    assert_row("step(col(0), col(1))", &[b("nan_quiet"), 0.0], POS_ZERO);
    assert_row("sign(col(0))", &[b("nan_quiet")], POS_ZERO);
}
