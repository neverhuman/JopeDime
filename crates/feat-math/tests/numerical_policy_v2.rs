//! First B07 numerical prerequisite. Helper parity does not admit complete
//! programs, fitted models, recipes or shipped generation under policy v2.
use feat_math::kernel::numerical_policy::{asinh_v2, reduce_v2, Reduction};
use feat_math::kernel::numerical_policy::{emit_helpers, NumericalLanguage, KERNEL_ID, POLICY_ID};
use feat_math::math_expr::{FaultSet, MathExpr};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::process::Command;

#[derive(Deserialize)]
struct Corpus {
    asinh: Vec<ScalarCase>,
    rows: Vec<RowCase>,
    mean_cancellation: Vec<MeanCase>,
}

#[derive(Deserialize)]
struct ScalarCase {
    input_bits: String,
    oracle_bits: String,
    fault_bits: u32,
}

#[derive(Deserialize)]
struct Expected {
    oracle_bits: String,
    fault_bits: u32,
}

#[derive(Deserialize)]
struct RowCase {
    input_bits: Vec<String>,
    expected: BTreeMap<String, Expected>,
}

#[derive(Deserialize)]
struct MeanCase {
    input_bits: Vec<String>,
    oracle_bits: String,
    fault_bits: u32,
}

fn corpus() -> Corpus {
    serde_json::from_str(include_str!(
        "fixtures/math_boundary/v2_numerical_policy.json"
    ))
    .unwrap()
}

fn value(bits: &str) -> f64 {
    f64::from_bits(u64::from_str_radix(bits, 16).unwrap())
}

fn reductions() -> [(Reduction, &'static str); 4] {
    [
        (Reduction::Sum, "sum"),
        (Reduction::L1, "l1"),
        (Reduction::Mean, "mean"),
        (Reduction::L2, "l2"),
    ]
}

fn check_oracle(actual: f64, expected: f64, context: &str) {
    assert!(actual.is_finite(), "{context}: nonfinite output");
    if expected == 0.0 {
        assert_eq!(actual.to_bits(), expected.to_bits(), "{context}: zero bits");
    } else if expected.is_normal() {
        assert_eq!(
            actual.is_sign_negative(),
            expected.is_sign_negative(),
            "{context}: sign"
        );
        assert!(
            actual.abs().to_bits().abs_diff(expected.abs().to_bits()) <= 2,
            "{context}: actual={actual:?}, oracle={expected:?}"
        );
    } else {
        assert!(
            (actual - expected).abs() <= f64::from_bits(1),
            "{context}: subnormal bound"
        );
    }
}

mod emitted_matrix;
mod oracle_cases;
mod reduction_cases;
