//! Fault reporting of the scalar evaluator (kernel2 WP-1.1 to WP-1.3).
//!
//! - The fault classes: bits, names and severities are append-only (WP-1.1).
//! - The fault tables of WP-1.2 and WP-1.8, row by row, plus a case for every class, for the
//!   boundaries of each guard and for the order in which faults accumulate, through the
//!   interpreter and the compiled form. Values are compared with the legacy entry points, and
//!   with recorded bits where the value is free of libm.
//! - On every (expression, row) of the compatibility corpus, and every element of every row, every
//!   traced form (the interpreter's row, strict row and element calls; the compiled row and
//!   element calls; the compiled map) returns the legacy value bit for bit (0.0 for a legacy
//!   error, as the stage fallback computes it); a Hard fault is raised exactly when the legacy
//!   call returns an error, with the class of that error's kind; and the compiled form reports
//!   exactly the interpreter's faults (WP-1.2, WP-1.3).
//! - The same on 10,000 random expressions from chained edits, on every corpus row (WP-1.3).
//!
//! Inputs come from PR 01's compatibility corpus, `tests/fixtures/compat_freeze/
//! mathexpr_values.v1.json`: its rows and its expressions. The values recorded there are never
//! used as expectations here, because they depend on the host's libm; every check compares two
//! evaluations made by this build.
//!
//! Two ignored probes compare two builds rather than two paths of one build. Run them in release,
//! where inlining is on, with `cargo test --release --locked -p feat-math --test fault_trace --
//! --ignored --nocapture`:
//!
//! - `legacy_value_digest` prints one sha256 over every result the legacy entry points return on
//!   the corpus: the interpreter and the compiled form, in row, element and map context. Two
//!   builds return the same legacy values exactly when they print the same digest.
//! - `legacy_eval_row_throughput` times the legacy entry points on the corpus.

use feat_math::math_expr::{
    CompiledMathExpr, ExpressionScope, FaultClass, FaultSet, FaultSeverity, MathEvalError, MathExpr,
};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::hint::black_box;
use std::time::Instant;

// ---------------------------------------------------------------------------------------------
// The fault classes (WP-1.1)
// ---------------------------------------------------------------------------------------------

/// The append-only table: bit, class, stable name and severity.
#[rustfmt::skip]
const CLASSES: [(u32, FaultSet, &str, FaultSeverity); 17] = [
    (0, FaultSet::DIV_NEAR_ZERO, "DIV_NEAR_ZERO", FaultSeverity::Domain),
    (1, FaultSet::LOG_DOMAIN, "LOG_DOMAIN", FaultSeverity::Domain),
    (2, FaultSet::LOG1P_DOMAIN, "LOG1P_DOMAIN", FaultSeverity::Domain),
    (3, FaultSet::SQRT_PROJECTED, "SQRT_PROJECTED", FaultSeverity::Domain),
    (4, FaultSet::POW_NONFINITE, "POW_NONFINITE", FaultSeverity::Domain),
    (5, FaultSet::POW_EXP_CLAMPED, "POW_EXP_CLAMPED", FaultSeverity::Saturation),
    (6, FaultSet::EXP_ARG_CLAMPED, "EXP_ARG_CLAMPED", FaultSeverity::Saturation),
    (7, FaultSet::SIGMOID_ARG_CLAMPED, "SIGMOID_ARG_CLAMPED", FaultSeverity::Saturation),
    (8, FaultSet::NONFINITE_SANITIZED, "NONFINITE_SANITIZED", FaultSeverity::Domain),
    (9, FaultSet::MAGNITUDE_CLAMPED, "MAGNITUDE_CLAMPED", FaultSeverity::Saturation),
    (10, FaultSet::INVALID_CLIP_BOUNDS, "INVALID_CLIP_BOUNDS", FaultSeverity::Hard),
    (11, FaultSet::COLUMN_OOB, "COLUMN_OOB", FaultSeverity::Hard),
    (12, FaultSet::ILLEGAL_SCOPE, "ILLEGAL_SCOPE", FaultSeverity::Hard),
    (13, FaultSet::NONFINITE_INPUT, "NONFINITE_INPUT", FaultSeverity::Domain),
    (14, FaultSet::EMPTY_ROW_REDUCTION, "EMPTY_ROW_REDUCTION", FaultSeverity::Domain),
    (15, FaultSet::COMPILE_FAILED, "COMPILE_FAILED", FaultSeverity::Hard),
    (16, FaultSet::PROBABILITY_CLAMPED, "PROBABILITY_CLAMPED", FaultSeverity::Saturation),
];

#[test]
fn fault_bits_names_and_severities_are_append_only() {
    // A slice, not an array: an appended class must not change the constant's type.
    let all: &'static [FaultClass] = FaultClass::ALL;
    assert_eq!(all.len(), CLASSES.len());
    for (class, (bit, set, name, severity)) in all.iter().copied().zip(CLASSES) {
        assert_eq!(set.bits(), 1 << bit, "{name}");
        assert_eq!(class.index(), bit as usize, "{name}");
        assert_eq!(class.set(), set, "{name}");
        assert_eq!(FaultSet::from(class), set, "{name}");
        assert_eq!(class.name(), name);
        assert_eq!(class.to_string(), name);
        assert_eq!(class.severity(), severity, "{name}");
        assert_eq!(set.len(), 1, "{name}");
        assert_eq!(set.iter().collect::<Vec<_>>(), [class], "{name}");
        assert_eq!(set.to_string(), format!("{{{name}}}"));
        assert_eq!(set.severity(), Some(severity), "{name}");
    }
    assert_eq!(FaultSet::ALL.bits(), 0x1ffff);
    // Each class has exactly one severity: the three masks partition the classes.
    assert_eq!(FaultSeverity::Hard.classes().bits(), 0x9c00);
    assert_eq!(FaultSeverity::Domain.classes().bits(), 0x611f);
    assert_eq!(FaultSeverity::Saturation.classes().bits(), 0x102e0);
    assert_eq!(FaultSeverity::Saturation.name(), "saturation");
    assert_eq!(FaultSeverity::Domain.name(), "domain");
    assert_eq!(FaultSeverity::Hard.name(), "hard");
    assert_eq!(FaultSeverity::Hard.to_string(), "hard");
    assert!(FaultSeverity::Saturation < FaultSeverity::Domain);
    assert!(FaultSeverity::Domain < FaultSeverity::Hard);
    assert_eq!(std::mem::size_of::<FaultSet>(), 4);
}

#[test]
fn fault_sets_combine_as_sets() {
    let div = FaultSet::DIV_NEAR_ZERO;
    let sanitized = FaultSet::NONFINITE_SANITIZED;
    let both = div.union(sanitized);
    assert_eq!(both.bits(), 0x101);
    assert_eq!(both, sanitized.union(div));
    assert_eq!(div.union(div), div);
    assert!(both.contains(div) && both.contains(sanitized) && both.contains(both));
    assert!(both.contains(FaultSet::EMPTY));
    assert!(!div.contains(both) && !div.contains(sanitized));
    assert!(both.intersects(div) && !div.intersects(sanitized));
    assert!(FaultSet::EMPTY.is_empty() && !both.is_empty());
    assert_eq!(FaultSet::default(), FaultSet::EMPTY);
    assert_eq!(both.len(), 2);
    assert_eq!(
        both.iter().collect::<Vec<_>>(),
        [FaultClass::DivNearZero, FaultClass::NonfiniteSanitized]
    );
    assert_eq!(
        [FaultClass::NonfiniteSanitized, FaultClass::DivNearZero]
            .into_iter()
            .collect::<FaultSet>(),
        both
    );
    assert_eq!(both.to_string(), "{DIV_NEAR_ZERO, NONFINITE_SANITIZED}");
    assert_eq!(
        format!("{both:?}"),
        "FaultSet({DIV_NEAR_ZERO, NONFINITE_SANITIZED})"
    );
    assert_eq!(FaultSet::EMPTY.to_string(), "{}");
    assert_eq!(FaultSet::EMPTY.severity(), None);
    assert_eq!(both.severity(), Some(FaultSeverity::Domain));
    assert_eq!(
        both.union(FaultSet::COLUMN_OOB).severity(),
        Some(FaultSeverity::Hard)
    );
    assert_eq!(FaultSet::from_bits(0x101), Some(both));
    assert_eq!(FaultSet::from_bits(0), Some(FaultSet::EMPTY));
    assert_eq!(
        FaultSet::from_bits(FaultSet::ALL.bits()),
        Some(FaultSet::ALL)
    );
    assert_eq!(FaultSet::from_bits(!FaultSet::ALL.bits()), None);
    assert_eq!(FaultSet::from_bits(u32::MAX), None);
}

#[test]
fn fault_names_pass_the_shipped_text_lint() {
    let severities = [
        FaultSeverity::Saturation,
        FaultSeverity::Domain,
        FaultSeverity::Hard,
    ];
    let names = FaultClass::ALL
        .iter()
        .map(|class| class.name())
        .chain(severities.iter().map(|severity| severity.name()));
    for name in names {
        let hits = feat_math::transpile::lint_shipped_text(name);
        assert!(hits.is_empty(), "{name}: {hits:?}");
    }
}

// ---------------------------------------------------------------------------------------------
// The fault tables (WP-1.2, WP-1.8)
// ---------------------------------------------------------------------------------------------

const POS_ZERO: u64 = 0x0000_0000_0000_0000;
const NEG_ZERO: u64 = 0x8000_0000_0000_0000;
const MAX_ABS: u64 = 0x426d_1a94_a200_0000; // 1e12, the output filter's clamp

fn union_of(sets: &[FaultSet]) -> FaultSet {
    sets.iter()
        .fold(FaultSet::EMPTY, |all, set| all.union(*set))
}

/// One evaluation and what it must report.
struct Case {
    source: &'static str,
    row: Vec<f64>,
    /// `None` is row context (`eval_row`); `Some(x)` is element context (`eval_element`).
    x: Option<f64>,
    /// The value's bits where they are free of libm; `None` compares with the legacy value only.
    bits: Option<u64>,
    faults: FaultSet,
}

impl Case {
    /// The row, and the element in element context, for failure messages.
    fn row_text(&self) -> String {
        match self.x {
            None => format!("{:?}", self.row),
            Some(x) => format!("{:?} at x = {x:?}", self.row),
        }
    }
}

fn row_case(source: &'static str, row: &[f64], bits: Option<u64>, faults: FaultSet) -> Case {
    Case {
        source,
        row: row.to_vec(),
        x: None,
        bits,
        faults,
    }
}

fn element_case(
    source: &'static str,
    x: f64,
    row: &[f64],
    bits: Option<u64>,
    faults: FaultSet,
) -> Case {
    Case {
        source,
        row: row.to_vec(),
        x: Some(x),
        bits,
        faults,
    }
}

/// The WP-1.2 table (workplan lines 591-609), in its order.
fn wp_1_2_table() -> Vec<Case> {
    use feat_math::math_expr::FaultSet as F;
    vec![
        row_case("1/col(0)", &[0.0], Some(POS_ZERO), F::DIV_NEAR_ZERO),
        row_case("0/col(0)", &[1.0], Some(POS_ZERO), F::EMPTY),
        row_case("log(col(0))", &[-1.0], Some(POS_ZERO), F::LOG_DOMAIN),
        row_case("log(col(0))", &[1.0], Some(POS_ZERO), F::EMPTY),
        row_case("log1p(col(0))", &[-1.0], Some(POS_ZERO), F::LOG1P_DOMAIN),
        row_case("sqrt(col(0))", &[-1.0], Some(POS_ZERO), F::SQRT_PROJECTED),
        // Recorded at the base: +0.0 (the workplan guessed -0.0), and not a fault.
        row_case("sqrt(col(0))", &[-0.0], Some(POS_ZERO), F::EMPTY),
        row_case("exp(col(0))", &[27.0], None, F::EMPTY),
        row_case("exp(col(0))", &[28.0], Some(MAX_ABS), F::MAGNITUDE_CLAMPED),
        row_case(
            "exp(col(0))",
            &[31.0],
            Some(MAX_ABS),
            union_of(&[F::EXP_ARG_CLAMPED, F::MAGNITUDE_CLAMPED]),
        ),
        row_case(
            "asinh(col(0))",
            &[1e155],
            Some(POS_ZERO),
            F::NONFINITE_SANITIZED,
        ),
        row_case(
            "col(0)*col(0)",
            &[1e200],
            Some(POS_ZERO),
            F::NONFINITE_SANITIZED,
        ),
        row_case(
            "col(0)",
            &[f64::NAN],
            Some(POS_ZERO),
            union_of(&[F::NONFINITE_INPUT, F::NONFINITE_SANITIZED]),
        ),
        row_case("col(3)", &[1.0], Some(POS_ZERO), F::COLUMN_OOB),
        row_case("x", &[1.0], Some(POS_ZERO), F::ILLEGAL_SCOPE),
        // The workplan lists only INVALID_CLIP_BOUNDS. Reading the NaN bound col(1) raises
        // NONFINITE_INPUT first (WP-1.2 step 5), and the faults raised before a Hard error stay.
        row_case(
            "clip(col(0), col(1), col(1))",
            &[1.0, f64::NAN],
            Some(POS_ZERO),
            union_of(&[F::NONFINITE_INPUT, F::INVALID_CLIP_BOUNDS]),
        ),
        row_case(
            "row_min",
            &[],
            Some(POS_ZERO),
            union_of(&[F::EMPTY_ROW_REDUCTION, F::NONFINITE_SANITIZED]),
        ),
    ]
}

/// The WP-1.8 table (workplan lines 774-779), then the other reductions over an empty row and
/// over a row with a non-finite cell. An empty sum is -0.0 with this toolchain (PR 01's finding),
/// so "0.0" in the workplan is a value, not a sign.
fn wp_1_8_table() -> Vec<Case> {
    use feat_math::math_expr::FaultSet as F;
    let sanitized_input = union_of(&[F::NONFINITE_INPUT, F::NONFINITE_SANITIZED]);
    vec![
        row_case("row_mean", &[], Some(NEG_ZERO), F::EMPTY_ROW_REDUCTION),
        row_case("row_sum", &[], Some(NEG_ZERO), F::EMPTY),
        row_case("row_std", &[5.0], Some(POS_ZERO), F::EMPTY),
        row_case(
            "row_mean",
            &[f64::NAN, 1.0],
            Some(POS_ZERO),
            sanitized_input,
        ),
        row_case(
            "row_max",
            &[f64::NEG_INFINITY],
            Some(POS_ZERO),
            sanitized_input,
        ),
        // Undefined over an empty row: mean, std, min and max. True identities: sum, l1, l2.
        row_case("row_std", &[], Some(NEG_ZERO), F::EMPTY_ROW_REDUCTION),
        row_case(
            "row_max",
            &[],
            Some(POS_ZERO),
            union_of(&[F::EMPTY_ROW_REDUCTION, F::NONFINITE_SANITIZED]),
        ),
        row_case("row_l1", &[], Some(NEG_ZERO), F::EMPTY),
        row_case("row_l2", &[], Some(NEG_ZERO), F::EMPTY),
        // Any non-finite cell, for every reduction; min absorbs the NaN but keeps the fault.
        row_case(
            "row_sum",
            &[1.0, f64::INFINITY],
            Some(POS_ZERO),
            sanitized_input,
        ),
        row_case("row_l1", &[f64::NAN, 1.0], Some(POS_ZERO), sanitized_input),
        row_case(
            "row_l2",
            &[1.0, f64::NEG_INFINITY],
            Some(POS_ZERO),
            sanitized_input,
        ),
        row_case("row_std", &[1.0, f64::NAN], Some(POS_ZERO), sanitized_input),
        row_case(
            "row_min",
            &[1.0, f64::NAN],
            Some(1f64.to_bits()),
            F::NONFINITE_INPUT,
        ),
    ]
}

/// Every class the evaluator raises besides those above, the boundaries of its guard, the
/// branches that absorb a NaN, and the order in which faults accumulate.
fn class_cases() -> Vec<Case> {
    use feat_math::math_expr::FaultSet as F;
    let sanitized_input = union_of(&[F::NONFINITE_INPUT, F::NONFINITE_SANITIZED]);
    let generated_nonfinite = union_of(&[F::POW_NONFINITE, F::NONFINITE_SANITIZED]);
    vec![
        // DIV_NEAR_ZERO is strictly below 1e-12, and 1/1e-12 is exactly 1e12, not clamped.
        row_case("1/col(0)", &[1e-12], Some(MAX_ABS), F::EMPTY),
        row_case("1/col(0)", &[-0.0], Some(POS_ZERO), F::DIV_NEAR_ZERO),
        // LOG_DOMAIN covers tiny positives up to 1e-300; a NaN argument is not in its branch.
        row_case("log(col(0))", &[1e-300], Some(POS_ZERO), F::LOG_DOMAIN),
        row_case("log(col(0))", &[f64::NAN], Some(POS_ZERO), sanitized_input),
        // SQRT_PROJECTED: max(0.0) replaces a NaN argument too.
        row_case(
            "sqrt(col(0))",
            &[f64::NAN],
            Some(POS_ZERO),
            union_of(&[F::NONFINITE_INPUT, F::SQRT_PROJECTED]),
        ),
        // POW_NONFINITE: a non-finite power of a finite base and a finite (clamped) exponent.
        row_case("col(0)^0.5", &[-1.0], Some(POS_ZERO), generated_nonfinite),
        row_case(
            "col(0)^col(1)",
            &[0.0, -1.0],
            Some(POS_ZERO),
            generated_nonfinite,
        ),
        row_case(
            "col(0)^col(1)",
            &[1e200, 8.0],
            Some(POS_ZERO),
            generated_nonfinite,
        ),
        row_case(
            "col(0)^col(1)",
            &[2.0, f64::NAN],
            Some(POS_ZERO),
            sanitized_input,
        ),
        // POW_EXP_CLAMPED: outside [-8, 8] only. An infinite exponent is clamped to 8.
        row_case(
            "2^col(0)",
            &[100.0],
            Some(256f64.to_bits()),
            F::POW_EXP_CLAMPED,
        ),
        row_case("2^col(0)", &[8.0], Some(256f64.to_bits()), F::EMPTY),
        row_case(
            "col(0)^col(1)",
            &[1e200, f64::INFINITY],
            Some(POS_ZERO),
            union_of(&[
                F::NONFINITE_INPUT,
                F::POW_EXP_CLAMPED,
                F::POW_NONFINITE,
                F::NONFINITE_SANITIZED,
            ]),
        ),
        // EXP_ARG_CLAMPED and SIGMOID_ARG_CLAMPED: outside [-30, 30] only.
        row_case("exp(col(0))", &[30.0], Some(MAX_ABS), F::MAGNITUDE_CLAMPED),
        row_case("exp(col(0))", &[-31.0], None, F::EXP_ARG_CLAMPED),
        row_case("sigmoid(col(0))", &[31.0], None, F::SIGMOID_ARG_CLAMPED),
        row_case("sigmoid(col(0))", &[-30.0], None, F::EMPTY),
        // MAGNITUDE_CLAMPED: strictly above 1e12 in magnitude, either sign.
        row_case(
            "col(0)",
            &[-1e13],
            Some(MAX_ABS | NEG_ZERO),
            F::MAGNITUDE_CLAMPED,
        ),
        // NONFINITE_INPUT: x as well as a column.
        element_case(
            "x",
            f64::INFINITY,
            &[f64::INFINITY],
            Some(POS_ZERO),
            sanitized_input,
        ),
        element_case("x + col(0)", 2.0, &[2.0], Some(4f64.to_bits()), F::EMPTY),
        // A branch that absorbs a NaN input still leaves the input fault.
        row_case(
            "min(col(0), col(1))",
            &[f64::NAN, 1.0],
            Some(1f64.to_bits()),
            F::NONFINITE_INPUT,
        ),
        row_case(
            "step(col(0), 0)",
            &[f64::NAN],
            Some(POS_ZERO),
            F::NONFINITE_INPUT,
        ),
        // A NaN generated at a guarded site is reported there, even when a parent absorbs it...
        row_case(
            "min(col(0)^0.5, 1)",
            &[-1.0],
            Some(1f64.to_bits()),
            F::POW_NONFINITE,
        ),
        // ...but a product has no guard: its overflow, absorbed by min, raises nothing (see
        // "Not reported" in docs/internal/kernel2/math-domain-table.md).
        row_case(
            "min(col(0)*col(0) - col(0)*col(0), 1)",
            &[1e200],
            Some(1f64.to_bits()),
            F::EMPTY,
        ),
        // The same NaN absorbed by ^ with a zero exponent: pow(NaN, 0) is 1.0, which is finite,
        // so neither ^ test fires.
        row_case(
            "(col(0)*col(0) - col(0)*col(0))^0",
            &[1e200],
            Some(1f64.to_bits()),
            F::EMPTY,
        ),
        // pow(1, NaN) is 1.0 too; a NaN read from the input keeps its input fault.
        row_case(
            "1^col(0)",
            &[f64::NAN],
            Some(1f64.to_bits()),
            F::NONFINITE_INPUT,
        ),
        // The overflowed product as a divisor: 1/inf is +0.0, and the divisor guard tests only
        // |b| < 1e-12.
        row_case("1/(col(0)*col(0))", &[1e200], Some(POS_ZERO), F::EMPTY),
        // Faults accumulate in evaluation order until a Hard error ends the evaluation.
        row_case(
            "1/col(1) + col(5)",
            &[1.0, 0.0],
            Some(POS_ZERO),
            union_of(&[F::DIV_NEAR_ZERO, F::COLUMN_OOB]),
        ),
        row_case(
            "col(5) + 1/col(1)",
            &[1.0, 0.0],
            Some(POS_ZERO),
            F::COLUMN_OOB,
        ),
        // Two NaN bounds from arithmetic, as in PR 01's clip pins; one NaN bound is no error.
        row_case(
            "clip(1,(-1)^0.5,(-1)^0.5)",
            &[],
            Some(POS_ZERO),
            union_of(&[F::POW_NONFINITE, F::INVALID_CLIP_BOUNDS]),
        ),
        row_case(
            "clip(1,(-1)^0.5,3)",
            &[],
            Some(3f64.to_bits()),
            F::POW_NONFINITE,
        ),
        // A row summary read twice reports its faults on each read; the set is the same.
        row_case(
            "row_min + row_min",
            &[],
            Some(POS_ZERO),
            union_of(&[F::EMPTY_ROW_REDUCTION, F::NONFINITE_SANITIZED]),
        ),
    ]
}

fn all_cases() -> Vec<Case> {
    let mut cases = wp_1_2_table();
    cases.extend(wp_1_8_table());
    cases.extend(class_cases());
    cases
}

/// The Hard class of a legacy evaluation error, by the kinds compat_freeze records.
fn hard_class_of(error: &anyhow::Error) -> FaultClass {
    let message = error.to_string();
    if error.downcast_ref::<MathEvalError>() == Some(&MathEvalError::InvalidClipBounds) {
        FaultClass::InvalidClipBounds
    } else if message.starts_with("variable x is only valid") {
        FaultClass::IllegalScope
    } else if message.starts_with("column index") {
        FaultClass::ColumnOob
    } else {
        panic!("an evaluation error of no recorded kind: {message}")
    }
}

/// Where a check is, for failure messages. It is formatted only when a check fails.
#[derive(Clone, Copy)]
struct At<'a> {
    expr: &'a str,
    row: &'a str,
    element: Option<usize>,
}

impl<'a> At<'a> {
    fn new(expr: &'a str, row: &'a str) -> Self {
        At {
            expr,
            row,
            element: None,
        }
    }

    fn element(self, j: usize) -> Self {
        At {
            element: Some(j),
            ..self
        }
    }
}

impl fmt::Display for At<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} on {}", self.expr, self.row)?;
        match self.element {
            Some(j) => write!(f, " at element {j}"),
            None => Ok(()),
        }
    }
}

/// A traced result as bits, so that NaN and the sign of zero compare exactly.
fn key((value, faults): (f64, FaultSet)) -> (u64, FaultSet) {
    (value.to_bits(), faults)
}

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|value| value.to_bits()).collect()
}

/// A traced result against the legacy result of the same call: the legacy value bit for bit
/// with no Hard fault, or, for a legacy error, 0.0 with exactly one Hard class, of that error's
/// kind.
fn check_against_legacy(
    at: At<'_>,
    legacy: &anyhow::Result<f64>,
    (value, faults): (f64, FaultSet),
) {
    match legacy {
        Ok(expected) => {
            assert_eq!(
                value.to_bits(),
                expected.to_bits(),
                "{at}: traced {value:?}, legacy {expected:?}"
            );
            assert_ne!(
                faults.severity(),
                Some(FaultSeverity::Hard),
                "{at}: a legacy value with the faults {faults}"
            );
        }
        Err(error) => {
            assert_eq!(value.to_bits(), POS_ZERO, "{at}: a legacy error is 0.0");
            let hard: Vec<FaultClass> = faults
                .iter()
                .filter(|class| class.severity() == FaultSeverity::Hard)
                .collect();
            assert_eq!(
                hard,
                [hard_class_of(error)],
                "{at}: the legacy error {error:#} with the faults {faults}"
            );
        }
    }
}

/// `eval_row_traced_strict`: the compat-shape value and faults, or exactly the legacy error.
fn check_strict(
    at: At<'_>,
    expr: &MathExpr,
    row: &[f64],
    legacy: &anyhow::Result<f64>,
    traced: (f64, FaultSet),
) {
    match (expr.eval_row_traced_strict(row), legacy) {
        (Ok(strict), Ok(_)) => assert_eq!(key(strict), key(traced), "{at}: strict"),
        (Err(strict), Err(legacy)) => {
            assert_eq!(strict.to_string(), legacy.to_string(), "{at}: strict");
            assert_eq!(
                strict.downcast_ref::<MathEvalError>(),
                legacy.downcast_ref::<MathEvalError>(),
                "{at}: strict error type"
            );
        }
        (strict, legacy) => panic!("{at}: strict {strict:?}, legacy {legacy:?}"),
    }
}

/// Counts for the record: evaluations, those with any fault, legacy errors by their Hard class,
/// and evaluations per class.
#[derive(Default)]
struct Tally {
    evaluations: usize,
    faulted: usize,
    errors: BTreeMap<&'static str, usize>,
    classes: BTreeMap<&'static str, usize>,
}

impl Tally {
    fn add(&mut self, legacy: &anyhow::Result<f64>, faults: FaultSet) {
        self.evaluations += 1;
        self.faulted += usize::from(!faults.is_empty());
        if let Err(error) = legacy {
            *self.errors.entry(hard_class_of(error).name()).or_default() += 1;
        }
        for class in faults.iter() {
            *self.classes.entry(class.name()).or_default() += 1;
        }
    }
}

impl fmt::Display for Tally {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "evaluations={} faulted={} legacy_errors={:?} classes={:?}",
            self.evaluations, self.faulted, self.errors, self.classes
        )
    }
}

/// The interpreter's faults in row context and in element context.
#[derive(Default)]
struct Tallies {
    rows: Tally,
    elements: Tally,
}

/// Every traced entry point on one row, against the legacy entry points and against each other:
/// the interpreter in row context (compat shape and strict) and at each element, the compiled
/// form in row context and at each element, and the compiled map of the row.
fn check_every_form(
    at: At<'_>,
    expr: &MathExpr,
    compiled: &CompiledMathExpr,
    cells: &[f64],
    tallies: &mut Tallies,
) {
    let legacy = expr.eval_row(cells);
    let traced = expr.eval_row_traced(cells);
    check_against_legacy(at, &legacy, traced);
    check_strict(at, expr, cells, &legacy, traced);
    assert_eq!(
        key(compiled.eval_row_traced(cells)),
        key(traced),
        "{at}: compiled eval_row_traced"
    );
    tallies.rows.add(&legacy, traced.1);
    let (values, faults) = compiled.map_row_traced(cells);
    let legacy_map = compiled.map_row(cells);
    assert_eq!(
        bits(&values),
        bits(&legacy_map),
        "{at}: map_row_traced values"
    );
    assert_eq!(faults.len(), cells.len(), "{at}: map_row_traced cells");
    for (j, &x) in cells.iter().enumerate() {
        let at = at.element(j);
        let legacy = expr.eval_element(x, cells);
        let traced = expr.eval_element_traced(x, cells);
        check_against_legacy(at, &legacy, traced);
        assert_eq!(
            key(compiled.eval_element_traced(x, cells)),
            key(traced),
            "{at}: compiled eval_element_traced"
        );
        assert_eq!(
            key((values[j], faults[j])),
            key(traced),
            "{at}: map_row_traced cell"
        );
        tallies.elements.add(&legacy, traced.1);
    }
}

/// One case through the interpreter.
fn check_interpreter(case: &Case) {
    let row_text = case.row_text();
    let at = At::new(case.source, &row_text);
    let expr = MathExpr::parse(case.source).unwrap_or_else(|error| panic!("{at}: {error}"));
    let (legacy, traced) = match case.x {
        None => (expr.eval_row(&case.row), expr.eval_row_traced(&case.row)),
        Some(x) => (
            expr.eval_element(x, &case.row),
            expr.eval_element_traced(x, &case.row),
        ),
    };
    check_against_legacy(at, &legacy, traced);
    if let Some(bits) = case.bits {
        assert_eq!(
            traced.0.to_bits(),
            bits,
            "{at}: expected {:?}, got {:?}",
            f64::from_bits(bits),
            traced.0
        );
    }
    assert_eq!(traced.1, case.faults, "{at}");
    if case.x.is_none() {
        check_strict(at, &expr, &case.row, &legacy, traced);
    }
}

/// One case through the compiled form, then every traced form on the case's row.
fn check_compiled(case: &Case) {
    let row_text = case.row_text();
    let at = At::new(case.source, &row_text);
    let expr = MathExpr::parse(case.source).unwrap_or_else(|error| panic!("{at}: {error}"));
    let compiled = expr
        .compile()
        .unwrap_or_else(|error| panic!("{at}: {error}"));
    let traced = match case.x {
        None => compiled.eval_row_traced(&case.row),
        Some(x) => compiled.eval_element_traced(x, &case.row),
    };
    if let Some(bits) = case.bits {
        assert_eq!(traced.0.to_bits(), bits, "{at}: compiled value");
    }
    assert_eq!(traced.1, case.faults, "{at}: compiled");
    check_every_form(at, &expr, &compiled, &case.row, &mut Tallies::default());
}

#[test]
fn the_interpreter_reports_the_fault_tables() {
    for case in all_cases() {
        check_interpreter(&case);
    }
}

#[test]
fn the_compiled_form_reports_the_fault_tables() {
    for case in all_cases() {
        check_compiled(&case);
    }
}

#[test]
fn a_call_with_the_wrong_arity_from_a_deserialized_ast_is_a_compile_failure() {
    // parse and from_ast refuse this; serde decoding is a compatibility interface and does not.
    let expr: MathExpr = serde_json::from_value(serde_json::json!({
        "root": {"Call": ["Sin", [{"Const": 1.0}, {"Const": 2.0}]]}
    }))
    .expect("the historical AST encoding");
    let legacy = expr.eval_row(&[]).expect_err("a legacy arity error");
    assert_eq!(legacy.to_string(), "Sin received 2 arguments");
    assert_eq!(
        key(expr.eval_row_traced(&[])),
        (POS_ZERO, FaultSet::COMPILE_FAILED)
    );
    let strict = expr
        .eval_row_traced_strict(&[])
        .expect_err("the legacy error");
    assert_eq!(strict.to_string(), legacy.to_string());
    assert_eq!(
        key(expr.eval_element_traced(1.0, &[1.0])),
        (POS_ZERO, FaultSet::COMPILE_FAILED)
    );
    // The stage refuses the same AST at compile time, which it reports as COMPILE_FAILED.
    assert!(expr.compile().is_err());
}

#[test]
fn map_cells_report_exactly_what_their_own_evaluation_reaches() {
    use feat_math::math_expr::FaultSet as F;
    // In map_row the element-independent instructions run once per row, before any element; in
    // each of these the row pass and an element's own evaluation stop at different points.
    let cases = [
        // The row pass fails at col(9), but each element divides by x - x first.
        (
            "1/(x-x)+col(9)",
            vec![1.0],
            vec![union_of(&[F::DIV_NEAR_ZERO, F::COLUMN_OOB])],
        ),
        // Element -1 fails at the clip, before 1/col(0), which the row pass evaluates for all.
        (
            "clip(1,x^0.5,x^0.5)+1/col(0)",
            vec![0.0, -1.0],
            vec![
                F::DIV_NEAR_ZERO,
                union_of(&[F::POW_NONFINITE, F::INVALID_CLIP_BOUNDS]),
            ],
        ),
        // The row pass fails at c2, but the element fails at the clip first: another Hard class.
        (
            "clip(x,c0,c0) + c2",
            vec![f64::NAN],
            vec![union_of(&[F::NONFINITE_INPUT, F::INVALID_CLIP_BOUNDS])],
        ),
        // An empty row has no cell, so no fault, whatever its row pass raised.
        ("row_min + col(0)", vec![], vec![]),
    ];
    for (source, row, expected) in cases {
        let expr = MathExpr::parse(source).expect("parses");
        let compiled = expr.compile().expect("compiles");
        let (values, faults) = compiled.map_row_traced(&row);
        assert_eq!(faults, expected, "{source} on {row:?}");
        assert_eq!(bits(&values), bits(&compiled.map_row(&row)), "{source}");
        check_every_form(
            At::new(source, &format!("{row:?}")),
            &expr,
            &compiled,
            &row,
            &mut Tallies::default(),
        );
    }
}

#[test]
fn the_expression_forms_and_fault_sets_stay_shareable_across_threads() {
    fn shareable<T: Send + Sync>() {}
    shareable::<MathExpr>();
    shareable::<CompiledMathExpr>();
    shareable::<FaultSet>();
}

// ---------------------------------------------------------------------------------------------
// The compatibility corpus
// ---------------------------------------------------------------------------------------------

/// One corpus row: its label and its cells.
struct Row {
    label: String,
    cells: Vec<f64>,
}

/// PR 01's corpus: every row and every expression source.
struct Corpus {
    rows: Vec<Row>,
    expressions: Vec<String>,
}

fn corpus() -> Corpus {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/compat_freeze/mathexpr_values.v1.json"
    );
    let text = std::fs::read_to_string(path).unwrap_or_else(|error| panic!("{path}: {error}"));
    let file: serde_json::Value =
        serde_json::from_str(&text).expect("mathexpr_values.v1.json parses");
    let rows: Vec<Row> = file["rows"]
        .as_array()
        .expect("rows")
        .iter()
        .map(|row| Row {
            label: row["label"].as_str().expect("row label").to_string(),
            // The empty row is recorded as "": no cells, not one empty token.
            cells: row["values"]
                .as_str()
                .expect("row values")
                .split(' ')
                .filter(|token| !token.is_empty())
                .map(|token| {
                    f64::from_bits(
                        u64::from_str_radix(token, 16)
                            .unwrap_or_else(|error| panic!("{token}: {error}")),
                    )
                })
                .collect(),
        })
        .collect();
    let expressions: Vec<String> = file["expressions"]
        .as_array()
        .expect("expressions")
        .iter()
        .map(|expression| expression["source"].as_str().expect("source").to_string())
        .collect();
    assert!(
        rows.iter().any(|row| row.cells.is_empty()) && !expressions.is_empty(),
        "the corpus has an empty row and at least one expression"
    );
    Corpus { rows, expressions }
}

#[test]
fn every_traced_form_equals_the_legacy_values_on_the_compat_corpus() {
    let corpus = corpus();
    let mut tallies = Tallies::default();
    for source in &corpus.expressions {
        let expr = MathExpr::parse(source).expect("corpus expression parses");
        let compiled = expr.compile().expect("corpus expression compiles");
        for row in &corpus.rows {
            check_every_form(
                At::new(source, &row.label),
                &expr,
                &compiled,
                &row.cells,
                &mut tallies,
            );
        }
    }
    eprintln!(
        "PROPERTY expressions={} rows={}",
        corpus.expressions.len(),
        corpus.rows.len()
    );
    eprintln!("PROPERTY row context: {}", tallies.rows);
    eprintln!("PROPERTY element context: {}", tallies.elements);
    assert!(tallies.rows.evaluations > 0 && tallies.elements.evaluations > 0);
}

// ---------------------------------------------------------------------------------------------
// Random expressions (WP-1.3)
// ---------------------------------------------------------------------------------------------

/// A recombination donor with every row summary, a negation, `x` and column reads: the edit
/// operator's leaves draw only row_mean among the summaries, and it never negates, so without
/// this the summary cache and most reductions would never meet the compiled form.
const DONOR: &str =
    "row_sum - row_l1 + row_l2 * row_min / row_max ^ row_std + row_mean * (-x) - c0 / c2 + sqrt(c1)";

/// Chained random expressions for `seeds`, each checked on every corpus row through every traced
/// form. Each seed edits the parent with `mutate_v1`, or recombines it with `DONOR` through
/// `crossover_v1` at every third seed, within a scope that changes every 10 seeds (element or row
/// context) and every 100 seeds (0, 1, 2, 3 or 8 columns); a parent outside the new scope
/// restarts from `0.1`.
fn random_expressions_agree(seeds: std::ops::Range<u64>) {
    let corpus = corpus();
    let donor = MathExpr::parse(DONOR).expect("the donor parses");
    let restart = MathExpr::parse("0.1").expect("the restart parses");
    let mut parent = restart.clone();
    let mut distinct = BTreeSet::new();
    let mut changed = 0usize;
    let mut tallies = Tallies::default();
    for seed in seeds.clone() {
        let scope = ExpressionScope {
            columns: [0, 1, 2, 3, 8][(seed / 100 % 5) as usize],
            elementwise: seed / 10 % 2 == 0,
            max_nodes: 64,
            max_depth: 16,
        };
        if scope.validate(&parent).is_err() {
            parent = restart.clone();
        }
        let edit = if seed % 3 == 0 {
            parent.crossover_v1(&donor, seed, scope)
        } else {
            parent.mutate_v1(seed, scope)
        }
        .unwrap_or_else(|error| panic!("seed {seed}: {error}"));
        changed += usize::from(edit.changed);
        let expr = edit.expression;
        let compiled = expr
            .compile()
            .unwrap_or_else(|error| panic!("seed {seed}: {error}"));
        let ast = format!("{:?}", expr.ast());
        let name = format!("seed {seed}, {ast}");
        for row in &corpus.rows {
            check_every_form(
                At::new(&name, &row.label),
                &expr,
                &compiled,
                &row.cells,
                &mut tallies,
            );
        }
        distinct.insert(ast);
        parent = expr;
    }
    eprintln!(
        "DIFFERENTIAL seeds={seeds:?} expressions={} changed={changed} distinct={} rows={}",
        seeds.end - seeds.start,
        distinct.len(),
        corpus.rows.len()
    );
    eprintln!("DIFFERENTIAL row context: {}", tallies.rows);
    eprintln!("DIFFERENTIAL element context: {}", tallies.elements);
}

#[test]
fn random_expressions_agree_across_forms_seeds_0_to_2499() {
    random_expressions_agree(0..2_500);
}

#[test]
fn random_expressions_agree_across_forms_seeds_2500_to_4999() {
    random_expressions_agree(2_500..5_000);
}

#[test]
fn random_expressions_agree_across_forms_seeds_5000_to_7499() {
    random_expressions_agree(5_000..7_500);
}

#[test]
fn random_expressions_agree_across_forms_seeds_7500_to_9999() {
    random_expressions_agree(7_500..10_000);
}

// ---------------------------------------------------------------------------------------------
// Probes that compare two builds (ignored; run them in release)
// ---------------------------------------------------------------------------------------------

/// A legacy result as a token: the value's bits, or the error's message.
fn token(result: anyhow::Result<f64>) -> String {
    match result {
        Ok(value) => format!("{:016x}", value.to_bits()),
        Err(error) => format!("ERR:{error}"),
    }
}

#[test]
#[ignore = "release probe: prints a digest of every legacy result on the corpus; compare it between two builds"]
fn legacy_value_digest() {
    let corpus = corpus();
    let mut digest = Sha256::new();
    let mut results = 0usize;
    let mut feed = |token: String| {
        digest.update(token.as_bytes());
        digest.update(b"\n");
        results += 1;
    };
    for source in &corpus.expressions {
        let expr = MathExpr::parse(source).expect("corpus expression parses");
        let compiled = expr.compile().expect("corpus expression compiles");
        for row in &corpus.rows {
            let cells = row.cells.as_slice();
            feed(token(expr.eval_row(cells)));
            feed(token(compiled.eval_row(cells)));
            for &x in cells {
                feed(token(expr.eval_element(x, cells)));
                feed(token(compiled.eval_element(x, cells)));
            }
            for value in compiled.map_row(cells) {
                feed(token(Ok(value)));
            }
        }
    }
    let hex: String = digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    println!(
        "LEGACY-DIGEST sha256={hex} results={results} expressions={} rows={}",
        corpus.expressions.len(),
        corpus.rows.len()
    );
}

/// Minimum and median nanoseconds of `passes` timed runs of `pass`, after five warm-up runs.
fn time_passes(passes: usize, mut pass: impl FnMut() -> u64) -> (u128, u128) {
    for _ in 0..5 {
        black_box(pass());
    }
    let mut nanos: Vec<u128> = (0..passes)
        .map(|_| {
            let start = Instant::now();
            black_box(pass());
            start.elapsed().as_nanos()
        })
        .collect();
    nanos.sort_unstable();
    (nanos[0], nanos[passes / 2])
}

fn report(path: &str, evaluations: usize, (min, median): (u128, u128)) {
    println!(
        "EVAL-BENCH path=\"{path}\" evaluations={evaluations} min_ns={min} median_ns={median} \
         ns_per_eval_min={:.3} ns_per_eval_median={:.3}",
        min as f64 / evaluations as f64,
        median as f64 / evaluations as f64
    );
}

fn fold(result: anyhow::Result<f64>) -> u64 {
    result.map_or(1, f64::to_bits)
}

#[test]
#[ignore = "perf probe: legacy evaluator throughput on the corpus; run in release with --nocapture"]
fn legacy_eval_row_throughput() {
    const PASSES: usize = 201;
    let corpus = corpus();
    let expressions: Vec<(MathExpr, CompiledMathExpr)> = corpus
        .expressions
        .iter()
        .map(|source| {
            let expr = MathExpr::parse(source).expect("corpus expression parses");
            let compiled = expr.compile().expect("corpus expression compiles");
            (expr, compiled)
        })
        .collect();
    let rows: Vec<&[f64]> = corpus.rows.iter().map(|row| row.cells.as_slice()).collect();
    let cells: usize = rows.iter().map(|row| row.len()).sum();
    // The pairs where eval_row returns a value: the arithmetic path, without error allocation.
    let value_pairs: Vec<(&MathExpr, &CompiledMathExpr, &[f64])> = expressions
        .iter()
        .flat_map(|(expr, compiled)| rows.iter().map(move |row| (expr, compiled, *row)))
        .filter(|(expr, _, row)| expr.eval_row(row).is_ok())
        .collect();

    report(
        "interpreter eval_row, all pairs",
        expressions.len() * rows.len(),
        time_passes(PASSES, || {
            let mut acc = 0u64;
            for (expr, _) in &expressions {
                for row in &rows {
                    acc ^= fold(expr.eval_row(black_box(row)));
                }
            }
            acc
        }),
    );
    report(
        "interpreter eval_row, value pairs",
        value_pairs.len(),
        time_passes(PASSES, || {
            let mut acc = 0u64;
            for (expr, _, row) in &value_pairs {
                acc ^= fold(expr.eval_row(black_box(row)));
            }
            acc
        }),
    );
    report(
        "compiled eval_row, value pairs",
        value_pairs.len(),
        time_passes(PASSES, || {
            let mut acc = 0u64;
            for (_, compiled, row) in &value_pairs {
                acc ^= fold(compiled.eval_row(black_box(row)));
            }
            acc
        }),
    );
    report(
        "interpreter eval_element, all cells",
        expressions.len() * cells,
        time_passes(PASSES, || {
            let mut acc = 0u64;
            for (expr, _) in &expressions {
                for row in &rows {
                    for &x in *row {
                        acc ^= fold(expr.eval_element(black_box(x), row));
                    }
                }
            }
            acc
        }),
    );
    report(
        "compiled map_row, all cells",
        expressions.len() * cells,
        time_passes(PASSES, || {
            let mut acc = 0u64;
            for (_, compiled) in &expressions {
                for row in &rows {
                    for value in compiled.map_row(black_box(row)) {
                        acc ^= value.to_bits();
                    }
                }
            }
            acc
        }),
    );
}
