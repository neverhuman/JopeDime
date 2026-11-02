use super::*;

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

pub(super) const POS_ZERO: u64 = 0x0000_0000_0000_0000;
const NEG_ZERO: u64 = 0x8000_0000_0000_0000;
const MAX_ABS: u64 = 0x426d_1a94_a200_0000; // 1e12, the output filter's clamp

fn union_of(sets: &[FaultSet]) -> FaultSet {
    sets.iter()
        .fold(FaultSet::EMPTY, |all, set| all.union(*set))
}

/// One evaluation and what it must report.
pub(super) struct Case {
    pub(super) source: &'static str,
    pub(super) row: Vec<f64>,
    /// `None` is row context (`eval_row`); `Some(x)` is element context (`eval_element`).
    pub(super) x: Option<f64>,
    /// The value's bits where they are free of libm; `None` compares with the legacy value only.
    pub(super) bits: Option<u64>,
    pub(super) faults: FaultSet,
}

impl Case {
    /// The row, and the element in element context, for failure messages.
    pub(super) fn row_text(&self) -> String {
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

pub(super) fn all_cases() -> Vec<Case> {
    let mut cases = wp_1_2_table();
    cases.extend(wp_1_8_table());
    cases.extend(class_cases());
    cases
}
