use super::*;

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
pub(super) struct At<'a> {
    expr: &'a str,
    row: &'a str,
    element: Option<usize>,
}

impl<'a> At<'a> {
    pub(super) fn new(expr: &'a str, row: &'a str) -> Self {
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
pub(super) struct Tallies {
    pub(super) rows: Tally,
    pub(super) elements: Tally,
}

/// Every traced entry point on one row, against the legacy entry points and against each other:
/// the interpreter in row context (compat shape and strict) and at each element, the compiled
/// form in row context and at each element, and the compiled map of the row.
pub(super) fn check_every_form(
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
