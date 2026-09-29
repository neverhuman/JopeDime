//! Fault reporting of the scalar evaluator (kernel2 WP-1.1 to WP-1.3).
//!
//! - The fault classes: bits, names and severities are append-only (WP-1.1).
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

use feat_math::math_expr::{CompiledMathExpr, FaultClass, FaultSet, FaultSeverity, MathExpr};
use sha2::{Digest, Sha256};
use std::hint::black_box;
use std::time::Instant;

// ---------------------------------------------------------------------------------------------
// The fault classes (WP-1.1)
// ---------------------------------------------------------------------------------------------

/// The append-only table: bit, class, stable name and severity.
#[rustfmt::skip]
const CLASSES: [(u32, FaultSet, &str, FaultSeverity); 16] = [
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
];

#[test]
fn fault_bits_names_and_severities_are_append_only() {
    assert_eq!(FaultClass::ALL.len(), CLASSES.len());
    for (class, (bit, set, name, severity)) in FaultClass::ALL.into_iter().zip(CLASSES) {
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
    assert_eq!(FaultSet::ALL.bits(), 0xffff);
    // Each class has exactly one severity: the three masks partition the classes.
    assert_eq!(FaultSeverity::Hard.classes().bits(), 0x9c00);
    assert_eq!(FaultSeverity::Domain.classes().bits(), 0x611f);
    assert_eq!(FaultSeverity::Saturation.classes().bits(), 0x02e0);
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
    let names: Vec<&str> = FaultClass::ALL
        .iter()
        .map(|class| class.name())
        .chain(
            [
                FaultSeverity::Saturation,
                FaultSeverity::Domain,
                FaultSeverity::Hard,
            ]
            .iter()
            .map(|severity| severity.name()),
        )
        .collect();
    for name in names {
        let hits = feat_math::transpile::lint_shipped_text(name);
        assert!(hits.is_empty(), "{name}: {hits:?}");
    }
}

/// One corpus row: its cells.
struct Row {
    cells: Vec<f64>,
}

/// PR 01's corpus: every row, and every expression with whether it is a map-context expression.
struct Corpus {
    rows: Vec<Row>,
    expressions: Vec<(String, bool)>,
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
    let expressions: Vec<(String, bool)> = file["expressions"]
        .as_array()
        .expect("expressions")
        .iter()
        .map(|expression| {
            let context = expression["context"].as_str().expect("context");
            assert!(
                context == "row" || context == "map",
                "unknown context {context}"
            );
            (
                expression["source"].as_str().expect("source").to_string(),
                context == "map",
            )
        })
        .collect();
    assert!(
        rows.iter().any(|row| row.cells.is_empty()) && !expressions.is_empty(),
        "the corpus has an empty row and at least one expression"
    );
    Corpus { rows, expressions }
}

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
    for (source, _) in &corpus.expressions {
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
        .map(|(source, _)| {
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
