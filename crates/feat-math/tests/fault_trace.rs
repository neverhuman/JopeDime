//! Fault reporting of the scalar evaluator (kernel2 WP-1.1 to WP-1.3).
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

use feat_math::math_expr::{CompiledMathExpr, MathExpr};
use sha2::{Digest, Sha256};
use std::hint::black_box;
use std::time::Instant;

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
