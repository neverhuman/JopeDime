use super::*;

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
