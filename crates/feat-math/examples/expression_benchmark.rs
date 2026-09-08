//! Same-source microbenchmark; does not qualify whole-search performance.
use feat_math::{math_expr::MathExpr, Pcg32};
use serde_json::{json, Value};
use std::{fs, hint::black_box, time::Instant};

fn measure<F: Fn(&[f64]) -> Vec<f64>>(rows: &[Vec<f64>], repetitions: usize, eval: F) -> f64 {
    let begun = Instant::now();
    for _ in 0..repetitions {
        for row in rows {
            black_box(eval(black_box(row)));
        }
    }
    begun.elapsed().as_secs_f64()
}

fn main() -> anyhow::Result<()> {
    let bytes = fs::read(std::env::args().nth(1).expect("manifest path required"))?;
    let manifest: Value = serde_json::from_slice(&bytes)?;
    anyhow::ensure!(
        manifest["schema"] == "kernel.expression-benchmark-manifest.v1",
        "manifest version"
    );
    let n = manifest["rows"].as_u64().unwrap() as usize;
    let repetitions = manifest["repetitions"].as_u64().unwrap() as usize;
    anyhow::ensure!(
        n > 0 && n <= 4096 && repetitions > 0 && repetitions <= 100,
        "probe budget"
    );
    let mut cases = Vec::new();
    for seed in manifest["seeds"].as_array().unwrap() {
        let seed = seed.as_u64().unwrap();
        for width in manifest["widths"].as_array().unwrap() {
            let width = width.as_u64().unwrap() as usize;
            anyhow::ensure!(width > 0 && width <= 512, "probe width");
            let mut rng = Pcg32::new(seed, 20260908);
            let rows: Vec<Vec<f64>> = (0..n)
                .map(|_| (0..width).map(|_| rng.range_f64(-1.0, 1.0)).collect())
                .collect();
            for source in manifest["expressions"].as_array().unwrap() {
                let expr = MathExpr::parse(source.as_str().unwrap())?;
                let begun = Instant::now();
                let compiled = expr.compile()?;
                let compilation_seconds = begun.elapsed().as_secs_f64();
                let reference = |row: &[f64]| {
                    row.iter()
                        .map(|x| expr.eval_element(*x, row).unwrap_or(0.0))
                        .collect::<Vec<_>>()
                };
                let registers = |row: &[f64]| compiled.map_row(row);
                // Warm both implementations and check every result before timing.
                for row in &rows {
                    anyhow::ensure!(
                        reference(row)
                            .iter()
                            .map(|x| x.to_bits())
                            .eq(registers(row).iter().map(|x| x.to_bits())),
                        "reference mismatch"
                    );
                }
                let (reference_seconds, register_seconds) = if seed % 2 == 0 {
                    (
                        measure(&rows, repetitions, reference),
                        measure(&rows, repetitions, registers),
                    )
                } else {
                    let register_seconds = measure(&rows, repetitions, registers);
                    (measure(&rows, repetitions, reference), register_seconds)
                };
                cases.push(json!({"seed":seed,"width":width,"source":source,"reference_seconds":reference_seconds,"register_seconds":register_seconds,"compilation_seconds":compilation_seconds}));
            }
        }
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "schema":"kernel.expression-benchmark.v1", "manifest":manifest,
            "manifest_identity":feat_math::kernel::codec::identity("expression-benchmark-manifest", &bytes),
            "cases":cases,"qualified_for_deployment":false
        }))?
    );
    Ok(())
}
