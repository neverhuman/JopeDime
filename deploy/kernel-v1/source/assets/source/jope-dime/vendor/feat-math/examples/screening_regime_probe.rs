//! Development counterexample: identical ridge surrogates may hide different
//! actual learners. Read the frozen manifest before executing; no final data.
use feat_math::{fit_predict, GenomeIr, Pcg32};
use serde_json::{json, Value};
use std::{fs, time::Instant};

fn main() -> anyhow::Result<()> {
    let path = std::env::args()
        .nth(1)
        .expect("screening manifest path required");
    let bytes = fs::read(path)?;
    let manifest: Value = serde_json::from_slice(&bytes)?;
    anyhow::ensure!(
        manifest["schema"] == "kernel.screening-probe-manifest.v1",
        "manifest version"
    );
    let train = manifest["training_rows"].as_u64().unwrap() as usize;
    let test = manifest["evaluation_rows"].as_u64().unwrap() as usize;
    let width = manifest["features"].as_u64().unwrap() as usize;
    anyhow::ensure!(
        train > 0 && train <= 4096 && test > 0 && test <= 4096 && width == 4,
        "probe dimensions"
    );
    let genomes: Vec<_> = manifest["genomes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| GenomeIr::parse(v.as_str().unwrap()))
        .collect::<anyhow::Result<_>>()?;
    let mut cases = Vec::new();
    for seed in manifest["seeds"].as_array().unwrap() {
        let seed = seed.as_u64().unwrap();
        let mut rng = Pcg32::new(seed, 20260908);
        let rows: Vec<Vec<f64>> = (0..train + test)
            .map(|_| (0..width).map(|_| rng.range_f64(-1.0, 1.0)).collect())
            .collect();
        let training = rows[..train].to_vec();
        let evaluation = rows[train..].to_vec();
        for mechanism in manifest["mechanisms"].as_array().unwrap() {
            let mechanism = mechanism.as_str().unwrap();
            anyhow::ensure!(
                matches!(mechanism, "linear" | "interaction"),
                "probe mechanism"
            );
            let y: Vec<_> = rows
                .iter()
                .map(|r| {
                    if mechanism == "linear" {
                        r[0]
                    } else {
                        r[0] * r[1]
                    }
                })
                .collect();
            let mut models = Vec::new();
            // Alternate measurement order across seeds to reduce timing bias.
            for i in (0..genomes.len()).map(|i| {
                if seed % 2 == 0 {
                    i
                } else {
                    genomes.len() - 1 - i
                }
            }) {
                let begun = Instant::now();
                let prediction = fit_predict(&genomes[i], &training, &y[..train], &evaluation)?;
                let seconds = begun.elapsed().as_secs_f64();
                let mse = prediction
                    .iter()
                    .zip(&y[train..])
                    .map(|(p, y)| (p - y) * (p - y))
                    .sum::<f64>()
                    / test as f64;
                anyhow::ensure!(mse.is_finite(), "nonfinite probe loss");
                models.push(json!({"genome":manifest["genomes"][i],"mse":mse,"fit_and_predict_seconds":seconds}));
            }
            cases.push(json!({"seed":seed,"mechanism":mechanism,"models":models}));
        }
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "schema":"kernel.screening-probe.v1", "manifest":manifest,
            "manifest_identity":feat_math::kernel::codec::identity("screening-manifest", &bytes),
            "cases":cases,"qualified_for_deployment":false
        }))?
    );
    Ok(())
}
