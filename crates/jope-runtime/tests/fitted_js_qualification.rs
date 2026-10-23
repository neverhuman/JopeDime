//! Fitted JavaScript qualification: every transform and head in the JavaScript slice, fitted
//! in Rust, exported with its fitted state by `jope_runtime::fitted_js::emit`, and run under Node
//! without refitting. Each prediction must agree with `Model::predict` to
//! `1e-8 * (1 + |reference|)`, the tolerance the TypeScript target states; JavaScript uses
//! ECMAScript math, so transcendental stages may differ from the platform libm by an ulp.
//!
//! A missing `node` is a skip locally and a failure under `JOPE_PARITY_STRICT=1`, as in
//! `portable_parity.rs`.

use feat_math::genome::GenomeIr;
use feat_math::model::fit;
use jope_runtime::fitted_js::emit;
use std::io::ErrorKind;
use std::process::Command;

const GENOMES: &[&str] = &[
    "(pipe X (standardize) (ridge))",
    "(pipe X (robust_scale append=true) (ridge))",
    "(pipe X (rank) (ridge))",
    "(pipe X (blom append=true) (ridge))",
    "(pipe X (quantile_bins bins=8) (ridge))",
    "(pipe X (bin_onehot bins=4 append=true) (ridge))",
    "(pipe X (hinge knots=3) (ridge))",
    "(pipe X (winsor_clip q=0.05) (ridge))",
    "(pipe X (cheb degree=3 append=true) (ridge))",
    "(pipe X (poly_interact append=true) (ridge))",
    "(pipe X (rff_cos n=8) (ridge))",
    "(pipe X (rff_sin n=16 seed=2) (ridge))",
    "(pipe X (rand_proj k=4) (ridge))",
    "(pipe X (landmark_dist k=4) (ridge))",
    "(pipe X (oblique_hinge n=4) (ridge))",
    "(pipe X (row_stats append=true) (ridge))",
    "(pipe X (pair_prod append=true topk=4) (ridge))",
    "(pipe X (target_bin_encode bins=4 prior=1) (ridge))",
    "(pipe X (sigmoid append=true) (ridge))",
    "(pipe X (step t=0.5 append=true) (ridge))",
    "(pipe X (norm_row append=true) (ridge))",
    "(pipe X (row_diff_stats append=true) (ridge))",
    "(pipe X (robust_scale) (hinge knots=3 append=true) (elasticnet))",
];

/// Deterministic rows in [0, 1) from a fixed linear congruential sequence.
fn rows(n: usize, width: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            (0..width)
                .map(|_| {
                    state = state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    (state >> 11) as f64 / (1u64 << 53) as f64
                })
                .collect()
        })
        .collect()
}

#[test]
fn fitted_javascript_matches_rust_prediction_for_every_supported_stage() {
    let width = 5;
    let train = rows(160, width, 7);
    let y: Vec<f64> = train
        .iter()
        .map(|r| 2.0 * r[0] - r[1] * r[2] + (3.0 * r[3]).sin() + 0.5 * r[4])
        .collect();
    // Held rows plus values outside the training range on both sides.
    let mut test = rows(40, width, 99);
    test.push(vec![-0.5; width]);
    test.push(vec![1.5; width]);
    test.push(vec![0.0; width]);

    let dir = tempfile::tempdir().unwrap();
    let mut checked = 0;
    for dsl in GENOMES {
        let genome = GenomeIr::parse(dsl).unwrap_or_else(|e| panic!("{dsl}: {e:#}"));
        let model = fit(&genome, &train, &y).unwrap_or_else(|e| panic!("{dsl}: {e:#}"));
        let expected = model.predict(&test).unwrap();
        let source = emit(&genome, &model, width).unwrap_or_else(|e| panic!("{dsl}: {e:#}"));
        assert!(
            source.contains("function modelFittedHandle()"),
            "{dsl}: no fitted handle"
        );

        let program = dir.path().join("program.js");
        let flat = serde_json::to_string(&test.concat()).unwrap();
        std::fs::write(
            &program,
            format!(
                "{source}\nconst h=modelFittedHandle(),x=Float64Array.from({flat}),n={rows};\n\
                 const out=new Float64Array(n);\n\
                 const status=modelPredictNumericBatch(h,x,x.length,n,{width},null,0,out,n);\n\
                 if(status!==0)throw new Error('status '+status);\n\
                 process.stdout.write(JSON.stringify(Array.from(out)));\n",
                rows = test.len()
            ),
        )
        .unwrap();
        let output = match Command::new("node").arg(&program).output() {
            Ok(output) => output,
            Err(error) if error.kind() == ErrorKind::NotFound => {
                assert!(
                    std::env::var_os("JOPE_PARITY_STRICT").is_none_or(|v| v != "1"),
                    "fitted JavaScript: node is unavailable and JOPE_PARITY_STRICT=1"
                );
                eprintln!("fitted JavaScript SKIP: node is unavailable");
                return;
            }
            Err(error) => panic!("could not start node: {error}"),
        };
        assert!(
            output.status.success(),
            "{dsl}: node failed:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let actual: Vec<f64> = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(actual.len(), expected.len(), "{dsl}");
        for (row, (a, e)) in actual.iter().zip(&expected).enumerate() {
            assert!(
                (a - e).abs() <= 1e-8 * (1.0 + e.abs()),
                "{dsl}: row {row}: JavaScript {a} vs Rust {e}"
            );
        }
        checked += 1;
    }
    assert_eq!(checked, GENOMES.len());
}

#[test]
fn fitted_javascript_refuses_a_model_from_another_plan() {
    let train = rows(60, 3, 3);
    let y: Vec<f64> = train.iter().map(|r| r[0] + r[1]).collect();
    let fitted_on = GenomeIr::parse("(pipe X (standardize) (ridge))").unwrap();
    let model = fit(&fitted_on, &train, &y).unwrap();
    let other = GenomeIr::parse("(pipe X (robust_scale) (ridge))").unwrap();
    let error = emit(&other, &model, 3).unwrap_err();
    assert!(
        format!("{error:#}").contains("fitted standardize where the plan has robust_scale"),
        "{error:#}"
    );
    let longer = GenomeIr::parse("(pipe X (standardize) (rank) (ridge))").unwrap();
    let error = emit(&longer, &model, 3).unwrap_err();
    assert!(
        format!("{error:#}").contains("1 fitted stages for 2 planned stages"),
        "{error:#}"
    );
}
