//! Gate E2 for Python: every operation and head the capability registry marks Python-supported,
//! fitted in Rust, exported with its fitted state and run under `python3` without refitting.
//! Each prediction must agree with `Model::predict` to `abs 1e-12 + rel 1e-10` (spec §10.5).
//! A missing `python3` is a skip locally and a failure under `JOPE_PARITY_STRICT=1`, as in
//! `tests/fitted_js_qualification.rs`.

use super::render::{render, Py};
use super::*;
use crate::math_pipeline::MathPipeline;
use feat_math::evolve::api::capability::{CapabilityKind, Language, Registry};
use feat_math::genome::GenomeIr;
use feat_math::heads::FittedHead;
use feat_math::model::{fit, Model};
use feat_math::ops::FittedOp;
use std::io::ErrorKind;
use std::path::Path;
use std::process::Command;

const WIDTH: usize = 5;

/// Hand-picked programs beyond the registry probes: non-default parameters, appended stages,
/// every target kind and multi-stage chains.
const PROGRAMS: &[&str] = &[
    "(pipe X (ridge))",
    "(pipe X (standardize append=true) (ridge lambda_frac=0.1))",
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
    "(pipe X (landmark_dist k=4 gamma=0.25) (ridge))",
    "(pipe X (oblique_hinge n=4) (ridge))",
    "(pipe X (row_stats append=true) (ridge))",
    "(pipe X (pair_prod append=true topk=4) (ridge))",
    "(pipe X (target_bin_encode bins=4 prior=1) (ridge))",
    "(pipe X (sigmoid append=true) (ridge))",
    "(pipe X (step t=0.25 append=true) (ridge))",
    "(pipe X (norm_row append=true) (ridge))",
    "(pipe X (row_diff_stats append=true) (ridge))",
    "(pipe X (polar append=true) (ridge))",
    "(pipe X (lag k=2 append=true) (ridge))",
    "(pipe X (roll k=3 append=true) (ridge))",
    "(pipe X (fft_mag k=9 append=true) (ridge))",
    "(pipe X (row_spectral append=true) (ridge))",
    "(pipe X (svd_project k=3 iters=10) (ridge))",
    "(pipe X (nystrom k=8 gamma=0.5) (ridge))",
    "(pipe X (map expr=\"sigmoid(col(0))\") (ridge))",
    "(pipe X (derive expr=\"row_mean + c0\") (ridge))",
    "(pipe X (symlog c=0.3 append=true) (ridge))",
    "(pipe X (select_corr k=3) (ridge))",
    "(pipe X (select_corr k=2 by=var append=true) (ridge))",
    "(pipe X (impute strategy=mean add_indicator=true) (ridge))",
    "(pipe X (impute strategy=const value=0.5) (ridge))",
    "(pipe X (pair_ratio topk=4 append=true) (ridge))",
    "(pipe X (spline knots=4 append=true) (ridge))",
    "(pipe X (robust_scale) (hinge knots=3 append=true) (elasticnet))",
    "(pipe X (standardize) (rff_cos n=8 append=true) (ridge_gcv))",
    "(pipe X (elasticnet alpha=0.001 l1_ratio=0.5 sweeps=60))",
    "(pipe X (gbm rounds=20 depth=3))",
    "(pipe X (ensemble lambda_frac=0.05 rounds=10 depth=2))",
    "(pipe X (histgbm2 rounds=20 leaves=8 bags=2 loss=huber))",
    "(pipe X (krr m=50))",
    "(pipe X (gpr m=50 steps=5))",
    "(pipe X (gpr kernel=rq+linear m=50 steps=5))",
    "(pipe X (ksmooth k=16 gamma=2.0))",
    "(pipe X (stack))",
    "(pipe X (portfolio))",
    "(pipe X (portfolio preset=none add1=group_mean add2=knn_loo repeats=1))",
    "(pipe X (target kind=rank_gauss) (ridge))",
    "(pipe X (target kind=log1p) (standardize) (ridge))",
    "(pipe X (target kind=winsor q=0.1) (ridge))",
    "(pipe X (target kind=zscore) (gbm rounds=10 depth=2))",
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

fn train() -> (Vec<Vec<f64>>, Vec<f64>) {
    let x = rows(160, WIDTH, 7);
    let y = x
        .iter()
        .map(|r| 2.0 * r[0] - r[1] * r[2] + (3.0 * r[3]).sin() + 0.5 * r[4] + 1.5)
        .collect();
    (x, y)
}

/// Held rows, three training rows, values outside the training range on both sides, signed
/// zero, exact zeros (the imputer's missing marker), a constant row and non-finite cells (the
/// engine zeroes them).
fn test_rows() -> Vec<Vec<f64>> {
    let mut test = rows(40, WIDTH, 99);
    test.extend(train().0.into_iter().take(3));
    test.push(vec![-0.5; WIDTH]);
    test.push(vec![1.5; WIDTH]);
    test.push(vec![0.0; WIDTH]);
    test.push(vec![-0.0; WIDTH]);
    test.push(vec![0.25; WIDTH]);
    test.push(vec![1e6, -1e6, 0.5, 0.0, 1.0]);
    test.push(vec![f64::NAN, 0.5, f64::INFINITY, f64::NEG_INFINITY, 0.3]);
    test
}

fn py_float(v: f64) -> String {
    if v.is_nan() {
        "float(\"nan\")".into()
    } else if v.is_infinite() {
        format!("float(\"{}inf\")", if v < 0.0 { "-" } else { "" })
    } else {
        format!("{v:?}")
    }
}

fn py_rows(rows: &[Vec<f64>]) -> String {
    let rows: Vec<String> = rows
        .iter()
        .map(|r| {
            format!(
                "[{}]",
                r.iter()
                    .map(|&v| py_float(v))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })
        .collect();
    format!("[{}]", rows.join(", "))
}

/// `JOPE_PYTHON` selects another interpreter, e.g. the oldest supported one (3.9).
fn python() -> std::ffi::OsString {
    std::env::var_os("JOPE_PYTHON").unwrap_or_else(|| "python3".into())
}

/// Run `call` (an expression over the loaded module namespace `m`) under `python3 -I -B` and
/// read one `repr` float per line. `None` when python3 is absent and strict mode is off.
fn run_python(dir: &Path, source: &str, call: &str) -> Option<Result<Vec<f64>, String>> {
    let predictor = dir.join("predictor.py");
    let driver = dir.join("driver.py");
    std::fs::write(&predictor, source).unwrap();
    std::fs::write(
        &driver,
        format!(
            "import runpy, sys\nm = runpy.run_path(sys.argv[1], run_name=\"predictor\")\n\
                 out = {call}\nsys.stdout.write(\"\\n\".join(repr(v) for v in out))\n"
        ),
    )
    .unwrap();
    let output = match Command::new(python())
        .args(["-I", "-B"])
        .arg(&driver)
        .arg(&predictor)
        .output()
    {
        Ok(output) => output,
        Err(error) if error.kind() == ErrorKind::NotFound => {
            assert!(
                std::env::var_os("JOPE_PARITY_STRICT").is_none_or(|v| v != "1"),
                "fitted Python: python3 is unavailable and JOPE_PARITY_STRICT=1"
            );
            eprintln!("fitted Python SKIP: python3 is unavailable");
            return None;
        }
        Err(error) => panic!("could not start python3: {error}"),
    };
    if !output.status.success() {
        return Some(Err(String::from_utf8_lossy(&output.stderr).into_owned()));
    }
    let text = String::from_utf8(output.stdout).unwrap();
    Some(Ok(text
        .lines()
        .map(|line| {
            line.parse::<f64>()
                .unwrap_or_else(|e| panic!("{line:?}: {e}"))
        })
        .collect()))
}

fn within_tolerance(actual: f64, expected: f64) -> bool {
    if expected.is_nan() {
        return actual.is_nan();
    }
    (actual - expected).abs() <= 1e-12 + 1e-10 * expected.abs()
}

/// The registry's probe program for one Python-supported operation or head.
fn probe(kind: CapabilityKind, name: &str) -> String {
    match (kind, name) {
        (_, "target") => "(pipe X (target kind=winsor) (ridge))".into(),
        (_, "map" | "derive") => format!("(pipe X ({name} expr=\"col(0)\") (ridge))"),
        (CapabilityKind::Head, _) => format!("(pipe X ({name}))"),
        _ => format!("(pipe X ({name}) (ridge))"),
    }
}

/// Fit, export and run one program; returns `(rows, bit-identical rows, max abs error)`.
fn check(dir: &Path, dsl: &str) -> Option<(usize, usize, f64)> {
    let (x, y) = train();
    let test = test_rows();
    let genome = GenomeIr::parse(dsl).unwrap_or_else(|e| panic!("{dsl}: {e:#}"));
    let model = fit(&genome, &x, &y).unwrap_or_else(|e| panic!("{dsl}: {e:#}"));
    let expected = model.predict(&test).unwrap();
    let source = emit(&genome, &model, WIDTH).unwrap_or_else(|e| panic!("{dsl}: {e:#}"));
    assert!(
        source.contains("\ndef fitted_model():\n"),
        "{dsl}: no fitted model"
    );
    let actual = run_python(
        dir,
        &source,
        &format!("m[\"fitted_predict\"]({})", py_rows(&test)),
    )?
    .unwrap_or_else(|stderr| panic!("{dsl}: python3 failed:\n{stderr}"));
    assert_eq!(actual.len(), expected.len(), "{dsl}");
    let mut exact = 0;
    let mut worst = 0.0_f64;
    for (row, (a, e)) in actual.iter().zip(&expected).enumerate() {
        assert!(
            within_tolerance(*a, *e),
            "{dsl}: row {row}: Python {a:?} vs Rust {e:?}"
        );
        exact += usize::from(a.to_bits() == e.to_bits());
        worst = worst.max((a - e).abs());
    }
    Some((expected.len(), exact, worst))
}

mod evaluation;
mod pipeline;
mod validation;
