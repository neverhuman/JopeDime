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

#[test]
fn fitted_python_matches_rust_for_every_registry_supported_operation_and_head() {
    let dir = tempfile::tempdir().unwrap();
    let registry = Registry::current();
    let mut programs: Vec<String> = registry
        .entries
        .iter()
        .filter(|row| matches!(row.kind, CapabilityKind::Operation | CapabilityKind::Head))
        .filter(|row| registry.supports(row.kind, row.name, Language::Python))
        .map(|row| probe(row.kind, row.name))
        .collect();
    let probes = programs.len();
    assert!(probes >= 50, "only {probes} Python-supported registry rows");
    programs.extend(PROGRAMS.iter().map(|dsl| dsl.to_string()));
    for dsl in &programs {
        let Some((rows, exact, worst)) = check(dir.path(), dsl) else {
            return;
        };
        eprintln!("E2 python {dsl}: {exact}/{rows} rows bit-identical, max abs error {worst:e}");
    }
}

/// Spec §10.5 decision flips for tree and threshold heads: on randomized rows, the share of
/// rows whose Python prediction leaves the numeric tolerance must stay at or under 0.1%.
#[test]
fn tree_and_threshold_programs_do_not_flip_on_randomized_rows() {
    const TREES: &[&str] = &[
        "(pipe X (gbm rounds=20 depth=3))",
        "(pipe X (ensemble rounds=10 depth=2))",
        "(pipe X (histgbm2 rounds=20 leaves=8 bags=2))",
        "(pipe X (stack))",
        "(pipe X (step t=0.5) (quantile_bins bins=8 append=true) (ridge))",
        "(pipe X (bin_onehot bins=4) (target_bin_encode bins=4 append=true) (ridge))",
    ];
    let dir = tempfile::tempdir().unwrap();
    let (x, y) = train();
    // Rows over [-0.25, 1.25): the training range and both sides of it.
    let random: Vec<Vec<f64>> = rows(2000, WIDTH, 2026)
        .into_iter()
        .map(|r| r.into_iter().map(|v| 1.5 * v - 0.25).collect())
        .collect();
    for dsl in TREES {
        let genome = GenomeIr::parse(dsl).unwrap();
        let model = fit(&genome, &x, &y).unwrap();
        let expected = model.predict(&random).unwrap();
        let source = emit(&genome, &model, WIDTH).unwrap();
        let Some(actual) = run_python(
            dir.path(),
            &source,
            &format!("m[\"fitted_predict\"]({})", py_rows(&random)),
        ) else {
            return;
        };
        let actual = actual.unwrap_or_else(|stderr| panic!("{dsl}: python3 failed:\n{stderr}"));
        assert_eq!(actual.len(), expected.len(), "{dsl}");
        let flips = actual
            .iter()
            .zip(&expected)
            .filter(|(a, e)| !within_tolerance(**a, **e))
            .count();
        let exact = actual
            .iter()
            .zip(&expected)
            .filter(|(a, e)| a.to_bits() == e.to_bits())
            .count();
        eprintln!(
            "E2 python flips {dsl}: {flips}/{} rows, {exact} bit-identical",
            random.len()
        );
        assert!(flips * 1000 <= random.len(), "{dsl}: {flips} flips");
    }
}

/// The portfolio state keeps every active member; duplicated training rows give the
/// group-mean member weight, so its table is exported and looked up row by row.
#[test]
fn portfolio_exports_its_active_members_including_group_means() {
    use feat_math::portfolio::FittedMemberState as S;
    let distinct = rows(40, WIDTH, 11);
    let x: Vec<Vec<f64>> = (0..4).flat_map(|_| distinct.clone()).collect();
    let y: Vec<f64> = x
        .iter()
        .enumerate()
        .map(|(i, r)| (7.0 * r[0]).sin() + r[1] * r[4] + 0.01 * (i % 4) as f64)
        .collect();
    let mut duplicated_test = distinct.clone();
    duplicated_test.extend(rows(10, WIDTH, 12));
    let (plain_x, plain_y) = train();
    let dir = tempfile::tempdir().unwrap();
    for (dsl, x, y, test, min_active) in [
        ("(pipe X (portfolio))", &plain_x, &plain_y, test_rows(), 2),
        (
            "(pipe X (portfolio preset=none add1=group_mean add2=knn_loo repeats=1))",
            &x,
            &y,
            duplicated_test,
            2,
        ),
    ] {
        let genome = GenomeIr::parse(dsl).unwrap();
        let model = fit(&genome, x, y).unwrap();
        let Model::Pipeline { head, .. } = &model else {
            panic!("{dsl}: not a pipeline");
        };
        let FittedHead::Portfolio(p) = head.as_ref() else {
            panic!("{dsl}: not a portfolio head");
        };
        assert!(!p.mean_only, "{dsl}: mean-only portfolio");
        let active = p
            .members
            .iter()
            .zip(&p.weights)
            .filter(|(m, w)| **w > 0.0 && matches!(m.state, S::Fitted(_) | S::GroupMean(_)))
            .count();
        assert!(active >= min_active, "{dsl}: {active} active members");
        if dsl.contains("group_mean") {
            assert!(
                p.members
                    .iter()
                    .zip(&p.weights)
                    .any(|(m, w)| *w > 0.0 && matches!(m.state, S::GroupMean(_))),
                "{dsl}: no weighted group-mean member"
            );
        }
        let expected = model.predict(&test).unwrap();
        let source = emit(&genome, &model, WIDTH).unwrap();
        assert!(source.contains("FITTED_HEAD = (False, "), "{dsl}");
        let Some(actual) = run_python(
            dir.path(),
            &source,
            &format!("m[\"fitted_predict\"]({})", py_rows(&test)),
        ) else {
            return;
        };
        let actual = actual.unwrap_or_else(|stderr| panic!("{dsl}: python3 failed:\n{stderr}"));
        assert_eq!(actual.len(), expected.len(), "{dsl}");
        for (row, (a, e)) in actual.iter().zip(&expected).enumerate() {
            assert_eq!(
                a.to_bits(),
                e.to_bits(),
                "{dsl}: row {row}: Python {a:?} vs Rust {e:?}"
            );
        }
        eprintln!(
            "E2 python {dsl}: {active} active members, {} rows bit-identical",
            test.len()
        );
    }
}

/// `emit_math_pipeline` against `MathPipeline::predict_cells`: raw cells through the f64 input
/// contract (missing tokens, booleans, Rust's number grammar and whitespace set, clamping, a
/// constant and an unselected column, a permuted selection), then the target inverse.
#[test]
fn math_pipeline_contract_matches_the_decoder() {
    use crate::math_pipeline::{InputContract, NumericColumn, PipelineState, TargetScale};
    use crate::math_pipeline::{INPUT_PART, PIPELINE_PART};
    use std::collections::BTreeMap;
    let column = |name: &str, min: f64, max: f64, constant: bool| NumericColumn {
        name: name.into(),
        min,
        max,
        constant,
    };
    let columns = vec![
        column("a", -2.0, 3.0, false),
        column("b", 0.0, 10.0, false),
        column("c", 1.0, 2.0, false),
        column("d", -1e3, 1e3, false),
        column("e", 0.0, 1.0, false),
        column("f", 5.0, 5.0, true),
        column("g", 0.0, 1.0, false),
    ];
    let decode = |dsl: &str, target: TargetScale| {
        let (x, y) = train();
        let model = fit(&GenomeIr::parse(dsl).unwrap(), &x, &y).unwrap();
        let input = InputContract::new(columns.clone(), vec![4, 0, 5, 2, 1], target).unwrap();
        let parts = BTreeMap::from([
            (
                PIPELINE_PART.to_string(),
                serde_json::to_vec(&PipelineState {
                    dsl: dsl.into(),
                    model,
                })
                .unwrap(),
            ),
            (INPUT_PART.to_string(), serde_json::to_vec(&input).unwrap()),
        ]);
        MathPipeline::decode(&parts, PIPELINE_PART).unwrap()
    };
    let cells: &[&str] = &[
        "0.5",
        "",
        "NA",
        " nan ",
        "?",
        "#N/A",
        "None",
        "null",
        "true",
        "Yes",
        "FALSE",
        "no",
        "1e3",
        ".5",
        "5.",
        "+2",
        "-0",
        "-0.0",
        "1_000",
        "\u{a0}0.25\u{3000}",
        "\u{1c}3",
        "\u{661}\u{662}",
        "inf",
        "-Infinity",
        "1e400",
        "1e-400",
        "0x10",
        "2.5e-1",
        "1e",
        "e5",
        ".",
        "+",
        "  -1.5  ",
        "\t7\n",
        "3.0\u{85}",
        "K",
        "1,5",
        "1.5.2",
        "TRUE",
        "nAn",
        "\u{feff}1",
        "-2",
        "3",
        "1.75",
        "-1000",
        "999.5",
        "0.999",
        "1e-3",
    ];
    let mut owned: Vec<Vec<Option<String>>> = Vec::new();
    for r in 0..cells.len() {
        owned.push(
            (0..7)
                .map(|j| {
                    let k = (r * 7 + j * 3) % (cells.len() + 1);
                    cells.get(k).map(|c| c.to_string())
                })
                .collect(),
        );
    }
    let rows: Vec<Vec<Option<&str>>> = owned
        .iter()
        .map(|row| row.iter().map(|c| c.as_deref()).collect())
        .collect();
    let literal = render(&Py::List(
        owned
            .iter()
            .map(|row| {
                Py::List(
                    row.iter()
                        .map(|c| c.clone().map_or(Py::None, Py::Str))
                        .collect(),
                )
            })
            .collect(),
    ))
    .unwrap();
    let dir = tempfile::tempdir().unwrap();
    for dsl in [
        "(pipe X (standardize) (ridge))",
        "(pipe X (target kind=zscore) (hinge knots=3) (gbm rounds=10 depth=2))",
    ] {
        let pipeline = decode(
            dsl,
            TargetScale {
                name: "y".into(),
                min: -4.0,
                scale: 2.5,
            },
        );
        let expected = pipeline.predict_cells(&rows).unwrap();
        let source = String::from_utf8(emit_math_pipeline(&pipeline).unwrap()).unwrap();
        let Some(actual) = run_python(
            dir.path(),
            &source,
            &format!("m[\"predict_rows\"]({literal})"),
        ) else {
            return;
        };
        let actual = actual.unwrap_or_else(|stderr| panic!("{dsl}: python3 failed:\n{stderr}"));
        assert_eq!(actual.len(), expected.len(), "{dsl}");
        for (row, (a, e)) in actual.iter().zip(&expected).enumerate() {
            assert!(
                a.to_bits() == e.to_bits(),
                "{dsl}: row {row} {:?}: Python {a:?} vs Rust {e:?}",
                owned[row]
            );
        }
        // Numbers are accepted as cells and read as their value.
        let numbers = run_python(
                dir.path(),
                &source,
                "m[\"predict_rows\"]([[0.5, 1, None, 2.5, -1, 7, True]]) + m[\"predict_rows\"]([[\"0.5\", \"1\", None, \"2.5\", \"-1\", \"7\", \"true\"]])",
            )
            .unwrap()
            .unwrap();
        assert_eq!(numbers[0].to_bits(), numbers[1].to_bits(), "{dsl}");
        // A constant target scale predicts its minimum; an empty input predicts nothing.
        let flat = decode(
            dsl,
            TargetScale {
                name: "y".into(),
                min: 7.0,
                scale: 0.0,
            },
        );
        let source = String::from_utf8(emit_math_pipeline(&flat).unwrap()).unwrap();
        let out = run_python(
            dir.path(),
            &source,
            "m[\"predict_rows\"]([[None] * 7]) + m[\"predict_rows\"]([])",
        )
        .unwrap()
        .unwrap();
        assert_eq!(out, flat.predict_cells(&[vec![None; 7]]).unwrap(), "{dsl}");
    }
    // Failures agree: a short row, and a target inverse that overflows f64.
    let pipeline = decode(
        "(pipe X (standardize) (ridge))",
        TargetScale {
            name: "y".into(),
            min: 1e308,
            scale: 1e308,
        },
    );
    let source = String::from_utf8(emit_math_pipeline(&pipeline).unwrap()).unwrap();
    let overflow = vec![Some("1e9"); 7];
    for (rust, call, python) in [
        (
            pipeline.predict_cells(std::slice::from_ref(&overflow)),
            "m[\"predict_rows\"]([[\"1e9\"] * 7])",
            "ArithmeticError: prediction undefined: row 0: non-finite target prediction",
        ),
        (
            pipeline.predict_cells(&[vec![Some("1"); 6]]),
            "m[\"predict_rows\"]([[\"1\"] * 6])",
            "ValueError: row 0 has 6 cells; the model reads 7",
        ),
    ] {
        let rust = format!("{:#}", rust.unwrap_err());
        assert!(
            rust.contains("non-finite target prediction") || rust.contains("row has 6 cells"),
            "{call}: {rust}"
        );
        let stderr = run_python(dir.path(), &source, call).unwrap().unwrap_err();
        assert!(stderr.contains(python), "{call}: {stderr}");
    }
}

#[test]
fn fitted_python_refuses_a_model_from_another_plan() {
    let (x, y) = train();
    let fitted_on = GenomeIr::parse("(pipe X (standardize) (ridge))").unwrap();
    let model = fit(&fitted_on, &x, &y).unwrap();
    for (other, message) in [
        (
            "(pipe X (robust_scale) (ridge))",
            "fitted standardize where the plan has robust_scale",
        ),
        (
            "(pipe X (standardize) (rank) (ridge))",
            "1 fitted stages for 2 planned stages",
        ),
        (
            "(pipe X (standardize) (elasticnet))",
            "fitted head ridge does not match the planned head elasticnet",
        ),
        (
            "(pipe X (target kind=zscore) (standardize) (ridge))",
            "no fitted target transform where the plan has one",
        ),
    ] {
        let error = emit(&GenomeIr::parse(other).unwrap(), &model, WIDTH).unwrap_err();
        assert!(format!("{error:#}").contains(message), "{other}: {error:#}");
    }
    let target = fit(
        &GenomeIr::parse("(pipe X (target kind=zscore) (ridge))").unwrap(),
        &x,
        &y,
    )
    .unwrap();
    let error = emit(
        &GenomeIr::parse("(pipe X (target kind=winsor) (ridge))").unwrap(),
        &target,
        WIDTH,
    )
    .unwrap_err();
    assert!(
        format!("{error:#}").contains("target transform kind differs"),
        "{error:#}"
    );
}

#[test]
fn fitted_python_refuses_contradictory_prediction_parameters() {
    let (x, y) = train();
    let cases = [
        (
            "(pipe X (standardize append=true) (ridge))",
            "(pipe X (standardize) (ridge))",
            "append",
        ),
        (
            "(pipe X (step t=0.1) (ridge))",
            "(pipe X (step t=0.9) (ridge))",
            "t",
        ),
        (
            "(pipe X (cheb degree=3) (ridge))",
            "(pipe X (cheb degree=4) (ridge))",
            "degree",
        ),
        (
            "(pipe X (quantile_bins bins=4) (ridge))",
            "(pipe X (quantile_bins bins=8) (ridge))",
            "bins",
        ),
        (
            "(pipe X (bin_onehot bins=3) (ridge))",
            "(pipe X (bin_onehot bins=4) (ridge))",
            "bins",
        ),
        (
            "(pipe X (landmark_dist k=4 gamma=0.25) (ridge))",
            "(pipe X (landmark_dist k=4 gamma=1) (ridge))",
            "gamma",
        ),
        (
            "(pipe X (nystrom k=4 gamma=0.25) (ridge))",
            "(pipe X (nystrom k=4 gamma=1) (ridge))",
            "gamma",
        ),
        (
            "(pipe X (symlog c=0.5) (ridge))",
            "(pipe X (symlog c=2) (ridge))",
            "c",
        ),
        (
            "(pipe X (lag k=1) (ridge))",
            "(pipe X (lag k=2) (ridge))",
            "k",
        ),
        (
            "(pipe X (roll k=1) (ridge))",
            "(pipe X (roll k=2) (ridge))",
            "k",
        ),
        (
            "(pipe X (map expr=\"col(0)\") (ridge))",
            "(pipe X (map expr=\"sigmoid(col(0))\") (ridge))",
            "expr",
        ),
        (
            "(pipe X (ksmooth k=8))",
            "(pipe X (ksmooth k=16))",
            "head k",
        ),
        (
            "(pipe X (ksmooth gamma=0.5))",
            "(pipe X (ksmooth gamma=2))",
            "head gamma",
        ),
        (
            "(pipe X (gpr kernel=rq+linear m=50 steps=2))",
            "(pipe X (gpr kernel=rbf+linear m=50 steps=2))",
            "head kernel",
        ),
    ];
    for (fitted, declared, parameter) in cases {
        let model = fit(&GenomeIr::parse(fitted).unwrap(), &x, &y).unwrap();
        let error = emit(&GenomeIr::parse(declared).unwrap(), &model, WIDTH).unwrap_err();
        assert!(
            format!("{error:#}").contains(&format!("{parameter} parameter differs")),
            "{fitted} / {declared}: {error:#}"
        );
    }
    let genome = GenomeIr::parse("(pipe X (ridge))").unwrap();
    let model = fit(&genome, &x, &y).unwrap();
    assert!(
        emit(&genome, &model, WIDTH - 1).is_err(),
        "wrong input width accepted"
    );
    assert!(
        emit(&genome, &model, WIDTH).is_ok(),
        "zero-stage matching model refused"
    );
}

#[test]
fn fitted_python_refuses_non_finite_state_symbolic_models_and_f32_contracts() {
    let (x, y) = train();
    let genome = GenomeIr::parse("(pipe X (standardize) (ridge))").unwrap();
    let mut model = fit(&genome, &x, &y).unwrap();
    if let Model::Pipeline { ops, .. } = &mut model {
        if let FittedOp::Standardize { means, .. } = &mut ops[0] {
            means[1] = f64::NAN;
        }
    }
    let error = emit(&genome, &model, WIDTH).unwrap_err();
    assert!(
        format!("{error:#}").contains("non-finite fitted state"),
        "{error:#}"
    );

    let symbolic = Model::Symbolic {
        expr: feat_math::math_expr::MathExpr::parse("col(0)").unwrap(),
        a: 0.0,
        b: 1.0,
    };
    let error = emit(&genome, &symbolic, WIDTH).unwrap_err();
    assert!(
        format!("{error:#}").contains("needs a pipeline model"),
        "{error:#}"
    );

    let input = include_bytes!("../tests/fixtures/math-pipeline/input.json");
    let state = include_bytes!("../tests/fixtures/math-pipeline/pipeline.json");
    let parts = std::collections::BTreeMap::from([
        (crate::math_pipeline::INPUT_PART.into(), input.to_vec()),
        (crate::math_pipeline::PIPELINE_PART.into(), state.to_vec()),
    ]);
    let legacy = MathPipeline::decode(&parts, crate::math_pipeline::PIPELINE_PART).unwrap();
    let error = emit_math_pipeline(&legacy).unwrap_err();
    assert!(
        format!("{error:#}").contains("prediction_version 1"),
        "{error:#}"
    );
}

/// Spec §10.5 determinism: the same model and emitter produce byte-identical source.
#[test]
fn fitted_python_is_byte_identical_for_the_same_model() {
    let (x, y) = train();
    for dsl in [
        "(pipe X (rank) (pair_prod topk=4 append=true) (gbm rounds=5))",
        "(pipe X (portfolio))",
    ] {
        let genome = GenomeIr::parse(dsl).unwrap();
        let first = emit(&genome, &fit(&genome, &x, &y).unwrap(), WIDTH).unwrap();
        let again = emit(&genome, &fit(&genome, &x, &y).unwrap(), WIDTH).unwrap();
        assert_eq!(
            crate::sha256_hex(first.as_bytes()),
            crate::sha256_hex(again.as_bytes()),
            "{dsl}"
        );
    }
}
