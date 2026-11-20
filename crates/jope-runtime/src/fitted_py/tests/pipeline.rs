use super::*;

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

/// Zero-inflated industrial columns (such as `concrete-strength` `blast_furnace_slag == 0.0` on
/// over 50% of training rows) place exact tree/quantile thresholds on the image of `0.0`. Under
/// the old `(i + 17 * j) % 64 / 63.0` probe grid, row 47 hit `inputs[1] == 0.0` (`47 + 17 == 64`),
/// where a sub-ULP transcendental difference (`fdlibm`/`musl` vs host `libm`) flipped the split
/// and failed `python3 winner.py --self-check` at row 47.
#[test]
fn math_pipeline_self_check_avoids_zero_inflated_split_flips() {
    use crate::math_pipeline::{
        InputContract, NumericColumn, PipelineState, TargetScale, INPUT_PART, PIPELINE_PART,
    };
    use std::collections::BTreeMap;

    let mut x = rows(240, 8, 2026);
    for (idx, row) in x.iter_mut().enumerate() {
        if idx % 5 != 0 {
            // 80% mass at 0.0 in column 1 -> median/quantile tree cut lands directly on f(0.0).
            row[1] = 0.0;
        } else {
            row[1] = 0.002 * ((idx % 11) + 1) as f64;
        }
    }
    let y: Vec<f64> = x
        .iter()
        .map(|r| {
            let jump = if r[1] > 0.0 { 25.0 } else { -15.0 };
            40.0 + jump + 6.0 * r[0] - 4.0 * r[2] + 2.5 * r[3]
        })
        .collect();

    let dsl = "(pipe X (standardize) (symlog c=1.0) (histgbm depth=4 lr=0.15 rounds=80))";
    let genome = GenomeIr::parse(dsl).unwrap();
    let glibc_model = fit(&genome, &x, &y).unwrap();

    // Simulate `aarch64-unknown-linux-musl` Engine Lambda (`musl` libm) vs `x86_64` host Python
    // (`glibc` libm): shifting `edges[1][0]` down by 1 ULP leaves `v_musl <= edge` on the Engine
    // while `v_glibc == edge + 1 ULP` crosses the split boundary at `inputs[1] == 0.0` (row 47).
    let mut musl_model = glibc_model.clone();
    if let Model::Pipeline { head, .. } = &mut musl_model {
        if let FittedHead::Gbm(gbm) = head.as_mut() {
            gbm.edges[1][0] = f64::from_bits(gbm.edges[1][0].to_bits() + 1);
        }
    }

    let dir = tempfile::tempdir().unwrap();

    // 1. Reproduce the exact legacy `(i + 17 * j) % 64 / 63.0` row-47 failure under 1-ULP libm drift.
    let legacy_rows: Vec<Vec<f64>> = (0..64)
        .map(|i| (0..8).map(|j| ((i + 17 * j) % 64) as f64 / 63.0).collect())
        .collect();
    let legacy_expected = glibc_model.predict(&legacy_rows).unwrap();
    let legacy_checks: Vec<CheckRow> = legacy_rows
        .into_iter()
        .zip(legacy_expected)
        .map(|(inputs, expected)| CheckRow { inputs, expected })
        .collect();
    let legacy_py = emit_with_checks(&genome, &musl_model, 8, &legacy_checks).unwrap();
    let legacy_path = dir.path().join("legacy_winner.py");
    std::fs::write(&legacy_path, &legacy_py).unwrap();
    let legacy_out = match Command::new(python())
        .args(["-I", "-B"])
        .arg(&legacy_path)
        .arg("--self-check")
        .output()
    {
        Ok(out) => out,
        Err(err) if err.kind() == ErrorKind::NotFound => return,
        Err(err) => panic!("python3 spawn failed: {err}"),
    };
    assert!(
        !legacy_out.status.success(),
        "legacy row-47 probe should fail"
    );
    let legacy_stderr = String::from_utf8_lossy(&legacy_out.stderr);
    assert!(
        legacy_stderr.contains("self_check failed at row 47"),
        "expected row 47 failure, got:\n{legacy_stderr}"
    );

    // 2. Prove `select_pipeline_self_check_rows` (`emit_math_pipeline` & `fitted_rs::emit_pipeline`)
    // passes both unshifted and under 1-ULP cross-libm drift (`glibc_model` expected vs `musl_model` predictor).
    let screened = crate::math_pipeline::select_pipeline_self_check_rows(&glibc_model, 8).unwrap();
    assert_eq!(screened.len(), 64);
    for (row, _) in &screened {
        assert!(row.iter().all(|&v| v > 0.01 && v < 0.99));
    }
    let screened_checks: Vec<CheckRow> = screened
        .into_iter()
        .map(|(inputs, expected)| CheckRow { inputs, expected })
        .collect();
    let cross_libm_py = emit_with_checks(&genome, &musl_model, 8, &screened_checks).unwrap();
    let cross_path = dir.path().join("cross_libm_winner.py");
    std::fs::write(&cross_path, &cross_libm_py).unwrap();
    let cross_out = Command::new(python())
        .args(["-I", "-B"])
        .arg(&cross_path)
        .arg("--self-check")
        .output()
        .unwrap();
    assert!(
        cross_out.status.success(),
        "screened cross-libm self_check failed:\n{}",
        String::from_utf8_lossy(&cross_out.stderr)
    );

    let columns: Vec<NumericColumn> = (0..8)
        .map(|j| NumericColumn {
            name: format!("col_{j}"),
            min: 0.0,
            max: 100.0,
            constant: false,
        })
        .collect();
    let input = InputContract::new(
        columns,
        (0..8).collect(),
        TargetScale {
            name: "compressive_strength_mpa".into(),
            min: 2.33,
            scale: 80.27,
        },
    )
    .unwrap();
    let parts = BTreeMap::from([
        (
            PIPELINE_PART.to_string(),
            serde_json::to_vec(&PipelineState {
                dsl: dsl.into(),
                model: glibc_model,
            })
            .unwrap(),
        ),
        (INPUT_PART.to_string(), serde_json::to_vec(&input).unwrap()),
    ]);
    let pipeline = MathPipeline::decode(&parts, PIPELINE_PART).unwrap();

    let py_source = String::from_utf8(emit_math_pipeline(&pipeline).unwrap()).unwrap();
    let py_path = dir.path().join("winner.py");
    std::fs::write(&py_path, &py_source).unwrap();
    let status = Command::new(python())
        .args(["-I", "-B"])
        .arg(&py_path)
        .arg("--self-check")
        .output()
        .unwrap();
    assert!(
        status.status.success(),
        "winner.py --self-check failed:\n{}",
        String::from_utf8_lossy(&status.stderr)
    );

    let rs_source = crate::fitted_rs::emit_pipeline(&pipeline).unwrap();
    let rs_path = dir.path().join("winner.rs");
    let rs_bin = dir.path().join("winner-test");
    std::fs::write(&rs_path, &rs_source).unwrap();
    let compile = Command::new("rustc")
        .args(["--edition=2021", "--test"])
        .arg(&rs_path)
        .arg("-o")
        .arg(&rs_bin)
        .output()
        .unwrap();
    assert!(
        compile.status.success(),
        "rustc failed:\n{}",
        String::from_utf8_lossy(&compile.stderr)
    );
    let rs_run = Command::new(&rs_bin).output().unwrap();
    assert!(
        rs_run.status.success(),
        "winner.rs self_check failed:\n{}{}",
        String::from_utf8_lossy(&rs_run.stdout),
        String::from_utf8_lossy(&rs_run.stderr)
    );
}
