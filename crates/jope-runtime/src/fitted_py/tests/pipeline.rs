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
