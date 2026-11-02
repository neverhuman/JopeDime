use super::super::csv::RawTable;
use super::super::pipeline::fit_table;
use super::super::schema::{ColumnSpec, DataSchema, RowSemantics};
use super::super::types::{Objective, TaskSpec};
use super::*;
use crate::GenomeIr;
use std::process::Command;

fn ordered_schema() -> DataSchema {
    DataSchema {
        columns: vec![
            ColumnSpec::categorical("group"),
            ColumnSpec::numeric("time"),
            ColumnSpec::numeric("value"),
        ],
        row_semantics: RowSemantics::Ordered {
            group_by: vec!["group".to_string()],
            order_by: vec!["time".to_string()],
        },
    }
}

fn predictions(output: &[u8]) -> Vec<f64> {
    String::from_utf8(output.to_vec())
        .unwrap()
        .lines()
        .map(|line| line.parse().unwrap())
        .collect()
}

fn assert_row_op_artifacts(op: &str, index: usize) {
    let schema = ordered_schema();
    let training = RawTable::new(
        vec!["group".into(), "time".into(), "value".into()],
        vec![
            vec!["A".into(), "1".into(), "1".into()],
            vec!["A".into(), "2".into(), "2".into()],
            vec!["B".into(), "1".into(), "10".into()],
            vec!["B".into(), "2".into(), "20".into()],
            vec!["nan".into(), "1".into(), "5".into()],
            vec!["nan".into(), "2".into(), "6".into()],
        ],
    )
    .unwrap();
    let task = TaskSpec::regression(Objective::R2).unwrap();
    let genome = GenomeIr::parse(&format!(
        "(pipe X (input numeric=median categorical=ordered prior=0) ({op}) (ridge))"
    ))
    .unwrap();
    let model = fit_table(
        &genome,
        &task,
        &schema,
        &training,
        &[1.0, 2.0, 10.0, 20.0, 5.0, 6.0],
    )
    .unwrap();
    // Reverse chronological caller order, equal-order A rows (ordinal
    // tie-break), fitted A/B tails, and an unseen all-missing C value.
    let csv = "group,time,value\nA,4,4\nA,3,3\nA,3,5\nB,3,30\nC,1,\nD,1,10000000000000000\nD,2,1\nD,3,-10000000000000000\n,1,\n,2,\nnan,3,inf\nnan,4,nan\n";
    let test = RawTable::from_csv(csv).unwrap();
    let runtime = model.predict_table(&test).unwrap();
    let emitted = emit_bound(&model, &crate::transpile::EmitOptions::default()).unwrap();
    let portable_error = emit_bound(
        &model,
        &crate::transpile::EmitOptions {
            math_kernel: crate::transpile::MathKernel::Portable,
            ..crate::transpile::EmitOptions::default()
        },
    )
    .expect_err("bound emission must not silently ignore Portable");
    assert!(portable_error.to_string().contains("portable"));

    let scratch = tempfile::Builder::new()
        .prefix(&format!("feat-math-v3-row-op-emission-{index}-"))
        .tempdir()
        .unwrap();
    let temp = scratch.path();
    if super::typescript_support::enabled() {
        let bundle =
            crate::typescript::emit_bound(&model, &crate::transpile::EmitOptions::default())
                .unwrap();
        super::typescript_support::run(
            &temp.join("typescript"),
            &bundle.source,
            &bundle.kernel,
            &serde_json::json!({"rows":test.rows}),
            &runtime,
            true,
        );
    }
    if super::go_support::enabled() {
        let bundle =
            crate::go::emit_bound(&model, &crate::transpile::EmitOptions::default()).unwrap();
        super::go_support::run(
            &temp.join("go"),
            &bundle.source,
            &bundle.kernel,
            &serde_json::json!({"rows":test.rows}),
            &runtime,
            true,
        );
    }
    let csv_path = temp.join("test.csv");
    let python_path = temp.join("model.py");
    let rust_path = temp.join("model.rs");
    let binary_path = temp.join("model");
    std::fs::write(&csv_path, csv).unwrap();
    std::fs::write(&python_path, emitted.python).unwrap();
    std::fs::write(&rust_path, emitted.rust).unwrap();
    let compilation = Command::new("rustc")
        .args(["--edition=2021", "-O"])
        .arg(&rust_path)
        .arg("-o")
        .arg(&binary_path)
        .output()
        .unwrap();
    assert!(
        compilation.status.success(),
        "generated Rust failed: {}",
        String::from_utf8_lossy(&compilation.stderr)
    );
    let python = Command::new("python3")
        .arg(&python_path)
        .arg(&csv_path)
        .output()
        .unwrap();
    let rust = Command::new(&binary_path).arg(&csv_path).output().unwrap();
    assert!(
        python.status.success(),
        "generated Python failed: {}",
        String::from_utf8_lossy(&python.stderr)
    );
    assert!(
        rust.status.success(),
        "generated Rust failed: {}",
        String::from_utf8_lossy(&rust.stderr)
    );
    let python_predictions = predictions(&python.stdout);
    let rust_predictions = predictions(&rust.stdout);
    assert_eq!(python_predictions.len(), runtime.len());
    assert_eq!(rust_predictions.len(), runtime.len());
    for (row, ((python, rust), runtime)) in python_predictions
        .iter()
        .zip(&rust_predictions)
        .zip(&runtime)
        .enumerate()
    {
        assert_eq!(python.to_bits(), runtime.to_bits(), "Python row {row}");
        assert_eq!(rust.to_bits(), runtime.to_bits(), "Rust row {row}");
    }
    if op.starts_with("roll") {
        assert_eq!(
            python_predictions[7].to_bits(),
            runtime[7].to_bits(),
            "Python cancellation-window row must use the runtime's sequential fold"
        );
        assert_eq!(
            rust_predictions[7].to_bits(),
            runtime[7].to_bits(),
            "Rust cancellation-window row must use the runtime's sequential fold"
        );
    }

    let stale_csv = "group,time,value\nA,2,99\n";
    let stale_table = RawTable::from_csv(stale_csv).unwrap();
    assert!(model.predict_table(&stale_table).is_err());
    let stale_path = temp.join("stale.csv");
    std::fs::write(&stale_path, stale_csv).unwrap();
    let stale_python = Command::new("python3")
        .arg(&python_path)
        .arg(&stale_path)
        .output()
        .unwrap();
    let stale_rust = Command::new(&binary_path)
        .arg(&stale_path)
        .output()
        .unwrap();
    assert!(
        !stale_python.status.success(),
        "generated Python accepted a non-future fitted-tail row"
    );
    assert!(
        !stale_rust.status.success(),
        "generated Rust accepted a non-future fitted-tail row"
    );

    let mut corrupt_model = model.clone();
    let BoundPredictor::Pipeline { ops, .. } = &mut corrupt_model.predictor else {
        unreachable!()
    };
    match &mut ops[0] {
        V3Op::Lag { tails, .. } | V3Op::Roll { tails, .. } => {
            tails[0].last_order[0] = SemanticKey::Missing;
        }
        _ => unreachable!(),
    }
    assert!(
        emit_bound(&corrupt_model, &crate::transpile::EmitOptions::default())
            .unwrap_err()
            .to_string()
            .contains("tail state hash mismatch"),
        "bound emission must reject corrupt fitted-tail state"
    );
}

#[test]
fn emitted_lag_and_roll_use_fitted_group_tails() {
    assert_row_op_artifacts("lag k=2", 0);
    assert_row_op_artifacts("roll k=3", 1);
}
