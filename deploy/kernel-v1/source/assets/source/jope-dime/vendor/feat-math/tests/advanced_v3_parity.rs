//! Public-boundary advanced-v3 parity lane.
//!
//! Private subsystem assertions live beside the implementation in
//! `src/advanced_v3/tests.rs`. This target proves that a caller can fit and
//! emit through only the public API, then obtain bit-identical predictions
//! from the runtime, generated Python, and generated Rust.

use feat_math::advanced_v3::ARTIFACT_SCHEMA;
use feat_math::transpile::EmitOptions;
use feat_math::{
    emit_bound, fit_table, ColumnSpec, DataSchema, GenomeIr, Objective, RawTable, RowSemantics,
    TaskSpec,
};
use std::path::Path;
use std::process::Command;

fn schema() -> DataSchema {
    DataSchema {
        columns: vec![
            ColumnSpec::numeric("amount"),
            ColumnSpec::categorical("country"),
        ],
        row_semantics: RowSemantics::Iid,
    }
}

fn training_table() -> RawTable {
    RawTable::new(
        vec!["amount".into(), "country".into()],
        vec![
            vec!["0".into(), "US".into()],
            vec!["0.1".into(), "US".into()],
            vec!["0.2".into(), "CA".into()],
            vec!["0.3".into(), "CA".into()],
            vec!["1.0".into(), "FR".into()],
            vec!["1.1".into(), "FR".into()],
            vec!["1.2".into(), "DE".into()],
            vec!["".into(), "DE".into()],
        ],
    )
    .unwrap()
}

fn require_toolchains() {
    for tool in ["python3", "rustc"] {
        let output = Command::new(tool)
            .arg("--version")
            .output()
            .unwrap_or_else(|error| panic!("{tool} must be on PATH: {error}"));
        assert!(output.status.success(), "{tool} --version failed");
    }
}

fn parse_predictions(output: &[u8]) -> Vec<f64> {
    String::from_utf8(output.to_vec())
        .unwrap()
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| line.parse().unwrap())
        .collect()
}

fn assert_case(
    index: usize,
    temp: &Path,
    csv_path: &Path,
    test: &RawTable,
    task: &TaskSpec,
    genome_source: &str,
    target: &[f64],
) {
    let genome = GenomeIr::parse(genome_source).unwrap();
    let model = fit_table(&genome, task, &schema(), &training_table(), target).unwrap();
    let runtime = model.predict_table(test).unwrap();
    let emitted = emit_bound(&model, &EmitOptions::default()).unwrap();
    assert!(emitted.manifest.contains(ARTIFACT_SCHEMA));

    let python_path = temp.join(format!("model-{index}.py"));
    let rust_path = temp.join(format!("model-{index}.rs"));
    let binary_path = temp.join(format!("model-{index}"));
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
        "artifact {index} rustc failed: {}",
        String::from_utf8_lossy(&compilation.stderr)
    );

    let python = Command::new("python3")
        .arg(&python_path)
        .arg(csv_path)
        .output()
        .unwrap();
    let rust = Command::new(&binary_path).arg(csv_path).output().unwrap();
    assert!(
        python.status.success(),
        "artifact {index} Python failed: {}",
        String::from_utf8_lossy(&python.stderr)
    );
    assert!(
        rust.status.success(),
        "artifact {index} Rust failed: {}",
        String::from_utf8_lossy(&rust.stderr)
    );

    let python = parse_predictions(&python.stdout);
    let rust = parse_predictions(&rust.stdout);
    assert_eq!(python.len(), runtime.len(), "artifact {index}: Python rows");
    assert_eq!(rust.len(), runtime.len(), "artifact {index}: Rust rows");
    for (row, ((python, rust), runtime)) in python.iter().zip(&rust).zip(&runtime).enumerate() {
        assert_eq!(
            python.to_bits(),
            runtime.to_bits(),
            "artifact {index} row {row}: Python/runtime"
        );
        assert_eq!(
            rust.to_bits(),
            runtime.to_bits(),
            "artifact {index} row {row}: Rust/runtime"
        );
    }
}

#[test]
fn public_runtime_python_and_rust_predictions_match_bitwise() {
    require_toolchains();
    let temp =
        std::env::temp_dir().join(format!("feat-math-v3-public-parity-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&temp);
    std::fs::create_dir_all(&temp).unwrap();
    let csv_path = temp.join("test.csv");
    let csv = "amount,country\n0.05,US\n1.3,\"unknown, place\"\n,DE\n";
    std::fs::write(&csv_path, csv).unwrap();
    let test = RawTable::from_csv(csv).unwrap();

    assert_case(
        0,
        &temp,
        &csv_path,
        &test,
        &TaskSpec::binary(Objective::LogLoss).unwrap(),
        "(pipe X (input numeric=median missing=indicator categorical=hybrid max_onehot=16 min_count=2 prior=10 folds=5) (logit_ridge class_weight=balanced l2=0.1))",
        &[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
    );
    assert_case(
        1,
        &temp,
        &csv_path,
        &test,
        &TaskSpec::regression(Objective::R2).unwrap(),
        "(pipe X (target kind=zscore) (input numeric=median categorical=frequency) (standardize) (target_bin_encode bins=8 prior=10 oof=5) (maxabs_scale) (ridge))",
        &[0.0, 0.2, 0.4, 0.6, 2.0, 2.2, 2.4, 2.6],
    );

    let _ = std::fs::remove_dir_all(&temp);
}
