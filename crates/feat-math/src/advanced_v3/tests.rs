use super::*;
use crate::GenomeIr;
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

/// A fresh scratch directory, removed when the returned handle drops (also on panic).
fn scratch_dir(tag: &str) -> tempfile::TempDir {
    tempfile::Builder::new()
        .prefix(&format!("feat-math-v3-{tag}-"))
        .tempdir()
        .unwrap()
}

fn write_test_csv(temp: &std::path::Path, content: &str) -> (std::path::PathBuf, RawTable) {
    let csv = temp.join("test.csv");
    std::fs::write(&csv, content).unwrap();
    let table = RawTable::from_csv(content).unwrap();
    (csv, table)
}

/// Parity is a hard invariant: a missing toolchain must FAIL the suite,
/// never silently skip it (tests/golden.rs precedent).
fn require_toolchains() {
    Command::new("python3")
        .arg("--version")
        .output()
        .expect("python3 must be on PATH");
    Command::new("rustc")
        .arg("--version")
        .output()
        .expect("rustc must be on PATH");
}

/// Emit, compile, run both artifacts on `csv`, and require BIT-EXACT
/// agreement (to_bits) with `predict_table` on every row.
fn assert_emitted_matches(
    model: &BoundModel,
    test: &RawTable,
    csv: &std::path::Path,
    temp: &std::path::Path,
    index: usize,
) {
    let emitted = emit_bound(model, &crate::transpile::EmitOptions::default()).unwrap();
    let py = temp.join(format!("model-{index}.py"));
    let rs = temp.join(format!("model-{index}.rs"));
    let bin = temp.join(format!("model-{index}"));
    std::fs::write(&py, emitted.python).unwrap();
    std::fs::write(&rs, emitted.rust).unwrap();
    let compile = Command::new("rustc")
        .args(["--edition=2021", "-O"])
        .arg(&rs)
        .arg("-o")
        .arg(&bin)
        .output()
        .unwrap();
    assert!(
        compile.status.success(),
        "artifact {index} rustc failed: {}",
        String::from_utf8_lossy(&compile.stderr)
    );
    let python = Command::new("python3").arg(&py).arg(csv).output().unwrap();
    let rust = Command::new(&bin).arg(csv).output().unwrap();
    assert!(
        python.status.success(),
        "artifact {index} python failed: {}",
        String::from_utf8_lossy(&python.stderr)
    );
    assert!(
        rust.status.success(),
        "artifact {index} rust failed: {}",
        String::from_utf8_lossy(&rust.stderr)
    );
    let python: Vec<f64> = String::from_utf8(python.stdout)
        .unwrap()
        .lines()
        .map(|line| line.parse().unwrap())
        .collect();
    let rust: Vec<f64> = String::from_utf8(rust.stdout)
        .unwrap()
        .lines()
        .map(|line| line.parse().unwrap())
        .collect();
    let runtime = model.predict_table(test).unwrap();
    assert_eq!(python.len(), runtime.len(), "artifact {index}: python rows");
    assert_eq!(rust.len(), runtime.len(), "artifact {index}: rust rows");
    for (row, ((python, rust), runtime)) in python.iter().zip(&rust).zip(&runtime).enumerate() {
        assert!(
            python.to_bits() == runtime.to_bits(),
            "artifact {index} row {row}: python={python:?} runtime={runtime:?}"
        );
        assert!(
            rust.to_bits() == runtime.to_bits(),
            "artifact {index} row {row}: rust={rust:?} runtime={runtime:?}"
        );
    }
}

mod admission;
mod emission;
mod emission_legacy;
mod heads;
mod schema;
