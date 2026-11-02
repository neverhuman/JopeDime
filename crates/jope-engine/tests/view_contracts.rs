use jope_parallel::{manifest::*, prepare::*};
use serde_json::json;
use std::{fs, path::PathBuf, process::Command};
#[path = "support/fit_authority.rs"]
mod authority;

const FENCED: &str = "mandatory fit authority service unavailable: fitting is fenced";

pub(super) fn fixture() -> (tempfile::TempDir, PathBuf) {
    let t = tempfile::tempdir().unwrap();
    let root = t.path();
    fs::create_dir(root.join("manifests")).unwrap();
    fs::write(root.join("Cargo.toml"), "[workspace]\n").unwrap();
    fs::create_dir_all(root.join("data/originals")).unwrap();
    let raw = (0..12)
        .map(|i| vec![i.to_string(), (100 + i * 3).to_string()])
        .collect::<Vec<_>>();
    let mut source = "x,target\n".to_string();
    for row in &raw {
        source.push_str(&format!("{},{}\n", row[0], row[1]));
    }
    fs::write(root.join("data/originals/raw.csv"), source).unwrap();
    let pp = Preprocessing::fit(&raw[..8], "regression").unwrap();
    let prepared = root.join("data/prepared/example");
    let dir = prepared.join("cut-0");
    fs::create_dir_all(&dir).unwrap();
    write_json(&dir.join("preprocessing.json"), &pp).unwrap();
    let tr = csv_bytes(&pp.transform(&raw[..8]).unwrap()).unwrap();
    let va = csv_bytes(&pp.transform(&raw[8..10]).unwrap()).unwrap();
    let p = json!({"schema":"parallel.prepared.v1","dataset":"example","family":"example","task":"regression","meta_partition":"meta_train","seed":SEED,"family_partition_frozen":false,"clean_evaluation":true,"clean_status":"raw_reconstructed","original_train_sha256":"0".repeat(64),"original_test_sha256":"1".repeat(64),"selected_train_indices":(0..10).collect::<Vec<_>>(),"selected_test_indices":[0,1],"target_last":true,"rows_total":12,"cuts":[{"index":0,"train_indices":(0..8).collect::<Vec<_>>(),"validation_indices":[8,9],"training_sha256":hash(&tr),"validation_sha256":hash(&va),"preprocessing_sha256":file_hash(&dir.join("preprocessing.json")).unwrap()}],"storage":{"source":"data/originals/raw.csv","source_sha256":file_hash(&root.join("data/originals/raw.csv")).unwrap(),"feature_columns":["x"],"target_column":"target","original_training_row_map":(0..10).collect::<Vec<_>>()}});
    write_json(&prepared.join("prepared.json"), &p).unwrap();
    (t, prepared)
}

pub(super) fn clock_search(
    prepared: &std::path::Path,
    output: &std::path::Path,
    evaluations: &str,
) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_jope-parallel"))
        .env("JOPE_CPU_EVALUATOR", env!("CARGO_BIN_EXE_jope-parallel"))
        .arg("--root")
        .arg(workspace_root(prepared).unwrap())
        .args([
            "search",
            "--cut",
            "0",
            "--seconds",
            "90",
            "--qualification",
            "--evaluations",
            evaluations,
        ])
        .arg("--prepared")
        .arg(prepared)
        .arg("--output")
        .arg(output)
        .env("RAYON_NUM_THREADS", "1")
        .env("OMP_NUM_THREADS", "1")
        .output()
        .unwrap()
}
pub(super) fn seeded_search(
    root: &std::path::Path,
    prepared: &std::path::Path,
    output: &std::path::Path,
    seed: &str,
) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_jope-parallel"))
        .env("JOPE_CPU_EVALUATOR", env!("CARGO_BIN_EXE_jope-parallel"))
        .arg("--root")
        .arg(root)
        .args([
            "search",
            "--cut",
            "0",
            "--evaluations",
            "2",
            "--seconds",
            "180",
            "--qualification",
            "--rng-seed",
            seed,
        ])
        .arg("--prepared")
        .arg(prepared)
        .arg("--output")
        .arg(output)
        .output()
        .unwrap()
}

mod admission;
mod resume;
mod seed;
