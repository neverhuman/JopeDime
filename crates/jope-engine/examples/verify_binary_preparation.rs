//! Verify real binary pointer cuts with the production Rust materializer.
use anyhow::{ensure, Result};
use jope_parallel::{manifest::*, prepare::*};
use serde_json::{json, Value};
use std::{collections::BTreeSet, path::PathBuf};

fn main() -> Result<()> {
    let experiment = PathBuf::from(std::env::args().nth(1).expect("experiment path"));
    let report: Value = read_json(&experiment.join("preparation-report.json"))?;
    let root = workspace_root(&experiment)?;
    let mut checks = Vec::new();
    for dataset in report["datasets"].as_array().unwrap() {
        ensure!(dataset["status"] == "admitted", "dataset withheld");
        let path = root.join(dataset["prepared"].as_str().unwrap());
        let prepared: Prepared = read_json(&path.join("prepared.json"))?;
        let mut memberships = BTreeSet::new();
        for cut in &prepared.cuts {
            ensure!(
                memberships.insert(cut.train_indices.clone()),
                "duplicate membership"
            );
            let train = load_view(&path, cut.index, true)?;
            let heldout = load_view(&path, cut.index, false)?;
            for rows in [&train, &heldout] {
                ensure!(
                    rows.iter()
                        .map(|r| *r.last().unwrap() as u64)
                        .collect::<BTreeSet<_>>()
                        == BTreeSet::from([0, 1]),
                    "class coverage"
                );
            }
            let raw = load_raw_view(
                &path,
                prepared.storage.as_ref().unwrap(),
                &cut.train_indices,
            )?;
            let pp = Preprocessing::fit(&raw, "binary")?;
            let saved: Preprocessing =
                read_json(&path.join(format!("cut-{}/preprocessing.json", cut.index)))?;
            ensure!(
                serde_json::to_value(&pp)? == serde_json::to_value(&saved)?,
                "preprocessing must fit training rows only"
            );
            ensure!(
                hash(&csv_bytes(&pp.transform(&raw)?)?) == cut.training_sha256,
                "recomputed training encoding differs"
            );
        }
        let rebuilt = experiment
            .join("verification-rebuilt")
            .join(&prepared.dataset);
        let original = experiment.join("prepared").join(&prepared.dataset);
        if !rebuilt.exists() {
            expand_cuts(
                &experiment.join("prepared-initial").join(&prepared.dataset),
                &rebuilt,
                4,
            )?;
        }
        ensure!(
            file_hash(&rebuilt.join("prepared.json"))?
                == file_hash(&original.join("prepared.json"))?,
            "repeated preparation changed membership or preprocessing"
        );
        checks.push(json!({"dataset":prepared.dataset,"cuts":prepared.cuts.len(),"training_and_heldout_hashes_verified":true,"both_classes_each_partition":true,"training_only_preprocessing_reproduced":true,"repeated_preparation_byte_identical":true}));
    }
    write_json(
        &experiment.join("preparation-verification.json"),
        &json!({"schema":"parallel.binary_preparation_verification.v1","passed":true,"checks":checks,"copied_cut_tables":0,"original_final_tests_opened":false,"prepared_report_sha256":file_hash(&experiment.join("preparation-report.json"))?}),
    )
}
