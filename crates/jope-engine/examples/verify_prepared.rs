//! Replay pointer cuts against their shared backing tables without fitting GP
//! or opening original final-test CSVs. Produces a compact immutable receipt.
use anyhow::{ensure, Context, Result};
use jope_parallel::{manifest::*, prepare::*};
use rayon::prelude::*;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

fn verify(root: Option<&Path>, prepared: &Path) -> Result<Value> {
    let p: Prepared = read_json(&prepared.join("prepared.json"))?;
    ensure!(
        p.target_last && p.rows_total <= ROW_LIMIT,
        "invalid prepared shape/cap"
    );
    ensure!(p.storage.is_some(), "verification requires pointer storage");
    let mut cuts = Vec::new();
    for cut in &p.cuts {
        let training = match root {
            Some(root) => load_view_in_root(root, prepared, cut.index, true)?,
            None => load_view(prepared, cut.index, true)?,
        };
        let validation = match root {
            Some(root) => load_view_in_root(root, prepared, cut.index, false)?,
            None => load_view(prepared, cut.index, false)?,
        };
        ensure!(!training.is_empty() && !validation.is_empty(), "empty cut");
        cuts.push(json!({"cut":cut.index,"training_rows":training.len(),"validation_rows":validation.len(),"training_sha256":cut.training_sha256,"validation_sha256":cut.validation_sha256}));
    }
    for entry in walkdir::WalkDir::new(prepared) {
        ensure!(
            !entry?.path().extension().is_some_and(|ext| ext == "csv"),
            "cut contains copied CSV payload"
        );
    }
    Ok(
        json!({"dataset":p.dataset,"prepared":prepared,"prepared_sha256":file_hash(&prepared.join("prepared.json"))?,"family":p.family,"partition":p.meta_partition,"cuts":cuts,"passed":true,"original_final_test_csv_opened":false,"copied_dataset_bytes":0}),
    )
}

fn main() -> Result<()> {
    let args = std::env::args_os()
        .skip(1)
        .map(PathBuf::from)
        .collect::<Vec<_>>();
    ensure!(
        [2, 3].contains(&args.len()),
        "usage: verify_prepared PREPARED_COHORT_DIR OUTPUT_JSON [EXPLICIT_DATA_ROOT]"
    );
    ensure!(
        !args[1].exists(),
        "receipt exists; use a new immutable path"
    );
    let mut paths = std::fs::read_dir(&args[0])?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<std::io::Result<Vec<_>>>()?;
    paths.retain(|path| path.join("prepared.json").is_file());
    paths.sort();
    ensure!(!paths.is_empty(), "no prepared datasets found");
    let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build()?;
    let datasets = pool.install(|| {
        paths
            .par_iter()
            .map(|path| {
                verify(args.get(2).map(PathBuf::as_path), path)
                    .with_context(|| format!("verify {}", path.display()))
            })
            .collect::<Result<Vec<_>>>()
    })?;
    let cuts = datasets
        .iter()
        .map(|d| d["cuts"].as_array().unwrap().len())
        .sum::<usize>();
    write_json(
        &args[1],
        &json!({"schema":"parallel.prepared_replay.v1","passed":true,"datasets":datasets,"cut_views_replayed":cuts*2,"worker_threads":4,"original_final_test_csv_opened":false,"copied_dataset_bytes":0}),
    )?;
    println!(
        "{}",
        json!({"passed":true,"datasets":paths.len(),"cut_views_replayed":cuts*2,"receipt":args[1]})
    );
    Ok(())
}
