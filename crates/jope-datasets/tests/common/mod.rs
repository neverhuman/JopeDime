//! A synthetic `quant`-shaped tree for the integration tests.
#![allow(dead_code)]

use serde_json::{json, Value};
use std::path::Path;

pub const A: &str = "0123456789abcdef";
pub const B: &str = "b0b0b0b0b0b0b0b0";
pub const C: &str = "cccccccccccccccc";
pub const D: &str = "d1d2d3d4d5d6d7d8";
pub const E: &str = "e0e0e0e0e0e0e0e0";
pub const F: &str = "f0f0f0f0f0f0f0f0";

/// A 64-hex canonical sha whose first 16 hex digits are `hash`.
pub fn canonical(hash: &str) -> String {
    format!("{hash}{}", "0".repeat(48))
}

pub fn write_dataset(dir: &Path, train: &str, test: &str, meta: Value) {
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(dir.join("train.csv"), train).unwrap();
    std::fs::write(dir.join("test.csv"), test).unwrap();
    std::fs::write(
        dir.join("meta.json"),
        serde_json::to_vec_pretty(&meta).unwrap(),
    )
    .unwrap();
}

pub fn real_meta(hash: &str, seed: u64) -> Value {
    json!({
        "sha256": canonical(hash),
        "hash_algorithm": "sha256:first16",
        "dataset_hash": hash,
        "origin": "real",
        "task_type": "regression",
        "source_platform": "pmlb",
        "source_dataset_id": format!("ds-{hash}"),
        "source_url": format!("https://example.invalid/{hash}"),
        "source_license": "MIT (PMLB)",
        "retrieved_at": "2026-06-27T23:54:41+00:00",
        "sampling": {"method": "quota", "seed": seed},
        "target_column": "target"
    })
}

pub fn csv(rows: u64, columns: u64, salt: u64) -> String {
    let mut s = String::new();
    for r in 0..rows {
        let line: Vec<String> = (0..columns)
            .map(|c| format!("{}.{}", (r * 7 + c * 3 + salt) % 10, salt))
            .collect();
        s.push_str(&line.join(","));
        s.push('\n');
    }
    s
}

/// Layout:
/// - `regression/A` real, good
/// - `regression/synth_B` synthetic, good
/// - `regression/C` meta.sha256 disagrees with the directory name (reject)
/// - `regression/F` incomplete (train.csv only; reject)
/// - `regression/_dupes_quarantine/E` quarantined copy (never ingested)
/// - `timeseries/A` byte-identical copy of `regression/A` (dataset-hash duplicate -> aliases)
/// - `<root>/A` root-level copy (duplicate by path rule, not hashed)
/// - `binary/D` and `synthetic/binary/D` identical (binary duplicate by path rule)
pub fn build_tree(root: &Path) {
    let a = (csv(8, 4, 1), csv(2, 4, 2));
    write_dataset(
        &root.join("regression").join(A),
        &a.0,
        &a.1,
        real_meta(A, 11),
    );
    write_dataset(
        &root.join("timeseries").join(A),
        &a.0,
        &a.1,
        real_meta(A, 11),
    );
    write_dataset(&root.join(A), &a.0, &a.1, real_meta(A, 11));
    write_dataset(
        &root.join("regression").join(format!("synth_{B}")),
        &csv(6, 3, 3),
        &csv(3, 3, 4),
        json!({
            "sha256": canonical(B), "hash_algorithm": "sha256:first16", "origin": "synthetic",
            "task_type": "regression", "generator_name": "synthetic_regression_mixed",
            "generator_version": "2.0.0", "generator_seed": 5, "generator_attempt_seed": 6,
            "generator_recipe": {"latent_dim": 3}, "sampling": {"method": "generated"}
        }),
    );
    write_dataset(
        &root.join("regression").join(C),
        &csv(5, 2, 5),
        &csv(2, 2, 6),
        real_meta(A, 12),
    );
    std::fs::create_dir_all(root.join("regression").join(F)).unwrap();
    std::fs::write(root.join("regression").join(F).join("train.csv"), "1,2\n").unwrap();
    write_dataset(
        &root.join("regression").join("_dupes_quarantine").join(E),
        &csv(4, 2, 7),
        &csv(1, 2, 8),
        real_meta(E, 13),
    );
    let d = (csv(10, 5, 9), csv(4, 5, 10));
    let mut dm = real_meta(D, 14);
    dm["task_type"] = json!("binary");
    dm["origin"] = json!("synthetic");
    dm["generator_name"] = json!("synthetic_binary_mixed_tabular");
    dm["generator_version"] = json!("0.2.0");
    write_dataset(&root.join("binary").join(D), &d.0, &d.1, dm.clone());
    write_dataset(
        &root.join("synthetic").join("binary").join(D),
        &d.0,
        &d.1,
        dm,
    );
    std::fs::write(
        root.join("regression").join("AGENT_CHAT.md"),
        "not a dataset\n",
    )
    .unwrap();
}
