//! jope-pipeline must depend on neither jope-learning nor jope-parallel (the engine).
//!
//! jope-learning already depends on jope-pipeline, so the reverse edge would be a cycle: the
//! learning store's HWM ingest and champion bank consume pipeline receipts and seeds files, and
//! the pipeline reads only those files back. The engine's GP path cannot measure in a shipped
//! binary (see `search.rs`), so the pipeline must never reach it either.
//!
//! Two checks, because Cargo answers two different questions:
//!
//! * `cargo metadata --no-deps`: jope-pipeline's own manifest declares no dependency on either
//!   crate in any section (normal, build, dev, optional or not). This is the edge that would make
//!   a cycle.
//! * `cargo tree -p jope-pipeline`: the package's resolved closure, with default features and with
//!   every jope-pipeline feature, reaches neither. `cargo metadata`'s `resolve` cannot answer this:
//!   it unifies features across the whole workspace, and there hyperion's own default
//!   `batch-worker` feature (which links the engine for the `hyperion-batch-worker` binary) shows
//!   up as `jope-pipeline -> jope-policy -> hyperion -> jope-parallel` although jope-policy
//!   depends on hyperion with default features off and no jope-pipeline build ever compiles the
//!   engine.

use serde_json::Value;
use std::collections::BTreeSet;
use std::process::Command;

const FORBIDDEN: [&str; 2] = ["jope-learning", "jope-parallel"];
const MANIFEST: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../Cargo.toml");

fn cargo() -> Command {
    Command::new(std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_owned()))
}

fn run(command: &mut Command) -> Vec<u8> {
    let output = command.output().expect("run cargo");
    assert!(
        output.status.success(),
        "{command:?} failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    output.stdout
}

/// Every package name a workspace package's manifest declares, in every dependency section.
fn declared(metadata: &Value, package: &str) -> BTreeSet<String> {
    metadata["packages"]
        .as_array()
        .unwrap()
        .iter()
        .find(|p| p["name"] == package)
        .unwrap_or_else(|| panic!("{package} is not a workspace package"))["dependencies"]
        .as_array()
        .unwrap()
        .iter()
        .map(|d| d["name"].as_str().unwrap().to_owned())
        .collect()
}

/// The package names `cargo tree -p jope-pipeline` resolves (normal and build edges).
fn resolved(features: &[&str]) -> BTreeSet<String> {
    let stdout = run(cargo()
        .args([
            "tree",
            "--locked",
            "--offline",
            "--manifest-path",
            MANIFEST,
            "-p",
            "jope-pipeline",
            "--edges",
            "normal,build",
            "--prefix",
            "none",
            "--format",
            "{p}",
        ])
        .args(features));
    String::from_utf8(stdout)
        .unwrap()
        .lines()
        .filter_map(|line| line.split_whitespace().next())
        .map(str::to_owned)
        .collect()
}

#[test]
fn jope_pipeline_declares_no_dependency_on_the_learning_store_or_the_engine() {
    let metadata: Value = serde_json::from_slice(&run(cargo().args([
        "metadata",
        "--format-version",
        "1",
        "--no-deps",
        "--locked",
        "--offline",
        "--manifest-path",
        MANIFEST,
    ])))
    .unwrap();
    let pipeline = declared(&metadata, "jope-pipeline");
    for forbidden in FORBIDDEN {
        assert!(
            !pipeline.contains(forbidden),
            "jope-pipeline's manifest declares {forbidden}; the learning store and the engine \
             consume pipeline files, never the reverse"
        );
    }
    // Positive controls: the declared set is real, and the edge that makes the reverse a cycle
    // is where it should be.
    for expected in [
        "feat-math",
        "jope-datasets",
        "rayon",
        "jope-policy",
        "tempfile",
    ] {
        assert!(
            pipeline.contains(expected),
            "declared set misses {expected}"
        );
    }
    assert!(declared(&metadata, "jope-learning").contains("jope-pipeline"));
}

#[test]
fn jope_pipeline_resolves_neither_the_learning_store_nor_the_engine_under_any_of_its_features() {
    for features in [
        vec![],
        vec!["--features", "neural-search,qualified-context,timing-study"],
    ] {
        let reached = resolved(&features);
        assert!(reached.contains("feat-math") && reached.contains("jope-datasets"));
        for forbidden in FORBIDDEN {
            assert!(
                !reached.contains(forbidden),
                "jope-pipeline {features:?} resolves {forbidden}"
            );
        }
        // The optional policy stack is in the closure exactly when its feature is on.
        assert_eq!(reached.contains("jope-policy"), !features.is_empty());
    }
}
