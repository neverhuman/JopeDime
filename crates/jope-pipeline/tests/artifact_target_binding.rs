use feat_math::{artifact_manifest::*, evolve::FaultPolicy};
use jope_pipeline::{
    clock::Clock,
    emit,
    kernel2::{publication, receipt::Receipt, run, search::Refinement},
    run::{Policy, RunOptions},
    search::Budget,
};
use std::{fs, path::Path};
#[path = "../../feat-math/tests/go_support/mod.rs"]
mod go_support;
#[path = "../../feat-math/tests/typescript_support/mod.rs"]
mod typescript_support;
fn scratch() -> tempfile::TempDir {
    let root = Path::new(env!("CARGO_TARGET_TMPDIR"));
    fs::create_dir_all(root).unwrap();
    tempfile::tempdir_in(root).unwrap()
}
fn rows() -> Vec<Vec<f64>> {
    (0..10)
        .map(|i| vec![i as f64, 2.0 * i as f64 + 1.0])
        .collect()
}
#[path = "artifact_target_binding/publication.rs"]
mod publication_tests;
#[path = "artifact_target_binding/runtime.rs"]
mod runtime;
#[path = "artifact_target_binding/writer.rs"]
mod writer;
