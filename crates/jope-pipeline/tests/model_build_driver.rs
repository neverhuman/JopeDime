//! The production entry point for the native bake-off and outer refit (SC-DEC-184W item 1).
//!
//! These fixtures drive the same code the `model-build-native` subcommand runs: the request file is
//! read back from disk, and the artifact is written through `artifact::write_new`.
//!
//! Authored for `cargo check --tests`. Execution is deferred to JA-06 / C14.
mod model_build_support;

use clap::Parser;
use jope_datasets::catalog::Task;
use jope_pipeline::cli::{run, Cli};
use jope_pipeline::model_build::artifact;
use jope_pipeline::model_build::driver::{
    read_native_request, run_native_request, NativeRequest, REQUEST_SCHEMA,
};
use jope_pipeline::model_build::BudgetCounters;
use model_build_support::*;
use std::path::PathBuf;

fn request(candidates: &[&str]) -> NativeRequest {
    let u = regression_union();
    NativeRequest {
        schema: REQUEST_SCHEMA.into(),
        plan: plan(Task::Regression, &u),
        data: u.data.clone(),
        selection_len: SELECTION,
        report_labels: u.report_labels.clone(),
        report_groups: vec![],
        candidates: candidates.iter().map(|c| c.to_string()).collect(),
        budget: BudgetCounters::UNLIMITED,
        wall_limit_ms: None,
    }
}

/// A fresh directory for one test's request and output files.
fn workdir(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("model-build-driver")
        .join(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn write_request(dir: &std::path::Path, request: &NativeRequest) -> PathBuf {
    let path = dir.join("request.json");
    std::fs::write(&path, serde_json::to_vec(request).unwrap()).unwrap();
    path
}

#[test]
fn the_request_file_runs_the_native_driver_and_writes_the_artifact_once() {
    let dir = workdir("driver-writes-once");
    let request_path = write_request(&dir, &request(&["rust.ridge"]));
    let output = dir.join("ensemble.json");
    let read = read_native_request(&request_path).unwrap();
    assert_eq!(read, request(&["rust.ridge"]));
    let out = run_native_request(&read, Some(&output)).unwrap();
    assert_eq!(out.receipt.selected.as_deref(), Some("rust.ridge"));
    assert_eq!(
        artifact::read(&output).unwrap(),
        out.persisted.clone().unwrap()
    );
    assert!(
        run_native_request(&read, Some(&output)).is_err(),
        "an existing artifact must not be overwritten"
    );
}

#[test]
fn the_cli_subcommand_reaches_the_same_driver_and_writes_the_artifact() {
    let dir = workdir("cli-writes");
    let request_path = write_request(&dir, &request(&["rust.ridge"]));
    let output = dir.join("ensemble.json");
    let cli = Cli::try_parse_from([
        "jope-pipeline",
        "model-build-native",
        "--request",
        request_path.to_str().unwrap(),
        "--output",
        output.to_str().unwrap(),
    ])
    .unwrap();
    run(cli).unwrap();
    assert!(artifact::read(&output).is_ok());
}

#[test]
fn a_request_without_an_output_path_runs_and_writes_nothing() {
    let dir = workdir("no-output");
    let out = run_native_request(&request(&["rust.ridge"]), None).unwrap();
    assert!(out.persisted.is_some());
    assert!(std::fs::read_dir(&dir).unwrap().count() == 0);
}

#[test]
fn an_unknown_candidate_is_refused_before_any_fit() {
    assert!(run_native_request(&request(&["rust.not_a_head"]), None).is_err());
}

#[test]
fn a_different_schema_is_refused() {
    let mut bad = request(&["rust.ridge"]);
    bad.schema = "jope.something-else/v1".into();
    assert!(run_native_request(&bad, None).is_err());
}

#[test]
fn report_labels_that_do_not_cover_the_report_rows_are_refused() {
    let mut bad = request(&["rust.ridge"]);
    bad.report_labels.pop();
    assert!(run_native_request(&bad, None).is_err());
}

#[test]
fn plan_labels_that_differ_from_the_data_targets_are_refused_by_the_driver() {
    let mut bad = request(&["rust.ridge"]);
    bad.plan.labels[0] = "999".into();
    assert!(run_native_request(&bad, None).is_err());
}

#[test]
fn an_unknown_request_field_is_refused_when_parsed() {
    let dir = workdir("unknown-field");
    let mut value = serde_json::to_value(request(&["rust.ridge"])).unwrap();
    value["surprise"] = serde_json::json!(1);
    let path = dir.join("request.json");
    std::fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
    assert!(read_native_request(&path).is_err());
}
