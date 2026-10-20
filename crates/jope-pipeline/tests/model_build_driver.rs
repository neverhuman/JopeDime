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
        budget: BudgetCounters {
            candidate_evaluations: 16,
            fold_fits: 80,
            outer_refits: 16,
            ensemble_attempts: 1,
            neural_gp_work_units: 0,
        },
        wall_limit_ms: None,
    }
}

/// A fresh directory for one test's request and output files.
fn workdir(name: &str) -> tempfile::TempDir {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    std::fs::create_dir_all(&root).unwrap();
    tempfile::Builder::new()
        .prefix(name)
        .tempdir_in(root)
        .unwrap()
}

fn write_request(dir: &std::path::Path, request: &NativeRequest) -> PathBuf {
    let path = dir.join("request.json");
    std::fs::write(&path, serde_json::to_vec(request).unwrap()).unwrap();
    path
}

#[test]
fn the_request_file_runs_the_native_driver_and_writes_the_artifact_once() {
    let dir = workdir("driver-writes-once");
    let request_path = write_request(dir.path(), &request(&["rust.ridge"]));
    let output = dir.path().join("ensemble.json");
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
    let request_path = write_request(dir.path(), &request(&["rust.ridge"]));
    let output = dir.path().join("ensemble.json");
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
    assert!(std::fs::read_dir(dir.path()).unwrap().count() == 0);
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
    let path = dir.path().join("request.json");
    std::fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
    assert!(read_native_request(&path).is_err());
}

#[test]
fn native_request_unlimited_work_caps_are_refused_at_admission() {
    for index in 0..5 {
        let mut req = request(&["rust.ridge"]);
        let caps = [
            &mut req.budget.candidate_evaluations,
            &mut req.budget.fold_fits,
            &mut req.budget.outer_refits,
            &mut req.budget.ensemble_attempts,
            &mut req.budget.neural_gp_work_units,
        ];
        *caps.into_iter().nth(index).unwrap() = u64::MAX;
        let result = run_native_request(&req, None);
        assert!(
            result.is_err(),
            "unlimited native work cap {index} was accepted"
        );
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("budget must be finite"));
    }
}

#[test]
fn native_request_maximum_wall_sentinel_is_refused_at_admission() {
    let mut req = request(&["rust.ridge"]);
    req.wall_limit_ms = Some(u64::MAX);
    req.budget.candidate_evaluations = 0;
    let error = run_native_request(&req, None).unwrap_err().to_string();
    assert!(
        error.contains("wall budget must be finite"),
        "wrong refusal: {error}"
    );
}

#[test]
fn native_request_duplicate_roster_is_refused_before_budgeted_work() {
    let mut req = request(&["rust.ridge", "rust.ridge"]);
    req.budget.candidate_evaluations = 0;
    let error = run_native_request(&req, None).unwrap_err().to_string();
    assert!(
        error.contains("duplicate candidate rust.ridge"),
        "wrong refusal: {error}"
    );
}

#[test]
fn native_request_finite_work_without_wall_limit_still_executes() {
    let req = request(&["rust.ridge"]);
    assert_eq!(req.wall_limit_ms, None);
    let result = run_native_request(&req, None).unwrap();
    assert_eq!(result.receipt.selected.as_deref(), Some("rust.ridge"));
}

#[test]
fn native_request_identity_rejects_the_old_unframed_protocol() {
    use jope_pipeline::model_build::identity::{AdmittedIdentity, ExecutionIdentity};
    use jope_pipeline::model_build::native::data_digest;
    let req = request(&["rust.ridge"]);
    let mut identity = ExecutionIdentity::new(
        &req.plan.request_sha256,
        &req.plan.digest(),
        &data_digest(&req.data),
        &req.plan.row_ids,
        &req.plan.labels,
        &req.plan.group_ids,
        &req.data.row_ids[req.selection_len..],
        &req.report_labels,
        &req.report_groups,
    )
    .unwrap();
    assert_eq!(identity.protocol, "outer-selection-refit-report/v2");
    AdmittedIdentity::admit(
        identity.clone(),
        &req.plan,
        &req.data,
        req.selection_len,
        &req.report_labels,
    )
    .unwrap();
    identity.protocol = "outer-selection-refit-report/v1".into();
    let error = AdmittedIdentity::admit(
        identity,
        &req.plan,
        &req.data,
        req.selection_len,
        &req.report_labels,
    )
    .unwrap_err();
    assert!(error.to_string().contains("protocol differs"));
}
