//! Deterministic fixtures shared by the model-build integration tests (SC-DEC-184M C06).
//!
//! Every fixture is generated from row indices, so a fresh process rebuilds byte-identical data.
//! The regression and binary unions each hold 20 selection rows and 6 report rows.
#![allow(dead_code)]

use jope_datasets::catalog::Task;
use jope_pipeline::model_build::native::{data_digest, NativeData, NativeFitter};
use jope_pipeline::model_build::{
    bakeoff_roster, BakeoffPlan, CandidateSpec, ExecutionIdentity, APPROVED_FOLD_COUNT,
};

pub const REQUEST: &str = "2222222222222222222222222222222222222222222222222222222222222222";
pub const SELECTION: usize = 20;
pub const REPORT: usize = 6;

/// Feature rows, targets and labels for both sides, in one row order: selection first.
pub struct Union {
    pub data: NativeData,
    pub selection_labels: Vec<String>,
    pub report_labels: Vec<String>,
    pub classes: Vec<String>,
}

fn union(
    features: Vec<Vec<f64>>,
    feature_names: &[&str],
    targets: Vec<f64>,
    labels: Vec<String>,
    classes: Vec<String>,
) -> Union {
    let n = SELECTION + REPORT;
    assert_eq!(features.len(), n);
    let data = NativeData {
        row_ids: (0..n).map(|i| format!("row-{i}")).collect(),
        feature_names: feature_names.iter().map(|s| s.to_string()).collect(),
        features,
        targets,
    };
    Union {
        data,
        selection_labels: labels[..SELECTION].to_vec(),
        report_labels: labels[SELECTION..].to_vec(),
        classes,
    }
}

pub fn regression_union() -> Union {
    let n = SELECTION + REPORT;
    let features: Vec<Vec<f64>> = (0..n)
        .map(|i| vec![i as f64, ((i * 7) % 5) as f64])
        .collect();
    let targets: Vec<f64> = features
        .iter()
        .enumerate()
        .map(|(i, r)| 2.0 * r[0] + 1.0 + 0.5 * r[1] + 0.1 * ((i % 4) as f64))
        .collect();
    let labels = targets.iter().map(|t| t.to_string()).collect();
    union(features, &["x0", "x1"], targets, labels, vec![])
}

pub fn binary_union() -> Union {
    let n = SELECTION + REPORT;
    let features: Vec<Vec<f64>> = (0..n)
        .map(|i| vec![i as f64, ((i * 3) % 4) as f64])
        .collect();
    let targets: Vec<f64> = (0..n)
        .map(|i| if (i * 5) % 7 >= 3 { 1.0 } else { 0.0 })
        .collect();
    let labels = targets
        .iter()
        .map(|t| if *t == 1.0 { "yes" } else { "no" }.to_string())
        .collect();
    union(
        features,
        &["x0", "x1"],
        targets,
        labels,
        vec!["no".into(), "yes".into()],
    )
}

/// The identity over the whole union: request, union bytes, both sides' rows and labels.
pub fn identity(u: &Union) -> ExecutionIdentity {
    let (selection_ids, report_ids) = u.data.row_ids.split_at(SELECTION);
    ExecutionIdentity::new(
        REQUEST,
        &data_digest(&u.data),
        selection_ids,
        &u.selection_labels,
        &[],
        report_ids,
        &u.report_labels,
    )
    .unwrap()
}

/// The same identity, with the report labels replaced.
pub fn identity_with_report_labels(u: &Union, report_labels: &[String]) -> ExecutionIdentity {
    let (selection_ids, report_ids) = u.data.row_ids.split_at(SELECTION);
    ExecutionIdentity::new(
        REQUEST,
        &data_digest(&u.data),
        selection_ids,
        &u.selection_labels,
        &[],
        report_ids,
        report_labels,
    )
    .unwrap()
}

pub fn split_fitter(task: Task, u: &Union, identity: &ExecutionIdentity) -> NativeFitter {
    NativeFitter::new_split(
        task,
        u.classes.clone(),
        u.data.clone(),
        SELECTION,
        &u.selection_labels,
        &u.report_labels,
        identity,
    )
    .unwrap()
}

/// The bake-off plan over the selection rows: five folds by row index, no groups.
pub fn plan(task: Task, u: &Union) -> BakeoffPlan {
    BakeoffPlan {
        task,
        classes: u.classes.clone(),
        row_ids: u.data.row_ids[..SELECTION].to_vec(),
        labels: u.selection_labels.clone(),
        folds: (0..SELECTION)
            .map(|i| (i % APPROVED_FOLD_COUNT) as u32)
            .collect(),
        group_ids: vec![],
        request_sha256: REQUEST.into(),
    }
}

pub fn report_ids(u: &Union) -> Vec<String> {
    u.data.row_ids[SELECTION..].to_vec()
}

/// The named candidates from the task's roster, in roster order.
pub fn roster_of(task: Task, ids: &[&str]) -> Vec<CandidateSpec> {
    bakeoff_roster(task)
        .into_iter()
        .filter(|spec| ids.contains(&spec.id))
        .collect()
}

pub fn selection_rows() -> Vec<usize> {
    (0..SELECTION).collect()
}

pub fn report_rows() -> Vec<usize> {
    (SELECTION..SELECTION + REPORT).collect()
}
