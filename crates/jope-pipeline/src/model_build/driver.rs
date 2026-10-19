//! Production entry point for the native bake-off and outer refit (SC-DEC-184W item 1).
//!
//! The `model-build-native` CLI subcommand reads one `NativeRequest` file and calls
//! `run_native_request`. The request carries the validated plan, the data, the report side and the
//! budget. The identity is derived here from those inputs, never supplied separately, and it must
//! pass `AdmittedIdentity::admit` before any fitter is built. The record is written only when an
//! output path is given, and never overwrites an existing file.
//!
//! This path runs the native heads only. It is not a model-build route: the neural-to-GP stage and
//! its checkpoint are not involved, so nothing here reports a neural result.
use super::automl::{run_automl, AutomlOutput};
use super::candidates::{bakeoff_roster, CandidateSpec};
use super::identity::{AdmittedIdentity, ExecutionIdentity};
use super::native::{data_digest, NativeData, NativeFitter};
use super::selection::{BakeoffPlan, BudgetCounters, EvaluationBudget};
use anyhow::{anyhow, ensure, Context, Result};
use jope_datasets::catalog::Task;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::Duration;

pub const REQUEST_SCHEMA: &str = "jope.model-build-native-request/v1";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeRequest {
    pub schema: String,
    pub plan: BakeoffPlan,
    pub data: NativeData,
    /// Rows `0..selection_len` of `data` are the selection side, matching `plan.row_ids`.
    pub selection_len: usize,
    /// Labels of the report rows, which follow the selection rows in `data`.
    pub report_labels: Vec<String>,
    /// Groups of the report rows. Required when the plan is grouped.
    #[serde(default)]
    pub report_groups: Vec<String>,
    /// Candidate ids from the task's roster. Each must name a roster entry.
    pub candidates: Vec<String>,
    pub budget: BudgetCounters,
    #[serde(default)]
    pub wall_limit_ms: Option<u64>,
}

pub fn read_native_request(path: &Path) -> Result<NativeRequest> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("parsing {}", path.display()))
}

/// Runs the admitted native bake-off and outer refit for one request. `output`, when given, names a
/// new file for the persisted ensemble record.
pub fn run_native_request(request: &NativeRequest, output: Option<&Path>) -> Result<AutomlOutput> {
    ensure!(
        request.schema == REQUEST_SCHEMA,
        "unknown request schema {}",
        request.schema
    );
    let rows = request.data.row_ids.len();
    ensure!(
        request.selection_len <= rows
            && request.selection_len + request.report_labels.len() == rows,
        "the report labels do not cover the report rows of the data"
    );
    let plan = &request.plan;
    let (selection_ids, report_ids) = request.data.row_ids.split_at(request.selection_len);
    let identity = ExecutionIdentity::new(
        &plan.request_sha256,
        &plan.digest(),
        &data_digest(&request.data),
        selection_ids,
        &plan.labels,
        &plan.group_ids,
        report_ids,
        &request.report_labels,
        &request.report_groups,
    )?;
    let admitted = AdmittedIdentity::admit(
        identity,
        plan,
        &request.data,
        request.selection_len,
        &request.report_labels,
    )?;
    let fitter = NativeFitter::for_plan(
        plan,
        request.data.clone(),
        request.selection_len,
        &request.report_labels,
        admitted.identity(),
    )?;
    let roster = roster_for(plan.task, &request.candidates)?;
    let mut budget = EvaluationBudget::with_limits(request.budget);
    if let Some(ms) = request.wall_limit_ms {
        budget = budget.with_wall_limit(Duration::from_millis(ms));
    }
    run_automl(
        plan,
        &roster,
        &fitter,
        admitted.identity(),
        report_ids,
        &request.report_labels,
        &mut budget,
        output,
    )
}

fn roster_for(task: Task, ids: &[String]) -> Result<Vec<CandidateSpec>> {
    let roster = bakeoff_roster(task);
    ids.iter()
        .map(|id| {
            roster
                .iter()
                .find(|spec| spec.id == id)
                .cloned()
                .ok_or_else(|| anyhow!("candidate {id} is not in the {task:?} roster"))
        })
        .collect()
}
