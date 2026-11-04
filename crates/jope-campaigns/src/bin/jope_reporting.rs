//! Native, non-scheduling operational materialization loop.
//!
//! This service owns periodic status/telemetry refresh and visibility into
//! terminal promotion candidates. It never consumes panels, schedules work,
//! advances a model pointer, or converts adaptive ratings into learning claims.

use anyhow::{ensure, Context, Result};
use clap::Parser;
use jope_dime_regressor::four_task::JopeFourTaskTelemetryV1;
use jope_dime_regressor::recovery_contracts::RecoveryContinuityObservationV1;
use jope_dime_regressor::selfplay_ingest;
use jope_dime_regressor::selfplay_runtime::{atomic_json, default_root_str, now_ms, sha256_file};
use jope_dime_regressor::selfplay_shadow::{
    Task, TaskPromotionEvaluationV3, TaskPromotionObservationSetV3,
};
use serde::Serialize;
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::thread;
use std::time::Duration;

const SCHEMA: &str = "JopeReportingStatus.v1";

#[derive(Parser, Debug)]
#[command(name = "jope-reporting")]
struct Args {
    #[arg(long, default_value = default_root_str())]
    root: PathBuf,
    #[arg(long, default_value_t = 30)]
    interval_seconds: u64,
    #[arg(long)]
    once: bool,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize)]
struct TerminalLearningInventory {
    terminal_evaluations: usize,
    promotion_cas_receipts: usize,
    immutable_learning_reports: usize,
    unpublished_terminal_candidates: Vec<String>,
}

#[derive(Debug, Serialize)]
struct ReportingStatus {
    schema: &'static str,
    generated_at_ms: u128,
    four_task_telemetry_path: PathBuf,
    four_task_telemetry_sha256: String,
    status_path: PathBuf,
    status_sha256: String,
    recovery_report_sha256: Option<String>,
    recovery_continuity_sha256: Option<String>,
    learning: TerminalLearningInventory,
    learning_reports_published: usize,
    scheduling_authority: bool,
    pointer_authority: bool,
    healthy: bool,
    blockers: Vec<String>,
}

#[path = "jope_reporting/learning.rs"]
mod learning;
#[path = "jope_reporting/recovery.rs"]
mod recovery;
#[cfg(test)]
#[path = "jope_reporting/tests.rs"]
mod tests;

pub(crate) use learning::*;
pub(crate) use recovery::*;

fn refresh(root: &Path) -> Result<ReportingStatus> {
    let mut status = selfplay_ingest::build_status(root)?;
    let generated_at_ms = now_ms();
    let recovery = materialize_recovery(root, generated_at_ms)?;
    let release_ready = recovery.get("release_gate").and_then(Value::as_str) == Some("ready");
    let failed_gates = recovery
        .get("gates")
        .and_then(Value::as_object)
        .into_iter()
        .flatten()
        .filter(|(_, value)| value.as_bool() != Some(true))
        .map(|(name, _)| name.clone())
        .collect::<Vec<_>>();
    status["release_gate"] = json!({
        "healthy": release_ready,
        "blocking_gates": failed_gates,
        "report": "ops/recovery-report.v2.json",
    });
    atomic_json(&root.join("ops/status-v3.json"), &status)?;
    atomic_json(&root.join("ops/status.json"), &status)?;
    let telemetry_path = root.join("ops/four-task-telemetry.v1.json");
    let status_path = root.join("ops/status-v3.json");
    let learning_reports_published = publish_terminal_learning_reports(root)?;
    let learning = terminal_learning_inventory(root);
    let mut blockers = Vec::new();
    let recovery_path = root.join("ops/recovery-report.v2.json");
    let continuity_path = root.join("ops/recovery-continuity.v1.json");
    let recovery = fs::read(&recovery_path)
        .ok()
        .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok());
    let continuity = fs::read(&continuity_path)
        .ok()
        .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok());
    if recovery.as_ref().is_none_or(|value| {
        value.get("schema").and_then(Value::as_str) != Some("JopeRecoveryReport.v2")
            || value.get("healthy").and_then(Value::as_bool) != Some(true)
            || value.get("release_gate").and_then(Value::as_str) != Some("ready")
    }) {
        blockers.push("recovery report is absent, invalid, or not ready".to_owned());
    }
    if continuity.as_ref().is_none_or(|value| {
        value.get("schema").and_then(Value::as_str) != Some("JopeRecoveryContinuity.v1")
            || value.get("release_ready").and_then(Value::as_bool) != Some(true)
    }) {
        blockers.push("86,400-second recovery continuity is not ready".to_owned());
    }
    if !learning.unpublished_terminal_candidates.is_empty() {
        blockers.push("terminal candidates await immutable JopeLearningReport.v2".to_owned());
    }
    let report = ReportingStatus {
        schema: SCHEMA,
        generated_at_ms,
        four_task_telemetry_sha256: sha256_file(&telemetry_path)?,
        status_sha256: sha256_file(&status_path)?,
        recovery_report_sha256: recovery_path
            .is_file()
            .then(|| sha256_file(&recovery_path))
            .transpose()?,
        recovery_continuity_sha256: continuity_path
            .is_file()
            .then(|| sha256_file(&continuity_path))
            .transpose()?,
        four_task_telemetry_path: telemetry_path,
        status_path,
        learning,
        learning_reports_published,
        scheduling_authority: false,
        pointer_authority: false,
        healthy: blockers.is_empty(),
        blockers,
    };
    atomic_json(
        &root.join("ops/reporting-status.v1.json"),
        &serde_json::to_value(&report)?,
    )?;
    Ok(report)
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        args.interval_seconds > 0 && args.interval_seconds <= 300,
        "reporting interval must be within 1..=300 seconds"
    );
    loop {
        let report = refresh(&args.root)?;
        println!("{}", serde_json::to_string(&report)?);
        if args.once {
            return Ok(());
        }
        thread::sleep(Duration::from_secs(args.interval_seconds));
    }
}
