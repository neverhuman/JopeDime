//! Connected-run events: one schema, two destinations.
//!
//! A connected CLI reports the same facts about a local run to two places:
//!
//! - the owner's VEOX account, as `veox.connected-run-events/v1` (`POST /api/cli/runs/:id/events`),
//!   where the run page draws its leaderboard and data profile; and
//! - veox-telemetry (`POST /v1/telemetry`), de-identified: bucketed counts, scores in tenths, a
//!   random run pseudonym, `source: cli`. Only when the person turned telemetry on.
//!
//! Neither copy carries a row of the dataset. A profile is one summary line per column (name, type,
//! missing and distinct counts) and only the account copy has the names.
use anyhow::{Context, Result};
use jope_api::{events::EventBody, ids::DatasetId, RunSnapshot};
use jope_client::Transport;
use serde_json::{json, Value};
use std::collections::BTreeMap;

pub const SCHEMA: &str = "veox.connected-run-events/v1";
/// The app's per-request ceiling.
pub const APP_BATCH: usize = 500;
/// Smaller than ingest's 500 so a batch also stays inside its 256 KiB body limit.
pub const TELEMETRY_BATCH: usize = 200;
/// Events one connected run may hold on the app side.
pub const MAX_RUN_EVENTS: usize = 5_000;
const MAX_COLUMNS: usize = 2_000;
/// Engine events read for one projection; far above a normal run, and a bound on a broken one.
const MAX_ENGINE_EVENTS: usize = 50_000;

/// Text bounded to `max` bytes on a character boundary, control characters dropped.
fn bounded(value: &str, max: usize) -> String {
    let mut out = String::new();
    for c in value.chars().filter(|c| !c.is_control()) {
        if out.len() + c.len_utf8() > max {
            break;
        }
        out.push(c);
    }
    out
}

/// An id from the app's alphabet `[A-Za-z0-9_.:-]`, at most 64 characters.
fn token(value: &str) -> String {
    value
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | ':' | '-') {
                c
            } else {
                '_'
            }
        })
        .take(64)
        .collect()
}

fn finite(value: Option<f64>) -> Value {
    value
        .filter(|v| v.is_finite())
        .map_or(Value::Null, |v| json!(v))
}

fn wire(id: String, sequence: u64, at: &str, event: Value) -> Value {
    json!({"id": id, "sequence": sequence.max(1).to_string(), "at": at, "event": event})
}

/// What the CLI knew when it started a run and the engine does not repeat in its events: the
/// target, the resolved task, the objective, the table's name and profile. Written locally at
/// run start (`crate::cloud::remember_run`), read back when the run is synced.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RunContext {
    pub label: String,
    pub target: String,
    pub task: Option<jope_api::Task>,
    pub objective: Option<String>,
    #[serde(default)]
    pub profile: Option<jope_api::DatasetProfile>,
}

impl RunContext {
    pub fn from_spec(
        spec: &jope_api::RunSpec,
        file_name: &str,
        profile: Option<jope_api::DatasetProfile>,
    ) -> Self {
        Self {
            label: bounded(&format!("{file_name} · {}", spec.target), 160),
            target: spec.target.clone(),
            task: (spec.task != jope_api::Task::Auto).then_some(spec.task),
            objective: match &spec.objective {
                jope_api::ObjectiveSpec::Name(name) => Some(name.clone()),
                _ => None,
            },
            profile,
        }
    }
}

/// Whether a metric is minimized or maximized, for the names the engine reports.
pub fn direction(metric: &str) -> Option<&'static str> {
    match metric.to_ascii_lowercase().as_str() {
        "rmse" | "mse" | "mae" | "logloss" | "log_loss" | "brier" | "mape" | "rmsle" => {
            Some("minimize")
        }
        "auc" | "roc_auc" | "accuracy" | "balanced_accuracy" | "f1" | "r2" | "precision"
        | "recall" | "pr_auc" => Some("maximize"),
        _ => None,
    }
}

fn task(value: jope_api::Task) -> &'static str {
    match value {
        jope_api::Task::Regression => "regression",
        jope_api::Task::Binary => "binary_classification",
        jope_api::Task::Multiclass => "multiclass_classification",
        jope_api::Task::Auto => "unknown",
    }
}

/// The app-side events for one run, in engine order, from its durable event log and (when the run
/// names its dataset) the dataset's column profile.
pub fn collect<T: Transport>(
    local: &T,
    snapshot: &RunSnapshot,
    context: Option<&RunContext>,
) -> Result<Vec<Value>> {
    let mut engine = Vec::new();
    let mut after = jope_api::decimal::DecimalU64(0);
    loop {
        let page = local
            .run_events(&snapshot.run_id, after, 256)
            .context("read run events")?;
        if page.events.is_empty() || page.next_after.0 <= after.0 {
            break;
        }
        after = page.next_after;
        engine.extend(page.events);
        if engine.len() >= MAX_ENGINE_EVENTS {
            break;
        }
    }
    let profile = snapshot
        .evaluation_receipt
        .as_ref()
        .and_then(|r| r.identity.dataset_id.parse::<DatasetId>().ok())
        .and_then(|id| local.dataset_inspect(&id).ok())
        .or_else(|| context.and_then(|c| c.profile.clone()));
    Ok(project_with(snapshot, &engine, profile.as_ref(), context))
}

/// The pure half of [`collect`]: engine events (and an optional profile) to app events. Ids are
/// derived from the engine's own sequence, so a re-sync sends the same ids with the same bytes
/// and the app counts them as duplicates.
#[cfg(test)]
pub fn project(
    snapshot: &RunSnapshot,
    engine: &[jope_api::events::PublicEvent],
    profile: Option<&jope_api::DatasetProfile>,
) -> Vec<Value> {
    project_with(snapshot, engine, profile, None)
}

/// [`project`] with what the CLI remembered at run start. The engine's CLI runs emit no plan
/// event, so without it the account would get no `run_started` and a `score` metric: the
/// metric then comes from the champion or the trials, the target and task from the context.
pub fn project_with(
    snapshot: &RunSnapshot,
    engine: &[jope_api::events::PublicEvent],
    profile: Option<&jope_api::DatasetProfile>,
    context: Option<&RunContext>,
) -> Vec<Value> {
    let mut out = Vec::new();
    let first_at = engine
        .first()
        .map(|e| e.time.clone())
        .unwrap_or_else(|| "1970-01-01T00:00:00Z".into());
    if let Some(profile) = profile {
        let columns: Vec<Value> = profile
            .columns
            .iter()
            .take(MAX_COLUMNS)
            .map(|c| {
                json!({"name": bounded(&c.name, 256), "dataType": bounded(&c.data_type, 40),
                    "missing": c.missing.0.to_string(), "distinct": c.distinct.0.to_string()})
            })
            .filter(|c| c["name"] != "" && c["dataType"] != "")
            .collect();
        if !columns.is_empty() {
            out.push(wire(
                "profile-train".into(),
                1,
                &first_at,
                json!({"kind": "dataset_profiled", "role": "train",
                    "rows": profile.rows.0.to_string(), "columns": columns}),
            ));
        }
    }
    let mut hypotheses: BTreeMap<&str, &str> = BTreeMap::new();
    let has_plan = engine.iter().any(|e| matches!(e.event, EventBody::Plan(_)));
    let reported = snapshot
        .best
        .as_ref()
        .map(|b| b.metric.clone())
        .or_else(|| {
            engine.iter().find_map(|e| match &e.event {
                EventBody::ExperimentFinished(done) if !done.metric.is_empty() => {
                    Some(done.metric.clone())
                }
                EventBody::AlgorithmAccepted(a) if !a.metric.is_empty() => Some(a.metric.clone()),
                _ => None,
            })
        })
        .or_else(|| context.and_then(|c| c.objective.clone()))
        .map(|m| bounded(&m, 80))
        .filter(|m| !m.is_empty());
    let mut metric = reported.clone().unwrap_or_else(|| String::from("score"));
    if let (false, Some(context)) = (has_plan, context) {
        let direction = direction(&metric);
        out.push(wire(
            "context-start".into(),
            1,
            &first_at,
            json!({"kind": "run_started",
                "task": context.task.map_or("unknown", task),
                "target": bounded(&context.target, 256),
                "metric": metric,
                "direction": direction}),
        ));
    }
    for event in engine {
        let seq = event.sequence.0;
        match &event.event {
            EventBody::Plan(plan) => {
                metric = bounded(&plan.objective.name, 80);
                out.push(wire(
                    format!("plan-{seq}"),
                    seq,
                    &event.time,
                    json!({"kind": "run_started", "task": task(plan.task.value),
                    "target": context.map(|c| bounded(&c.target, 256)),
                    "metric": metric, "direction": match plan.objective.direction {
                        jope_api::Direction::Minimize => "minimize",
                        jope_api::Direction::Maximize => "maximize",
                    }}),
                ));
            }
            EventBody::ExperimentStarted(started) => {
                hypotheses.insert(&started.experiment_id, &started.hypothesis);
            }
            EventBody::ExperimentFinished(done) => {
                let name = hypotheses
                    .get(done.experiment_id.as_str())
                    .map(|h| bounded(h, 160))
                    .filter(|h| !h.is_empty())
                    .unwrap_or_else(|| bounded(&done.experiment_id, 160));
                let trial = token(&done.experiment_id);
                let trial_metric = match bounded(&done.metric, 80) {
                    m if m.is_empty() => metric.clone(),
                    m => m,
                };
                out.push(wire(
                    format!("trial-{seq}"),
                    seq,
                    &event.time,
                    json!({"kind": "trial_scored",
                        "trialId": if trial.is_empty() { format!("t{seq}") } else { trial },
                        "algorithm": if name.is_empty() { format!("candidate {seq}") } else { name },
                        "outcome": match done.outcome {
                            jope_api::events::ExperimentOutcome::Improved => "improved",
                            jope_api::events::ExperimentOutcome::NotImproved => "not_improved",
                            jope_api::events::ExperimentOutcome::Failed => "failed",
                        },
                        "metric": trial_metric,
                        "cv": finite(done.value), "holdout": null}),
                ));
            }
            EventBody::RunFinished(done) => {
                let outcome = match snapshot.state {
                    jope_api::RunState::Cancelled => "cancelled",
                    jope_api::RunState::Failed => "failed",
                    _ => "completed",
                };
                let elapsed = done
                    .budget_used
                    .wall_seconds
                    .parse::<f64>()
                    .ok()
                    .filter(|v| v.is_finite() && *v >= 0.0)
                    .unwrap_or(0.0);
                out.push(wire(
                    format!("finish-{seq}"),
                    seq,
                    &event.time,
                    json!({"kind": "run_finished", "outcome": outcome, "metric": metric,
                        "cv": finite(done.scores.cv), "holdout": finite(done.scores.holdout),
                        "test": finite(done.scores.test),
                        "evaluations": done.budget_used.evaluations.0.to_string(),
                        "elapsedSeconds": format!("{elapsed}")}),
                ));
            }
            _ => {}
        }
    }
    out.truncate(MAX_RUN_EVENTS);
    out
}

/// The event that records an explicit model publication.
pub fn model_published(sha256: &str, at: &str, sequence: u64) -> Value {
    wire(
        format!("model-{}", &sha256[..sha256.len().min(16)]),
        sequence,
        at,
        json!({"kind": "model_published", "sha256": sha256}),
    )
}

mod telemetry;
#[cfg(test)]
pub use telemetry::{bucket, tenth};
pub use telemetry::{environment, telemetry, telemetry_model, Pseudonyms};

#[cfg(test)]
mod tests;
