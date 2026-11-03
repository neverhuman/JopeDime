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
pub fn collect<T: Transport>(local: &T, snapshot: &RunSnapshot) -> Result<Vec<Value>> {
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
        .and_then(|id| local.dataset_inspect(&id).ok());
    Ok(project(snapshot, &engine, profile.as_ref()))
}

/// The pure half of [`collect`]: engine events (and an optional profile) to app events. Ids are
/// derived from the engine's own sequence, so a re-sync sends the same ids with the same bytes
/// and the app counts them as duplicates.
pub fn project(
    snapshot: &RunSnapshot,
    engine: &[jope_api::events::PublicEvent],
    profile: Option<&jope_api::DatasetProfile>,
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
    let mut metric = String::from("score");
    for event in engine {
        let seq = event.sequence.0;
        match &event.event {
            EventBody::Plan(plan) => {
                metric = bounded(&plan.objective.name, 80);
                out.push(wire(
                    format!("plan-{seq}"),
                    seq,
                    &event.time,
                    json!({"kind": "run_started", "task": task(plan.task.value), "target": null,
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
