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
pub use telemetry::{bucket, environment, telemetry, telemetry_model, tenth, Pseudonyms};

#[cfg(test)]
mod tests {
    use super::*;

    fn snapshot() -> RunSnapshot {
        serde_json::from_value(json!({
            "run_id": "run_01ARZ3NDEKTSV4RRFFQ69G5FAV", "revision": "3", "state": "completed",
            "phase": "finalizing", "pending": null, "effort": "fast",
            "budget_used": {"evaluations": "2", "wall_seconds": "12.5"},
            "budget_total": {"evaluations": "10", "wall_seconds": "60"},
            "best": null, "event_watermark": "9", "warnings": [], "completion_reason": null
        }))
        .expect("snapshot fixture")
    }

    fn events() -> Vec<jope_api::events::PublicEvent> {
        let run = "run_01ARZ3NDEKTSV4RRFFQ69G5FAV";
        let t = "2026-11-02T14:00:00Z";
        serde_json::from_value(json!([
            {"run_id": run, "sequence": "3", "time": t, "kind": "experiment.started",
                "body": {"experiment_id": "exp 1", "hypothesis": "ridge on CANARY_COL_7f3a"}},
            {"run_id": run, "sequence": "4", "time": t, "kind": "experiment.finished",
                "body": {"experiment_id": "exp 1", "outcome": "improved", "metric": "rmse",
                    "value": 0.61, "failure": null}},
            {"run_id": run, "sequence": "5", "time": t, "kind": "experiment.finished",
                "body": {"experiment_id": "exp2", "outcome": "failed", "metric": "rmse",
                    "value": null, "failure": null}},
            {"run_id": run, "sequence": "9", "time": t, "kind": "run.finished",
                "body": {"completion_reason": "converged", "champion": null,
                    "scores": {"cv": 0.61, "holdout": 0.65, "test": null},
                    "budget_used": {"evaluations": "2", "wall_seconds": "12.5"},
                    "nondeterminism": {"count": "0", "experiment_ids": []}}}
        ]))
        .expect("event fixture")
    }

    fn profile() -> jope_api::DatasetProfile {
        serde_json::from_value(json!({"dataset_id": "ds_0123456789abcdef01234567",
            "name": "CANARY_NAME", "bytes": "100", "rows": "7342917",
            "columns": [{"name": "CANARY_COL_7f3a", "data_type": "numeric", "missing": "3", "distinct": "7"},
                        {"name": "y", "data_type": "categorical", "missing": "0", "distinct": "2"}],
            "target_suggestions": [], "warnings": []}))
        .expect("profile fixture")
    }

    #[test]
    fn app_events_carry_trials_finish_and_column_summaries_only() {
        let app = project(&snapshot(), &events(), Some(&profile()));
        let kinds: Vec<_> = app.iter().map(|e| e["event"]["kind"].clone()).collect();
        assert_eq!(
            kinds,
            [
                "dataset_profiled",
                "trial_scored",
                "trial_scored",
                "run_finished"
            ]
        );
        assert_eq!(app[1]["event"]["algorithm"], "ridge on CANARY_COL_7f3a");
        assert_eq!(app[1]["event"]["trialId"], "exp_1");
        assert_eq!(app[1]["event"]["cv"], 0.61);
        assert_eq!(app[2]["event"]["outcome"], "failed");
        assert_eq!(app[3]["event"]["holdout"], 0.65);
        assert_eq!(app[3]["event"]["elapsedSeconds"], "12.5");
        // The profile is column summaries: these four keys and nothing else, and not the name.
        for column in app[0]["event"]["columns"].as_array().unwrap() {
            let mut keys: Vec<_> = column.as_object().unwrap().keys().cloned().collect();
            keys.sort();
            assert_eq!(keys, ["dataType", "distinct", "missing", "name"]);
        }
        assert!(!app[0].to_string().contains("CANARY_NAME"));
        // Re-projection is byte-identical, so a re-sync is a duplicate, not a conflict.
        assert_eq!(app, project(&snapshot(), &events(), Some(&profile())));
    }

    #[test]
    fn telemetry_has_no_names_no_exact_counts_and_no_local_ids() {
        let app = project(&snapshot(), &events(), Some(&profile()));
        let ids = Pseudonyms {
            install_id: "0123456789abcdef0123456789abcdef",
            session_id: "1123456789abcdef0123456789abcdef",
            run_id: "2123456789abcdef0123456789abcdef",
            environment: "development",
        };
        let mut n = 0;
        let sent = telemetry(&app, &ids, 1_790_000_000_000, || {
            n += 1;
            format!("{n:032x}")
        });
        assert_eq!(sent.len(), 4);
        let text = serde_json::to_string(&sent.iter().map(|(_, v)| v).collect::<Vec<_>>()).unwrap();
        for canary in [
            "CANARY",
            "7342917",
            "run_01ARZ",
            "exp 1",
            "exp_1",
            "ridge",
            "rmse",
            "0.61",
        ] {
            assert!(!text.contains(canary), "{canary} leaked: {text}");
        }
        assert_eq!(sent[0].1["event"]["dataset"]["rows"], "B1mTo10m");
        assert_eq!(sent[1].1["event"]["score"], "T6");
        assert_eq!(sent[2].1["event"]["outcome"], "failed");
        assert_eq!(sent[3].1["event"]["outcome"], "succeeded");
        assert_eq!(sent[3].1["event"]["algorithm"]["trials"], 2);
        assert_eq!(sent[0].1["source"], "cli");
        assert_eq!(sent[0].0, "profile-train");
    }

    #[test]
    fn buckets_and_tenths_match_veox_telemetry() {
        assert_eq!(bucket(0), "Zero");
        assert_eq!(bucket(999), "B100To1k");
        assert_eq!(bucket(1_000), "B1kTo10k");
        assert_eq!(bucket(u64::MAX), "B1bOrMore");
        assert_eq!(tenth(f64::NAN), "T0");
        assert_eq!(tenth(0.61), "T6");
        assert_eq!(tenth(4.0), "T10");
        assert_eq!(environment("https://app.veox.ai"), "production");
        assert_eq!(environment("http://127.0.0.1:8091"), "development");
    }
}
