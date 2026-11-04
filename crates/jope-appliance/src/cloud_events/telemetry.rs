use serde_json::{json, Value};

// ---- de-identified projection (veox-telemetry envelope, schema_version 1) ----

/// A count rounded down to its decade, spelled as veox-telemetry's `Bucket`.
pub fn bucket(n: u64) -> &'static str {
    match n {
        0 => "Zero",
        1..=9 => "B1To10",
        10..=99 => "B10To100",
        100..=999 => "B100To1k",
        1_000..=9_999 => "B1kTo10k",
        10_000..=99_999 => "B10kTo100k",
        100_000..=999_999 => "B100kTo1m",
        1_000_000..=9_999_999 => "B1mTo10m",
        10_000_000..=99_999_999 => "B10mTo100m",
        100_000_000..=999_999_999 => "B100mTo1b",
        _ => "B1bOrMore",
    }
}

/// A fraction in tenths, spelled as veox-telemetry's `Tenth`; NaN and out-of-range clamp.
pub fn tenth(value: f64) -> String {
    let f = if value.is_nan() {
        0.0
    } else {
        value.clamp(0.0, 1.0)
    };
    format!("T{}", (f * 10.0).floor() as u8)
}

fn shape(profiled: &Value, classes: u64) -> Value {
    let columns = profiled["columns"].as_array().cloned().unwrap_or_default();
    let rows = profiled["rows"]
        .as_str()
        .and_then(|r| r.parse::<u64>().ok())
        .unwrap_or(0);
    let kind = |c: &Value| {
        let t = c["dataType"].as_str().unwrap_or("").to_ascii_lowercase();
        if matches!(
            t.as_str(),
            "numeric" | "number" | "float" | "int" | "integer"
        ) {
            0
        } else if t.contains("date") || t.contains("time") {
            3
        } else if t == "text" {
            2
        } else {
            1
        }
    };
    let count = |k| columns.iter().filter(|c| kind(c) == k).count() as u64;
    let missing: u128 = columns
        .iter()
        .filter_map(|c| c["missing"].as_str()?.parse::<u64>().ok())
        .map(u128::from)
        .sum();
    let cells = u128::from(rows) * columns.len() as u128;
    let missing = if cells == 0 {
        0.0
    } else {
        missing as f64 / cells as f64
    };
    // `schema` is null: a keyed hash needs a tenant key this CLI does not hold.
    json!({"rows": bucket(rows), "columns": bucket(columns.len() as u64),
        "numeric_columns": bucket(count(0)), "categorical_columns": bucket(count(1)),
        "text_columns": bucket(count(2)), "datetime_columns": bucket(count(3)),
        "missing": tenth(missing), "target_classes": bucket(classes), "schema": null})
}

/// The fixed, non-identifying identity of one telemetry batch.
pub struct Pseudonyms<'a> {
    pub install_id: &'a str,
    pub session_id: &'a str,
    /// Random per local run, minted for telemetry; never the local run id or the cloud id.
    pub run_id: &'a str,
    pub environment: &'a str,
}

/// One veox-telemetry envelope per app event that has a de-identified form, keyed by the app
/// event's id so a re-sync does not send it twice. `new_id` mints each envelope's random id.
pub fn telemetry(
    app: &[Value],
    ids: &Pseudonyms<'_>,
    ts_ms: u64,
    mut new_id: impl FnMut() -> String,
) -> Vec<(String, Value)> {
    let profiled = app
        .iter()
        .find(|e| e["event"]["kind"] == "dataset_profiled")
        .map(|e| shape(&e["event"], 0));
    let task = app
        .iter()
        .find(|e| e["event"]["kind"] == "run_started")
        .and_then(|e| e["event"]["task"].as_str())
        .unwrap_or("unknown")
        .to_owned();
    let trials = app
        .iter()
        .filter(|e| e["event"]["kind"] == "trial_scored")
        .count() as u32;
    let empty = json!({"rows":"Zero","columns":"Zero","numeric_columns":"Zero",
        "categorical_columns":"Zero","text_columns":"Zero","datetime_columns":"Zero",
        "missing":"T0","target_classes":"Zero","schema":null});
    let dataset = profiled.clone().unwrap_or(empty);
    let mut out = Vec::new();
    for event in app {
        let body = &event["event"];
        let id = event["id"].as_str().unwrap_or_default().to_owned();
        let score = |key: &str| body[key].as_f64().map(tenth);
        let projected = match body["kind"].as_str() {
            Some("dataset_profiled") => {
                Some(json!({"event": "dataset_profiled", "dataset": dataset}))
            }
            Some("run_started") => Some(json!({"event": "run_started", "dataset": dataset,
                "task": task, "budget_secs": 0})),
            Some("trial_scored") => Some(json!({"event": "trial_scored", "run_id": ids.run_id,
                "family": "other", "outcome": body["outcome"],
                "score": score("cv").or_else(|| score("holdout"))})),
            Some("run_finished") => {
                let wall = body["elapsedSeconds"]
                    .as_str()
                    .and_then(|s| s.parse::<f64>().ok())
                    .filter(|v| v.is_finite() && *v >= 0.0)
                    .map_or(0, |v| v.min(f64::from(u32::MAX)) as u32);
                let outcome = match body["outcome"].as_str() {
                    Some("completed") => json!("succeeded"),
                    Some("cancelled") => json!("cancelled"),
                    _ => json!({"failed": "unknown"}),
                };
                Some(
                    json!({"event": "run_finished", "run_id": ids.run_id, "outcome": outcome,
                    "metrics": {"wall_secs": wall, "queue_secs": 0, "compute_secs": wall,
                        "peak_memory": "Zero", "accelerator": "unknown",
                        "score": score("cv").or_else(|| score("holdout")), "beat_baseline": null},
                    "algorithm": {"family": "other", "trials": trials, "ensemble_size": 1}}),
                )
            }
            _ => None,
        };
        if let Some(projected) = projected {
            out.push((
                id,
                json!({"schema_version": 1, "install_id": ids.install_id,
                    "session_id": ids.session_id, "event_id": new_id(), "ts_ms": ts_ms,
                    "consent_tier": "basic", "source": "cli", "environment": ids.environment,
                    "event": projected}),
            ));
        }
    }
    out
}

/// The de-identified form of an explicit model publication.
pub fn telemetry_model(ids: &Pseudonyms<'_>, bytes: u64, ts_ms: u64, event_id: String) -> Value {
    json!({"schema_version": 1, "install_id": ids.install_id, "session_id": ids.session_id,
        "event_id": event_id, "ts_ms": ts_ms, "consent_tier": "basic", "source": "cli",
        "environment": ids.environment,
        "event": {"event": "model_published", "run_id": ids.run_id, "size": bucket(bytes)}})
}

/// Which veox-telemetry environment a workspace origin reports into. Only the production app
/// is `production`; a missing field would mean production, so it is always set.
pub fn environment(origin: &str) -> &'static str {
    match origin {
        "https://app.veox.ai" | "https://cloud.veox.ai" => "production",
        o if o.contains("stage") => "staging",
        _ => "development",
    }
}
