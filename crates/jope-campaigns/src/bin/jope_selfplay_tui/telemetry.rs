use super::*;

/// Seeding curve (`ops/seeding-curve.v1.json`, written by the ingestion
/// reconcile loop): per model-checkpoint generation, the DL strict-win share
/// and DL observation share for the tab-4 generations panel DL-share column.
pub(super) fn read_seeding_curve(root: &Path) -> Option<SeedingCurveView> {
    let bytes = fs::read(root.join("ops/seeding-curve.v1.json")).ok()?;
    let value = serde_json::from_slice::<Value>(&bytes).ok()?;
    let points = value
        .get("curve")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .map(|point| SeedingPoint {
            checkpoint_short: string_at(point, &["model_checkpoint_sha256"])
                .unwrap_or("?")
                .chars()
                .take(8)
                .collect(),
            dl_win_share_pct: f64_at(point, &["dl_strict_win_share_pct"]),
            dl_obs_share_pct: f64_at(point, &["observations", "dl_observation_share"])
                .map(|share| share * 100.0),
            strict_wins_total: u64_at(point, &["strict_wins_total"]).unwrap_or(0),
            observations_caught_up: bool_at(point, &["observations", "caught_up"]).unwrap_or(false),
        })
        .collect();
    Some(SeedingCurveView {
        points,
        fetched_at_ms: now_ms(),
    })
}

pub(super) fn read_watchdog(root: &Path) -> Option<TrainerWatchdogView> {
    let bytes = fs::read(root.join("ops/trainer-watchdog-v1.json")).ok()?;
    let value = serde_json::from_slice::<Value>(&bytes).ok()?;
    Some(TrainerWatchdogView {
        labels: u64_at(&value, &["labels"]).unwrap_or(0),
        snapshot_watermark: u64_at(&value, &["snapshot_watermark"]).unwrap_or(0),
        last_evidence_watermark: u64_at(&value, &["last_evidence_watermark"]).unwrap_or(0),
        next_label_threshold: u64_at(&value, &["next_label_threshold"]).unwrap_or(0),
        pending_unfrozen_labels: u64_at(&value, &["pending_unfrozen_labels"]).unwrap_or(0),
        checkpoint_sha256: string_at(&value, &["checkpoint_sha256"])
            .unwrap_or("-")
            .to_owned(),
        action: string_at(&value, &["action"]).unwrap_or("-").to_owned(),
        active_trainers: value
            .get("active_trainers")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .map(str::to_owned)
            .collect(),
        generated_at_ms: u64_at(&value, &["generated_at_ms"]).unwrap_or(0) as u128,
    })
}

pub(super) fn parse_accepted_task_fronts(value: Option<&Value>) -> Vec<AcceptedTaskFront> {
    let Some(tasks) = value
        .and_then(|value| value.get("tasks"))
        .and_then(Value::as_object)
    else {
        return Vec::new();
    };
    ["binary", "regression"]
        .into_iter()
        .filter_map(|task| {
            let row = tasks.get(task)?;
            Some(AcceptedTaskFront {
                task: task.to_owned(),
                checkpoint_sha256: string_at(row, &["checkpoint_sha256"])
                    .unwrap_or("-")
                    .to_owned(),
                generation: u64_at(row, &["generation"]).unwrap_or(0),
                updated_at_ms: u64_at(row, &["updated_at_ms"]).unwrap_or(0) as u128,
            })
        })
        .collect()
}

/// Alarm files rendered in the alerts panel, in display order.
///
/// `ops/utilization-alarms.v1.json` is written by jope_utilization_supervisor
/// and is deliberately a SEPARATE file from `health-alarms.v1.json`:
/// jope_selfplay_expansion_driver.rs:116-141 halts dataset activation on any
/// critical row in health-alarms, so putting a critical `cluster_idle` row
/// there would stop the flywheel activating datasets *because the cluster is
/// idle* — the exact starvation loop that emptied all 1,052 slots on
/// 2026-07-28. The TUI is a display, so it can safely read both.
pub(super) const ALARM_SOURCES: &[(&str, &str)] = &[
    ("ops/utilization-alarms.v1.json", "utilization"),
    ("ops/health-alarms.v1.json", "alarms"),
];

pub(super) fn read_health_alarms(root: &Path, states: &mut Vec<SourceState>) -> Vec<String> {
    let mut rows = Vec::new();
    for (source, label) in ALARM_SOURCES {
        rows.extend(read_alarm_file(&root.join(source), label, states));
    }
    rows
}

pub(super) fn read_alarm_file(
    path: &Path,
    label: &str,
    states: &mut Vec<SourceState>,
) -> Vec<String> {
    let Ok(bytes) = fs::read(path) else {
        return Vec::new();
    };
    match serde_json::from_slice::<Value>(&bytes) {
        Ok(value) => value
            .get("alarms")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|alarm| {
                let name = string_at(alarm, &["alarm"])?;
                let severity = string_at(alarm, &["severity"]).unwrap_or("warn");
                let observed = alarm
                    .get("observed")
                    .map(Value::to_string)
                    .unwrap_or_else(|| "-".to_owned());
                let threshold = alarm
                    .get("threshold")
                    .map(Value::to_string)
                    .unwrap_or_else(|| "-".to_owned());
                Some(format!(
                    "ALARM[{severity}] {name}: {observed} (thr {threshold})"
                ))
            })
            .collect(),
        Err(error) => {
            states.push(SourceState {
                name: label.to_owned(),
                healthy: false,
                note: format!("invalid JSON: {error}"),
            });
            Vec::new()
        }
    }
}

pub(super) fn push_ring(ring: &mut VecDeque<GameplayEvent>, event: GameplayEvent) {
    if ring.len() >= GAMEPLAY_RING {
        ring.pop_front();
    }
    ring.push_back(event);
}
