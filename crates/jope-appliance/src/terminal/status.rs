//! Compact status and activity lines for the inline terminal.

use jope_api::{
    Effort, MetricSample, MetricValue, RunPhase, RunSnapshot, RunState, Severity, WarningState,
};

/// Renders the status and activity rows for the terminal's live region.
///
/// Optional fields are removed in the spec's order as the available width shrinks:
/// GPU, RAM, CPU, then experiment rate.
pub fn render(
    snapshot: &RunSnapshot,
    telemetry: &[MetricSample],
    width: usize,
) -> (String, String) {
    let optional = [
        metric_field(telemetry, "host.gpu.util", "GPU", |value, _sample| {
            format!("{value:.0}%")
        }),
        metric_field(telemetry, "host.cpu.process_rss", "RAM", format_bytes),
        cpu_field(telemetry),
        rate_field(snapshot),
    ];

    let warning = snapshot
        .warnings
        .iter()
        .find(|warning| {
            warning.state == WarningState::Raised
                && matches!(warning.severity, Severity::Warn | Severity::Error)
        })
        .map(|warning| format!("[warning: {}]", warning.code));

    let mut status = format!(
        "{} · {} · {} · {} · {} exp",
        state(snapshot.state),
        effort(snapshot.effort),
        elapsed_cap(snapshot),
        best(snapshot),
        snapshot.budget_used.evaluations.0
    );
    if let Some(warning) = &warning {
        status.push_str(" · ");
        status.push_str(warning);
    }

    for field in optional.into_iter().flatten() {
        if status.chars().count() + 3 + field.chars().count() <= width {
            status.push_str(" · ");
            status.push_str(&field);
        }
    }

    // The compact core still carries every required run field at the card's minimum width.
    if status.chars().count() > width {
        status = compact_core(snapshot, warning.as_deref());
    }
    let status = truncate(&status, width);
    let activity = match snapshot.phase {
        RunPhase::Preparing => "phase: preparing",
        RunPhase::Encoding => "phase: encoding",
        RunPhase::Seeding => "phase: seeding",
        RunPhase::Searching => "phase: searching",
        RunPhase::Finalizing => "phase: finalizing",
    };
    (status, truncate(activity, width))
}

fn metric_field(
    samples: &[MetricSample],
    name: &str,
    label: &str,
    format: impl Fn(f64, &MetricSample) -> String,
) -> Option<String> {
    let Some(sample) = samples.iter().find(|sample| sample.name == name) else {
        return Some(format!("{label} n/a"));
    };
    Some(numeric_value(sample).map_or_else(
        || format!("{label} n/a"),
        |value| format!("{label} {}", format(value, sample)),
    ))
}

fn cpu_field(samples: &[MetricSample]) -> Option<String> {
    let cores = metric_number(samples, "cpu.cores");
    let host_pct = metric_number(samples, "cpu.host_pct");
    if cores.is_none() && host_pct.is_none() {
        return Some("CPU n/a".to_owned());
    }
    let cores = cores.map_or_else(|| "n/a".to_owned(), |value| format!("{value:.1}"));
    let host_pct = host_pct.map_or_else(|| "n/a".to_owned(), |value| format!("{value:.0}%"));
    Some(format!("CPU {cores} cores ({host_pct})"))
}

fn rate_field(snapshot: &RunSnapshot) -> Option<String> {
    let elapsed = snapshot.budget_used.wall_seconds.parse::<f64>().ok()?;
    if !elapsed.is_finite() || elapsed <= 0.0 {
        return Some("n/a exp/s".to_owned());
    }
    Some(format!(
        "{:.1} exp/s",
        snapshot.budget_used.evaluations.0 as f64 / elapsed
    ))
}

fn metric_number(samples: &[MetricSample], name: &str) -> Option<f64> {
    samples
        .iter()
        .find(|sample| sample.name == name)
        .and_then(numeric_value)
}

fn numeric_value(sample: &MetricSample) -> Option<f64> {
    let value = match sample.value.as_ref()? {
        MetricValue::Number(value) => *value,
        MetricValue::Decimal(value) => value.0 as f64,
        MetricValue::Boolean(_) => return None,
    };
    value.is_finite().then_some(value)
}

fn format_bytes(value: f64, _sample: &MetricSample) -> String {
    format!("{:.1} GB", value / 1_000_000_000.0)
}

fn state(state: RunState) -> &'static str {
    match state {
        RunState::Running => "running",
        RunState::Pausing => "pausing",
        RunState::Paused => "paused",
        RunState::Completed => "completed",
        RunState::Cancelled => "cancelled",
        RunState::Failed => "failed",
    }
}

fn effort(effort: Effort) -> &'static str {
    match effort {
        Effort::Fast => "fast",
        Effort::Medium => "medium",
        Effort::High => "high",
        Effort::Ultra => "ultra",
    }
}

fn elapsed_cap(snapshot: &RunSnapshot) -> String {
    let elapsed = snapshot
        .budget_used
        .wall_seconds
        .parse::<f64>()
        .ok()
        .filter(|value| value.is_finite() && *value >= 0.0)
        .unwrap_or(0.0) as u64;
    let cap = snapshot.budget_total.wall_seconds.0;
    format!("{}/{}", clock(elapsed), clock(cap))
}

fn clock(seconds: u64) -> String {
    let minutes = seconds / 60;
    let seconds = seconds % 60;
    if minutes >= 60 {
        format!("{:02}:{:02}:{:02}", minutes / 60, minutes % 60, seconds)
    } else {
        format!("{minutes:02}:{seconds:02}")
    }
}

fn best(snapshot: &RunSnapshot) -> String {
    snapshot.best.as_ref().map_or_else(
        || "best n/a".to_owned(),
        |best| {
            if best.cv_value.is_finite() {
                format!("best {:.3}", best.cv_value)
            } else {
                "best n/a".to_owned()
            }
        },
    )
}

fn compact_core(snapshot: &RunSnapshot, warning: Option<&str>) -> String {
    let state = match snapshot.state {
        RunState::Running => "run",
        RunState::Pausing => "pause",
        RunState::Paused => "paused",
        RunState::Completed => "done",
        RunState::Cancelled => "cancel",
        RunState::Failed => "failed",
    };
    let effort = match snapshot.effort {
        Effort::Fast => "fast",
        Effort::Medium => "med",
        Effort::High => "high",
        Effort::Ultra => "ultra",
    };
    let elapsed = snapshot
        .budget_used
        .wall_seconds
        .parse::<f64>()
        .ok()
        .filter(|value| value.is_finite() && *value >= 0.0)
        .unwrap_or(0.0) as u64;
    let best = snapshot
        .best
        .as_ref()
        .filter(|best| best.cv_value.is_finite())
        .map_or_else(|| "n/a".to_owned(), |best| format!("{:.3}", best.cv_value));
    let mut core = format!(
        "{state} {effort} {}/{} b{best} {}exp",
        compact_clock(elapsed),
        compact_clock(snapshot.budget_total.wall_seconds.0),
        snapshot.budget_used.evaluations.0
    );
    if let Some(warning) = warning {
        core.push(' ');
        core.push_str(warning);
    }
    core
}

fn compact_clock(seconds: u64) -> String {
    let minutes = seconds / 60;
    format!("{}:{:02}", minutes, seconds % 60)
}

fn truncate(value: &str, width: usize) -> String {
    if value.chars().count() <= width {
        return value.to_owned();
    }
    value.chars().take(width).collect()
}

#[cfg(test)]
mod tests {
    use jope_api::{
        decimal::DecimalU64, ids::RunId, BudgetTotal, BudgetUsed, Effort, MetricSample,
        MetricValue, RunPhase, RunSnapshot, RunState,
    };

    use super::render;

    fn snapshot() -> RunSnapshot {
        RunSnapshot {
            run_id: RunId::new(),
            revision: DecimalU64(1),
            state: RunState::Running,
            phase: RunPhase::Searching,
            pending: None,
            effort: Effort::Medium,
            budget_used: BudgetUsed {
                evaluations: DecimalU64(41),
                wall_seconds: "23.4".to_owned(),
            },
            budget_total: BudgetTotal {
                evaluations: DecimalU64(128),
                wall_seconds: DecimalU64(60),
            },
            best: None,
            event_watermark: DecimalU64(0),
            warnings: Vec::new(),
            completion_reason: None,
            evaluation_receipt: None,
        }
    }

    fn sample(name: &str, value: Option<MetricValue>) -> MetricSample {
        MetricSample {
            name: name.to_owned(),
            value,
            unit: String::new(),
            scope: "host".to_owned(),
            window_ms: 1_000,
            age_ms: 0,
            availability: "available".to_owned(),
        }
    }

    fn full_telemetry() -> Vec<MetricSample> {
        vec![
            sample("host.gpu.util", Some(MetricValue::Number(72.0))),
            sample(
                "host.cpu.process_rss",
                Some(MetricValue::Decimal(DecimalU64(900_000_000))),
            ),
            sample("cpu.cores", Some(MetricValue::Number(6.8))),
            sample("cpu.host_pct", Some(MetricValue::Number(85.0))),
        ]
    }

    #[test]
    fn golden_status_and_activity_lines_fit_40_80_and_120_columns() {
        let snapshot = snapshot();
        let telemetry = full_telemetry();
        let (at_40, activity_40) = render(&snapshot, &telemetry, 40);
        let (at_80, activity_80) = render(&snapshot, &telemetry, 80);
        let (at_120, activity_120) = render(&snapshot, &telemetry, 120);

        assert_eq!(at_40, "run med 0:23/1:00 bn/a 41exp");
        assert_eq!(
            at_80,
            "running · medium · 00:23/01:00 · best n/a · 41 exp · GPU 72% · RAM 0.9 GB"
        );
        assert_eq!(at_120, "running · medium · 00:23/01:00 · best n/a · 41 exp · GPU 72% · RAM 0.9 GB · CPU 6.8 cores (85%) · 1.8 exp/s");
        assert_eq!(activity_40, "phase: searching");
        assert_eq!(activity_80, "phase: searching");
        assert_eq!(activity_120, "phase: searching");
    }

    #[test]
    fn null_sensors_render_as_unavailable_and_fields_drop_in_priority_order() {
        let telemetry = vec![
            sample("host.gpu.util", None),
            sample("host.cpu.process_rss", None),
            sample("cpu.cores", None),
            sample("cpu.host_pct", None),
        ];
        let (line, _) = render(&snapshot(), &telemetry, 120);
        assert!(line.contains("GPU n/a"));
        assert!(line.contains("RAM n/a"));
        assert!(line.contains("CPU n/a"));

        let (short, _) = render(&snapshot(), &full_telemetry(), 80);
        assert!(short.contains("RAM"));
        assert!(!short.contains("CPU"));
        assert!(!short.contains("exp/s"));
        assert!(short.contains("GPU"));
    }
}
