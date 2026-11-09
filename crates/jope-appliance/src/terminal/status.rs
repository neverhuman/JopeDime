//! Compact status and activity lines for the inline terminal.

use jope_api::{
    CompletionReason, Effort, MetricSample, MetricValue, RunPhase, RunSnapshot, RunState, Severity,
    WarningState,
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

    // Live machine fields describe a run in progress; a finished run shows none (#2143).
    let finished = is_terminal(snapshot.state);
    for field in optional.into_iter().flatten().filter(|_| !finished) {
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
    (status, truncate(&activity(snapshot), width))
}

/// The activity row. A finished run reports how it ended, never the phase it ended in: the
/// engine leaves `phase: finalizing` on a completed run (#2135), which read as still running.
fn activity(snapshot: &RunSnapshot) -> String {
    let phase = match snapshot.phase {
        RunPhase::Preparing => "preparing",
        RunPhase::Encoding => "encoding",
        RunPhase::Seeding => "seeding",
        RunPhase::Searching => "searching",
        RunPhase::Finalizing => "finalizing",
    };
    match snapshot.state {
        RunState::Completed => match snapshot.completion_reason {
            Some(reason) => format!("finished · {}", completion_reason_name(reason)),
            None => "finished".to_owned(),
        },
        RunState::Cancelled => "finished · cancelled".to_owned(),
        RunState::Failed => "finished · failed".to_owned(),
        _ => format!("phase: {phase}"),
    }
}

/// Whether a run has ended.
pub fn is_terminal(state: RunState) -> bool {
    matches!(
        state,
        RunState::Completed | RunState::Cancelled | RunState::Failed
    )
}

pub fn completion_reason_name(reason: CompletionReason) -> &'static str {
    match reason {
        CompletionReason::BudgetEvaluations => "evaluation budget reached",
        CompletionReason::BudgetTime => "time budget reached",
        CompletionReason::Converged => "converged",
        CompletionReason::EffortReduced => "effort reduced",
    }
}

pub fn state_name(state: RunState) -> &'static str {
    self::state(state)
}

pub fn effort_name(value: Effort) -> &'static str {
    effort(value)
}

/// The snapshot with its elapsed time advanced to `since` (the time the client has followed
/// it) while it runs, when the engine reports less: the engine records a run's elapsed time as
/// it ends, so the status clock otherwise stays at 00:00 for the whole run.
pub fn with_live_clock(snapshot: &RunSnapshot, since: Option<std::time::Duration>) -> RunSnapshot {
    let mut live = snapshot.clone();
    let Some(since) = since.filter(|_| !is_terminal(snapshot.state)) else {
        return live;
    };
    let reported = snapshot
        .budget_used
        .wall_seconds
        .parse::<f64>()
        .ok()
        .filter(|value| value.is_finite())
        .unwrap_or(0.0);
    if since.as_secs_f64() > reported {
        live.budget_used.wall_seconds = format!("{:.1}", since.as_secs_f64());
    }
    live
}

/// One line for a finished run: how it ended and its champion.
pub fn finished_line(snapshot: &RunSnapshot) -> String {
    let ended = match (snapshot.state, snapshot.completion_reason) {
        (RunState::Completed, Some(reason)) => {
            format!("Run finished · {}", completion_reason_name(reason))
        }
        (RunState::Completed, None) => "Run finished".to_owned(),
        (state, _) => format!("Run {}", self::state(state)),
    };
    match &snapshot.best {
        Some(best) if best.cv_value.is_finite() => format!(
            "{ended} · champion {} (CV {} {:.3}) · {} candidates evaluated",
            best.algorithm_id,
            super::sanitize_user_string(&best.metric),
            best.cv_value,
            snapshot.budget_used.evaluations.0
        ),
        _ => format!(
            "{ended} · no champion · {} candidates evaluated",
            snapshot.budget_used.evaluations.0
        ),
    }
}

/// #2134 (client side): when the search stopped on its time budget, say so plainly, with how
/// many of the planned candidates it evaluated and, when it ran past the budget, by how much.
/// Budget enforcement itself is the engine's; this only reports what the snapshot records.
pub fn budget_stop_line(snapshot: &RunSnapshot) -> Option<String> {
    if snapshot.completion_reason != Some(CompletionReason::BudgetTime) {
        return None;
    }
    let done = snapshot.budget_used.evaluations.0;
    let planned = snapshot.budget_total.evaluations.0;
    let cap = snapshot.budget_total.wall_seconds.0;
    let mut line = if planned > done {
        format!("Stopped at the time budget ({cap} s) after {done} of {planned} candidates.")
    } else {
        format!("Stopped at the time budget ({cap} s) after {done} candidates.")
    };
    let elapsed = snapshot
        .budget_used
        .wall_seconds
        .parse::<f64>()
        .ok()
        .filter(|value| value.is_finite());
    if let Some(elapsed) = elapsed.filter(|elapsed| *elapsed >= cap as f64 + 1.0) {
        line.push_str(&format!(
            " Finishing the candidates in flight took it to {elapsed:.0} s."
        ));
    }
    if planned > done {
        line.push_str(" A longer budget (/effort --time, or --budget) evaluates more.");
    }
    Some(line)
}

/// The one-line next step after a result (#2143). `tui` selects slash commands over `jain`
/// subcommands. Runs on your own data are the veox Cloud's (owner rule 2026-11-06: the local,
/// weightless engine is for the bundled demos and anonymous use), so the own-data step is the
/// login while signed out and a cloud run once signed in.
pub fn next_steps(run: &str, connected: bool, tui: bool) -> String {
    let predict = format!(
        "predict with `jain predict --model {run} --input new_rows.csv --output predictions.csv`"
    );
    let code = if tui {
        "get the code with /export --out DIR · /copy result copies this result".to_owned()
    } else {
        format!("get the code with `jain export {run} --kind bundle --lang python --out DIR`")
    };
    let own = match (tui, connected) {
        (true, false) => "your own data: /login (accounts open from the waitlist: https://app.veox.ai/register); it trains in the veox Cloud, and jAIn asks before uploading",
        (true, true) => "your own data: /open FILE.csv, then /run; it trains in the veox Cloud, and jAIn asks before uploading",
        (false, false) => "your own data: `jain auth login` (accounts open from the waitlist: https://app.veox.ai/register); it trains in the veox Cloud, and jAIn asks before uploading",
        (false, true) => "your own data: `jain run FILE.csv --target COLUMN`; it trains in the veox Cloud, and jAIn asks before uploading",
    };
    format!("Next: {predict} · {code} · {own}")
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
    fn a_finished_run_shows_its_end_not_its_last_phase_or_live_fields() {
        let mut finished = snapshot();
        finished.state = RunState::Completed;
        // The engine leaves the last phase on a completed run (#2135).
        finished.phase = RunPhase::Finalizing;
        finished.completion_reason = Some(jope_api::CompletionReason::BudgetTime);
        let (status, activity) = render(&finished, &full_telemetry(), 120);
        assert_eq!(
            status,
            "completed · medium · 00:23/01:00 · best n/a · 41 exp"
        );
        assert_eq!(activity, "finished · time budget reached");
        finished.state = RunState::Cancelled;
        assert_eq!(render(&finished, &[], 80).1, "finished · cancelled");
    }

    #[test]
    fn time_budget_stops_are_said_plainly_with_counts_and_overrun() {
        let mut finished = snapshot();
        finished.state = RunState::Completed;
        assert_eq!(super::budget_stop_line(&finished), None);
        finished.completion_reason = Some(jope_api::CompletionReason::BudgetTime);
        finished.budget_used.evaluations = DecimalU64(15);
        finished.budget_used.wall_seconds = "68.2".into();
        assert_eq!(
            super::budget_stop_line(&finished).unwrap(),
            "Stopped at the time budget (60 s) after 15 of 128 candidates. Finishing the candidates in flight took it to 68 s. A longer budget (/effort --time, or --budget) evaluates more."
        );
        finished.budget_used.wall_seconds = "59.9".into();
        finished.budget_used.evaluations = DecimalU64(128);
        assert_eq!(
            super::budget_stop_line(&finished).unwrap(),
            "Stopped at the time budget (60 s) after 128 candidates."
        );
    }

    #[test]
    fn next_steps_name_predict_code_own_data_and_login_until_connected() {
        let run = "run_01JBD8Q6N4V2T5Y8K3M7P9R1S0";
        let tui = super::next_steps(run, false, true);
        assert!(tui.starts_with(
            "Next: predict with `jain predict --model run_01JBD8Q6N4V2T5Y8K3M7P9R1S0"
        ));
        assert!(tui.contains("/login") && tui.contains("trains in the veox Cloud"));
        assert!(tui.contains("accounts open from the waitlist: https://app.veox.ai/register"));
        assert!(tui.contains("asks before uploading") && tui.contains("/copy result"));
        let connected = super::next_steps(run, true, true);
        assert!(!connected.contains("/login") && connected.contains("/open FILE.csv, then /run;"));
        assert!(super::next_steps(run, true, false)
            .contains("`jain run FILE.csv --target COLUMN`; it trains in the veox Cloud"));
        let cli = super::next_steps(run, false, false);
        assert!(
            cli.contains("jain export run_01JBD8Q6N4V2T5Y8K3M7P9R1S0")
                && cli.contains("jain auth login")
        );
        assert!(!cli.contains(" /"), "{cli}");
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

    #[test]
    fn the_clock_moves_while_the_engine_reports_no_elapsed_time() {
        let mut running = snapshot();
        running.budget_used.wall_seconds = "0".into();
        let live = super::with_live_clock(&running, Some(std::time::Duration::from_secs(75)));
        assert!(
            render(&live, &[], 200).0.contains("01:15/01:00"),
            "{}",
            render(&live, &[], 200).0
        );
        // The engine's own figure wins when it is larger, and a finished run keeps its own.
        running.budget_used.wall_seconds = "90".into();
        let live = super::with_live_clock(&running, Some(std::time::Duration::from_secs(75)));
        assert!(render(&live, &[], 200).0.contains("01:30/01:00"));
        let mut done = snapshot();
        done.state = RunState::Completed;
        done.budget_used.wall_seconds = "12".into();
        let kept = super::with_live_clock(&done, Some(std::time::Duration::from_secs(75)));
        assert!(render(&kept, &[], 200).0.contains("00:12/01:00"));
        assert_eq!(super::with_live_clock(&running, None), running);
    }
}
