use super::*;

pub(crate) fn render_overview(
    frame: &mut Frame<'_>,
    dashboard: &Dashboard,
    refresh_ms: u64,
    area: Rect,
) {
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Length(4),
            Constraint::Length(1),
            Constraint::Length(7),
            Constraint::Min(10),
            Constraint::Length(9),
            Constraint::Length(8),
            Constraint::Length(1),
        ])
        .split(area);
    render_header(frame, dashboard, rows[0]);
    render_global(frame, dashboard, rows[1]);
    frame.render_widget(compact_funnel_line(dashboard), rows[2]);
    render_four_task_scoreboard(frame, dashboard, rows[3]);
    let main = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(62), Constraint::Percentage(38)])
        .split(rows[4]);
    render_nodes(frame, dashboard, main[0]);
    render_queue(frame, dashboard, main[1]);
    let learning = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(rows[5]);
    render_learning(frame, dashboard, learning[0]);
    render_wave(frame, dashboard, learning[1]);
    let tail = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(45), Constraint::Percentage(55)])
        .split(rows[6]);
    render_alerts(frame, dashboard, tail[0]);
    render_events(frame, dashboard, tail[1]);
    frame.render_widget(footer_line(dashboard, refresh_ms), rows[7]);
}

pub(crate) fn stale_suffix(now_ms_value: u128, fetched_at_ms: u128) -> String {
    if fetched_at_ms == 0 {
        return " [NO DATA]".to_owned();
    }
    let age = now_ms_value.saturating_sub(fetched_at_ms);
    if age > STALE_MS {
        format!(" [STALE {}s]", age / 1_000)
    } else {
        String::new()
    }
}

pub(crate) fn sparkline(values: &[f64]) -> String {
    const BARS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    let finite = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if finite.is_empty() {
        return "-".to_owned();
    }
    let minimum = finite.iter().copied().fold(f64::INFINITY, f64::min);
    let maximum = finite.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let range = (maximum - minimum).max(f64::EPSILON);
    finite
        .iter()
        .map(|value| {
            let bucket = (((value - minimum) / range) * 7.0).round() as usize;
            BARS[bucket.min(7)]
        })
        .collect()
}

pub(crate) fn origin_label(origin: Origin) -> (&'static str, Color) {
    match origin {
        Origin::DlProposal => ("dl", Color::Magenta),
        Origin::GpOffspring => ("gp", Color::Cyan),
        Origin::LocalElite => ("elite", Color::Green),
        Origin::CommonWinner => ("robust", Color::Blue),
        Origin::Unknown => ("?", Color::DarkGray),
    }
}

#[path = "overview/gameplay_tab.rs"]
mod gameplay_tab;
#[cfg(test)]
#[path = "overview/gpu_tab.rs"]
mod gpu_tab;

pub(crate) use gameplay_tab::*;
#[cfg(test)]
pub(crate) use gpu_tab::*;

pub(crate) fn phase_label(trainer: &TrainerDetail) -> String {
    if trainer.phase.starts_with("validation") {
        format!(
            "epoch {}/{} • VALIDATION {}/{} • training batches done",
            trainer.epoch, trainer.epochs, trainer.validation_batch, trainer.validation_batches
        )
    } else {
        format!(
            "epoch {}/{} • batch {}/{} ({:.0}%) • {}",
            trainer.epoch,
            trainer.epochs,
            trainer.batch,
            trainer.total_batches,
            trainer.pct,
            trainer.phase
        )
    }
}

pub(crate) fn trend_arrow(prior: f64, last: f64) -> &'static str {
    if last < prior {
        "↓"
    } else if last > prior {
        "↑"
    } else {
        "→"
    }
}

// The per-epoch chart series (train/val loss, pairwise, ablation) are produced
// by `continuous_series` from the cross-generation `DlHistory`; the old
// single-run extractors keyed on one trainer's history were what emptied the
// chart on job turnover and have been removed.

/// Training arms in a stable display order: real trainers first (so a proposal
/// or encoder GPU job never steals arm 0), each keeping its collector ordering.
pub(crate) fn training_arms(dashboard: &Dashboard) -> Vec<&TrainerDetail> {
    let mut arms: Vec<&TrainerDetail> =
        dashboard.trainers.iter().filter(|t| t.is_trainer).collect();
    arms.extend(dashboard.trainers.iter().filter(|t| !t.is_trainer));
    arms
}

/// Newest metric row for an arm, or None while its metrics file is still empty.
/// Callers must render "warming up" rather than borrowing another arm's numbers.
pub(crate) fn arm_latest(trainer: &TrainerDetail) -> Option<&TrainerMetricPoint> {
    trainer.history.last()
}

pub(crate) fn gpu_for_job<'a>(dashboard: &'a Dashboard, job_id: &str) -> Option<&'a GpuRow> {
    dashboard
        .gpus
        .iter()
        .find(|gpu| gpu.owner_job.as_deref() == Some(job_id))
}
