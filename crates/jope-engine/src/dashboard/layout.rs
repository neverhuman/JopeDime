use super::campaign::{goal_phase, gpu_reason, milestone_lines};
use super::details::{
    comparison_lines, decimal_count, neural_lines, teacher_evolution_lines, teacher_lines,
};
use super::evidence::{decimal, number, phase, rmse, status_age};
use super::prelude::*;
use super::tasks::{policy_task_state, workflow_rows};

#[derive(Default)]
pub struct History {
    pub(crate) gpu: [VecDeque<u64>; 8],
    last: u64,
    pub(crate) worker_offset: usize,
    pub compact_offsets: [u16; 5],
    pub learning: std::sync::Arc<crate::learning_history::Snapshot>,
    pub learning_navigation: crate::learning_history::Navigation,
    pub learning_details: bool,
}
impl History {
    pub(crate) fn update(&mut self, h: &Hardware) {
        if h.timestamp == self.last {
            return;
        }
        self.last = h.timestamp;
        for g in &h.gpus {
            if let Some(history) = self.gpu.get_mut(g.index) {
                history.push_back(g.utilization as u64);
                if history.len() > 120 {
                    history.pop_front();
                }
            }
        }
    }
}
pub(crate) fn compact_layout(area: Rect) -> bool {
    area.width < 100 || area.height < 40
}

// Reflow whole records before scrolling so narrow terminals retain every field,
// including long paths. Full layouts retain their existing tables and charts.
pub(crate) fn compact_wrap(lines: Vec<Line<'static>>, width: u16) -> Vec<Line<'static>> {
    let mut output = Vec::new();
    for line in lines {
        let text = line
            .spans
            .iter()
            .map(|span| span.content.as_ref())
            .collect::<String>();
        for paragraph in text.split('\n') {
            let mut current = String::new();
            let mut used = 0;
            for word in paragraph.split_whitespace() {
                let word_width = Span::raw(word).width();
                if used > 0 && used + 1 + word_width > width as usize {
                    output.push(Line::raw(std::mem::take(&mut current)));
                    used = 0;
                }
                if used > 0 {
                    current.push(' ');
                    used += 1;
                }
                for ch in word.chars() {
                    let size = Span::raw(ch.to_string()).width();
                    if used + size > width as usize && !current.is_empty() {
                        output.push(Line::raw(std::mem::take(&mut current)));
                        used = 0;
                    }
                    current.push(ch);
                    used += size;
                }
            }
            output.push(Line::raw(current));
        }
    }
    output
}
/// Observed search, training and run evidence rendered together.
#[derive(Clone, Copy)]
pub struct DashboardData<'a> {
    pub gp: &'a Value,
    pub neural: &'a Value,
    pub run: &'a Value,
    pub events: &'a str,
    pub history: &'a History,
}

pub(crate) fn compact_lines(
    view: usize,
    h: &Hardware,
    data: DashboardData<'_>,
    width: u16,
) -> Vec<Line<'static>> {
    let DashboardData {
        gp,
        neural,
        run,
        events,
        history,
    } = data;

    let policy = if run["policy"].is_object() {
        &run["policy"]
    } else {
        neural
    };
    let (status, detail) = policy_task_state(run, policy);
    let mut lines = Vec::new();
    match view {
        0 => {
            lines.push(Line::raw(format!("Policy optimization: {status}")));
            lines.push(Line::raw(detail));
            for row in workflow_rows(run, gp, neural) {
                if row.name == "Policy optimization" {
                    continue;
                }
                lines.push(Line::raw(format!(
                    "{}: {} · {}",
                    row.name, row.status, row.progress
                )));
                lines.push(Line::raw(row.detail));
            }
            lines.extend(milestone_lines(run, gp, neural));
            lines.extend(teacher_lines(run));
            lines.extend(comparison_lines(run));
            lines.push(Line::raw(gpu_reason(run, neural, h)));
        }
        1 => {
            let fresh =
                h.timestamp > 0 && h.timestamp <= now() && now().saturating_sub(h.timestamp) <= 3;
            lines.push(Line::raw(if fresh {
                "Hardware: current observation"
            } else {
                "Hardware: STALE / UNAVAILABLE"
            }));
            lines.push(Line::raw(format!(
                "GPU devices observed: {} / 8",
                h.gpus.len()
            )));
            for g in &h.gpus {
                lines.push(Line::raw(format!(
                    "GPU {}: {:.1}% · {:.1}/{:.1} GiB · {:.0} W · {:.0} C",
                    g.index,
                    g.utilization,
                    g.memory_mb / 1024.,
                    g.memory_total_mb / 1024.,
                    g.watts,
                    g.temperature
                )));
            }
            lines.push(Line::raw(format!(
                "RAM {:.1}/{:.1} GiB; available {:.1}; disk free {:.1} GiB",
                h.ram_used_gb, h.ram_total_gb, h.available_gb, h.disk_free_gb
            )));
            lines.push(Line::raw(format!(
                "CPU total {:.1}% · load {}",
                h.cpu_total,
                h.load.trim()
            )));
            for (cpu, activity) in h.cpu.iter().enumerate() {
                lines.push(Line::raw(format!("CPU {cpu}: {activity:.1}%")));
            }
            for numa in &h.numa {
                lines.push(Line::raw(format!(
                    "NUMA {}: {:.1}% [{}]",
                    numa.node, numa.activity, numa.cpus
                )));
            }
        }
        2 => {
            lines.push(Line::raw(format!("Policy optimization: {status}")));
            lines.push(Line::raw(detail));
            lines.push(Line::raw("Training fit is not generalization."));
            lines.push(Line::raw("Validation is adaptive; compare the same panel."));
            lines.push(Line::raw(format!(
                "Saved checkpoint: {}",
                run["selected_checkpoint"]["checkpoint"]
                    .as_str()
                    .unwrap_or("unavailable")
            )));
            lines.push(Line::raw(format!(
                "Full saved history: {} runs · {} optimizer records",
                history.learning.runs.len(),
                history
                    .learning
                    .runs
                    .iter()
                    .map(|run| run.points.len())
                    .sum::<usize>()
            )));
            // Latest losses before the per-run list, so they stay in the first visible rows.
            let (summary, notice) = crate::learning_history::latest_summary(&history.learning);
            lines.extend(summary.into_iter().map(Line::raw));
            lines.push(Line::raw(notice));
            for saved in &history.learning.runs {
                lines.push(Line::raw(format!(
                    "{}: {} · {} records · {} attempts",
                    saved.path.display(),
                    saved.status,
                    saved.points.len(),
                    saved.attempts
                )));
                if let Some(point) = saved.points.last() {
                    lines.push(Line::raw(format!(
                        "Saved step {} · training NLL {:.5}",
                        point.step, point.train
                    )));
                }
            }
            lines.extend(history.learning.errors.iter().cloned().map(Line::raw));
            if history.learning_details {
                lines.extend(neural_lines(policy));
            }
            lines.push(Line::raw(
                "Enlarge for full-history charts; d toggles optimizer details.",
            ));
        }
        3 => {
            lines.push(Line::raw(format!(
                "GP search: {} · {}",
                phase(gp),
                status_age(gp)
            )));
            lines.push(Line::raw(format!(
                "{} active · {} queued · {} completed · {} failed",
                decimal_count(gp, "active"),
                number(gp, "queued"),
                number(gp, "completed_views"),
                number(gp, "failed_views")
            )));
            for worker in gp["active_rows"].as_array().into_iter().flatten() {
                lines.push(Line::raw(format!(
                    "Worker {} · {} · cut {} · {} evaluations · RMSE {} · {} s",
                    number(worker, "slot"),
                    worker["dataset"].as_str().unwrap_or("unknown"),
                    number(worker, "cut"),
                    number(worker, "evaluations"),
                    rmse(worker, "best_rmse"),
                    decimal(worker, "elapsed_seconds")
                )));
            }
            lines.extend(teacher_evolution_lines(run));
            lines.extend(comparison_lines(run));
        }
        _ => {
            lines.push(Line::raw(format!(
                "Campaign: {}",
                goal_phase(run, neural, gp)
            )));
            lines.push(Line::raw(format!(
                "{} saved contexts · {} families",
                number(&run["contexts"], "completed"),
                number(&run["contexts"], "families")
            )));
            lines.push(Line::raw(
                "Final tests stay sealed. Displayed validation is adaptive.",
            ));
            lines.push(Line::raw("Recent events (newest first):"));
            lines.extend(
                events
                    .lines()
                    .rev()
                    .take(30)
                    .map(|line| Line::raw(line.to_owned())),
            );
        }
    }
    lines.extend(
        h.errors
            .iter()
            .map(|error| Line::raw(format!("Hardware error: {error}"))),
    );
    compact_wrap(lines, width)
}
pub(crate) fn render_compact(f: &mut Frame, view: usize, h: &Hardware, data: DashboardData<'_>) {
    let DashboardData {
        gp,
        neural,
        run,
        events,
        history,
    } = data;

    let areas = Layout::vertical([
        Constraint::Length(1),
        Constraint::Length(1),
        Constraint::Min(1),
        Constraint::Length(1),
    ])
    .split(f.area());
    let title = [
        "Overview",
        "Hardware",
        "Deep learning",
        "GP search",
        "Data / Events",
    ][view.min(4)];
    f.render_widget(
        Paragraph::new(format!("JOPE / PARALLEL · {title}"))
            .style(Style::default().fg(BLUE).bold()),
        areas[0],
    );
    f.render_widget(
        Paragraph::new("1 Overview 2 HW 3 DL 4 GP 5 Events"),
        areas[1],
    );
    let lines = compact_lines(
        view,
        h,
        crate::dashboard::DashboardData {
            gp,
            neural,
            run,
            events,
            history,
        },
        areas[2].width,
    );
    let maximum = lines
        .len()
        .saturating_sub(areas[2].height as usize)
        .min(u16::MAX as usize) as u16;
    let offset = history.compact_offsets[view.min(4)].min(maximum);
    let count = lines.len();
    f.render_widget(Paragraph::new(lines).scroll((offset, 0)), areas[2]);
    f.render_widget(
        Paragraph::new(format!(
            "↑↓ scroll {}/{} · ←→ tabs · q detach",
            u32::from(offset) + 1,
            count
        )),
        areas[3],
    );
}
