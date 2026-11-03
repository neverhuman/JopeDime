use super::campaign::{goal_phase, gpu_reason, milestone_lines};
use super::details::{
    binary_replay_detail, comparison_lines, decimal_count, learning_chart, neural_lines,
    teacher_evolution_lines, teacher_lines,
};
use super::evidence::{age, completed, decimal, duration, number, panel, phase, rmse, status_age};
use super::layout::{DashboardData, History, compact_layout, render_compact};
use super::prelude::*;
use super::tasks::{TaskRow, label, policy_task_state, workflow_rows};

mod gp_search;
mod hardware;
mod learning;
mod overview;
mod provenance;

pub fn render(f: &mut Frame, view: usize, h: &Hardware, data: DashboardData<'_>) {
    let DashboardData {
        gp,
        neural,
        run,
        events,
        history,
    } = data;

    let observed_gp = crate::telemetry::observed_status(gp, now());
    let gp = &observed_gp;
    f.render_widget(
        Block::default().style(Style::default().bg(Color::Rgb(9, 17, 28)).fg(INK)),
        f.area(),
    );
    if f.area().width < 48 || f.area().height < 15 {
        f.render_widget(
            Paragraph::new(
                "JOPE / PARALLEL\nResize to at least 48 × 15.\nq detaches; services keep running.",
            )
            .block(panel(" Terminal size ")),
            f.area(),
        );
        return;
    }
    if compact_layout(f.area()) {
        render_compact(
            f,
            view,
            h,
            crate::dashboard::DashboardData {
                gp,
                neural,
                run,
                events,
                history,
            },
        );
        return;
    }
    let areas = Layout::vertical([
        Constraint::Length(3),
        Constraint::Length(3),
        Constraint::Min(1),
        Constraint::Length(1),
    ])
    .split(f.area());
    let stale = h.timestamp == 0 || now().saturating_sub(h.timestamp) > 3;
    f.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(" JOPE / PARALLEL  ", Style::default().fg(BLUE).bold()),
            Span::raw(format!(
                "8 × A100  ·  {} CPUs  ·  {}",
                h.cpu.len(),
                goal_phase(run, neural, gp)
            )),
        ]))
        .block(panel(
            " LEARNING OPERATIONS · live receipts, saved evidence ",
        )),
        areas[0],
    );
    f.render_widget(
        Tabs::new([
            "1 Overview",
            "2 Hardware",
            "3 Deep learning",
            "4 GP search",
            "5 Data / Events",
        ])
        .select(view)
        .highlight_style(Style::default().fg(Color::Rgb(9, 17, 28)).bg(BLUE).bold())
        .style(Style::default().fg(MUTED))
        .block(panel(" Navigate ")),
        areas[1],
    );
    let body = areas[2];
    match view {
        0 => overview::render(f, body, h, run, gp, neural, stale),
        1 => hardware::render(f, body, h, history, stale),
        2 if !history.learning_details => learning::render_history(f, body, history, run),
        2 => learning::render_policy(f, body, run, neural),
        3 => gp_search::render(f, body, run, gp, history),
        _ => provenance::render(f, body, h, run, gp, neural, events),
    }
    f.render_widget(
        Paragraph::new(if stale {
            " TELEMETRY STALE / UNAVAILABLE · q detaches; workers keep running"
        } else if h.gpus.len() != 8 || h.cpu.is_empty() {
            " PARTIAL TELEMETRY · inspect Hardware / Events · q detaches; workers keep running"
        } else {
            " LIVE · 1 Hz telemetry / 4 fps · ← → or 1–5 navigate · q detaches"
        })
        .style(Style::default().fg(if stale {
            Color::Yellow
        } else {
            Color::DarkGray
        })),
        areas[3],
    );
}
