use super::tui_dashboard::blocked_banner;
use super::*;

mod compact;
mod dl_charts;
mod global_status;
mod overview;
mod scoreboard;

pub(super) use compact::*;
pub(super) use dl_charts::*;
pub(super) use global_status::*;
pub(super) use overview::*;
pub(super) use scoreboard::*;

pub(super) fn render(frame: &mut Frame<'_>, dashboard: &Dashboard, refresh_ms: u64, ui: UiState) {
    let mut area = frame.area();
    // The blocked/throttled banner pre-empts the first row of EVERY tab and
    // both compact layouts: if the flywheel is starved, no view is allowed to
    // look calm.
    if let Some(text) = blocked_banner(dashboard) {
        let rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(1), Constraint::Min(0)])
            .split(area);
        frame.render_widget(
            Paragraph::new(format!("▲▲▲  {text}  ▲▲▲")).style(
                Style::default()
                    .fg(Color::White)
                    .bg(Color::Red)
                    .add_modifier(Modifier::BOLD),
            ),
            rows[0],
        );
        area = rows[1];
    }
    if area.width < 100 || area.height < 32 {
        match ui.tab {
            Tab::Overview | Tab::Gameplay => render_compact(frame, dashboard, refresh_ms, area),
            Tab::GpuDl | Tab::Dl => render_compact_gpu_dl(frame, dashboard, refresh_ms, area),
        }
        return;
    }
    match ui.tab {
        Tab::Overview => render_overview(frame, dashboard, refresh_ms, area),
        Tab::GpuDl => render_gpu_tab(frame, dashboard, refresh_ms, area),
        Tab::Gameplay => render_gameplay_tab(frame, dashboard, refresh_ms, area),
        Tab::Dl => render_dl_tab(frame, dashboard, refresh_ms, area, ui.arm),
    }
}
