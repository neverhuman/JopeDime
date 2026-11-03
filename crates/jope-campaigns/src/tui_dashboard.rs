use super::*;

#[path = "tui_dashboard/build.rs"]
mod build;
pub(crate) use build::build_dashboard;
#[path = "tui_dashboard/campaign.rs"]
mod campaign;
pub(crate) use campaign::enrich_gpu_admission;
pub(crate) use campaign::parse_campaign;
#[path = "tui_dashboard/cluster.rs"]
mod cluster;
pub(crate) use cluster::aggregate_cluster;
pub(crate) use cluster::parse_families;
pub(crate) use cluster::parse_nodes;
#[path = "tui_dashboard/queue.rs"]
mod queue;
pub(crate) use queue::parse_events;
pub(crate) use queue::parse_failures;
pub(crate) use queue::parse_learning;
pub(crate) use queue::parse_queue;
pub(crate) use queue::parse_wave;
#[path = "tui_dashboard/alerts.rs"]
mod alerts;
pub(crate) use alerts::blocked_banner;
pub(crate) use alerts::cockpit_state;
pub(crate) use alerts::collect_alerts;
pub(crate) use alerts::why_idle_chain;

#[cfg(test)]
pub(super) fn render(frame: &mut Frame<'_>, dashboard: &Dashboard, refresh_ms: u64, ui: UiState) {
    super::tui_render::render(frame, dashboard, refresh_ms, ui);
}
