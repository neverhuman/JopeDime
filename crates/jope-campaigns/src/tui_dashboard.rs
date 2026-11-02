use super::*;

#[path = "tui_dashboard/build.rs"]
mod build;
pub(super) use build::build_dashboard;
#[path = "tui_dashboard/campaign.rs"]
mod campaign;
pub(super) use campaign::enrich_gpu_admission;
pub(super) use campaign::infrastructure_attempt_number;
pub(super) use campaign::parse_campaign;
#[path = "tui_dashboard/cluster.rs"]
mod cluster;
pub(super) use cluster::aggregate_cluster;
pub(super) use cluster::parse_families;
pub(super) use cluster::parse_nodes;
#[path = "tui_dashboard/queue.rs"]
mod queue;
pub(super) use queue::parse_events;
pub(super) use queue::parse_failures;
pub(super) use queue::parse_learning;
pub(super) use queue::parse_queue;
pub(super) use queue::parse_wave;
#[path = "tui_dashboard/alerts.rs"]
mod alerts;
pub(super) use alerts::add_stale_source_alert;
pub(super) use alerts::blocked_banner;
pub(super) use alerts::cockpit_state;
pub(super) use alerts::collect_alerts;
pub(super) use alerts::why_idle_chain;

#[path = "../tui_render.rs"]
mod tui_render;
