//! JOPE flywheel control plane: the self-play services, watchdogs, garbage
//! collectors, alarms, TUI, task-learning controller and training
//! archive/history/replica tools — one binary with argv[0] dispatch (see
//! `jope_campaigns::dispatch`), so the historical `jope_selfplay_*` names used
//! by the systemd units keep working through installed links.
//!
//! The library half of the control plane (`selfplay_*`, `selfplay_shadow`,
//! `spec_policy`, contracts, paths) lives in `jope-campaigns`, which both
//! binaries share.

pub mod commands;
/// The freshness wire protocol shared by `jope_selfplay_freshness` and its MCP adapter.
pub mod freshness_protocol;

pub use jope_campaigns::dispatch;

/// The `jope-selfplay` binary's program table.
pub static TABLE: dispatch::Table = dispatch::Table {
    binary: "jope-selfplay",
    about: "JOPE flywheel control plane: queue, supply, wave, promotion, propose, expansion, corpus freezer, watchdogs, gc, alarms, TUI, task-learning controller, training archive/history/replica",
    programs: commands::PROGRAMS,
};
