//! `jope-telemetry`: low-overhead training and flywheel telemetry (TELEMETRY_SPEC.md).
//!
//! Module owners:
//! - luna_tel_io: the event core, collectors, sinks, the PyO3 binding and `jtel tail`/`diff`
//!   (not yet staged).
//! - luna_tel_train: `train`, `gp`, `flywheel`, `alarm`, `import`, the `record` interface and
//!   `hooks` (the trainer-side hooks ported from the former Python `jope.telemetry` package),
//!   `summary` and `fleet` (the `jtel summary` and `jtel fleet` reports).

pub mod alarm;
pub mod api;
pub mod collectors;
pub mod core;
pub mod cpu_quota;
pub mod event;
pub mod fleet;
pub mod flywheel;
pub mod gp;
pub mod hooks;
pub mod import;
pub mod manifest;
pub mod record;
pub mod sink;
pub mod summary;
pub mod train;

#[cfg(feature = "python")]
mod python;
