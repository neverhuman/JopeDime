//! `jope-telemetry`: low-overhead training and flywheel telemetry (TELEMETRY_SPEC.md).
//!
//! Module owners:
//! - luna_tel_io: the event core, collectors, sinks, the PyO3 binding and `jtel tail`/`diff`
//!   (not yet staged).
//! - luna_tel_train: `train`, `gp`, `flywheel`, `alarm`, `import` and the `record` interface.

pub mod alarm;
pub mod api;
pub mod collectors;
pub mod core;
pub mod event;
pub mod flywheel;
pub mod gp;
pub mod import;
pub mod manifest;
pub mod record;
pub mod sink;
pub mod summary;
pub mod train;

#[cfg(feature = "python")]
mod python;
