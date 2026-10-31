//! Bounded telemetry recording and training/flywheel reports.
//!
//! [`core`] manages producer queues and the asynchronous event drain; [`event`],
//! [`manifest`] and [`sink`] define the recorded events, run metadata and output sinks.
//! [`record`] provides the low-rate typed record interface used by [`hooks`].
//!
//! [`train`], [`gp`], [`flywheel`] and [`alarm`] build training and evolutionary-search
//! summaries. [`api`] exposes JSON entry points for these reports; [`summary`] and
//! [`fleet`] read recorded runs. [`cpu_quota`], [`collectors`] and [`nvml_device`]
//! supply CPU-capacity accounting and host/device probes.
//!
//! The `jtel` binary provides `tail`, `import`, `parquet`, `recover`, `summary`, `diff`
//! and `fleet` commands. The default `parquet` feature enables the Parquet sink and
//! the CLI's Parquet regeneration and recovery operations. Use `--no-default-features`
//! for the JSONL-only build. The optional `python` feature adds the PyO3 bindings;
//! the Rust interfaces and `jtel` do not require it.

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
pub mod nvml_device;
pub mod record;
pub mod sink;
pub mod summary;
pub mod train;

#[cfg(feature = "python")]
mod python;
