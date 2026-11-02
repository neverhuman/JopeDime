//! W8 product integration: the publisher that turns the engine's durable events into committed
//! W8 progress.
//!
//! [`Driver`] appends verified engine events to the durable outbox, commits them through the
//! conditional publisher, and drives flush and checkpoint cadence. A corpus freeze, checkpoint
//! commit, HWM change, or shutdown flushes at once; size and interval triggers handle ordinary
//! progress. W8 checkpoints follow committed optimizer steps or controlled shutdown.
//!
//! **Configuration.** Outbox limits and cadence are explicit [`Config`] values. The adopted
//! defaults and their approval state are reported; a configuration file can override them, and
//! the report identifies those values as operator configured.
//!
//! **Store.** The remote is the store explicitly opened by the caller. The CLI uses only the
//! store named by `--store` or `JOPE_LEARNING_URL`; tests use file and memory stores. No default
//! bucket is selected here.
//!
//! **Restart.** The driver re-derives the run manifest, committed and pending objects, and last
//! checkpoint from the outbox and remote closure. A restart flushes owed work, and replaying the
//! same bytes yields the same outbox objects.

mod driver;
mod types;

pub use driver::Driver;
pub use types::{
    CheckpointStep, Config, ConfigReport, Step, SystemClock, APPROVAL, CONFIG_SCHEMA, OVERRIDDEN,
};
