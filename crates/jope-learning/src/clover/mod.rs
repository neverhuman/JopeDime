//! Model-free CLOVER Phase 0 contracts. These APIs do not launch execution or training.
mod identity;
mod store;
mod visibility;

pub use identity::{Budget, ExperimentIdentity, InitialState, SourceIdentity};
pub use store::{AttemptOutcome, EvidenceStore, RunStore};
pub use visibility::{project_encoder_input, EncoderInput, Observation, Visibility};

pub const SCHEMA: &str = "clover.experiment/v1";
pub const DAILY_BYTES_CAP: u64 = 25_000_000_000;
pub const DAILY_PUTS_CAP: u64 = 1_500_000;
