//! CLOVER identity, evidence and bounded discovery contracts. No optimizer training.
pub mod expansion;
mod identity;
pub mod round_qual;
mod store;
mod visibility;

pub use identity::{Budget, ExperimentIdentity, InitialState, SourceIdentity};
pub use store::{AttemptOutcome, EvidenceStore, RunStore};
pub use visibility::{project_encoder_input, EncoderInput, Observation, Visibility};

/// Schema identifier for  values.
pub const SCHEMA: &str = "clover.experiment/v1";
/// Constant identifying the daily bytes cap.
pub const DAILY_BYTES_CAP: u64 = 25_000_000_000;
/// Constant identifying the daily puts cap.
pub const DAILY_PUTS_CAP: u64 = 1_500_000;
