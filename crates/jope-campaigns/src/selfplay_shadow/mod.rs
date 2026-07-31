//! Causal, cache-exact self-play contracts for the shadow ratchet.
//!
//! This module is deliberately additive.  The production generation-zero
//! services continue to serve their accepted anchor while shadow artifacts use
//! their own schemas and namespace.  The module owns policy and validation;
//! callers own model inference, the authoritative bridge, and evaluator work.

mod contracts;
mod flywheel;
mod policy;
mod state;
mod store;
mod v2;
mod v3;

pub use contracts::*;
pub use flywheel::*;
pub use policy::*;
pub use state::*;
pub use store::*;
pub use v2::*;
pub use v3::*;

pub const EVALUATIONS_PER_CONTEXT: usize = 256;
pub const DATASETS_PER_COHORT: usize = 64;
pub const DIRECT_BUDGETS: [usize; 5] = [8, 16, 32, 48, 64];
pub const MIN_PARENT_DIRECT_OUTCOMES: usize = 2_048;
pub const ENCODING_CHUNK_SIZE: usize = 256;
pub const ENCODING_TRANCHE_SIZE: usize = 512;
pub const ENCODING_RETRY_INTERVAL_MS: u64 = 120_000;
pub const FIRST_TOUCH_RELEASE_FRACTION: f64 = 0.30;
