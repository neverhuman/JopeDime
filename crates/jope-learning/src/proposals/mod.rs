//! Proposal delivery over immutable, already accepted draw records.
//!
//! This module is a draft I26 worker API. It does not create draw records, execution grants, or
//! optimizer credit. The pipeline/CLI fence-and-release hook is coordinator-owned and must be
//! wired before this helper can authorize a production fit.

pub mod outbox;
pub mod receiving;

#[cfg(test)]
mod tests;
