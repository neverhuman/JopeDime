//! Verified model-bundle handling and encoder APIs for JopeDime.
//!
//! This crate groups model bundle validation and loading, foundation and target
//! encoders, JOPE policy inference, and the optional DOPE training/session APIs.
//! The `dope` feature gates the latter module. Encoder and inference behavior is
//! defined by the repository's frozen contracts and parity fixtures; exposing an
//! API here does not by itself qualify a complete learning or serving loop.

pub mod bundle;
pub mod cli;
pub mod data;
pub mod foundation;
pub mod jope_policy;
pub mod landmark_args;
pub mod metrics;
pub mod model;
pub mod target_encoder;

/// Frozen Gen31 PyTorch oracle used by consumers to qualify their JOPE proposal path.
pub const JOPE_GEN31_PARITY_FIXTURE_JSON: &[u8] =
    include_bytes!("../tests/fixtures/jope_gen31_context_014b93f43dbaa1a4.json");

#[cfg(feature = "dope")]
pub mod dope_encoder;
