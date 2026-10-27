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
