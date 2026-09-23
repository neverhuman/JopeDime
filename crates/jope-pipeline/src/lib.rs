//! The CPU spine of the product pipeline: `dataset -> prepare -> search -> select
//! -> emit -> receipt`.
//!
//! Nothing here is a new numerical method. Each stage composes what the workspace
//! already owns: `jope-datasets` resolves and verifies the dataset offline,
//! [`prepare`] partitions its training rows into folds with
//! `feat_math::folds::fold_of`, [`search`] runs a genetic-programming search
//! whose every fit is `feat_math::fit`, and [`emit`] transpiles the champion to
//! standalone Python, Rust, C++ and JavaScript. Every stage records what it did —
//! including what it skipped and why — in a `jope.pipeline-receipt/v2`.
//!
//! This crate deliberately does **not** depend on `jope-engine`. The engine's GP
//! search cannot measure anything in a shipped binary — `eval_registry/authority.rs`
//! blocks every capability unconditionally outside `cfg(test)`, and
//! `fit_boundary.rs` fences fitting behind a service socket — so driving it here
//! produced a bootstrap genome on every run and nothing else. [`search`] carries
//! the full argument.
//!
//! Two stages of the owner's end state are deliberately absent and are named in
//! the receipt rather than silently passed over: the encode/propose stage
//! (`--policy`) and the S3 / router-training stages.

pub mod cli;
pub mod dataset;
pub mod emit;
#[cfg(target_os = "linux")]
pub mod execution;
pub mod experiment;
pub mod experiment_v2;
pub mod front_door;
pub mod prepare;
pub mod receipt;
pub mod run;
pub mod search;
pub mod select;
#[cfg(target_os = "linux")]
pub mod supervisor;

pub use receipt::{Receipt, RECEIPT_SCHEMA_ID};
pub use run::{run, Policy, RunOptions};

/// Version stamped into every receipt.
pub const TOOL_VERSION: &str = concat!("jope-pipeline/", env!("CARGO_PKG_VERSION"));

/// The default fold seed, matching the workspace's own `manifest::SEED` so a
/// preparation built here lands on the same folds the rest of the tree uses.
pub const DEFAULT_SEED: u64 = 20260907;

/// Lowercase hex sha256, the one digest form every receipt field carries.
pub fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    format!("{:x}", Sha256::digest(bytes))
}

/// Lowercase hex sha256 of a file's bytes.
pub fn sha256_file(path: &std::path::Path) -> anyhow::Result<String> {
    use anyhow::Context;
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    Ok(sha256_hex(&bytes))
}
