//! The `jopedime` appliance runtime (`deploy/appliance/SPECIFICATION.md` §4.1, §6, §13).
//!
//! It owns the local table contract, the method registry, native fitting and prediction,
//! bounded automatic selection, the fitted-state file and the run receipt. Numerical work is not
//! reimplemented here: fitting and prediction call `feat_math::advanced_v3`, and the roster,
//! raw-cell encoding and bound prediction were *moved* here from the frozen-panel adapter
//! (`jope-bench`) and `jope-pipeline`, which now re-export them, so the research panel and the
//! appliance execute the same code.
//!
//! The crate depends on `feat-math` (without its C++ runtime) and small pure-Rust crates only.
//! It never fetches data or models, runs a subprocess, or reads `JOPE_*` research variables
//! (`crates/jope-appliance/tests/dependency_fence.rs` checks the shipped closure).
#![forbid(unsafe_code)]

pub mod api;
pub mod automl;
pub mod backend;
pub mod budget;
pub mod error;
pub mod fitted;
pub mod limits;
pub mod methods;
pub mod model_object;
pub mod model_session;
pub mod receipt;
pub mod table;
pub mod task;

pub use error::{Failure, FailureKind};

/// Version of the runtime contract a fitted file and a receipt name.
pub const RUNTIME_ABI: &str = "jope.appliance-runtime/v1";

/// Lower-case hex sha256.
pub fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    hex::encode(Sha256::digest(bytes))
}

pub mod strict_json;
