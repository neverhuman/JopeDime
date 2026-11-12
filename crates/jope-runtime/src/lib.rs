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
pub mod calibration;
pub mod error;
pub mod fitness;
pub mod fitness_classification;
pub mod fitted;
pub mod fitted_js;
pub mod fitted_py;
pub mod fitted_rs;
pub mod landmark_args;
pub mod limits;
pub mod math_pipeline;
pub mod meter;
pub mod methods;
pub mod model_object;
pub mod model_session;
pub mod output_safety;
mod prediction_narration;
pub mod receipt;
pub mod service;
pub mod table;
pub mod task;

pub use error::{Failure, FailureKind};

/// Version of the runtime contract a fitted file and a receipt name.
pub const RUNTIME_ABI: &str = "jope.appliance-runtime/v1";

/// Lowercase hex of raw bytes. This does not hash.
pub fn encode_hex(bytes: &[u8]) -> String {
    hex::encode(bytes)
}

/// Lower-case hex sha256.
pub fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    encode_hex(&Sha256::digest(bytes))
}

#[cfg(test)]
mod tests {
    use super::{encode_hex, sha256_hex};

    #[test]
    fn encode_hex_does_not_hash_and_sha256_matches_abc() {
        assert_eq!(encode_hex(&[0x0a, 0xff]), "0aff");
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}

pub mod strict_json;
