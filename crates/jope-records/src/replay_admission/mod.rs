//! Controller-adopted, purpose-bound admission from a complete retained exposure closure.
//!
//! An artifact store is trusted control-plane configuration, never request-supplied authority.
//! A digest authenticates retained bytes relative to the separately adopted policy; it is not
//! a signature, a policy adoption decision, or permission to open protected payloads.
mod core;
mod labels;
mod types;

pub use core::{AdmissionVerifier, ArtifactStore, RevocationSnapshot};
pub use types::*;

pub const POLICY_SCHEMA: &str = "jope.replay-admission-policy/v2";
pub const EXPOSURE_SCHEMA: &str = "jope.replay-exposure/v2";
pub const ADMISSION_SCHEMA: &str = "jope.replay-admission/v2";
pub const ADOPTION_SCHEMA: &str = "jope.replay-policy-adoption/v2";
/// Privileged controller configuration, never a candidate-controlled pathname.
pub const ADOPTION_PATH: &str = "replay/adopted-policy.json";
/// Operator-installed component authority review artifact, never a request pathname.
pub const COMPONENT_AUTHORITY_PATH: &str = "replay/component-authority.json";
pub const COMPONENT_AUTHORITY_SCHEMA: &str = "jope.component-authority/v1";
const MAX_NODES: usize = 128;
const MAX_DEPTH: usize = 32;

fn is_hex64(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}
fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    format!("{:x}", Sha256::digest(bytes))
}

#[cfg(test)]
mod tests;
