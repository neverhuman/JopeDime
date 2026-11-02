//! Draft, versioned typed-edit policy ABI.
//!
//! This is an additive challenger. It does not alter the 401-token grammar or accepted model
//! identities. I08's graph/effect types and I25's fit-closure adapters are still draft inputs;
//! `ResidualClosure` records the corresponding binding without claiming those adapters exist.

use anyhow::{ensure, Result};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

#[path = "edit_policy/context.rs"]
mod context;
#[path = "edit_policy/policy.rs"]
mod policy;
#[cfg(test)]
#[path = "edit_policy/tests.rs"]
mod tests;

pub use context::*;
pub use policy::*;

pub const EDIT_POLICY_SCHEMA: &str = "jope.typed-edit-policy.v1";
pub const CONTEXT_ABI_SCHEMA: &str = "jope.context-abi.v2";
pub const EDIT_ARCHITECTURE: &str = "N1_factorized_typed_mixture_v1";
pub const CONTEXT_FEATURES: usize = 2048;
pub const MAX_PROGRAM_TOKENS: usize = 128;
pub const MAX_SCHEMA_COLUMNS: usize = 4096;
pub const REFERENCE_BACKEND: &str = "cpu_f64_reference_v1";
const DIGEST_LEN: usize = 64;

fn sorted_unique(mut values: Vec<String>) -> Result<Vec<String>> {
    ensure!(
        values.iter().all(|value| !value.is_empty()),
        "empty selected column key"
    );
    values.sort();
    ensure!(
        values.windows(2).all(|pair| pair[0] != pair[1]),
        "duplicate selected column key"
    );
    Ok(values)
}

fn schema_digest(columns: &[ColumnDescriptor]) -> Result<String> {
    digest_json(&(CONTEXT_ABI_SCHEMA, columns))
}

fn digest_json(value: &impl Serialize) -> Result<String> {
    let bytes = serde_json::to_vec(value)?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn is_digest(value: &str) -> bool {
    value.len() == DIGEST_LEN
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}
