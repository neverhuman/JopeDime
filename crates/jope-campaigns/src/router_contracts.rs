//! Public task-neutral contracts for Jain's per-family V2 router.
//!
//! These types contain no protected search implementation. They describe the
//! feature view, member route, validation boundary, and opaque receipts that
//! can be shared by Core, CLI, and a cloud client.

use serde::Serialize;
use sha2::{Digest, Sha256};

pub const ROUTER_SEEDS_V2: [u64; 2] = [303, 404];
pub const ROUTER_FOLDS_V2: usize = 7;

fn canonical_sha256<T: Serialize>(value: &T) -> Result<String, String> {
    let bytes = serde_json::to_vec(value).map_err(|error| format!("canonical JSON: {error}"))?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn validate_sha256(name: &str, value: &str) -> Result<(), String> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(format!("{name} is not a lowercase SHA-256 digest"));
    }
    Ok(())
}

mod predictions;
mod routes;
mod validation;

pub use predictions::*;
pub use routes::*;
pub use validation::*;
