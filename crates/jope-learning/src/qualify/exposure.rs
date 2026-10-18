//! Versioned exposure receipt consumed by the verified qualification adapter.
//!
//! This is an additive draft wire format. It validates and hashes declared exposure; it does
//! not itself grant access to sealed labels or establish a governing adoption decision.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const EXPOSURE_SCHEMA: &str = "jope.qualify-exposure.v1";

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExposureManifest {
    pub schema: String,
    pub roster_sha256: String,
    pub evidence_sha256: String,
    pub intended_use: String,
    pub exposed_families: Vec<String>,
    pub held_out_families: Vec<String>,
}

impl ExposureManifest {
    /// Validate the declaration and return its canonical content digest.
    pub fn validate(&self) -> Result<String, String> {
        if self.schema != EXPOSURE_SCHEMA {
            return Err(format!("unexpected exposure schema {:?}", self.schema));
        }
        for (name, digest) in [
            ("roster_sha256", &self.roster_sha256),
            ("evidence_sha256", &self.evidence_sha256),
        ] {
            if !is_hex64(digest) {
                return Err(format!("{name} must be 64 lowercase hex characters"));
            }
        }
        if !matches!(self.intended_use.as_str(), "report" | "research_adoption") {
            return Err("unsupported intended use".into());
        }
        let exposed = distinct_nonempty("exposed_families", &self.exposed_families)?;
        let held_out = distinct_nonempty("held_out_families", &self.held_out_families)?;
        if !exposed.is_disjoint(&held_out) {
            return Err("a family is both exposed and held out".into());
        }
        let canonical = serde_json::to_vec(&serde_json::to_value(self).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
        Ok(sha256_hex(&canonical))
    }
}

pub(crate) fn distinct_nonempty<'a>(
    field: &str,
    values: &'a [String],
) -> Result<BTreeSet<&'a str>, String> {
    let mut out = BTreeSet::new();
    for value in values {
        if value.trim().is_empty() {
            return Err(format!("{field} contains an empty identifier"));
        }
        if !out.insert(value.as_str()) {
            return Err(format!("{field} contains duplicate identifier {value:?}"));
        }
    }
    Ok(out)
}

pub(crate) fn is_hex64(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

pub(crate) fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    hex::encode(Sha256::digest(bytes))
}
