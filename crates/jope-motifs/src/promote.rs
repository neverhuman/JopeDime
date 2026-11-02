//! Additive qualification for one motif in a new research vocabulary.
//!
//! This module does not mutate `MotifTable`: its V0 verifier intentionally admits only
//! quarantined entries. A caller must bind the candidate to a verified table and pass a
//! source-verified validation report before an admission artifact can be produced.

use crate::fragment::{self, stage_texts};
use crate::source::QualifiedSet;
use crate::table::{Lifecycle, MotifTable, Promotion};
use crate::validate::{self, ValidateOptions, Validation, VALIDATION_SCHEMA};
use crate::NAME_PREFIX;
use feat_math::dsl::{parse_dsl, DslExpr, Stage};
use feat_math::GenomeIr;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

pub const ADMISSION_SCHEMA: &str = "jope.motif-admission/v1";
pub const CONTRAST_SCHEMA: &str = "jope.motif-contrast/v1";
pub const VOCABULARY_SCHEMA: &str = "jope.motif-vocabulary-migration/v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AdmissionError(pub &'static str);

impl std::fmt::Display for AdmissionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0)
    }
}

impl std::error::Error for AdmissionError {}

type Result<T> = std::result::Result<T, AdmissionError>;

mod candidate;
pub use candidate::{ExpansionLimits, ExpansionWitness, MotifCandidate, ProbeSimilarity};
mod contrast;
pub use contrast::{
    ContrastObservation, EvidenceBoundary, EvidenceRole, MetricDirection, PreregisteredContrast,
    ValidatedContrast,
};
mod migration;
pub use migration::VocabularyMigration;
mod admission;
pub use admission::MotifAdmission;

fn motif_id_fn(expansion: &str) -> String {
    let mut bytes = b"jope-motif-expansion/v1\0".to_vec();
    bytes.extend_from_slice(expansion.as_bytes());
    sha256(&bytes)
}

fn motif_name_fn(id: &str) -> String {
    format!("{NAME_PREFIX}{}", id.get(..16).unwrap_or(id))
}

fn is_hex64(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|b| b.is_ascii_hexdigit())
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
