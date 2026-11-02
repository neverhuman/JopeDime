//! `verify-claims`: scoped publication claims against an evidence dependency graph
//! (W12-T01/T05/T07).
//!
//! Evidence nodes bind a subject, scope, source commit, kind and parents. A claim must match the
//! subject, scope and source of every node it cites, and its stage must be backed by a node of
//! an admissible kind; a synthetic fixture never backs live, measurement, promotion or
//! historical claims. Revoking a node revokes exactly its descendants; any claim citing a
//! revoked node must itself be recorded as revoked. History is append-only. Digests
//! authenticate bytes, not the truth of a producer.
//!
//! Synthetic identity (W12-T07) is read from the evidence itself, not from its declared kind:
//! a file under the public conformance tree (by path or by resolved location), or verified JSON
//! bytes carrying a `synthetic_identity` of the conformance schema, can never be declared
//! real-corpus, live, measurement, decision or historical evidence, and a live, measurement,
//! promotion or historical claim can never cite it.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};

use crate::py::{self, Key, PosixPath};
use crate::pyjson::{self, Dict, Value};
use crate::{require, synthetic, Error, Result};

pub const SCHEMA: &str = "jope.core-ip-claims/v1";
pub const KINDS: [&str; 8] = [
    "source-trace",
    "synthetic-fixture",
    "real-corpus-fixture",
    "hosted-ci",
    "live-host",
    "measurement",
    "decision",
    "historical-record",
];
pub const CLAIM_FIELDS: [&str; 16] = [
    "id",
    "statement",
    "subject",
    "metric",
    "units",
    "denominator",
    "comparison",
    "baseline",
    "scope",
    "source_commit",
    "observed_utc",
    "uncertainty",
    "limitations",
    "stage",
    "evidence",
    "status",
];
pub const STATUSES: [&str; 3] = ["active", "revoked", "superseded"];
/// Claim stages that synthetic evidence can never support.
pub const NOT_SYNTHETIC_STAGES: [&str; 4] = ["live", "measurement", "promotion", "historical"];
/// Kinds that attest real data, live operation, measurement, a decision or retained history,
/// which evidence of synthetic identity can never be.
pub const NOT_SYNTHETIC_KINDS: [&str; 5] = [
    "real-corpus-fixture",
    "live-host",
    "measurement",
    "decision",
    "historical-record",
];
pub const SCOPE: &str =
    "claim-to-evidence binding and revocation closure; evidence truth needs independent review";

/// Evidence kinds each claim stage accepts (`None`: not a stage).
pub fn stage_kinds(stage: &str) -> Option<&'static [&'static str]> {
    Some(match stage {
        "implementation" => &["source-trace"],
        "fixture" => &["synthetic-fixture", "real-corpus-fixture"],
        "hosted-ci" => &["hosted-ci"],
        "live" => &["live-host"],
        "measurement" => &["measurement"],
        "promotion" => &["decision"],
        "historical" => &["historical-record"],
        _ => return None,
    })
}

/// The summary `verify-claims` prints.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Summary {
    pub evidence: usize,
    pub verified_bytes: usize,
    pub custody_unverified: usize,
    pub revoked_evidence: Vec<String>,
    pub claims: BTreeMap<String, usize>,
}

impl Summary {
    pub fn to_json(&self) -> Value {
        let mut dict = Dict::new();
        dict.insert("evidence", self.evidence.into());
        dict.insert("verified_bytes", self.verified_bytes.into());
        dict.insert("custody_unverified", self.custody_unverified.into());
        dict.insert(
            "revoked_evidence",
            Value::List(
                self.revoked_evidence
                    .iter()
                    .map(|n| n.as_str().into())
                    .collect(),
            ),
        );
        dict.insert(
            "claims",
            Value::Dict(
                self.claims
                    .iter()
                    .map(|(k, v)| (k.clone(), Value::from(*v)))
                    .collect(),
            ),
        );
        dict.insert("scope", SCOPE.into());
        Value::Dict(dict)
    }
}

/// Repository evidence is always verified; custody evidence when present or required.
/// Returns the evidence file when its bytes were verified on this host.
mod revoked;
mod verify;

pub use revoked::*;
pub use verify::*;

pub(super) use verify::parents;
