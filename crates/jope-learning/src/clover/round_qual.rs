//! Offline self-play round qualification diagnostic report (`jope.selfplay-round-qualification/v1`).
//!
//! Evaluates multi-round self-play ratchet outputs (`clover.selfplay-ratchet/v1`), once-only
//! test-split score receipts (`jope.pipeline-test-score/v1`), R5 parity records, a parent-cluster
//! mapping, and an optional pre-registration packet (`jope.selfplay-round1-prereg/v1`) against a
//! caller-supplied [`ExperimentIdentity`] and [`RoundQualPlan`].
//!
//! **Provenance and pre-registration binding (`B2`) and offline hard stop (`R1`/`R2`/`R3`):**
//! - Every [`RoundQualReport`] emitted by this offline verifier sets `qualified_claim: false`,
//!   `promotion: false`, and `l6_claim: false` **unconditionally** (literal `false`, never derived
//!   solely from `qualify::promotion::GENERALIZATION_CLAIM`).
//! - When no pre-registration packet is supplied (`prereg_bytes: None`), the report sets
//!   `provenance_binding_status: "unverified_caller_asserted_inputs"` and includes
//!   `"provenance_and_prereg_binding_unimplemented"` in `disqualification_reasons`.
//! - When `prereg_bytes: Some(bytes)` is supplied (`qualify_rounds_with_prereg` or
//!   `jope-pools qualify-rounds --prereg <PATH>`), the verifier parses `bytes` through a strict
//!   recursive JSON visitor that rejects duplicate object keys at every depth before constructing
//!   `serde_json::Value`, checks `plan.prereg_sha256` against either raw `sha256_hex(bytes)` or
//!   `canonical_sha256(prereg)`, validates the `jope.selfplay-round1-prereg/v1` schema, cross-checks
//!   all pinned contract fields against `plan`, `identity`, `parent_of_context`, and the caller-supplied
//!   ratchet outputs (`seed_cap_b`, `rounds.len()`, `measured_tilt_policy`, `max_conditional_base_kl`,
//!   `max_new_support_mass`, and per-round `kl_source == "measured_tilt_leaf_policy"`), and checks
//!   that per-round `ordinary_search_sha256`, `ordinary_observations_sha256`,
//!   `ratchet_search_sha256`, and `ratchet_observations_sha256` are 64-character lowercase hex
//!   strings (`is_hex64` format check only). Because this offline verifier does not ingest or
//!   re-hash the underlying search/observation artifact files or authenticate the caller-supplied
//!   `execution_grant`, every `--prereg` report (both unpinned and pinned) appends
//!   `"prereg_receipts_and_execution_grant_caller_asserted_unverified"` and
//!   `"qualified_execution_mode_disabled_offline_verifier_only"` to `disqualification_reasons`.
//!   While any pre-registration field is `"UNPINNED"` or `execution_grant.authorized == false`, the
//!   report sets
//!   `provenance_binding_status: "prereg_contract_fields_checked_unpinned_no_execution_grant"` and
//!   also appends `"prereg_unpinned_or_no_execution_grant"`. When all pre-registration fields are
//!   syntactically pinned and `execution_grant.authorized == true`, the report sets
//!   `provenance_binding_status: "prereg_contract_fields_checked_receipts_and_grant_caller_asserted_unverified"`.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use anyhow::{bail, ensure, Context, Result};
use jope_pipeline::test_score::{ScoredDataset, TEST_SCORE_SCHEMA_ID};
use serde::de::{self, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;

use crate::canon::{canonical_sha256, is_hex64, sha256_hex};
use crate::clover::expansion::MeasuredTiltPolicyConfig;
use crate::clover::ExperimentIdentity;
use crate::pools::bootstrap::{cluster_bootstrap_ci, PairedInterval, Verdict};
use crate::qualify::promotion::{GENERALIZATION_CLAIM, MIN_RESAMPLES};
use crate::support::DecomposedSupportKl;

/// True if `s` is empty/whitespace or matches a case-insensitive placeholder sentinel
/// (`UNPINNED`, `TBD`, `TODO`, `PENDING`, `PLACEHOLDER`, `UNBOUND`).
fn is_placeholder_sentinel(s: &str) -> bool {
    let trimmed = s.trim();
    trimmed.is_empty()
        || matches!(
            trimmed.to_ascii_uppercase().as_str(),
            "UNPINNED" | "TBD" | "TODO" | "PENDING" | "PLACEHOLDER" | "UNBOUND"
        )
}

mod model;
mod qualifier;
#[cfg(test)]
mod tests;
mod validation;

pub use model::*;
pub use qualifier::{qualify_rounds, qualify_rounds_with_prereg};
pub use validation::parse_parent_map;
