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

/// Schema identifier for the offline self-play round qualification report.
pub const ROUND_QUAL_SCHEMA: &str = "jope.selfplay-round-qualification/v1";

/// Expected schema identifier for per-context self-play ratchet outputs.
pub const RATCHET_SCHEMA: &str = "clover.selfplay-ratchet/v1";

/// Expected schema identifier for the Round-1 pre-registration packet.
pub const PREREG_SCHEMA: &str = "jope.selfplay-round1-prereg/v1";

/// Expected `endpoints.secondary_status` disclosure string in `jope.selfplay-round1-prereg/v1`.
pub const PREREG_SECONDARY_STATUS: &str =
    "DIAGNOSTIC_VERIFIER_ONLY_PRODUCER_SCRATCH_UNEQUAL_RESOURCES_SINGLE_ENDPOINT_ALPHA_0_025";

/// F10 tie tolerance on log-RMSE ratios: `|ln(cand) - ln(ref)| <= 1e-12` is treated as zero (`0.0`).
pub const F10_TIE_TOLERANCE: f64 = 1e-12;

/// Maximum single-endpoint alpha permitted while joint Holm-Bonferroni multiplicity is unimplemented.
pub const MAX_SINGLE_ENDPOINT_ALPHA: f64 = 0.025;

/// Provenance binding disclosure string emitted when no `--prereg` packet is supplied.
pub const PROVENANCE_BINDING_STATUS: &str = "unverified_caller_asserted_inputs";

/// Provenance binding disclosure string emitted when `--prereg` matches `plan.prereg_sha256`
/// (raw or canonical) and all pinned contract fields match, but one or more fields remain
/// `"UNPINNED"` or `execution_grant.authorized == false`.
pub const PREREG_UNPINNED_BINDING_STATUS: &str =
    "prereg_contract_fields_checked_unpinned_no_execution_grant";

/// Provenance binding disclosure string emitted when `--prereg` matches `plan.prereg_sha256`
/// (raw or canonical), all pre-registration contract fields are syntactically pinned with
/// `execution_grant.authorized == true`, and per-round search/observation digest fields pass
/// 64-hex format checks, while underlying search/observation artifacts and the execution grant
/// remain caller-asserted and unverified by this offline tool.
pub const PREREG_PINNED_BINDING_STATUS: &str =
    "prereg_contract_fields_checked_receipts_and_grant_caller_asserted_unverified";

/// Disqualification reason appended on every `--prereg` report because underlying search/observation
/// artifacts and the execution grant are caller-asserted rather than externally verified.
pub const PREREG_CALLER_ASSERTED_REASON: &str =
    "prereg_receipts_and_execution_grant_caller_asserted_unverified";

/// Multiplicity adjustment disclosure string emitted on every report in this build.
pub const MULTIPLICITY_ADJUSTMENT_STATUS: &str = "unimplemented_single_endpoint_alpha_cap_0_025";

fn default_alpha() -> f64 {
    MAX_SINGLE_ENDPOINT_ALPHA
}

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

fn value_contains_placeholder_sentinel(v: &Value) -> Option<String> {
    match v {
        Value::String(s) => is_placeholder_sentinel(s).then(|| s.clone()),
        Value::Array(arr) => arr.iter().find_map(value_contains_placeholder_sentinel),
        Value::Object(map) => map.values().find_map(value_contains_placeholder_sentinel),
        _ => None,
    }
}

/// Recursive `serde_json::Value` wrapper whose `Deserialize` implementation rejects duplicate
/// object keys at every nesting depth (including inside arrays and nested maps) before
/// constructing a `serde_json::Map`.
struct StrictJsonValue(Value);

impl<'de> Deserialize<'de> for StrictJsonValue {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        struct StrictValueVisitor;

        impl<'de> Visitor<'de> for StrictValueVisitor {
            type Value = StrictJsonValue;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a valid JSON value without duplicate object keys")
            }

            fn visit_bool<E>(self, v: bool) -> std::result::Result<Self::Value, E> {
                Ok(StrictJsonValue(Value::Bool(v)))
            }

            fn visit_i64<E>(self, v: i64) -> std::result::Result<Self::Value, E> {
                Ok(StrictJsonValue(Value::Number(v.into())))
            }

            fn visit_u64<E>(self, v: u64) -> std::result::Result<Self::Value, E> {
                Ok(StrictJsonValue(Value::Number(v.into())))
            }

            fn visit_f64<E: de::Error>(self, v: f64) -> std::result::Result<Self::Value, E> {
                serde_json::Number::from_f64(v)
                    .map(|n| StrictJsonValue(Value::Number(n)))
                    .ok_or_else(|| de::Error::custom("non-finite float in JSON"))
            }

            fn visit_str<E>(self, v: &str) -> std::result::Result<Self::Value, E> {
                Ok(StrictJsonValue(Value::String(v.to_owned())))
            }

            fn visit_string<E>(self, v: String) -> std::result::Result<Self::Value, E> {
                Ok(StrictJsonValue(Value::String(v)))
            }

            fn visit_none<E>(self) -> std::result::Result<Self::Value, E> {
                Ok(StrictJsonValue(Value::Null))
            }

            fn visit_unit<E>(self) -> std::result::Result<Self::Value, E> {
                Ok(StrictJsonValue(Value::Null))
            }

            fn visit_some<D: Deserializer<'de>>(
                self,
                deserializer: D,
            ) -> std::result::Result<Self::Value, D::Error> {
                Deserialize::deserialize(deserializer)
            }

            fn visit_seq<A: SeqAccess<'de>>(
                self,
                mut seq: A,
            ) -> std::result::Result<Self::Value, A::Error> {
                let mut vec = Vec::new();
                while let Some(elem) = seq.next_element::<StrictJsonValue>()? {
                    vec.push(elem.0);
                }
                Ok(StrictJsonValue(Value::Array(vec)))
            }

            fn visit_map<M: MapAccess<'de>>(
                self,
                mut access: M,
            ) -> std::result::Result<Self::Value, M::Error> {
                let mut map = serde_json::Map::new();
                while let Some((k, v)) = access.next_entry::<String, StrictJsonValue>()? {
                    if map.contains_key(&k) {
                        return Err(de::Error::custom(format!("duplicate JSON object key: {k}")));
                    }
                    map.insert(k, v.0);
                }
                Ok(StrictJsonValue(Value::Object(map)))
            }
        }

        deserializer.deserialize_any(StrictValueVisitor)
    }
}

/// Parse a JSON document into [`serde_json::Value`], rejecting duplicate object keys at all
/// nesting depths and rejecting trailing characters after the root value.
pub fn parse_strict_json_value(bytes: &[u8], what: &str) -> Result<Value> {
    let mut de = serde_json::Deserializer::from_slice(bytes);
    let StrictJsonValue(val) = StrictJsonValue::deserialize(&mut de)
        .with_context(|| format!("{what}: invalid JSON or duplicate object key"))?;
    de.end()
        .with_context(|| format!("{what}: trailing characters after JSON value"))?;
    Ok(val)
}

/// Pre-registration field that is either explicitly marked `"UNPINNED"` or bound to a value `T`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum UnpinnedOr<T> {
    /// The `Unpinned` case of `UnpinnedOr`.
    Unpinned,
    /// The `Pinned` case of `UnpinnedOr`.
    Pinned(T),
}

impl<T> UnpinnedOr<T> {
    /// Returns whether unpinned.
    pub fn is_unpinned(&self) -> bool {
        matches!(self, Self::Unpinned)
    }

    /// Performs the as pinned operation and returns an error if it cannot be completed.
    pub fn as_pinned(&self) -> Option<&T> {
        match self {
            Self::Unpinned => None,
            Self::Pinned(v) => Some(v),
        }
    }
}

impl<T: Serialize> Serialize for UnpinnedOr<T> {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        match self {
            Self::Unpinned => serializer.serialize_str("UNPINNED"),
            Self::Pinned(v) => v.serialize(serializer),
        }
    }
}

impl<'de, T: serde::de::DeserializeOwned> Deserialize<'de> for UnpinnedOr<T> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        let StrictJsonValue(val) = StrictJsonValue::deserialize(deserializer)?;
        if val.as_str() == Some("UNPINNED") {
            return Ok(Self::Unpinned);
        }
        if let Some(bad) = value_contains_placeholder_sentinel(&val) {
            return Err(de::Error::custom(format!(
                "invalid placeholder or case/whitespace sentinel {bad:?}; use exact \"UNPINNED\" or a concrete pinned value"
            )));
        }
        serde_json::from_value(val)
            .map(Self::Pinned)
            .map_err(de::Error::custom)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// `PreregEndpoints` is a record for the prereg endpoints contract in this module: Offline self-play round qualification diagnostic report (jope.selfplay-round-qualification/v1).
pub struct PreregEndpoints {
    /// Primary recorded for this prereg endpoints.
    pub primary: Endpoint,
    /// Secondary recorded for this prereg endpoints.
    pub secondary: Endpoint,
    /// Secondary status recorded for this prereg endpoints.
    pub secondary_status: String,
    /// Multiplicity procedure recorded for this prereg endpoints.
    pub multiplicity_procedure: String,
    /// Cli single endpoint alpha cap recorded for this prereg endpoints.
    pub cli_single_endpoint_alpha_cap: f64,
    /// Alpha total recorded for this prereg endpoints.
    pub alpha_total: f64,
    /// Alpha primary strict recorded for this prereg endpoints.
    pub alpha_primary_strict: f64,
    /// F10 tie tolerance log ratio recorded for this prereg endpoints.
    pub f10_tie_tolerance_log_ratio: f64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// `PreregRosterAndFamilies` is a record for the prereg roster and families contract in this module: Offline self-play round qualification diagnostic report (jope.selfplay-round-qualification/v1).
pub struct PreregRosterAndFamilies {
    /// Unit of inference recorded for this prereg roster and families.
    pub unit_of_inference: String,
    /// Roster role recorded for this prereg roster and families.
    pub roster_role: String,
    /// SHA-256 digest identifying the associated sealed roster.
    pub sealed_roster_sha256: UnpinnedOr<String>,
    /// SHA-256 digest identifying the associated parent cluster map.
    pub parent_cluster_map_sha256: UnpinnedOr<String>,
    /// G23 sealed boundary ruling recorded for this prereg roster and families.
    pub g23_sealed_boundary_ruling: UnpinnedOr<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// `PreregSampleSizeAndMargin` is a record for the prereg sample size and margin contract in this module: Offline self-play round qualification diagnostic report (jope.selfplay-round-qualification/v1).
pub struct PreregSampleSizeAndMargin {
    /// Lower bound on parent clusters.
    pub min_parent_clusters: UnpinnedOr<usize>,
    /// Practical margin log ratio recorded for this prereg sample size and margin.
    pub practical_margin_log_ratio: UnpinnedOr<f64>,
    /// C158 g judge binding receipt recorded for this prereg sample size and margin.
    pub c158_g_judge_binding_receipt: UnpinnedOr<String>,
    /// Bootstrap resamples recorded for this prereg sample size and margin.
    pub bootstrap_resamples: u32,
    /// Bootstrap seed recorded for this prereg sample size and margin.
    pub bootstrap_seed: u64,
    /// Bootstrap method recorded for this prereg sample size and margin.
    pub bootstrap_method: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// `PreregKlGuard` is a record for the prereg KL guard contract in this module: Offline self-play round qualification diagnostic report (jope.selfplay-round-qualification/v1).
pub struct PreregKlGuard {
    /// Upper bound on conditional base KL.
    pub max_conditional_base_kl: f64,
    /// Upper bound on new support mass.
    pub max_new_support_mass: f64,
    /// Allow kl frozen rounds recorded for this prereg KL guard.
    pub allow_kl_frozen_rounds: bool,
    /// Collection of allowlisted measured sources values carried by this prereg KL guard.
    pub allowlisted_measured_sources: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// `PreregArmsAndRounds` is a record for the prereg arms and rounds contract in this module: Offline self-play round qualification diagnostic report (jope.selfplay-round-qualification/v1).
pub struct PreregArmsAndRounds {
    /// Rounds recorded for this prereg arms and rounds.
    pub rounds: usize,
    /// Seed cap b recorded for this prereg arms and rounds.
    pub seed_cap_b: usize,
    /// Equal attempts per round required recorded for this prereg arms and rounds.
    pub equal_attempts_per_round_required: bool,
    /// Equal resource comparison required recorded for this prereg arms and rounds.
    pub equal_resource_comparison_required: bool,
    /// Measured tilt policy recorded for this prereg arms and rounds.
    pub measured_tilt_policy: MeasuredTiltPolicyConfig,
    /// Kl guard recorded for this prereg arms and rounds.
    pub kl_guard: PreregKlGuard,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// `PreregSequentialLooks` is a record for the prereg sequential looks contract in this module: Offline self-play round qualification diagnostic report (jope.selfplay-round-qualification/v1).
pub struct PreregSequentialLooks {
    /// Collection of reference earlier looks values carried by this prereg sequential looks.
    pub reference_earlier_looks: Vec<String>,
    /// Collection of candidate earlier looks when distinct values carried by this prereg sequential looks.
    pub candidate_earlier_looks_when_distinct: Vec<String>,
    /// Reject omitted earlier looks field recorded for this prereg sequential looks.
    pub reject_omitted_earlier_looks_field: bool,
    /// Reject registry reset both empty when distinct recorded for this prereg sequential looks.
    pub reject_registry_reset_both_empty_when_distinct: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// `PreregHeldOutAndParityRules` is a record for the prereg held out and parity rules contract in this module: Offline self-play round qualification diagnostic report (jope.selfplay-round-qualification/v1).
pub struct PreregHeldOutAndParityRules {
    /// Test score schema recorded for this prereg held out and parity rules.
    pub test_score_schema: String,
    /// Pinned test score registry recorded for this prereg held out and parity rules.
    pub pinned_test_score_registry: UnpinnedOr<String>,
    /// Sequential same registry looks recorded for this prereg held out and parity rules.
    pub sequential_same_registry_looks: PreregSequentialLooks,
    /// Require feeds selection false recorded for this prereg held out and parity rules.
    pub require_feeds_selection_false: bool,
    /// R5 parity failure disposition recorded for this prereg held out and parity rules.
    pub r5_parity_failure_disposition: String,
    /// Reject unreferenced score or parity records recorded for this prereg held out and parity rules.
    pub reject_unreferenced_score_or_parity_records: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// `PreregExecutionGrant` is a record for the prereg execution grant contract in this module: Offline self-play round qualification diagnostic report (jope.selfplay-round-qualification/v1).
pub struct PreregExecutionGrant {
    /// Authorized recorded for this prereg execution grant.
    pub authorized: bool,
    /// Owner admission locator recorded for this prereg execution grant.
    pub owner_admission_locator: UnpinnedOr<String>,
    /// Hosts recorded for this prereg execution grant.
    pub hosts: UnpinnedOr<Vec<String>>,
    /// Cpu ceiling percent recorded for this prereg execution grant.
    pub cpu_ceiling_percent: UnpinnedOr<u32>,
    /// Memory floor free percent recorded for this prereg execution grant.
    pub memory_floor_free_percent: u32,
    /// Gpu seconds budget recorded for this prereg execution grant.
    pub gpu_seconds_budget: u64,
    /// Deadline utc recorded for this prereg execution grant.
    pub deadline_utc: UnpinnedOr<String>,
}

/// Pre-registration packet (`jope.selfplay-round1-prereg/v1`) for Round-1 self-play qualification.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SelfPlayPrereg {
    /// Schema identifier and version for the serialized value.
    pub schema: String,
    /// Identifier for the associated study.
    pub study_id: String,
    /// Outcome or lifecycle status recorded for this value.
    pub status: String,
    /// Qualification report schema recorded for this self play prereg.
    pub qualification_report_schema: String,
    /// Provenance binding status recorded for this self play prereg.
    pub provenance_binding_status: String,
    /// SHA-256 digest identifying the associated experiment identity.
    pub experiment_identity_sha256: UnpinnedOr<String>,
    /// Source commit at draft recorded for this self play prereg.
    pub source_commit_at_draft: String,
    /// Endpoints recorded for this self play prereg.
    pub endpoints: PreregEndpoints,
    /// Roster and families recorded for this self play prereg.
    pub roster_and_families: PreregRosterAndFamilies,
    /// Sample size and margin recorded for this self play prereg.
    pub sample_size_and_margin: PreregSampleSizeAndMargin,
    /// Arms and rounds recorded for this self play prereg.
    pub arms_and_rounds: PreregArmsAndRounds,
    /// Held out and parity rules recorded for this self play prereg.
    pub held_out_and_parity_rules: PreregHeldOutAndParityRules,
    /// Execution grant recorded for this self play prereg.
    pub execution_grant: PreregExecutionGrant,
}

impl SelfPlayPrereg {
    /// True if any required pre-registration field remains `"UNPINNED"`.
    pub fn has_unpinned_fields(&self) -> bool {
        self.experiment_identity_sha256.is_unpinned()
            || self.roster_and_families.sealed_roster_sha256.is_unpinned()
            || self
                .roster_and_families
                .parent_cluster_map_sha256
                .is_unpinned()
            || self
                .roster_and_families
                .g23_sealed_boundary_ruling
                .is_unpinned()
            || self
                .sample_size_and_margin
                .min_parent_clusters
                .is_unpinned()
            || self
                .sample_size_and_margin
                .practical_margin_log_ratio
                .is_unpinned()
            || self
                .sample_size_and_margin
                .c158_g_judge_binding_receipt
                .is_unpinned()
            || self
                .held_out_and_parity_rules
                .pinned_test_score_registry
                .is_unpinned()
            || self.execution_grant.owner_admission_locator.is_unpinned()
            || self.execution_grant.hosts.is_unpinned()
            || self.execution_grant.cpu_ceiling_percent.is_unpinned()
            || self.execution_grant.deadline_utc.is_unpinned()
    }
}

/// Caller-asserted provenance grade of the candidate and reference runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceGrade {
    /// Caller-asserted pre-registered confirmation input (diagnostic only until provenance binding lands).
    Qualified,
    /// Scratch discovery or R22/inner-cut exploration.
    NonQualifyingScratch,
    /// Reused development pools (e.g., `jope.j9-folds/v1`).
    ReusedDevelopment,
}

/// Paired comparison endpoint across rounds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Endpoint {
    /// Final ratchet round champion vs. Round-0 ratchet incumbent on the held-out test split.
    FinalVsRound0Incumbent,
    /// Final ratchet round champion vs. final ordinary-GP round champion (`ordinary_champion_genome`)
    /// on the held-out test split.
    RatchetVsOrdinaryFinalRound,
}

/// Pre-registered evaluation plan for [`qualify_rounds`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoundQualPlan {
    /// SHA-256 digest identifying the associated experiment identity.
    pub experiment_identity_sha256: String,
    /// SHA-256 digest identifying the associated prereg.
    pub prereg_sha256: String,
    /// Source grade recorded for this round qual plan.
    pub source_grade: SourceGrade,
    /// Pinned test-score registry path/identifier that all matched `QualTestScore` records must share.
    pub pinned_test_score_registry: String,
    /// Minimum required parent clusters (`>= 2`).
    pub min_parent_clusters: usize,
    /// Minimum required lower bound on parent-cluster bootstrap mean log-RMSE reduction
    /// `ln(test_rmse_reference) - ln(test_rmse_candidate)` (`>= 0.0`).
    pub practical_margin_log_ratio: f64,
    /// Bootstrap resamples recorded for this round qual plan.
    pub bootstrap_resamples: u32,
    /// Bootstrap seed recorded for this round qual plan.
    pub bootstrap_seed: u64,
    #[serde(default = "default_alpha")]
    /// Alpha recorded for this round qual plan.
    pub alpha: f64,
    /// Endpoint recorded for this round qual plan.
    pub endpoint: Endpoint,
    /// Allow kl frozen rounds recorded for this round qual plan.
    pub allow_kl_frozen_rounds: bool,
    /// Maximum admissible `KL(pi_theta(. | S_base) || pi_ref(. | S_base))` (`>= 0.0`, e.g. `0.20`).
    pub max_conditional_base_kl: f64,
    /// Maximum admissible probability mass allocated to newly admitted `N+1` leaves (`in [0, 1]`, e.g. `0.15`).
    pub max_new_support_mass: f64,
    /// Optional explicit list of expected context SHA-256 identifiers; when empty, the keys of
    /// `parent_of_context` define the expected context set.
    #[serde(default)]
    pub expected_contexts: Vec<String>,
}

/// Strict test-split score input record for [`qualify_rounds`].
///
/// Unlike [`jope_pipeline::test_score::TestScore`], which uses `#[serde(default)]` on `earlier_looks`
/// and silently turns an omitted legacy field into `[]`, [`QualTestScore`] deserializes
/// `earlier_looks` as `Option<Vec<String>>` so omitted or `null` history is detected and disqualified.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QualTestScore {
    /// Schema identifier and version for the serialized value.
    pub schema: String,
    /// Tool version recorded for this qual test score.
    pub tool_version: String,
    /// Timestamp at which the associated created occurred.
    pub created_at: String,
    /// Split recorded for this qual test score.
    pub split: String,
    /// SHA-256 digest of the receipt that authenticates this result.
    pub receipt_sha256: String,
    /// Dataset recorded for this qual test score.
    pub dataset: ScoredDataset,
    /// Canonical program or genome representation.
    pub genome: String,
    /// SHA-256 digest of the canonical program source.
    pub program_sha256: String,
    /// Test rmse recorded for this qual test score.
    pub test_rmse: f64,
    /// Test r2 recorded for this qual test score.
    pub test_r2: f64,
    /// Validation rmse recorded for this qual test score.
    pub validation_rmse: f64,
    /// Registry recorded for this qual test score.
    pub registry: String,
    #[serde(default)]
    /// Optional earlier looks value; `None` means it is not available.
    pub earlier_looks: Option<Vec<String>>,
    /// Feeds selection recorded for this qual test score.
    pub feeds_selection: bool,
    /// Seconds recorded for this qual test score.
    pub seconds: f64,
}

/// R5 cross-language / runtime parity verification record for one `(dataset_id, genome)` pair.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ParityRecord {
    /// Identifier for the associated dataset.
    pub dataset_id: String,
    /// Canonical program or genome representation.
    pub genome: String,
    /// Passed recorded for this parity record.
    pub passed: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Optional failure reason value; `None` means it is not available.
    pub failure_reason: Option<String>,
}

/// Borrowed inputs to [`qualify_rounds`] and [`qualify_rounds_with_prereg`].
pub struct RoundQualInputs<'a> {
    /// Plan recorded for this round qual inputs.
    pub plan: &'a RoundQualPlan,
    /// Identity recorded for this round qual inputs.
    pub identity: &'a ExperimentIdentity,
    /// Ratchet outputs recorded for this round qual inputs.
    pub ratchet_outputs: &'a [Value],
    /// Test scores recorded for this round qual inputs.
    pub test_scores: &'a [QualTestScore],
    /// Parity recorded for this round qual inputs.
    pub parity: &'a [ParityRecord],
    /// Parent of context recorded for this round qual inputs.
    pub parent_of_context: &'a BTreeMap<String, String>,
}

/// Per-context audit row inside [`RoundQualReport`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContextQualRow {
    /// Identifier for the associated context.
    pub context_id: String,
    /// Identifier for the associated parent cluster.
    pub parent_cluster_id: String,
    /// Rounds observed recorded for this context qual row.
    pub rounds_observed: usize,
    /// Optional candidate genome value; `None` means it is not available.
    pub candidate_genome: Option<String>,
    /// Optional reference genome value; `None` means it is not available.
    pub reference_genome: Option<String>,
    /// Optional candidate test RMSE value; `None` means it is not available.
    pub candidate_test_rmse: Option<f64>,
    /// Optional reference test RMSE value; `None` means it is not available.
    pub reference_test_rmse: Option<f64>,
    /// `ln(test_rmse_candidate) - ln(test_rmse_reference)` (lower is better; `< 0` means candidate beat reference).
    pub paired_log_ratio: Option<f64>,
    /// `ln(test_rmse_reference) - ln(test_rmse_candidate)` (higher is better; `> 0` means candidate beat reference).
    pub paired_log_reduction: Option<f64>,
    /// Optional tie within f10 tolerance value; `None` means it is not available.
    pub tie_within_f10_tolerance: Option<bool>,
    /// Optional candidate parity passed value; `None` means it is not available.
    pub candidate_parity_passed: Option<bool>,
    /// Optional reference parity passed value; `None` means it is not available.
    pub reference_parity_passed: Option<bool>,
    /// Registry matches plan recorded for this context qual row.
    pub registry_matches_plan: bool,
    /// Earlier looks clean recorded for this context qual row.
    pub earlier_looks_clean: bool,
    /// Feeds selection clean recorded for this context qual row.
    pub feeds_selection_clean: bool,
    /// Equal attempts all rounds recorded for this context qual row.
    pub equal_attempts_all_rounds: bool,
    /// Equal resource comparison recorded for this context qual row.
    pub equal_resource_comparison: bool,
    /// Kl guard evaluated recorded for this context qual row.
    pub kl_guard_evaluated: bool,
    /// Any round kl frozen recorded for this context qual row.
    pub any_round_kl_frozen: bool,
    /// Included in paired interval recorded for this context qual row.
    pub included_in_paired_interval: bool,
    /// Collection of exclusion reasons values carried by this context qual row.
    pub exclusion_reasons: Vec<String>,
}

/// Output document of [`qualify_rounds`] (`jope.selfplay-round-qualification/v1`).
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RoundQualReport {
    /// Schema identifier and version for the serialized value.
    pub schema: String,
    /// SHA-256 digest identifying the associated experiment identity.
    pub experiment_identity_sha256: String,
    /// SHA-256 digest identifying the associated prereg.
    pub prereg_sha256: String,
    /// Provenance binding status recorded for this round qual report.
    pub provenance_binding_status: String,
    /// Multiplicity adjustment recorded for this round qual report.
    pub multiplicity_adjustment: String,
    /// Source grade recorded for this round qual report.
    pub source_grade: SourceGrade,
    /// Pinned test score registry recorded for this round qual report.
    pub pinned_test_score_registry: String,
    /// Endpoint recorded for this round qual report.
    pub endpoint: Endpoint,
    /// Alpha recorded for this round qual report.
    pub alpha: f64,
    /// Lower bound on parent clusters.
    pub min_parent_clusters: usize,
    /// Practical margin log ratio recorded for this round qual report.
    pub practical_margin_log_ratio: f64,
    /// Allow kl frozen rounds recorded for this round qual report.
    pub allow_kl_frozen_rounds: bool,
    /// Upper bound on conditional base KL.
    pub max_conditional_base_kl: f64,
    /// Upper bound on new support mass.
    pub max_new_support_mass: f64,
    /// N ratchet contexts recorded for this round qual report.
    pub n_ratchet_contexts: usize,
    /// N missing expected contexts recorded for this round qual report.
    pub n_missing_expected_contexts: usize,
    /// N complete paired contexts recorded for this round qual report.
    pub n_complete_paired_contexts: usize,
    /// N observed parent clusters recorded for this round qual report.
    pub n_observed_parent_clusters: usize,
    /// N insufficient rounds contexts recorded for this round qual report.
    pub n_insufficient_rounds_contexts: usize,
    /// N missing test scores recorded for this round qual report.
    pub n_missing_test_scores: usize,
    /// N registry mismatch violations recorded for this round qual report.
    pub n_registry_mismatch_violations: usize,
    /// N earlier looks violations recorded for this round qual report.
    pub n_earlier_looks_violations: usize,
    /// N feeds selection violations recorded for this round qual report.
    pub n_feeds_selection_violations: usize,
    /// N missing parity recorded for this round qual report.
    pub n_missing_parity: usize,
    /// N failed parity recorded for this round qual report.
    pub n_failed_parity: usize,
    /// N unreferenced test scores recorded for this round qual report.
    pub n_unreferenced_test_scores: usize,
    /// N unreferenced parity records recorded for this round qual report.
    pub n_unreferenced_parity_records: usize,
    /// N unequal attempt rounds recorded for this round qual report.
    pub n_unequal_attempt_rounds: usize,
    /// N unequal resource contexts recorded for this round qual report.
    pub n_unequal_resource_contexts: usize,
    /// N unmeasured kl contexts recorded for this round qual report.
    pub n_unmeasured_kl_contexts: usize,
    /// N kl frozen contexts recorded for this round qual report.
    pub n_kl_frozen_contexts: usize,
    /// Policy optimizer updates recorded for this round qual report.
    pub policy_optimizer_updates: u64,
    /// Paired interval over `ln(test_rmse_candidate) - ln(test_rmse_reference)` (lower is better).
    pub paired_log_ratio: Option<PairedInterval>,
    /// Paired interval over `ln(test_rmse_reference) - ln(test_rmse_candidate)` (higher is better).
    pub paired_log_reduction: Option<PairedInterval>,
    /// Verdict recorded for this round qual report.
    pub verdict: Verdict,
    /// Margin cleared recorded for this round qual report.
    pub margin_cleared: bool,
    /// Sealed family boundary adopted recorded for this round qual report.
    pub sealed_family_boundary_adopted: bool,
    /// True only when all caller-asserted diagnostic checks (statistical margin, parent clusters,
    /// sequential same-registry looks, parity, equal attempts/resources, and measured KL guard)
    /// passed on the supplied inputs. Does not establish provenance or a qualified claim.
    pub caller_asserted_diagnostic_checks_passed: bool,
    /// Qualified claim recorded for this round qual report.
    pub qualified_claim: bool,
    /// Promotion recorded for this round qual report.
    pub promotion: bool,
    /// L6 claim recorded for this round qual report.
    pub l6_claim: bool,
    /// Generalization claim recorded for this round qual report.
    pub generalization_claim: String,
    /// Collection of disqualification reasons values carried by this round qual report.
    pub disqualification_reasons: Vec<String>,
    /// Collection of contexts values carried by this round qual report.
    pub contexts: Vec<ContextQualRow>,
}

impl RoundQualReport {
    /// Canonical SHA-256 of this report without float re-formatting drift.
    pub fn canonical_sha256(&self) -> Result<String> {
        canonical_sha256(self)
    }
}

/// Parse a context-to-parent-cluster map from JSON (`{"ctx": "parent"}`) or 2-column TSV
/// (`ctx<TAB>parent`), rejecting duplicate context keys in both formats and rejecting extra TSV
/// columns.
pub fn parse_parent_map(bytes: &[u8]) -> Result<BTreeMap<String, String>> {
    let text = std::str::from_utf8(bytes).context("parent_map: invalid UTF-8")?;
    let trimmed = text.trim();
    ensure!(!trimmed.is_empty(), "parent_map must not be empty");
    if trimmed.starts_with('{') {
        struct NoDupMapVisitor;
        impl<'de> Visitor<'de> for NoDupMapVisitor {
            type Value = BTreeMap<String, String>;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str(
                    "a JSON object mapping context_id to parent_cluster_id without duplicate keys",
                )
            }

            fn visit_map<M>(self, mut access: M) -> std::result::Result<Self::Value, M::Error>
            where
                M: MapAccess<'de>,
            {
                let mut map = BTreeMap::new();
                while let Some((k, v)) = access.next_entry::<String, String>()? {
                    let key = k.trim().to_owned();
                    let val = v.trim().to_owned();
                    if key.is_empty() || val.is_empty() {
                        return Err(de::Error::custom(
                            "parent_map JSON contains empty context_id or parent_cluster_id",
                        ));
                    }
                    if map.insert(key.clone(), val).is_some() {
                        return Err(de::Error::custom(format!(
                            "duplicate context key in parent_map JSON: {key}"
                        )));
                    }
                }
                Ok(map)
            }
        }
        let mut de = serde_json::Deserializer::from_str(trimmed);
        let map = serde::Deserializer::deserialize_map(&mut de, NoDupMapVisitor)
            .context("parent_map: JSON object")?;
        de.end()
            .context("parent_map: trailing characters after JSON object")?;
        ensure!(!map.is_empty(), "parent_map must not be empty");
        return Ok(map);
    }

    let mut map = BTreeMap::new();
    for (i, raw_line) in trimmed.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        ensure!(
            cols.len() == 2,
            "parent_map line {}: expected exactly 2 tab-separated columns (context_id<TAB>parent_id), got {}",
            i + 1,
            cols.len()
        );
        let ctx = cols[0].trim().to_owned();
        let parent = cols[1].trim().to_owned();
        ensure!(
            !ctx.is_empty() && !parent.is_empty(),
            "parent_map line {}: empty context_id or parent_id",
            i + 1
        );
        ensure!(
            map.insert(ctx.clone(), parent).is_none(),
            "parent_map line {}: duplicate context key {ctx}",
            i + 1
        );
    }
    ensure!(!map.is_empty(), "parent_map must not be empty");
    Ok(map)
}

fn validate_plan_and_identity(
    plan: &RoundQualPlan,
    identity: &ExperimentIdentity,
) -> Result<String> {
    let computed_id = identity
        .sha256()
        .context("invalid ExperimentIdentity for round qualification")?;
    ensure!(
        is_hex64(&plan.experiment_identity_sha256),
        "plan.experiment_identity_sha256 must be a 64-hex lowercase digest"
    );
    ensure!(
        computed_id == plan.experiment_identity_sha256,
        "experiment identity sha256 mismatch: identity={computed_id}, plan={}",
        plan.experiment_identity_sha256
    );
    ensure!(
        is_hex64(&plan.prereg_sha256),
        "plan.prereg_sha256 must be a 64-hex lowercase digest"
    );
    ensure!(
        !plan.pinned_test_score_registry.trim().is_empty(),
        "plan.pinned_test_score_registry must be non-empty"
    );
    ensure!(
        plan.min_parent_clusters >= 2,
        "plan.min_parent_clusters must be at least 2 for parent-cluster bootstrap inference"
    );
    ensure!(
        plan.practical_margin_log_ratio.is_finite() && plan.practical_margin_log_ratio >= 0.0,
        "plan.practical_margin_log_ratio must be finite and >= 0"
    );
    ensure!(
        plan.bootstrap_resamples >= MIN_RESAMPLES,
        "plan.bootstrap_resamples ({}) must be >= {MIN_RESAMPLES}",
        plan.bootstrap_resamples
    );
    ensure!(
        plan.alpha.is_finite() && plan.alpha > 0.0 && plan.alpha <= MAX_SINGLE_ENDPOINT_ALPHA,
        "plan.alpha ({}) must be finite and in (0, {MAX_SINGLE_ENDPOINT_ALPHA}] while joint Holm-Bonferroni adjustment is unimplemented",
        plan.alpha
    );
    ensure!(
        plan.max_conditional_base_kl.is_finite() && plan.max_conditional_base_kl >= 0.0,
        "plan.max_conditional_base_kl ({}) must be finite and >= 0",
        plan.max_conditional_base_kl
    );
    ensure!(
        plan.max_new_support_mass.is_finite() && (0.0..=1.0).contains(&plan.max_new_support_mass),
        "plan.max_new_support_mass ({}) must be finite and in [0, 1]",
        plan.max_new_support_mass
    );
    Ok(computed_id)
}

fn validate_and_bind_prereg(
    prereg_bytes: &[u8],
    plan: &RoundQualPlan,
    identity: &ExperimentIdentity,
    parent_of_context: &BTreeMap<String, String>,
) -> Result<(SelfPlayPrereg, bool)> {
    let raw_sha = sha256_hex(prereg_bytes);
    let prereg_val = parse_strict_json_value(prereg_bytes, "prereg")?;
    let canon_sha = canonical_sha256(&prereg_val).context("prereg: canonical sha256")?;
    ensure!(
        plan.prereg_sha256 == raw_sha || plan.prereg_sha256 == canon_sha,
        "prereg_sha256 mismatch: plan.prereg_sha256={} != sha256(prereg_bytes)={raw_sha} (canonical={canon_sha})",
        plan.prereg_sha256
    );

    let prereg: SelfPlayPrereg =
        serde_json::from_value(prereg_val).context("prereg: schema validation")?;
    ensure!(
        prereg.schema == PREREG_SCHEMA,
        "unsupported prereg schema {:?}; expected {PREREG_SCHEMA}",
        prereg.schema
    );
    ensure!(
        !is_placeholder_sentinel(&prereg.study_id) && !is_placeholder_sentinel(&prereg.status),
        "prereg study_id and status must be non-empty and not placeholder sentinels"
    );
    ensure!(
        prereg.qualification_report_schema == ROUND_QUAL_SCHEMA,
        "prereg qualification_report_schema {:?} != {ROUND_QUAL_SCHEMA}",
        prereg.qualification_report_schema
    );
    ensure!(
        prereg.source_commit_at_draft.len() == 40
            && prereg
                .source_commit_at_draft
                .bytes()
                .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()),
        "prereg source_commit_at_draft must be a 40-character lowercase hex commit sha"
    );

    // 1. Endpoints and multiplicity invariants
    ensure!(
        prereg.endpoints.primary == Endpoint::FinalVsRound0Incumbent
            && prereg.endpoints.secondary == Endpoint::RatchetVsOrdinaryFinalRound,
        "prereg endpoints must bind primary=final_vs_round0_incumbent and secondary=ratchet_vs_ordinary_final_round"
    );
    ensure!(
        prereg.endpoints.secondary_status == PREREG_SECONDARY_STATUS,
        "prereg secondary_status {:?} != {PREREG_SECONDARY_STATUS}",
        prereg.endpoints.secondary_status
    );
    ensure!(
        prereg.endpoints.cli_single_endpoint_alpha_cap.is_finite()
            && (prereg.endpoints.cli_single_endpoint_alpha_cap - MAX_SINGLE_ENDPOINT_ALPHA).abs()
                <= 1e-15,
        "prereg cli_single_endpoint_alpha_cap ({}) != {MAX_SINGLE_ENDPOINT_ALPHA}",
        prereg.endpoints.cli_single_endpoint_alpha_cap
    );
    ensure!(
        prereg.endpoints.alpha_primary_strict.is_finite()
            && (prereg.endpoints.alpha_primary_strict - MAX_SINGLE_ENDPOINT_ALPHA).abs() <= 1e-15,
        "prereg alpha_primary_strict ({}) != {MAX_SINGLE_ENDPOINT_ALPHA}",
        prereg.endpoints.alpha_primary_strict
    );
    ensure!(
        prereg.endpoints.f10_tie_tolerance_log_ratio.is_finite()
            && (prereg.endpoints.f10_tie_tolerance_log_ratio - F10_TIE_TOLERANCE).abs() <= 1e-18,
        "prereg f10_tie_tolerance_log_ratio ({}) != {F10_TIE_TOLERANCE}",
        prereg.endpoints.f10_tie_tolerance_log_ratio
    );
    ensure!(
        plan.alpha <= prereg.endpoints.cli_single_endpoint_alpha_cap + 1e-15,
        "plan.alpha ({}) exceeds prereg cli_single_endpoint_alpha_cap ({})",
        plan.alpha,
        prereg.endpoints.cli_single_endpoint_alpha_cap
    );

    // 2. Roster and parent-cluster invariants
    ensure!(
        prereg.roster_and_families.unit_of_inference == "parent_dataset_cluster",
        "prereg unit_of_inference must be \"parent_dataset_cluster\""
    );
    ensure!(
        prereg.roster_and_families.roster_role == "confirmation_family_held_out",
        "prereg roster_role must be \"confirmation_family_held_out\""
    );
    if let Some(roster_sha) = prereg.roster_and_families.sealed_roster_sha256.as_pinned() {
        ensure!(
            is_hex64(roster_sha) && roster_sha == &identity.data_roster_sha256,
            "prereg sealed_roster_sha256 ({roster_sha}) != identity.data_roster_sha256 ({})",
            identity.data_roster_sha256
        );
    }
    if let Some(parent_sha) = prereg
        .roster_and_families
        .parent_cluster_map_sha256
        .as_pinned()
    {
        let expected_parent_sha = canonical_sha256(parent_of_context)?;
        ensure!(
            is_hex64(parent_sha) && parent_sha == &expected_parent_sha,
            "prereg parent_cluster_map_sha256 ({parent_sha}) != canonical_sha256(parent_of_context) ({expected_parent_sha})"
        );
    }
    if let Some(ruling) = prereg
        .roster_and_families
        .g23_sealed_boundary_ruling
        .as_pinned()
    {
        ensure!(
            !is_placeholder_sentinel(ruling),
            "prereg g23_sealed_boundary_ruling must be non-empty and not a placeholder when pinned"
        );
    }

    // 3. Sample size, margin, and bootstrap invariants
    ensure!(
        prereg.sample_size_and_margin.bootstrap_resamples == plan.bootstrap_resamples,
        "prereg bootstrap_resamples ({}) != plan.bootstrap_resamples ({})",
        prereg.sample_size_and_margin.bootstrap_resamples,
        plan.bootstrap_resamples
    );
    ensure!(
        prereg.sample_size_and_margin.bootstrap_seed == plan.bootstrap_seed,
        "prereg bootstrap_seed ({}) != plan.bootstrap_seed ({})",
        prereg.sample_size_and_margin.bootstrap_seed,
        plan.bootstrap_seed
    );
    ensure!(
        prereg.sample_size_and_margin.bootstrap_method == "parent_cluster_percentile_splitmix64",
        "prereg bootstrap_method must be \"parent_cluster_percentile_splitmix64\""
    );
    if let Some(&min_clusters) = prereg
        .sample_size_and_margin
        .min_parent_clusters
        .as_pinned()
    {
        ensure!(
            min_clusters >= 2 && min_clusters == plan.min_parent_clusters,
            "prereg min_parent_clusters ({min_clusters}) != plan.min_parent_clusters ({})",
            plan.min_parent_clusters
        );
    }
    if let Some(&margin) = prereg
        .sample_size_and_margin
        .practical_margin_log_ratio
        .as_pinned()
    {
        ensure!(
            margin.is_finite()
                && margin >= 0.0
                && (margin - plan.practical_margin_log_ratio).abs() <= 1e-12,
            "prereg practical_margin_log_ratio ({margin}) != plan.practical_margin_log_ratio ({})",
            plan.practical_margin_log_ratio
        );
    }
    if let Some(c158) = prereg
        .sample_size_and_margin
        .c158_g_judge_binding_receipt
        .as_pinned()
    {
        ensure!(
            !is_placeholder_sentinel(c158),
            "prereg c158_g_judge_binding_receipt must be non-empty and not a placeholder when pinned"
        );
    }

    // 4. Arms, rounds, measured-tilt policy, and KL guard invariants
    ensure!(
        prereg.arms_and_rounds.rounds >= 2,
        "prereg arms_and_rounds.rounds ({}) must be >= 2",
        prereg.arms_and_rounds.rounds
    );
    ensure!(
        prereg.arms_and_rounds.seed_cap_b >= 1,
        "prereg arms_and_rounds.seed_cap_b ({}) must be >= 1",
        prereg.arms_and_rounds.seed_cap_b
    );
    ensure!(
        prereg.arms_and_rounds.equal_attempts_per_round_required
            && prereg.arms_and_rounds.equal_resource_comparison_required,
        "prereg equal_attempts_per_round_required and equal_resource_comparison_required must be true"
    );
    prereg
        .arms_and_rounds
        .measured_tilt_policy
        .validate()
        .context("prereg measured_tilt_policy invalid")?;
    ensure!(
        (prereg.arms_and_rounds.kl_guard.max_conditional_base_kl - plan.max_conditional_base_kl)
            .abs()
            <= 1e-12,
        "prereg max_conditional_base_kl ({}) != plan.max_conditional_base_kl ({})",
        prereg.arms_and_rounds.kl_guard.max_conditional_base_kl,
        plan.max_conditional_base_kl
    );
    ensure!(
        (prereg.arms_and_rounds.kl_guard.max_new_support_mass - plan.max_new_support_mass).abs()
            <= 1e-12,
        "prereg max_new_support_mass ({}) != plan.max_new_support_mass ({})",
        prereg.arms_and_rounds.kl_guard.max_new_support_mass,
        plan.max_new_support_mass
    );
    ensure!(
        prereg.arms_and_rounds.kl_guard.allow_kl_frozen_rounds == plan.allow_kl_frozen_rounds,
        "prereg allow_kl_frozen_rounds ({}) != plan.allow_kl_frozen_rounds ({})",
        prereg.arms_and_rounds.kl_guard.allow_kl_frozen_rounds,
        plan.allow_kl_frozen_rounds
    );
    ensure!(
        prereg.arms_and_rounds.kl_guard.allowlisted_measured_sources
            == vec!["measured_tilt_leaf_policy".to_owned()],
        "prereg allowlisted_measured_sources must match [\"measured_tilt_leaf_policy\"]"
    );

    // 5. Held-out test-score and parity rules
    ensure!(
        prereg.held_out_and_parity_rules.test_score_schema == TEST_SCORE_SCHEMA_ID,
        "prereg test_score_schema {:?} != {TEST_SCORE_SCHEMA_ID}",
        prereg.held_out_and_parity_rules.test_score_schema
    );
    if let Some(reg) = prereg
        .held_out_and_parity_rules
        .pinned_test_score_registry
        .as_pinned()
    {
        ensure!(
            !is_placeholder_sentinel(reg) && reg == &plan.pinned_test_score_registry,
            "prereg pinned_test_score_registry ({reg}) != plan.pinned_test_score_registry ({})",
            plan.pinned_test_score_registry
        );
    }
    let looks = &prereg
        .held_out_and_parity_rules
        .sequential_same_registry_looks;
    ensure!(
        looks.reference_earlier_looks.is_empty()
            && looks.candidate_earlier_looks_when_distinct
                == vec!["<reference.program_sha256>".to_owned()]
            && looks.reject_omitted_earlier_looks_field
            && looks.reject_registry_reset_both_empty_when_distinct,
        "prereg sequential_same_registry_looks contract does not match B1 verifier rules"
    );
    ensure!(
        prereg
            .held_out_and_parity_rules
            .require_feeds_selection_false
            && prereg
                .held_out_and_parity_rules
                .r5_parity_failure_disposition
                == "charge_as_program_failure_never_drop"
            && prereg
                .held_out_and_parity_rules
                .reject_unreferenced_score_or_parity_records,
        "prereg held_out_and_parity_rules contract does not match R5/M5 verifier rules"
    );

    // 6. Experiment identity & execution grant admission gate
    if let Some(id_sha) = prereg.experiment_identity_sha256.as_pinned() {
        ensure!(
            is_hex64(id_sha) && id_sha == &plan.experiment_identity_sha256,
            "prereg experiment_identity_sha256 ({id_sha}) != plan.experiment_identity_sha256 ({})",
            plan.experiment_identity_sha256
        );
    }
    if let Some(loc) = prereg.execution_grant.owner_admission_locator.as_pinned() {
        ensure!(
            !is_placeholder_sentinel(loc),
            "prereg owner_admission_locator must be non-empty and not a placeholder when pinned"
        );
    }
    if let Some(hosts) = prereg.execution_grant.hosts.as_pinned() {
        ensure!(
            !hosts.is_empty() && hosts.iter().all(|h| !is_placeholder_sentinel(h)),
            "prereg execution_grant.hosts must be non-empty and contain no placeholders when pinned"
        );
    }
    if let Some(&cpu_cap) = prereg.execution_grant.cpu_ceiling_percent.as_pinned() {
        ensure!(
            (1..=100).contains(&cpu_cap),
            "prereg execution_grant.cpu_ceiling_percent ({cpu_cap}) must be in 1..=100"
        );
    }
    ensure!(
        (1..=100).contains(&prereg.execution_grant.memory_floor_free_percent),
        "prereg execution_grant.memory_floor_free_percent ({}) must be in 1..=100",
        prereg.execution_grant.memory_floor_free_percent
    );
    if let Some(deadline) = prereg.execution_grant.deadline_utc.as_pinned() {
        ensure!(
            !is_placeholder_sentinel(deadline),
            "prereg execution_grant.deadline_utc must be non-empty and not a placeholder when pinned"
        );
    }

    ensure!(
        !is_placeholder_sentinel(&prereg.study_id),
        "prereg study_id must be non-empty and not a placeholder"
    );
    let has_unpinned = prereg.has_unpinned_fields();
    if prereg.execution_grant.authorized {
        ensure!(
            !has_unpinned,
            "prereg execution_grant.authorized is true while one or more required prereg fields remain UNPINNED"
        );
        let status_upper = prereg.status.to_ascii_uppercase();
        ensure!(
            !is_placeholder_sentinel(&prereg.status)
                && !status_upper.contains("UNPINNED")
                && !status_upper.contains("NO_EXECUTION_GRANT"),
            "prereg execution_grant.authorized is true while status {:?} indicates unpinned or no execution grant",
            prereg.status
        );
        ensure!(
            prereg.provenance_binding_status == PREREG_PINNED_BINDING_STATUS,
            "prereg provenance_binding_status {:?} != {PREREG_PINNED_BINDING_STATUS} when authorized",
            prereg.provenance_binding_status
        );
    } else {
        ensure!(
            prereg.status == "PRE_REGISTERED_UNPINNED_NO_EXECUTION_GRANT",
            "prereg status {:?} must be PRE_REGISTERED_UNPINNED_NO_EXECUTION_GRANT when execution_grant.authorized is false",
            prereg.status
        );
        ensure!(
            prereg.provenance_binding_status == PREREG_UNPINNED_BINDING_STATUS,
            "prereg provenance_binding_status {:?} != {PREREG_UNPINNED_BINDING_STATUS} when execution_grant.authorized is false",
            prereg.provenance_binding_status
        );
    }
    let fully_pinned_and_authorized = !has_unpinned && prereg.execution_grant.authorized;
    Ok((prereg, fully_pinned_and_authorized))
}

fn ensure_prereg_round_provenance(
    context_id: &str,
    raw_ratchet: &Value,
    rounds: &[Value],
    prereg: &SelfPlayPrereg,
) -> Result<()> {
    let observed_seed_cap = raw_ratchet
        .get("seed_cap_b")
        .and_then(Value::as_u64)
        .with_context(|| {
            format!("context {context_id} missing integer 'seed_cap_b' required by prereg binding")
        })?;
    ensure!(
        observed_seed_cap == prereg.arms_and_rounds.seed_cap_b as u64,
        "context {context_id} seed_cap_b ({observed_seed_cap}) != prereg arms_and_rounds.seed_cap_b ({})",
        prereg.arms_and_rounds.seed_cap_b
    );
    let observed_max_base_kl = raw_ratchet
        .get("max_conditional_base_kl")
        .and_then(Value::as_f64)
        .filter(|v| v.is_finite())
        .with_context(|| {
            format!(
                "context {context_id} missing finite 'max_conditional_base_kl' required by prereg binding"
            )
        })?;
    ensure!(
        (observed_max_base_kl - prereg.arms_and_rounds.kl_guard.max_conditional_base_kl).abs()
            <= 1e-12,
        "context {context_id} max_conditional_base_kl ({observed_max_base_kl}) != prereg max_conditional_base_kl ({})",
        prereg.arms_and_rounds.kl_guard.max_conditional_base_kl
    );
    let observed_max_new_mass = raw_ratchet
        .get("max_new_support_mass")
        .and_then(Value::as_f64)
        .filter(|v| v.is_finite())
        .with_context(|| {
            format!(
                "context {context_id} missing finite 'max_new_support_mass' required by prereg binding"
            )
        })?;
    ensure!(
        (observed_max_new_mass - prereg.arms_and_rounds.kl_guard.max_new_support_mass).abs()
            <= 1e-12,
        "context {context_id} max_new_support_mass ({observed_max_new_mass}) != prereg max_new_support_mass ({})",
        prereg.arms_and_rounds.kl_guard.max_new_support_mass
    );
    let policy_val = raw_ratchet
        .get("measured_tilt_policy")
        .filter(|v| !v.is_null())
        .with_context(|| {
            format!(
                "context {context_id} missing 'measured_tilt_policy' object required by prereg binding"
            )
        })?;
    let observed_policy: MeasuredTiltPolicyConfig = serde_json::from_value(policy_val.clone())
        .with_context(|| format!("context {context_id} invalid 'measured_tilt_policy'"))?;
    observed_policy
        .validate()
        .with_context(|| format!("context {context_id} invalid 'measured_tilt_policy'"))?;
    let expected_policy = &prereg.arms_and_rounds.measured_tilt_policy;
    ensure!(
        (observed_policy.temperature_tau - expected_policy.temperature_tau).abs() <= 1e-12
            && (observed_policy.advantage_clip - expected_policy.advantage_clip).abs() <= 1e-12
            && (observed_policy.ref_new_support_mass - expected_policy.ref_new_support_mass).abs()
                <= 1e-12
            && (observed_policy.policy_new_support_cap - expected_policy.policy_new_support_cap)
                .abs()
                <= 1e-12
            && (observed_policy.step_size_eta - expected_policy.step_size_eta).abs() <= 1e-12,
        "context {context_id} measured_tilt_policy ({observed_policy:?}) != prereg measured_tilt_policy ({expected_policy:?})"
    );
    ensure!(
        rounds.len() == prereg.arms_and_rounds.rounds,
        "context {context_id} rounds.len() ({}) != prereg arms_and_rounds.rounds ({})",
        rounds.len(),
        prereg.arms_and_rounds.rounds
    );
    for (round_idx, round) in rounds.iter().enumerate() {
        let kl_source = round.get("kl_source").and_then(Value::as_str);
        ensure!(
            kl_source == Some("measured_tilt_leaf_policy"),
            "context {context_id} round {round_idx} kl_source {:?} != \"measured_tilt_leaf_policy\" required by prereg binding",
            kl_source
        );
        for genome_field in ["ratchet_champion_genome", "ordinary_champion_genome"] {
            let valid_str = round
                .get(genome_field)
                .and_then(Value::as_str)
                .is_some_and(|s| !is_placeholder_sentinel(s));
            ensure!(
                valid_str,
                "context {context_id} round {round_idx} missing non-empty string '{genome_field}' required by prereg binding"
            );
        }
        for sha_field in [
            "ordinary_search_sha256",
            "ordinary_observations_sha256",
            "ratchet_search_sha256",
            "ratchet_observations_sha256",
        ] {
            let valid_sha = round
                .get(sha_field)
                .and_then(Value::as_str)
                .is_some_and(is_hex64);
            ensure!(
                valid_sha,
                "context {context_id} round {round_idx} field '{sha_field}' must be a 64-hex lowercase digest"
            );
        }
    }
    Ok(())
}

fn find_unique_test_score<'a>(
    scores: &'a [QualTestScore],
    context_id: &str,
    genome: &str,
    used_indices: &mut BTreeSet<usize>,
) -> Result<Option<&'a QualTestScore>> {
    let mut matches = Vec::new();
    for (idx, ts) in scores.iter().enumerate() {
        if (ts.dataset.id == context_id || ts.dataset.dataset_hash == context_id)
            && ts.genome == genome
        {
            matches.push((idx, ts));
        }
    }
    ensure!(
        matches.len() <= 1,
        "ambiguous test score matches ({}) for context {context_id} genome {genome}",
        matches.len()
    );
    if let Some((idx, ts)) = matches.into_iter().next() {
        used_indices.insert(idx);
        Ok(Some(ts))
    } else {
        Ok(None)
    }
}

fn find_unique_parity<'a>(
    parity: &'a [ParityRecord],
    context_id: &str,
    genome: &str,
    used_indices: &mut BTreeSet<usize>,
) -> Result<Option<&'a ParityRecord>> {
    let mut matches = Vec::new();
    for (idx, pr) in parity.iter().enumerate() {
        if pr.dataset_id == context_id && pr.genome == genome {
            matches.push((idx, pr));
        }
    }
    ensure!(
        matches.len() <= 1,
        "ambiguous parity record matches ({}) for context {context_id} genome {genome}",
        matches.len()
    );
    if let Some((idx, pr)) = matches.into_iter().next() {
        used_indices.insert(idx);
        Ok(Some(pr))
    } else {
        Ok(None)
    }
}

fn ensure_valid_round_structure(context_id: &str, round_idx: usize, round: &Value) -> Result<()> {
    let observed_round = round
        .get("round")
        .and_then(Value::as_u64)
        .with_context(|| {
            format!("context {context_id} round[{round_idx}] missing integer 'round'")
        })?;
    ensure!(
        observed_round == round_idx as u64,
        "context {context_id} round[{round_idx}] has out-of-order round={observed_round}; expected {round_idx}"
    );
    let cum_prior = round
        .get("cumulative_prior_rounds")
        .and_then(Value::as_u64)
        .with_context(|| {
            format!(
                "context {context_id} round[{round_idx}] missing integer 'cumulative_prior_rounds'"
            )
        })?;
    ensure!(
        cum_prior == round_idx as u64,
        "context {context_id} round[{round_idx}] has cumulative_prior_rounds={cum_prior}; expected {round_idx}"
    );

    for field in [
        "ordinary_best_rmse",
        "ordinary_best_q",
        "ratchet_best_rmse",
        "ratchet_best_q",
        "surplus_q_vs_ordinary",
    ] {
        let Some(val) = round.get(field) else {
            bail!("context {context_id} round {round_idx} missing numeric field {field}");
        };
        let Some(num) = val.as_f64() else {
            bail!("context {context_id} round {round_idx} field {field} is not a finite number");
        };
        ensure!(
            num.is_finite(),
            "context {context_id} round {round_idx} field {field} is non-finite"
        );
    }
    if let Some(ord_genome) = round.get("ordinary_champion_genome") {
        ensure!(
            ord_genome
                .as_str()
                .is_some_and(|s| !is_placeholder_sentinel(s)),
            "context {context_id} round {round_idx} field ordinary_champion_genome must be a non-empty string"
        );
    }
    Ok(())
}

/// Validate a single round's KL guard fields fail-closed.
/// Returns `(kl_measured_and_valid, frozen_by_kl_guard)`.
fn evaluate_round_kl_guard(
    round: &Value,
    max_conditional_base_kl: f64,
    max_new_support_mass: f64,
) -> (bool, bool) {
    let Some(kl_source) = round.get("kl_source").and_then(Value::as_str) else {
        return (false, false);
    };
    if !matches!(
        kl_source,
        "measured_tilt_leaf_policy" | "caller_evaluated_leaf_log_probs"
    ) {
        let frozen = round
            .get("seed_update_frozen_by_kl_guard")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        return (false, frozen);
    }
    if !round.get("kl_malformed_reason").is_none_or(Value::is_null) {
        return (false, true);
    }
    let Some(decomposed_val) = round.get("decomposed_kl") else {
        return (false, false);
    };
    let Ok(decomposed) = serde_json::from_value::<DecomposedSupportKl>(decomposed_val.clone())
    else {
        return (false, false);
    };
    let numbers_valid = [
        decomposed.conditional_base_kl,
        decomposed.conditional_new_kl,
        decomposed.base_support_mass,
        decomposed.new_support_mass,
        decomposed.ref_new_support_mass,
        decomposed.bernoulli_allocation_kl,
        decomposed.total_kl,
    ]
    .into_iter()
    .all(|x| x.is_finite() && x >= 0.0);
    let masses_valid = [
        decomposed.base_support_mass,
        decomposed.new_support_mass,
        decomposed.ref_new_support_mass,
    ]
    .into_iter()
    .all(|m| m <= 1.0 + 1e-12)
        && (decomposed.base_support_mass + decomposed.new_support_mass - 1.0).abs() <= 1e-9;
    let expected_total_kl = decomposed.bernoulli_allocation_kl
        + decomposed.base_support_mass * decomposed.conditional_base_kl
        + decomposed.new_support_mass * decomposed.conditional_new_kl;
    let chain_rule_valid =
        expected_total_kl.is_finite() && (decomposed.total_kl - expected_total_kl).abs() <= 1e-9;
    if !numbers_valid || !masses_valid || !chain_rule_valid {
        return (false, false);
    }
    let expected_admissible = decomposed.conditional_base_kl <= max_conditional_base_kl + 1e-12
        && decomposed.new_support_mass <= max_new_support_mass + 1e-12;
    if decomposed.admissible != expected_admissible {
        return (false, false);
    }
    let Some(frozen) = round
        .get("seed_update_frozen_by_kl_guard")
        .and_then(Value::as_bool)
    else {
        return (false, false);
    };
    if frozen != !expected_admissible {
        return (false, frozen);
    }
    (true, frozen)
}

/// Evaluate multi-round self-play outputs against test-split scores, parity records, parent
/// lineage clusters, and pre-registered diagnostic checks without an external pre-registration packet.
pub fn qualify_rounds(inputs: RoundQualInputs<'_>) -> Result<RoundQualReport> {
    qualify_rounds_with_prereg(inputs, None)
}

/// Evaluate multi-round self-play outputs against test-split scores, parity records, parent
/// lineage clusters, and an optional pre-registration packet (`jope.selfplay-round1-prereg/v1`).
pub fn qualify_rounds_with_prereg(
    inputs: RoundQualInputs<'_>,
    prereg_bytes: Option<&[u8]>,
) -> Result<RoundQualReport> {
    let RoundQualInputs {
        plan,
        identity,
        ratchet_outputs,
        test_scores,
        parity,
        parent_of_context,
    } = inputs;

    let identity_sha = validate_plan_and_identity(plan, identity)?;
    ensure!(
        !ratchet_outputs.is_empty(),
        "ratchet_outputs must not be empty"
    );
    ensure!(
        !parent_of_context.is_empty(),
        "parent_of_context must not be empty"
    );
    for (ctx, parent) in parent_of_context {
        ensure!(
            !ctx.trim().is_empty() && !parent.trim().is_empty(),
            "parent_of_context contains empty context or parent cluster id"
        );
    }

    let verified_prereg = match prereg_bytes {
        Some(bytes) => Some(validate_and_bind_prereg(
            bytes,
            plan,
            identity,
            parent_of_context,
        )?),
        None => None,
    };

    let mut expected_context_set: BTreeSet<String> = if plan.expected_contexts.is_empty() {
        parent_of_context.keys().cloned().collect()
    } else {
        let mut s = BTreeSet::new();
        for ctx in &plan.expected_contexts {
            let trimmed = ctx.trim();
            ensure!(
                !trimmed.is_empty(),
                "plan.expected_contexts contains an empty context id"
            );
            ensure!(
                parent_of_context.contains_key(trimmed),
                "plan.expected_contexts context {trimmed} is missing from parent_of_context"
            );
            ensure!(
                s.insert(trimmed.to_owned()),
                "duplicate context {trimmed} in plan.expected_contexts"
            );
        }
        s
    };

    // Validate all QualTestScore records upfront, enforce 1-to-1 dataset id <-> dataset_hash
    // fingerprint coherence, and reject duplicates or ambiguous aliases.
    let mut seen_dataset_genome = BTreeSet::new();
    let mut seen_hash_prog = BTreeSet::new();
    let mut dataset_by_id: BTreeMap<String, ScoredDataset> = BTreeMap::new();
    let mut dataset_by_hash: BTreeMap<String, ScoredDataset> = BTreeMap::new();

    for ts in test_scores {
        ensure!(
            ts.schema == TEST_SCORE_SCHEMA_ID,
            "unsupported test score schema {:?}; expected {TEST_SCORE_SCHEMA_ID}",
            ts.schema
        );
        ensure!(
            ts.split == "test",
            "test score record for dataset {} has split {:?}; expected \"test\"",
            ts.dataset.id,
            ts.split
        );
        ensure!(
            !ts.dataset.id.trim().is_empty()
                && !ts.dataset.dataset_hash.trim().is_empty()
                && !ts.dataset.train_sha256.trim().is_empty()
                && !ts.dataset.test_sha256.trim().is_empty()
                && ts.dataset.train_rows > 0
                && ts.dataset.test_rows > 0,
            "test score record for dataset {:?} has empty dataset identifiers or zero rows",
            ts.dataset.id
        );
        ensure!(
            !ts.genome.trim().is_empty() && is_hex64(&ts.program_sha256),
            "test score record for dataset {} has empty genome or invalid program_sha256",
            ts.dataset.id
        );
        let expected_prog_sha = sha256_hex(ts.genome.as_bytes());
        ensure!(
            ts.program_sha256 == expected_prog_sha,
            "test score record for dataset {} genome {} has program_sha256={} != sha256(genome)={expected_prog_sha}",
            ts.dataset.id,
            ts.genome,
            ts.program_sha256
        );
        ensure!(
            ts.test_rmse.is_finite()
                && ts.test_rmse > 0.0
                && ts.validation_rmse.is_finite()
                && ts.test_r2.is_finite()
                && ts.seconds.is_finite(),
            "test score record for dataset {} genome {} has non-finite or non-positive metrics",
            ts.dataset.id,
            ts.genome
        );

        if let Some(prev) = dataset_by_id.get(&ts.dataset.id) {
            ensure!(
                prev == &ts.dataset,
                "inconsistent ScoredDataset fingerprint for dataset.id {}",
                ts.dataset.id
            );
        } else {
            dataset_by_id.insert(ts.dataset.id.clone(), ts.dataset.clone());
        }
        if let Some(prev) = dataset_by_hash.get(&ts.dataset.dataset_hash) {
            ensure!(
                prev == &ts.dataset,
                "inconsistent ScoredDataset fingerprint for dataset.dataset_hash {}",
                ts.dataset.dataset_hash
            );
        } else {
            dataset_by_hash.insert(ts.dataset.dataset_hash.clone(), ts.dataset.clone());
        }
        // Prevent cross-dataset alias collision where dataset A's id equals dataset B's dataset_hash.
        if let Some(other) = dataset_by_hash.get(&ts.dataset.id) {
            ensure!(
                other == &ts.dataset,
                "ambiguous dataset identifier collision: dataset.id {} matches another dataset's dataset_hash",
                ts.dataset.id
            );
        }
        if let Some(other) = dataset_by_id.get(&ts.dataset.dataset_hash) {
            ensure!(
                other == &ts.dataset,
                "ambiguous dataset identifier collision: dataset.dataset_hash {} matches another dataset's id",
                ts.dataset.dataset_hash
            );
        }

        ensure!(
            seen_dataset_genome.insert((ts.dataset.id.clone(), ts.genome.clone())),
            "duplicate (dataset, genome) test score record: ({}, {})",
            ts.dataset.id,
            ts.genome
        );
        ensure!(
            seen_hash_prog.insert((ts.dataset.dataset_hash.clone(), ts.program_sha256.clone())),
            "duplicate (dataset_hash, program_sha256) test score record: ({}, {})",
            ts.dataset.dataset_hash,
            ts.program_sha256
        );
    }

    // Validate all ParityRecord entries for non-empty identifiers.
    let mut seen_parity = BTreeSet::new();
    for pr in parity {
        ensure!(
            !pr.dataset_id.trim().is_empty() && !pr.genome.trim().is_empty(),
            "parity record has empty dataset_id or genome"
        );
        ensure!(
            seen_parity.insert((pr.dataset_id.clone(), pr.genome.clone())),
            "duplicate parity record for ({}, {})",
            pr.dataset_id,
            pr.genome
        );
    }

    let mut seen_contexts = BTreeSet::new();
    let mut used_test_score_indices = BTreeSet::new();
    let mut used_parity_indices = BTreeSet::new();

    let mut n_insufficient_rounds_contexts = 0usize;
    let mut n_missing_test_scores = 0usize;
    let mut n_registry_mismatch_violations = 0usize;
    let mut n_earlier_looks_violations = 0usize;
    let mut n_feeds_selection_violations = 0usize;
    let mut n_missing_parity = 0usize;
    let mut n_failed_parity = 0usize;
    let mut n_unequal_attempt_rounds = 0usize;
    let mut n_unequal_resource_contexts = 0usize;
    let mut n_unmeasured_kl_contexts = 0usize;
    let mut n_kl_frozen_contexts = 0usize;
    let mut any_ratchet_marked_non_qualified = false;
    let mut total_policy_optimizer_updates = 0u64;

    let mut context_rows = Vec::with_capacity(ratchet_outputs.len());
    let mut cluster_log_ratios: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let mut cluster_log_reductions: BTreeMap<String, Vec<f64>> = BTreeMap::new();

    for raw in ratchet_outputs {
        let schema = raw
            .get("schema")
            .and_then(Value::as_str)
            .unwrap_or_default();
        ensure!(
            schema == RATCHET_SCHEMA,
            "unsupported ratchet output schema {schema:?}; expected {RATCHET_SCHEMA}"
        );
        let context_id = raw
            .get("context_sha256")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .context("ratchet output missing non-empty context_sha256")?
            .to_owned();
        ensure!(
            seen_contexts.insert(context_id.clone()),
            "duplicate ratchet output for context {context_id}"
        );

        let roster_sha = raw
            .get("roster_sha256")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .with_context(|| format!("context {context_id} missing non-empty roster_sha256"))?;
        ensure!(
            roster_sha == identity.data_roster_sha256,
            "context {context_id} roster_sha256 ({roster_sha}) != identity.data_roster_sha256 ({})",
            identity.data_roster_sha256
        );

        let parent_id = parent_of_context
            .get(&context_id)
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .with_context(|| format!("context {context_id} has no parent mapping"))?
            .to_owned();
        if !plan.expected_contexts.is_empty() {
            ensure!(
                expected_context_set.contains(&context_id),
                "context {context_id} is not in plan.expected_contexts"
            );
        }
        expected_context_set.remove(&context_id);

        let qual_tag = raw
            .get("qualification")
            .and_then(Value::as_str)
            .unwrap_or("scratch_only");
        if qual_tag != "qualified" {
            any_ratchet_marked_non_qualified = true;
        }

        let updates = raw
            .get("policy_optimizer_updates")
            .and_then(Value::as_u64)
            .with_context(|| {
                format!("context {context_id} missing integer policy_optimizer_updates")
            })?;
        total_policy_optimizer_updates = total_policy_optimizer_updates.saturating_add(updates);

        let equal_resource_comparison = raw
            .get("equal_resource_comparison")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        if !equal_resource_comparison {
            n_unequal_resource_contexts += 1;
        }

        let rounds = raw
            .get("rounds")
            .and_then(Value::as_array)
            .with_context(|| format!("context {context_id} missing rounds array"))?;

        for (idx, round) in rounds.iter().enumerate() {
            ensure_valid_round_structure(&context_id, idx, round)?;
        }
        if let Some((ref prereg, _)) = verified_prereg {
            ensure_prereg_round_provenance(&context_id, raw, rounds, prereg)?;
        }

        let mut exclusion_reasons = Vec::new();
        if rounds.len() < 2 {
            n_insufficient_rounds_contexts += 1;
            exclusion_reasons.push("insufficient_rounds".to_owned());
        }

        let mut equal_attempts_all_rounds = true;
        let mut any_round_unmeasured_kl = false;
        let mut any_round_kl_frozen = false;
        for round in rounds {
            if round
                .get("equal_attempts_in_round")
                .and_then(Value::as_bool)
                != Some(true)
            {
                equal_attempts_all_rounds = false;
                n_unequal_attempt_rounds += 1;
            }
            let (kl_valid, kl_frozen) = evaluate_round_kl_guard(
                round,
                plan.max_conditional_base_kl,
                plan.max_new_support_mass,
            );
            if !kl_valid {
                any_round_unmeasured_kl = true;
            }
            if kl_frozen {
                any_round_kl_frozen = true;
            }
        }
        if raw.get("kl_guard_evaluated").and_then(Value::as_bool) != Some(true) {
            any_round_unmeasured_kl = true;
        }
        match raw
            .get("any_round_frozen_by_kl_guard")
            .and_then(Value::as_bool)
        {
            Some(top_frozen) => {
                if top_frozen != any_round_kl_frozen {
                    any_round_unmeasured_kl = true;
                }
                if top_frozen {
                    any_round_kl_frozen = true;
                }
            }
            None => {
                any_round_unmeasured_kl = true;
            }
        }
        if any_round_unmeasured_kl {
            n_unmeasured_kl_contexts += 1;
        }
        if any_round_kl_frozen {
            n_kl_frozen_contexts += 1;
        }

        let candidate_genome = rounds
            .last()
            .and_then(|r| r.get("ratchet_champion_genome"))
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_owned);

        let reference_genome = match plan.endpoint {
            Endpoint::FinalVsRound0Incumbent => rounds
                .first()
                .and_then(|r| r.get("ratchet_champion_genome"))
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_owned),
            Endpoint::RatchetVsOrdinaryFinalRound => {
                let Some(last_round) = rounds.last() else {
                    bail!(
                        "context {context_id} has no rounds for Endpoint::RatchetVsOrdinaryFinalRound"
                    );
                };
                let ord = last_round
                    .get("ordinary_champion_genome")
                    .and_then(Value::as_str)
                    .map(str::trim)
                    .filter(|s| !is_placeholder_sentinel(s))
                    .map(str::to_owned)
                    .with_context(|| {
                        format!(
                            "context {context_id} final round missing non-empty string 'ordinary_champion_genome' required by Endpoint::RatchetVsOrdinaryFinalRound"
                        )
                    })?;
                Some(ord)
            }
        };

        let cand_ts = match candidate_genome.as_deref() {
            Some(g) => {
                find_unique_test_score(test_scores, &context_id, g, &mut used_test_score_indices)?
            }
            None => None,
        };
        let ref_ts = match reference_genome.as_deref() {
            Some(g) => {
                find_unique_test_score(test_scores, &context_id, g, &mut used_test_score_indices)?
            }
            None => None,
        };

        if cand_ts.is_none() || ref_ts.is_none() {
            n_missing_test_scores += 1;
            exclusion_reasons.push("missing_test_score".to_owned());
        }

        if let (Some(c), Some(r)) = (cand_ts, ref_ts) {
            ensure!(
                c.dataset == r.dataset,
                "context {context_id} candidate and reference test scores have mismatched ScoredDataset fingerprints"
            );
        }

        let mut registry_matches_plan = true;
        let mut earlier_looks_clean = true;
        let mut feeds_selection_clean = true;

        for ts in [ref_ts, cand_ts].into_iter().flatten() {
            if ts.registry != plan.pinned_test_score_registry {
                registry_matches_plan = false;
            }
            if ts.feeds_selection {
                feeds_selection_clean = false;
            }
        }

        match (ref_ts, cand_ts) {
            (Some(r), Some(c)) => match (&r.earlier_looks, &c.earlier_looks) {
                (Some(ref_looks), Some(cand_looks)) => {
                    if r.program_sha256 == c.program_sha256 {
                        if !ref_looks.is_empty() || !cand_looks.is_empty() {
                            earlier_looks_clean = false;
                        }
                    } else {
                        let expected_cand_looks = vec![r.program_sha256.clone()];
                        if !registry_matches_plan
                            || !ref_looks.is_empty()
                            || cand_looks != &expected_cand_looks
                        {
                            earlier_looks_clean = false;
                        }
                    }
                }
                _ => {
                    earlier_looks_clean = false;
                }
            },
            (Some(only), None) | (None, Some(only)) => {
                if only.earlier_looks.as_ref().is_none_or(|v| !v.is_empty()) {
                    earlier_looks_clean = false;
                }
            }
            (None, None) => {}
        }

        if !registry_matches_plan {
            n_registry_mismatch_violations += 1;
            exclusion_reasons.push("test_score_registry_mismatch".to_owned());
        }
        if !earlier_looks_clean {
            n_earlier_looks_violations += 1;
            exclusion_reasons.push("earlier_looks_sequence_violation".to_owned());
        }
        if !feeds_selection_clean {
            n_feeds_selection_violations += 1;
            exclusion_reasons.push("test_score_feeds_selection".to_owned());
        }

        let cand_parity = match candidate_genome.as_deref() {
            Some(g) => find_unique_parity(parity, &context_id, g, &mut used_parity_indices)?,
            None => None,
        };
        let ref_parity = match reference_genome.as_deref() {
            Some(g) => find_unique_parity(parity, &context_id, g, &mut used_parity_indices)?,
            None => None,
        };

        let candidate_parity_passed = cand_parity.map(|p| p.passed);
        let reference_parity_passed = ref_parity.map(|p| p.passed);

        if cand_parity.is_none() || ref_parity.is_none() {
            n_missing_parity += 1;
            exclusion_reasons.push("missing_parity_record".to_owned());
        }
        if candidate_parity_passed == Some(false) || reference_parity_passed == Some(false) {
            n_failed_parity += 1;
            exclusion_reasons.push("parity_failure".to_owned());
        }

        let (paired_log_ratio, paired_log_reduction, tie_within_f10_tolerance) =
            match (cand_ts, ref_ts) {
                (Some(c), Some(r)) => {
                    let raw_ratio = c.test_rmse.ln() - r.test_rmse.ln();
                    ensure!(
                        raw_ratio.is_finite(),
                        "context {context_id} produced non-finite log RMSE ratio"
                    );
                    let tie = raw_ratio.abs() <= F10_TIE_TOLERANCE;
                    let ratio = if tie { 0.0 } else { raw_ratio };
                    let reduction = if tie { 0.0 } else { -raw_ratio };
                    (Some(ratio), Some(reduction), Some(tie))
                }
                _ => (None, None, None),
            };

        let included_in_paired_interval = rounds.len() >= 2
            && paired_log_ratio.is_some()
            && paired_log_reduction.is_some()
            && candidate_parity_passed == Some(true)
            && reference_parity_passed == Some(true);

        if included_in_paired_interval {
            if let (Some(log_ratio), Some(log_reduction)) = (paired_log_ratio, paired_log_reduction)
            {
                cluster_log_ratios
                    .entry(parent_id.clone())
                    .or_default()
                    .push(log_ratio);
                cluster_log_reductions
                    .entry(parent_id.clone())
                    .or_default()
                    .push(log_reduction);
            }
        }

        context_rows.push(ContextQualRow {
            context_id,
            parent_cluster_id: parent_id,
            rounds_observed: rounds.len(),
            candidate_genome,
            reference_genome,
            candidate_test_rmse: cand_ts.map(|t| t.test_rmse),
            reference_test_rmse: ref_ts.map(|t| t.test_rmse),
            paired_log_ratio,
            paired_log_reduction,
            tie_within_f10_tolerance,
            candidate_parity_passed,
            reference_parity_passed,
            registry_matches_plan,
            earlier_looks_clean,
            feeds_selection_clean,
            equal_attempts_all_rounds,
            equal_resource_comparison,
            kl_guard_evaluated: !any_round_unmeasured_kl,
            any_round_kl_frozen,
            included_in_paired_interval,
            exclusion_reasons,
        });
    }

    let n_missing_expected_contexts = expected_context_set.len();
    let n_unreferenced_test_scores = test_scores
        .len()
        .saturating_sub(used_test_score_indices.len());
    let n_unreferenced_parity_records = parity.len().saturating_sub(used_parity_indices.len());

    let n_complete_paired_contexts = context_rows
        .iter()
        .filter(|r| r.included_in_paired_interval)
        .count();
    let n_observed_parent_clusters = cluster_log_reductions.len();

    let (paired_log_ratio, paired_log_reduction) =
        if n_complete_paired_contexts >= 2 && n_observed_parent_clusters >= 2 {
            let ratio_groups: Vec<Vec<f64>> = cluster_log_ratios.into_values().collect();
            let reduction_groups: Vec<Vec<f64>> = cluster_log_reductions.into_values().collect();
            let ratio_ci = cluster_bootstrap_ci(
                &ratio_groups,
                plan.bootstrap_resamples as usize,
                plan.bootstrap_seed,
                plan.alpha,
            )?;
            let reduction_ci = cluster_bootstrap_ci(
                &reduction_groups,
                plan.bootstrap_resamples as usize,
                plan.bootstrap_seed,
                plan.alpha,
            )?;
            (Some(ratio_ci), Some(reduction_ci))
        } else {
            (None, None)
        };

    let verdict = paired_log_reduction
        .as_ref()
        .map(|pi| pi.verdict)
        .unwrap_or(Verdict::Inconclusive);
    let margin_cleared = paired_log_reduction
        .as_ref()
        .is_some_and(|pi| pi.cluster_bootstrap.low > plan.practical_margin_log_ratio);

    let sealed_family_boundary_adopted = !GENERALIZATION_CLAIM.starts_with("none:");

    let mut disqualification_reasons = Vec::new();
    if plan.source_grade != SourceGrade::Qualified || any_ratchet_marked_non_qualified {
        disqualification_reasons.push("source_grade_not_qualified".to_owned());
    }
    if n_missing_expected_contexts > 0 {
        disqualification_reasons.push("missing_expected_contexts".to_owned());
    }
    if n_insufficient_rounds_contexts > 0 {
        disqualification_reasons.push("insufficient_rounds".to_owned());
    }
    if n_missing_test_scores > 0 {
        disqualification_reasons.push("missing_test_scores".to_owned());
    }
    if n_registry_mismatch_violations > 0 {
        disqualification_reasons.push("test_score_registry_mismatch".to_owned());
    }
    if n_earlier_looks_violations > 0 {
        disqualification_reasons.push("test_split_earlier_looks_sequence_violation".to_owned());
    }
    if n_feeds_selection_violations > 0 {
        disqualification_reasons.push("test_score_feeds_selection".to_owned());
    }
    if n_missing_parity > 0 {
        disqualification_reasons.push("missing_parity_record".to_owned());
    }
    if n_failed_parity > 0 {
        disqualification_reasons.push("parity_failure".to_owned());
    }
    if n_unreferenced_test_scores > 0 {
        disqualification_reasons.push("unreferenced_test_score_records".to_owned());
    }
    if n_unreferenced_parity_records > 0 {
        disqualification_reasons.push("unreferenced_parity_records".to_owned());
    }
    if n_unequal_attempt_rounds > 0 {
        disqualification_reasons.push("unequal_attempts_in_round".to_owned());
    }
    if n_unequal_resource_contexts > 0 {
        disqualification_reasons.push("unequal_resource_comparison".to_owned());
    }
    if n_unmeasured_kl_contexts > 0 {
        disqualification_reasons.push("kl_guard_unmeasured".to_owned());
    }
    if n_kl_frozen_contexts > 0 && !plan.allow_kl_frozen_rounds {
        disqualification_reasons.push("kl_guard_frozen_round".to_owned());
    }
    if n_observed_parent_clusters < plan.min_parent_clusters {
        disqualification_reasons.push("insufficient_parent_clusters".to_owned());
    }
    if paired_log_reduction.is_none() {
        disqualification_reasons.push("paired_interval_not_computed".to_owned());
    } else if !margin_cleared {
        disqualification_reasons.push("margin_not_cleared".to_owned());
    }

    let caller_asserted_diagnostic_checks_passed = disqualification_reasons.is_empty();
    let provenance_binding_status = match verified_prereg {
        None => {
            disqualification_reasons.push("provenance_and_prereg_binding_unimplemented".to_owned());
            PROVENANCE_BINDING_STATUS.to_owned()
        }
        Some((_, pinned)) => {
            if !pinned {
                disqualification_reasons.push("prereg_unpinned_or_no_execution_grant".to_owned());
            }
            disqualification_reasons.push(PREREG_CALLER_ASSERTED_REASON.to_owned());
            disqualification_reasons
                .push("qualified_execution_mode_disabled_offline_verifier_only".to_owned());
            if pinned {
                PREREG_PINNED_BINDING_STATUS.to_owned()
            } else {
                PREREG_UNPINNED_BINDING_STATUS.to_owned()
            }
        }
    };
    if !sealed_family_boundary_adopted {
        disqualification_reasons.push("no_adopted_sealed_family_boundary".to_owned());
    }

    Ok(RoundQualReport {
        schema: ROUND_QUAL_SCHEMA.to_owned(),
        experiment_identity_sha256: identity_sha,
        prereg_sha256: plan.prereg_sha256.clone(),
        provenance_binding_status,
        multiplicity_adjustment: MULTIPLICITY_ADJUSTMENT_STATUS.to_owned(),
        source_grade: plan.source_grade,
        pinned_test_score_registry: plan.pinned_test_score_registry.clone(),
        endpoint: plan.endpoint,
        alpha: plan.alpha,
        min_parent_clusters: plan.min_parent_clusters,
        practical_margin_log_ratio: plan.practical_margin_log_ratio,
        allow_kl_frozen_rounds: plan.allow_kl_frozen_rounds,
        max_conditional_base_kl: plan.max_conditional_base_kl,
        max_new_support_mass: plan.max_new_support_mass,
        n_ratchet_contexts: ratchet_outputs.len(),
        n_missing_expected_contexts,
        n_complete_paired_contexts,
        n_observed_parent_clusters,
        n_insufficient_rounds_contexts,
        n_missing_test_scores,
        n_registry_mismatch_violations,
        n_earlier_looks_violations,
        n_feeds_selection_violations,
        n_missing_parity,
        n_failed_parity,
        n_unreferenced_test_scores,
        n_unreferenced_parity_records,
        n_unequal_attempt_rounds,
        n_unequal_resource_contexts,
        n_unmeasured_kl_contexts,
        n_kl_frozen_contexts,
        policy_optimizer_updates: total_policy_optimizer_updates,
        paired_log_ratio,
        paired_log_reduction,
        verdict,
        margin_cleared,
        sealed_family_boundary_adopted,
        caller_asserted_diagnostic_checks_passed,
        qualified_claim: false,
        promotion: false,
        l6_claim: false,
        generalization_claim: GENERALIZATION_CLAIM.to_owned(),
        disqualification_reasons,
        contexts: context_rows,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clover::{Budget, InitialState, SourceIdentity, SCHEMA};
    use serde_json::json;

    fn sample_identity() -> ExperimentIdentity {
        let hex64 = |b: u8| std::iter::repeat_n(b as char, 64).collect::<String>();
        let mut guards = BTreeMap::new();
        guards.insert("quantile".to_owned(), hex64(b'1'));
        guards.insert("label_visibility".to_owned(), hex64(b'2'));
        guards.insert("hold_recovery".to_owned(), hex64(b'3'));
        ExperimentIdentity {
            schema: SCHEMA.to_owned(),
            source: SourceIdentity {
                commit: "525be65b38540eb64b84c9bf36f7da6812812f2f".to_owned(),
                closure_sha256: hex64(b'a'),
            },
            initial_state: InitialState::Checkpoint {
                parameters_sha256: hex64(b'b'),
                optimizer_sha256: hex64(b'c'),
                rng_state_sha256: hex64(b'd'),
            },
            evaluator_sha256: hex64(b'e'),
            parity_contract_sha256: hex64(b'f'),
            support_sha256: hex64(b'4'),
            objective_sha256: hex64(b'5'),
            data_roster_sha256: hex64(b'6'),
            seed: 20_261_016,
            budget: Budget {
                attempts: 64,
                wall_seconds: 120,
                cpu_seconds: 240,
                gpu_seconds: 0,
                memory_bytes: 1_073_741_824,
                daily_bytes: 1_000_000,
                daily_puts: 1_000,
            },
            guards,
        }
    }

    fn sample_plan(identity: &ExperimentIdentity) -> RoundQualPlan {
        RoundQualPlan {
            experiment_identity_sha256: identity.sha256().expect("valid identity"),
            prereg_sha256: "9".repeat(64),
            source_grade: SourceGrade::Qualified,
            pinned_test_score_registry: "fixture-registry".to_owned(),
            min_parent_clusters: 3,
            practical_margin_log_ratio: 0.05,
            bootstrap_resamples: 1000,
            bootstrap_seed: 20_261_016,
            alpha: 0.025,
            endpoint: Endpoint::FinalVsRound0Incumbent,
            allow_kl_frozen_rounds: false,
            max_conditional_base_kl: 0.20,
            max_new_support_mass: 0.15,
            expected_contexts: Vec::new(),
        }
    }

    fn sample_decomposed_kl(admissible: bool) -> Value {
        let conditional_base_kl = if admissible { 0.05 } else { 0.35 };
        let conditional_new_kl = 0.01;
        let base_support_mass = 0.90;
        let new_support_mass = 0.10;
        let ref_new_support_mass = 0.08;
        let bernoulli_allocation_kl = 0.002;
        let total_kl = bernoulli_allocation_kl
            + base_support_mass * conditional_base_kl
            + new_support_mass * conditional_new_kl;
        json!({
            "conditional_base_kl": conditional_base_kl,
            "conditional_new_kl": conditional_new_kl,
            "base_support_mass": base_support_mass,
            "new_support_mass": new_support_mass,
            "ref_new_support_mass": ref_new_support_mass,
            "bernoulli_allocation_kl": bernoulli_allocation_kl,
            "total_kl": total_kl,
            "admissible": admissible
        })
    }

    fn sample_ratchet_output(
        identity: &ExperimentIdentity,
        context_id: &str,
        round0_genome: &str,
        final_genome: &str,
        equal_resource: bool,
        equal_attempts: bool,
        kl_frozen: bool,
    ) -> Value {
        json!({
            "schema": RATCHET_SCHEMA,
            "qualification": "qualified",
            "roster_sha256": identity.data_roster_sha256,
            "context_sha256": context_id,
            "equal_resource_comparison": equal_resource,
            "kl_guard_evaluated": true,
            "any_round_frozen_by_kl_guard": kl_frozen,
            "policy_optimizer_updates": 0,
            "rounds": [
                {
                    "round": 0,
                    "cumulative_prior_rounds": 0,
                    "ordinary_best_rmse": 1.0,
                    "ordinary_best_q": 0.5,
                    "ratchet_best_rmse": 1.0,
                    "ratchet_best_q": 0.5,
                    "surplus_q_vs_ordinary": 0.0,
                    "ratchet_champion_genome": round0_genome,
                    "equal_attempts_in_round": equal_attempts,
                    "kl_source": "measured_tilt_leaf_policy",
                    "kl_malformed_reason": Value::Null,
                    "decomposed_kl": sample_decomposed_kl(true),
                    "seed_update_frozen_by_kl_guard": false
                },
                {
                    "round": 1,
                    "cumulative_prior_rounds": 1,
                    "ordinary_best_rmse": 0.95,
                    "ordinary_best_q": 0.51,
                    "ratchet_best_rmse": 0.70,
                    "ratchet_best_q": 0.58,
                    "surplus_q_vs_ordinary": 0.07,
                    "ratchet_champion_genome": final_genome,
                    "equal_attempts_in_round": equal_attempts,
                    "kl_source": "measured_tilt_leaf_policy",
                    "kl_malformed_reason": Value::Null,
                    "decomposed_kl": sample_decomposed_kl(!kl_frozen),
                    "seed_update_frozen_by_kl_guard": kl_frozen
                }
            ]
        })
    }

    fn sample_test_score(
        context_id: &str,
        genome: &str,
        test_rmse: f64,
        earlier_looks: Vec<String>,
    ) -> QualTestScore {
        QualTestScore {
            schema: TEST_SCORE_SCHEMA_ID.to_owned(),
            tool_version: "0.3.0".to_owned(),
            created_at: "2026-10-16T17:30:00Z".to_owned(),
            split: "test".to_owned(),
            receipt_sha256: "a".repeat(64),
            dataset: ScoredDataset {
                id: context_id.to_owned(),
                dataset_hash: context_id.to_owned(),
                train_sha256: "b".repeat(64),
                test_sha256: "c".repeat(64),
                train_rows: 100,
                test_rows: 50,
            },
            genome: genome.to_owned(),
            program_sha256: sha256_hex(genome.as_bytes()),
            test_rmse,
            test_r2: 0.85,
            validation_rmse: test_rmse * 0.98,
            registry: "fixture-registry".to_owned(),
            earlier_looks: Some(earlier_looks),
            feeds_selection: false,
            seconds: 0.1,
        }
    }

    #[test]
    fn clean_synthetic_fixture_passes_caller_asserted_diagnostics_only() {
        let identity = sample_identity();
        let plan = sample_plan(&identity);
        let mut parent_map = BTreeMap::new();
        let mut ratchets = Vec::new();
        let mut scores = Vec::new();
        let mut parity = Vec::new();

        for (ctx, parent, ref_rmse, cand_rmse) in [
            ("ctx1", "parent_a", 1.00, 0.75),
            ("ctx2", "parent_b", 1.10, 0.80),
            ("ctx3", "parent_c", 0.90, 0.65),
            ("ctx4", "parent_c", 1.20, 0.85),
        ] {
            parent_map.insert(ctx.to_owned(), parent.to_owned());
            let g0 = format!("g0_{ctx}");
            let g1 = format!("g1_{ctx}");
            let ref_score = sample_test_score(ctx, &g0, ref_rmse, Vec::new());
            let cand_score =
                sample_test_score(ctx, &g1, cand_rmse, vec![ref_score.program_sha256.clone()]);
            ratchets.push(sample_ratchet_output(
                &identity, ctx, &g0, &g1, true, true, false,
            ));
            scores.push(ref_score);
            scores.push(cand_score);
            parity.push(ParityRecord {
                dataset_id: ctx.to_owned(),
                genome: g0,
                passed: true,
                failure_reason: None,
            });
            parity.push(ParityRecord {
                dataset_id: ctx.to_owned(),
                genome: g1,
                passed: true,
                failure_reason: None,
            });
        }

        let report = qualify_rounds(RoundQualInputs {
            plan: &plan,
            identity: &identity,
            ratchet_outputs: &ratchets,
            test_scores: &scores,
            parity: &parity,
            parent_of_context: &parent_map,
        })
        .expect("valid qualification report");

        assert_eq!(report.schema, ROUND_QUAL_SCHEMA);
        assert_eq!(report.n_ratchet_contexts, 4);
        assert_eq!(report.n_complete_paired_contexts, 4);
        assert_eq!(report.n_observed_parent_clusters, 3);
        assert_eq!(report.verdict, Verdict::Greater);
        assert!(report.margin_cleared);
        assert!(report.caller_asserted_diagnostic_checks_passed);
        assert!(!report.sealed_family_boundary_adopted);
        assert!(!report.qualified_claim);
        assert!(!report.promotion);
        assert!(!report.l6_claim);
        assert_eq!(
            report.disqualification_reasons,
            vec![
                "provenance_and_prereg_binding_unimplemented".to_owned(),
                "no_adopted_sealed_family_boundary".to_owned(),
            ]
        );
        assert_eq!(report.canonical_sha256().expect("sha256").len(), 64);
    }

    #[test]
    fn f10_tie_tolerance_and_extreme_positive_rmse_log_difference() {
        let identity = sample_identity();
        let mut plan = sample_plan(&identity);
        plan.min_parent_clusters = 2;
        let mut parent_map = BTreeMap::new();
        parent_map.insert("ctx1".to_owned(), "p1".to_owned());
        parent_map.insert("ctx2".to_owned(), "p2".to_owned());

        let ratchets = vec![
            sample_ratchet_output(&identity, "ctx1", "g0_1", "g1_1", true, true, false),
            sample_ratchet_output(&identity, "ctx2", "g0_2", "g1_2", true, true, false),
        ];
        let s1_ref = sample_test_score("ctx1", "g0_1", 1.0, Vec::new());
        let s1_cand = sample_test_score(
            "ctx1",
            "g1_1",
            1.0 * (1.0 - 5e-13),
            vec![s1_ref.program_sha256.clone()],
        );
        // Extreme positive RMSEs where (c / r) would underflow/overflow if computed via division,
        // whereas ln(c) - ln(r) stays finite.
        let s2_ref = sample_test_score("ctx2", "g0_2", 1e200, Vec::new());
        let s2_cand =
            sample_test_score("ctx2", "g1_2", 1e-200, vec![s2_ref.program_sha256.clone()]);
        let scores = vec![s1_ref, s1_cand, s2_ref, s2_cand];
        let parity = vec![
            ParityRecord {
                dataset_id: "ctx1".to_owned(),
                genome: "g0_1".to_owned(),
                passed: true,
                failure_reason: None,
            },
            ParityRecord {
                dataset_id: "ctx1".to_owned(),
                genome: "g1_1".to_owned(),
                passed: true,
                failure_reason: None,
            },
            ParityRecord {
                dataset_id: "ctx2".to_owned(),
                genome: "g0_2".to_owned(),
                passed: true,
                failure_reason: None,
            },
            ParityRecord {
                dataset_id: "ctx2".to_owned(),
                genome: "g1_2".to_owned(),
                passed: true,
                failure_reason: None,
            },
        ];

        let report = qualify_rounds(RoundQualInputs {
            plan: &plan,
            identity: &identity,
            ratchet_outputs: &ratchets,
            test_scores: &scores,
            parity: &parity,
            parent_of_context: &parent_map,
        })
        .expect("report");

        assert_eq!(report.contexts[0].tie_within_f10_tolerance, Some(true));
        assert_eq!(report.contexts[0].paired_log_ratio, Some(0.0));
        assert_eq!(report.contexts[0].paired_log_reduction, Some(0.0));
        assert!(report.contexts[1]
            .paired_log_reduction
            .expect("finite reduction")
            .is_finite());
        assert!(report.contexts[1].paired_log_reduction.unwrap() > 900.0);
    }
}
