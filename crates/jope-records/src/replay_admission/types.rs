use crate::wire_contract::MissingReason;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// Local episode permission is deliberately a different purpose from every global use.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Purpose {
    GlobalPolicyTraining,
    RepresentationLearning,
    RetrievalIndexUpdate,
    Distillation,
    EpisodeFit,
    Audit,
}
impl Purpose {
    pub fn is_global(self) -> bool {
        !matches!(self, Self::EpisodeFit | Self::Audit)
    }
}

/// Supplied by the controller's separately governed adoption configuration, never a candidate.
/// This lane does not install a policy, issue signatures, or authorize an unsealing.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolicyPin {
    pub schema: String,
    pub policy_sha256: String,
    pub issuer: String,
    pub epoch: u64,
}

/// Immutable verifier output manifest. Every receipt must be independently admitted into this
/// exact set by the controller before it can be consumed. Schema conformance alone is insufficient.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdmissionPolicy {
    pub schema: String,
    pub issuer: String,
    pub epoch: u64,
    pub effective_from_ms: u64,
    pub effective_until_ms: u64,
    /// Preregistered future evaluation boundary; a successor must name a new one.
    pub evaluation_boundary_sha256: String,
    pub purposes: BTreeSet<Purpose>,
    pub receipts: BTreeSet<String>,
    /// A new epoch never changes the original role or observation time of an old receipt.
    pub supersedes: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactValue {
    pub f64_bits: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QualityLabel {
    pub metric: String,
    /// I02 exact finite binary64 spelling; no raw float in a new identity payload.
    pub value: ExactValue,
    pub unit: String,
    pub denominator: u64,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TimeoutCause {
    ProgramDeadline,
    Infrastructure,
    Unknown,
}

/// Outcome claims stay unverified until their complete exposure receipt is in the pinned
/// verifier manifest. In particular, I02's caller-constructible evidence Boolean is not used.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
pub enum Observation {
    Proposed,
    Completed {
        quality: Option<QualityLabel>,
        elapsed_ms: Option<u64>,
        policy_target_sha256: Option<String>,
    },
    Timeout {
        lower_bound_ms: Option<u64>,
        cause: TimeoutCause,
    },
    Failed {
        elapsed_ms: Option<u64>,
    },
}

/// Original role, family partition, within-task view, fidelity and producer remain separate.
/// Parents include *all* material inputs, including summaries, selections and embeddings.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExposureReceipt {
    pub schema: String,
    pub subject_sha256: String,
    pub family: String,
    pub meta_partition: String,
    pub original_role: String,
    pub within_task_view: String,
    pub fidelity: String,
    pub producer: String,
    pub policy_epoch: u64,
    pub observed_at_ms: u64,
    pub episode: Option<String>,
    pub purposes: BTreeSet<Purpose>,
    /// None means unknown exposure, not an empty exposure history.
    pub prior_exposure: Option<Vec<String>>,
    pub parents: Vec<String>,
    pub observation: Observation,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "availability", rename_all = "snake_case")]
pub enum Label<T> {
    Available {
        value: T,
        unit: String,
        denominator: u64,
    },
    Missing {
        #[serde(serialize_with = "super::labels::serialize_missing_reason")]
        reason: MissingReason,
        detail: &'static str,
    },
}
impl<T> Label<T> {
    pub fn value(&self) -> Option<&T> {
        match self {
            Self::Available { value, .. } => Some(value),
            Self::Missing { .. } => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(
    tag = "observation_type",
    content = "milliseconds",
    rename_all = "snake_case"
)]
pub enum RuntimeLabel {
    Exact(u64),
    RightCensored(u64),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct LabelEligibility {
    pub quality: Label<QualityLabel>,
    pub runtime: Label<RuntimeLabel>,
    pub failure: Label<bool>,
    pub policy: Label<String>,
}

/// Cannot be deserialized or built from a role string. Only the verifier constructs it.
#[derive(Debug, Serialize)]
pub struct ExposureClosure {
    pub(super) policy_sha256: String,
    pub(super) policy_id: String,
    pub(super) receipt_sha256: String,
    pub(super) subject_sha256: String,
    pub(super) ancestors: BTreeSet<String>,
    #[serde(skip)]
    pub(super) subjects: BTreeSet<String>,
}

/// A purpose/subject-bound capability, not a general "eligible" flag. Every ingress recomputes
/// it against fresh retained bytes and revocations, rather than trusting a serialized capability.
///
/// ```compile_fail
/// use jope_records::replay_admission::ReplayAdmission;
/// let forged: ReplayAdmission = serde_json::from_str(r#"{"eligible":true,"role":"train"}"#).unwrap();
/// ```
#[derive(Debug, Serialize)]
pub struct ReplayAdmission {
    pub(super) schema: &'static str,
    pub(super) purpose: Purpose,
    pub(super) episode: Option<String>,
    pub(super) closure: ExposureClosure,
    pub(super) labels: LabelEligibility,
}

impl ReplayAdmission {
    pub fn labels(&self) -> &LabelEligibility {
        &self.labels
    }
    pub fn purpose(&self) -> Purpose {
        self.purpose
    }
    pub fn subject_sha256(&self) -> &str {
        &self.closure.subject_sha256
    }
    pub fn receipt_sha256(&self) -> &str {
        &self.closure.receipt_sha256
    }
    pub fn policy_sha256(&self) -> &str {
        &self.closure.policy_sha256
    }
    pub fn source_subjects(&self) -> &BTreeSet<String> {
        &self.closure.subjects
    }
}

/// A fitting-component capability whose complete ancestry is strictly FIT-only training.
/// Generic episode admission may permit inner validation; this capability never does.
///
/// ```compile_fail
/// use jope_records::replay_admission::FitAdmission;
/// let forged: FitAdmission = serde_json::from_str(r#"{"eligible":true}"#).unwrap();
/// ```
#[derive(Debug, Serialize)]
pub struct FitAdmission {
    pub(super) admission: ReplayAdmission,
}
impl FitAdmission {
    /// Borrow the underlying purpose/subject-bound proof without widening this capability.
    pub fn admission(&self) -> &ReplayAdmission {
        &self.admission
    }
}

/// A review artifact binding the controller's adopted pin and imported revocation state.
/// Presence is not a signature, an adoption action, or permission to trust a new issuer.
/// Installation into the separately governed operator state remains an explicit owner boundary.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ComponentAuthoritySnapshot {
    pub schema: String,
    /// SHA256 of exact canonical bytes at the existing ADOPTION_PATH.
    pub adoption_sha256: String,
    /// Exact HEAD.json and batch byte hashes, with paths relative to the trusted State root.
    pub revocation_files: BTreeMap<String, String>,
    /// All digests imported by the verified local revocation ledger; no invented global claim.
    pub revoked_digests: BTreeSet<String>,
}
