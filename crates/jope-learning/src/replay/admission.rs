//! Pure, purpose-bound admission from a controller-pinned receipt closure.
//!
//! The pin is trusted controller configuration, separate from submitted evidence. A digest
//! authenticates bytes relative to that pin; it is not a signature or an adoption decision.
//! Never obtain the pin from a row, bank export, qualification, or signature-shaped JSON.
use crate::canon::{is_hex64, sha256_hex};
use crate::revocation::Revocations;
use crate::state::State;
use anyhow::{bail, ensure, Context, Result};
use jope_records::wire_contract::{CanonicalPayload, ExactF64, MissingReason};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const POLICY_SCHEMA: &str = "jope.replay-admission-policy/v2";
pub const EXPOSURE_SCHEMA: &str = "jope.replay-exposure/v2";
pub const ADMISSION_SCHEMA: &str = "jope.replay-admission/v2";
pub const ADOPTION_SCHEMA: &str = "jope.replay-policy-adoption/v2";
/// Privileged controller configuration, never a candidate-controlled pathname.
pub const ADOPTION_PATH: &str = "replay/adopted-policy.json";
const MAX_NODES: usize = 128;
const MAX_DEPTH: usize = 32;

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
        #[serde(serialize_with = "serialize_missing_reason")]
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

fn available<T>(value: T, unit: &str, denominator: u64) -> Label<T> {
    Label::Available {
        value,
        unit: unit.into(),
        denominator,
    }
}
fn serialize_missing_reason<S: serde::Serializer>(
    reason: &MissingReason,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    serializer.serialize_str(reason.as_str())
}
fn missing<T>(reason: MissingReason, detail: &'static str) -> Label<T> {
    Label::Missing { reason, detail }
}

fn labels(observation: &Observation) -> Result<LabelEligibility> {
    let mut out = LabelEligibility {
        quality: missing(MissingReason::Unscheduled, "not_measured"),
        runtime: missing(MissingReason::Unscheduled, "not_measured"),
        failure: missing(MissingReason::Unscheduled, "not_measured"),
        policy: missing(MissingReason::NotApplicable, "no_measured_policy_target"),
    };
    match observation {
        Observation::Proposed => {}
        Observation::Completed {
            quality,
            elapsed_ms,
            policy_target_sha256,
        } => {
            out.quality = missing(MissingReason::FailedClosed, "quality_unavailable");
            out.runtime = missing(MissingReason::FailedClosed, "duration_unavailable");
            if let Some(q) = quality {
                ExactF64::from_hex("/quality/f64_bits", &q.value.f64_bits)?;
                ensure!(
                    !q.metric.trim().is_empty() && !q.unit.trim().is_empty() && q.denominator > 0,
                    "HELD: invalid quality unit/denominator"
                );
                out.quality = available(q.clone(), &q.unit, q.denominator);
            }
            if let Some(ms) = elapsed_ms {
                out.runtime = available(RuntimeLabel::Exact(*ms), "ms", 1);
            }
            out.failure = available(false, "attempt", 1);
            if let Some(target) = policy_target_sha256 {
                ensure!(is_hex64(target), "HELD: malformed measured policy target");
                out.policy = available(target.clone(), "target", 1);
            }
        }
        Observation::Timeout {
            lower_bound_ms,
            cause,
        } => {
            out.quality = missing(MissingReason::BudgetExhausted, "timeout_has_no_quality");
            out.policy = missing(
                MissingReason::BudgetExhausted,
                "timeout_has_no_policy_target",
            );
            out.runtime = missing(MissingReason::FailedClosed, "timeout_bound_unavailable");
            if let Some(ms) = lower_bound_ms {
                ensure!(*ms > 0, "HELD: invalid timeout bound");
                out.runtime = available(RuntimeLabel::RightCensored(*ms), "ms", 1);
            }
            if matches!(cause, TimeoutCause::ProgramDeadline) {
                out.failure = available(true, "attempt", 1);
            } else {
                out.failure = missing(
                    MissingReason::FailedClosed,
                    "no_program_failure_observation",
                );
            }
        }
        Observation::Failed { elapsed_ms } => {
            out.quality = missing(MissingReason::FailedClosed, "failed_attempt_has_no_quality");
            out.runtime = missing(MissingReason::FailedClosed, "duration_unavailable");
            if let Some(ms) = elapsed_ms {
                out.runtime = available(RuntimeLabel::Exact(*ms), "ms", 1);
            }
            out.failure = available(true, "attempt", 1);
        }
    }
    Ok(out)
}

/// Cannot be deserialized or built from a role string. Only the verifier constructs it.
#[derive(Debug, Serialize)]
pub struct ExposureClosure {
    policy_sha256: String,
    policy_id: String,
    receipt_sha256: String,
    subject_sha256: String,
    ancestors: BTreeSet<String>,
}

/// A purpose/subject-bound capability, not a general "eligible" flag. Every ingress recomputes
/// it against fresh retained bytes and revocations, rather than trusting a serialized capability.
///
/// ```compile_fail
/// use jope_learning::replay::admission::ReplayAdmission;
/// let forged: ReplayAdmission = serde_json::from_str(r#"{"eligible":true,"role":"train"}"#).unwrap();
/// ```
#[derive(Debug, Serialize)]
pub struct ReplayAdmission {
    schema: &'static str,
    purpose: Purpose,
    episode: Option<String>,
    closure: ExposureClosure,
    labels: LabelEligibility,
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
}

/// All content references use the existing State byte-SHA namespace. Each new payload is also
/// checked by I02's strict canonical codec and has its distinct domain-separated identity.
fn retained<T: DeserializeOwned>(state: &State, digest: &str, schema: &str) -> Result<(T, String)> {
    ensure!(is_hex64(digest), "HELD: invalid proof digest");
    let bytes = state
        .read_blob(digest)?
        .context("HELD: missing exposure proof")?;
    ensure!(sha256_hex(&bytes) == digest, "HELD: altered exposure proof");
    let payload = CanonicalPayload::from_canonical_bytes(&bytes)?;
    ensure!(
        payload.schema_id() == schema,
        "HELD: unsupported exposure proof version"
    );
    Ok((
        serde_json::from_slice(&bytes)?,
        payload.domain_separated_id()?.sha256_hex().to_owned(),
    ))
}

/// The lifetime binds the verifier to the controller's evidence store. Request data cannot
/// replace its policy pin. No method here adopts a policy or writes a role/exposure receipt.
pub struct AdmissionVerifier<'a> {
    state: &'a State,
    pin: PolicyPin,
}
impl<'a> AdmissionVerifier<'a> {
    /// Only the controller's installed adoption pin is read. There is deliberately no public
    /// constructor accepting a candidate's pin, issuer, verifier Boolean or policy pathname.
    pub fn from_adopted_policy(state: &'a State) -> Result<Self> {
        let pin = Self::adopted_pin(state)?;
        let verifier = Self { state, pin };
        verifier.policy()?;
        Ok(verifier)
    }

    fn adopted_pin(state: &State) -> Result<PolicyPin> {
        let bytes = std::fs::read(state.root().join(ADOPTION_PATH))
            .context("HELD: no controller-adopted replay policy")?;
        let payload = CanonicalPayload::from_canonical_bytes(&bytes)?;
        ensure!(
            payload.schema_id() == ADOPTION_SCHEMA,
            "HELD: unknown policy adoption version"
        );
        Ok(serde_json::from_slice(&bytes)?)
    }

    fn policy(&self) -> Result<(AdmissionPolicy, String)> {
        ensure!(
            Self::adopted_pin(self.state)? == self.pin,
            "HELD: controller policy changed; reconstruct verifier under the new epoch"
        );
        let (policy, id): (AdmissionPolicy, _) =
            retained(self.state, &self.pin.policy_sha256, POLICY_SCHEMA)?;
        ensure!(
            policy.issuer == self.pin.issuer
                && !policy.issuer.trim().is_empty()
                && policy.epoch == self.pin.epoch
                && policy.effective_from_ms < policy.effective_until_ms
                && !policy.receipts.is_empty()
                && policy.receipts.len() <= 100_000
                && policy.receipts.iter().all(|d| is_hex64(d))
                && is_hex64(&policy.evaluation_boundary_sha256)
                && self
                    .state
                    .read_blob(&policy.evaluation_boundary_sha256)?
                    .is_some()
                && policy.supersedes.as_ref().is_none_or(|d| is_hex64(d)),
            "HELD: unadopted issuer/policy epoch"
        );
        if let Some(previous) = &policy.supersedes {
            let (parent, _): (AdmissionPolicy, _) = retained(self.state, previous, POLICY_SCHEMA)?;
            ensure!(
                parent.issuer == policy.issuer
                    && parent.epoch < policy.epoch
                    && parent.effective_from_ms < policy.effective_from_ms
                    && parent.evaluation_boundary_sha256 != policy.evaluation_boundary_sha256,
                "HELD: prospective revision lacks a new future evaluation boundary"
            );
        }
        Ok((policy, id))
    }

    pub fn require_state(&self, state: &State) -> Result<()> {
        ensure!(
            self.state.root() == state.root(),
            "HELD: verifier belongs to a different state"
        );
        Ok(())
    }

    pub fn admit(
        &self,
        subject_sha256: &str,
        receipt_sha256: &str,
        purpose: Purpose,
        episode: Option<&str>,
        at_ms: u64,
    ) -> Result<ReplayAdmission> {
        let (policy, policy_id) = self.policy()?;
        ensure!(
            (policy.effective_from_ms..policy.effective_until_ms).contains(&at_ms)
                && policy.purposes.contains(&purpose),
            "HELD: purpose or policy epoch not effective"
        );
        let revocations = Revocations::load(self.state)?;
        revocations.refuse("replay policy", [&*self.pin.policy_sha256, &*policy_id])?;
        revocations.refuse(
            "future evaluation boundary",
            [policy.evaluation_boundary_sha256.as_str()],
        )?;
        let mut closure = BTreeMap::new();
        self.visit(
            receipt_sha256,
            &policy,
            &revocations,
            &mut BTreeSet::new(),
            &mut closure,
            0,
        )?;
        let root = closure
            .get(receipt_sha256)
            .context("HELD: missing exposure root")?;
        ensure!(
            root.subject_sha256 == subject_sha256,
            "HELD: proof belongs to another subject"
        );
        for receipt in closure.values() {
            ensure!(
                receipt.observed_at_ms <= at_ms && receipt.purposes.contains(&purpose),
                "HELD: purpose not permitted by every ancestor"
            );
            if purpose.is_global() {
                ensure!(receipt.meta_partition == "train_search"
                    && matches!(receipt.original_role.as_str(), "train" | "train_fit")
                    && matches!(receipt.within_task_view.as_str(), "training" | "inner_validation")
                    && receipt.prior_exposure.as_ref().is_some_and(|roles|
                        roles.iter().all(|r| matches!(r.as_str(), "train" | "train_fit"))),
                    "HELD: protected role/view/exposure ancestry cannot enter global learning/retrieval");
            } else if purpose == Purpose::EpisodeFit {
                ensure!(
                    episode.is_some_and(|e| !e.is_empty() && receipt.episode.as_deref() == Some(e))
                        && matches!(
                            receipt.within_task_view.as_str(),
                            "training" | "inner_validation"
                        )
                        && !matches!(
                            receipt.original_role.as_str(),
                            "test_adaptive" | "judge" | "sealed" | "meta_test"
                        ),
                    "HELD: local fitting is outside its declared episode/view"
                );
            }
        }
        let eligibility = labels(&root.observation)?;
        Ok(ReplayAdmission {
            schema: ADMISSION_SCHEMA,
            purpose,
            episode: episode.map(str::to_owned),
            labels: eligibility,
            closure: ExposureClosure {
                policy_sha256: self.pin.policy_sha256.clone(),
                policy_id,
                receipt_sha256: receipt_sha256.into(),
                subject_sha256: subject_sha256.into(),
                ancestors: closure.keys().cloned().collect(),
            },
        })
    }

    pub fn revalidate(&self, admission: &ReplayAdmission, at_ms: u64) -> Result<ReplayAdmission> {
        ensure!(
            admission.closure.policy_sha256 == self.pin.policy_sha256,
            "HELD: admission belongs to a different policy epoch"
        );
        self.admit(
            admission.subject_sha256(),
            admission.receipt_sha256(),
            admission.purpose,
            admission.episode.as_deref(),
            at_ms,
        )
    }

    fn visit(
        &self,
        digest: &str,
        policy: &AdmissionPolicy,
        revocations: &Revocations,
        active: &mut BTreeSet<String>,
        found: &mut BTreeMap<String, ExposureReceipt>,
        depth: usize,
    ) -> Result<()> {
        ensure!(!active.contains(digest), "HELD: cyclic exposure ancestry");
        if found.contains_key(digest) {
            return Ok(());
        }
        ensure!(
            depth <= MAX_DEPTH && found.len() + active.len() < MAX_NODES,
            "HELD: exposure closure exceeds its bound"
        );
        ensure!(
            policy.receipts.contains(digest),
            "HELD: missing admitted issuer proof"
        );
        revocations.refuse("exposure receipt", [digest])?;
        let (receipt, id): (ExposureReceipt, _) = retained(self.state, digest, EXPOSURE_SCHEMA)?;
        ensure!(
            receipt.policy_epoch == policy.epoch
                && receipt.observed_at_ms >= policy.effective_from_ms
                && receipt.observed_at_ms < policy.effective_until_ms,
            "HELD: prospective role revision cannot upgrade an old observation"
        );
        ensure!(
            matches!(
                receipt.meta_partition.as_str(),
                "train_search" | "meta_dev" | "qualification" | "sealed_final"
            ) && !receipt.family.trim().is_empty()
                && !receipt.producer.trim().is_empty()
                && !receipt.fidelity.trim().is_empty()
                && !receipt.original_role.trim().is_empty()
                && receipt.prior_exposure.is_some(),
            "HELD: missing or ambiguous exposure provenance"
        );
        revocations.refuse("exposure subject", [&*receipt.subject_sha256, &*id])?;
        ensure!(
            is_hex64(&receipt.subject_sha256)
                && self.state.read_blob(&receipt.subject_sha256)?.is_some(),
            "HELD: missing exposure subject"
        );
        if let Observation::Completed {
            policy_target_sha256: Some(target),
            ..
        } = &receipt.observation
        {
            ensure!(
                is_hex64(target) && self.state.read_blob(target)?.is_some(),
                "HELD: missing policy target"
            );
            revocations.refuse("policy target", [target.as_str()])?;
        }
        labels(&receipt.observation)?;
        ensure!(
            receipt.parents.len() <= MAX_NODES
                && receipt.parents.iter().collect::<BTreeSet<_>>().len() == receipt.parents.len(),
            "HELD: ambiguous parent roster"
        );
        active.insert(digest.into());
        for parent in &receipt.parents {
            self.visit(parent, policy, revocations, active, found, depth + 1)?;
        }
        active.remove(digest);
        if found.insert(digest.into(), receipt).is_some() {
            bail!("HELD: duplicate exposure receipt");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cyclic_ancestry_is_held_before_reuse_or_content_traversal() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/task-evidence/I04/scratch");
        std::fs::create_dir_all(&root).unwrap();
        let dir = tempfile::tempdir_in(root).unwrap();
        let state = State::at(dir.path());
        let digest = "a".repeat(64);
        let policy = AdmissionPolicy {
            schema: POLICY_SCHEMA.into(),
            issuer: "fixture".into(),
            epoch: 1,
            effective_from_ms: 0,
            effective_until_ms: 1000,
            evaluation_boundary_sha256: state.put_blob(b"future evaluation boundary").unwrap(),
            purposes: BTreeSet::from([Purpose::Audit]),
            receipts: BTreeSet::from([digest.clone()]),
            supersedes: None,
        };
        let verifier = AdmissionVerifier {
            state: &state,
            pin: PolicyPin {
                schema: ADOPTION_SCHEMA.into(),
                policy_sha256: "b".repeat(64),
                issuer: "fixture".into(),
                epoch: 1,
            },
        };
        // A valid content-addressed cycle would require a SHA preimage; nevertheless the
        // traversal itself refuses an active ancestor before considering memoized nodes.
        let error = verifier
            .visit(
                &digest,
                &policy,
                &Revocations::default(),
                &mut BTreeSet::from([digest.clone()]),
                &mut BTreeMap::new(),
                1,
            )
            .unwrap_err();
        assert!(error.to_string().contains("cyclic exposure ancestry"));
    }
}
