use super::labels::labels;
use super::*;
use crate::wire_contract::CanonicalPayload;
use anyhow::{bail, ensure, Context, Result};
use serde::de::DeserializeOwned;
use std::collections::{BTreeMap, BTreeSet};

/// A verified, immutable revocation snapshot for one admission attempt.
/// Implementations must fail closed on corrupt or unavailable revocation state.
pub trait RevocationSnapshot {
    fn refuse(&self, what: &str, digests: &[&str]) -> Result<()>;
}

/// Trusted controller-owned access to adopted configuration and retained evidence.
/// Implementations must never source adoption bytes, store identity or revocations from
/// a candidate/Fusion request. The verifier neither adopts policy nor authenticates a new issuer.
pub trait ArtifactStore {
    type StoreId: Eq + ?Sized;
    type Revocations: RevocationSnapshot;

    fn store_id(&self) -> &Self::StoreId;
    fn adopted_policy_bytes(&self) -> Result<Vec<u8>>;
    fn read_blob(&self, digest: &str) -> Result<Option<Vec<u8>>>;
    fn revocations(&self) -> Result<Self::Revocations>;
}

/// All content references use the existing State byte-SHA namespace. Each new payload is also
/// checked by I02's strict canonical codec and has its distinct domain-separated identity.
fn retained<T: DeserializeOwned, S: ArtifactStore + ?Sized>(
    state: &S,
    digest: &str,
    schema: &str,
) -> Result<(T, String)> {
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
pub struct AdmissionVerifier<'a, S: ArtifactStore + ?Sized> {
    state: &'a S,
    pin: PolicyPin,
}
impl<'a, S: ArtifactStore + ?Sized> AdmissionVerifier<'a, S> {
    /// Only the controller's installed adoption pin is read. There is deliberately no public
    /// constructor accepting a candidate's pin, issuer, verifier Boolean or policy pathname.
    pub fn from_adopted_policy(state: &'a S) -> Result<Self> {
        let pin = Self::adopted_pin(state)?;
        let verifier = Self { state, pin };
        verifier.policy()?;
        Ok(verifier)
    }

    fn adopted_pin(state: &S) -> Result<PolicyPin> {
        let bytes = state
            .adopted_policy_bytes()
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

    pub fn require_state(&self, state: &S) -> Result<()> {
        ensure!(
            self.state.store_id() == state.store_id(),
            "HELD: verifier belongs to a different state"
        );
        Ok(())
    }

    /// Preserve the existing general purpose/episode rules, including inner validation.
    pub fn admit(
        &self,
        subject_sha256: &str,
        receipt_sha256: &str,
        purpose: Purpose,
        episode: Option<&str>,
        at_ms: u64,
    ) -> Result<ReplayAdmission> {
        self.admit_inner(
            subject_sha256,
            receipt_sha256,
            purpose,
            episode,
            at_ms,
            false,
        )
    }

    /// Admit foundation/target fitting components only after the complete receipt closure
    /// establishes train_search, train/train_fit, training view and train-only prior exposure.
    /// No subject or target payload is opened until every ancestor passes these checks.
    pub fn admit_fitting_components(
        &self,
        subject_sha256: &str,
        receipt_sha256: &str,
        episode: &str,
        at_ms: u64,
    ) -> Result<FitAdmission> {
        Ok(FitAdmission {
            admission: self.admit_inner(
                subject_sha256,
                receipt_sha256,
                Purpose::EpisodeFit,
                Some(episode),
                at_ms,
                true,
            )?,
        })
    }

    fn admit_inner(
        &self,
        subject_sha256: &str,
        receipt_sha256: &str,
        purpose: Purpose,
        episode: Option<&str>,
        at_ms: u64,
        fitting_only: bool,
    ) -> Result<ReplayAdmission> {
        let (policy, policy_id) = self.policy()?;
        ensure!(
            (policy.effective_from_ms..policy.effective_until_ms).contains(&at_ms)
                && policy.purposes.contains(&purpose),
            "HELD: purpose or policy epoch not effective"
        );
        let revocations = self.state.revocations()?;
        revocations.refuse("replay policy", &[&self.pin.policy_sha256, &policy_id])?;
        revocations.refuse(
            "future evaluation boundary",
            &[policy.evaluation_boundary_sha256.as_str()],
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
                ensure!(
                    receipt.meta_partition == "train_search"
                        && matches!(receipt.original_role.as_str(), "train" | "train_fit")
                        && matches!(
                            receipt.within_task_view.as_str(),
                            "training" | "inner_validation"
                        )
                        && receipt.prior_exposure.as_ref().is_some_and(|roles| roles
                            .iter()
                            .all(|r| matches!(r.as_str(), "train" | "train_fit"))),
                    "HELD: protected role/view/exposure ancestry cannot enter global learning/retrieval"
                );
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
            if fitting_only {
                ensure!(
                    receipt.meta_partition == "train_search"
                        && matches!(receipt.original_role.as_str(), "train" | "train_fit")
                        && receipt.within_task_view == "training"
                        && receipt.prior_exposure.as_ref().is_some_and(|roles| roles
                            .iter()
                            .all(|r| matches!(r.as_str(), "train" | "train_fit"))),
                    "HELD: fitting components require FIT-only training ancestry"
                );
            }
        }
        // The entire metadata closure is permitted before ANY subject/target payload is read.
        for receipt in closure.values() {
            ensure!(
                self.state.read_blob(&receipt.subject_sha256)?.is_some(),
                "HELD: missing exposure subject"
            );
            if let Observation::Completed {
                policy_target_sha256: Some(target),
                ..
            } = &receipt.observation
            {
                ensure!(
                    self.state.read_blob(target)?.is_some(),
                    "HELD: missing policy target"
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
                subjects: closure.values().map(|r| r.subject_sha256.clone()).collect(),
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
        revocations: &S::Revocations,
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
        revocations.refuse("exposure receipt", &[digest])?;
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
        revocations.refuse("exposure subject", &[&receipt.subject_sha256, &id])?;
        ensure!(
            is_hex64(&receipt.subject_sha256),
            "HELD: missing exposure subject"
        );
        if let Observation::Completed {
            policy_target_sha256: Some(target),
            ..
        } = &receipt.observation
        {
            ensure!(is_hex64(target), "HELD: missing policy target");
            revocations.refuse("policy target", &[target.as_str()])?;
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
    use crate::replay_admission::tests::MockStore;

    #[test]
    fn cyclic_ancestry_is_held_before_reuse_or_content_traversal() {
        let state = MockStore::default();
        let digest = "a".repeat(64);
        let policy = AdmissionPolicy {
            schema: POLICY_SCHEMA.into(),
            issuer: "fixture".into(),
            epoch: 1,
            effective_from_ms: 0,
            effective_until_ms: 1000,
            evaluation_boundary_sha256: "c".repeat(64),
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
        // A content-addressed cycle requires a SHA preimage. The traversal must still
        // reject an active ancestor before reading any proof or memoized content.
        let error = verifier
            .visit(
                &digest,
                &policy,
                &state.revocations().unwrap(),
                &mut BTreeSet::from([digest.clone()]),
                &mut BTreeMap::new(),
                1,
            )
            .unwrap_err();
        assert!(error.to_string().contains("cyclic exposure ancestry"));
    }
}
