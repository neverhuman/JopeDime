use super::*;
use crate::wire_contract::CanonicalPayload;
use anyhow::{ensure, Result};
use serde::Serialize;
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Default)]
pub(super) struct MockStore {
    id: String,
    blobs: RefCell<BTreeMap<String, Vec<u8>>>,
    adoption: RefCell<Vec<u8>>,
    reads: RefCell<Vec<String>>,
    revoked: RefCell<BTreeSet<String>>,
}
pub(super) struct Snapshot(BTreeSet<String>);
impl RevocationSnapshot for Snapshot {
    fn refuse(&self, what: &str, digests: &[&str]) -> Result<()> {
        ensure!(
            digests.iter().all(|d| !self.0.contains(*d)),
            "revoked {what}"
        );
        Ok(())
    }
}
impl ArtifactStore for MockStore {
    type StoreId = str;
    type Revocations = Snapshot;
    fn store_id(&self) -> &str {
        &self.id
    }
    fn adopted_policy_bytes(&self) -> Result<Vec<u8>> {
        Ok(self.adoption.borrow().clone())
    }
    fn read_blob(&self, digest: &str) -> Result<Option<Vec<u8>>> {
        self.reads.borrow_mut().push(digest.into());
        Ok(self.blobs.borrow().get(digest).cloned())
    }
    fn revocations(&self) -> Result<Snapshot> {
        Ok(Snapshot(self.revoked.borrow().clone()))
    }
}
impl MockStore {
    fn put(&self, bytes: &[u8]) -> String {
        let digest = sha256_hex(bytes);
        self.blobs
            .borrow_mut()
            .insert(digest.clone(), bytes.to_vec());
        digest
    }
    fn retain<T: Serialize>(&self, value: &T) -> String {
        let payload =
            CanonicalPayload::parse_and_canonicalize(&serde_json::to_vec(value).unwrap()).unwrap();
        self.put(payload.as_bytes())
    }
    fn adopt(&self, receipts: &[String]) {
        let policy = AdmissionPolicy {
            schema: POLICY_SCHEMA.into(),
            issuer: "fixture-controller".into(),
            epoch: 1,
            effective_from_ms: 0,
            effective_until_ms: 1000,
            evaluation_boundary_sha256: self.put(b"separate future evaluation boundary"),
            purposes: purposes(),
            receipts: receipts.iter().cloned().collect(),
            supersedes: None,
        };
        let pin = PolicyPin {
            schema: ADOPTION_SCHEMA.into(),
            policy_sha256: self.retain(&policy),
            issuer: policy.issuer,
            epoch: 1,
        };
        let payload =
            CanonicalPayload::parse_and_canonicalize(&serde_json::to_vec(&pin).unwrap()).unwrap();
        *self.adoption.borrow_mut() = payload.as_bytes().to_vec();
    }
    fn never_read(&self, digests: &[String]) {
        for digest in digests {
            assert!(
                !self.reads.borrow().contains(digest),
                "protected payload opened: {digest}"
            );
        }
    }
}
fn purposes() -> BTreeSet<Purpose> {
    BTreeSet::from([
        Purpose::GlobalPolicyTraining,
        Purpose::EpisodeFit,
        Purpose::Audit,
    ])
}
fn receipt(store: &MockStore, subject: &str) -> ExposureReceipt {
    ExposureReceipt {
        schema: EXPOSURE_SCHEMA.into(),
        subject_sha256: subject.into(),
        family: "fixture".into(),
        meta_partition: "train_search".into(),
        original_role: "train".into(),
        within_task_view: "training".into(),
        fidelity: "measured".into(),
        producer: "fixture-controller".into(),
        policy_epoch: 1,
        observed_at_ms: 100,
        episode: Some("episode-a".into()),
        purposes: purposes(),
        prior_exposure: Some(vec!["train_fit".into()]),
        parents: vec![],
        observation: Observation::Completed {
            quality: Some(QualityLabel {
                metric: "rmse".into(),
                value: ExactValue {
                    f64_bits: format!("{:016x}", 0.5_f64.to_bits()),
                },
                unit: "target".into(),
                denominator: 3,
            }),
            elapsed_ms: Some(13),
            policy_target_sha256: Some(store.put(b"target payload")),
        },
    }
}

#[test]
fn train_fit_closure_produces_opaque_subject_bound_capability_and_labels() {
    let store = MockStore::default();
    let parent = store.put(b"parent FIT input");
    let parent_proof = store.retain(&receipt(&store, &parent));
    let subject = store.put(b"foundation and target outputs");
    let mut r = receipt(&store, &subject);
    r.parents.push(parent_proof.clone());
    let proof = store.retain(&r);
    store.adopt(&[parent_proof, proof.clone()]);
    let verifier = AdmissionVerifier::from_adopted_policy(&store).unwrap();
    let fit = verifier
        .admit_fitting_components(&subject, &proof, "episode-a", 500)
        .unwrap();
    assert_eq!(fit.admission().purpose(), Purpose::EpisodeFit);
    assert_eq!(fit.admission().subject_sha256(), subject);
    assert_eq!(fit.admission().receipt_sha256(), proof);
    assert_eq!(
        fit.admission()
            .labels()
            .quality
            .value()
            .unwrap()
            .denominator,
        3
    );
    assert_eq!(
        fit.admission().labels().runtime.value(),
        Some(&RuntimeLabel::Exact(13))
    );
    assert_eq!(fit.admission().labels().failure.value(), Some(&false));
    verifier.require_state(&store).unwrap();
    verifier.revalidate(fit.admission(), 501).unwrap();
}

#[test]
fn protected_ancestor_rejects_before_any_subject_or_target_payload_is_opened() {
    let store = MockStore::default();
    let parent = store.put(b"sealed ancestor subject");
    let mut sealed = receipt(&store, &parent);
    sealed.meta_partition = "sealed_final".into();
    sealed.original_role = "sealed".into();
    let parent_proof = store.retain(&sealed);
    let subject = store.put(b"innocuous derivative subject");
    let mut r = receipt(&store, &subject);
    r.parents.push(parent_proof.clone());
    let proof = store.retain(&r);
    store.adopt(&[parent_proof, proof.clone()]);
    let verifier = AdmissionVerifier::from_adopted_policy(&store).unwrap();
    for purpose in [Purpose::GlobalPolicyTraining, Purpose::EpisodeFit] {
        assert!(verifier
            .admit(&subject, &proof, purpose, Some("episode-a"), 500)
            .is_err());
    }
    assert!(verifier
        .admit_fitting_components(&subject, &proof, "episode-a", 500)
        .is_err());
    store.never_read(&[subject, parent, sha256_hex(b"target payload")]);
}

#[test]
fn all_ancestor_metadata_is_checked_before_payloads_even_for_audit() {
    for future in [false, true] {
        let store = MockStore::default();
        let parent = store.put(b"ancestor payload");
        let mut ancestor = receipt(&store, &parent);
        if future {
            ancestor.observed_at_ms = 600;
        } else {
            ancestor.purposes.remove(&Purpose::Audit);
        }
        let ancestor_proof = store.retain(&ancestor);
        let subject = store.put(b"root payload");
        let mut r = receipt(&store, &subject);
        r.parents.push(ancestor_proof.clone());
        let proof = store.retain(&r);
        store.adopt(&[ancestor_proof, proof.clone()]);
        let verifier = AdmissionVerifier::from_adopted_policy(&store).unwrap();
        assert!(verifier
            .admit(&subject, &proof, Purpose::Audit, None, 500)
            .is_err());
        store.never_read(&[subject, parent, sha256_hex(b"target payload")]);
    }
}

#[test]
fn generic_episode_keeps_inner_validation_but_fit_capability_refuses_it_without_reads() {
    let store = MockStore::default();
    let subject = store.put(b"inner validation subject");
    let mut r = receipt(&store, &subject);
    r.within_task_view = "inner_validation".into();
    let proof = store.retain(&r);
    store.adopt(std::slice::from_ref(&proof));
    let verifier = AdmissionVerifier::from_adopted_policy(&store).unwrap();
    assert!(verifier
        .admit_fitting_components(&subject, &proof, "episode-a", 500)
        .is_err());
    store.never_read(&[subject.clone(), sha256_hex(b"target payload")]);
    verifier
        .admit(
            &subject,
            &proof,
            Purpose::EpisodeFit,
            Some("episode-a"),
            500,
        )
        .unwrap();
}

#[test]
fn fitting_capability_refuses_wrong_partition_role_prior_and_episode_before_payloads() {
    for case in 0..4 {
        let store = MockStore::default();
        let subject = store.put(b"subject");
        let mut r = receipt(&store, &subject);
        match case {
            0 => r.meta_partition = "meta_dev".into(),
            1 => r.original_role = "train_query".into(),
            2 => r.prior_exposure = Some(vec!["judge".into()]),
            _ => r.episode = Some("another-episode".into()),
        }
        let proof = store.retain(&r);
        store.adopt(std::slice::from_ref(&proof));
        let verifier = AdmissionVerifier::from_adopted_policy(&store).unwrap();
        assert!(verifier
            .admit_fitting_components(&subject, &proof, "episode-a", 500)
            .is_err());
        store.never_read(&[subject, sha256_hex(b"target payload")]);
    }
}

#[test]
fn revocation_is_fresh_and_adopted_pin_cannot_change_under_existing_verifier() {
    let store = MockStore::default();
    let subject = store.put(b"subject");
    let proof = store.retain(&receipt(&store, &subject));
    store.adopt(std::slice::from_ref(&proof));
    let verifier = AdmissionVerifier::from_adopted_policy(&store).unwrap();
    let first = verifier
        .admit_fitting_components(&subject, &proof, "episode-a", 500)
        .unwrap();
    store.revoked.borrow_mut().insert(proof.clone());
    assert!(verifier.revalidate(first.admission(), 501).is_err());
    store.revoked.borrow_mut().clear();
    let mut changed: PolicyPin = serde_json::from_slice(&store.adoption.borrow()).unwrap();
    changed.issuer = "different-controller".into();
    let payload =
        CanonicalPayload::parse_and_canonicalize(&serde_json::to_vec(&changed).unwrap()).unwrap();
    *store.adoption.borrow_mut() = payload.as_bytes().to_vec();
    let error = verifier
        .revalidate(first.admission(), 501)
        .unwrap_err()
        .to_string();
    assert!(error.contains("controller policy changed"));
    let foreign = MockStore {
        id: "another-state".into(),
        ..MockStore::default()
    };
    assert!(verifier.require_state(&foreign).is_err());
}

#[test]
fn wrong_subject_is_refused_without_opening_root_or_target() {
    let store = MockStore::default();
    let subject = store.put(b"subject");
    let proof = store.retain(&receipt(&store, &subject));
    store.adopt(std::slice::from_ref(&proof));
    let verifier = AdmissionVerifier::from_adopted_policy(&store).unwrap();
    assert!(verifier
        .admit_fitting_components(&"a".repeat(64), &proof, "episode-a", 500)
        .is_err());
    store.never_read(&[subject, sha256_hex(b"target payload")]);
}
