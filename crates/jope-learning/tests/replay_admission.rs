//! Tiny retained proof chains through the production admission and ingress APIs. Policy pins
//! are synthetic controller configuration; no fixture installs a product policy.
use jope_learning::canon::{canonical_json, canonical_sha256};
use jope_learning::ledger::Ledger;
use jope_learning::replay::admission::*;
use jope_learning::{bank, pools::source, state::State};
use jope_records::wire_contract::CanonicalPayload;
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

const NOW: u64 = 500;

#[path = "support/optional_dope.rs"]
mod optional_dope;

fn purposes() -> BTreeSet<Purpose> {
    [
        Purpose::GlobalPolicyTraining,
        Purpose::RepresentationLearning,
        Purpose::RetrievalIndexUpdate,
        Purpose::Distillation,
        Purpose::EpisodeFit,
        Purpose::Audit,
    ]
    .into_iter()
    .collect()
}
fn retain<T: serde::Serialize>(state: &State, value: &T) -> String {
    let payload =
        CanonicalPayload::parse_and_canonicalize(&serde_json::to_vec(value).unwrap()).unwrap();
    state.put_blob(payload.as_bytes()).unwrap()
}
fn scratch() -> PathBuf {
    let target =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/task-evidence/I04/scratch");
    std::fs::create_dir_all(&target).unwrap();
    tempfile::Builder::new()
        .prefix("replay-proof-")
        .tempdir_in(target)
        .unwrap()
        .keep()
}
fn quality() -> QualityLabel {
    QualityLabel {
        metric: "rmse".into(),
        value: ExactValue {
            f64_bits: format!("{:016x}", 0.5_f64.to_bits()),
        },
        unit: "target".into(),
        denominator: 3,
    }
}
fn receipt(state: &State, subject: &str) -> ExposureReceipt {
    ExposureReceipt {
        schema: EXPOSURE_SCHEMA.into(),
        subject_sha256: subject.into(),
        family: "fixture-family".into(),
        meta_partition: "train_search".into(),
        original_role: "train".into(),
        within_task_view: "training".into(),
        fidelity: "measured_cpu_cv".into(),
        producer: "fixture-controller-verifier".into(),
        policy_epoch: 1,
        observed_at_ms: 100,
        episode: Some("episode-a".into()),
        purposes: purposes(),
        prior_exposure: Some(vec![]),
        parents: vec![],
        observation: Observation::Completed {
            quality: Some(quality()),
            elapsed_ms: Some(13),
            policy_target_sha256: Some(state.put_blob(b"measured policy target").unwrap()),
        },
    }
}
fn policy(state: &State, receipts: &[String], epoch: u64, from: u64) -> PolicyPin {
    let policy = AdmissionPolicy {
        schema: POLICY_SCHEMA.into(),
        issuer: "fixture-controller".into(),
        epoch,
        effective_from_ms: from,
        effective_until_ms: 1000,
        evaluation_boundary_sha256: state
            .put_blob(format!("future-boundary-{epoch}-{from}").as_bytes())
            .unwrap(),
        purposes: purposes(),
        receipts: receipts.iter().cloned().collect(),
        supersedes: None,
    };
    PolicyPin {
        schema: ADOPTION_SCHEMA.into(),
        policy_sha256: retain(state, &policy),
        issuer: policy.issuer,
        epoch,
    }
}
fn bind(state: &State, pin: PolicyPin) -> anyhow::Result<AdmissionVerifier<'_>> {
    // Test controller installs the trusted configuration. Production exposes no installer and
    // no candidate-facing constructor taking a PolicyPin.
    let payload =
        CanonicalPayload::parse_and_canonicalize(&serde_json::to_vec(&pin).unwrap()).unwrap();
    let path = state.root().join(ADOPTION_PATH);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, payload.as_bytes()).unwrap();
    AdmissionVerifier::from_adopted_policy(state)
}
fn verifier<'a>(state: &'a State, receipts: &[String]) -> AdmissionVerifier<'a> {
    bind(state, policy(state, receipts, 1, 0)).unwrap()
}
fn refused(result: anyhow::Result<ReplayAdmission>, expected: &str) {
    let message = format!("{:#}", result.unwrap_err());
    println!("observed refusal: {message}");
    assert!(
        message.contains(expected),
        "expected {expected:?}: {message}"
    );
}

#[test]
fn authenticated_train_chain_has_independent_quality_runtime_failure_and_policy_labels() {
    let dir = scratch();
    let state = State::at(&dir);
    let parent = state.put_blob(b"train parent").unwrap();
    let parent_proof = retain(&state, &receipt(&state, &parent));
    let subject = state.put_blob(b"train derivative").unwrap();
    let mut r = receipt(&state, &subject);
    r.parents.push(parent_proof.clone());
    let proof = retain(&state, &r);
    let verifier = verifier(&state, &[parent_proof, proof.clone()]);
    for purpose in [
        Purpose::GlobalPolicyTraining,
        Purpose::RepresentationLearning,
        Purpose::RetrievalIndexUpdate,
        Purpose::Distillation,
    ] {
        let admitted = verifier
            .admit(&subject, &proof, purpose, None, NOW)
            .unwrap();
        state
            .put_blob(&serde_json::to_vec(&admitted).unwrap())
            .unwrap();
        assert_eq!(admitted.labels().quality.value(), Some(&quality()));
        assert_eq!(
            admitted.labels().runtime.value(),
            Some(&RuntimeLabel::Exact(13))
        );
        assert_eq!(admitted.labels().failure.value(), Some(&false));
        assert!(admitted.labels().policy.value().is_some());
        assert_eq!(admitted.purpose(), purpose);
        assert_eq!(admitted.subject_sha256(), subject);
        assert!(serde_json::to_string(&admitted)
            .unwrap()
            .contains(ADMISSION_SCHEMA));
    }
}

#[test]
fn forged_role_boolean_signature_and_unadmitted_proof_cannot_issue_capability() {
    let dir = scratch();
    let state = State::at(&dir);
    let subject = state.put_blob(b"sealed original").unwrap();
    let mut protected = receipt(&state, &subject);
    protected.original_role = "sealed".into();
    let proof = retain(&state, &protected);
    let verifier = verifier(&state, std::slice::from_ref(&proof));
    let mut forged = serde_json::to_value(&protected).unwrap();
    forged["original_role"] = json!("train");
    forged["eligible"] = json!(true);
    forged["independent_verifier_passed"] = json!(true);
    forged["signature"] = json!({"signer_role":"offline_verifier","signature_hex":"a".repeat(64)});
    assert!(serde_json::from_value::<ExposureReceipt>(forged.clone()).is_err());
    // Even a well-formed forgery with all tempting fields removed lacks issuer admission.
    forged.as_object_mut().unwrap().remove("eligible");
    forged
        .as_object_mut()
        .unwrap()
        .remove("independent_verifier_passed");
    forged.as_object_mut().unwrap().remove("signature");
    let forged_proof = retain(&state, &forged);
    refused(
        verifier.admit(
            &subject,
            &forged_proof,
            Purpose::GlobalPolicyTraining,
            None,
            NOW,
        ),
        "issuer proof",
    );
    refused(
        verifier.admit(&subject, &proof, Purpose::GlobalPolicyTraining, None, NOW),
        "protected",
    );
    let mut pin = policy(&state, &[proof], 1, 0);
    pin.issuer = "copied issuer".into();
    assert!(bind(&state, pin).is_err());
}

#[test]
fn sealed_and_judge_summaries_embeddings_and_selections_inherit_every_parent() {
    let dir = scratch();
    let state = State::at(&dir);
    for role in ["sealed", "judge", "meta_test"] {
        for derivative in ["summary", "embedding", "selection"] {
            let parent = state.put_blob(format!("{role} parent").as_bytes()).unwrap();
            let mut r = receipt(&state, &parent);
            r.original_role = role.into();
            let parent_proof = retain(&state, &r);
            let subject = state
                .put_blob(format!("{role} {derivative}").as_bytes())
                .unwrap();
            let mut r = receipt(&state, &subject);
            r.parents.push(parent_proof.clone());
            let proof = retain(&state, &r);
            let verifier = verifier(&state, &[parent_proof, proof.clone()]);
            for purpose in [
                Purpose::GlobalPolicyTraining,
                Purpose::RepresentationLearning,
                Purpose::RetrievalIndexUpdate,
                Purpose::Distillation,
            ] {
                refused(
                    verifier.admit(&subject, &proof, purpose, None, NOW),
                    "protected",
                );
            }
        }
    }
}

#[test]
fn test_adaptive_representation_and_copied_role_with_exposure_are_refused() {
    let dir = scratch();
    let state = State::at(&dir);
    let subject = state.put_blob(b"historical adaptive result").unwrap();
    for original in ["test_adaptive", "train"] {
        let mut r = receipt(&state, &subject);
        r.original_role = original.into();
        r.prior_exposure = Some(vec!["test_adaptive".into()]);
        let proof = retain(&state, &r);
        let verifier = verifier(&state, std::slice::from_ref(&proof));
        refused(
            verifier.admit(&subject, &proof, Purpose::RepresentationLearning, None, NOW),
            "protected",
        );
    }
}

#[test]
fn episode_fit_permission_does_not_grant_global_use_or_outer_label_access() {
    let dir = scratch();
    let state = State::at(&dir);
    let subject = state.put_blob(b"qualification training view").unwrap();
    let mut r = receipt(&state, &subject);
    r.meta_partition = "qualification".into();
    let proof = retain(&state, &r);
    let verifier = verifier(&state, std::slice::from_ref(&proof));
    verifier
        .admit(
            &subject,
            &proof,
            Purpose::EpisodeFit,
            Some("episode-a"),
            NOW,
        )
        .unwrap();
    refused(
        verifier.admit(
            &subject,
            &proof,
            Purpose::EpisodeFit,
            Some("episode-b"),
            NOW,
        ),
        "episode",
    );
    refused(
        verifier.admit(&subject, &proof, Purpose::GlobalPolicyTraining, None, NOW),
        "protected",
    );
    r.within_task_view = "judge_only_outer_labels".into();
    let proof = retain(&state, &r);
    let verifier = bind(&state, policy(&state, std::slice::from_ref(&proof), 1, 0)).unwrap();
    refused(
        verifier.admit(
            &subject,
            &proof,
            Purpose::EpisodeFit,
            Some("episode-a"),
            NOW,
        ),
        "episode",
    );
}

#[test]
fn timeout_supplies_only_censored_cost_and_valid_failure_without_quality_or_policy() {
    let dir = scratch();
    let state = State::at(&dir);
    let subject = state.put_blob(b"timed out attempt").unwrap();
    for bound in [Some(90_000), None] {
        let mut r = receipt(&state, &subject);
        r.observation = Observation::Timeout {
            lower_bound_ms: bound,
            cause: TimeoutCause::ProgramDeadline,
        };
        let proof = retain(&state, &r);
        let verifier = verifier(&state, std::slice::from_ref(&proof));
        let a = verifier
            .admit(&subject, &proof, Purpose::GlobalPolicyTraining, None, NOW)
            .unwrap();
        state.put_blob(&serde_json::to_vec(&a).unwrap()).unwrap();
        assert!(a.labels().quality.value().is_none());
        assert!(a.labels().policy.value().is_none());
        assert_eq!(a.labels().failure.value(), Some(&true));
        assert_eq!(
            a.labels().runtime.value(),
            bound.as_ref().map(|_| &RuntimeLabel::RightCensored(90_000))
        );
        if bound.is_some() {
            assert!(
                matches!(&a.labels().runtime, Label::Available { unit, denominator: 1, .. } if unit == "ms")
            );
        }
    }
}

#[test]
fn proposal_has_no_fabricated_labels_and_failure_is_not_quality() {
    let dir = scratch();
    let state = State::at(&dir);
    let subject = state.put_blob(b"unmeasured proposal").unwrap();
    let mut r = receipt(&state, &subject);
    r.observation = Observation::Proposed;
    let proof = retain(&state, &r);
    let verifier = verifier(&state, std::slice::from_ref(&proof));
    let a = verifier
        .admit(&subject, &proof, Purpose::GlobalPolicyTraining, None, NOW)
        .unwrap();
    assert!(
        a.labels().quality.value().is_none()
            && a.labels().runtime.value().is_none()
            && a.labels().failure.value().is_none()
            && a.labels().policy.value().is_none()
    );
    r.observation = Observation::Failed {
        elapsed_ms: Some(7),
    };
    let proof = retain(&state, &r);
    let verifier = bind(&state, policy(&state, std::slice::from_ref(&proof), 1, 0)).unwrap();
    let a = verifier
        .admit(&subject, &proof, Purpose::GlobalPolicyTraining, None, NOW)
        .unwrap();
    assert_eq!(a.labels().failure.value(), Some(&true));
    assert_eq!(a.labels().runtime.value(), Some(&RuntimeLabel::Exact(7)));
    assert!(a.labels().quality.value().is_none() && a.labels().policy.value().is_none());
}

#[test]
fn absent_unknown_altered_and_self_referential_ancestry_remain_held() {
    let dir = scratch();
    let state = State::at(&dir);
    let subject = state.put_blob(b"ancestry subject").unwrap();
    let absent = "a".repeat(64);
    let mut r = receipt(&state, &subject);
    r.parents = vec![absent.clone()];
    let proof = retain(&state, &r);
    let verifier = verifier(&state, &[proof.clone(), absent.clone()]);
    refused(
        verifier.admit(&subject, &proof, Purpose::GlobalPolicyTraining, None, NOW),
        "missing exposure proof",
    );
    std::fs::write(state.blob_path(&absent), b"{\"parents\":[\"self\"]}").unwrap();
    refused(
        verifier.admit(&subject, &proof, Purpose::GlobalPolicyTraining, None, NOW),
        "corrupt",
    );
    r.parents.clear();
    r.prior_exposure = None;
    let proof = retain(&state, &r);
    let verifier = bind(&state, policy(&state, std::slice::from_ref(&proof), 1, 0)).unwrap();
    refused(
        verifier.admit(&subject, &proof, Purpose::GlobalPolicyTraining, None, NOW),
        "ambiguous",
    );
}

#[test]
fn ancestor_revocation_invalidates_cached_admission_and_only_its_descendants() {
    let dir = scratch();
    let state = State::at(&dir);
    let parent = state.put_blob(b"revoked parent").unwrap();
    let p = retain(&state, &receipt(&state, &parent));
    let child = state.put_blob(b"child").unwrap();
    let mut r = receipt(&state, &child);
    r.parents.push(p.clone());
    let c = retain(&state, &r);
    let other = state.put_blob(b"unrelated").unwrap();
    let o = retain(&state, &receipt(&state, &other));
    let verifier = verifier(&state, &[p.clone(), c.clone(), o.clone()]);
    let admitted = verifier
        .admit(&child, &c, Purpose::RetrievalIndexUpdate, None, NOW)
        .unwrap();
    Ledger::open(state.root(), "revocations")
        .append(
            &[json!({"schema":"jope.revocation-record/v1",
        "node":"parent", "sha256":p, "revoked_node":"parent", "issuer":"fixture-controller",
        "cause":"fixture revoked ancestor", "effective_utc":"2026-10-18T00:00:00Z", "sequence":1,
        "claims_sha256":"b".repeat(64)})],
            json!({"fixture":true}),
        )
        .unwrap();
    refused(verifier.revalidate(&admitted, NOW), "RevokedEvidence");
    verifier
        .admit(&other, &o, Purpose::RetrievalIndexUpdate, None, NOW)
        .unwrap();
}

#[test]
fn prospective_revision_never_relabels_old_observations_or_erasures_of_exposure() {
    let dir = scratch();
    let state = State::at(&dir);
    let subject = state.put_blob(b"old sealed claim").unwrap();
    let mut r = receipt(&state, &subject);
    r.original_role = "sealed".into();
    let old = retain(&state, &r);
    let before = state.read_blob(&old).unwrap().unwrap();
    let pin = policy(&state, std::slice::from_ref(&old), 2, 200);
    let verifier = bind(&state, pin).unwrap();
    refused(
        verifier.admit(&subject, &old, Purpose::RepresentationLearning, None, NOW),
        "prospective",
    );
    r.policy_epoch = 2;
    r.observed_at_ms = 250;
    r.original_role = "train".into();
    r.prior_exposure = Some(vec!["sealed".into()]);
    let changed = retain(&state, &r);
    let verifier = bind(
        &state,
        policy(&state, std::slice::from_ref(&changed), 2, 200),
    )
    .unwrap();
    refused(
        verifier.admit(
            &subject,
            &changed,
            Purpose::RepresentationLearning,
            None,
            NOW,
        ),
        "protected",
    );
    assert_eq!(state.read_blob(&old).unwrap().unwrap(), before);
    // Independently admitted future train evidence has a new identity and boundary.
    let fresh = state.put_blob(b"new future train input").unwrap();
    r.subject_sha256 = fresh.clone();
    r.prior_exposure = Some(vec![]);
    let proof = retain(&state, &r);
    let verifier = bind(&state, policy(&state, std::slice::from_ref(&proof), 2, 200)).unwrap();
    verifier
        .admit(&fresh, &proof, Purpose::RepresentationLearning, None, NOW)
        .unwrap();
}

#[test]
fn versioned_coverage_ingress_binds_actual_document_before_global_use() {
    let dir = scratch();
    let state = State::at(&dir);
    let doc: Value =
        serde_json::from_str(include_str!("fixtures/replay-admission-v2/coverage.json")).unwrap();
    let subject = state.put_blob(&canonical_json(&doc).unwrap()).unwrap();
    let mut r = receipt(&state, &subject);
    r.observation = Observation::Proposed;
    let proof = retain(&state, &r);
    let verifier = verifier(&state, std::slice::from_ref(&proof));
    let (rows, _, _) = source::coverage_rows_for(
        &doc,
        &"b".repeat(64),
        &verifier,
        &proof,
        Purpose::RetrievalIndexUpdate,
        NOW,
    )
    .unwrap();
    assert_eq!(rows.len(), 1);
    let mut forged = doc.clone();
    forged["label"] = json!("forged train");
    assert!(source::coverage_rows_for(
        &forged,
        &"b".repeat(64),
        &verifier,
        &proof,
        Purpose::RetrievalIndexUpdate,
        NOW
    )
    .is_err());
    assert!(source::coverage_rows_for(
        &doc,
        &"b".repeat(64),
        &verifier,
        &proof,
        Purpose::GlobalPolicyTraining,
        NOW
    )
    .is_err());
}

#[test]
fn global_bank_export_requires_subject_bound_admission_and_filters_before_ranking() {
    let dir = scratch();
    let state = State::at(&dir);
    let entry: bank::BankEntry =
        serde_json::from_str(include_str!("fixtures/replay-admission-v2/bank-entry.json")).unwrap();
    let event = bank::BankEvent {
        schema: bank::EVENT_SCHEMA.into(),
        entry: entry.clone(),
        entry_sha256: canonical_sha256(&entry).unwrap(),
        ts_unix: 100,
    };
    Ledger::open(state.root(), bank::LEDGER)
        .append(
            &[serde_json::to_value(event).unwrap()],
            json!({"fixture":true}),
        )
        .unwrap();
    let subject = state.put_blob(&canonical_json(&entry).unwrap()).unwrap();
    let proof = retain(&state, &receipt(&state, &subject));
    let verifier = verifier(&state, std::slice::from_ref(&proof));
    let proofs = BTreeMap::from([(subject.clone(), proof.clone())]);
    assert!(bank::export_for(
        &state,
        &verifier,
        Purpose::RetrievalIndexUpdate,
        &BTreeMap::new(),
        NOW
    )
    .is_err());
    let exported = bank::export_for(
        &state,
        &verifier,
        Purpose::RetrievalIndexUpdate,
        &proofs,
        NOW,
    )
    .unwrap();
    assert_eq!(exported.entries().len(), 1);
    let mut restricted = receipt(&state, &subject);
    restricted.original_role = "sealed".into();
    let proof = retain(&state, &restricted);
    let verifier = bind(&state, policy(&state, std::slice::from_ref(&proof), 1, 0)).unwrap();
    assert!(bank::export_for(
        &state,
        &verifier,
        Purpose::RetrievalIndexUpdate,
        &BTreeMap::from([(subject, proof)]),
        NOW
    )
    .is_err());
}

#[test]
fn infrastructure_or_unknown_timeout_does_not_fabricate_program_failure() {
    let dir = scratch();
    let state = State::at(&dir);
    let subject = state.put_blob(b"missing program outcome").unwrap();
    for cause in [TimeoutCause::Infrastructure, TimeoutCause::Unknown] {
        let mut r = receipt(&state, &subject);
        r.observation = Observation::Timeout {
            lower_bound_ms: None,
            cause,
        };
        let proof = retain(&state, &r);
        let verifier = verifier(&state, std::slice::from_ref(&proof));
        let a = verifier
            .admit(&subject, &proof, Purpose::GlobalPolicyTraining, None, NOW)
            .unwrap();
        assert!(a.labels().failure.value().is_none() && a.labels().runtime.value().is_none());
        assert!(a.labels().quality.value().is_none() && a.labels().policy.value().is_none());
    }
}

#[test]
fn wrong_subject_purpose_epoch_version_and_expired_policy_are_held() {
    let dir = scratch();
    let state = State::at(&dir);
    let subject = state.put_blob(b"train observation").unwrap();
    let mut r = receipt(&state, &subject);
    r.purposes = BTreeSet::from([Purpose::Audit]);
    let proof = retain(&state, &r);
    let verifier = verifier(&state, std::slice::from_ref(&proof));
    refused(
        verifier.admit(&subject, &proof, Purpose::RetrievalIndexUpdate, None, NOW),
        "purpose",
    );
    refused(
        verifier.admit(&"b".repeat(64), &proof, Purpose::Audit, None, NOW),
        "another subject",
    );
    refused(
        verifier.admit(&subject, &proof, Purpose::Audit, None, 1000),
        "not effective",
    );
    r.schema = "jope.replay-exposure/v99".into();
    let proof = retain(&state, &r);
    let verifier = bind(&state, policy(&state, std::slice::from_ref(&proof), 1, 0)).unwrap();
    refused(
        verifier.admit(&subject, &proof, Purpose::Audit, None, NOW),
        "version",
    );
}

#[test]
fn malformed_or_noncanonical_claim_cannot_be_made_authoritative_by_listing_its_digest() {
    let dir = scratch();
    let state = State::at(&dir);
    let subject = state.put_blob(b"parser target").unwrap();
    let valid = serde_json::to_string(&receipt(&state, &subject)).unwrap();
    for raw in [
        format!(" {valid}"),
        valid.replacen("\"parents\":[]", "\"parents\":[],\"parents\":[]", 1),
    ] {
        let proof = state.put_blob(raw.as_bytes()).unwrap();
        let verifier = verifier(&state, std::slice::from_ref(&proof));
        assert!(verifier
            .admit(&subject, &proof, Purpose::Audit, None, NOW)
            .is_err());
    }
}

#[test]
fn diamond_ancestry_is_valid_but_missing_branch_duplicate_parent_and_excess_depth_are_held() {
    let dir = scratch();
    let state = State::at(&dir);
    let subject = state.put_blob(b"shared input").unwrap();
    let leaf = retain(&state, &receipt(&state, &subject));
    let mut r = receipt(&state, &subject);
    r.parents.push(leaf.clone());
    r.producer = "left".into();
    let left = retain(&state, &r);
    r.producer = "right".into();
    let right = retain(&state, &r);
    r.producer = "diamond".into();
    r.parents = vec![left.clone(), right.clone()];
    let root = retain(&state, &r);
    let verifier = verifier(
        &state,
        &[leaf.clone(), left.clone(), right.clone(), root.clone()],
    );
    verifier
        .admit(&subject, &root, Purpose::RetrievalIndexUpdate, None, NOW)
        .unwrap();
    let verifier = bind(
        &state,
        policy(&state, &[leaf.clone(), left.clone(), root.clone()], 1, 0),
    )
    .unwrap();
    refused(
        verifier.admit(&subject, &root, Purpose::RetrievalIndexUpdate, None, NOW),
        "issuer proof",
    );
    r.parents = vec![leaf.clone(), leaf.clone()];
    let root = retain(&state, &r);
    let verifier = bind(&state, policy(&state, &[leaf.clone(), root.clone()], 1, 0)).unwrap();
    refused(
        verifier.admit(&subject, &root, Purpose::RetrievalIndexUpdate, None, NOW),
        "ambiguous parent",
    );
    let mut proofs = vec![leaf];
    for n in 0..34 {
        r.parents = vec![proofs.last().unwrap().clone()];
        r.producer = format!("depth-{n}");
        proofs.push(retain(&state, &r));
    }
    let verifier = bind(&state, policy(&state, &proofs, 1, 0)).unwrap();
    refused(
        verifier.admit(
            &subject,
            proofs.last().unwrap(),
            Purpose::RetrievalIndexUpdate,
            None,
            NOW,
        ),
        "bound",
    );
}

#[test]
fn candidate_policy_json_and_role_strings_cannot_install_controller_adoption() {
    let dir = scratch();
    let state = State::at(&dir);
    let subject = state.put_blob(b"candidate-owned document").unwrap();
    let proof = retain(&state, &receipt(&state, &subject));
    // Complete, canonical policy bytes in the blob store still do not install authority.
    let _candidate_pin = policy(&state, &[proof], 1, 0);
    let error = match AdmissionVerifier::from_adopted_policy(&state) {
        Ok(_) => panic!("candidate bytes installed controller authority"),
        Err(error) => error,
    };
    assert!(error
        .to_string()
        .contains("no controller-adopted replay policy"));
}

#[test]
fn successor_policy_requires_new_boundary_and_invalidates_stale_verifier() {
    let dir = scratch();
    let state = State::at(&dir);
    let subject = state.put_blob(b"epoch-one train input").unwrap();
    let first = retain(&state, &receipt(&state, &subject));
    let pin = policy(&state, std::slice::from_ref(&first), 1, 0);
    let old = bind(&state, pin.clone()).unwrap();
    let old_policy: AdmissionPolicy =
        serde_json::from_slice(&state.read_blob(&pin.policy_sha256).unwrap().unwrap()).unwrap();
    let fresh = state.put_blob(b"epoch-two train input").unwrap();
    let mut r = receipt(&state, &fresh);
    r.policy_epoch = 2;
    r.observed_at_ms = 250;
    let proof = retain(&state, &r);
    let mut successor = old_policy.clone();
    successor.epoch = 2;
    successor.effective_from_ms = 200;
    successor.supersedes = Some(pin.policy_sha256);
    successor.receipts = BTreeSet::from([proof.clone()]);
    let invalid_pin = PolicyPin {
        schema: ADOPTION_SCHEMA.into(),
        issuer: successor.issuer.clone(),
        epoch: 2,
        policy_sha256: retain(&state, &successor),
    };
    assert!(bind(&state, invalid_pin).is_err());
    successor.evaluation_boundary_sha256 =
        state.put_blob(b"new future evaluation boundary").unwrap();
    let valid_pin = PolicyPin {
        schema: ADOPTION_SCHEMA.into(),
        issuer: successor.issuer.clone(),
        epoch: 2,
        policy_sha256: retain(&state, &successor),
    };
    let current = bind(&state, valid_pin).unwrap();
    current
        .admit(&fresh, &proof, Purpose::RepresentationLearning, None, NOW)
        .unwrap();
    refused(
        old.admit(&subject, &first, Purpose::RepresentationLearning, None, NOW),
        "controller policy changed",
    );
}
