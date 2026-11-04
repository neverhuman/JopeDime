use super::*;

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
    let doc: Value = serde_json::from_str(include_str!(
        "../fixtures/replay-admission-v2/coverage.json"
    ))
    .unwrap();
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
    let entry: bank::BankEntry = serde_json::from_str(include_str!(
        "../fixtures/replay-admission-v2/bank-entry.json"
    ))
    .unwrap();
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
