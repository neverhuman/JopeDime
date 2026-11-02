use super::*;

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
