use super::*;
fn lineage_fixture() -> DescendantLineagePolicy {
    use crate::native_loop::lineage::*;
    fn put(b: &mut LineageBundle, schema: &str, value: &Value) -> String {
        let bytes = canonical_bytes(value).unwrap();
        let id = object_id(schema, &bytes).unwrap();
        b.evidence
            .insert(id.clone(), String::from_utf8(bytes).unwrap());
        id
    }
    let h = |s: &str| sha(s.as_bytes());
    let protocol = serde_json::json!({"objective_sha256":h("objective"),"recipe_sha256":h("recipe"),"architecture_sha256":h("architecture"),"vocabulary_sha256":h("vocabulary"),"support_sha256":h("support")});
    let mut bundle: LineageBundle = serde_json::from_value(serde_json::json!({
        "lineage":{"schema":LINEAGE_SCHEMA,"anchor":{"checkpoint_sha256":h("anchor"),"weights_sha256":h("anchor weights"),"optimizer_step":60000,"objective_sha256":null,"committed_updates":null,"architecture_sha256":h("architecture"),"vocabulary_sha256":h("vocabulary"),"support_sha256":h("support")},"protocol":protocol,"edges":[]},"evidence":{}})).unwrap();
    let epoch = OptimizerEpoch {
        schema: EPOCH_SCHEMA.into(),
        origin_checkpoint_sha256: h("anchor"),
        reset_nonce_sha256: h("reset"),
        recipe_sha256: h("recipe"),
    };
    for i in 0..2 {
        let parent = if i == 0 {
            h("anchor")
        } else {
            h("checkpoint0")
        };
        let child = h(&format!("checkpoint{i}"));
        let intent = serde_json::json!({"schema":INTENT_SCHEMA,"parent_checkpoint_sha256":parent,"protocol":protocol,"corpus_sha256":h("corpus"),"optimizer_epoch_sha256":epoch.sha256().unwrap(),"initial_step":i*3000,"target_step":(i+1)*3000,"execution_partition_sha256":h("partition"),"sampler_sha256":h("sampler")});
        let intent_id = object_id(INTENT_SCHEMA, &canonical_bytes(&intent).unwrap()).unwrap();
        let eligible = put(
            &mut bundle,
            ELIGIBILITY_SCHEMA,
            &serde_json::json!({"schema":ELIGIBILITY_SCHEMA,"purpose":"training","intent_sha256":intent_id,"corpus_sha256":h("corpus"),"role_exposure_closure_sha256":h("fixture role closure"),"source_roles":["train"]}),
        );
        let credit = put(
            &mut bundle,
            CREDIT_SCHEMA,
            &serde_json::json!({"schema":CREDIT_SCHEMA,"commit_sha256":h(&format!("commit{i}")),"parent_checkpoint_sha256":parent,"checkpoint_sha256":child,"optimizer_epoch_sha256":epoch.sha256().unwrap(),"intent_sha256":intent_id,"eligibility_sha256":eligible,"rank":0,"world":1,"initial_update":i*3000,"target_update":(i+1)*3000}),
        );
        let edge = serde_json::json!({"schema":EDGE_SCHEMA,"previous_edge_sha256":bundle.lineage.edges.last(),"parent_checkpoint_sha256":parent,"checkpoint_sha256":child,"weights_sha256":h(&format!("weights{i}")),"optimizer_sha256":h(&format!("optimizer{i}")),"mode":if i==0 {"weight_warm_start"} else {"new_corpus_transition"},"epoch":epoch,"intent":intent,"committed_credit_sha256":[credit]});
        let id = put(&mut bundle, EDGE_SCHEMA, &edge);
        bundle.lineage.edges.push(id);
    }
    DescendantLineagePolicy {
        schema: DESCENDANT_POLICY_SCHEMA.into(),
        run: "descendant run".into(),
        scientific_policy_sha256: h("science"),
        previous_completion_sha256: h("completion"),
        bundle,
    }
}

#[test]
fn descendant_verifier_requires_both_committed_blocks_and_scoped_opt_in() {
    let (_t, _root, _key, base, _provisioner) = native_grants::tests::fixture();
    let prior = base.native_grant_delegation.as_ref().unwrap();
    let mut next = prior.clone();
    let policy = lineage_fixture();
    next.run = policy.run.clone();
    next.scientific_policy_sha256 = policy.scientific_policy_sha256.clone();
    next.initial_optimizer_step = 6000;
    let completion = NativeGrantArtifact {
        path: "/data/completion.json".into(),
        sha256: policy.previous_completion_sha256.clone(),
    };
    let v = validate_descendant_window(&policy, &next).unwrap();
    assert_eq!(
        (
            v.local_step(),
            v.committed_updates(),
            v.cumulative_updates()
        ),
        (6000, 6000, None)
    );
    validate_descendant_transition(&policy, None, &next, prior, Some(&completion)).unwrap();
    assert!(validate_descendant_transition(&policy, None, &next, prior, None).is_err());
    let mut short = policy.clone();
    short.bundle.lineage.edges.truncate(1);
    next.initial_optimizer_step = 3000;
    validate_descendant_window(&short, &next).unwrap();
    assert!(validate_descendant_transition(&short, None, &next, prior, Some(&completion)).is_err());
    next.run = "wrong scope".into();
    assert!(validate_descendant_window(&short, &next).is_err());
    next.run = policy.run.clone();
    next.initial_optimizer_step = 70000;
    assert!(validate_descendant_window(&policy, &next).is_err());
}

#[test]
fn validly_signed_current_digest_rotation_without_opt_in_keeps_legacy_window() {
    let (_t, _root, _key, base, _provisioner) = native_grants::tests::fixture();
    let reviewer = SigningKey::from_bytes(&[81; 32]);
    let pk = hex::encode(reviewer.verifying_key().to_bytes());
    let mut r = transition(
        &base,
        &reviewer,
        EpochTransitionMode::ReviewedNewRunSuccessor,
    );
    assert_eq!(r.body.next_policy.witness_source_sha256, source_sha256());
    assert_eq!(
        r.body.next_policy.authority_source_sha256,
        authority_source_sha256()
    );
    assert!(descendant_policy(&r.body.next_policy).unwrap().is_none());
    r.body
        .next_policy
        .native_grant_delegation
        .as_mut()
        .unwrap()
        .initial_optimizer_step = 3000;
    r.signature = sign(&reviewer, TRANSITION_DOMAIN, &r.body).unwrap();
    verify(&pk, TRANSITION_DOMAIN, &r.body, &r.signature).unwrap();
    let error = validate_transition_with_key(&base, &base, &r, 50, &pk).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("invalid reviewed successor optimizer window"),
        "{error}"
    );
    // Neither a forged body nor an expired/foreign issuer obtains a new window.
    r.body
        .next_policy
        .native_grant_delegation
        .as_mut()
        .unwrap()
        .initial_optimizer_step = 66000;
    assert!(validate_transition_with_key(&base, &base, &r, 50, &pk).is_err());
    r.signature = sign(&reviewer, TRANSITION_DOMAIN, &r.body).unwrap();
    assert!(validate_transition_with_key(&base, &base, &r, 100, &pk).is_err());
    assert!(
        validate_transition_with_key(&base, &base, &r, 50, &base.authority_public_key).is_err()
    );
}

#[test]
fn forged_or_stripped_opt_in_projection_breaks_the_transition_signature() {
    let (_t, _root, _key, base, _provisioner) = native_grants::tests::fixture();
    let reviewer = SigningKey::from_bytes(&[81; 32]);
    let pk = hex::encode(reviewer.verifying_key().to_bytes());
    let r = transition(
        &base,
        &reviewer,
        EpochTransitionMode::ReviewedNewRunSuccessor,
    );
    // The coordinator's future field is signed inside next_policy. This
    // projection test does not claim the absent DTO hunk is integrated.
    let mut body = serde_json::to_value(&r.body).unwrap();
    body["next_policy"]["descendant_lineage_policy"] =
        serde_json::to_value(lineage_fixture()).unwrap();
    let signature = sign(&reviewer, TRANSITION_DOMAIN, &body).unwrap();
    verify(&pk, TRANSITION_DOMAIN, &body, &signature).unwrap();
    let mut forged = body.clone();
    forged["next_policy"]["descendant_lineage_policy"]["run"] =
        Value::String("foreign scope".into());
    assert!(verify(&pk, TRANSITION_DOMAIN, &forged, &signature).is_err());
    body["next_policy"]
        .as_object_mut()
        .unwrap()
        .remove("descendant_lineage_policy");
    assert!(verify(&pk, TRANSITION_DOMAIN, &body, &signature).is_err());
}
fn transition(
    base: &WitnessPolicy,
    reviewer: &SigningKey,
    mode: EpochTransitionMode,
) -> SignedWitnessEpochTransition {
    let mut next = base.clone();
    next.authority_epoch = sha(b"next epoch");
    let d = next.native_grant_delegation.as_mut().unwrap();
    d.authority_epoch = next.authority_epoch.clone();
    d.grant_template.authority_epoch = next.authority_epoch.clone();
    if mode == EpochTransitionMode::ReviewedNewRunSuccessor {
        d.run = "next reviewed run".into();
        d.initial_optimizer_step = 66000;
        d.total_evaluation_budget = 8;
    }
    let a = NativeGrantArtifact {
        path: "/data/review/evidence.json".into(),
        sha256: sha(b"exact evidence"),
    };
    let body = WitnessEpochTransitionBody {
        schema: "parallel.witness_epoch_transition.v1".into(),
        immutable_base_policy_sha256: base.sha256().unwrap(),
        previous_effective_policy_sha256: base.sha256().unwrap(),
        previous_transition_sha256: ZERO.into(),
        previous_witness_sequence: 7,
        previous_witness_tip_sha256: sha(b"original head"),
        authority_recovery_receipt: a.clone(),
        run_approval: a.clone(),
        frozen_config: a.clone(),
        previous_run_completion: if mode == EpochTransitionMode::ReviewedNewRunSuccessor {
            Some(a)
        } else {
            None
        },
        mode,
        next_update_sequence: 2,
        previous_update_sha256: sha(b"original grant"),
        original_reserved_evaluations: 3,
        historical_authorization_count: 0,
        historical_budget_accounting: None,
        historical_accounting: None,
        cumulative_evaluation_limit: next
            .native_grant_delegation
            .as_ref()
            .unwrap()
            .total_evaluation_budget,
        not_before_unix_ms: 1,
        expires_unix_ms: 100,
        next_policy: next,
    };
    SignedWitnessEpochTransition {
        authorization_public_key: hex::encode(reviewer.verifying_key().to_bytes()),
        signature: sign(reviewer, TRANSITION_DOMAIN, &body).unwrap(),
        body,
    }
}
#[test]
fn same_window_cannot_extend_budget_deadline_or_relabel_initial_optimizer() {
    let (_t, _root, _key, base, _provisioner) = native_grants::tests::fixture();
    let reviewer = SigningKey::from_bytes(&[81; 32]);
    let pk = hex::encode(reviewer.verifying_key().to_bytes());
    let original = transition(&base, &reviewer, EpochTransitionMode::SameWindowRecovery);
    validate_transition_with_key(&base, &base, &original, 50, &pk).unwrap();
    for change in 0..4 {
        let mut r = original.clone();
        let d = r.body.next_policy.native_grant_delegation.as_mut().unwrap();
        match change {
            0 => {
                d.total_evaluation_budget += 1;
                r.body.cumulative_evaluation_limit += 1;
            }
            1 => d.deadline_unix_ms += 1,
            2 => d.initial_optimizer_step = 66000,
            _ => r.body.next_policy.authority_instance = sha(b"empty replacement authority"),
        };
        r.signature = sign(&reviewer, TRANSITION_DOMAIN, &r.body).unwrap();
        assert!(validate_transition_with_key(&base, &base, &r, 50, &pk).is_err());
    }
    assert!(validate_transition_with_key(&base, &base, &original, 100, &pk).is_err());
    assert!(
        validate_transition_with_key(&base, &base, &original, 50, &base.authority_public_key)
            .is_err()
    );
}
#[test]
fn successor_requires_separate_completion_and_exact6000_window() {
    let (_t, _root, _key, base, _provisioner) = native_grants::tests::fixture();
    let reviewer = SigningKey::from_bytes(&[81; 32]);
    let pk = hex::encode(reviewer.verifying_key().to_bytes());
    let r = transition(
        &base,
        &reviewer,
        EpochTransitionMode::ReviewedNewRunSuccessor,
    );
    validate_transition_with_key(&base, &base, &r, 50, &pk).unwrap();
    assert!(r.body.next_policy.validate().is_err()); // not an unextended base
    let mut bad = r.clone();
    bad.body.previous_run_completion = None;
    bad.signature = sign(&reviewer, TRANSITION_DOMAIN, &bad.body).unwrap();
    assert!(validate_transition_with_key(&base, &base, &bad, 50, &pk).is_err());
    let mut bad = r;
    bad.body
        .next_policy
        .native_grant_delegation
        .as_mut()
        .unwrap()
        .initial_optimizer_step = 63000;
    bad.signature = sign(&reviewer, TRANSITION_DOMAIN, &bad.body).unwrap();
    assert!(validate_transition_with_key(&base, &base, &bad, 50, &pk).is_err());
}
#[test]
fn signed_original_transition_ack_binds_head_budget_policy_and_offline_time() {
    let (_t, _root, _key, base, _provisioner) = native_grants::tests::fixture();
    let reviewer = SigningKey::from_bytes(&[81; 32]);
    let pk = hex::encode(reviewer.verifying_key().to_bytes());
    let r = transition(&base, &reviewer, EpochTransitionMode::SameWindowRecovery);
    let ack_key = SigningKey::from_bytes(&[42; 32]);
    let body = WitnessEpochTransitionAckBody {
        schema: "parallel.witness_epoch_transition_ack.v1".into(),
        immutable_base_policy_sha256: base.sha256().unwrap(),
        witness_instance: base.witness_instance.clone(),
        request_sha256: sha(&bytes(&r).unwrap()),
        effective_policy_sha256: r.body.next_policy.sha256().unwrap(),
        witness_sequence: 8,
        previous_record_sha256: r.body.previous_witness_tip_sha256.clone(),
        committed_unix_ms: 50,
    };
    let p = WitnessEpochProof {
        epoch_transition: r,
        ack: SignedWitnessEpochTransitionAck {
            signature: sign(&ack_key, TRANSITION_ACK_DOMAIN, &body).unwrap(),
            body,
        },
    };
    let effective = verify_epoch_chain_checked(&base, std::slice::from_ref(&p), Some(&pk)).unwrap();
    assert_eq!(
        effective.authority_epoch,
        p.epoch_transition.body.next_policy.authority_epoch
    );
    let mut altered = p.clone();
    altered.ack.body.previous_record_sha256 = sha(b"other ledger");
    assert!(verify_epoch_chain_checked(&base, &[altered], Some(&pk)).is_err());
    assert!(verify_epoch_chain_checked(&base, &[p.clone(), p], Some(&pk)).is_err());
}
#[test]
fn initial_bootstrap_preserves_original_static_budget_and_unknown_identities() {
    let (_t, _root, _key, mut base, _provisioner) = native_grants::tests::fixture();
    let reviewer = SigningKey::from_bytes(&[81; 32]);
    let pk = hex::encode(reviewer.verifying_key().to_bytes());
    let mut r = transition(&base, &reviewer, EpochTransitionMode::SameWindowRecovery);
    base.native_grant_delegation = None;
    base.native_fit_trust = Some(NativeWitnessTrust {
        authorization_public_key: hex::encode(
            SigningKey::from_bytes(&[82; 32]).verifying_key().to_bytes(),
        ),
        grant_sha256: sha(b"original static grant"),
    });
    r.body.immutable_base_policy_sha256 = base.sha256().unwrap();
    r.body.previous_effective_policy_sha256 = base.sha256().unwrap();
    r.body.mode = EpochTransitionMode::InitialDelegationBootstrap;
    r.body.original_reserved_evaluations = 0;
    r.body.next_update_sequence = 1;
    r.body.previous_update_sha256 = ZERO.into();
    r.body.historical_authorization_count = 15;
    let accounting = HistoricalBudgetAccounting {
        schema: "jope.original_static_grant_budget.v1".into(),
        original_policy_sha256: base.sha256().unwrap(),
        original_static_grant_sha256: sha(b"original static grant"),
        original_approved_candidate_budget: 16,
        original_authorization_count: 15,
        physical_starts: None,
        unknown_or_held: BTreeMap::from([(
            sha(b"retained ambiguous ridge"),
            r.body.authority_recovery_receipt.clone(),
        )]),
        source_closure: vec![r.body.authority_recovery_receipt.clone()],
        authority_recovery_receipt_sha256: r.body.authority_recovery_receipt.sha256.clone(),
    };
    r.body.historical_budget_accounting = Some(NativeGrantArtifact {
        path: "/data/review/original-budget.json".into(),
        sha256: sha(&bytes(&accounting).unwrap()),
    });
    r.body.historical_accounting = Some(accounting);
    r.signature = sign(&reviewer, TRANSITION_DOMAIN, &r.body).unwrap();
    validate_transition_with_key(&base, &base, &r, 50, &pk).unwrap();
    let mut missing = r.clone();
    missing.body.historical_accounting = None;
    missing.signature = sign(&reviewer, TRANSITION_DOMAIN, &missing.body).unwrap();
    assert!(validate_transition_with_key(&base, &base, &missing, 50, &pk).is_err());
    let mut changed = r;
    changed
        .body
        .historical_accounting
        .as_mut()
        .unwrap()
        .unknown_or_held
        .clear();
    changed.signature = sign(&reviewer, TRANSITION_DOMAIN, &changed.body).unwrap();
    assert!(validate_transition_with_key(&base, &base, &changed, 50, &pk).is_err());
}
