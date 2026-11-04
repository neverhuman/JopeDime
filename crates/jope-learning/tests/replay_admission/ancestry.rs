use super::*;

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
