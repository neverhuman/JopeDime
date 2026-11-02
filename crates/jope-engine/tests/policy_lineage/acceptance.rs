use super::*;

#[test]
fn accepted60000_reset3000_then6000_is_a_native_v3_successor() {
    let b = fixture();
    let v = native(&b).unwrap().unwrap();
    assert_eq!(v.local_step(), 6000);
    assert_eq!(v.committed_updates(), 6000);
    assert_eq!(v.cumulative_updates(), None);
    assert_eq!(v.lineage().anchor.optimizer_step, 60000);
    assert_eq!(v.edges()[0].mode, ContinuationMode::WeightWarmStart);
    assert_eq!(v.edges()[1].epoch, v.edges()[0].epoch);
    let w = warm_start::window(warm_start::LOOP_V3, 6000, 0).unwrap();
    assert_eq!(
        (w.initial, w.target, w.warm_start_parent_step),
        (6000, 9000, None)
    );
}

#[test]
fn actual_checkpoint_bindings_cannot_be_replaced_by_lineage_labels() {
    use serde_json::json;
    let b = fixture();
    let verified = b.verify().unwrap();
    for e in verified.edges() {
        let state = json!({"schema":"parallel.checkpoint.v1","step":e.intent.target_step,
            "model_sha256":e.weights_sha256,"optimizer_sha256":e.optimizer_sha256,
            "optimizer_epoch_sha256":e.intent.optimizer_epoch_sha256,
            "optimizer_recipe_sha256":e.epoch.recipe_sha256,
            "objective_sha256":e.intent.protocol.objective_sha256,
            "vocabulary_sha256":e.intent.protocol.vocabulary_sha256,"corpus_sha256":e.intent.corpus_sha256});
        check_checkpoint(&state, e).unwrap();
        for field in [
            "model_sha256",
            "optimizer_sha256",
            "optimizer_epoch_sha256",
            "optimizer_recipe_sha256",
            "objective_sha256",
            "vocabulary_sha256",
            "corpus_sha256",
        ] {
            let mut forged = state.clone();
            forged[field] = json!(h("different identity"));
            assert!(check_checkpoint(&forged, e).is_err(), "{field}");
        }
        let mut relabeled = state;
        relabeled["step"] = json!(70000);
        assert!(check_checkpoint(&relabeled, e).is_err());
    }
}
#[test]
fn random70000_and_relabelled3000_cannot_impersonate_a_successor() {
    let b = fixture();
    let (p, mut n, mut c) = configs(&b);
    n.initial_optimizer_step = 70000;
    assert!(native_loop::verify_versioned_successor(&n, &p, &c, None, None).is_err());
    n.initial_optimizer_step = 3000;
    c.target_optimizer_step = 3000;
    assert!(native_loop::verify_versioned_successor(&n, &p, &c, Some(&b), None).is_err());
    let mut forged = b.clone();
    let mut e = edge(&forged, 1);
    e.intent.target_step = 70000;
    replace(&mut forged, 1, &e);
    assert!(native(&forged).is_err());
}

#[test]
fn legacy_v1_still_refuses_the_same_reset_epoch_completion() {
    let b = fixture();
    let (p, mut n, c) = configs(&b);
    n.schema = warm_start::LOOP_V1.into();
    n.initial_optimizer_step = 66000;
    let error = native_loop::verify_versioned_successor(&n, &p, &c, None, None).unwrap_err();
    assert!(error.to_string().contains("no successor protocol"));
    assert!(native(&b).is_ok());
}

#[test]
fn legacy_exposure_cannot_be_certified_by_relabeling_the_anchor() {
    let mut b = fixture();
    b.lineage.anchor.committed_updates = Some(60000);
    let error = b.verify().unwrap_err();
    assert!(error.to_string().contains("cannot be relabeled"));
}
#[test]
fn missing_altered_forked_and_cyclic_edges_are_refused() {
    let good = fixture();
    let mut b = good.clone();
    b.evidence.remove(&b.lineage.edges[0]);
    assert!(native(&b).is_err());
    let mut b = good.clone();
    b.evidence.get_mut(&b.lineage.edges[0]).unwrap().push(' ');
    assert!(native(&b).is_err());
    let mut b = good.clone();
    b.lineage.edges.swap(0, 1);
    assert!(b.verify().is_err());
    let mut b = good.clone();
    b.lineage.edges.push(b.lineage.edges[0].clone());
    assert!(b.verify().is_err());
    let mut b = good;
    let mut e = edge(&b, 1);
    e.checkpoint_sha256 = b.lineage.anchor.checkpoint_sha256.clone();
    replace(&mut b, 1, &e);
    assert!(b.verify().unwrap_err().to_string().contains("cycle"));
}
#[test]
fn objective_recipe_architecture_vocabulary_and_support_mismatches_are_refused() {
    for case in 0..5 {
        let mut b = fixture();
        let mut e = edge(&b, 1);
        match case {
            0 => e.intent.protocol.objective_sha256 = h("different"),
            1 => e.intent.protocol.recipe_sha256 = h("different"),
            2 => e.intent.protocol.architecture_sha256 = h("different"),
            3 => e.intent.protocol.vocabulary_sha256 = h("different"),
            _ => e.intent.protocol.support_sha256 = h("different"),
        }
        replace(&mut b, 1, &e);
        assert!(native(&b).is_err(), "case {case}");
    }
}
#[test]
fn empty_reused_and_incomplete_rank_credit_is_refused() {
    let mut b = fixture();
    let mut e = edge(&b, 1);
    e.committed_credit_sha256.clear();
    replace(&mut b, 1, &e);
    assert!(native(&b).is_err());
    let mut b = fixture();
    let mut e = edge(&b, 1);
    e.committed_credit_sha256 = edge(&b, 0).committed_credit_sha256;
    replace(&mut b, 1, &e);
    assert!(native(&b).is_err());
    let mut b = fixture();
    let mut e = edge(&b, 1);
    let mut c: CommittedUpdateCredit =
        serde_json::from_str(&b.evidence[&e.committed_credit_sha256[0]]).unwrap();
    c.world = 2;
    e.committed_credit_sha256[0] = put(&mut b, CREDIT_SCHEMA, &c);
    replace(&mut b, 1, &e);
    assert!(native(&b).is_err());
}
#[test]
fn duplicate_delivery_adds_no_credit_and_forked_commit_cannot_be_reused() {
    let mut b = fixture();
    let repeated = b.evidence.clone();
    b.evidence.extend(repeated);
    assert_eq!(b.verify().unwrap().committed_updates(), 6000);
    let first = edge(&b, 0);
    let old: CommittedUpdateCredit =
        serde_json::from_str(&b.evidence[&first.committed_credit_sha256[0]]).unwrap();
    let mut second = edge(&b, 1);
    let mut c: CommittedUpdateCredit =
        serde_json::from_str(&b.evidence[&second.committed_credit_sha256[0]]).unwrap();
    c.commit_sha256 = old.commit_sha256;
    second.committed_credit_sha256[0] = put(&mut b, CREDIT_SCHEMA, &c);
    replace(&mut b, 1, &second);
    assert!(b.verify().unwrap_err().to_string().contains("reused"));
}
#[test]
fn stale_intent_and_evaluation_role_evidence_are_refused() {
    let mut b = fixture();
    let mut e = edge(&b, 1);
    let mut c: CommittedUpdateCredit =
        serde_json::from_str(&b.evidence[&e.committed_credit_sha256[0]]).unwrap();
    c.intent_sha256 = h("stale intent");
    e.committed_credit_sha256[0] = put(&mut b, CREDIT_SCHEMA, &c);
    replace(&mut b, 1, &e);
    assert!(native(&b).is_err());
    let mut b = fixture();
    let mut e = edge(&b, 1);
    let mut c: CommittedUpdateCredit =
        serde_json::from_str(&b.evidence[&e.committed_credit_sha256[0]]).unwrap();
    let mut eligibility: TrainingEligibility =
        serde_json::from_str(&b.evidence[&c.eligibility_sha256]).unwrap();
    eligibility.source_roles = BTreeSet::from(["meta_test".into()]);
    c.eligibility_sha256 = put(&mut b, ELIGIBILITY_SCHEMA, &eligibility);
    e.committed_credit_sha256[0] = put(&mut b, CREDIT_SCHEMA, &c);
    replace(&mut b, 1, &e);
    assert!(native(&b).is_err());
}
