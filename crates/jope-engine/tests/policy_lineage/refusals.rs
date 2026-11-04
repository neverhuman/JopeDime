use super::*;

#[test]
fn overflow_depth_and_byte_limits_are_enforced() {
    let mut b = fixture();
    b.lineage.edges.clear();
    b.evidence.clear();
    append(&mut b, 0, 9_007_199_254_740_990);
    append(&mut b, 0, 9_007_199_254_740_991);
    assert!(b.verify().unwrap_err().to_string().contains("overflow"));
    b.lineage.anchor.committed_updates = Some(u64::MAX);
    assert!(b.verify().is_err());
    let mut b = fixture();
    b.lineage.edges.resize(MAX_LINEAGE_EDGES + 1, h("extra"));
    assert!(b.verify().unwrap_err().to_string().contains("bound"));
    assert!(object_id(EDGE_SCHEMA, &vec![b' '; MAX_LINEAGE_BYTES + 1]).is_err());
    let mut b = fixture();
    b.evidence
        .insert(h("oversized"), "x".repeat(MAX_EVIDENCE_BYTES + 1));
    assert!(b.verify().unwrap_err().to_string().contains("bound"));
}

#[test]
fn the_explicit_sixty_four_edge_boundary_is_accepted() {
    let mut b = fixture();
    for index in 2..MAX_LINEAGE_EDGES as u64 {
        append(&mut b, index * 3000, (index + 1) * 3000);
    }
    assert_eq!(b.verify().unwrap().edges().len(), 64);
    append(&mut b, 64 * 3000, 65 * 3000);
    assert!(b.verify().is_err());
}
#[test]
fn reset_carry_and_exact_resume_are_distinct() {
    for mode in [
        ContinuationMode::ColdInitialization,
        ContinuationMode::ExactResume,
        ContinuationMode::WeightWarmStart,
    ] {
        let mut b = fixture();
        let mut e = edge(&b, 1);
        e.mode = mode;
        replace(&mut b, 1, &e);
        assert!(b.verify().is_err());
    }
    let mut b = fixture();
    let mut e = edge(&b, 0);
    e.mode = ContinuationMode::NewCorpusTransition;
    replace(&mut b, 0, &e);
    assert!(b.verify().is_err());
}
#[test]
fn three_thousand_is_a_descendant_but_not_a_completed_two_block_run() {
    let mut b = fixture();
    b.lineage.edges.truncate(1);
    let v = b.verify().unwrap();
    assert_eq!(v.local_step(), 3000);
    assert_eq!(v.committed_updates(), 3000);
    let good = fixture();
    let (p, n, c) = configs(&good);
    assert!(native_loop::verify_versioned_successor(&n, &p, &c, Some(&b), None).is_err());
}
#[test]
fn subsequent_v3_successor_preserves_the_verified_prefix() {
    let mut b = fixture();
    let prior = b.verify().unwrap();
    let (_, p, _) = configs(&b);
    append(&mut b, 6000, 9000);
    append(&mut b, 9000, 12000);
    let mut n = p.clone();
    n.run = "third run".into();
    n.initial_optimizer_step = 12000;
    n.accepted_checkpoint = artifact(edge(&b, 3).checkpoint_sha256);
    let a = artifact(h("completion artifact"));
    let c = RunCompletion {
        schema: warm_start::COMPLETION_V3.into(),
        run: p.run.clone(),
        config: a.clone(),
        initial_optimizer_step: 6000,
        target_optimizer_step: 12000,
        checkpoint: n.accepted_checkpoint.clone(),
        corpus: n.initial_corpus.clone(),
        accepted_parent_decision: a.clone(),
        final_proposals: BTreeMap::from([(Arm::Model, a.clone()), (Arm::Shuffled, a)]),
    };
    assert!(native_loop::verify_versioned_successor(&n, &p, &c, Some(&b), Some(&prior)).is_ok());
    assert!(native_loop::verify_versioned_successor(&n, &p, &c, Some(&b), None).is_err());
    b.lineage.anchor.weights_sha256 = h("another root");
    assert!(native_loop::verify_versioned_successor(&n, &p, &c, Some(&b), Some(&prior)).is_err());
}
#[test]
fn versioned_bytes_roundtrip_and_identity_domains_are_distinct() {
    let b = fixture();
    let encoded = canonical_bytes(&b).unwrap();
    let decoded: LineageBundle = serde_json::from_slice(&encoded).unwrap();
    assert_eq!(canonical_bytes(&decoded).unwrap(), encoded);
    assert!(decoded.verify().is_ok());
    assert_ne!(
        object_id(EDGE_SCHEMA, b"{}").unwrap(),
        object_id(INTENT_SCHEMA, b"{}").unwrap()
    );
    let mut v = serde_json::to_value(&b.lineage).unwrap();
    v["unknown_authority"] = true.into();
    assert!(serde_json::from_value::<PolicyLineage>(v).is_err());
}

#[cfg(feature = "neural")]
#[test]
fn neural_continuation_and_retained_ancestry_bridges_require_the_actual_epoch() {
    use jope_parallel::{
        neural,
        task_contract::training_partition,
        training::{
            ancestry::{self, Genealogy, Node},
            continuation::{check_versioned_parent, Current, Mode},
            objective,
        },
    };
    use serde_json::json;
    let config = serde_json::to_value(neural::Config::regression(401)).unwrap();
    let mut b = fixture();
    b.lineage.edges.clear();
    b.evidence.clear();
    b.lineage.protocol.objective_sha256 = objective::identity_sha256();
    b.lineage.protocol.architecture_sha256 = objective::identity_digest(&config);
    b.lineage.anchor.architecture_sha256 = b.lineage.protocol.architecture_sha256.clone();
    let partition = objective::identity_digest(&training_partition(1, 256, 2048).unwrap());
    append_with_inputs(&mut b, 0, 3000, partition.clone(), h("fixture sampler"));
    append_with_inputs(&mut b, 3000, 6000, partition, h("fixture sampler"));
    let verified = b.verify().unwrap();
    let tip = verified.tip();
    let current = Current {
        corpus_sha256: &tip.intent.corpus_sha256,
        vocabulary_sha256: &tip.intent.protocol.vocabulary_sha256,
        world: 1,
        microbatch: 256,
        global_batch: 2048,
        seed: 7,
        sampler_id: "fixture sampler",
        config: &config,
        objective_sha256: &tip.intent.protocol.objective_sha256,
    };
    let mut state = json!({"step":6000,"objective":objective::identity(),
        "objective_sha256":objective::identity_sha256(),"model_sha256":tip.weights_sha256,
        "optimizer_sha256":tip.optimizer_sha256,"optimizer_epoch_sha256":tip.intent.optimizer_epoch_sha256,
        "optimizer_recipe_sha256":tip.epoch.recipe_sha256,
        "vocabulary_sha256":current.vocabulary_sha256,"corpus_sha256":current.corpus_sha256,
        "execution_partition":training_partition(1,256,2048).unwrap(),"world":1,"global_batch":2048,
        "seed":7,"sampler_id":"fixture sampler"});
    let check = |state: &serde_json::Value| {
        check_versioned_parent(
            Mode::ExactResume,
            state,
            &config,
            &current,
            None,
            &tip.checkpoint_sha256,
            &verified,
        )
    };
    let report = check(&state).unwrap();
    assert_eq!(
        report["policy_lineage"]["committed_additional_updates"],
        6000
    );
    state["optimizer_epoch_sha256"] = json!(h("another epoch"));
    assert!(check(&state).is_err());
    state["optimizer_epoch_sha256"] = json!(tip.intent.optimizer_epoch_sha256);
    state["sampler_id"] = json!("another sampler");
    assert!(check(&state).is_err());

    // Retention flags below are fixture inputs, not an on-disk durability proof.
    let node = |digest: String, step, state| Node {
        digest,
        dir: "fixture".into(),
        step,
        state,
        weights_retained: true,
        optimizer_retained: true,
    };
    let anchor = &b.lineage.anchor;
    let mut g = Genealogy {
        nodes: BTreeMap::from([(
            anchor.checkpoint_sha256.clone(),
            node(
                anchor.checkpoint_sha256.clone(),
                60000,
                json!({"model_sha256":anchor.weights_sha256}),
            ),
        )]),
        edges: BTreeMap::new(),
    };
    for e in verified.edges() {
        g.nodes.insert(e.checkpoint_sha256.clone(), node(e.checkpoint_sha256.clone(), e.intent.target_step,
            json!({"schema":"parallel.checkpoint.v1","step":e.intent.target_step,
            "model_sha256":e.weights_sha256,"optimizer_sha256":e.optimizer_sha256,
            "optimizer_recipe_sha256":e.epoch.recipe_sha256,
            "optimizer_epoch_sha256":e.intent.optimizer_epoch_sha256,"objective_sha256":e.intent.protocol.objective_sha256,
            "vocabulary_sha256":e.intent.protocol.vocabulary_sha256,"corpus_sha256":e.intent.corpus_sha256})));
    }
    assert_eq!(
        ancestry::verified_lineage_exposure(&g, &b).unwrap()["cumulative_committed_updates"],
        serde_json::Value::Null
    );
    g.nodes
        .get_mut(&tip.checkpoint_sha256)
        .unwrap()
        .optimizer_retained = false;
    assert!(ancestry::verified_lineage_exposure(&g, &b).is_err());
}
