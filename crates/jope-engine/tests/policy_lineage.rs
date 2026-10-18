//! Non-authoritative CPU receipts. No trainer, authority service or grant issuance.
use anyhow::Result;
use jain_smartcluster::{JobId, ResourceRequest, Timestamp};
use jope_parallel::{
    manifest::hash,
    native_genome::Arm,
    native_graph::Artifact,
    native_loop::{self, lineage::*, warm_start, Config, RunCompletion},
    task_contract::ContinuationMode,
};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};

fn h(s: &str) -> String {
    hash(s.as_bytes())
}
fn put<T: Serialize>(bundle: &mut LineageBundle, schema: &str, value: &T) -> String {
    let bytes = canonical_bytes(value).unwrap();
    let id = object_id(schema, &bytes).unwrap();
    bundle
        .evidence
        .insert(id.clone(), String::from_utf8(bytes).unwrap());
    id
}
fn edge(bundle: &LineageBundle, index: usize) -> LineageEdge {
    serde_json::from_str(&bundle.evidence[&bundle.lineage.edges[index]]).unwrap()
}
fn replace(bundle: &mut LineageBundle, index: usize, edge: &LineageEdge) {
    bundle.lineage.edges[index] = put(bundle, EDGE_SCHEMA, edge);
}
fn append(bundle: &mut LineageBundle, initial: u64, target: u64) {
    append_with_inputs(bundle, initial, target, h("world1"), h("sampler"));
}
fn append_with_inputs(
    bundle: &mut LineageBundle,
    initial: u64,
    target: u64,
    partition_sha256: String,
    sampler_sha256: String,
) {
    let prior = bundle
        .lineage
        .edges
        .len()
        .checked_sub(1)
        .map(|i| edge(bundle, i));
    let parent = prior
        .as_ref()
        .map_or(bundle.lineage.anchor.checkpoint_sha256.clone(), |p| {
            p.checkpoint_sha256.clone()
        });
    let reset = prior.is_none() || initial == 0;
    let epoch = if reset {
        OptimizerEpoch {
            schema: EPOCH_SCHEMA.into(),
            origin_checkpoint_sha256: parent.clone(),
            reset_nonce_sha256: h(&format!("fixture reset{}", bundle.lineage.edges.len())),
            recipe_sha256: bundle.lineage.protocol.recipe_sha256.clone(),
        }
    } else {
        prior.as_ref().unwrap().epoch.clone()
    };
    let intent = TrainingIntent {
        schema: INTENT_SCHEMA.into(),
        parent_checkpoint_sha256: parent.clone(),
        protocol: bundle.lineage.protocol.clone(),
        corpus_sha256: h("corpus"),
        optimizer_epoch_sha256: epoch.sha256().unwrap(),
        initial_step: initial,
        target_step: target,
        execution_partition_sha256: partition_sha256,
        sampler_sha256,
    };
    let intent_id = object_id(INTENT_SCHEMA, &canonical_bytes(&intent).unwrap()).unwrap();
    let eligibility = TrainingEligibility {
        schema: ELIGIBILITY_SCHEMA.into(),
        purpose: "training".into(),
        intent_sha256: intent_id.clone(),
        corpus_sha256: intent.corpus_sha256.clone(),
        role_exposure_closure_sha256: h("fixture role closure"),
        source_roles: BTreeSet::from(["train".into()]),
    };
    let eligible_id = put(bundle, ELIGIBILITY_SCHEMA, &eligibility);
    let checkpoint = h(&format!("checkpoint{target}"));
    let credit = CommittedUpdateCredit {
        schema: CREDIT_SCHEMA.into(),
        commit_sha256: h(&format!("commit{target}")),
        parent_checkpoint_sha256: parent.clone(),
        checkpoint_sha256: checkpoint.clone(),
        optimizer_epoch_sha256: intent.optimizer_epoch_sha256.clone(),
        intent_sha256: intent_id,
        eligibility_sha256: eligible_id,
        rank: 0,
        world: 1,
        initial_update: initial,
        target_update: target,
    };
    let credit_id = put(bundle, CREDIT_SCHEMA, &credit);
    let edge = LineageEdge {
        schema: EDGE_SCHEMA.into(),
        previous_edge_sha256: bundle.lineage.edges.last().cloned(),
        parent_checkpoint_sha256: parent,
        checkpoint_sha256: checkpoint,
        weights_sha256: h(&format!("weights{target}")),
        optimizer_sha256: h(&format!("optimizer{target}")),
        mode: if reset {
            ContinuationMode::WeightWarmStart
        } else {
            ContinuationMode::NewCorpusTransition
        },
        epoch,
        intent,
        committed_credit_sha256: vec![credit_id],
    };
    let id = put(bundle, EDGE_SCHEMA, &edge);
    bundle.lineage.edges.push(id);
}
fn fixture() -> LineageBundle {
    let protocol = ProtocolIdentity {
        objective_sha256: h("objective"),
        recipe_sha256: h("recipe"),
        architecture_sha256: h("architecture"),
        vocabulary_sha256: h("vocabulary"),
        support_sha256: h("support"),
    };
    let anchor = PolicyAnchor {
        checkpoint_sha256: h("accepted60000"),
        weights_sha256: h("accepted weights"),
        optimizer_step: 60000,
        objective_sha256: None,
        committed_updates: None,
        architecture_sha256: protocol.architecture_sha256.clone(),
        vocabulary_sha256: protocol.vocabulary_sha256.clone(),
        support_sha256: protocol.support_sha256.clone(),
    };
    let mut b = LineageBundle {
        lineage: PolicyLineage {
            schema: LINEAGE_SCHEMA.into(),
            anchor,
            protocol,
            edges: vec![],
        },
        evidence: BTreeMap::new(),
    };
    append(&mut b, 0, 3000);
    append(&mut b, 3000, 6000);
    b
}
fn artifact(d: String) -> Artifact {
    Artifact {
        path: format!("fixture/{d}.json").into(),
        sha256: d,
    }
}
fn configs(b: &LineageBundle) -> (Config, Config, RunCompletion) {
    let a = artifact(h("artifact"));
    let previous = Config {
        schema: warm_start::LOOP_V2.into(),
        run: "reset run".into(),
        accepted_checkpoint: artifact(b.lineage.anchor.checkpoint_sha256.clone()),
        initial_corpus: a.clone(),
        accepted_decision: a.clone(),
        scientific_policy: a.clone(),
        label_admission_policy: a.clone(),
        fit_policy: "fit.json".into(),
        run_approval: "approval.json".into(),
        resident_job: JobId::new(),
        workers: 48,
        cycles: 2,
        initial_optimizer_step: 60000,
        predecessor_run: None,
        updates_per_block: 3000,
        global_batch: 2048,
        microbatch: 256,
        direct_seconds: 30,
        continuation_evaluations: 24,
        continuation_seconds: 54,
        cpu_resources: ResourceRequest::cpu_and_ram(1000, 1024 * 1024 * 1024),
        task_deadline: Timestamp::from_unix_millis(u64::MAX),
        qualification: false,
        preparation_overlay: None,
        cuts: 1,
        cut_start: 0,
        materialization: None,
    };
    let mut next = previous.clone();
    next.schema = warm_start::LOOP_V3.into();
    next.run = "descendant run".into();
    next.predecessor_run = Some(a.clone());
    next.initial_optimizer_step = 6000;
    next.accepted_checkpoint = artifact(edge(b, 1).checkpoint_sha256);
    next.initial_corpus = artifact(h("corpus"));
    let completed = RunCompletion {
        schema: warm_start::COMPLETION_V2.into(),
        run: previous.run.clone(),
        config: a.clone(),
        initial_optimizer_step: 0,
        target_optimizer_step: 6000,
        checkpoint: next.accepted_checkpoint.clone(),
        corpus: next.initial_corpus.clone(),
        accepted_parent_decision: a.clone(),
        final_proposals: BTreeMap::from([(Arm::Model, a.clone()), (Arm::Shuffled, a)]),
    };
    (previous, next, completed)
}
fn native(b: &LineageBundle) -> Result<Option<VerifiedLineage>> {
    let (p, n, c) = configs(b);
    native_loop::verify_versioned_successor(&n, &p, &c, Some(b), None)
}

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
    let config = serde_json::to_value(neural::Config::default()).unwrap();
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
