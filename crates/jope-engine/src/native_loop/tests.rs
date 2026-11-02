use super::*;
fn r(g: u64, n: &str, role: NativeLoopRole) -> Request {
    Request {
        schema: "jope.native_loop_request.v1".into(),
        run: "r".into(),
        generation: g,
        id: n.into(),
        role,
        parameters: json!({}),
        predecessors: BTreeMap::new(),
    }
}
#[test]
fn independently_valid_graph_admission_with_same_ids_cannot_replace_loop_binding() {
    let request = r(0, "same", NativeLoopRole::Aggregate);
    let a = Artifact {
        path: "task.json".into(),
        sha256: "1".repeat(64),
    };
    let admission = Admission {
        schema: "jope.native_graph_admission.v1".into(),
        manifest: a.clone(),
        predecessors: BTreeMap::new(),
        delivery: Delivery::NativeJob,
    };
    let attempt: jain_smartcluster::FencedAttempt = serde_json::from_value(
        json!({"job_id":jain_smartcluster::JobId::new(),"number":1,"fence":1}),
    )
    .unwrap();
    let mut result = VerifiedResult {
        schema: "jope.native_graph_result.v1".into(),
        admission_sha256: hash(&serde_json::to_vec(&admission).unwrap()),
        task_manifest_sha256: a.sha256.clone(),
        task_id: request.id.clone(),
        run: request.run.clone(),
        generation: 0,
        receipt_id: "original".into(),
        native_attempt: attempt,
        worker_completion: a.clone(),
        output: a.clone(),
    };
    verify_result_binding(&request, &admission, &result).unwrap();
    let mut different = admission.clone();
    different.manifest.sha256 = "2".repeat(64);
    // This digest correctly authenticates a different real graph admission,
    // but names/run/generation alone must not make it the loop's task.
    result.admission_sha256 = hash(&serde_json::to_vec(&different).unwrap());
    result.task_manifest_sha256 = different.manifest.sha256;
    assert!(verify_result_binding(&request, &admission, &result).is_err());
}
#[test]
fn committed_training_is_not_a_quality_or_serving_acceptance() {
    let a = Artifact {
        path: "checkpoint.json".into(),
        sha256: "1".repeat(64),
    };
    let mut decision = ProposalParentDecision {
        schema: "jope.proposal_parent_decision.v1".into(),
        run: "r".into(),
        generation: 0,
        checkpoint: a.clone(),
        previous_parent: a.clone(),
        training_result: a.clone(),
        comparison: a.clone(),
        assessment: a.clone(),
        accepted: true,
    };
    assert!(verify_parent_decision(
        &decision,
        &json!({"committed":true}),
        "r",
        0,
        crate::native_loop::ParentEvidence {
            cp: &a,
            previous: &a,
            train: &a,
            comparison: &a
        },
    )
    .is_err());
    let review = json!({"schema":"jope.proposal_parent_quality_review.v1","checkpoint":a,"previous_parent":a,"training_result":a,"comparison":a,
        "accepted":true,"sealed_test_used":false,"serving_promotion":false,"reviewer":"unit-test reviewer","rationale":"unit-test evidence decision"});
    verify_parent_decision(
        &decision,
        &review,
        "r",
        0,
        crate::native_loop::ParentEvidence {
            cp: &a,
            previous: &a,
            train: &a,
            comparison: &a,
        },
    )
    .unwrap();
    decision.checkpoint.sha256 = "2".repeat(64);
    assert!(verify_parent_decision(
        &decision,
        &review,
        "r",
        0,
        crate::native_loop::ParentEvidence {
            cp: &a,
            previous: &a,
            train: &a,
            comparison: &a
        },
    )
    .is_err());
}
#[test]
fn unequal_valid_arms_remain_preserved_but_do_not_invent_matches() {
    let t = tempfile::tempdir().unwrap();
    let mut refs = Vec::new();
    let reference = Artifact {
        path: "source.json".into(),
        sha256: "a".repeat(64),
    };
    for (i, arm, slot) in [
        (0, Arm::Model, 0),
        (1, Arm::Generic, 0),
        (2, Arm::Shuffled, 0),
        (3, Arm::Model, 1),
        (4, Arm::Generic, 1),
    ] {
        let c = Candidate {
            schema: "jope.native_candidate.v1".into(),
            run: "r".into(),
            generation: 0,
            arm,
            matched_slot: slot,
            context: crate::corpus::ContextRow {
                dataset: "d".into(),
                family: "f".into(),
                cut: 0,
                partition: "meta_train".into(),
                path: "context".into(),
                sha256: "b".repeat(64),
                prepared: "prepared".into(),
                prepared_sha256: "c".repeat(64),
            },
            genome: "(pipe X (ridge))".into(),
            genome_sha256: "d".repeat(64),
            proposal: reference.clone(),
            parent_checkpoint: reference.clone(),
            sampling_seed: 1,
            origin_capture: reference.clone(),
            origin_row: 0,
            generic_draw: None,
        };
        refs.push(frozen(t.path(), &t.path().join(format!("candidates/{i}.json")), &c).unwrap());
    }
    let compared = matched_comparisons(t.path(), 0, &refs).unwrap();
    assert_eq!(compared["paired"].as_array().unwrap().len(), 1);
    assert_eq!(compared["unmatched"].as_array().unwrap().len(), 1);
    assert_eq!(
        compared["unmatched"][0]["arms"].as_object().unwrap().len(),
        2
    );
    refs.push(refs[0].clone());
    assert!(matched_comparisons(t.path(), 0, &refs).is_err());
}
#[test]
fn shared_backlog_counts_and_continuation_quarter_cap() {
    let direct = (0..500)
        .map(|i| r(2, &i.to_string(), NativeLoopRole::Direct))
        .collect();
    let cont = (0..500)
        .map(|i| r(1, &i.to_string(), NativeLoopRole::Continuation))
        .collect();
    let out = replenish(48, 20, 4, direct, cont);
    assert_eq!(out.len(), 120);
    assert_eq!(
        out.iter()
            .filter(|r| r.role == NativeLoopRole::Continuation)
            .count(),
        8
    );
    assert!(replenish(48, 90, 6, vec![r(0, "x", NativeLoopRole::Direct)], vec![]).is_empty());
}
#[test]
fn generations_not_newest_first_and_no_continuation_starvation() {
    let out = replenish(
        48,
        0,
        0,
        vec![
            r(3, "new", NativeLoopRole::Direct),
            r(0, "old", NativeLoopRole::Direct),
        ],
        vec![r(1, "continuation", NativeLoopRole::Continuation)],
    );
    assert_eq!(out[0].role, NativeLoopRole::Continuation);
    assert_eq!(out[1].generation, 0);
    assert_eq!(out[2].generation, 3);
}
#[test]
fn generation_seed_replays_but_changes_on_parent_or_generation() {
    let s = generation_seed("r", 0, "a").unwrap();
    assert_eq!(s, generation_seed("r", 0, "a").unwrap());
    assert_ne!(s, generation_seed("r", 1, "a").unwrap());
    assert_ne!(s, generation_seed("r", 0, "b").unwrap());
}
#[test]
fn conflicting_durable_intent_cannot_extend_original_window() {
    let t = tempfile::tempdir().unwrap();
    let p = t.path().join("state/window.json");
    frozen(t.path(), &p, &json!({"initial":60000,"target":63000})).unwrap();
    assert!(frozen(t.path(), &p, &json!({"initial":60000,"target":66000})).is_err());
}
#[test]
fn successor_requires_exact_completed_window_and_new_run_identity() {
    let a = Artifact {
        path: "a.json".into(),
        sha256: "a".repeat(64),
    };
    let previous = Config {
        preparation_overlay: None,
        schema: "jope.native_loop.v1".into(),
        run: "original".into(),
        accepted_checkpoint: a.clone(),
        initial_corpus: a.clone(),
        accepted_decision: a.clone(),
        scientific_policy: a.clone(),
        label_admission_policy: a.clone(),
        fit_policy: "policy.json".into(),
        run_approval: "/etc/jope-dime/test-run-approval.json".into(),
        resident_job: JobId::new(),
        workers: 48,
        cycles: 2,
        initial_optimizer_step: 60000,
        predecessor_run: None,
        updates_per_block: 3000,
        global_batch: 2048,
        microbatch: 256,
        direct_seconds: 30,
        continuation_evaluations: 16,
        continuation_seconds: 60,
        cpu_resources: ResourceRequest::cpu_and_ram(1000, 1024 * 1024 * 1024),
        task_deadline: Timestamp::from_unix_millis(u64::MAX),
        qualification: false,
        cuts: 1,
        cut_start: 0,
        materialization: None,
    };
    previous.validate().unwrap();
    let mut scoped = previous.clone();
    scoped.preparation_overlay = Some(a.clone());
    assert!(scoped.validate().is_err());
    scoped.continuation_evaluations = 24;
    scoped.continuation_seconds = 54;
    scoped.validate().unwrap();
    scoped.continuation_seconds = 53;
    assert!(scoped.validate().is_err());
    scoped.continuation_seconds = 54;
    scoped.continuation_evaluations = 23;
    assert!(scoped.validate().is_err());
    let completed = RunCompletion {
        schema: "jope.native_loop_completion.v1".into(),
        run: previous.run.clone(),
        config: a.clone(),
        initial_optimizer_step: 60000,
        target_optimizer_step: 66000,
        checkpoint: a.clone(),
        corpus: a.clone(),
        accepted_parent_decision: a.clone(),
        final_proposals: BTreeMap::from([(Arm::Model, a.clone()), (Arm::Shuffled, a.clone())]),
    };
    let mut next = previous.clone();
    next.run = "successor".into();
    next.initial_optimizer_step = 66000;
    next.predecessor_run = Some(a);
    next.validate().unwrap();
    verify_successor(&next, &previous, &completed).unwrap();
    next.initial_optimizer_step = 69000;
    assert!(next.validate().is_err());
    assert!(verify_successor(&next, &previous, &completed).is_err());
    next.initial_optimizer_step = 66000;
    next.run = previous.run.clone();
    assert!(verify_successor(&next, &previous, &completed).is_err());
    next.run = "successor".into();
    next.predecessor_run = None;
    assert!(next.validate().is_err());
    assert_eq!(previous.initial_optimizer_step, 60000);

    // D12: `jope.native_loop.v2` starts the accepted step-60000 parent as a weight warm
    // start. It has no predecessor and no successor, and the v1 windows are unchanged.
    let mut warm = previous.clone();
    warm.schema = warm_start::LOOP_V2.into();
    warm.validate().unwrap();
    assert_eq!(
        (
            previous.train_window(0).unwrap(),
            previous.train_window(1).unwrap()
        ),
        (
            warm_start::Window {
                initial: 60000,
                target: 63000,
                warm_start_parent_step: None
            },
            warm_start::Window {
                initial: 63000,
                target: 66000,
                warm_start_parent_step: None
            }
        )
    );
    assert_eq!(
        warm.train_window(0).unwrap(),
        warm_start::Window {
            initial: 0,
            target: 3000,
            warm_start_parent_step: Some(60000)
        }
    );
    let mut warm_successor = warm.clone();
    warm_successor.predecessor_run = Some(completed.config.clone());
    warm_successor.initial_optimizer_step = 66000;
    assert!(warm_successor.validate().is_err());
    let mut from_warm = previous.clone();
    from_warm.run = "successor".into();
    from_warm.initial_optimizer_step = 66000;
    from_warm.predecessor_run = Some(completed.config.clone());
    let error = verify_successor(&from_warm, &warm, &completed).unwrap_err();
    assert!(
        error.to_string().contains("no successor protocol"),
        "{error}"
    );
    for schema in ["jope.native_loop.v4", ""] {
        let mut unknown = previous.clone();
        unknown.schema = schema.into();
        assert!(unknown.validate().is_err());
    }
}
