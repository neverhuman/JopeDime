use super::*;

#[test]
fn promotion_compare_and_swap_mutates_only_one_task_pointer() {
    let directory = tempdir().unwrap();
    let root = directory.path();
    fs::create_dir_all(root.join("training")).unwrap();
    let parent_path = root.join("parent.model.pt");
    let child_path = root.join("child.model.pt");
    fs::write(&parent_path, b"parent").unwrap();
    fs::write(&child_path, b"child").unwrap();
    let parent_digest = Sha256Digest::parse(sha256_file(&parent_path).unwrap()).unwrap();
    let child_digest = Sha256Digest::parse(sha256_file(&child_path).unwrap()).unwrap();
    atomic_json(
        &root.join("training/accepted-by-task.json"),
        &json!({
            "schema": "JopeSelfplayAcceptedByTask.v1",
            "global_checkpoint_sha256": parent_digest,
            "global_pointer": null,
            "tasks": {
                "binary": {
                    "checkpoint": parent_path,
                    "checkpoint_sha256": parent_digest,
                    "promotion_tier": null,
                    "source": "test",
                    "updated_at_ms": 1
                },
                "regression": {
                    "checkpoint": parent_path,
                    "checkpoint_sha256": parent_digest,
                    "promotion_tier": null,
                    "source": "test",
                    "updated_at_ms": 1
                }
            },
            "updated_at_ms": 1
        }),
    )
    .unwrap();
    let args = Args {
        root: root.to_owned(),
        scq: PathBuf::from("/bin/false"),
        runs_root: root.join("runs"),
        trainer_script: root.join("trainer"),
        common_script: root.join("common"),
        packed_script: root.join("packed"),
        prime_source: root.join("prime"),
        submit: false,
        once: true,
        interval_seconds: 1,
        seed_base: 1,
    };
    promote_task_pointer(
        &args,
        Task::Regression,
        &AcceptedPointer {
            digest: parent_digest.clone(),
            path: parent_path,
        },
        &ModelRef {
            accepted_parent_checkpoint_sha256: parent_digest.clone(),
            checkpoint_sha256: child_digest.clone(),
            checkpoint_path: child_path,
        },
        &[
            "receipt-1".to_owned(),
            "receipt-2".to_owned(),
            "receipt-3".to_owned(),
        ],
    )
    .unwrap();
    let pointer = read_json(&root.join("training/accepted-by-task.json")).unwrap();
    assert_eq!(
        pointer["tasks"]["regression"]["checkpoint_sha256"],
        child_digest.as_str()
    );
    assert_eq!(
        pointer["tasks"]["binary"]["checkpoint_sha256"],
        parent_digest.as_str()
    );
    assert!(pointer["global_checkpoint_sha256"].is_null());
}

#[test]
fn promotion_panels_are_consumed_only_when_each_run_is_scheduled() {
    use jope_dime_regressor::selfplay_shadow::{
        build_sealed_panel_pool_v2, ClusteredConfidenceBoundsV3, DatasetOriginV2,
        SealedPanelCandidateV2, TaskPromotionMetricsV3,
    };

    let directory = tempdir().unwrap();
    let root = directory.path();
    fs::create_dir_all(root.join("training/sealed-panels/consumed")).unwrap();
    let candidates = (0..5_120_usize)
        .map(|index| SealedPanelCandidateV2 {
            task: if index % 2 == 0 {
                Task::Binary
            } else {
                Task::Regression
            },
            context_identity: format!("context-{index}"),
            context_sha256: Sha256Digest::parse(format!("{index:064x}")).unwrap(),
            source_family_sha256: Sha256Digest::parse(format!("{:064x}", index + 10_000)).unwrap(),
            near_duplicate_cluster_sha256: Sha256Digest::parse(format!("{:064x}", index + 20_000))
                .unwrap(),
            origin: DatasetOriginV2::Real,
            base_seal_sha256: digest("base-seal"),
            catalog_sha256: digest("catalog"),
            train_sha256: digest(&format!("train-{index}")),
            test_sha256: digest(&format!("test-{index}")),
            context_facts_sha256: digest(&format!("facts-{index}")),
            prior_use_index_sha256: digest("prior-use"),
            zero_prior_use_proof_sha256:
                jope_dime_regressor::selfplay_shadow::sealed_zero_prior_use_proof_v2(
                    &Sha256Digest::parse(format!("{index:064x}")).unwrap(),
                    &digest("base-seal"),
                    &digest("catalog"),
                    &digest("prior-use"),
                    1,
                ),
            admission_watermark: 1,
        })
        .collect::<Vec<_>>();
    let pool = build_sealed_panel_pool_v2(candidates, 1).unwrap();
    atomic_json(
        &root.join("training/sealed-panel-pool.v2.json"),
        &serde_json::to_value(pool).unwrap(),
    )
    .unwrap();
    let parent_path = root.join("parent.model.pt");
    let candidate_path = root.join("candidate.model.pt");
    fs::write(&parent_path, b"parent").unwrap();
    fs::write(&candidate_path, b"candidate").unwrap();
    let parent = Sha256Digest::parse(sha256_file(&parent_path).unwrap()).unwrap();
    let candidate = Sha256Digest::parse(sha256_file(&candidate_path).unwrap()).unwrap();
    atomic_json(
        &root.join("training/accepted-by-task.json"),
        &json!({
            "schema": "JopeSelfplayAcceptedByTask.v1",
            "global_checkpoint_sha256": parent,
            "global_pointer": null,
            "tasks": {
                "binary": {
                    "checkpoint": parent_path,
                    "checkpoint_sha256": parent,
                    "promotion_tier": null,
                    "source": "fixture",
                    "updated_at_ms": 1
                },
                "regression": {
                    "checkpoint": parent_path,
                    "checkpoint_sha256": parent,
                    "promotion_tier": null,
                    "source": "fixture",
                    "updated_at_ms": 1
                }
            },
            "updated_at_ms": 1,
        }),
    )
    .unwrap();
    let args = Args {
        root: root.to_owned(),
        scq: PathBuf::from("/bin/false"),
        runs_root: root.join("runs"),
        trainer_script: root.join("trainer"),
        common_script: root.join("common"),
        packed_script: root.join("packed"),
        prime_source: root.join("prime"),
        submit: false,
        once: true,
        interval_seconds: 1,
        seed_base: 1,
    };
    schedule_promotion_run(&args, Task::Binary, &parent, &candidate, 10, 0).unwrap();
    let consumed = || {
        fs::read_dir(root.join("training/sealed-panels/consumed"))
            .unwrap()
            .count()
    };
    assert_eq!(consumed(), 1);
    assert!(root
        .join(format!(
            "training/task-promotion-evaluations/pending/binary-{}-run0.v3.json",
            candidate
        ))
        .is_file());
    assert!(!root
        .join(format!(
            "training/task-promotion-evaluations/pending/binary-{}-run1.v3.json",
            candidate
        ))
        .exists());

    schedule_promotion_run(&args, Task::Binary, &parent, &candidate, 10, 0).unwrap();
    assert_eq!(consumed(), 1, "run zero scheduling must be idempotent");
    let consumption_path = fs::read_dir(root.join("training/sealed-panels/consumed"))
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    let consumption_sha = Sha256Digest::parse(sha256_file(&consumption_path).unwrap()).unwrap();
    let evaluation = TaskPromotionEvaluationV3 {
        schema: "TaskPromotionEvaluation.v3".to_owned(),
        task: Task::Binary,
        accepted_parent_checkpoint_sha256: parent.clone(),
        candidate_checkpoint_sha256: candidate.clone(),
        panel_consumption_sha256: consumption_sha,
        run_index: 0,
        metrics: TaskPromotionMetricsV3 {
            contexts: 128,
            confidence: ClusteredConfidenceBoundsV3 {
                resamples: 10_000,
                seed: 11,
                candidate_vs_accepted_lower: -0.1,
                candidate_vs_gp_lower: -0.1,
                conditional_advantage_lower: 0.01,
            },
            maximum_slice_regression: 0.1,
            validity: 1.0,
            replay_verification: 1.0,
            failure_rate_increase: 0.0,
            latency_ratio: 1.0,
            strict_wins: 1,
            equal_budget_wins: 1,
        },
        accepted: false,
        rejection_reasons: vec!["fixture rejection".to_owned()],
        evaluated_at_ms: 2,
    };
    evaluation.validate().unwrap();
    atomic_json(
        &root
            .join("training/task-promotion-evaluations/rejected")
            .join("run0.v3.json"),
        &serde_json::to_value(evaluation).unwrap(),
    )
    .unwrap();
    let mut state = ControllerState::default();
    state.production_candidates.insert(
        Task::Binary,
        ModelRef {
            accepted_parent_checkpoint_sha256: parent.clone(),
            checkpoint_sha256: candidate.clone(),
            checkpoint_path: candidate_path,
        },
    );
    let accepted = accepted_pointers(root).unwrap();
    assert!(apply_task_canary_acceptances(&args, &mut state, &accepted)
        .unwrap()
        .is_empty());
    assert!(!state.production_candidates.contains_key(&Task::Binary));
    assert_eq!(consumed(), 1, "rejection must not consume run one or two");
    assert!(!root
        .join(format!(
            "training/task-promotion-evaluations/pending/binary-{}-run1.v3.json",
            candidate
        ))
        .exists());
}

#[test]
fn first_optional_promotion_uses_absent_to_candidate_cas() {
    let directory = tempdir().unwrap();
    let root = directory.path();
    fs::create_dir_all(root.join("training/bootstrap-evaluation-anchors")).unwrap();
    let anchor_path = root.join("multiclass-anchor.model.pt");
    let child_path = root.join("multiclass-child.model.pt");
    fs::write(&anchor_path, b"anchor").unwrap();
    fs::write(&child_path, b"child").unwrap();
    let anchor_digest = Sha256Digest::parse(sha256_file(&anchor_path).unwrap()).unwrap();
    let child_digest = Sha256Digest::parse(sha256_file(&child_path).unwrap()).unwrap();
    atomic_json(
        &root.join("training/bootstrap-evaluation-anchors/multiclass.v1.json"),
        &json!({
            "schema": "TaskBootstrapEvaluationAnchor.v1",
            "task": "multiclass",
            "checkpoint": anchor_path,
            "checkpoint_sha256": anchor_digest,
            "accepted": false,
            "created_at_ms": 1,
        }),
    )
    .unwrap();
    atomic_json(
        &root.join("training/accepted-by-task.json"),
        &json!({
            "schema": "JopeSelfplayAcceptedByTask.v1",
            "global_checkpoint_sha256": null,
            "global_pointer": null,
            "tasks": {},
            "updated_at_ms": 1,
        }),
    )
    .unwrap();
    let args = Args {
        root: root.to_owned(),
        scq: PathBuf::from("/bin/false"),
        runs_root: root.join("runs"),
        trainer_script: root.join("trainer"),
        common_script: root.join("common"),
        packed_script: root.join("packed"),
        prime_source: root.join("prime"),
        submit: false,
        once: true,
        interval_seconds: 1,
        seed_base: 1,
    };
    promote_task_pointer(
        &args,
        Task::Multiclass,
        &AcceptedPointer {
            digest: anchor_digest.clone(),
            path: anchor_path,
        },
        &ModelRef {
            accepted_parent_checkpoint_sha256: anchor_digest,
            checkpoint_sha256: child_digest.clone(),
            checkpoint_path: child_path,
        },
        &["one".to_owned(), "two".to_owned(), "three".to_owned()],
    )
    .unwrap();
    let pointer = read_json(&root.join("training/accepted-by-task.json")).unwrap();
    assert_eq!(
        pointer["tasks"]["multiclass"]["checkpoint_sha256"],
        child_digest.as_str()
    );
    let anchor =
        read_json(&root.join("training/bootstrap-evaluation-anchors/multiclass.v1.json")).unwrap();
    assert_eq!(anchor["accepted"], false);
}

#[test]
fn three_sequential_passes_advance_the_pointer_once() {
    use jope_dime_regressor::selfplay_shadow::{
        build_sealed_panel_pool_v2, ClusteredConfidenceBoundsV3, DatasetOriginV2,
        SealedPanelCandidateV2, TaskPromotionMetricsV3,
    };

    let directory = tempdir().unwrap();
    let root = directory.path();
    fs::create_dir_all(root.join("training/sealed-panels/consumed")).unwrap();
    let pool = build_sealed_panel_pool_v2(
        (0..5_120_usize)
            .map(|index| SealedPanelCandidateV2 {
                task: if index % 2 == 0 {
                    Task::Binary
                } else {
                    Task::Regression
                },
                context_identity: format!("context-{index}"),
                context_sha256: Sha256Digest::parse(format!("{index:064x}")).unwrap(),
                source_family_sha256: Sha256Digest::parse(format!("{:064x}", index + 10_000))
                    .unwrap(),
                near_duplicate_cluster_sha256: Sha256Digest::parse(format!(
                    "{:064x}",
                    index + 20_000
                ))
                .unwrap(),
                origin: DatasetOriginV2::Real,
                base_seal_sha256: digest("base-seal"),
                catalog_sha256: digest("catalog"),
                train_sha256: digest(&format!("train-{index}")),
                test_sha256: digest(&format!("test-{index}")),
                context_facts_sha256: digest(&format!("facts-{index}")),
                prior_use_index_sha256: digest("prior-use"),
                zero_prior_use_proof_sha256:
                    jope_dime_regressor::selfplay_shadow::sealed_zero_prior_use_proof_v2(
                        &Sha256Digest::parse(format!("{index:064x}")).unwrap(),
                        &digest("base-seal"),
                        &digest("catalog"),
                        &digest("prior-use"),
                        1,
                    ),
                admission_watermark: 1,
            })
            .collect(),
        1,
    )
    .unwrap();
    atomic_json(
        &root.join("training/sealed-panel-pool.v2.json"),
        &serde_json::to_value(pool).unwrap(),
    )
    .unwrap();
    let parent_path = root.join("parent.model.pt");
    let candidate_path = root.join("candidate.model.pt");
    fs::write(&parent_path, b"parent").unwrap();
    fs::write(&candidate_path, b"candidate").unwrap();
    let parent = Sha256Digest::parse(sha256_file(&parent_path).unwrap()).unwrap();
    let candidate = Sha256Digest::parse(sha256_file(&candidate_path).unwrap()).unwrap();
    atomic_json(
        &root.join("training/accepted-by-task.json"),
        &json!({
            "schema": "JopeSelfplayAcceptedByTask.v1",
            "global_checkpoint_sha256": parent,
            "global_pointer": null,
            "tasks": {
                "binary": {"checkpoint": parent_path, "checkpoint_sha256": parent},
                "regression": {"checkpoint": parent_path, "checkpoint_sha256": parent}
            },
            "updated_at_ms": 1,
        }),
    )
    .unwrap();
    let args = Args {
        root: root.to_owned(),
        scq: PathBuf::from("/bin/false"),
        runs_root: root.join("runs"),
        trainer_script: root.join("trainer"),
        common_script: root.join("common"),
        packed_script: root.join("packed"),
        prime_source: root.join("prime"),
        submit: false,
        once: true,
        interval_seconds: 1,
        seed_base: 1,
    };
    for run_index in 0..3 {
        schedule_promotion_run(&args, Task::Binary, &parent, &candidate, 100, run_index).unwrap();
    }
    let consumptions = fs::read_dir(root.join("training/sealed-panels/consumed"))
        .unwrap()
        .filter_map(|entry| entry.ok())
        .map(|entry| {
            let path = entry.path();
            let receipt: SealedPanelConsumptionV1 =
                serde_json::from_value(read_json(&path).unwrap()).unwrap();
            (path, receipt)
        })
        .collect::<Vec<_>>();
    assert_eq!(consumptions.len(), 3);
    for (path, consumption) in consumptions {
        let evaluation = TaskPromotionEvaluationV3 {
            schema: "TaskPromotionEvaluation.v3".to_owned(),
            task: Task::Binary,
            accepted_parent_checkpoint_sha256: parent.clone(),
            candidate_checkpoint_sha256: candidate.clone(),
            panel_consumption_sha256: Sha256Digest::parse(sha256_file(&path).unwrap()).unwrap(),
            run_index: consumption.run_index,
            metrics: TaskPromotionMetricsV3 {
                contexts: 128,
                confidence: ClusteredConfidenceBoundsV3 {
                    resamples: 10_000,
                    seed: consumption.seed,
                    candidate_vs_accepted_lower: 0.01,
                    candidate_vs_gp_lower: 0.01,
                    conditional_advantage_lower: 0.01,
                },
                maximum_slice_regression: 0.0,
                validity: 1.0,
                replay_verification: 1.0,
                failure_rate_increase: 0.0,
                latency_ratio: 1.0,
                strict_wins: 1,
                equal_budget_wins: 1,
            },
            accepted: true,
            rejection_reasons: Vec::new(),
            evaluated_at_ms: consumption.run_index as u128 + 1,
        };
        evaluation.validate().unwrap();
        atomic_json(
            &root
                .join("training/task-promotion-evaluations/accepted")
                .join(format!("run{}.v3.json", consumption.run_index)),
            &serde_json::to_value(evaluation).unwrap(),
        )
        .unwrap();
    }
    let mut state = ControllerState::default();
    state.production_candidates.insert(
        Task::Binary,
        ModelRef {
            accepted_parent_checkpoint_sha256: parent.clone(),
            checkpoint_sha256: candidate.clone(),
            checkpoint_path: candidate_path,
        },
    );
    let accepted = accepted_pointers(root).unwrap();
    assert_eq!(
        apply_task_canary_acceptances(&args, &mut state, &accepted).unwrap(),
        vec![Task::Binary]
    );
    let pointer = read_json(&root.join("training/accepted-by-task.json")).unwrap();
    assert_eq!(
        pointer["tasks"]["binary"]["checkpoint_sha256"],
        candidate.as_str()
    );
    assert!(!state.production_candidates.contains_key(&Task::Binary));
}
