use super::*;

fn promotion_observations(lift: f64) -> Vec<PromotionObservationV3> {
    (0..PROMOTION_MIN_CONTEXTS)
        .map(|index| PromotionObservationV3 {
            context_sha256: digest(&format!("context-{index}")),
            source_family_sha256: digest(&format!("family-{index}")),
            candidate_score: 1.0 + lift,
            accepted_score: 1.0,
            gp_score: 1.0,
            candidate_conditional_advantage: 0.02,
            accepted_conditional_advantage: 0.01,
            valid: true,
            replay_verified: true,
            candidate_failed: false,
            accepted_failed: false,
            candidate_latency_ms: 100,
            accepted_latency_ms: 100,
            strict_win: index == 0,
            equal_budget_win: index == 1,
            slices: BTreeMap::from([
                ("origin".to_owned(), "real".to_owned()),
                ("size".to_owned(), "small".to_owned()),
                ("missingness".to_owned(), "none".to_owned()),
                ("imbalance".to_owned(), "balanced".to_owned()),
                ("target_shape".to_owned(), "balanced".to_owned()),
            ]),
        })
        .collect()
}

#[test]
fn clustered_promotion_gates_require_external_and_gp_lift() {
    let accepted = evaluate_task_promotion_v3(
        Task::Binary,
        digest("parent"),
        digest("candidate"),
        digest("panel"),
        0,
        &promotion_observations(0.01),
        11,
        1,
    )
    .unwrap();
    assert!(accepted.accepted);
    let rejected = evaluate_task_promotion_v3(
        Task::Binary,
        digest("parent"),
        digest("candidate"),
        digest("panel"),
        0,
        &promotion_observations(0.0),
        11,
        1,
    )
    .unwrap();
    assert!(!rejected.accepted);
}

#[test]
fn recipe_change_v2_is_recipe_scoped_and_diagnosis_matched() {
    let previous = digest("previous-recipe");
    let diagnosis = PlateauDiagnosisV2 {
        schema: PLATEAU_DIAGNOSIS_V2_SCHEMA.to_owned(),
        task: Task::Binary,
        unchanged_parent_checkpoint_sha256: digest("parent"),
        current_recipe_identity_sha256: Some(previous.as_str().to_owned()),
        counted_candidates: ["one", "two", "three"]
            .into_iter()
            .map(|candidate| PlateauCountedCandidateV1 {
                candidate_checkpoint_sha256: digest(candidate),
                recipe_identity_sha256: Some(previous.as_str().to_owned()),
            })
            .collect(),
        recipe_change_directive_present: false,
        proposal_validity: 1.0,
        gp_disadvantage: 0.04,
        calibration_summary: "calibrated".to_owned(),
        data_slice_summary: "family-disjoint".to_owned(),
        latency_summary: "worst candidate/accepted latency ratio 1.000000".to_owned(),
        replay_summary: "3/3 replay-verified".to_owned(),
        failure_classes: BTreeSet::from([PlateauFailureClassV2::GpDisadvantage]),
        diagnosed_at_ms: 1,
    };
    diagnosis.validate().unwrap();
    let mut directive = RecipeChangeDirectiveV2 {
        schema: RECIPE_CHANGE_DIRECTIVE_V2_SCHEMA.to_owned(),
        task: Task::Binary,
        unchanged_parent_checkpoint_sha256: diagnosis.unchanged_parent_checkpoint_sha256.clone(),
        answers_diagnosis_sha256: digest("diagnosis"),
        previous_recipe_identity_sha256: Some(previous.clone()),
        next_recipe_identity_sha256: digest("next-recipe"),
        failure_classes: diagnosis.failure_classes.clone(),
        changed_knobs: BTreeSet::from([RecipeKnobV2::ExactCausalPairGpComparatorMixture]),
        created_at_ms: 2,
    };
    directive.validate(&diagnosis).unwrap();

    directive.changed_knobs = BTreeSet::from([RecipeKnobV2::HardNegativeValiditySampling]);
    assert!(directive.validate(&diagnosis).is_err());
    directive.changed_knobs = BTreeSet::from([RecipeKnobV2::ExactCausalPairGpComparatorMixture]);
    directive.next_recipe_identity_sha256 = previous;
    assert!(directive.validate(&diagnosis).is_err());
}

#[test]
fn packed_v3_rejects_observations_copy_and_calibration_falls_back() {
    let column = |name: &str, rows| PackedColumnV3 {
        path: PathBuf::from(name),
        sha256: digest(name),
        dtype: "u1".to_owned(),
        rows,
    };
    let mut corpus = PackedCorpusV3 {
        schema: PACKED_CORPUS_V3_SCHEMA.to_owned(),
        watermark: 10,
        catalog_binding_sha256: digest("catalog"),
        selected_row_index: column("selected.u64", 10),
        task_column: column("tasks.u1", 10),
        split_column: column("splits.u1", 10),
        exact_pair_sidecar: column("pairs.jsonl", 0),
        label_shards: vec![PackedLabelShardV3 {
            first_receipt: 1,
            last_receipt: 10,
            rows: 10,
            path: PathBuf::from("labels/one.shard"),
            sha256: digest("shard"),
            reused_from_corpus_sha256: None,
        }],
        created_at_ms: 1,
    };
    corpus.validate().unwrap();
    corpus.selected_row_index.path = PathBuf::from("observations.jsonl");
    assert!(corpus.validate().is_err());

    let fallback = ranking_fallback(&CalibrationEvidenceV1 {
        error_correlation_lower_95: -0.1,
        interval_coverage: 0.8,
        target_coverage: 0.95,
        native_head_accuracy: 0.51,
        frequency_baseline_accuracy: 0.50,
    })
    .unwrap();
    assert_eq!(
        fallback.uncertainty,
        UncertaintyRankingSourceV1::EmpiricalEnsembleDispersion
    );
    assert!(!fallback.native_head_enabled);
}

#[test]
fn recursive_claim_requires_a_child_of_an_accepted_child() {
    let first = AcceptedGenerationV2 {
        task: Task::Binary,
        parent_checkpoint_sha256: digest("root"),
        child_checkpoint_sha256: digest("child-1"),
        independent_external_passes: vec![digest("pass-1"), digest("pass-2"), digest("pass-3")],
        accepted_at_ms: 1,
    };
    let second = AcceptedGenerationV2 {
        task: Task::Binary,
        parent_checkpoint_sha256: first.child_checkpoint_sha256.clone(),
        child_checkpoint_sha256: digest("child-2"),
        independent_external_passes: vec![digest("pass-4"), digest("pass-5"), digest("pass-6")],
        accepted_at_ms: 2,
    };
    let progress = FlywheelProgressV2 {
        schema: FLYWHEEL_PROGRESS_V2_SCHEMA.to_owned(),
        generated_at_ms: 3,
        accepted_generations: vec![first, second],
        causal_evidence_ledger_sha256: digest("ledger"),
        external_evaluation_sha256: BTreeMap::new(),
        throughput: ThroughputProgressV2 {
            queue_depth: 4_096,
            ready_reserve: 4_096,
            dynamic_target: 4_096,
            p95_context_to_ready_ms: 1,
            freeze_to_publish_ms: 1,
            corpus_lag_labels: 0,
            supply_target_fraction: 1_000,
            disk_used_percent: 50,
            expansion_paused: false,
        },
        catalog_coverage: BTreeMap::from([(Task::Binary, 0.5)]),
        real_data_share: BTreeMap::from([(Task::Binary, 0.4)]),
        plateau: BTreeMap::new(),
        self_improving: true,
        recursive_improvement_proven: true,
    };
    progress.validate().unwrap();
}
