use super::*;

fn digest(value: &str) -> Sha256Digest {
    Sha256Digest::of_bytes(value.as_bytes())
}

fn preflight(parent: &Sha256Digest) -> Vec<EvidencePreflightCohortV1> {
    (0..PREFLIGHT_COHORTS_REQUIRED)
        .map(|sequence| EvidencePreflightCohortV1 {
            sequence,
            task: Task::Binary,
            accepted_parent_checkpoint_sha256: parent.clone(),
            contexts: PREFLIGHT_CONTEXTS_PER_COHORT,
            requested_slots: PREFLIGHT_CONTEXTS_PER_COHORT * EVIDENCE_SLOTS_PER_CONTEXT,
            valid_slots: PREFLIGHT_CONTEXTS_PER_COHORT * EVIDENCE_SLOTS_PER_CONTEXT,
            completed: true,
        })
        .collect()
}

fn epoch(parent: &Sha256Digest) -> CausalEvidenceEpochV1 {
    CausalEvidenceEpochV1 {
        schema: CAUSAL_EVIDENCE_EPOCH_SCHEMA.to_owned(),
        task: Task::Binary,
        accepted_parent_checkpoint_sha256: parent.clone(),
        start_wave: 300,
        runtime_policy_sha256: digest("runtime"),
        proposer_policy_sha256: digest("proposer"),
        evaluator_policy_sha256: digest("evaluator"),
        catalog_eligibility_sha256: digest("eligibility"),
        requested_contexts: vec![EvidenceEpochContextV1 {
            context_identity: "context".to_owned(),
            context_sha256: digest("context"),
            requested_wave: 300,
        }],
        slots_per_context: EVIDENCE_SLOTS_PER_CONTEXT,
        preflight_cohorts: preflight(parent),
        opened_at_ms: 1,
    }
}

#[test]
fn evidence_epoch_requires_preflight_and_is_non_resettable() {
    let parent = digest("parent");
    let original = epoch(&parent);
    original.validate().unwrap();
    let mut changed = original.clone();
    changed.start_wave += 1;
    assert!(ensure_evidence_epoch_immutable(&original, &changed).is_err());
    let mut failed_preflight = original.clone();
    failed_preflight.preflight_cohorts[9].valid_slots = 500;
    assert!(failed_preflight.validate().is_err());
}

#[test]
fn proposal_cache_identity_survives_catalog_append_and_escalates() {
    let key = ContextProposalKeyV3 {
        task: Task::Regression,
        accepted_checkpoint_sha256: digest("parent"),
        context_sha256: digest("context"),
        seed: 7,
        decoding_policy_sha256: digest("decode"),
        reservoir_size: 64,
    };
    let first = context_proposal_cache_path(Path::new("/cache/catalog-a"), &key).unwrap();
    let second = context_proposal_cache_path(Path::new("/cache/catalog-a"), &key).unwrap();
    assert_eq!(first, second);
    assert_eq!(key.next_reservoir_size(), Some(128));

    let proposals = (0..EVIDENCE_SLOTS_PER_CONTEXT)
        .map(|slot| CachedContextProposalV3 {
            slot,
            candidate_sha256: digest(&format!("candidate-{slot}")),
            canonical_genome: format!("genome-{slot}"),
            decoder: "nucleus".to_owned(),
            decoder_seed: 7,
            tokens: vec![1, 2, 3],
            head_type: "ridge".to_owned(),
            reward_mean: 1.0,
            reward_uncertainty: 0.1,
            reward_lcb: 0.95,
            instability_probability: 0.1,
            selection_score: 0.85,
            candidate_risk_reason: None,
        })
        .collect::<Vec<_>>();
    let slots = (0..EVIDENCE_SLOTS_PER_CONTEXT)
        .map(|slot| ProposalSlotReceiptV3 {
            slot,
            status: ProposalSlotStatusV3::Ready,
            candidate_sha256: Some(digest(&format!("candidate-{slot}"))),
            rejections: BTreeMap::new(),
        })
        .collect();
    let artifact = ContextProposalArtifactV3 {
        schema: CONTEXT_PROPOSAL_ARTIFACT_V3_SCHEMA.to_owned(),
        key,
        generated_at_ms: 1,
        decoded_candidates: 64,
        rejection_counts: all_rejection_counts(),
        slots,
        proposals,
    };
    artifact.validate().unwrap();
}

#[test]
fn a_missing_slot_is_permanent_only_after_reservoir_256() {
    let mut artifact = ContextProposalArtifactV3 {
        schema: CONTEXT_PROPOSAL_ARTIFACT_V3_SCHEMA.to_owned(),
        key: ContextProposalKeyV3 {
            task: Task::Binary,
            accepted_checkpoint_sha256: digest("parent"),
            context_sha256: digest("context"),
            seed: 7,
            decoding_policy_sha256: digest("decode"),
            reservoir_size: 128,
        },
        generated_at_ms: 1,
        decoded_candidates: 128,
        rejection_counts: all_rejection_counts(),
        slots: (0..EVIDENCE_SLOTS_PER_CONTEXT)
            .map(|slot| ProposalSlotReceiptV3 {
                slot,
                status: ProposalSlotStatusV3::Ready,
                candidate_sha256: Some(digest(&format!("candidate-{slot}"))),
                rejections: BTreeMap::new(),
            })
            .collect(),
        proposals: (0..EVIDENCE_SLOTS_PER_CONTEXT)
            .map(|slot| CachedContextProposalV3 {
                slot,
                candidate_sha256: digest(&format!("candidate-{slot}")),
                canonical_genome: format!("genome-{slot}"),
                decoder: "nucleus".to_owned(),
                decoder_seed: 7,
                tokens: vec![1, 2, 3],
                head_type: "ridge".to_owned(),
                reward_mean: 1.0,
                reward_uncertainty: 0.1,
                reward_lcb: 0.95,
                instability_probability: 0.1,
                selection_score: 0.85,
                candidate_risk_reason: None,
            })
            .collect(),
    };
    artifact.slots[7] = ProposalSlotReceiptV3 {
        slot: 7,
        status: ProposalSlotStatusV3::PermanentFailure,
        candidate_sha256: None,
        rejections: BTreeMap::from([(ProposalRejectionReasonV3::Invalid, 256)]),
    };
    assert!(artifact.validate().is_err());
    artifact.key.reservoir_size = 256;
    artifact.proposals.pop();
    artifact.validate().unwrap();
}

fn provenance(license: Option<&str>) -> DatasetProvenanceV2 {
    DatasetProvenanceV2 {
        schema: DATASET_PROVENANCE_V2_SCHEMA.to_owned(),
        provider: "OpenML".to_owned(),
        provider_version: "1".to_owned(),
        upstream_root: "openml:1".to_owned(),
        license_spdx: license.map(str::to_owned),
        manual_license_approved: false,
        raw_sha256: BTreeMap::from([("data".to_owned(), digest("raw"))]),
        origin: DatasetOriginV2::Real,
        generator_superfamily: None,
        exact_family_sha256: digest("family"),
        near_duplicate_cluster_sha256: digest("cluster"),
    }
}

fn binary_profile() -> DatasetProfileV1 {
    DatasetProfileV1 {
        schema: DATASET_PROFILE_V1_SCHEMA.to_owned(),
        rows: 1_000,
        encoded_features: 20,
        missing_fraction: 0.0,
        class_count: Some(2),
        minority_majority_ratio: Some(0.2),
        numeric_target_distinct_finite: None,
        finite_target_fraction: None,
        size_slice: DatasetSizeSliceV1::Small,
        missingness_slice: MissingnessSliceV1::None,
        target_shape_slice: TargetShapeSliceV1::Balanced,
        embedding_novelty: 1.0,
        sensitive_or_pii_like: false,
        ambiguous_target: false,
        sparse_text: false,
        grouped_or_time_series: false,
        duplicate_or_leakage_prone: false,
    }
}

#[test]
fn license_and_task_eligibility_fail_closed() {
    assert!(
        binary_profile()
            .eligibility(Task::Binary, &provenance(Some("CC-BY-4.0")))
            .unwrap()
            .eligible
    );
    assert!(
        !binary_profile()
            .eligibility(Task::Binary, &provenance(None))
            .unwrap()
            .eligible
    );
}

#[test]
fn public_discovery_is_versioned_and_cross_provider_duplicates_cluster() {
    let openml = deterministic_openml_discovery(vec![
        OpenMlDatasetMetadataV1 {
            dataset_id: 2,
            version: 1,
            name: "later".to_owned(),
            license: Some("CC-BY-4.0".to_owned()),
            default_target_attribute: Some("target".to_owned()),
            original_data_url: None,
            download_url: "https://example.test/2".to_owned(),
            rows: 1_000,
            features: 10,
            missing_values: 0,
        },
        OpenMlDatasetMetadataV1 {
            dataset_id: 1,
            version: 1,
            name: "first".to_owned(),
            license: Some("CC0-1.0".to_owned()),
            default_target_attribute: Some("target".to_owned()),
            original_data_url: None,
            download_url: "https://example.test/1".to_owned(),
            rows: 1_000,
            features: 10,
            missing_values: 0,
        },
    ]);
    assert_eq!(openml[0].dataset_id, 1);
    let raw = BTreeMap::from([("data".to_owned(), digest("same bytes"))]);
    assert_eq!(
        cross_provider_duplicate_cluster(&raw).unwrap(),
        cross_provider_duplicate_cluster(&raw).unwrap()
    );
}

#[test]
fn synthetic_family_identity_has_only_stable_components() {
    let family = SyntheticFamilyKeyV2 {
        generator_name: "make_classification".to_owned(),
        generator_version: "2".to_owned(),
        scenario: "covariate_shift".to_owned(),
        target_family: "binary".to_owned(),
        feature_family_set: BTreeSet::from(["numeric".to_owned()]),
        shift_pathology_class: "missing_at_random".to_owned(),
    };
    assert_eq!(
        family.exact_family_sha256().unwrap(),
        family.clone().exact_family_sha256().unwrap()
    );
}

fn panel_pool() -> SealedPanelPoolV2 {
    let entries = (0..1_000)
        .map(|index| {
            let pool = match index {
                0..=699 => PanelPoolKindV2::Training,
                700..=799 => PanelPoolKindV2::Development,
                800..=949 => PanelPoolKindV2::Promotion,
                _ => PanelPoolKindV2::Audit,
            };
            SealedPanelEntryV2 {
                task: Task::Binary,
                context_identity: format!("context-{index}"),
                context_sha256: digest(&format!("context-{index}")),
                source_family_sha256: digest(&format!("family-{index}")),
                near_duplicate_cluster_sha256: digest(&format!("cluster-{index}")),
                origin: DatasetOriginV2::Real,
                pool,
                promotion_panel: (pool == PanelPoolKindV2::Promotion).then(|| "panel-a".to_owned()),
            }
        })
        .collect();
    SealedPanelPoolV2 {
        schema: SEALED_PANEL_POOL_V2_SCHEMA.to_owned(),
        entries,
        created_at_ms: 1,
    }
}

#[test]
fn sealed_panels_are_family_disjoint_and_single_use() {
    let pool = panel_pool();
    pool.validate().unwrap();
    let contexts = pool
        .promotion_panel(Task::Binary, "panel-a")
        .into_iter()
        .take(PROMOTION_MIN_CONTEXTS)
        .map(|entry| entry.context_sha256.clone())
        .collect();
    let receipt = SealedPanelConsumptionV1 {
        schema: SEALED_PANEL_CONSUMPTION_V1_SCHEMA.to_owned(),
        task: Task::Binary,
        panel: "panel-a".to_owned(),
        candidate_checkpoint_sha256: digest("candidate"),
        run_index: 0,
        seed: 9,
        context_sha256: contexts,
        consumed_at_ms: 1,
    };
    validate_panel_consumption(&pool, &receipt, &[]).unwrap();
    assert!(validate_panel_consumption(&pool, &receipt, std::slice::from_ref(&receipt)).is_err());
}

#[test]
fn sealed_panel_builder_creates_three_real_disjoint_panels_per_task() {
    let candidates = (0..5_120)
        .map(|index| SealedPanelCandidateV2 {
            task: if index % 2 == 0 {
                Task::Binary
            } else {
                Task::Regression
            },
            context_identity: format!("context-{index}"),
            context_sha256: Sha256Digest::parse(format!("{index:064x}")).unwrap(),
            source_family_sha256: Sha256Digest::parse(format!(
                "{:064x}",
                10_000 + (index / 4) * 2 + index % 2
            ))
            .unwrap(),
            near_duplicate_cluster_sha256: Sha256Digest::parse(format!(
                "{:064x}",
                20_000 + (index / 4) * 2 + index % 2
            ))
            .unwrap(),
            origin: if index < 1_024 {
                DatasetOriginV2::Real
            } else {
                DatasetOriginV2::Synthetic
            },
            base_seal_sha256: digest("base-seal"),
            catalog_sha256: digest("catalog"),
            train_sha256: digest(&format!("train-{index}")),
            test_sha256: digest(&format!("test-{index}")),
            context_facts_sha256: digest(&format!("facts-{index}")),
            prior_use_index_sha256: digest("prior-use"),
            zero_prior_use_proof_sha256: sealed_zero_prior_use_proof_v2(
                &Sha256Digest::parse(format!("{index:064x}")).unwrap(),
                &digest("base-seal"),
                &digest("catalog"),
                &digest("prior-use"),
                1,
            ),
            admission_watermark: 1,
        })
        .collect();
    let pool = build_sealed_panel_pool_v2(candidates, 1).unwrap();
    pool.validate().unwrap();
    for task in [Task::Binary, Task::Regression] {
        let panels = pool
            .entries
            .iter()
            .filter(|entry| entry.task == task && entry.pool == PanelPoolKindV2::Promotion)
            .filter_map(|entry| entry.promotion_panel.as_deref())
            .collect::<BTreeSet<_>>();
        assert_eq!(panels.len(), 3);
        assert!(panels
            .iter()
            .all(|panel| pool.promotion_panel(task, panel).len() == 128));
        let mut family_panels = BTreeMap::new();
        for entry in pool
            .entries
            .iter()
            .filter(|entry| entry.task == task && entry.pool == PanelPoolKindV2::Promotion)
        {
            let panel = entry.promotion_panel.as_deref().unwrap();
            assert!(family_panels
                .insert(&entry.source_family_sha256, panel)
                .is_none_or(|existing| existing == panel));
        }
    }
}

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
