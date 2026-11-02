use super::*;
use feat_math::evolve::random_canonical_pipeline;
use tempfile::TempDir;

fn hash(number: u64) -> String {
    format!("{number:064x}")
}

fn identities(count: usize, seed_start: u64) -> Vec<CandidateIdentityV1> {
    let mut result = Vec::new();
    let mut seen = BTreeSet::new();
    let mut seed = seed_start;
    while result.len() < count {
        let canonical = random_canonical_pipeline(seed, 6);
        seed += 1;
        let identity = CandidateIdentityV1::from_canonical(canonical).unwrap();
        if seen.insert(identity.semantic_sha256.clone()) {
            result.push(identity);
        }
    }
    result
}

fn seed_request() -> SeedInferenceRequestV2 {
    SeedInferenceRequestV2 {
        schema: SEED_REQUEST_V2_SCHEMA.into(),
        learning_wave_sha256: hash(1),
        dataset_encoding_sha256: hash(2),
        checkpoint_sha256: hash(3),
        corrector_sha256: hash(4),
        registry_sha256: hash(5),
        vocabulary_sha256: hash(6),
        state_snapshot_sha256: hash(7),
        draw_seed: 17,
        required_count: MODEL_PROPOSAL_COUNT,
    }
}

fn seed_response(identities: &[CandidateIdentityV1]) -> SeedInferenceResponseV2 {
    let request = seed_request();
    SeedInferenceResponseV2 {
        schema: SEED_RESPONSE_V2_SCHEMA.into(),
        request_sha256: request.request_sha256().unwrap(),
        proposals: identities
            .iter()
            .enumerate()
            .map(|(index, candidate)| ModelProposalV2 {
                candidate: candidate.clone(),
                reward_prediction: index as f64,
                disagreement_prediction: (index % 17) as f64,
                novelty_prediction: (index % 11) as f64,
                temperature_logit: (index % 7) as f64 / 10.0,
            })
            .collect(),
    }
}

fn gate(kind: WaveGateKindV1, manifest_number: u64) -> WaveGateReceiptV1 {
    let dataset_count = match kind {
        WaveGateKindV1::Offline119 => 119,
        WaveGateKindV1::Prospective64 => 64,
        WaveGateKindV1::RecurringTrainOnly | WaveGateKindV1::Corrector => 2_090,
    };
    WaveGateReceiptV1 {
        kind: kind.clone(),
        receipt_sha256: hash(100 + manifest_number),
        dataset_manifest_sha256: hash(manifest_number),
        dataset_count,
        dataset_sha256s: (0..dataset_count)
            .map(|index| hash(manifest_number * 100_000 + index as u64))
            .collect(),
        train_only: matches!(
            kind,
            WaveGateKindV1::RecurringTrainOnly | WaveGateKindV1::Corrector
        ),
        passed: true,
        genome_validity: 0.995,
        mean_normalized_regret_reduction: 0.11,
        median_robust_r2_improvement: 0.01,
        median_wall_time_ratio: 1.0,
        safety_delta: 0.0,
        pairwise_accuracy_delta: 0.0,
        normalized_regret_bootstrap_lower: 0.001,
        capture_1_delta: 0.0,
        capture_3_delta: 0.0,
        validity_delta: 0.0,
        median_time_to_champion_delta: 0.0,
        corrector_grouped_time_forward_passed: true,
        corrector_pop256_system_passed: true,
    }
}

fn release(name: &str, number: u64) -> ArtifactReleaseV1 {
    ArtifactReleaseV1 {
        release: name.into(),
        sha256: hash(number),
    }
}

fn bootstrap_wave() -> LearningWaveManifestV1 {
    LearningWaveManifestV1 {
        schema: WAVE_SCHEMA.into(),
        parent_wave_sha256: None,
        accepted_at_unix_seconds: 10_000,
        training_label_cutoff: 50_000,
        replay_manifest_sha256: hash(10),
        corpus_manifest_sha256: hash(11),
        policy_checkpoint_sha256: hash(12),
        resume_checkpoint_sha256: hash(13),
        corrector_sha256: hash(14),
        registry: release("registry-1", 15),
        vocabulary: release("vocabulary-1", 16),
        encoding_catalog: release("encoding-1", 17),
        evaluator: release("evaluator-1", 18),
        sidecar: release("sidecar-1", 19),
        gate_receipts: vec![
            gate(WaveGateKindV1::Offline119, 20),
            gate(WaveGateKindV1::Prospective64, 21),
            gate(WaveGateKindV1::Corrector, 22),
        ],
        labels_since_parent: 50_000,
        seconds_since_parent: 10_000,
        synthetic_data_ratio: 0.1,
        pilot_only: false,
        publication_sha256: String::new(),
    }
    .seal()
    .unwrap()
}

fn recurring_wave(
    parent_sha256: String,
    parent: &LearningWaveManifestV1,
) -> LearningWaveManifestV1 {
    LearningWaveManifestV1 {
        schema: WAVE_SCHEMA.into(),
        parent_wave_sha256: Some(parent_sha256),
        accepted_at_unix_seconds: parent.accepted_at_unix_seconds + MIN_WAVE_AGE_SECONDS,
        training_label_cutoff: parent.training_label_cutoff + MIN_WAVE_LABELS,
        replay_manifest_sha256: hash(30),
        corpus_manifest_sha256: hash(31),
        policy_checkpoint_sha256: hash(32),
        resume_checkpoint_sha256: hash(33),
        corrector_sha256: hash(34),
        registry: release("registry-1", 15),
        vocabulary: release("vocabulary-1", 16),
        encoding_catalog: release("encoding-1", 17),
        evaluator: release("evaluator-1", 18),
        sidecar: release("sidecar-1", 19),
        gate_receipts: vec![
            gate(WaveGateKindV1::RecurringTrainOnly, 23),
            gate(WaveGateKindV1::Corrector, 24),
        ],
        labels_since_parent: MIN_WAVE_LABELS,
        seconds_since_parent: MIN_WAVE_AGE_SECONDS,
        synthetic_data_ratio: 0.1,
        pilot_only: false,
        publication_sha256: String::new(),
    }
    .seal()
    .unwrap()
}

fn state_with_roles(ids: &[CandidateIdentityV1]) -> DatasetLearningStateV2 {
    let track_1_history = vec![
        TrackHistoryEntryV2 {
            sequence: 1,
            candidate: ids[0].clone(),
            authority: SelectionAuthorityV2::TrainOnlyExact,
            robust_fitness: 0.1,
            receipt_sha256: hash(40),
            committed_at_unix_seconds: 1,
        },
        TrackHistoryEntryV2 {
            sequence: 2,
            candidate: ids[2].clone(),
            authority: SelectionAuthorityV2::TrainOnlyExact,
            robust_fitness: 0.2,
            receipt_sha256: hash(41),
            committed_at_unix_seconds: 2,
        },
    ];
    let track_2_history = vec![TrackHistoryEntryV2 {
        sequence: 1,
        candidate: ids[3].clone(),
        authority: SelectionAuthorityV2::FeedbackTestTrack2,
        robust_fitness: 0.3,
        receipt_sha256: hash(42),
        committed_at_unix_seconds: 2,
    }];
    DatasetLearningStateV2 {
        schema: DATASET_STATE_SCHEMA.into(),
        dataset_sha256: hash(50),
        state_sequence: 0,
        observations: vec![],
        label_revisions: vec![],
        track_1_history,
        track_2_history,
        global_cv_frontier: (0..5)
            .map(|index| GlobalCvFrontierMemberV2 {
                candidate: ids[index].clone(),
                robust_fitness: 1.0 - index as f64 / 10.0,
                comparable_fold_count: 7,
                confirmed: index == 0,
                receipt_sha256: hash(60 + index as u64),
            })
            .collect(),
        confirmed_cv_champion: Some(ids[0].clone()),
        // Shared semantic anchor is intentional.
        latest_exact_track_1_frontier: Some(ids[0].clone()),
        latest_model_prediction: Some(PredictionSnapshotV2 {
            candidate: ids[33].clone(),
            checkpoint_sha256: hash(70),
            prediction: 0.5,
            recorded_at_unix_seconds: 3,
        }),
        latest_corrector_prediction: Some(PredictionSnapshotV2 {
            candidate: ids[1].clone(),
            checkpoint_sha256: hash(71),
            prediction: 0.6,
            recorded_at_unix_seconds: 3,
        }),
        legacy_anchor_import: None,
        counters: LearningCountersV2::default(),
        struggle: StruggleStateV2::default(),
        scheduling: SchedulingMetadataV2::default(),
    }
}

fn generation_for_state(
    ids: &[CandidateIdentityV1],
    state: &DatasetLearningStateV2,
    wave_sha256: String,
) -> Generation0ReceiptV3 {
    let archive = ids[0..32].to_vec();
    let model = ids[32..96]
        .iter()
        .enumerate()
        .map(|(index, candidate)| RetainedModelProposalV2 {
            proposal: ModelProposalV2 {
                candidate: candidate.clone(),
                reward_prediction: index as f64,
                disagreement_prediction: index as f64,
                novelty_prediction: index as f64,
                temperature_logit: index as f64,
            },
            bucket: if index < 32 {
                ModelRetentionBucketV2::Reward
            } else if index < 48 {
                ModelRetentionBucketV2::Disagreement
            } else if index < 56 {
                ModelRetentionBucketV2::Novelty
            } else {
                ModelRetentionBucketV2::SeededTemperature
            },
            bucket_rank: if index < 32 {
                index
            } else if index < 48 {
                index - 32
            } else if index < 56 {
                index - 48
            } else {
                index - 56
            },
        })
        .collect();
    let families = [
        GprKernelFamilyV3::Rbf,
        GprKernelFamilyV3::Matern12,
        GprKernelFamilyV3::Matern32,
        GprKernelFamilyV3::Matern52,
        GprKernelFamilyV3::RationalQuadratic,
        GprKernelFamilyV3::Periodic,
        GprKernelFamilyV3::Linear,
        GprKernelFamilyV3::WhiteComposite,
    ];
    let gpr = ids[96..104].iter().cloned().zip(families).collect();
    let mutation = ids[104..164]
        .iter()
        .enumerate()
        .map(|(index, candidate)| {
            (
                candidate.clone(),
                1_000 + index as u64,
                ids[index % 32].semantic_sha256.clone(),
            )
        })
        .collect();
    let crossover = ids[164..224]
        .iter()
        .enumerate()
        .map(|(index, candidate)| {
            (
                candidate.clone(),
                2_000 + index as u64,
                ids[index % 31].semantic_sha256.clone(),
                ids[index % 31 + 1].semantic_sha256.clone(),
            )
        })
        .collect();
    let random = ids[224..256]
        .iter()
        .enumerate()
        .map(|(index, candidate)| (candidate.clone(), 3_000 + index as u64))
        .collect();
    let roles = vec![
        (
            GenerationRoleV3::ConfirmedCvChampion,
            ids[0].semantic_sha256.clone(),
        ),
        (
            GenerationRoleV3::LatestExactTrack1Frontier,
            ids[0].semantic_sha256.clone(),
        ),
        (
            GenerationRoleV3::CheckpointRewardTop1,
            ids[32].semantic_sha256.clone(),
        ),
        (
            GenerationRoleV3::LatestModelPrediction,
            ids[33].semantic_sha256.clone(),
        ),
        (
            GenerationRoleV3::LatestCorrectorPrediction,
            ids[1].semantic_sha256.clone(),
        ),
        (
            GenerationRoleV3::LatestTrack1Winner,
            ids[2].semantic_sha256.clone(),
        ),
        (
            GenerationRoleV3::LatestTrack2Winner,
            ids[3].semantic_sha256.clone(),
        ),
    ]
    .into_iter()
    .map(|(role, semantic_sha256)| GenerationRoleBindingV3 {
        role,
        semantic_sha256,
    })
    .collect();
    assemble_generation_zero_v3(
        wave_sha256,
        state.snapshot_sha256().unwrap(),
        state.dataset_sha256.clone(),
        hash(80),
        hash(81),
        hash(82),
        43,
        archive,
        model,
        gpr,
        mutation,
        crossover,
        random,
        roles,
    )
    .unwrap()
}

#[test]
fn v1_receipts_remain_readable_but_are_not_promoted() {
    assert!(matches!(
        read_seed_inference(br#"{"schema":"SeedInferenceResponse.v1","seeds":[]}"#).unwrap(),
        ReadableSeedInference::V1(_)
    ));
    assert!(matches!(
        read_generation_zero(br#"{"schema":"Generation0Receipt.v2","population":[]}"#).unwrap(),
        ReadableGeneration0::Legacy(_)
    ));
    assert!(matches!(
        read_admission(br#"{"schema":"AttemptAdmissionManifest.v2"}"#).unwrap(),
        ReadableAdmission::Legacy(_)
    ));
    assert!(matches!(
        read_telemetry(br#"{"schema":"RatchetLiveTelemetry.v1"}"#).unwrap(),
        ReadableTelemetry::Legacy(_)
    ));
}

#[test]
fn rust_model_retention_is_deterministic_and_exact() {
    let ids = identities(100, 1);
    let request = seed_request();
    let response = seed_response(&ids[..96]);
    let first = response.retain_64(&request, &BTreeSet::new()).unwrap();
    let second = response.retain_64(&request, &BTreeSet::new()).unwrap();
    assert_eq!(first, second);
    let counts = first.iter().fold(BTreeMap::new(), |mut counts, item| {
        *counts.entry(item.bucket.clone()).or_insert(0) += 1;
        counts
    });
    assert_eq!(counts[&ModelRetentionBucketV2::Reward], 32);
    assert_eq!(counts[&ModelRetentionBucketV2::Disagreement], 16);
    assert_eq!(counts[&ModelRetentionBucketV2::Novelty], 8);
    assert_eq!(counts[&ModelRetentionBucketV2::SeededTemperature], 8);

    let excluded = ids[..33]
        .iter()
        .map(|identity| identity.semantic_sha256.clone())
        .collect();
    assert!(response.retain_64(&request, &excluded).is_err());
}

#[test]
fn generation_v3_enforces_topology_and_shared_anchor_roles() {
    let ids = identities(256, 10_000);
    let state = state_with_roles(&ids);
    let generation = generation_for_state(&ids, &state, hash(90));
    generation.validate_against_state(&state).unwrap();
    assert_eq!(
        generation.role_bindings[0].semantic_sha256,
        generation.role_bindings[1].semantic_sha256
    );
    assert_eq!(
        generation
            .candidates
            .iter()
            .filter(|candidate| candidate.candidate.semantic_sha256
                == generation.role_bindings[0].semantic_sha256)
            .count(),
        1
    );

    let mut drifted = generation.clone();
    drifted.candidates[100].candidate = drifted.candidates[99].candidate.clone();
    assert!(drifted.validate().is_err());
}

#[test]
fn archive_and_offspring_refills_are_deterministic_and_fail_closed() {
    let ids = identities(96, 15_000);
    let state = state_with_roles(&ids);
    let archive = deterministic_archive_32(&state, &ids).unwrap();
    assert_eq!(archive.len(), 32);
    assert_eq!(archive[0], ids[0]);
    assert_eq!(archive[1], ids[1]);
    assert_eq!(archive[2], ids[2]);
    assert_eq!(archive[3], ids[3]);

    let occupied = archive
        .iter()
        .map(|candidate| candidate.semantic_sha256.clone())
        .collect::<BTreeSet<_>>();
    let first = build_bounded_offspring_pools(&archive, &occupied, 123, 6, 8_192).unwrap();
    let second = build_bounded_offspring_pools(&archive, &occupied, 123, 6, 8_192).unwrap();
    assert_eq!(first.mutation, second.mutation);
    assert_eq!(first.crossover, second.crossover);
    assert_eq!(first.random, second.random);
    assert_eq!(first.mutation.len(), 60);
    assert_eq!(first.crossover.len(), 60);
    assert_eq!(first.random.len(), 32);
    assert!(build_bounded_offspring_pools(&archive, &occupied, 123, 6, 1).is_err());
}

#[test]
fn track_2_identity_never_enters_native_ranking_features() {
    let ids = identities(40, 20_000);
    let state = state_with_roles(&ids);
    let features = NativeFitnessFeaturesV1::build(&state, hash(99), 0.2, 5, 0.1, 10.0).unwrap();
    let serialized = String::from_utf8(canonical_json(&features).unwrap()).unwrap();
    assert!(!serialized.contains("track_2"));
    assert!(!serialized.contains(&ids[3].semantic_sha256));
    assert!((features.family_shrunk_residual - 0.1 / 3.0).abs() < 1e-12);
}

#[test]
fn wave_parentage_thresholds_gates_and_holdout_firewall_are_closed() {
    let parent = bootstrap_wave();
    parent.validate(None, &BTreeSet::new()).unwrap();
    let parent_sha = parent.manifest_sha256().unwrap();
    let wave = recurring_wave(parent_sha.clone(), &parent);
    wave.validate(Some((&parent_sha, &parent)), &BTreeSet::from([hash(999)]))
        .unwrap();

    let mut early = wave.clone();
    early.seconds_since_parent -= 1;
    early.publication_sha256 = early.publication_digest().unwrap();
    assert!(early
        .validate(Some((&parent_sha, &parent)), &BTreeSet::new())
        .is_err());

    let mut leaked = wave.clone();
    let sealed = leaked.gate_receipts[0].dataset_sha256s[0].clone();
    leaked.publication_sha256 = leaked.publication_digest().unwrap();
    assert!(leaked
        .validate(Some((&parent_sha, &parent)), &BTreeSet::from([sealed]))
        .is_err());
}

#[test]
fn append_only_dataset_state_is_content_addressed_and_rejects_rewrites() {
    let ids = identities(40, 30_000);
    let initial = state_with_roles(&ids);
    initial.validate().unwrap();
    let temporary = TempDir::new().unwrap();
    let store = RatchetStore::new(temporary.path());
    store.commit_dataset_state(&initial, 10).unwrap();

    let key = ObservationKeyV2 {
        dataset_sha256: initial.dataset_sha256.clone(),
        semantic_sha256: ids[10].semantic_sha256.clone(),
        evaluator_sha256: hash(101),
        folds_sha256: hash(102),
        protocol_sha256: hash(103),
    };
    let mut next = initial.clone();
    next.observations.push(PhysicalObservationV2 {
        sequence: 1,
        observation_sha256: hash(104),
        key: key.clone(),
        receipt_sha256: hash(105),
        observed_at_unix_seconds: 11,
        provenance_complete: true,
        synthetic: false,
    });
    next.label_revisions.push(LabelRevisionV2 {
        sequence: 1,
        key,
        revision: 1,
        observation_sha256: hash(104),
        revision_receipt_sha256: hash(106),
        committed_at_unix_seconds: 12,
        value: LabelValueV2 {
            status: LabelStatusV2::Exact,
            robust_fitness: Some(0.4),
            wall_time_seconds: Some(1.0),
            comparison_won: None,
        },
    });
    next.state_sequence = 2;
    next.counters.provenance_complete_labels = 1;
    next.counters.exact_labels = 1;
    store.commit_dataset_state(&next, 12).unwrap();
    assert_eq!(
        store
            .load_dataset_state(&initial.dataset_sha256)
            .unwrap()
            .unwrap()
            .1,
        next
    );

    let mut rewritten = next.clone();
    rewritten.observations[0].receipt_sha256 = hash(107);
    assert!(store.commit_dataset_state(&rewritten, 13).is_err());
}

fn replay_manifest() -> ReplayManifestV2 {
    replay_manifest_at(0, 25_000)
}

fn replay_manifest_at(offset: u64, label_cutoff: u64) -> ReplayManifestV2 {
    let reservoir_groups = (0..200)
        .map(|index| hash(offset + 20_000 + index))
        .collect::<Vec<_>>();
    let steps = (0..REPLAY_STEPS)
        .map(|step| {
            let datasets = (0..DATASETS_PER_MICRO_BATCH)
                .map(|index| hash(offset + 10_000 + (step * 8 + index) as u64))
                .collect::<Vec<_>>();
            let revisions = datasets
                .iter()
                .enumerate()
                .map(|(index, dataset)| ReplayLabelRevisionV2 {
                    group_sha256: reservoir_groups[(step * 8 + index) % 200].clone(),
                    dataset_sha256: dataset.clone(),
                    revision: 1,
                    label_revision_sha256: hash(offset + 30_000 + (step * 8 + index) as u64),
                })
                .collect();
            ReplayStepV2 {
                step,
                datasets,
                revisions,
            }
        })
        .collect::<Vec<_>>();
    ReplayManifestV2 {
        schema: REPLAY_V2_SCHEMA.into(),
        label_cutoff,
        reservoir_manifest_sha256: sha256_bytes(&canonical_json(&reservoir_groups).unwrap()),
        reservoir_group_count: 200,
        reservoir_groups,
        scheduled_datasets: steps[0].datasets.clone(),
        steps,
    }
}

fn publish_head_for_wave(
    store: &RatchetStore,
    wave: &LearningWaveManifestV1,
    base_checkpoint_sha256: String,
    replay_offset: u64,
    published_at_unix_seconds: u64,
) {
    let replay = replay_manifest_at(replay_offset, wave.training_label_cutoff);
    let batch = LearningMicroBatchManifestV1 {
        schema: MICRO_BATCH_SCHEMA.into(),
        batch_index: replay_offset,
        serving_wave_sha256: wave.parent_wave_sha256.clone().unwrap_or_else(|| hash(900)),
        base_training_checkpoint_sha256: base_checkpoint_sha256,
        candidate_training_checkpoint_sha256: wave.policy_checkpoint_sha256.clone(),
        candidate_resume_checkpoint_sha256: wave.resume_checkpoint_sha256.clone(),
        replay_manifest_sha256: replay.manifest_sha256().unwrap(),
        scheduler_draws: replay
            .scheduled_datasets
            .iter()
            .enumerate()
            .map(|(index, dataset)| SchedulerDrawV4 {
                draw_index: replay_offset + index as u64,
                lane: SchedulerLaneV4::Challenge,
                dataset_sha256: dataset.clone(),
                state_snapshot_sha256: hash(replay_offset + 60_000 + index as u64),
                draw_seed: replay_offset + index as u64,
            })
            .collect(),
        update_steps: REPLAY_STEPS,
        completed: true,
    };
    let head = TrainingHeadManifestV1 {
        schema: TRAINING_HEAD_SCHEMA.into(),
        checkpoint_sha256: wave.policy_checkpoint_sha256.clone(),
        resume_checkpoint_sha256: wave.resume_checkpoint_sha256.clone(),
        corrector_sha256: wave.corrector_sha256.clone(),
        micro_batch_manifest_sha256: batch.manifest_sha256(&replay).unwrap(),
        replay_manifest_sha256: replay.manifest_sha256().unwrap(),
        update_steps: REPLAY_STEPS,
        label_cutoff: wave.training_label_cutoff,
        published_at_unix_seconds,
    };
    store.publish_training_head(&head, &batch, &replay).unwrap();
}

#[test]
fn replay_is_exactly_one_hundred_novel_eight_dataset_steps() {
    let replay = replay_manifest();
    replay.validate(&BTreeSet::new()).unwrap();
    let mut historical = BTreeSet::new();
    let mut first = replay.steps[0].datasets.clone();
    first.sort();
    historical.insert(first);
    assert!(replay.validate(&historical).is_err());
}

#[test]
fn scheduler_repays_measured_cpu_debt_at_70_15_10_5() {
    let mut scheduler = SchedulerStateV4::default();
    for _ in 0..200 {
        let lane = scheduler.next_lane().unwrap();
        scheduler.record_measured_cpu(lane, 100).unwrap();
    }
    assert_eq!(
        scheduler.measured_cpu_micros[&SchedulerLaneV4::Challenge],
        14_000
    );
    assert_eq!(
        scheduler.measured_cpu_micros[&SchedulerLaneV4::Uniform],
        3_000
    );
    assert_eq!(
        scheduler.measured_cpu_micros[&SchedulerLaneV4::StaleUnderexposed],
        2_000
    );
    assert_eq!(
        scheduler.measured_cpu_micros[&SchedulerLaneV4::GlobalCvConfirmation],
        1_000
    );
}

#[test]
fn serving_and_training_pointers_are_separate_and_recovery_is_idempotent() {
    let temporary = TempDir::new().unwrap();
    let store = RatchetStore::new(temporary.path());
    let replay = replay_manifest();
    let replay_sha = replay.manifest_sha256().unwrap();
    let batch = LearningMicroBatchManifestV1 {
        schema: MICRO_BATCH_SCHEMA.into(),
        batch_index: 1,
        serving_wave_sha256: hash(298),
        base_training_checkpoint_sha256: hash(299),
        candidate_training_checkpoint_sha256: hash(300),
        candidate_resume_checkpoint_sha256: hash(301),
        replay_manifest_sha256: replay_sha.clone(),
        scheduler_draws: replay
            .scheduled_datasets
            .iter()
            .enumerate()
            .map(|(index, dataset)| SchedulerDrawV4 {
                draw_index: index as u64,
                lane: SchedulerLaneV4::Challenge,
                dataset_sha256: dataset.clone(),
                state_snapshot_sha256: hash(50_000 + index as u64),
                draw_seed: index as u64,
            })
            .collect(),
        update_steps: REPLAY_STEPS,
        completed: true,
    };
    let head = TrainingHeadManifestV1 {
        schema: TRAINING_HEAD_SCHEMA.into(),
        checkpoint_sha256: hash(300),
        resume_checkpoint_sha256: hash(301),
        corrector_sha256: hash(304),
        micro_batch_manifest_sha256: batch.manifest_sha256(&replay).unwrap(),
        replay_manifest_sha256: replay_sha,
        update_steps: REPLAY_STEPS,
        label_cutoff: replay.label_cutoff,
        published_at_unix_seconds: 1,
    };
    let head_sha = store.publish_training_head(&head, &batch, &replay).unwrap();
    assert!(!temporary.path().join("waves/current.json").exists());
    assert_eq!(
        serde_json::from_slice::<RatchetArtifactPointerV1>(
            &fs::read(temporary.path().join("training-head.json")).unwrap()
        )
        .unwrap()
        .artifact_sha256,
        head_sha
    );

    let switches = RolloutSwitchesV1 {
        dataset_state_v2: true,
        scheduler_v4: true,
        replay_v2: true,
        generation_zero_v3: true,
        wave_authority: true,
    };
    let parent = bootstrap_wave();
    let parent_sha = store
        .publish_serving_wave(&parent, &switches, &BTreeSet::new())
        .unwrap();
    // Retrying identical publication is byte-idempotent.
    assert_eq!(
        store
            .publish_serving_wave(&parent, &switches, &BTreeSet::new())
            .unwrap(),
        parent_sha
    );
    let next = recurring_wave(parent_sha.clone(), &parent);
    assert!(store
        .publish_serving_wave(&next, &switches, &BTreeSet::new())
        .is_err());
    publish_head_for_wave(&store, &next, head.checkpoint_sha256.clone(), 100_000, 2);
    let next_sha = store
        .publish_serving_wave(&next, &switches, &BTreeSet::new())
        .unwrap();
    assert_eq!(store.load_serving_wave().unwrap().unwrap().0, next_sha);
    store.stop_admissions(&hash(400), 20_000).unwrap();
    store.restore_previous_wave(&parent_sha, 20_001).unwrap();
    assert_eq!(store.load_serving_wave().unwrap().unwrap().0, parent_sha);
    let mirror: RatchetArtifactPointerV1 =
        serde_json::from_slice(&fs::read(temporary.path().join("dl/current.json")).unwrap())
            .unwrap();
    assert!(!mirror.serving_authority);
}

#[test]
fn admission_freezes_latest_wave_and_state_while_in_flight_retries_stay_bound() {
    let temporary = TempDir::new().unwrap();
    let store = RatchetStore::new(temporary.path());
    let switches = RolloutSwitchesV1 {
        dataset_state_v2: true,
        scheduler_v4: true,
        replay_v2: true,
        generation_zero_v3: true,
        wave_authority: true,
    };
    let wave = bootstrap_wave();
    let wave_sha = store
        .publish_serving_wave(&wave, &switches, &BTreeSet::new())
        .unwrap();
    let ids = identities(256, 40_000);
    let state = state_with_roles(&ids);
    store.commit_dataset_state(&state, 11_000).unwrap();
    let generation = generation_for_state(&ids, &state, wave_sha.clone());
    let admission = AttemptAdmissionManifestV3 {
        schema: ADMISSION_V3_SCHEMA.into(),
        attempt_id: "fixture-attempt-1".into(),
        learning_wave_sha256: wave_sha.clone(),
        dataset_state_sha256: state.snapshot_sha256().unwrap(),
        scheduler_draw_sha256: hash(500),
        generation_zero_sha256: generation.receipt_sha256().unwrap(),
        checkpoint_sha256: wave.policy_checkpoint_sha256.clone(),
        corrector_sha256: wave.corrector_sha256.clone(),
        dataset_sha256: state.dataset_sha256.clone(),
        folds_sha256: hash(501),
        proof_slate_seed: 29,
        proof_slate: ids[..12]
            .iter()
            .enumerate()
            .map(|(index, candidate)| ProofSlateCandidateV3 {
                candidate: candidate.clone(),
                lane: if index < 10 {
                    ProofSlateLaneV3::Selectable
                } else {
                    ProofSlateLaneV3::CvOnlyCalibration
                },
            })
            .collect(),
    };
    let receipt = store.admit_attempt(&generation, &admission).unwrap();
    assert!(temporary
        .path()
        .join("attempts/fixture-attempt-1/admission.v3.json")
        .exists());

    let next = recurring_wave(wave_sha, &wave);
    publish_head_for_wave(&store, &next, hash(29), 200_000, 11_500);
    store
        .publish_serving_wave(&next, &switches, &BTreeSet::new())
        .unwrap();
    // Recovery of the immutable admitted attempt does not switch waves.
    assert_eq!(
        store.admit_attempt(&generation, &admission).unwrap(),
        receipt
    );
    let mut new_attempt = admission.clone();
    new_attempt.attempt_id = "fixture-attempt-2".into();
    assert!(store.admit_attempt(&generation, &new_attempt).is_err());

    store.stop_admissions(&hash(502), 12_000).unwrap();
    assert_eq!(
        store.admit_attempt(&generation, &admission).unwrap(),
        receipt
    );
}

#[test]
fn rollout_defaults_off_and_wave_authority_requires_all_dependencies() {
    let defaults: RolloutSwitchesV1 = serde_json::from_str("{}").unwrap();
    assert_eq!(defaults, RolloutSwitchesV1::default());
    defaults.validate().unwrap();
    let invalid = RolloutSwitchesV1 {
        wave_authority: true,
        ..RolloutSwitchesV1::default()
    };
    assert!(invalid.validate().is_err());
}
