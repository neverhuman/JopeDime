use super::super::generation::{GenerationRoleBindingV3, GenerationRoleV3, GprKernelFamilyV3};
use super::super::state::{
    GlobalCvFrontierMemberV2, LearningCountersV2, PredictionSnapshotV2, SchedulingMetadataV2,
    SelectionAuthorityV2, StruggleStateV2, TrackHistoryEntryV2,
};
use super::super::wave::{
    ArtifactReleaseV1, ModelProposalV2, RetainedModelProposalV2, WaveGateKindV1, WaveGateReceiptV1,
};
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

mod model_tests;
mod training_tests;
