use super::super::admission::{
    ProofSlateCandidateV3, ProofSlateLaneV3, ReplayLabelRevisionV2, ReplayStepV2,
};
use super::super::store::RatchetArtifactPointerV1;
use super::*;

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
