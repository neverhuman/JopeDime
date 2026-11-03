use super::super::super::state::{
    LabelRevisionV2, LabelStatusV2, LabelValueV2, ObservationKeyV2, PhysicalObservationV2,
};
use super::super::super::wave::{ReadableAdmission, ReadableGeneration0, ReadableTelemetry};
use super::*;

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
