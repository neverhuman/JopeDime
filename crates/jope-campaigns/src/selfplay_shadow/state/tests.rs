use super::*;

fn digest(character: char) -> Sha256Digest {
    Sha256Digest::parse(character.to_string().repeat(64)).unwrap()
}

fn front(task: Task, checkpoint: char, generation: u64) -> AcceptedFront {
    AcceptedFront {
        task,
        generation,
        checkpoint_sha256: digest(checkpoint),
        parent_checkpoint_sha256: (generation > 0).then(|| digest('0')),
        promotion_evidence_sha256: digest('e'),
        promoted_at_ms: 10,
    }
}

fn key() -> EvaluationKeyV2 {
    EvaluationKeyV2::new(
        Task::Binary,
        digest('a'),
        digest('b'),
        digest('c'),
        digest('d'),
    )
}

#[test]
fn supply_state_resumes_after_every_persisted_stage() {
    let mut state = CohortSupplyStateV1::new(
        Task::Binary,
        digest('a'),
        digest('b'),
        digest('c'),
        digest('d'),
        11,
    );
    let transitions = [
        (
            CohortSupplyStage::SubmitCohortProposer,
            Some("proposal-job".to_owned()),
            None,
            None,
        ),
        (
            CohortSupplyStage::FetchValidateProposals,
            Some("proposal-job".to_owned()),
            Some(digest('e')),
            None,
        ),
        (
            CohortSupplyStage::CacheFilterResample,
            None,
            Some(digest('e')),
            None,
        ),
        (
            CohortSupplyStage::BuildWave,
            None,
            None,
            Some("wave-1".to_owned()),
        ),
        (
            CohortSupplyStage::KeeperAcknowledgment,
            None,
            None,
            Some("wave-1".to_owned()),
        ),
    ];
    for (stage, job, pointer, wave) in transitions {
        assert!(state.advance(stage, job, pointer, wave).unwrap());
        state = serde_json::from_slice(&serde_json::to_vec(&state).unwrap()).unwrap();
        state.validate().unwrap();
    }
    assert!(!state
        .advance(
            CohortSupplyStage::KeeperAcknowledgment,
            Some("proposal-job".into()),
            Some(digest('e')),
            Some("wave-1".into()),
        )
        .unwrap());
    assert!(state
        .advance(
            CohortSupplyStage::KeeperAcknowledgment,
            Some("different-job".into()),
            None,
            None,
        )
        .is_err());
}

fn encoding_receipt(
    dataset_id: &str,
    content: char,
    attempt_number: u32,
    succeeded: bool,
) -> EncodingAdoptionReceiptV2 {
    EncodingAdoptionReceiptV2 {
        schema: "EncodingAdoptionReceipt.v2".into(),
        dataset: EncodingDatasetIdentity {
            dataset_id: dataset_id.into(),
            content_sha256: digest(content),
        },
        attempt_number,
        job_identity: format!("encode-{dataset_id}-{attempt_number}"),
        encoder_identity_sha256: digest('9'),
        disposition: if succeeded {
            EncodingAdoptionDisposition::Succeeded {
                encoded_train_sha256: digest('7'),
                encoded_test_sha256: digest('8'),
            }
        } else {
            EncodingAdoptionDisposition::Failed {
                error_class: "dataset-malformed".into(),
            }
        },
        recorded_at_ms: attempt_number as u64,
    }
}

#[test]
fn encoding_adoption_is_monotonic_per_dataset_across_partial_chunks() {
    let mut ledger = EncodingAdoptionLedgerV2::default();
    assert!(ledger.adopt(encoding_receipt("a", 'a', 1, true)).unwrap());
    assert!(ledger.adopt(encoding_receipt("b", 'b', 1, false)).unwrap());
    let mut ledger: EncodingAdoptionLedgerV2 =
        serde_json::from_slice(&serde_json::to_vec(&ledger).unwrap()).unwrap();
    assert!(!ledger.adopt(encoding_receipt("a", 'a', 2, false)).unwrap());
    assert!(ledger.adopt(encoding_receipt("b", 'b', 2, true)).unwrap());
    assert_eq!(ledger.successful.len(), 2);
    assert!(ledger.latest_failures.is_empty());

    let mut changed = encoding_receipt("a", 'a', 3, true);
    changed.disposition = EncodingAdoptionDisposition::Succeeded {
        encoded_train_sha256: digest('5'),
        encoded_test_sha256: digest('6'),
    };
    assert!(ledger.adopt(changed).is_err());
}

#[test]
fn task_compare_and_swap_does_not_move_the_other_front() {
    let mut accepted =
        AcceptedByTaskV2::new(front(Task::Binary, '1', 0), front(Task::Regression, '2', 0))
            .unwrap();
    let regression = accepted.regression.clone();
    accepted
        .compare_and_swap(Task::Binary, &digest('1'), digest('3'), digest('4'), 20)
        .unwrap();
    assert_eq!(accepted.binary.generation, 1);
    assert_eq!(accepted.regression, regression);
}

#[test]
fn lineage_is_hash_chained_and_causally_ordered() {
    let mut lineage = CausalLineage::default();
    for (index, stage) in [
        LineageStage::AcceptedParent,
        LineageStage::PolicyActions,
        LineageStage::Evaluations,
        LineageStage::CorpusShards,
        LineageStage::Candidate,
        LineageStage::Tournament,
        LineageStage::AcceptedChild,
        LineageStage::NextActions,
    ]
    .into_iter()
    .enumerate()
    {
        lineage
            .append(
                Task::Binary,
                0,
                digest('a'),
                stage,
                Sha256Digest::of_bytes(format!("artifact-{index}").as_bytes()),
                index as u64 + 1,
            )
            .unwrap();
    }
    lineage.validate().unwrap();
    assert!(lineage.complete_generation(Task::Binary, 0));
    assert!(lineage
        .append(
            Task::Regression,
            0,
            digest('b'),
            LineageStage::Evaluations,
            digest('c'),
            10,
        )
        .is_err());
}

#[test]
fn repeated_queue_submission_never_duplicates_evaluator_execution() {
    let mut queue = QueueReservationBook::default();
    assert_eq!(queue.reserve(key(), "wave-1").unwrap(), 1);
    assert_eq!(queue.reserve(key(), "wave-1").unwrap(), 1);
    queue.submit(&key(), "wave-1", 1, "eval-run-1").unwrap();
    queue.submit(&key(), "wave-1", 1, "eval-run-1").unwrap();
    assert!(queue.submit(&key(), "wave-1", 1, "eval-run-2").is_err());
    assert_eq!(queue.active_attempts(), 1);
    queue.finalize(&key(), "wave-1", 1, digest('e')).unwrap();
    queue.finalize(&key(), "wave-1", 1, digest('e')).unwrap();
    assert_eq!(queue.retry(&key(), "wave-1-retry").unwrap(), 2);
    assert_eq!(queue.active_attempts(), 1);
}

#[test]
fn admission_false_is_a_healthy_supply_hold() {
    let status = QueueStatusEnvelope {
        schema: "JopeSelfplayQueueKeeper.v3".into(),
        generated_at_ms: 1,
        admission_allowed: false,
        degraded: false,
        error: None,
        fail_closed_reason: Some("reservation pressure".into()),
        queued: 0,
        running: 0,
    };
    assert_eq!(
        interpret_queue_status(&status).unwrap(),
        SupplyAdmission::HealthyHold("reservation pressure".into())
    );
}

#[test]
fn bundle_drift_fails_closed() {
    let bundle = RuntimeBundleV3Identity {
        schema: RUNTIME_BUNDLE_SCHEMA.into(),
        bundle_sha256: digest('a'),
        configuration_sha256: digest('b'),
        evaluator_sha256: digest('c'),
        services: vec![ServiceIdentity {
            service: "queue".into(),
            bundle_sha256: digest('a'),
            configuration_sha256: digest('b'),
            evaluator_sha256: digest('d'),
            started_at_ms: 1,
        }],
    };
    assert!(bundle.validate().is_err());
}

#[test]
fn shared_training_slot_chooses_the_oldest_eligible_task_front() {
    let accepted =
        AcceptedByTaskV2::new(front(Task::Binary, '1', 0), front(Task::Regression, '2', 1))
            .unwrap();
    let evidence = BTreeMap::from([
        (
            Task::Binary,
            ParentEvidence {
                completed_parent_waves: 1,
                authoritative_dl_direct_outcomes: 2_048,
            },
        ),
        (
            Task::Regression,
            ParentEvidence {
                completed_parent_waves: 2,
                authoritative_dl_direct_outcomes: 4_096,
            },
        ),
    ]);
    assert_eq!(
        choose_shared_training_task(&accepted, &evidence),
        Some(Task::Binary)
    );
}
