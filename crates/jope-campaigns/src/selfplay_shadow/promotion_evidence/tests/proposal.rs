use super::*;

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
