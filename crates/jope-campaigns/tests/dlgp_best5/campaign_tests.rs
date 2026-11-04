use super::*;

#[test]
fn campaign_freezes_exact_20_plus_22_and_retains_nine_small_datasets() {
    let campaign = campaign();
    campaign.validate().unwrap();
    assert_eq!(campaign.datasets.len(), 42);
    assert_eq!(
        campaign
            .datasets
            .iter()
            .filter(|dataset| dataset.stage == DatasetStageV1::Pilot && dataset.train_rows < 300)
            .count(),
        9
    );
    let mut hash_drift = campaign.clone();
    hash_drift.datasets[0].train_sha256 = digest("substituted training bytes");
    assert!(hash_drift
        .validate()
        .unwrap_err()
        .contains("input-manifest identity or hash drift"));
    let mut filtered = campaign.clone();
    filtered
        .datasets
        .retain(|dataset| dataset.train_rows >= 300);
    assert!(filtered.validate().is_err());
    let mut substituted = campaign;
    substituted.checkpoint.artifact_sha256 = digest("newer checkpoint");
    assert!(substituted
        .validate()
        .unwrap_err()
        .contains("substitution is forbidden"));
}

#[test]
fn budget_ledger_charges_cache_hits_and_rejects_incomplete_coverage() {
    let control = BudgetLedgerV1 {
        logical_operations: 950,
        predicted_cpu_seconds: 7_200,
        mandatory_anchor_evaluations: 80,
        completed_anchor_evaluations: 80,
        mandatory_search_evaluations: 640,
        completed_search_evaluations: 640,
        mandatory_archive_evaluations: 160,
        completed_archive_evaluations: 160,
        mandatory_cv7_evaluations: 70,
        completed_cv7_evaluations: 70,
        arm_specific_operations: 0,
    };
    control.validate(Best5ArmV1::ConfidenceGpControl).unwrap();
    let mut lineage = control.clone();
    lineage.logical_operations = 1_090;
    lineage.arm_specific_operations = 140;
    lineage
        .validate(Best5ArmV1::EvolvabilityUcbLineages)
        .unwrap();
    let mut incomplete = control;
    incomplete.completed_cv7_evaluations = 69;
    incomplete.logical_operations = 949;
    assert!(incomplete
        .validate(Best5ArmV1::ConfidenceGpControl)
        .unwrap_err()
        .contains("incomplete_mandatory_coverage"));
}

#[test]
fn proposal_batch_normalizes_shuffled_completion_and_covers_all_42_once() {
    let campaign = campaign();
    let mut batch = ProposalBatchFreezeV1 {
        schema: ProposalBatchFreezeV1::SCHEMA.into(),
        campaign_freeze_sha256: campaign.identity().unwrap(),
        proposals: campaign
            .datasets
            .iter()
            .rev()
            .map(|dataset| proposal_for(&dataset.dataset_sha256))
            .collect(),
    };
    assert!(batch.validate(&campaign).is_err());
    batch.normalize_completion_order(&campaign);
    batch.validate(&campaign).unwrap();
    assert_eq!(batch.proposals.len(), 42);
    assert!(batch
        .proposals
        .iter()
        .all(|proposal| proposal.inference_ordinal == 1 && proposal.proposals.len() == 64));
}
