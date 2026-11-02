use super::*;

use std::path::Path;

pub fn run_sealed_one(
    campaign: &CampaignFreezeV1,
    mut selections: Vec<SelectionFreezeV1>,
    artifact_root: &Path,
    work_root: &Path,
) -> Result<SealedBundleV1, String> {
    campaign.validate()?;
    if selections.is_empty() {
        return Err("sealed-one requires a dataset-level selection set".into());
    }
    for selection in &selections {
        selection.validate()?;
    }
    selections.sort_by_key(|selection| selection.arm);
    let campaign_sha256 = campaign.identity()?;
    let dataset_sha256 = selections[0].dataset_sha256.clone();
    if selections.iter().any(|selection| {
        selection.campaign_freeze_sha256 != campaign_sha256
            || selection.dataset_sha256 != dataset_sha256
    }) {
        return Err("sealed-one selections are not one campaign dataset".into());
    }
    let dataset = campaign
        .datasets
        .iter()
        .find(|dataset| dataset.dataset_sha256 == dataset_sha256)
        .ok_or_else(|| "sealed-one dataset is outside the campaign".to_string())?;
    let mut selection_ids = selections
        .iter()
        .map(SelectionFreezeV1::identity)
        .collect::<Result<Vec<_>, _>>()?;
    selection_ids.sort();
    let claim = SealedClaimV1 {
        schema: SealedClaimV1::SCHEMA.into(),
        campaign_freeze_sha256: campaign_sha256,
        dataset_sha256: dataset_sha256.clone(),
        selection_freeze_sha256s: selection_ids,
        access_ordinal: 1,
    };
    let claim_sha256 = claim.identity()?;
    let claim_slot =
        sealed_claim_slot_identity(&claim.campaign_freeze_sha256, &claim.dataset_sha256)?;
    let store = ArtifactStoreV1::new(artifact_root);
    // The immutable claim is the atomic boundary and precedes the first test
    // path resolution in build_claimed_sealed_table.
    store.publish(&claim_slot, &claim)?;
    store.publish(&claim_sha256, &claim)?;
    let training = build_training_table(campaign, &dataset.dataset_id)?;
    let sealed = build_claimed_sealed_table(campaign, &dataset.dataset_id, &claim)?;
    let predictors = selections
        .iter()
        .map(|selection| selection.winner_predictor.clone())
        .collect::<Vec<_>>();
    let mut context = Best5EvaluationContext::new(campaign, artifact_root, work_root)?;
    let evaluated =
        context.evaluate_claimed_sealed(&training.table, &sealed.table, &claim, &predictors)?;
    let scores = selections
        .iter()
        .zip(&evaluated)
        .map(|(selection, value)| {
            let (r2, rmse) = prediction_metrics(&sealed.target_rows, &value.prediction)?;
            Ok(SealedScoreV1 {
                schema: SealedScoreV1::SCHEMA.into(),
                campaign_freeze_sha256: selection.campaign_freeze_sha256.clone(),
                selection_freeze_sha256: selection.identity()?,
                arm: selection.arm,
                dataset_sha256: selection.dataset_sha256.clone(),
                test_sha256: dataset.test_sha256.clone(),
                winner_semantic_sha256: selection.winner_semantic_sha256.clone(),
                predictor: selection.winner_predictor.clone(),
                test_access_ordinal: 1,
                r2,
                rmse,
                latency_seconds: value.runtime_seconds,
                validity: if value.prediction.valid { 1.0 } else { 0.0 },
                ancestry: AncestryV1 {
                    predictor: selection.winner_predictor.clone(),
                    founder_semantic_sha256: selection.winner_predictor.pipelines()[0]
                        .semantic_sha256
                        .clone(),
                    parent_semantic_sha256s: selection
                        .winner_predictor
                        .pipelines()
                        .into_iter()
                        .map(|pipeline| pipeline.semantic_sha256.clone())
                        .collect(),
                    model_seeded: true,
                },
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let bundle = SealedBundleV1 {
        schema: SealedBundleV1::SCHEMA.into(),
        claim,
        staging: SealedStagingAttestationV1 {
            schema: SealedStagingAttestationV1::SCHEMA.into(),
            claim_sha256,
            training: training.training_artifact,
            test: sealed.test_artifact,
        },
        selections,
        prediction_receipts: evaluated.into_iter().map(|value| value.receipt).collect(),
        scores,
    };
    bundle.validate()?;
    verify_sealed_bundle(campaign, &bundle, artifact_root)?;
    Ok(bundle)
}
