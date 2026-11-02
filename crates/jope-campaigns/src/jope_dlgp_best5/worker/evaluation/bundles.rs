use super::*;

pub fn prediction_metrics(
    target_rows: &[TargetRowV1],
    prediction: &PredictionObjectV1,
) -> Result<(f64, f64), String> {
    prediction.validate()?;
    let targets = target_rows
        .iter()
        .map(|row| (row.stable_row_id.as_str(), row.target))
        .collect::<BTreeMap<_, _>>();
    let actual = prediction
        .validation_row_ids
        .iter()
        .map(|row| {
            targets
                .get(row.as_str())
                .copied()
                .ok_or_else(|| format!("prediction names unknown target row {row}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let predicted = prediction
        .prediction_f64_bits
        .iter()
        .map(|bits| f64::from_bits(*bits))
        .collect::<Vec<_>>();
    if actual.len() != predicted.len() || actual.is_empty() {
        return Err("prediction and target rows are misaligned".into());
    }
    let mse = actual
        .iter()
        .zip(&predicted)
        .map(|(actual, predicted)| (actual - predicted).powi(2))
        .sum::<f64>()
        / actual.len() as f64;
    let mean = actual.iter().sum::<f64>() / actual.len() as f64;
    let variance = actual
        .iter()
        .map(|actual| (actual - mean).powi(2))
        .sum::<f64>();
    let r2 = if variance == 0.0 {
        if mse == 0.0 {
            1.0
        } else {
            -1.0
        }
    } else {
        1.0 - mse * actual.len() as f64 / variance
    }
    .clamp(-1.0, 1.0);
    Ok((r2, mse.sqrt()))
}

fn load_receipt_object(
    store: &ArtifactStoreV1,
    receipt: &PredictionReceiptV1,
) -> Result<PredictionObjectV1, String> {
    let object: PredictionObjectV1 = store.load(&receipt.key.identity()?)?;
    receipt.validate_object(&object)?;
    Ok(object)
}

/// Replays staged targets and content-addressed prediction objects. This is
/// the public finalizer's trust boundary; worker-supplied scalar metrics are
/// never accepted on their own.
pub fn verify_search_bundle(
    campaign: &CampaignFreezeV1,
    proposal_batch: &ProposalBatchFreezeV1,
    bundle: &SearchBundleV1,
    artifact_root: &Path,
) -> Result<(), String> {
    campaign.validate()?;
    bundle.validate()?;
    proposal_batch.validate(campaign)?;
    if bundle.staging.campaign_freeze_sha256 != campaign.identity()?
        || bundle.staging.proposal_batch_sha256 != proposal_batch.identity(campaign)?
    {
        return Err("search bundle belongs to a foreign campaign or proposal batch".into());
    }
    let dataset = campaign
        .datasets
        .iter()
        .find(|dataset| dataset.dataset_sha256 == bundle.result.dataset_sha256)
        .ok_or_else(|| "search bundle dataset is outside the campaign".to_string())?;
    if bundle.staging.training.sha256 != dataset.train_sha256
        || bundle.staging.metadata.sha256 != dataset.metadata_sha256
        || bundle.staging.training.path
            != Path::new(&campaign.sotarena_root)
                .join(&dataset.train_relative_path)
                .to_string_lossy()
        || bundle.staging.metadata.path
            != Path::new(&campaign.sotarena_root)
                .join(&dataset.metadata_relative_path)
                .to_string_lossy()
    {
        return Err("search staging paths or hashes differ from the campaign".into());
    }
    bundle.staging.training.verify("staged training")?;
    bundle.staging.metadata.verify("staged metadata")?;
    let rebuilt = build_training_table(campaign, &dataset.dataset_id)?;
    if rebuilt.target_rows != bundle.target_rows {
        return Err(
            "search staged targets or stable row identities were recomputed differently".into(),
        );
    }
    let split = build_search_split_plan(
        &dataset.dataset_sha256,
        &dataset.train_sha256,
        bundle.result.master_seed,
        &rebuilt.target_rows,
    )?;
    let cv7 = build_cv7_plan(
        &dataset.dataset_sha256,
        &dataset.train_sha256,
        bundle.result.master_seed,
        &rebuilt.target_rows,
    )?;
    if split != bundle.split_plan || cv7 != bundle.cv7_plan {
        return Err("search split or CV7 identity was not recomputed from staged targets".into());
    }
    let store = ArtifactStoreV1::new(artifact_root);
    let receipts = bundle
        .prediction_receipts
        .iter()
        .map(|receipt| Ok((receipt.identity()?, receipt)))
        .collect::<Result<BTreeMap<_, _>, String>>()?;
    for receipt in &bundle.prediction_receipts {
        load_receipt_object(&store, receipt)?;
    }
    for population in &bundle.result.populations {
        let split_sha256 =
            canonical_sha256(&bundle.split_plan.splits[population.generation as usize])?;
        for semantic in &population.semantic_sha256s {
            if !bundle.prediction_receipts.iter().any(|receipt| {
                receipt.key.split_sha256 == split_sha256 && receipt.key.semantic_sha256 == *semantic
            }) {
                return Err("generation population lacks evaluated prediction coverage".into());
            }
        }
    }
    for anchors in &bundle.anchor_receipt_sets {
        let mut r2 = anchors
            .prediction_receipt_sha256s
            .iter()
            .map(|identity| {
                let receipt = receipts
                    .get(identity)
                    .ok_or_else(|| "anchor references a missing prediction receipt".to_string())?;
                let object = load_receipt_object(&store, receipt)?;
                Ok(prediction_metrics(&rebuilt.target_rows, &object)?.0)
            })
            .collect::<Result<Vec<_>, String>>()?;
        r2.sort_by(f64::total_cmp);
        let median = 0.5 * (r2[4] + r2[5]);
        if (median - anchors.median_r2_clipped).abs() > 1e-12 {
            return Err("anchor median was altered after evaluation".into());
        }
    }
    for member in &bundle.result.archive {
        for evidence in &member.split_evidence {
            let receipt = receipts
                .get(&evidence.prediction_receipt_sha256)
                .ok_or_else(|| "archive references a missing prediction receipt".to_string())?;
            if receipt.key.semantic_sha256 != evidence.predictor.semantic_sha256()? {
                return Err("archive receipt binds another predictor".into());
            }
            let object = load_receipt_object(&store, receipt)?;
            let (r2, rmse) = prediction_metrics(&rebuilt.target_rows, &object)?;
            let anchor = &bundle.anchor_receipt_sets[evidence.split_index as usize];
            if (r2 - evidence.r2_clipped).abs() > 1e-12
                || (rmse - evidence.rmse).abs() > 1e-12
                || (anchor.median_r2_clipped - evidence.anchor_median_r2_clipped).abs() > 1e-12
            {
                return Err("archive metrics differ from resolved prediction objects".into());
            }
        }
    }
    for finalist in &bundle.result.finalists {
        let mut all_actual = Vec::new();
        let mut all_predicted = Vec::new();
        for (fold, identity) in finalist.fold_prediction_receipt_sha256s.iter().enumerate() {
            let receipt = receipts
                .get(identity)
                .ok_or_else(|| "CV7 references a missing prediction receipt".to_string())?;
            if receipt.key.semantic_sha256 != finalist.semantic_sha256 {
                return Err("CV7 receipt binds another predictor".into());
            }
            let object = load_receipt_object(&store, receipt)?;
            let (r2, _) = prediction_metrics(&rebuilt.target_rows, &object)?;
            if (r2 - finalist.fold_r2[fold]).abs() > 1e-12 {
                return Err("CV7 fold metric differs from its prediction object".into());
            }
            let target_map = rebuilt
                .target_rows
                .iter()
                .map(|row| (row.stable_row_id.as_str(), row.target))
                .collect::<BTreeMap<_, _>>();
            for (row, bits) in object
                .validation_row_ids
                .iter()
                .zip(&object.prediction_f64_bits)
            {
                all_actual.push(target_map[row.as_str()]);
                all_predicted.push(f64::from_bits(*bits));
            }
        }
        let mean = all_actual.iter().sum::<f64>() / all_actual.len() as f64;
        let sse = all_actual
            .iter()
            .zip(&all_predicted)
            .map(|(actual, predicted)| (actual - predicted).powi(2))
            .sum::<f64>();
        let sst = all_actual
            .iter()
            .map(|actual| (actual - mean).powi(2))
            .sum::<f64>();
        let pooled = if sst == 0.0 { -1.0 } else { 1.0 - sse / sst }.clamp(-1.0, 1.0);
        let rmse = (sse / all_actual.len() as f64).sqrt();
        if (pooled - finalist.pooled_oof_r2).abs() > 1e-12
            || (rmse - finalist.rmse).abs() > 1e-12
            || finalist.complexity != finalist.predictor.complexity()
        {
            return Err("CV7 aggregate metrics differ from resolved prediction objects".into());
        }
    }
    Ok(())
}

pub fn verify_sealed_bundle(
    campaign: &CampaignFreezeV1,
    bundle: &SealedBundleV1,
    artifact_root: &Path,
) -> Result<(), String> {
    campaign.validate()?;
    bundle.validate()?;
    if bundle.claim.campaign_freeze_sha256 != campaign.identity()? {
        return Err("sealed bundle belongs to a foreign campaign".into());
    }
    let dataset = campaign
        .datasets
        .iter()
        .find(|dataset| dataset.dataset_sha256 == bundle.claim.dataset_sha256)
        .ok_or_else(|| "sealed bundle dataset is outside the campaign".to_string())?;
    let arms = bundle
        .selections
        .iter()
        .map(|selection| selection.arm)
        .collect::<BTreeSet<_>>();
    match dataset.stage {
        DatasetStageV1::Pilot if arms != Best5ArmV1::ALL.into_iter().collect::<BTreeSet<_>>() => {
            return Err("pilot sealed job must cover all six selected arms".into());
        }
        DatasetStageV1::Confirmation
            if arms.len() != 2
                || !arms.contains(&Best5ArmV1::ConfidenceGpControl)
                || arms.iter().filter(|arm| arm.is_experiment()).count() != 1 =>
        {
            return Err("confirmation sealed job must cover winner plus control".into());
        }
        _ => {}
    }
    if bundle.staging.training.sha256 != dataset.train_sha256
        || bundle.staging.test.sha256 != dataset.test_sha256
    {
        return Err("sealed staged hashes differ from the frozen dataset".into());
    }
    bundle.staging.training.verify("sealed training")?;
    bundle.staging.test.verify("sealed test")?;
    let store = ArtifactStoreV1::new(artifact_root);
    let claim_slot = sealed_claim_slot_identity(
        &bundle.claim.campaign_freeze_sha256,
        &bundle.claim.dataset_sha256,
    )?;
    if !store.contains(&claim_slot)? {
        return Err("sealed dataset claim was not published before test access".into());
    }
    let published_claim: SealedClaimV1 = store.load(&claim_slot)?;
    if published_claim != bundle.claim {
        return Err("published sealed claim differs from the worker bundle".into());
    }
    let sealed = build_claimed_sealed_table(campaign, &dataset.dataset_id, &bundle.claim)?;
    let receipts = bundle
        .prediction_receipts
        .iter()
        .map(|receipt| Ok((receipt.key.semantic_sha256.clone(), receipt)))
        .collect::<Result<BTreeMap<_, _>, String>>()?;
    for score in &bundle.scores {
        let receipt = receipts
            .get(&score.winner_semantic_sha256)
            .ok_or_else(|| "sealed score lacks its prediction receipt".to_string())?;
        let prediction = load_receipt_object(&store, receipt)?;
        let (r2, rmse) = prediction_metrics(&sealed.target_rows, &prediction)?;
        let validity = if prediction.valid { 1.0 } else { 0.0 };
        if (r2 - score.r2).abs() > 1e-12
            || (rmse - score.rmse).abs() > 1e-12
            || (validity - score.validity).abs() > 1e-12
            || score.predictor.semantic_sha256()? != receipt.key.semantic_sha256
        {
            return Err("sealed score metrics differ from resolved test predictions".into());
        }
    }
    Ok(())
}

pub fn sealed_claim_slot_identity(
    campaign_freeze_sha256: &str,
    dataset_sha256: &str,
) -> Result<String, String> {
    crate::search_v3::validate_sha256("sealed claim campaign", campaign_freeze_sha256)?;
    crate::search_v3::validate_sha256("sealed claim dataset", dataset_sha256)?;
    Ok(raw_sha256(
        format!("best5-sealed-dataset-claim-v1|{campaign_freeze_sha256}|{dataset_sha256}")
            .as_bytes(),
    ))
}
