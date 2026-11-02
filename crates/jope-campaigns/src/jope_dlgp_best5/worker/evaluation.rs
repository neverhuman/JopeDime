use super::*;

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::process::Command;

use crate::jope_generalized_gp::{
    raw_sha256, ArtifactStoreV1, CorpusEvalAdapter, DatasetTableV1, EvaluatorFreezeV1,
    FrozenArtifactV1, GenomeOriginV1, GenomeV1, PopulationMemberV1, RowSplitV1, SealedTestTableV1,
};
use crate::search_v3::{canonical_sha256, PredictionObjectV1};
use serde::{Deserialize, Serialize};

pub struct EvaluatedPredictorV1 {
    pub predictor: FrozenPredictorV1,
    pub prediction: PredictionObjectV1,
    pub receipt: PredictionReceiptV1,
    pub runtime_seconds: f64,
}

pub struct Best5EvaluationContext {
    adapter: CorpusEvalAdapter,
    store: ArtifactStoreV1,
    evaluator_sha256: String,
    math_policy_sha256: String,
    backend_policy_sha256: String,
    next_logical_operation: u32,
    receipts: Vec<PredictionReceiptV1>,
}

impl Best5EvaluationContext {
    pub fn new(
        campaign: &CampaignFreezeV1,
        artifact_root: impl Into<std::path::PathBuf>,
        work_root: impl Into<std::path::PathBuf>,
    ) -> Result<Self, String> {
        campaign.validate()?;
        let evaluator = EvaluatorFreezeV1 {
            schema: EvaluatorFreezeV1::SCHEMA.into(),
            artifact: FrozenArtifactV1 {
                path: campaign.evaluator_artifact.path.clone(),
                sha256: campaign.evaluator_artifact.sha256.clone(),
                byte_length: campaign.evaluator_artifact.bytes,
            },
            // The executable bytes are the V1 evaluator protocol binding for
            // this unpublished campaign shape.
            evaluator_protocol_sha256: campaign.evaluator_sha256.clone(),
            math_policy_sha256: campaign.math_policy_sha256.clone(),
            backend_policy_sha256: campaign.backend_policy_sha256.clone(),
        };
        Ok(Self {
            adapter: CorpusEvalAdapter::new(
                evaluator,
                work_root,
                campaign.scq.candidate_timeout_seconds as u64,
            )?,
            store: ArtifactStoreV1::new(artifact_root),
            evaluator_sha256: campaign.evaluator_sha256.clone(),
            math_policy_sha256: campaign.math_policy_sha256.clone(),
            backend_policy_sha256: campaign.backend_policy_sha256.clone(),
            next_logical_operation: 1,
            receipts: Vec::new(),
        })
    }

    pub fn receipts(&self) -> &[PredictionReceiptV1] {
        &self.receipts
    }

    pub fn into_receipts(self) -> Vec<PredictionReceiptV1> {
        self.receipts
    }

    fn key(
        &self,
        dataset: &DatasetTableV1,
        split_sha256: &str,
        semantic_sha256: String,
    ) -> PredictionKeyV1 {
        PredictionKeyV1 {
            schema: PredictionKeyV1::SCHEMA.into(),
            dataset_sha256: dataset.dataset_sha256.clone(),
            training_sha256: dataset.training_sha256.clone(),
            split_sha256: split_sha256.into(),
            semantic_sha256,
            evaluator_sha256: self.evaluator_sha256.clone(),
            math_policy_sha256: self.math_policy_sha256.clone(),
            backend_policy_sha256: self.backend_policy_sha256.clone(),
        }
    }

    fn load(&self, key: &PredictionKeyV1) -> Result<Option<PredictionObjectV1>, String> {
        let identity = key.identity()?;
        if !self.store.contains(&identity)? {
            return Ok(None);
        }
        let prediction: PredictionObjectV1 = self.store.load(&identity)?;
        let receipt = PredictionReceiptV1 {
            schema: PredictionReceiptV1::SCHEMA.into(),
            key: key.clone(),
            prediction_object_sha256: canonical_sha256(&prediction)?,
            logical_operation_ordinal: 1,
        };
        receipt.validate_object(&prediction)?;
        Ok(Some(prediction))
    }

    fn publish(
        &self,
        key: &PredictionKeyV1,
        prediction: &PredictionObjectV1,
    ) -> Result<String, String> {
        let object_sha256 = canonical_sha256(prediction)?;
        let published = self.store.publish(&key.identity()?, prediction)?;
        if published != object_sha256 {
            return Err("prediction store returned a different content digest".into());
        }
        Ok(object_sha256)
    }

    fn charge(
        &mut self,
        key: PredictionKeyV1,
        prediction: PredictionObjectV1,
        predictor: FrozenPredictorV1,
        runtime_seconds: f64,
    ) -> Result<EvaluatedPredictorV1, String> {
        let receipt = PredictionReceiptV1 {
            schema: PredictionReceiptV1::SCHEMA.into(),
            key,
            prediction_object_sha256: canonical_sha256(&prediction)?,
            logical_operation_ordinal: self.next_logical_operation,
        };
        receipt.validate_object(&prediction)?;
        self.next_logical_operation = self
            .next_logical_operation
            .checked_add(1)
            .ok_or_else(|| "logical operation ordinal overflowed".to_string())?;
        self.receipts.push(receipt.clone());
        Ok(EvaluatedPredictorV1 {
            predictor,
            prediction,
            receipt,
            runtime_seconds,
        })
    }

    pub fn evaluate_search_split(
        &mut self,
        dataset: &DatasetTableV1,
        split: &DataSplitV1,
        predictors: &[FrozenPredictorV1],
    ) -> Result<Vec<EvaluatedPredictorV1>, String> {
        dataset.validate()?;
        if predictors.is_empty() || predictors.len() > POPULATION {
            return Err("Best5 evaluation batch requires one through 80 predictors".into());
        }
        let split_sha256 = canonical_sha256(split)?;
        let mut final_objects = BTreeMap::<String, PredictionObjectV1>::new();
        let mut missing_components = BTreeMap::<String, GenomeV1>::new();
        for predictor in predictors {
            predictor.validate()?;
            let semantic = predictor.semantic_sha256()?;
            let key = self.key(dataset, &split_sha256, semantic.clone());
            if let Some(prediction) = self.load(&key)? {
                final_objects.insert(semantic, prediction);
            } else {
                for pipeline in predictor.pipelines() {
                    let component_key =
                        self.key(dataset, &split_sha256, pipeline.semantic_sha256.clone());
                    if let Some(prediction) = self.load(&component_key)? {
                        final_objects
                            .entry(pipeline.semantic_sha256.clone())
                            .or_insert(prediction);
                    } else {
                        missing_components
                            .entry(pipeline.semantic_sha256.clone())
                            .or_insert_with(|| pipeline.clone());
                    }
                }
            }
        }
        if missing_components.len() > 128 {
            return Err(
                "component evaluation batch exceeds the immutable evaluator envelope".into(),
            );
        }
        if !missing_components.is_empty() {
            let candidates = missing_components.values().cloned().collect::<Vec<_>>();
            let legacy_split = RowSplitV1 {
                split_index: split.split_index,
                seed: split.split_seed,
                train_row_ids: split.train_row_ids.clone(),
                validation_row_ids: split.validation_row_ids.clone(),
                validation_groups: split.validation_groups.clone(),
                balance_counts: BTreeMap::new(),
            };
            let outcomes =
                self.adapter
                    .evaluate_search_split(dataset, &legacy_split, &candidates)?;
            if outcomes.len() != candidates.len() {
                return Err("evaluator omitted a component prediction".into());
            }
            for (candidate, outcome) in candidates.into_iter().zip(outcomes) {
                outcome.validate()?;
                let key = self.key(dataset, &split_sha256, candidate.semantic_sha256.clone());
                let mut prediction = outcome.prediction;
                prediction.fold_plan_sha256 = split_sha256.clone();
                prediction.fold_index = split.split_index;
                prediction.fit_sha256 = raw_sha256(
                    format!(
                        "best5-fit-v1|{}|{}|{}|{}",
                        dataset.training_sha256,
                        split_sha256,
                        candidate.semantic_sha256,
                        self.evaluator_sha256
                    )
                    .as_bytes(),
                );
                let object_sha256 = self.publish(&key, &prediction)?;
                if object_sha256 != canonical_sha256(&prediction)? {
                    return Err("component prediction publication changed bytes".into());
                }
                final_objects.insert(candidate.semantic_sha256, prediction);
            }
        }
        let mut evaluated = Vec::with_capacity(predictors.len());
        for predictor in predictors {
            let semantic = predictor.semantic_sha256()?;
            let key = self.key(dataset, &split_sha256, semantic.clone());
            let prediction = if let Some(prediction) = final_objects.get(&semantic) {
                prediction.clone()
            } else if let Some(prediction) = self.load(&key)? {
                prediction
            } else {
                let component_objects = predictor
                    .pipelines()
                    .into_iter()
                    .map(|pipeline| {
                        final_objects
                            .get(&pipeline.semantic_sha256)
                            .cloned()
                            .ok_or_else(|| "missing predictor component object".to_string())
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let component_values = component_objects
                    .iter()
                    .map(|object| {
                        object
                            .prediction_f64_bits
                            .iter()
                            .map(|bits| f64::from_bits(*bits))
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                let values = predictor.reconstruct(&component_values)?;
                let valid = component_objects.iter().all(|object| object.valid);
                let invalid_reason = (!valid).then(|| {
                    component_objects
                        .iter()
                        .filter_map(|object| object.invalid_reason.as_deref())
                        .collect::<Vec<_>>()
                        .join("; ")
                });
                let prediction = PredictionObjectV1 {
                    schema: PredictionObjectV1::SCHEMA.into(),
                    training_view_sha256: dataset.training_sha256.clone(),
                    fold_plan_sha256: split_sha256.clone(),
                    fold_index: split.split_index,
                    candidate_semantic_sha256: semantic.clone(),
                    evaluator_sha256: self.evaluator_sha256.clone(),
                    fit_sha256: raw_sha256(
                        format!("best5-composite-fit-v1|{}|{split_sha256}", semantic).as_bytes(),
                    ),
                    math_policy_sha256: self.math_policy_sha256.clone(),
                    backend_policy_sha256: self.backend_policy_sha256.clone(),
                    validation_row_ids: split.validation_row_ids.clone(),
                    prediction_shape: vec![values.len()],
                    prediction_f64_bits: values.into_iter().map(f64::to_bits).collect(),
                    valid,
                    invalid_reason,
                };
                self.publish(&key, &prediction)?;
                prediction
            };
            evaluated.push(self.charge(key, prediction, predictor.clone(), 0.0)?);
        }
        Ok(evaluated)
    }

    pub fn evaluate_claimed_sealed(
        &mut self,
        training: &DatasetTableV1,
        sealed: &SealedTestTableV1,
        claim: &SealedClaimV1,
        predictors: &[FrozenPredictorV1],
    ) -> Result<Vec<EvaluatedPredictorV1>, String> {
        claim.validate()?;
        if claim.dataset_sha256 != training.dataset_sha256
            || sealed.dataset_sha256 != training.dataset_sha256
            || predictors.len() != claim.selection_freeze_sha256s.len()
        {
            return Err("sealed evaluator inputs differ from the claimed dataset set".into());
        }
        let mut components = BTreeMap::<String, GenomeV1>::new();
        for predictor in predictors {
            predictor.validate()?;
            for pipeline in predictor.pipelines() {
                components
                    .entry(pipeline.semantic_sha256.clone())
                    .or_insert_with(|| pipeline.clone());
            }
        }
        let candidates = components.values().cloned().collect::<Vec<_>>();
        let claim_sha256 = claim.identity()?;
        let outcomes = self.adapter.evaluate_claimed_sealed_batch(
            training,
            sealed,
            &claim_sha256,
            &candidates,
        )?;
        let split_sha256 = raw_sha256(
            format!("claimed-sealed-v1|{claim_sha256}|{}", sealed.test_sha256).as_bytes(),
        );
        let component_objects = candidates
            .into_iter()
            .zip(outcomes)
            .map(|(candidate, outcome)| {
                outcome.validate()?;
                let key = self.key(training, &split_sha256, candidate.semantic_sha256.clone());
                self.publish(&key, &outcome.prediction)?;
                Ok((candidate.semantic_sha256, outcome.prediction))
            })
            .collect::<Result<BTreeMap<_, _>, String>>()?;
        let mut evaluated = Vec::with_capacity(predictors.len());
        for predictor in predictors {
            let semantic = predictor.semantic_sha256()?;
            let key = self.key(training, &split_sha256, semantic.clone());
            let components = predictor
                .pipelines()
                .into_iter()
                .map(|pipeline| {
                    component_objects
                        .get(&pipeline.semantic_sha256)
                        .cloned()
                        .ok_or_else(|| "sealed evaluator omitted a predictor component".to_string())
                })
                .collect::<Result<Vec<_>, _>>()?;
            let prediction = if components.len() == 1 {
                components[0].clone()
            } else {
                let values = predictor.reconstruct(
                    &components
                        .iter()
                        .map(|object| {
                            object
                                .prediction_f64_bits
                                .iter()
                                .map(|bits| f64::from_bits(*bits))
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>(),
                )?;
                PredictionObjectV1 {
                    schema: PredictionObjectV1::SCHEMA.into(),
                    training_view_sha256: training.training_sha256.clone(),
                    fold_plan_sha256: split_sha256.clone(),
                    fold_index: 0,
                    candidate_semantic_sha256: semantic.clone(),
                    evaluator_sha256: self.evaluator_sha256.clone(),
                    fit_sha256: raw_sha256(
                        format!("best5-sealed-composite-v1|{semantic}|{claim_sha256}").as_bytes(),
                    ),
                    math_policy_sha256: self.math_policy_sha256.clone(),
                    backend_policy_sha256: self.backend_policy_sha256.clone(),
                    validation_row_ids: sealed.test_row_ids.clone(),
                    prediction_shape: vec![values.len()],
                    prediction_f64_bits: values.into_iter().map(f64::to_bits).collect(),
                    valid: components.iter().all(|object| object.valid),
                    invalid_reason: components
                        .iter()
                        .find_map(|object| object.invalid_reason.clone()),
                }
            };
            self.publish(&key, &prediction)?;
            evaluated.push(self.charge(key, prediction, predictor.clone(), 0.0)?);
        }
        Ok(evaluated)
    }

    fn replace_with_derived_predictor(
        &mut self,
        dataset: &DatasetTableV1,
        predictor: FrozenPredictorV1,
        components: &[EvaluatedPredictorV1],
        replaced_receipt: &PredictionReceiptV1,
    ) -> Result<EvaluatedPredictorV1, String> {
        predictor.validate()?;
        let by_semantic = components
            .iter()
            .map(|value| Ok((value.predictor.semantic_sha256()?, value)))
            .collect::<Result<BTreeMap<_, _>, String>>()?;
        let selected = predictor
            .pipelines()
            .into_iter()
            .map(|pipeline| {
                by_semantic
                    .get(&pipeline.semantic_sha256)
                    .copied()
                    .ok_or_else(|| "derived predictor lacks a selected component".to_string())
            })
            .collect::<Result<Vec<_>, _>>()?;
        let values = predictor.reconstruct(
            &selected
                .iter()
                .map(|value| {
                    value
                        .prediction
                        .prediction_f64_bits
                        .iter()
                        .map(|bits| f64::from_bits(*bits))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>(),
        )?;
        let semantic = predictor.semantic_sha256()?;
        let key = self.key(
            dataset,
            &replaced_receipt.key.split_sha256,
            semantic.clone(),
        );
        let prediction = PredictionObjectV1 {
            schema: PredictionObjectV1::SCHEMA.into(),
            training_view_sha256: dataset.training_sha256.clone(),
            fold_plan_sha256: replaced_receipt.key.split_sha256.clone(),
            fold_index: selected[0].prediction.fold_index,
            candidate_semantic_sha256: semantic.clone(),
            evaluator_sha256: self.evaluator_sha256.clone(),
            fit_sha256: raw_sha256(
                format!(
                    "best5-derived-predictor-v1|{semantic}|{}",
                    replaced_receipt.key.split_sha256
                )
                .as_bytes(),
            ),
            math_policy_sha256: self.math_policy_sha256.clone(),
            backend_policy_sha256: self.backend_policy_sha256.clone(),
            validation_row_ids: selected[0].prediction.validation_row_ids.clone(),
            prediction_shape: vec![values.len()],
            prediction_f64_bits: values.into_iter().map(f64::to_bits).collect(),
            valid: selected.iter().all(|value| value.prediction.valid),
            invalid_reason: selected
                .iter()
                .find_map(|value| value.prediction.invalid_reason.clone()),
        };
        self.publish(&key, &prediction)?;
        let position = self
            .receipts
            .iter()
            .position(|receipt| receipt == replaced_receipt)
            .ok_or_else(|| "derived predictor replacement receipt is absent".to_string())?;
        let receipt = PredictionReceiptV1 {
            schema: PredictionReceiptV1::SCHEMA.into(),
            key,
            prediction_object_sha256: canonical_sha256(&prediction)?,
            logical_operation_ordinal: replaced_receipt.logical_operation_ordinal,
        };
        receipt.validate_object(&prediction)?;
        self.receipts[position] = receipt.clone();
        Ok(EvaluatedPredictorV1 {
            predictor,
            prediction,
            receipt,
            runtime_seconds: selected.iter().map(|value| value.runtime_seconds).sum(),
        })
    }
}

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
