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

#[derive(Clone)]
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

    pub(super) fn replace_with_derived_predictor(
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

mod bundles;
pub use bundles::{
    prediction_metrics, sealed_claim_slot_identity, verify_sealed_bundle, verify_search_bundle,
};
