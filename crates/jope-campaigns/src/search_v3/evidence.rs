use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use super::contracts::{
    CandidateProducerV1, RegisteredFitnessV1, SearchCandidateIdentityV1, SearchCandidatePlanV1,
    SearchRunContractV3, SearchTaskV1,
};
use super::identity::{canonical_sha256, validate_schema, validate_sha256};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PredictionObjectV1 {
    pub schema: String,
    pub training_view_sha256: String,
    pub fold_plan_sha256: String,
    pub fold_index: u32,
    pub candidate_semantic_sha256: String,
    pub evaluator_sha256: String,
    pub fit_sha256: String,
    pub math_policy_sha256: String,
    pub backend_policy_sha256: String,
    pub validation_row_ids: Vec<String>,
    pub prediction_shape: Vec<usize>,
    pub prediction_f64_bits: Vec<u64>,
    pub valid: bool,
    pub invalid_reason: Option<String>,
}

impl PredictionObjectV1 {
    pub const SCHEMA: &'static str = "PredictionObject.v1";

    pub fn validate(&self) -> Result<(), String> {
        validate_schema(&self.schema, Self::SCHEMA)?;
        for (name, digest) in [
            ("training_view_sha256", self.training_view_sha256.as_str()),
            ("fold_plan_sha256", self.fold_plan_sha256.as_str()),
            (
                "candidate_semantic_sha256",
                self.candidate_semantic_sha256.as_str(),
            ),
            ("evaluator_sha256", self.evaluator_sha256.as_str()),
            ("fit_sha256", self.fit_sha256.as_str()),
            ("math_policy_sha256", self.math_policy_sha256.as_str()),
            ("backend_policy_sha256", self.backend_policy_sha256.as_str()),
        ] {
            validate_sha256(name, digest)?;
        }
        if self.validation_row_ids.is_empty()
            || self.validation_row_ids.iter().any(String::is_empty)
            || self
                .validation_row_ids
                .iter()
                .collect::<BTreeSet<_>>()
                .len()
                != self.validation_row_ids.len()
        {
            return Err("prediction validation row IDs must be non-empty and unique".into());
        }
        if self.prediction_shape.is_empty()
            || self.prediction_shape[0] != self.validation_row_ids.len()
            || self.prediction_shape.contains(&0)
        {
            return Err("prediction shape must be non-empty and row-aligned".into());
        }
        let expected = self
            .prediction_shape
            .iter()
            .try_fold(1usize, |product, dimension| product.checked_mul(*dimension))
            .ok_or_else(|| "prediction shape overflows usize".to_string())?;
        if expected != self.prediction_f64_bits.len() {
            return Err(format!(
                "prediction shape requires {expected} f64 values, found {}",
                self.prediction_f64_bits.len()
            ));
        }
        let all_finite = self
            .prediction_f64_bits
            .iter()
            .all(|bits| f64::from_bits(*bits).is_finite());
        match (self.valid, self.invalid_reason.as_deref(), all_finite) {
            (true, None, true) => Ok(()),
            (false, Some(reason), _) if !reason.trim().is_empty() => Ok(()),
            (true, _, false) => Err("a valid prediction object contains a nonfinite value".into()),
            (true, Some(_), _) => {
                Err("a valid prediction object cannot carry invalid_reason".into())
            }
            (false, None, _) => Err("an invalid prediction object requires invalid_reason".into()),
            (false, Some(_), _) => Err("invalid_reason must not be empty".into()),
        }
    }

    pub fn identity(&self) -> Result<String, String> {
        self.validate()?;
        canonical_sha256(self)
    }

    fn values(&self) -> Vec<f64> {
        self.prediction_f64_bits
            .iter()
            .map(|bits| f64::from_bits(*bits))
            .collect()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScoringTapeV2 {
    pub schema: String,
    pub prediction_object_sha256: String,
    pub validation_row_ids: Vec<String>,
    pub truth_shape: Vec<usize>,
    pub truth_f64_bits: Vec<u64>,
    pub auxiliary_scale_f64_bits: Option<u64>,
}

impl ScoringTapeV2 {
    pub const SCHEMA: &'static str = "ScoringTape.v2";

    pub fn validate(&self) -> Result<(), String> {
        validate_schema(&self.schema, Self::SCHEMA)?;
        validate_sha256("prediction_object_sha256", &self.prediction_object_sha256)?;
        if self.validation_row_ids.is_empty()
            || self
                .validation_row_ids
                .iter()
                .collect::<BTreeSet<_>>()
                .len()
                != self.validation_row_ids.len()
            || self.truth_shape.is_empty()
            || self.truth_shape[0] != self.validation_row_ids.len()
            || self.truth_shape.contains(&0)
        {
            return Err(
                "scoring tape truth must be non-empty, unique-row, and shape aligned".into(),
            );
        }
        let expected = self
            .truth_shape
            .iter()
            .try_fold(1usize, |product, dimension| product.checked_mul(*dimension))
            .ok_or_else(|| "truth shape overflows usize".to_string())?;
        if expected != self.truth_f64_bits.len()
            || !self
                .truth_f64_bits
                .iter()
                .all(|bits| f64::from_bits(*bits).is_finite())
        {
            return Err("scoring tape truth bits are mis-sized or nonfinite".into());
        }
        if self
            .auxiliary_scale_f64_bits
            .is_some_and(|bits| !f64::from_bits(bits).is_finite() || f64::from_bits(bits) <= 0.0)
        {
            return Err("scoring tape auxiliary scale must be finite and positive".into());
        }
        Ok(())
    }

    pub fn identity(&self) -> Result<String, String> {
        self.validate()?;
        canonical_sha256(self)
    }

    fn truth(&self) -> Vec<f64> {
        self.truth_f64_bits
            .iter()
            .map(|bits| f64::from_bits(*bits))
            .collect()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScoreKeyV2 {
    pub schema: String,
    pub scoring_tape_sha256: String,
    pub registered_fitness_sha256: String,
}

impl ScoreKeyV2 {
    pub const SCHEMA: &'static str = "ScoreKey.v2";

    pub fn validate(&self) -> Result<(), String> {
        validate_schema(&self.schema, Self::SCHEMA)?;
        validate_sha256("scoring_tape_sha256", &self.scoring_tape_sha256)?;
        validate_sha256("registered_fitness_sha256", &self.registered_fitness_sha256)
    }

    pub fn identity(&self) -> Result<String, String> {
        self.validate()?;
        canonical_sha256(self)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TypedMetricValueV1 {
    pub task: SearchTaskV1,
    pub metric: String,
    pub value: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GuardValueV1 {
    pub name: String,
    pub value: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FitnessVectorV1 {
    pub schema: String,
    pub evidence_valid: bool,
    pub invalid_reason: Option<String>,
    pub constraints_satisfied: bool,
    pub primary: TypedMetricValueV1,
    pub stability_guards: Vec<GuardValueV1>,
    pub complexity: u64,
    pub candidate_semantic_sha256: String,
}

impl FitnessVectorV1 {
    pub const SCHEMA: &'static str = "FitnessVector.v1";

    pub fn validate(&self) -> Result<(), String> {
        validate_schema(&self.schema, Self::SCHEMA)?;
        validate_sha256("candidate_semantic_sha256", &self.candidate_semantic_sha256)?;
        if self.primary.metric.is_empty()
            || !self.primary.value.is_finite()
            || self
                .stability_guards
                .iter()
                .any(|guard| guard.name.is_empty() || !guard.value.is_finite())
            || !self
                .stability_guards
                .windows(2)
                .all(|window| window[0].name < window[1].name)
        {
            return Err("fitness metric and sorted guard values must be named and finite".into());
        }
        match (self.evidence_valid, self.invalid_reason.as_deref()) {
            (true, None) => Ok(()),
            (false, Some(reason)) if !reason.trim().is_empty() && !self.constraints_satisfied => {
                Ok(())
            }
            _ => Err("fitness validity, invalid_reason, and constraints are inconsistent".into()),
        }
    }

    /// Selection ordering. `Ordering::Less` means `self` ranks ahead of `other`.
    pub fn compare_for_selection(&self, other: &Self) -> Ordering {
        other
            .constraints_satisfied
            .cmp(&self.constraints_satisfied)
            .then_with(|| other.evidence_valid.cmp(&self.evidence_valid))
            .then_with(|| other.primary.value.total_cmp(&self.primary.value))
            .then_with(|| {
                for (left, right) in self.stability_guards.iter().zip(&other.stability_guards) {
                    let ordering = right.value.total_cmp(&left.value);
                    if ordering != Ordering::Equal {
                        return ordering;
                    }
                    let name_ordering = left.name.cmp(&right.name);
                    if name_ordering != Ordering::Equal {
                        return name_ordering;
                    }
                }
                other
                    .stability_guards
                    .len()
                    .cmp(&self.stability_guards.len())
            })
            .then_with(|| self.complexity.cmp(&other.complexity))
            .then_with(|| {
                self.candidate_semantic_sha256
                    .cmp(&other.candidate_semantic_sha256)
            })
    }
}

fn invalid_fitness(
    task: SearchTaskV1,
    metric: &str,
    candidate_semantic_sha256: &str,
    complexity: u64,
    reason: String,
) -> FitnessVectorV1 {
    FitnessVectorV1 {
        schema: FitnessVectorV1::SCHEMA.into(),
        evidence_valid: false,
        invalid_reason: Some(reason),
        constraints_satisfied: false,
        primary: TypedMetricValueV1 {
            task,
            metric: metric.into(),
            value: 0.0,
        },
        stability_guards: Vec::new(),
        complexity,
        candidate_semantic_sha256: candidate_semantic_sha256.into(),
    }
}

pub fn score_prediction(
    prediction: &PredictionObjectV1,
    tape: &ScoringTapeV2,
    fitness: &RegisteredFitnessV1,
    complexity: u64,
) -> Result<(ScoreKeyV2, FitnessVectorV1), String> {
    prediction.validate()?;
    tape.validate()?;
    let prediction_sha256 = prediction.identity()?;
    if tape.prediction_object_sha256 != prediction_sha256
        || tape.validation_row_ids != prediction.validation_row_ids
    {
        return Err(
            "scoring tape does not bind the supplied prediction object and row order".into(),
        );
    }
    let fitness_sha256 = fitness.identity()?;
    let key = ScoreKeyV2 {
        schema: ScoreKeyV2::SCHEMA.into(),
        scoring_tape_sha256: tape.identity()?,
        registered_fitness_sha256: fitness_sha256,
    };
    key.validate()?;
    let (task, metric) = match fitness {
        RegisteredFitnessV1::RobustR2 { .. } => (SearchTaskV1::Regression, "robust_r2"),
        RegisteredFitnessV1::BinaryAuc { .. } => (SearchTaskV1::Binary, "auc"),
        RegisteredFitnessV1::MulticlassMacroF1 { .. } => (SearchTaskV1::Multiclass, "macro_f1"),
        RegisteredFitnessV1::NegativeMase { .. } => (SearchTaskV1::Timeseries, "negative_mase"),
    };
    if !prediction.valid {
        let result = invalid_fitness(
            task,
            metric,
            &prediction.candidate_semantic_sha256,
            complexity,
            prediction.invalid_reason.clone().unwrap(),
        );
        result.validate()?;
        return Ok((key, result));
    }

    let predictions = prediction.values();
    let truth = tape.truth();
    let result = match fitness {
        RegisteredFitnessV1::RobustR2 { max_rmse } => {
            if prediction.prediction_shape != vec![truth.len()]
                || tape.truth_shape != vec![truth.len()]
            {
                return Err("regression prediction and truth must both be one-dimensional".into());
            }
            let mean = truth.iter().sum::<f64>() / truth.len() as f64;
            let squared_error = predictions
                .iter()
                .zip(&truth)
                .map(|(prediction, truth)| (prediction - truth).powi(2))
                .sum::<f64>();
            let total = truth
                .iter()
                .map(|truth| (truth - mean).powi(2))
                .sum::<f64>();
            let rmse = (squared_error / truth.len() as f64).sqrt();
            let r2 = if total <= f64::EPSILON {
                if squared_error <= f64::EPSILON {
                    1.0
                } else {
                    0.0
                }
            } else {
                1.0 - squared_error / total
            };
            FitnessVectorV1 {
                schema: FitnessVectorV1::SCHEMA.into(),
                evidence_valid: true,
                invalid_reason: None,
                constraints_satisfied: rmse <= *max_rmse,
                primary: TypedMetricValueV1 {
                    task,
                    metric: metric.into(),
                    value: r2,
                },
                stability_guards: vec![GuardValueV1 {
                    name: "negative_rmse".into(),
                    value: -rmse,
                }],
                complexity,
                candidate_semantic_sha256: prediction.candidate_semantic_sha256.clone(),
            }
        }
        RegisteredFitnessV1::BinaryAuc {
            lift_depths,
            max_log_loss,
        } => {
            if prediction.prediction_shape != vec![truth.len()]
                || tape.truth_shape != vec![truth.len()]
            {
                return Err("binary prediction and truth must both be one-dimensional".into());
            }
            if predictions.iter().any(|value| !(0.0..=1.0).contains(value))
                || truth.iter().any(|value| *value != 0.0 && *value != 1.0)
            {
                return Ok((
                    key,
                    invalid_fitness(
                        task,
                        metric,
                        &prediction.candidate_semantic_sha256,
                        complexity,
                        "binary values are outside their registered domains".into(),
                    ),
                ));
            }
            let positives = truth.iter().filter(|value| **value == 1.0).count();
            let negatives = truth.len() - positives;
            if positives == 0 || negatives == 0 {
                return Ok((
                    key,
                    invalid_fitness(
                        task,
                        metric,
                        &prediction.candidate_semantic_sha256,
                        complexity,
                        "binary validation truth lacks class support".into(),
                    ),
                ));
            }
            let mut pairs: Vec<_> = predictions
                .iter()
                .copied()
                .zip(truth.iter().copied())
                .collect();
            pairs.sort_by(|left, right| left.0.total_cmp(&right.0));
            let mut rank_sum = 0.0;
            let mut start = 0;
            while start < pairs.len() {
                let mut end = start + 1;
                while end < pairs.len() && pairs[end].0.to_bits() == pairs[start].0.to_bits() {
                    end += 1;
                }
                let average_rank = ((start + 1 + end) as f64) / 2.0;
                rank_sum += average_rank
                    * pairs[start..end]
                        .iter()
                        .filter(|(_, truth)| *truth == 1.0)
                        .count() as f64;
                start = end;
            }
            let auc = (rank_sum - (positives * (positives + 1) / 2) as f64)
                / (positives * negatives) as f64;
            let epsilon = 1e-15;
            let log_loss = predictions
                .iter()
                .zip(&truth)
                .map(|(prediction, truth)| {
                    let prediction = prediction.clamp(epsilon, 1.0 - epsilon);
                    -(truth * prediction.ln() + (1.0 - truth) * (1.0 - prediction).ln())
                })
                .sum::<f64>()
                / truth.len() as f64;
            pairs.sort_by(|left, right| right.0.total_cmp(&left.0));
            let prevalence = positives as f64 / truth.len() as f64;
            let mut guards = vec![GuardValueV1 {
                name: "negative_log_loss".into(),
                value: -log_loss,
            }];
            for depth in lift_depths {
                let count = ((*depth * pairs.len() as f64).ceil() as usize)
                    .max(1)
                    .min(pairs.len());
                let rate = pairs[..count]
                    .iter()
                    .filter(|(_, truth)| *truth == 1.0)
                    .count() as f64
                    / count as f64;
                guards.push(GuardValueV1 {
                    name: format!("lift_at_{:016x}", depth.to_bits()),
                    value: rate / prevalence,
                });
            }
            guards.sort_by(|left, right| left.name.cmp(&right.name));
            FitnessVectorV1 {
                schema: FitnessVectorV1::SCHEMA.into(),
                evidence_valid: true,
                invalid_reason: None,
                constraints_satisfied: log_loss <= *max_log_loss,
                primary: TypedMetricValueV1 {
                    task,
                    metric: metric.into(),
                    value: auc,
                },
                stability_guards: guards,
                complexity,
                candidate_semantic_sha256: prediction.candidate_semantic_sha256.clone(),
            }
        }
        RegisteredFitnessV1::MulticlassMacroF1 {
            classes,
            min_class_support,
            max_log_loss,
        } => {
            let classes = *classes as usize;
            if prediction.prediction_shape != vec![truth.len(), classes]
                || tape.truth_shape != vec![truth.len()]
            {
                return Err(
                    "multiclass prediction must be [rows, classes] with scalar class truth".into(),
                );
            }
            let mut support = vec![0u64; classes];
            let mut predicted = Vec::with_capacity(truth.len());
            let mut log_loss = 0.0;
            for (row, truth) in truth.iter().enumerate() {
                let class = *truth as usize;
                if *truth != class as f64 || class >= classes {
                    return Ok((
                        key,
                        invalid_fitness(
                            task,
                            metric,
                            &prediction.candidate_semantic_sha256,
                            complexity,
                            "multiclass truth contains an invalid class".into(),
                        ),
                    ));
                }
                support[class] += 1;
                let probabilities = &predictions[row * classes..(row + 1) * classes];
                let sum = probabilities.iter().sum::<f64>();
                if probabilities
                    .iter()
                    .any(|value| *value < 0.0 || *value > 1.0)
                    || (sum - 1.0).abs() > 1e-9
                {
                    return Ok((
                        key,
                        invalid_fitness(
                            task,
                            metric,
                            &prediction.candidate_semantic_sha256,
                            complexity,
                            "multiclass probabilities are outside the simplex".into(),
                        ),
                    ));
                }
                let winner = probabilities
                    .iter()
                    .enumerate()
                    .max_by(|left, right| {
                        left.1.total_cmp(right.1).then_with(|| right.0.cmp(&left.0))
                    })
                    .unwrap()
                    .0;
                predicted.push(winner);
                log_loss -= probabilities[class].clamp(1e-15, 1.0).ln();
            }
            log_loss /= truth.len() as f64;
            let mut macro_f1 = 0.0;
            for class in 0..classes {
                let mut true_positive = 0u64;
                let mut false_positive = 0u64;
                let mut false_negative = 0u64;
                for (row, truth) in truth.iter().enumerate() {
                    let actual = *truth as usize;
                    let predicted = predicted[row];
                    true_positive += u64::from(actual == class && predicted == class);
                    false_positive += u64::from(actual != class && predicted == class);
                    false_negative += u64::from(actual == class && predicted != class);
                }
                let denominator = 2 * true_positive + false_positive + false_negative;
                macro_f1 += if denominator == 0 {
                    0.0
                } else {
                    2.0 * true_positive as f64 / denominator as f64
                };
            }
            macro_f1 /= classes as f64;
            FitnessVectorV1 {
                schema: FitnessVectorV1::SCHEMA.into(),
                evidence_valid: true,
                invalid_reason: None,
                constraints_satisfied: support.iter().all(|count| *count >= *min_class_support)
                    && log_loss <= *max_log_loss,
                primary: TypedMetricValueV1 {
                    task,
                    metric: metric.into(),
                    value: macro_f1,
                },
                stability_guards: vec![
                    GuardValueV1 {
                        name: "minimum_class_support".into(),
                        value: *support.iter().min().unwrap() as f64,
                    },
                    GuardValueV1 {
                        name: "negative_log_loss".into(),
                        value: -log_loss,
                    },
                ],
                complexity,
                candidate_semantic_sha256: prediction.candidate_semantic_sha256.clone(),
            }
        }
        RegisteredFitnessV1::NegativeMase { max_mase } => {
            if prediction.prediction_shape != vec![truth.len()]
                || tape.truth_shape != vec![truth.len()]
            {
                return Err("timeseries prediction and truth must both be one-dimensional".into());
            }
            let Some(scale_bits) = tape.auxiliary_scale_f64_bits else {
                return Err("timeseries scoring tape requires the native MASE scale".into());
            };
            let scale = f64::from_bits(scale_bits);
            let mase = predictions
                .iter()
                .zip(&truth)
                .map(|(prediction, truth)| (prediction - truth).abs())
                .sum::<f64>()
                / truth.len() as f64
                / scale;
            FitnessVectorV1 {
                schema: FitnessVectorV1::SCHEMA.into(),
                evidence_valid: true,
                invalid_reason: None,
                constraints_satisfied: mase <= *max_mase,
                primary: TypedMetricValueV1 {
                    task,
                    metric: metric.into(),
                    value: -mase,
                },
                stability_guards: vec![GuardValueV1 {
                    name: "negative_native_mase".into(),
                    value: -mase,
                }],
                complexity,
                candidate_semantic_sha256: prediction.candidate_semantic_sha256.clone(),
            }
        }
    };
    result.validate()?;
    Ok((key, result))
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationRecordV2 {
    pub schema: String,
    pub candidate_slot: u32,
    pub fold_index: u32,
    pub prediction: PredictionObjectV1,
    pub scoring_tape: ScoringTapeV2,
    pub score_key: ScoreKeyV2,
    pub fitness: FitnessVectorV1,
}

impl EvaluationRecordV2 {
    pub const SCHEMA: &'static str = "SearchEvaluationRecord.v2";

    pub fn validate_structural(&self) -> Result<(), String> {
        validate_schema(&self.schema, Self::SCHEMA)?;
        self.prediction.validate()?;
        self.scoring_tape.validate()?;
        self.score_key.validate()?;
        self.fitness.validate()?;
        if self.fold_index != self.prediction.fold_index
            || self.fitness.candidate_semantic_sha256 != self.prediction.candidate_semantic_sha256
            || self.scoring_tape.prediction_object_sha256 != self.prediction.identity()?
            || self.score_key.scoring_tape_sha256 != self.scoring_tape.identity()?
        {
            return Err("evaluation record identities or fold binding disagree".into());
        }
        Ok(())
    }

    pub fn verify_score(
        &self,
        registered: &RegisteredFitnessV1,
        complexity: u64,
    ) -> Result<(), String> {
        self.validate_structural()?;
        let (key, fitness) =
            score_prediction(&self.prediction, &self.scoring_tape, registered, complexity)?;
        if key != self.score_key || fitness != self.fitness {
            return Err(
                "evaluation score does not recompute from prediction and scoring tape".into(),
            );
        }
        Ok(())
    }

    pub fn identity(&self) -> Result<String, String> {
        self.validate_structural()?;
        canonical_sha256(self)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CandidateEvidenceV1 {
    pub candidate: SearchCandidateIdentityV1,
    pub evaluations: BTreeMap<u32, EvaluationRecordV2>,
}

impl CandidateEvidenceV1 {
    pub fn validate_structural(&self) -> Result<(), String> {
        self.candidate.validate()?;
        for (fold, evaluation) in &self.evaluations {
            evaluation.validate_structural()?;
            if *fold != evaluation.fold_index
                || evaluation.candidate_slot != self.candidate.slot
                || evaluation.prediction.candidate_semantic_sha256 != self.candidate.semantic_sha256
            {
                return Err(
                    "candidate evidence has a mismatched slot, fold, or semantic identity".into(),
                );
            }
        }
        Ok(())
    }

    pub fn evidence_bitmap(&self) -> BTreeSet<u32> {
        self.evaluations.keys().copied().collect()
    }
}

fn aggregate(candidate: &CandidateEvidenceV1, folds: &[u32]) -> Result<FitnessVectorV1, String> {
    if folds.is_empty() {
        return Err("cannot aggregate an empty evidence bitmap".into());
    }
    let evaluations: Vec<_> = folds
        .iter()
        .map(|fold| {
            candidate
                .evaluations
                .get(fold)
                .ok_or_else(|| format!("candidate {} lacks fold {fold}", candidate.candidate.slot))
        })
        .collect::<Result<_, _>>()?;
    let first = &evaluations[0].fitness;
    if evaluations.iter().any(|evaluation| {
        evaluation.fitness.primary.task != first.primary.task
            || evaluation.fitness.primary.metric != first.primary.metric
            || evaluation.fitness.stability_guards.len() != first.stability_guards.len()
            || evaluation
                .fitness
                .stability_guards
                .iter()
                .zip(&first.stability_guards)
                .any(|(left, right)| left.name != right.name)
    }) {
        return Err("candidate fold fitness vectors do not share one registered shape".into());
    }
    let count = evaluations.len() as f64;
    let valid = evaluations
        .iter()
        .all(|evaluation| evaluation.fitness.evidence_valid);
    let mut result = FitnessVectorV1 {
        schema: FitnessVectorV1::SCHEMA.into(),
        evidence_valid: valid,
        invalid_reason: (!valid).then(|| "one or more declared evaluations are invalid".into()),
        constraints_satisfied: valid
            && evaluations
                .iter()
                .all(|evaluation| evaluation.fitness.constraints_satisfied),
        primary: TypedMetricValueV1 {
            task: first.primary.task,
            metric: first.primary.metric.clone(),
            value: evaluations
                .iter()
                .map(|evaluation| evaluation.fitness.primary.value)
                .sum::<f64>()
                / count,
        },
        stability_guards: first
            .stability_guards
            .iter()
            .enumerate()
            .map(|(index, guard)| GuardValueV1 {
                name: guard.name.clone(),
                value: evaluations
                    .iter()
                    .map(|evaluation| evaluation.fitness.stability_guards[index].value)
                    .sum::<f64>()
                    / count,
            })
            .collect(),
        complexity: first.complexity,
        candidate_semantic_sha256: candidate.candidate.semantic_sha256.clone(),
    };
    if !valid {
        result.primary.value = 0.0;
        result.stability_guards.clear();
    }
    result.validate()?;
    Ok(result)
}

fn rank_on_common_evidence(
    candidates: &[&CandidateEvidenceV1],
    folds: &[u32],
) -> Result<Vec<u32>, String> {
    let required: BTreeSet<_> = folds.iter().copied().collect();
    if candidates
        .iter()
        .any(|candidate| !required.is_subset(&candidate.evidence_bitmap()))
    {
        return Err("a comparison attempted to use unequal evidence bitmaps".into());
    }
    let mut ranked: Vec<_> = candidates
        .iter()
        .map(|candidate| Ok((candidate.candidate.slot, aggregate(candidate, folds)?)))
        .collect::<Result<_, String>>()?;
    ranked.sort_by(|left, right| left.1.compare_for_selection(&right.1));
    Ok(ranked.into_iter().map(|(slot, _)| slot).collect())
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PromotionScheduleV1 {
    pub schema: String,
    pub fold_count: u32,
    pub fold_one_slots: Vec<u32>,
    pub promoted_thirty_slots: Vec<u32>,
    pub fold_two_slots: Vec<u32>,
    pub finalist_eight_slots: Vec<u32>,
    pub complete_fold_slots: BTreeMap<u32, Vec<u32>>,
    pub fold_evaluation_count: u64,
}

impl PromotionScheduleV1 {
    pub const SCHEMA: &'static str = "SearchPromotionSchedule.v1";

    pub fn build(
        plan: &SearchCandidatePlanV1,
        evidence: &[CandidateEvidenceV1],
        fold_count: u32,
    ) -> Result<Self, String> {
        plan.validate()?;
        if fold_count < 3 || evidence.len() != 60 {
            return Err(
                "promotion schedule requires 60 candidates and at least three folds".into(),
            );
        }
        for (slot, candidate) in evidence.iter().enumerate() {
            candidate.validate_structural()?;
            if candidate.candidate != plan.candidates[slot] {
                return Err("candidate evidence must be in exact plan slot order".into());
            }
        }
        let all: Vec<_> = evidence.iter().collect();
        let fold_one_slots = rank_on_common_evidence(&all, &[0])?;
        let promoted_thirty_slots = fold_one_slots[..30].to_vec();
        let promoted: Vec<_> = promoted_thirty_slots
            .iter()
            .map(|slot| &evidence[*slot as usize])
            .collect();
        let fold_two_slots = rank_on_common_evidence(&promoted, &[0, 1])?;
        let finalist_eight_slots = fold_two_slots[..8].to_vec();
        let finalists: Vec<_> = finalist_eight_slots
            .iter()
            .map(|slot| &evidence[*slot as usize])
            .collect();
        let complete_folds: Vec<_> = (0..fold_count).collect();
        let final_rank = rank_on_common_evidence(&finalists, &complete_folds)?;

        let promoted_set: BTreeSet<_> = promoted_thirty_slots.iter().copied().collect();
        let finalist_set: BTreeSet<_> = finalist_eight_slots.iter().copied().collect();
        for candidate in evidence {
            let expected: BTreeSet<u32> = if finalist_set.contains(&candidate.candidate.slot) {
                (0..fold_count).collect()
            } else if promoted_set.contains(&candidate.candidate.slot) {
                [0, 1].into_iter().collect()
            } else {
                [0].into_iter().collect()
            };
            if candidate.evidence_bitmap() != expected {
                return Err(format!(
                    "candidate {} has evidence {:?}, expected {:?}",
                    candidate.candidate.slot,
                    candidate.evidence_bitmap(),
                    expected
                ));
            }
        }
        let complete_fold_slots = (0..fold_count)
            .map(|fold| {
                let slots = if fold < 2 {
                    if fold == 0 {
                        (0..60).collect()
                    } else {
                        promoted_thirty_slots.clone()
                    }
                } else {
                    finalist_eight_slots.clone()
                };
                (fold, slots)
            })
            .collect();
        Ok(Self {
            schema: Self::SCHEMA.into(),
            fold_count,
            fold_one_slots,
            promoted_thirty_slots,
            fold_two_slots,
            finalist_eight_slots: final_rank,
            complete_fold_slots,
            fold_evaluation_count: 60 + 30 + u64::from(fold_count - 2) * 8,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConfirmedCandidateV1 {
    pub first_confirmed_generation: u32,
    pub candidate: SearchCandidateIdentityV1,
    pub complete_evidence_bitmap: BTreeSet<u32>,
    pub aggregate_fitness: FitnessVectorV1,
    pub evidence_root_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConfirmedArchiveV1 {
    pub schema: String,
    pub candidates: Vec<ConfirmedCandidateV1>,
}

impl ConfirmedArchiveV1 {
    pub const SCHEMA: &'static str = "ConfirmedArchive.v1";

    pub fn empty() -> Self {
        Self {
            schema: Self::SCHEMA.into(),
            candidates: Vec::new(),
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        validate_schema(&self.schema, Self::SCHEMA)?;
        if !self
            .candidates
            .windows(2)
            .all(|window| window[0].candidate.semantic_sha256 < window[1].candidate.semantic_sha256)
        {
            return Err("confirmed archive must be unique and semantic-digest ordered".into());
        }
        for candidate in &self.candidates {
            candidate.candidate.validate()?;
            candidate.aggregate_fitness.validate()?;
            validate_sha256("evidence_root_sha256", &candidate.evidence_root_sha256)?;
            if candidate.complete_evidence_bitmap.len() < 3
                || candidate.aggregate_fitness.candidate_semantic_sha256
                    != candidate.candidate.semantic_sha256
            {
                return Err("confirmed archive contains incomplete or mismatched evidence".into());
            }
        }
        Ok(())
    }

    pub fn identity(&self) -> Result<String, String> {
        self.validate()?;
        canonical_sha256(self)
    }

    fn merged_generation(
        &self,
        generation: u32,
        schedule: &PromotionScheduleV1,
        evidence: &[CandidateEvidenceV1],
    ) -> Result<Self, String> {
        self.validate()?;
        let complete_folds: Vec<_> = (0..schedule.fold_count).collect();
        let mut merged: BTreeMap<String, ConfirmedCandidateV1> = self
            .candidates
            .iter()
            .cloned()
            .map(|candidate| (candidate.candidate.semantic_sha256.clone(), candidate))
            .collect();
        for slot in &schedule.finalist_eight_slots {
            let candidate = &evidence[*slot as usize];
            let archived = ConfirmedCandidateV1 {
                first_confirmed_generation: generation,
                candidate: candidate.candidate.clone(),
                complete_evidence_bitmap: complete_folds.iter().copied().collect(),
                aggregate_fitness: aggregate(candidate, &complete_folds)?,
                evidence_root_sha256: canonical_sha256(candidate)?,
            };
            match merged.get(&candidate.candidate.semantic_sha256) {
                Some(existing) if existing != &archived => {
                    return Err(
                        "same candidate semantic identity has differing confirmed evidence".into(),
                    );
                }
                Some(_) => {}
                None => {
                    merged.insert(candidate.candidate.semantic_sha256.clone(), archived);
                }
            }
        }
        let archive = Self {
            schema: Self::SCHEMA.into(),
            candidates: merged.into_values().collect(),
        };
        archive.validate()?;
        Ok(archive)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SelectionFreezeV2 {
    pub schema: String,
    pub archive_sha256: String,
    pub winner_semantic_sha256: String,
    pub confirmed_generations: Vec<u32>,
}

impl SelectionFreezeV2 {
    pub const SCHEMA: &'static str = "SelectionFreeze.v2";

    pub fn from_archive(archive: &ConfirmedArchiveV1) -> Result<Self, String> {
        archive.validate()?;
        let winner = archive
            .candidates
            .iter()
            .min_by(|left, right| {
                left.aggregate_fitness
                    .compare_for_selection(&right.aggregate_fitness)
            })
            .ok_or_else(|| "cannot freeze selection from an empty archive".to_string())?;
        let confirmed_generations = archive
            .candidates
            .iter()
            .map(|candidate| candidate.first_confirmed_generation)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        Ok(Self {
            schema: Self::SCHEMA.into(),
            archive_sha256: archive.identity()?,
            winner_semantic_sha256: winner.candidate.semantic_sha256.clone(),
            confirmed_generations,
        })
    }

    pub fn validate_against(&self, archive: &ConfirmedArchiveV1) -> Result<(), String> {
        validate_schema(&self.schema, Self::SCHEMA)?;
        validate_sha256("archive_sha256", &self.archive_sha256)?;
        validate_sha256("winner_semantic_sha256", &self.winner_semantic_sha256)?;
        if self != &Self::from_archive(archive)? {
            return Err(
                "selection freeze is not the total-order winner over the complete archive".into(),
            );
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenerationLedgerEntryV2 {
    pub slot: u32,
    pub candidate_semantic_sha256: String,
    pub ancestry_sha256: String,
    pub producer: CandidateProducerV1,
    pub seed: u64,
    pub complexity: u64,
    pub compute_units: u64,
    pub evaluation_sha256s: BTreeMap<u32, String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenerationLedgerV2 {
    pub schema: String,
    pub contract_sha256: String,
    pub generation: u32,
    pub previous_generation_state_sha256: Option<String>,
    pub slots: Vec<GenerationLedgerEntryV2>,
    pub archive_sha256: String,
    pub selection_sha256: String,
}

impl GenerationLedgerV2 {
    pub const SCHEMA: &'static str = "GenerationCommit.v2";
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenerationStateV2 {
    pub schema: String,
    pub contract: SearchRunContractV3,
    pub contract_sha256: String,
    pub generation: u32,
    pub previous_generation_state_sha256: Option<String>,
    pub candidate_plan: SearchCandidatePlanV1,
    pub evidence: Vec<CandidateEvidenceV1>,
    pub promotion_schedule: PromotionScheduleV1,
    pub archive: ConfirmedArchiveV1,
    pub selection: SelectionFreezeV2,
    pub ledger: GenerationLedgerV2,
    pub ledger_sha256: String,
}

impl GenerationStateV2 {
    pub const SCHEMA: &'static str = "SearchGenerationState.v2";

    pub fn validate_structural(&self) -> Result<(), String> {
        validate_schema(&self.schema, Self::SCHEMA)?;
        self.contract.validate()?;
        self.candidate_plan.validate()?;
        if self.generation != self.candidate_plan.generation
            || self.contract_sha256 != self.contract.identity()?
            || self.contract.candidate_plan_root_sha256 != self.candidate_plan.identity()?
            || self.promotion_schedule
                != PromotionScheduleV1::build(
                    &self.candidate_plan,
                    &self.evidence,
                    self.contract.budget.declared_fold_count,
                )?
        {
            return Err(
                "generation state contract, plan, or promotion schedule does not rebuild".into(),
            );
        }
        self.archive.validate()?;
        self.selection.validate_against(&self.archive)?;
        if self.ledger.schema != GenerationLedgerV2::SCHEMA
            || self.ledger.generation != self.generation
            || self.ledger.contract_sha256 != self.contract_sha256
            || self.ledger.previous_generation_state_sha256 != self.previous_generation_state_sha256
            || self.ledger.archive_sha256 != self.archive.identity()?
            || self.ledger.selection_sha256 != canonical_sha256(&self.selection)?
            || self.ledger_sha256 != canonical_sha256(&self.ledger)?
            || self.ledger.slots.len() != 60
        {
            return Err("generation semantic ledger does not bind the complete state".into());
        }
        for (slot, entry) in self.ledger.slots.iter().enumerate() {
            let candidate = &self.evidence[slot];
            let expected: BTreeMap<_, _> = candidate
                .evaluations
                .iter()
                .map(|(fold, evaluation)| Ok((*fold, evaluation.identity()?)))
                .collect::<Result<_, String>>()?;
            if entry.slot != slot as u32
                || entry.candidate_semantic_sha256 != candidate.candidate.semantic_sha256
                || entry.ancestry_sha256 != candidate.candidate.ancestry_sha256
                || entry.producer != candidate.candidate.producer
                || entry.seed != candidate.candidate.seed
                || entry.complexity != candidate.candidate.complexity
                || entry.compute_units != candidate.candidate.compute_units
                || entry.evaluation_sha256s != expected
            {
                return Err(format!(
                    "semantic ledger slot {slot} does not match evidence"
                ));
            }
        }
        Ok(())
    }

    pub fn identity(&self) -> Result<String, String> {
        self.validate_structural()?;
        canonical_sha256(self)
    }
}

pub fn build_generation_state(
    contract: SearchRunContractV3,
    candidate_plan: SearchCandidatePlanV1,
    mut evidence: Vec<CandidateEvidenceV1>,
    previous_generation_state_sha256: Option<String>,
    previous_archive: &ConfirmedArchiveV1,
) -> Result<GenerationStateV2, String> {
    contract.validate()?;
    candidate_plan.validate()?;
    if candidate_plan.generation > 0 && previous_generation_state_sha256.is_none() {
        return Err("non-genesis generation requires the previous complete state identity".into());
    }
    if let Some(previous) = &previous_generation_state_sha256 {
        validate_sha256("previous_generation_state_sha256", previous)?;
    }
    // Worker completion order is runtime topology, not scientific identity.
    evidence.sort_by_key(|candidate| candidate.candidate.slot);
    let contract_sha256 = contract.identity()?;
    let promotion_schedule = PromotionScheduleV1::build(
        &candidate_plan,
        &evidence,
        contract.budget.declared_fold_count,
    )?;
    let archive = previous_archive.merged_generation(
        candidate_plan.generation,
        &promotion_schedule,
        &evidence,
    )?;
    let selection = SelectionFreezeV2::from_archive(&archive)?;
    let slots = evidence
        .iter()
        .map(|candidate| {
            Ok(GenerationLedgerEntryV2 {
                slot: candidate.candidate.slot,
                candidate_semantic_sha256: candidate.candidate.semantic_sha256.clone(),
                ancestry_sha256: candidate.candidate.ancestry_sha256.clone(),
                producer: candidate.candidate.producer,
                seed: candidate.candidate.seed,
                complexity: candidate.candidate.complexity,
                compute_units: candidate.candidate.compute_units,
                evaluation_sha256s: candidate
                    .evaluations
                    .iter()
                    .map(|(fold, evaluation)| Ok((*fold, evaluation.identity()?)))
                    .collect::<Result<_, String>>()?,
            })
        })
        .collect::<Result<_, String>>()?;
    let ledger = GenerationLedgerV2 {
        schema: GenerationLedgerV2::SCHEMA.into(),
        contract_sha256: contract_sha256.clone(),
        generation: candidate_plan.generation,
        previous_generation_state_sha256: previous_generation_state_sha256.clone(),
        slots,
        archive_sha256: archive.identity()?,
        selection_sha256: canonical_sha256(&selection)?,
    };
    let ledger_sha256 = canonical_sha256(&ledger)?;
    let state = GenerationStateV2 {
        schema: GenerationStateV2::SCHEMA.into(),
        contract,
        contract_sha256,
        generation: candidate_plan.generation,
        previous_generation_state_sha256,
        candidate_plan,
        evidence,
        promotion_schedule,
        archive,
        selection,
        ledger,
        ledger_sha256,
    };
    state.validate_structural()?;
    Ok(state)
}
