use super::*;

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

    pub(super) fn values(&self) -> Vec<f64> {
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

    pub(super) fn truth(&self) -> Vec<f64> {
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
