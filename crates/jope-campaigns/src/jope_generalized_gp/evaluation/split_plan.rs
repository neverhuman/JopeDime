use super::*;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SplitRowV1 {
    pub stable_row_id: String,
    pub atomic_group: String,
    pub regression_target: Option<f64>,
    pub binary_target: Option<u8>,
}

impl SplitRowV1 {
    pub fn validate(&self, task: GpTaskV1) -> Result<(), String> {
        nonempty("stable row ID", &self.stable_row_id)?;
        nonempty("atomic group", &self.atomic_group)?;
        match task {
            GpTaskV1::Regression
                if self.regression_target.is_some_and(f64::is_finite)
                    && self.binary_target.is_none() =>
            {
                Ok(())
            }
            GpTaskV1::Binary
                if matches!(self.binary_target, Some(0 | 1))
                    && self.regression_target.is_none() =>
            {
                Ok(())
            }
            _ => Err("split row target does not match its task".into()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RowSplitV1 {
    pub split_index: u32,
    pub seed: u64,
    pub train_row_ids: Vec<String>,
    pub validation_row_ids: Vec<String>,
    pub validation_groups: Vec<String>,
    pub balance_counts: BTreeMap<String, u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RotatingSplitPlanV1 {
    pub schema: String,
    pub task: GpTaskV1,
    pub dataset_sha256: String,
    pub training_sha256: String,
    pub stable_row_ids: Vec<String>,
    pub row_groups: BTreeMap<String, String>,
    pub splits: Vec<RowSplitV1>,
}

impl RotatingSplitPlanV1 {
    pub const SCHEMA: &'static str = "JopeGeneralizedGpRotatingSplitPlan.v1";

    pub fn validate(&self) -> Result<(), String> {
        self.validate_k(Self::SCHEMA, 5, true)
    }

    pub(crate) fn validate_k(
        &self,
        expected_schema: &str,
        expected: usize,
        distinct_seeds: bool,
    ) -> Result<(), String> {
        schema(&self.schema, expected_schema)?;
        validate_sha256("split dataset", &self.dataset_sha256)?;
        validate_sha256("split training", &self.training_sha256)?;
        let all: BTreeSet<_> = self.stable_row_ids.iter().collect();
        if all.len() != self.stable_row_ids.len() || all.len() < expected {
            return Err("split plan stable rows are duplicate or insufficient".into());
        }
        if self.row_groups.len() != all.len()
            || self
                .row_groups
                .iter()
                .any(|(row, group)| !all.contains(row) || group.is_empty())
        {
            return Err("split plan must bind every stable row to an atomic group".into());
        }
        if self.splits.len() != expected {
            return Err(format!("split plan requires exactly {expected} partitions"));
        }
        let mut seeds = BTreeSet::new();
        let mut validations = BTreeSet::new();
        for (index, split) in self.splits.iter().enumerate() {
            if split.split_index != index as u32 {
                return Err("split indices must be contiguous".into());
            }
            seeds.insert(split.seed);
            let train: BTreeSet<_> = split.train_row_ids.iter().collect();
            let validation: BTreeSet<_> = split.validation_row_ids.iter().collect();
            if train.len() != split.train_row_ids.len()
                || validation.len() != split.validation_row_ids.len()
                || validation.is_empty()
                || !train.is_disjoint(&validation)
                || train.union(&validation).copied().collect::<BTreeSet<_>>() != all
            {
                return Err(format!(
                    "split {index} is not a complete disjoint partition"
                ));
            }
            let validation_groups: BTreeSet<_> = split.validation_groups.iter().collect();
            for row in &split.validation_row_ids {
                let group = self.row_groups.get(row).unwrap();
                if !validation_groups.contains(group) {
                    return Err(format!("split {index} omits validation atomic group"));
                }
            }
            for row in &split.train_row_ids {
                if validation_groups.contains(self.row_groups.get(row).unwrap()) {
                    return Err(format!("split {index} crosses an atomic group"));
                }
            }
            if self.task == GpTaskV1::Binary
                && (split.balance_counts.get("class_0").copied().unwrap_or(0) == 0
                    || split.balance_counts.get("class_1").copied().unwrap_or(0) == 0)
            {
                return Err(format!(
                    "binary split {index} lacks both validation classes"
                ));
            }
            validations.insert(split.validation_row_ids.clone());
        }
        if distinct_seeds && seeds.len() != expected {
            return Err("rotating split seeds must be distinct".into());
        }
        if validations.len() != expected {
            return Err("rotating validation partitions must be distinct".into());
        }
        Ok(())
    }

    pub fn identity(&self) -> Result<String, String> {
        self.validate()?;
        canonical_sha256(self)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Cv7PlanV1(pub RotatingSplitPlanV1);

impl Cv7PlanV1 {
    pub const SCHEMA: &'static str = "JopeGeneralizedGpCv7Plan.v1";

    pub fn validate(&self) -> Result<(), String> {
        self.0.validate_k(Self::SCHEMA, 7, false)
    }

    pub fn identity(&self) -> Result<String, String> {
        self.validate()?;
        canonical_sha256(self)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PredictionCacheKeyV1 {
    pub schema: String,
    pub dataset_sha256: String,
    pub training_sha256: String,
    pub split_sha256: String,
    pub candidate_semantic_sha256: String,
    pub evaluator_sha256: String,
    pub math_policy_sha256: String,
    pub backend_policy_sha256: String,
}

impl PredictionCacheKeyV1 {
    pub const SCHEMA: &'static str = "JopeGeneralizedGpPredictionKey.v1";

    pub fn validate(&self) -> Result<(), String> {
        schema(&self.schema, Self::SCHEMA)?;
        for (name, digest) in [
            ("prediction dataset", self.dataset_sha256.as_str()),
            ("prediction training", self.training_sha256.as_str()),
            ("prediction split", self.split_sha256.as_str()),
            (
                "prediction candidate",
                self.candidate_semantic_sha256.as_str(),
            ),
            ("prediction evaluator", self.evaluator_sha256.as_str()),
            ("prediction math policy", self.math_policy_sha256.as_str()),
            (
                "prediction backend policy",
                self.backend_policy_sha256.as_str(),
            ),
        ] {
            validate_sha256(name, digest)?;
        }
        Ok(())
    }

    pub fn identity(&self) -> Result<String, String> {
        self.validate()?;
        canonical_sha256(self)
    }
}
