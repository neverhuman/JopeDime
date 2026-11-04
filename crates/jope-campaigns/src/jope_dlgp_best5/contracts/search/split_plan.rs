use super::*;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TargetRowV1 {
    pub stable_row_id: String,
    pub atomic_group: String,
    pub target: f64,
}

impl TargetRowV1 {
    pub fn validate(&self) -> Result<(), String> {
        nonempty("stable row ID", &self.stable_row_id)?;
        nonempty("atomic group", &self.atomic_group)?;
        finite("regression target", self.target)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DataSplitV1 {
    pub split_index: u32,
    pub split_seed: u64,
    pub train_row_ids: Vec<String>,
    pub validation_row_ids: Vec<String>,
    pub validation_groups: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SearchSplitPlanV1 {
    pub schema: String,
    pub dataset_sha256: String,
    pub training_sha256: String,
    pub master_seed: u64,
    pub stable_row_ids: Vec<String>,
    pub row_groups: BTreeMap<String, String>,
    pub splits: Vec<DataSplitV1>,
}

impl SearchSplitPlanV1 {
    pub const SCHEMA: &'static str = "JopeDlgpBest5SearchSplitPlan.v1";

    pub fn validate(&self) -> Result<(), String> {
        self.validate_n(Self::SCHEMA, SEARCH_SPLITS, true)
    }

    pub(crate) fn validate_n(
        &self,
        expected_schema: &str,
        count: usize,
        distinct: bool,
    ) -> Result<(), String> {
        schema(&self.schema, expected_schema)?;
        validate_sha256("split dataset", &self.dataset_sha256)?;
        validate_sha256("split training", &self.training_sha256)?;
        if !SEARCH_SEEDS.contains(&self.master_seed) {
            return Err("split plan master seed must be 101 or 202".into());
        }
        let all: BTreeSet<_> = self.stable_row_ids.iter().collect();
        if all.len() != self.stable_row_ids.len() || all.len() < count {
            return Err("split rows are duplicate or insufficient".into());
        }
        if self.row_groups.len() != all.len()
            || self
                .row_groups
                .iter()
                .any(|(row, group)| !all.contains(row) || group.is_empty())
        {
            return Err("split plan must bind every row to an atomic group".into());
        }
        if self.splits.len() != count {
            return Err(format!("split plan requires exactly {count} splits"));
        }
        let mut validation_sets = BTreeSet::new();
        for (index, split) in self.splits.iter().enumerate() {
            if split.split_index != index as u32 {
                return Err("split indices must be contiguous".into());
            }
            let train: BTreeSet<_> = split.train_row_ids.iter().collect();
            let validation: BTreeSet<_> = split.validation_row_ids.iter().collect();
            if train.len() != split.train_row_ids.len()
                || validation.len() != split.validation_row_ids.len()
                || train.is_empty()
                || validation.is_empty()
                || !train.is_disjoint(&validation)
                || train.union(&validation).copied().collect::<BTreeSet<_>>() != all
            {
                return Err("split must be a complete disjoint 80/20 partition".into());
            }
            let groups: BTreeSet<_> = split.validation_groups.iter().collect();
            if validation
                .iter()
                .any(|row| !groups.contains(&self.row_groups[*row]))
                || train
                    .iter()
                    .any(|row| groups.contains(&self.row_groups[*row]))
            {
                return Err("split crosses or omits an atomic group".into());
            }
            let validation_fraction = validation.len() as f64 / all.len() as f64;
            let target_fraction = if expected_schema == Self::SCHEMA {
                0.20
            } else {
                1.0 / count as f64
            };
            let max_group = self
                .row_groups
                .values()
                .fold(BTreeMap::<&str, usize>::new(), |mut counts, group| {
                    *counts.entry(group).or_default() += 1;
                    counts
                })
                .values()
                .copied()
                .max()
                .unwrap_or(1) as f64
                / all.len() as f64;
            if (validation_fraction - target_fraction).abs() > max_group + 0.02 {
                return Err(
                    "atomic-aware validation partition has the wrong approximate size".into(),
                );
            }
            validation_sets.insert(split.validation_row_ids.clone());
        }
        if distinct && validation_sets.len() != count {
            return Err("all eight search validation partitions must be distinct".into());
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
pub struct Cv7PlanV1(pub SearchSplitPlanV1);

impl Cv7PlanV1 {
    pub const SCHEMA: &'static str = "JopeDlgpBest5Cv7Plan.v1";

    pub fn validate(&self) -> Result<(), String> {
        self.0.validate_n(Self::SCHEMA, 7, false)
    }

    pub fn identity(&self) -> Result<String, String> {
        self.validate()?;
        canonical_sha256(self)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PredictionKeyV1 {
    pub schema: String,
    pub dataset_sha256: String,
    pub training_sha256: String,
    pub split_sha256: String,
    pub semantic_sha256: String,
    pub evaluator_sha256: String,
    pub math_policy_sha256: String,
    pub backend_policy_sha256: String,
}

impl PredictionKeyV1 {
    pub const SCHEMA: &'static str = "JopeDlgpBest5PredictionKey.v1";

    pub fn validate(&self) -> Result<(), String> {
        schema(&self.schema, Self::SCHEMA)?;
        for (name, digest) in [
            ("dataset", self.dataset_sha256.as_str()),
            ("training", self.training_sha256.as_str()),
            ("split", self.split_sha256.as_str()),
            ("semantic", self.semantic_sha256.as_str()),
            ("evaluator", self.evaluator_sha256.as_str()),
            ("math policy", self.math_policy_sha256.as_str()),
            ("backend policy", self.backend_policy_sha256.as_str()),
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
