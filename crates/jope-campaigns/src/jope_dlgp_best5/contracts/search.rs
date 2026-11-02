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

/// A complete, deployable predictor. Residual and portfolio winners are
/// represented directly instead of pretending that their last component is
/// the selected model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LicenseTrackV1 {
    ProductionCompatible,
    ResearchOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExternalModelFamilyV1 {
    CatBoost,
    LightGbm,
    XgBoost,
    GpControl,
    TabFm,
    TabPfn,
    TabIcl,
    TabDpt,
}

impl ExternalModelFamilyV1 {
    pub fn license_track(self) -> LicenseTrackV1 {
        match self {
            Self::CatBoost | Self::LightGbm | Self::XgBoost | Self::GpControl => {
                LicenseTrackV1::ProductionCompatible
            }
            Self::TabFm | Self::TabPfn | Self::TabIcl | Self::TabDpt => {
                LicenseTrackV1::ResearchOnly
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LicenseIdentityV1 {
    pub spdx_expression: String,
    pub license_text_sha256: String,
    pub upstream_source: String,
    pub upstream_revision: String,
}

impl LicenseIdentityV1 {
    pub fn validate(&self) -> Result<(), String> {
        nonempty("SPDX expression", &self.spdx_expression)?;
        validate_sha256("license text", &self.license_text_sha256)?;
        nonempty("upstream source", &self.upstream_source)?;
        nonempty("upstream revision", &self.upstream_revision)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum FrozenPredictorComponentV1 {
    GpPipeline {
        pipeline: GenomeV1,
    },
    ExternalModel {
        family: ExternalModelFamilyV1,
        artifact_sha256: String,
        artifact_bytes: u64,
        inference_protocol_sha256: String,
        license: LicenseIdentityV1,
        complexity: u64,
    },
}

impl FrozenPredictorComponentV1 {
    pub fn validate(&self) -> Result<(), String> {
        match self {
            Self::GpPipeline { pipeline } => {
                pipeline.validate()?;
                if pipeline.task != GpTaskV1::Regression {
                    return Err("predictor component GP pipeline must be a regressor".into());
                }
                Ok(())
            }
            Self::ExternalModel {
                artifact_sha256,
                artifact_bytes,
                inference_protocol_sha256,
                license,
                ..
            } => {
                validate_sha256("external model artifact", artifact_sha256)?;
                if *artifact_bytes == 0 {
                    return Err("external model artifact byte length must be positive".into());
                }
                validate_sha256("external inference protocol", inference_protocol_sha256)?;
                license.validate()
            }
        }
    }

    pub fn identity(&self) -> Result<String, String> {
        self.validate()?;
        canonical_sha256(self)
    }

    pub fn license_track(&self) -> LicenseTrackV1 {
        match self {
            Self::GpPipeline { .. } => LicenseTrackV1::ProductionCompatible,
            Self::ExternalModel { family, .. } => family.license_track(),
        }
    }

    pub fn complexity(&self) -> u64 {
        match self {
            Self::GpPipeline { pipeline } => pipeline.complexity,
            Self::ExternalModel { complexity, .. } => *complexity,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum FrozenPredictorV1 {
    Single {
        pipeline: GenomeV1,
    },
    ResidualComposite {
        base: GenomeV1,
        residual_pipeline: GenomeV1,
        eta: f64,
    },
    SparsePortfolio {
        pipelines: Vec<GenomeV1>,
        weights: Vec<f64>,
    },
    ComponentSingle {
        component: FrozenPredictorComponentV1,
        license_track: LicenseTrackV1,
    },
    ComponentResidualComposite {
        base: FrozenPredictorComponentV1,
        residual: FrozenPredictorComponentV1,
        eta: f64,
        cross_fit_protocol_sha256: String,
        license_track: LicenseTrackV1,
    },
    ComponentSparsePortfolio {
        components: Vec<FrozenPredictorComponentV1>,
        weights: Vec<f64>,
        cross_fit_protocol_sha256: String,
        license_track: LicenseTrackV1,
    },
}

impl FrozenPredictorV1 {
    pub fn single(pipeline: GenomeV1) -> Self {
        Self::Single { pipeline }
    }

    pub fn pipelines(&self) -> Vec<&GenomeV1> {
        match self {
            Self::Single { pipeline } => vec![pipeline],
            Self::ResidualComposite {
                base,
                residual_pipeline,
                ..
            } => vec![base, residual_pipeline],
            Self::SparsePortfolio { pipelines, .. } => pipelines.iter().collect(),
            Self::ComponentSingle { component, .. } => match component {
                FrozenPredictorComponentV1::GpPipeline { pipeline } => vec![pipeline],
                FrozenPredictorComponentV1::ExternalModel { .. } => Vec::new(),
            },
            Self::ComponentResidualComposite { base, residual, .. } => [base, residual]
                .into_iter()
                .filter_map(|component| match component {
                    FrozenPredictorComponentV1::GpPipeline { pipeline } => Some(pipeline),
                    FrozenPredictorComponentV1::ExternalModel { .. } => None,
                })
                .collect(),
            Self::ComponentSparsePortfolio { components, .. } => components
                .iter()
                .filter_map(|component| match component {
                    FrozenPredictorComponentV1::GpPipeline { pipeline } => Some(pipeline),
                    FrozenPredictorComponentV1::ExternalModel { .. } => None,
                })
                .collect(),
        }
    }

    pub fn component_count(&self) -> usize {
        match self {
            Self::Single { .. } => 1,
            Self::ResidualComposite { .. } => 2,
            Self::SparsePortfolio { pipelines, .. } => pipelines.len(),
            Self::ComponentSingle { .. } => 1,
            Self::ComponentResidualComposite { .. } => 2,
            Self::ComponentSparsePortfolio { components, .. } => components.len(),
        }
    }

    pub fn license_track(&self) -> LicenseTrackV1 {
        match self {
            Self::Single { .. } | Self::ResidualComposite { .. } | Self::SparsePortfolio { .. } => {
                LicenseTrackV1::ProductionCompatible
            }
            Self::ComponentSingle { license_track, .. }
            | Self::ComponentResidualComposite { license_track, .. }
            | Self::ComponentSparsePortfolio { license_track, .. } => *license_track,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.component_count() == 0 || self.component_count() > 4 {
            return Err("frozen predictor requires one through four pipelines".into());
        }
        let mut semantics = BTreeSet::new();
        for pipeline in self.pipelines() {
            pipeline.validate()?;
            if pipeline.task != GpTaskV1::Regression
                || !semantics.insert(pipeline.semantic_sha256.as_str())
            {
                return Err("frozen predictor pipelines must be unique regressors".into());
            }
        }
        match self {
            Self::Single { .. } => Ok(()),
            Self::ResidualComposite { eta, .. } => {
                if eta.is_finite() && (0.0..=1.0).contains(eta) {
                    Ok(())
                } else {
                    Err("residual eta must be finite and in [0,1]".into())
                }
            }
            Self::SparsePortfolio { pipelines, weights } => {
                if pipelines.len() != weights.len()
                    || weights
                        .iter()
                        .any(|weight| !weight.is_finite() || *weight < 0.0)
                    || (weights.iter().sum::<f64>() - 1.0).abs() > 1e-10
                {
                    return Err(
                        "portfolio weights must be a nonnegative simplex aligned to pipelines"
                            .into(),
                    );
                }
                Ok(())
            }
            Self::ComponentSingle {
                component,
                license_track,
            } => {
                component.validate()?;
                validate_component_track(*license_track, std::slice::from_ref(component))
            }
            Self::ComponentResidualComposite {
                base,
                residual,
                eta,
                cross_fit_protocol_sha256,
                license_track,
            } => {
                base.validate()?;
                residual.validate()?;
                validate_sha256("residual cross-fit protocol", cross_fit_protocol_sha256)?;
                if !eta.is_finite() || !(0.0..=1.0).contains(eta) {
                    return Err("residual eta must be finite and in [0,1]".into());
                }
                if base.identity()? == residual.identity()? {
                    return Err("residual predictor components must be unique".into());
                }
                validate_component_track(*license_track, &[base.clone(), residual.clone()])
            }
            Self::ComponentSparsePortfolio {
                components,
                weights,
                cross_fit_protocol_sha256,
                license_track,
            } => {
                validate_sha256("portfolio cross-fit protocol", cross_fit_protocol_sha256)?;
                let mut identities = BTreeSet::new();
                for component in components {
                    component.validate()?;
                    if !identities.insert(component.identity()?) {
                        return Err("portfolio predictor components must be unique".into());
                    }
                }
                if components.len() != weights.len()
                    || weights
                        .iter()
                        .any(|weight| !weight.is_finite() || *weight < 0.0)
                    || (weights.iter().sum::<f64>() - 1.0).abs() > 1e-10
                {
                    return Err(
                        "portfolio weights must be a nonnegative simplex aligned to components"
                            .into(),
                    );
                }
                validate_component_track(*license_track, components)
            }
        }
    }

    pub fn semantic_sha256(&self) -> Result<String, String> {
        self.validate()?;
        match self {
            Self::Single { pipeline } => Ok(pipeline.semantic_sha256.clone()),
            Self::ResidualComposite { .. }
            | Self::SparsePortfolio { .. }
            | Self::ComponentSingle { .. }
            | Self::ComponentResidualComposite { .. }
            | Self::ComponentSparsePortfolio { .. } => canonical_sha256(self),
        }
    }

    pub fn complexity(&self) -> u64 {
        match self {
            Self::ComponentSingle { component, .. } => component.complexity(),
            Self::ComponentResidualComposite { base, residual, .. } => {
                base.complexity() + residual.complexity()
            }
            Self::ComponentSparsePortfolio { components, .. } => components
                .iter()
                .map(FrozenPredictorComponentV1::complexity)
                .sum(),
            _ => self
                .pipelines()
                .into_iter()
                .map(|pipeline| pipeline.complexity)
                .sum(),
        }
    }

    pub fn reconstruct(&self, component_predictions: &[Vec<f64>]) -> Result<Vec<f64>, String> {
        self.validate()?;
        if component_predictions.len() != self.component_count() || component_predictions.is_empty()
        {
            return Err("predictor reconstruction has incomplete component coverage".into());
        }
        let rows = component_predictions[0].len();
        if rows == 0
            || component_predictions
                .iter()
                .any(|values| values.len() != rows || values.iter().any(|value| !value.is_finite()))
        {
            return Err("predictor component predictions are nonfinite or row-misaligned".into());
        }
        let values = match self {
            Self::Single { .. } => component_predictions[0].clone(),
            Self::ResidualComposite { eta, .. } => component_predictions[0]
                .iter()
                .zip(&component_predictions[1])
                .map(|(base, residual)| base + eta * residual)
                .collect(),
            Self::SparsePortfolio { weights, .. } => (0..rows)
                .map(|row| {
                    weights
                        .iter()
                        .zip(component_predictions)
                        .map(|(weight, prediction)| weight * prediction[row])
                        .sum()
                })
                .collect(),
            Self::ComponentSingle { .. } => component_predictions[0].clone(),
            Self::ComponentResidualComposite { eta, .. } => component_predictions[0]
                .iter()
                .zip(&component_predictions[1])
                .map(|(base, residual)| base + eta * residual)
                .collect(),
            Self::ComponentSparsePortfolio { weights, .. } => (0..rows)
                .map(|row| {
                    weights
                        .iter()
                        .zip(component_predictions)
                        .map(|(weight, prediction)| weight * prediction[row])
                        .sum()
                })
                .collect(),
        };
        Ok(values)
    }
}

fn validate_component_track(
    track: LicenseTrackV1,
    components: &[FrozenPredictorComponentV1],
) -> Result<(), String> {
    let contains_research = components
        .iter()
        .any(|component| component.license_track() == LicenseTrackV1::ResearchOnly);
    if track == LicenseTrackV1::ProductionCompatible && contains_research {
        return Err(
            "research-only component cannot enter a production-compatible predictor".into(),
        );
    }
    if track == LicenseTrackV1::ResearchOnly && !contains_research {
        return Err(
            "research-only predictor must bind at least one research-only component".into(),
        );
    }
    Ok(())
}
