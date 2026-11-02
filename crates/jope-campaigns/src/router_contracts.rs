//! Public task-neutral contracts for Jain's per-family V2 router.
//!
//! These types contain no protected search implementation. They describe the
//! feature view, member route, validation boundary, and opaque receipts that
//! can be shared by Core, CLI, and a cloud client.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const ROUTER_SEEDS_V2: [u64; 2] = [303, 404];
pub const ROUTER_FOLDS_V2: usize = 7;

fn canonical_sha256<T: Serialize>(value: &T) -> Result<String, String> {
    let bytes = serde_json::to_vec(value).map_err(|error| format!("canonical JSON: {error}"))?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn validate_sha256(name: &str, value: &str) -> Result<(), String> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(format!("{name} is not a lowercase SHA-256 digest"));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeatureViewV2 {
    Raw,
    HyperionCellOnly,
    HyperionCellAppend,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum PredictorRouteV2 {
    FeatureView { view: FeatureViewV2 },
    RawCellBlend { alpha: f64 },
}

impl PredictorRouteV2 {
    pub fn raw() -> Self {
        Self::FeatureView {
            view: FeatureViewV2::Raw,
        }
    }

    pub fn cell_only() -> Self {
        Self::FeatureView {
            view: FeatureViewV2::HyperionCellOnly,
        }
    }

    pub fn append() -> Self {
        Self::FeatureView {
            view: FeatureViewV2::HyperionCellAppend,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        match self {
            Self::FeatureView { .. } => Ok(()),
            Self::RawCellBlend { alpha } if alpha.is_finite() && (0.0..=0.5).contains(alpha) => {
                Ok(())
            }
            Self::RawCellBlend { .. } => Err("blend alpha must be finite and in [0, 0.5]".into()),
        }
    }

    pub fn is_raw(&self) -> bool {
        matches!(
            self,
            Self::FeatureView {
                view: FeatureViewV2::Raw
            }
        )
    }

    pub fn input_width(&self, raw_width: usize) -> Result<usize, String> {
        self.validate()?;
        if raw_width == 0 {
            return Err("raw feature width must be positive".into());
        }
        match self {
            Self::FeatureView {
                view: FeatureViewV2::HyperionCellAppend,
            } => raw_width
                .checked_mul(2)
                .ok_or_else(|| "append feature width overflowed".into()),
            _ => Ok(raw_width),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum ValidationSchemeV2 {
    IidGrouped,
    TimeSeries {
        time_column: String,
        seasonal_period: usize,
        purge_groups: usize,
        embargo_groups: usize,
    },
}

impl ValidationSchemeV2 {
    pub fn validate(&self) -> Result<(), String> {
        match self {
            Self::IidGrouped => Ok(()),
            Self::TimeSeries {
                time_column,
                seasonal_period,
                ..
            } if !time_column.trim().is_empty() && *seasonal_period > 0 => Ok(()),
            Self::TimeSeries { .. } => {
                Err("time-series validation needs a time column and positive seasonality".into())
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SeedRouteMetricV2 {
    pub seed: u64,
    pub raw_loss: f64,
    pub routed_loss: f64,
    pub relative_improvement: f64,
    pub fold_wins: u32,
    pub valid: bool,
}

impl SeedRouteMetricV2 {
    fn validate(&self) -> Result<(), String> {
        if !ROUTER_SEEDS_V2.contains(&self.seed)
            || !self.raw_loss.is_finite()
            || !self.routed_loss.is_finite()
            || self.raw_loss < 0.0
            || self.routed_loss < 0.0
            || !self.relative_improvement.is_finite()
            || self.fold_wins > ROUTER_FOLDS_V2 as u32
        {
            return Err("seed route metric is invalid".into());
        }
        let expected = if self.raw_loss > 0.0 {
            (self.raw_loss - self.routed_loss) / self.raw_loss
        } else {
            0.0
        };
        if (self.relative_improvement - expected).abs() > 1e-12 {
            return Err("seed route improvement does not match its losses".into());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FamilyRouteDecisionV2 {
    pub schema: String,
    pub family_id: String,
    pub selected_route: PredictorRouteV2,
    pub matched_raw_fallback: PredictorRouteV2,
    pub per_seed_metrics: Vec<SeedRouteMetricV2>,
    pub blend_coefficient: Option<f64>,
    pub latency_ms: f64,
    pub valid: bool,
    pub receipt_sha256: String,
}

impl FamilyRouteDecisionV2 {
    pub const SCHEMA: &'static str = "JainFamilyRouteDecision.v2";

    pub fn validate(&self) -> Result<(), String> {
        if self.schema != Self::SCHEMA
            || self.family_id.trim().is_empty()
            || !self.latency_ms.is_finite()
            || self.latency_ms < 0.0
            || self.matched_raw_fallback != PredictorRouteV2::raw()
        {
            return Err("family route decision is malformed".into());
        }
        validate_sha256("family receipt", &self.receipt_sha256)?;
        self.selected_route.validate()?;
        if !self.valid && !self.selected_route.is_raw() {
            return Err("invalid family decisions must select raw".into());
        }
        match (&self.selected_route, self.blend_coefficient) {
            (PredictorRouteV2::RawCellBlend { alpha }, Some(coefficient))
                if (*alpha - coefficient).abs() <= 1e-12 => {}
            (PredictorRouteV2::RawCellBlend { .. }, _) => {
                return Err("blend decision has no matching coefficient".into());
            }
            (_, None) => {}
            (_, Some(_)) => return Err("nonblend decision carries a coefficient".into()),
        }
        if self.per_seed_metrics.len() != ROUTER_SEEDS_V2.len() {
            return Err("family decision does not contain both router seeds".into());
        }
        for (index, metric) in self.per_seed_metrics.iter().enumerate() {
            metric.validate()?;
            if metric.seed != ROUTER_SEEDS_V2[index] {
                return Err("seed metrics are not in canonical order".into());
            }
        }
        Ok(())
    }

    pub fn identity(&self) -> Result<String, String> {
        self.validate()?;
        canonical_sha256(self)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MemberInputPlanV2 {
    pub schema: String,
    pub member_id: String,
    pub family_id: String,
    pub route: PredictorRouteV2,
    pub raw_width: usize,
    pub predictor_sha256: String,
}

impl MemberInputPlanV2 {
    pub const SCHEMA: &'static str = "JainMemberInputPlan.v2";

    pub fn validate(&self) -> Result<(), String> {
        if self.schema != Self::SCHEMA
            || self.member_id.trim().is_empty()
            || self.family_id.trim().is_empty()
            || self.raw_width == 0
        {
            return Err("member input plan identity or width is invalid".into());
        }
        self.route.validate()?;
        validate_sha256("member predictor", &self.predictor_sha256)
    }

    pub fn input_width(&self) -> Result<usize, String> {
        self.validate()?;
        self.route.input_width(self.raw_width)
    }
}

pub fn validate_family_ensemble_v2(
    decisions: &[FamilyRouteDecisionV2],
    members: &[MemberInputPlanV2],
) -> Result<(), String> {
    let mut selected = BTreeMap::new();
    for decision in decisions {
        decision.validate()?;
        if selected
            .insert(decision.family_id.as_str(), &decision.selected_route)
            .is_some()
        {
            return Err("family decisions contain a duplicate family".into());
        }
    }
    let mut member_ids = BTreeSet::new();
    for member in members {
        member.validate()?;
        if !member_ids.insert(member.member_id.as_str()) {
            return Err("ensemble member IDs are duplicated".into());
        }
        let route = selected
            .get(member.family_id.as_str())
            .ok_or_else(|| format!("member family {} has no route", member.family_id))?;
        if *route != &member.route {
            return Err(format!(
                "member {} does not use its family route",
                member.member_id
            ));
        }
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PredictionBatchV2 {
    pub row_count: usize,
    pub output_width: usize,
    pub row_major_values: Vec<f64>,
    pub class_label_names: Vec<String>,
}

impl PredictionBatchV2 {
    pub const SCHEMA: &'static str = "JainPredictionBatch.v2";

    pub fn new(
        row_count: usize,
        output_width: usize,
        row_major_values: Vec<f64>,
        class_label_names: Vec<String>,
    ) -> Result<Self, String> {
        let value = Self {
            row_count,
            output_width,
            row_major_values,
            class_label_names,
        };
        value.validate()?;
        Ok(value)
    }

    pub fn scalar(values: Vec<f64>) -> Result<Self, String> {
        Self::new(values.len(), 1, values, Vec::new())
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.row_count == 0
            || self.output_width == 0
            || self.row_count.checked_mul(self.output_width) != Some(self.row_major_values.len())
            || self.row_major_values.iter().any(|value| !value.is_finite())
        {
            return Err("prediction batch shape or values are invalid".into());
        }
        let labels = self
            .class_label_names
            .iter()
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        if labels.len() != self.class_label_names.len()
            || self
                .class_label_names
                .iter()
                .any(|value| value.trim().is_empty())
            || (!self.class_label_names.is_empty()
                && self.class_label_names.len() != self.output_width)
        {
            return Err("prediction class/label order is invalid".into());
        }
        Ok(())
    }

    pub fn row(&self, index: usize) -> Result<&[f64], String> {
        self.validate()?;
        let start = index
            .checked_mul(self.output_width)
            .ok_or_else(|| "prediction row index overflowed".to_string())?;
        self.row_major_values
            .get(start..start + self.output_width)
            .ok_or_else(|| "prediction row is out of range".into())
    }

    pub fn class_order_sha256(&self) -> Result<Option<String>, String> {
        self.validate()?;
        if self.class_label_names.is_empty() {
            return Ok(None);
        }
        Ok(Some(canonical_sha256(&self.class_label_names)?))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BlendFitV2 {
    pub alpha: f64,
    pub denominator: f64,
    pub stable: bool,
}

/// Fit the raw-biased blend from inner OOF predictions only.  A near-zero or
/// nonfinite denominator intentionally becomes raw (`alpha = 0`).
pub fn fit_raw_biased_blend_v2(
    truth: &[f64],
    raw_oof: &[f64],
    cell_oof: &[f64],
) -> Result<BlendFitV2, String> {
    if truth.is_empty() || truth.len() != raw_oof.len() || truth.len() != cell_oof.len() {
        return Err("blend vectors are empty or have different lengths".into());
    }
    if truth
        .iter()
        .chain(raw_oof)
        .chain(cell_oof)
        .any(|value| !value.is_finite())
    {
        return Err("blend vectors contain a nonfinite value".into());
    }
    let (mut numerator, mut denominator) = (0.0, 0.0);
    for ((actual, raw), cell) in truth.iter().zip(raw_oof).zip(cell_oof) {
        let direction = cell - raw;
        numerator += direction * (actual - raw);
        denominator += direction * direction;
    }
    let floor = 1e-12 * truth.len() as f64;
    if !numerator.is_finite() || !denominator.is_finite() || denominator <= floor {
        return Ok(BlendFitV2 {
            alpha: 0.0,
            denominator,
            stable: false,
        });
    }
    Ok(BlendFitV2 {
        alpha: (numerator / denominator).clamp(0.0, 0.5),
        denominator,
        stable: true,
    })
}

pub fn apply_raw_biased_blend_v2(
    raw_predictions: &[f64],
    cell_predictions: &[f64],
    fit: BlendFitV2,
) -> Result<Vec<f64>, String> {
    if raw_predictions.is_empty()
        || raw_predictions.len() != cell_predictions.len()
        || !fit.alpha.is_finite()
        || !(0.0..=0.5).contains(&fit.alpha)
        || raw_predictions
            .iter()
            .chain(cell_predictions)
            .any(|value| !value.is_finite())
    {
        return Err("blend inputs or coefficient are invalid".into());
    }
    Ok(raw_predictions
        .iter()
        .zip(cell_predictions)
        .map(|(raw, cell)| raw + fit.alpha * (cell - raw))
        .collect())
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StrategyScreenReceiptV3 {
    pub schema: String,
    pub train_sha256: String,
    pub cell_sha256: String,
    pub view_identity_sha256: String,
    pub fold_bank_sha256: String,
    pub candidate_bank_sha256: String,
    pub task: String,
    pub configuration_sha256: String,
    pub binary_sha256: String,
    pub source_sha256: String,
    pub seed: u64,
    pub timeout_policy: String,
    pub evaluation_budget: u32,
    pub test_material_present: bool,
}

impl StrategyScreenReceiptV3 {
    pub const SCHEMA: &'static str = "JainStrategyScreenReceipt.v3";

    pub fn validate(&self) -> Result<(), String> {
        if self.schema != Self::SCHEMA
            || self.task.trim().is_empty()
            || self.timeout_policy.trim().is_empty()
            || self.evaluation_budget == 0
            || !ROUTER_SEEDS_V2.contains(&self.seed)
            || self.test_material_present
        {
            return Err("strategy receipt is incomplete or crosses the test boundary".into());
        }
        for (name, value) in [
            ("train", &self.train_sha256),
            ("cell", &self.cell_sha256),
            ("view", &self.view_identity_sha256),
            ("fold bank", &self.fold_bank_sha256),
            ("candidate bank", &self.candidate_bank_sha256),
            ("configuration", &self.configuration_sha256),
            ("binary", &self.binary_sha256),
            ("source", &self.source_sha256),
        ] {
            validate_sha256(name, value)?;
        }
        Ok(())
    }

    pub fn identity(&self) -> Result<String, String> {
        self.validate()?;
        canonical_sha256(self)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ValidationFoldV2 {
    pub origin: u32,
    pub fit_rows: Vec<usize>,
    pub hold_rows: Vec<usize>,
    pub fit_groups: Vec<String>,
    pub hold_groups: Vec<String>,
}

fn fold_rows(groups: &[Vec<usize>], range: std::ops::Range<usize>) -> Vec<usize> {
    groups
        .get(range)
        .unwrap_or(&[])
        .iter()
        .flatten()
        .copied()
        .collect()
}

fn group_names(rows: &[usize], atomic_groups: &[String]) -> Result<Vec<String>, String> {
    let mut names = BTreeSet::new();
    for row in rows {
        names.insert(
            atomic_groups
                .get(*row)
                .ok_or_else(|| "fold row is out of range".to_string())?
                .clone(),
        );
    }
    Ok(names.into_iter().collect())
}

fn timestamp_groups(times: &[i64]) -> Vec<Vec<usize>> {
    let mut ordered = (0..times.len()).collect::<Vec<_>>();
    ordered.sort_by_key(|row| (times[*row], *row));
    let mut result = Vec::new();
    for row in ordered {
        if result
            .last()
            .and_then(|group: &Vec<usize>| group.first())
            .is_some_and(|first| times[*first] == times[row])
        {
            result.last_mut().expect("timestamp group exists").push(row);
        } else {
            result.push(vec![row]);
        }
    }
    result
}

pub fn build_validation_folds_v2(
    times: Option<&[i64]>,
    atomic_groups: &[String],
    scheme: &ValidationSchemeV2,
) -> Result<Vec<ValidationFoldV2>, String> {
    scheme.validate()?;
    if atomic_groups.is_empty() || atomic_groups.iter().any(|value| value.trim().is_empty()) {
        return Err("folds require nonempty atomic groups".into());
    }
    match scheme {
        ValidationSchemeV2::IidGrouped => {
            let mut by_group = BTreeMap::<String, Vec<usize>>::new();
            for (row, group) in atomic_groups.iter().enumerate() {
                by_group.entry(group.clone()).or_default().push(row);
            }
            if by_group.len() < 14 {
                return Err("IidGrouped requires at least fourteen atomic groups".into());
            }
            let mut ordered = by_group.keys().cloned().collect::<Vec<_>>();
            ordered.sort_by_key(|group| {
                let mut digest = Sha256::new();
                digest.update(format!("jain-router-v2|303|{group}").as_bytes());
                format!("{:x}", digest.finalize())
            });
            let mut hold_groups = vec![Vec::new(); ROUTER_FOLDS_V2];
            for (index, group) in ordered.into_iter().enumerate() {
                hold_groups[index % ROUTER_FOLDS_V2].push(group);
            }
            let all_rows = (0..atomic_groups.len()).collect::<BTreeSet<_>>();
            let all_groups = by_group.keys().cloned().collect::<BTreeSet<_>>();
            let mut coverage = vec![0u8; atomic_groups.len()];
            let mut folds = Vec::new();
            for (index, groups) in hold_groups.into_iter().enumerate() {
                let hold_set = groups.into_iter().collect::<BTreeSet<_>>();
                let hold_rows = hold_set
                    .iter()
                    .flat_map(|group| by_group[group].iter().copied())
                    .collect::<Vec<_>>();
                let hold_row_set = hold_rows.iter().copied().collect::<BTreeSet<_>>();
                let fit_rows = all_rows
                    .difference(&hold_row_set)
                    .copied()
                    .collect::<Vec<_>>();
                if fit_rows.len() < 32 || hold_rows.len() < 2 || hold_set.len() < 2 {
                    return Err("IID fold violates the minimum row/group guards".into());
                }
                for row in &hold_rows {
                    coverage[*row] += 1;
                }
                folds.push(ValidationFoldV2 {
                    origin: index as u32,
                    fit_rows,
                    hold_rows,
                    fit_groups: all_groups.difference(&hold_set).cloned().collect(),
                    hold_groups: hold_set.into_iter().collect(),
                });
            }
            if coverage.iter().any(|value| *value != 1) {
                return Err("IID folds do not cover each row exactly once".into());
            }
            Ok(folds)
        }
        ValidationSchemeV2::TimeSeries {
            seasonal_period,
            purge_groups,
            embargo_groups,
            ..
        } => {
            let times = times.ok_or_else(|| "time values are required".to_string())?;
            if times.len() != atomic_groups.len() {
                return Err("time and group lengths differ".into());
            }
            let groups = timestamp_groups(times);
            let mut seen = BTreeMap::<String, usize>::new();
            for (index, rows) in groups.iter().enumerate() {
                for row in rows {
                    if let Some(previous) = seen.insert(atomic_groups[*row].clone(), index) {
                        if previous != index {
                            return Err("an atomic group spans multiple timestamps".into());
                        }
                    }
                }
            }
            let minimum_fit_rows = 32usize.max(2 * seasonal_period);
            let mut fit_end = 0;
            while fit_end < groups.len() && fold_rows(&groups, 0..fit_end).len() < minimum_fit_rows
            {
                fit_end += 1;
            }
            let remaining = groups.len().saturating_sub(fit_end + purge_groups);
            let hold_size = (remaining / 7).max(1);
            let mut folds = Vec::new();
            while folds.len() < 7 {
                let hold_start = fit_end + purge_groups;
                if hold_start >= groups.len() {
                    break;
                }
                let hold_end = (hold_start + hold_size).min(groups.len());
                let fit_rows = fold_rows(&groups, 0..fit_end);
                let hold_rows = fold_rows(&groups, hold_start..hold_end);
                if fit_rows.len() >= minimum_fit_rows && !hold_rows.is_empty() {
                    let fit_groups = group_names(&fit_rows, atomic_groups)?;
                    let hold_groups = group_names(&hold_rows, atomic_groups)?;
                    if fit_groups.iter().any(|group| hold_groups.contains(group)) {
                        return Err("time-series fold leaks an atomic group".into());
                    }
                    folds.push(ValidationFoldV2 {
                        origin: folds.len() as u32,
                        fit_rows,
                        hold_rows,
                        fit_groups,
                        hold_groups,
                    });
                }
                fit_end = hold_end + embargo_groups;
            }
            if folds.len() < 3 {
                return Err(format!("only {} valid time-series origins", folds.len()));
            }
            Ok(folds)
        }
    }
}

pub fn predictor_columns_without_time(
    columns: &[String],
    time_column: &str,
) -> Result<Vec<String>, String> {
    if time_column.trim().is_empty() || !columns.iter().any(|column| column == time_column) {
        return Err("time column is absent from the schema".into());
    }
    Ok(columns
        .iter()
        .filter(|column| column.as_str() != time_column)
        .cloned()
        .collect())
}

pub fn mase_v2(
    fit_targets: &[f64],
    hold_targets: &[f64],
    predictions: &[f64],
    seasonal_period: usize,
) -> Result<f64, String> {
    if seasonal_period == 0
        || fit_targets.len() <= seasonal_period
        || hold_targets.is_empty()
        || hold_targets.len() != predictions.len()
        || fit_targets
            .iter()
            .chain(hold_targets)
            .chain(predictions)
            .any(|value| !value.is_finite())
    {
        return Err("invalid MASE inputs".into());
    }
    let denominator = (seasonal_period..fit_targets.len())
        .map(|row| (fit_targets[row] - fit_targets[row - seasonal_period]).abs())
        .sum::<f64>()
        / (fit_targets.len() - seasonal_period) as f64;
    if denominator <= 0.0 || !denominator.is_finite() {
        return Err("MASE denominator is zero or nonfinite".into());
    }
    Ok(hold_targets
        .iter()
        .zip(predictions)
        .map(|(actual, predicted)| (actual - predicted).abs())
        .sum::<f64>()
        / hold_targets.len() as f64
        / denominator)
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelArtifactV4 {
    pub schema: String,
    pub task: String,
    pub selected_route: PredictorRouteV2,
    pub output_width: usize,
    pub class_label_names: Vec<String>,
    pub predictor_sha256: String,
}

impl ModelArtifactV4 {
    pub const SCHEMA: &'static str = "JainModelArtifact.v4";

    pub fn validate(&self) -> Result<(), String> {
        if self.schema != Self::SCHEMA || self.task.trim().is_empty() || self.output_width == 0 {
            return Err("model artifact v4 is invalid".into());
        }
        self.selected_route.validate()?;
        validate_sha256("model predictor", &self.predictor_sha256)?;
        if !self.class_label_names.is_empty() && self.class_label_names.len() != self.output_width {
            return Err("model class order does not match output width".into());
        }
        Ok(())
    }

    pub fn from_legacy_v3(
        task: impl Into<String>,
        output_width: usize,
        class_label_names: Vec<String>,
        predictor_sha256: String,
    ) -> Result<Self, String> {
        let artifact = Self {
            schema: Self::SCHEMA.into(),
            task: task.into(),
            selected_route: PredictorRouteV2::raw(),
            output_width,
            class_label_names,
            predictor_sha256,
        };
        artifact.validate()?;
        Ok(artifact)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunManifestV6 {
    pub schema: String,
    pub task: String,
    pub selected_route: PredictorRouteV2,
    pub output_width: usize,
    pub class_label_names: Vec<String>,
    pub model_artifact_sha256: String,
}

impl RunManifestV6 {
    pub const SCHEMA: &'static str = "JainRunManifest.v6";

    pub fn validate(&self) -> Result<(), String> {
        if self.schema != Self::SCHEMA || self.task.trim().is_empty() || self.output_width == 0 {
            return Err("run manifest v6 is invalid".into());
        }
        self.selected_route.validate()?;
        validate_sha256("run model artifact", &self.model_artifact_sha256)?;
        if !self.class_label_names.is_empty() && self.class_label_names.len() != self.output_width {
            return Err("run manifest class order does not match output width".into());
        }
        Ok(())
    }

    pub fn from_legacy_v5(
        task: impl Into<String>,
        output_width: usize,
        class_label_names: Vec<String>,
        model_artifact_sha256: String,
    ) -> Result<Self, String> {
        let manifest = Self {
            schema: Self::SCHEMA.into(),
            task: task.into(),
            selected_route: PredictorRouteV2::raw(),
            output_width,
            class_label_names,
            model_artifact_sha256,
        };
        manifest.validate()?;
        Ok(manifest)
    }
}
