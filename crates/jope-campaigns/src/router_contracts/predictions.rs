use super::{canonical_sha256, validate_sha256, ROUTER_SEEDS_V2};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

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
