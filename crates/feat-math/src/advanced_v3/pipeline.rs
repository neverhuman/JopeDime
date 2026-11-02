use super::calibration::{fit_calibration, CalibrationModel};
use super::csv::RawTable;
use super::feature_block::FeatureBlock;
use super::heads::{
    finite, fit_v3_head, validate_binary_target, validate_multiclass_target, V3Head,
};
use super::input::{str_param, FittedInput, InputConfig};
use super::schema::{DataSchema, RowSemantics};
use super::transforms::{fit_v3_op, V3Op};
use super::types::{
    stable_hash, stratified_fold_assignments, text_hash, TaskKind, TaskSpec, ARTIFACT_SCHEMA,
};
use crate::dsl::{DslExpr, ParamValue, Stage};
use crate::target_transform::{fit_target, is_target_op, FittedTarget};
use crate::{GenomeIr, Rows};
use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) enum BoundPredictor {
    Legacy(crate::model::Model),
    Pipeline {
        target: Option<FittedTarget>,
        ops: Vec<V3Op>,
        head: V3Head,
        calibration: CalibrationModel,
        missing_aware: bool,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BoundModel {
    pub artifact_schema: String,
    pub task: TaskSpec,
    pub schema: DataSchema,
    pub task_sha256: String,
    pub schema_sha256: String,
    pub vocabulary_sha256: String,
    pub genome_sha256: String,
    pub threshold: f64,
    pub(super) input: FittedInput,
    pub(super) predictor: BoundPredictor,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct MulticlassPrediction {
    pub classes: Vec<i64>,
    pub labels: Vec<i64>,
    pub probabilities: Vec<Vec<f64>>,
}

impl BoundModel {
    /// Verify every identity carried by a fitted artifact before prediction or
    /// export.  In particular, the vocabulary digest is derived from fitted
    /// input state; trusting the stored string would allow a mutated category
    /// table to retain a valid-looking artifact envelope.
    pub fn validate_binding(&self) -> Result<()> {
        if self.artifact_schema != ARTIFACT_SCHEMA {
            bail!(
                "unsupported bound artifact schema '{}'; expected {ARTIFACT_SCHEMA}",
                self.artifact_schema
            );
        }
        self.task.validate()?;
        self.schema.validate()?;
        if self.schema.sha256()? != self.schema_sha256 {
            bail!("bound model schema hash mismatch");
        }
        if stable_hash(&self.task)? != self.task_sha256 {
            bail!("bound model task hash mismatch");
        }
        if stable_hash(&self.input)? != self.vocabulary_sha256 {
            bail!("bound model vocabulary hash mismatch");
        }
        if !self.threshold.is_finite() {
            bail!("bound model threshold must be finite");
        }
        if let BoundPredictor::Pipeline { ops, .. } = &self.predictor {
            for op in ops {
                op.validate_fitted_state(&self.schema)?;
            }
        }
        Ok(())
    }

    pub fn predict_table(&self, table: &RawTable) -> Result<Vec<f64>> {
        table.validate_against(&self.schema)?;
        self.validate_binding()?;
        let mut block = self.input.transform(table)?;
        let mut predictions = match &self.predictor {
            BoundPredictor::Legacy(model) => model.predict(&block.values)?,
            BoundPredictor::Pipeline {
                target,
                ops,
                head,
                calibration,
                missing_aware,
            } => {
                if *missing_aware {
                    restore_missing_values(&mut block);
                }
                for op in ops {
                    block = op.apply_block(&block, table)?;
                }
                let predictions = calibration.apply(head.predict(&block.values)?);
                match target {
                    Some(target) => predictions
                        .iter()
                        .map(|value| target.final_clamp(target.inverse(*value)))
                        .collect(),
                    None => predictions,
                }
            }
        };
        match self.task.kind {
            TaskKind::Regression | TaskKind::Timeseries => {
                for value in &mut predictions {
                    *value = finite(*value);
                }
            }
            TaskKind::Binary => {
                for value in &mut predictions {
                    *value = if value.is_finite() {
                        value.clamp(0.0, 1.0)
                    } else {
                        0.5
                    };
                }
            }
            TaskKind::Multiclass => {
                for value in &mut predictions {
                    if !value.is_finite() || value.fract() != 0.0 {
                        bail!("multiclass head returned a non-integer class label");
                    }
                }
            }
        }
        Ok(predictions)
    }

    pub fn predict_multiclass_table(&self, table: &RawTable) -> Result<MulticlassPrediction> {
        if self.task.kind != TaskKind::Multiclass {
            bail!("multiclass prediction API requires a multiclass bound model");
        }
        table.validate_against(&self.schema)?;
        self.validate_binding()?;
        let mut block = self.input.transform(table)?;
        let BoundPredictor::Pipeline {
            target,
            ops,
            head,
            missing_aware,
            ..
        } = &self.predictor
        else {
            bail!("multiclass prediction requires a schema-bound pipeline");
        };
        if target.is_some() {
            bail!("multiclass bound models cannot carry target transforms");
        }
        if *missing_aware {
            restore_missing_values(&mut block);
        }
        for op in ops {
            block = op.apply_block(&block, table)?;
        }
        let (classes, probabilities) = head
            .multiclass_probabilities(&block.values)
            .ok_or_else(|| anyhow::anyhow!("bound model does not carry a multiclass head"))?;
        if probabilities.iter().any(|row| {
            row.len() != classes.len()
                || row
                    .iter()
                    .any(|value| !value.is_finite() || *value < 0.0 || *value > 1.0)
                || (row.iter().sum::<f64>() - 1.0).abs() > 1e-9
        }) {
            bail!("multiclass head returned invalid aligned probabilities");
        }
        let labels = probabilities
            .iter()
            .map(|row| {
                let index = row
                    .iter()
                    .enumerate()
                    .max_by(|left, right| left.1.total_cmp(right.1))
                    .map_or(0, |(index, _)| index);
                classes[index]
            })
            .collect();
        Ok(MulticlassPrediction {
            classes,
            labels,
            probabilities,
        })
    }

    pub fn predict_probabilities_table(
        &self,
        table: &RawTable,
    ) -> Result<(Vec<i64>, Vec<Vec<f64>>)> {
        let prediction = self.predict_multiclass_table(table)?;
        Ok((prediction.classes, prediction.probabilities))
    }

    pub fn predict_labels_table(&self, table: &RawTable) -> Result<Vec<i64>> {
        Ok(self.predict_multiclass_table(table)?.labels)
    }

    /// Fast-path numeric matrix prediction for compatibility-bound legacy models
    /// when the caller already holds validated finite `f64` rows (`&[Vec<f64>]`),
    /// avoiding `f64 -> String -> f64` table formatting and parsing while preserving
    /// full binding validation and `0 ulp` prediction identity.
    pub fn predict_numeric_rows(&self, x: &[Vec<f64>]) -> Result<Vec<f64>> {
        self.validate_binding()?;
        if x.is_empty() {
            bail!("table has no data rows");
        }
        let width = self.schema.columns.len();
        if x.iter().any(|row| row.len() != width) {
            bail!("numeric matrix column width does not match bound schema");
        }
        let values = self.input.transform_numeric_rows(x)?;
        let mut predictions = match &self.predictor {
            BoundPredictor::Legacy(model) => model.predict(&values)?,
            _ => bail!("predict_numeric_rows requires a legacy bound predictor"),
        };
        match self.task.kind {
            TaskKind::Regression | TaskKind::Timeseries => {
                for value in &mut predictions {
                    *value = finite(*value);
                }
            }
            TaskKind::Binary => {
                for value in &mut predictions {
                    *value = if value.is_finite() {
                        value.clamp(0.0, 1.0)
                    } else {
                        0.5
                    };
                }
            }
            TaskKind::Multiclass => {
                for value in &mut predictions {
                    if !value.is_finite() || value.fract() != 0.0 {
                        bail!("multiclass head returned a non-integer class label");
                    }
                }
            }
        }
        Ok(predictions)
    }
}

pub fn predict_table(model: &BoundModel, table: &RawTable) -> Result<Vec<f64>> {
    model.predict_table(table)
}

pub fn predict_multiclass_table(
    model: &BoundModel,
    table: &RawTable,
) -> Result<MulticlassPrediction> {
    model.predict_multiclass_table(table)
}

pub fn predict_probabilities_table(
    model: &BoundModel,
    table: &RawTable,
) -> Result<(Vec<i64>, Vec<Vec<f64>>)> {
    model.predict_probabilities_table(table)
}

pub fn predict_labels_table(model: &BoundModel, table: &RawTable) -> Result<Vec<i64>> {
    model.predict_labels_table(table)
}

/// Bind the frozen numeric legacy fitter to the V3 artifact envelope.
///
/// This is a compatibility boundary, not a second implementation of the
/// legacy pipeline.  The input is encoded with the historical zero-filled
/// numeric layout and the fitted predictor is produced by `model::fit`
/// unchanged.  New schema-aware pipelines continue to use [`fit_table`].
mod emission;
mod fit;
mod legacy;

pub use emission::check_direct_bound_emission_support;
pub use fit::{fit_table, validate_v3_genome};
use fit::{restore_missing_values, split_v3_pipeline};
pub use legacy::{
    bound_from_retained_legacy_model, fit_legacy_numeric_bound,
    try_bound_pipeline_from_retained_legacy_model,
};

#[cfg(test)]
mod h4_retained_shortcut_tests;
