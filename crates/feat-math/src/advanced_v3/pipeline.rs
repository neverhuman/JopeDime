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
pub fn fit_legacy_numeric_bound(
    genome: &GenomeIr,
    x: &[Vec<f64>],
    y: &[f64],
) -> Result<BoundModel> {
    if x.is_empty() || x.len() != y.len() {
        bail!("training matrix is empty or target length differs from row count");
    }
    let width = x[0].len();
    if width == 0 || x.iter().any(|row| row.len() != width) {
        bail!("legacy numeric training matrix is empty or ragged");
    }
    if x.iter().flatten().any(|value| !value.is_finite())
        || y.iter().any(|value| !value.is_finite())
    {
        bail!("legacy numeric bound models require finite inputs and targets");
    }
    let schema = DataSchema {
        columns: (0..width)
            .map(|index| super::schema::ColumnSpec::numeric(format!("c{index}")))
            .collect(),
        row_semantics: RowSemantics::Iid,
    };
    let table = RawTable {
        headers: schema
            .columns
            .iter()
            .map(|column| column.name.clone())
            .collect(),
        rows: x
            .iter()
            .map(|row| row.iter().map(|value| value.to_string()).collect())
            .collect(),
    };
    let task = TaskSpec::regression(super::types::Objective::Rmse)?;
    let config = InputConfig::compatibility();
    let (input, block) = FittedInput::fit_transform(config, &schema, &table, y, task.kind)?;
    let predictor = BoundPredictor::Legacy(crate::model::fit(genome, &block.values, y)?);
    Ok(BoundModel {
        artifact_schema: ARTIFACT_SCHEMA.to_string(),
        task_sha256: stable_hash(&task)?,
        schema_sha256: schema.sha256()?,
        vocabulary_sha256: stable_hash(&input)?,
        genome_sha256: text_hash(&genome.canonical()),
        threshold: 0.5,
        task,
        schema,
        input,
        predictor,
    })
}

/// Bind an already-fitted legacy `crate::model::Model` directly into the V3
/// `BoundModel` envelope without re-running `crate::model::fit` or formatting
/// `x` to strings. Produces a `BoundModel` byte-for-byte identical to
/// [`fit_legacy_numeric_bound`].
pub fn bound_from_retained_legacy_model(
    genome: &GenomeIr,
    x: &[Vec<f64>],
    y: &[f64],
    retained_model: crate::model::Model,
) -> Result<BoundModel> {
    if x.is_empty() || x.len() != y.len() {
        bail!("training matrix is empty or target length differs from row count");
    }
    let width = x[0].len();
    if width == 0 || x.iter().any(|row| row.len() != width) {
        bail!("legacy numeric training matrix is empty or ragged");
    }
    if x.iter().flatten().any(|value| !value.is_finite())
        || y.iter().any(|value| !value.is_finite())
    {
        bail!("legacy numeric bound models require finite inputs and targets");
    }
    let schema = DataSchema {
        columns: (0..width)
            .map(|index| super::schema::ColumnSpec::numeric(format!("c{index}")))
            .collect(),
        row_semantics: RowSemantics::Iid,
    };
    let task = TaskSpec::regression(super::types::Objective::Rmse)?;
    let input = FittedInput::compatibility_numeric(width);
    let predictor = BoundPredictor::Legacy(retained_model);
    Ok(BoundModel {
        artifact_schema: ARTIFACT_SCHEMA.to_string(),
        task_sha256: stable_hash(&task)?,
        schema_sha256: schema.sha256()?,
        vocabulary_sha256: stable_hash(&input)?,
        genome_sha256: text_hash(&genome.canonical()),
        threshold: 0.5,
        task,
        schema,
        input,
        predictor,
    })
}

/// When a pipeline uses only compatibility numeric input, legacy non-OOF transforms
/// within the 512-column dense cap, and a legacy regression head, its `BoundPredictor::Pipeline`
/// state is identical to the already-fitted `crate::model::Model::Pipeline`.
/// Returns `Ok(Some(BoundModel))` when that fast path applies, or `Ok(None)` when `fit_table`
/// should be executed.
pub fn try_bound_pipeline_from_retained_legacy_model(
    genome: &GenomeIr,
    task: &TaskSpec,
    schema: &DataSchema,
    retained_model: &crate::model::Model,
) -> Result<Option<BoundModel>> {
    if task.validate().is_err()
        || schema.validate().is_err()
        || validate_v3_genome(genome, task, schema).is_err()
    {
        return Ok(None);
    }
    if task.kind != TaskKind::Regression || schema.row_semantics != RowSemantics::Iid {
        return Ok(None);
    }
    if !schema
        .columns
        .iter()
        .all(|c| c.kind == super::schema::ColumnKind::Numeric)
    {
        return Ok(None);
    }
    let GenomeIr::Pipeline(dsl) = genome else {
        return Ok(None);
    };
    let parsed = split_v3_pipeline(dsl)?;
    if parsed.input.is_some() {
        return Ok(None);
    }
    for stage in parsed.transforms {
        if matches!(
            stage.op.as_str(),
            "maxabs_scale"
                | "power_transform"
                | "piecewise_linear"
                | "corr_screen"
                | "pls_project"
                | "lag"
                | "roll"
                | "spline"
                | "impute"
        ) {
            return Ok(None);
        }
        if stage.op == "target_bin_encode" && stage.params.contains_key("oof") {
            return Ok(None);
        }
    }
    let head_op = parsed.head.map_or("ridge", |s| s.op.as_str());
    if !crate::heads::is_head_op(head_op) {
        return Ok(None);
    }
    let crate::model::Model::Pipeline { target, ops, head } = retained_model else {
        return Ok(None);
    };
    if ops.len() != parsed.transforms.len() {
        return Ok(None);
    }
    let width = schema.columns.len();
    if crate::kernel::width::check_input_width(retained_model, width).is_err() {
        return Ok(None);
    }
    let input = FittedInput::compatibility_numeric(width);
    let v3_ops: Vec<V3Op> = ops.iter().cloned().map(V3Op::Legacy).collect();
    let v3_head = V3Head::Legacy((**head).clone());

    let probe_table = RawTable {
        headers: schema.columns.iter().map(|c| c.name.clone()).collect(),
        rows: vec![vec!["0.0".to_string(); width]],
    };
    let Ok(mut probe_block) = input.transform(&probe_table) else {
        return Ok(None);
    };
    let mut legacy_rows = vec![vec![0.0; width]];
    for ((stage, v3_op), leg_op) in parsed.transforms.iter().zip(v3_ops.iter()).zip(ops.iter()) {
        if crate::kernel::width::stage_name(leg_op) != stage.op.as_str() {
            return Ok(None);
        }
        if v3_op.validate_fitted_state(schema).is_err() {
            return Ok(None);
        }
        let Ok(next_block) = v3_op.apply_block(&probe_block, &probe_table) else {
            return Ok(None);
        };
        let Ok(next_legacy) = crate::ops::apply_op(leg_op, &legacy_rows) else {
            return Ok(None);
        };
        let v3_w = next_block.values.first().map_or(0, Vec::len);
        let leg_w = next_legacy.first().map_or(0, Vec::len);
        if v3_w == 0 || v3_w > crate::MAX_WIDTH || v3_w != leg_w {
            return Ok(None);
        }
        probe_block = next_block;
        legacy_rows = next_legacy;
    }
    let final_w = probe_block.values.first().map_or(0, Vec::len);
    if !crate::kernel::width::required_head_width(head).admits(final_w) {
        return Ok(None);
    }
    let Ok(v3_pred) = v3_head.predict(&probe_block.values) else {
        return Ok(None);
    };
    let Ok(leg_pred) = head.predict(&legacy_rows) else {
        return Ok(None);
    };
    if v3_pred.len() != 1 || leg_pred.len() != 1 || v3_pred[0].to_bits() != leg_pred[0].to_bits() {
        return Ok(None);
    }

    let bound = BoundModel {
        artifact_schema: ARTIFACT_SCHEMA.to_string(),
        task_sha256: stable_hash(task)?,
        schema_sha256: schema.sha256()?,
        vocabulary_sha256: stable_hash(&input)?,
        genome_sha256: text_hash(&genome.canonical()),
        threshold: 0.5,
        task: task.clone(),
        schema: schema.clone(),
        input,
        predictor: BoundPredictor::Pipeline {
            target: target.clone(),
            ops: v3_ops,
            head: v3_head,
            calibration: CalibrationModel::None,
            missing_aware: false,
        },
    };
    if bound.validate_binding().is_err() {
        return Ok(None);
    }
    Ok(Some(bound))
}

/// Static preflight check for direct `fit_table` + `emit_bound` support on an
/// all-numeric IID schema. Returns `Err((true, msg))` if `fit_table` will fail
/// closed (e.g. `generated dense width ... exceeds 512`) or `Err((false, msg))`
/// if `emit_bound` will fail closed (e.g. unsupported legacy head or transform),
/// avoiding a doomed redundant model fit while preserving exact error strings.
pub fn check_direct_bound_emission_support(
    genome: &GenomeIr,
    task: &TaskSpec,
    schema: &DataSchema,
) -> std::result::Result<(), (bool, String)> {
    if let Err(e) = validate_v3_genome(genome, task, schema) {
        return Err((true, format!("{e:#}")));
    }
    let GenomeIr::Pipeline(dsl) = genome else {
        return Err((
            false,
            "bound emission currently requires a v3 pipeline model".to_string(),
        ));
    };
    let parsed = match split_v3_pipeline(dsl) {
        Ok(p) => p,
        Err(e) => return Err((true, format!("{e:#}"))),
    };
    let input_cfg = match InputConfig::from_stage(parsed.input) {
        Ok(c) => c,
        Err(e) => return Err((true, format!("{e:#}"))),
    };
    if schema
        .columns
        .iter()
        .all(|c| c.kind == super::schema::ColumnKind::Numeric)
    {
        let mut cols = schema.columns.len();
        if parsed.input.is_some() {
            // Check if missing indicator is enabled via InputConfig serialization or stage param
            let _ = &input_cfg;
            if let Some(stage) = parsed.input {
                if matches!(str_param(stage, "missing", "indicator"), Ok("indicator")) {
                    cols *= 2;
                }
            }
        }
        let mut generated_dense = 0usize;
        let max_dense = super::feature_block::MAX_GENERATED_DENSE_WIDTH;
        for stage in parsed.transforms {
            let append = matches!(stage.params.get("append"), Some(ParamValue::Bool(true)))
                || stage.get_usize("append", 0) == 1;
            match stage.op.as_str() {
                "maxabs_scale" | "power_transform" | "standardize" | "norm_features"
                | "robust_scale" | "target_bin_encode" | "sigmoid" | "step" | "norm_row"
                | "winsor_clip" | "winsor_moments" => {
                    if append {
                        generated_dense += cols;
                        cols += cols;
                        if generated_dense > max_dense {
                            return Err((
                                true,
                                format!(
                                    "generated dense width {generated_dense} exceeds {max_dense}"
                                ),
                            ));
                        }
                    }
                }
                "corr_screen" => {
                    let req = stage.get_usize("k", 32).min(cols);
                    if append {
                        generated_dense += req;
                        cols += req;
                        if generated_dense > max_dense {
                            return Err((
                                true,
                                format!(
                                    "generated dense width {generated_dense} exceeds {max_dense}"
                                ),
                            ));
                        }
                    } else {
                        cols = req;
                    }
                }
                "select_corr" => {
                    let k = stage.get_usize("k", 32).clamp(1, 512).min(cols);
                    if append {
                        generated_dense += k;
                        cols += k;
                        if generated_dense > max_dense {
                            return Err((
                                true,
                                format!(
                                    "generated dense width {generated_dense} exceeds {max_dense}"
                                ),
                            ));
                        }
                    } else {
                        cols = k;
                    }
                }
                "piecewise_linear" => {
                    let bins = stage.get_usize("bins", 8);
                    let gen_w = cols.saturating_mul(bins);
                    if gen_w > max_dense {
                        return Err((
                            true,
                            format!("piecewise_linear derived width {gen_w} exceeds {max_dense}"),
                        ));
                    }
                    generated_dense = (if append { generated_dense } else { 0 }) + gen_w;
                    if generated_dense > max_dense {
                        return Err((
                            true,
                            format!("generated dense width {generated_dense} exceeds {max_dense}"),
                        ));
                    }
                    cols = (if append { cols } else { 0 }) + gen_w;
                }
                "spline" | "impute" | "lag" | "roll" => {
                    // Data-dependent width or table row op; defer to fit_table if needed
                }
                other => {
                    let gen_w = match other {
                        "pls_project" => stage.get_usize("k", 8),
                        "cheb" => cols * stage.get_usize("degree", 4).clamp(1, 8),
                        "poly_interact" => {
                            (cols + cols * cols.saturating_sub(1) / 2).min(crate::MAX_WIDTH)
                        }
                        "rand_proj" => stage.get_usize("k", 16).clamp(1, 256),
                        "rff_cos" | "rff_sin" | "oblique_hinge" => {
                            stage.get_usize("n", 32).clamp(1, 256)
                        }
                        "landmark_dist" | "nystrom" => stage.get_usize("k", 16).clamp(1, 128),
                        "svd_project" => stage.get_usize("k", 8).clamp(1, 64).min(cols),
                        "pair_prod" => {
                            let m = stage.get_usize("topk", 8).clamp(2, 24).min(cols);
                            m * m.saturating_sub(1) / 2
                        }
                        "pair_ratio" => {
                            let m = cols.min(32);
                            let max_p = m * m.saturating_sub(1);
                            stage.get_usize("topk", 8).clamp(1, 64).min(max_p)
                        }
                        "row_stats" => 10,
                        "row_spectral" | "row_diff_stats" => 4,
                        "fft_mag" => stage.get_usize("k", 8).clamp(1, (cols / 2).max(1)),
                        "polar" => (cols / 2) * 2,
                        "hinge" => cols * stage.get_usize("knots", 4).clamp(1, 16),
                        "quantile_bins" | "rank" | "blom" | "symlog" | "map" => cols,
                        "bin_onehot" => cols * stage.get_usize("bins", 6).clamp(2, 32),
                        "derive" => 1,
                        _ => 0,
                    };
                    if gen_w > 0 {
                        generated_dense = (if append { generated_dense } else { 0 }) + gen_w;
                        if generated_dense > max_dense {
                            return Err((
                                true,
                                format!(
                                    "generated dense width {generated_dense} exceeds {max_dense}"
                                ),
                            ));
                        }
                        cols = (if append { cols } else { 0 }) + gen_w;
                    }
                }
            }
        }
    }

    // Check head support in emit_bound_python order
    let head_op = parsed.head.map_or(
        match task.kind {
            TaskKind::Binary => "logit_ridge",
            TaskKind::Multiclass => "multiclass_ridge",
            TaskKind::Regression | TaskKind::Timeseries => "ridge",
        },
        |stage| stage.op.as_str(),
    );
    match head_op {
        "logit_ridge" | "tab_boost" | "extra_trees" | "gam_boost" | "pls" | "portfolio"
        | "ridge" => {}
        "multiclass_ridge"
        | "multiclass_ovr_logistic"
        | "multiclass_diagonal_lda"
        | "multiclass_nearest_centroid" => {
            return Err((
                false,
                "multiclass probability matrices are not supported by scalar bound emission"
                    .to_string(),
            ));
        }
        _ => {
            return Err((
                false,
                "this fitted legacy head is not supported by bound emission".to_string(),
            ));
        }
    }

    // Check transform support in emit_bound_rust order
    for stage in parsed.transforms {
        match stage.op.as_str() {
            "maxabs_scale" | "power_transform" | "piecewise_linear" | "corr_screen"
            | "pls_project" | "lag" | "roll" | "standardize" | "norm_features" | "robust_scale"
            | "target_bin_encode" | "sigmoid" | "step" | "norm_row" | "select_corr"
            | "winsor_clip" | "winsor_moments" | "impute" => {}
            _ => {
                return Err((
                    false,
                    "this legacy transform is not supported by bound emission".to_string(),
                ));
            }
        }
    }

    Ok(())
}

/// Fit a schema-bound v3 model. The leading `(input ...)` pseudo-stage is
/// removed before numerical pipeline fitting; its honest training encoding is
/// used for target-aware categorical columns.
pub fn fit_table(
    genome: &GenomeIr,
    task: &TaskSpec,
    schema: &DataSchema,
    table: &RawTable,
    y: &[f64],
) -> Result<BoundModel> {
    task.validate()?;
    schema.validate()?;
    table.validate_against(schema)?;
    if table.rows.is_empty() || table.rows.len() != y.len() {
        bail!("training table is empty or target length differs from row count");
    }
    if y.iter().any(|value| !value.is_finite()) {
        bail!("targets must be finite");
    }
    if task.kind == TaskKind::Binary {
        validate_binary_target(y, table.rows.len())?;
    } else if task.kind == TaskKind::Multiclass {
        validate_multiclass_target(y, table.rows.len())?;
    }
    validate_v3_genome(genome, task, schema)?;

    match genome {
        GenomeIr::Symbolic { .. } | GenomeIr::Portfolio(_) => {
            if !matches!(task.kind, TaskKind::Regression | TaskKind::Timeseries) {
                bail!("closed-form and explicit portfolio programs are regression-only");
            }
            let config = InputConfig::compatibility();
            let (input, block) = FittedInput::fit_transform(config, schema, table, y, task.kind)?;
            let predictor = BoundPredictor::Legacy(crate::model::fit(genome, &block.values, y)?);
            Ok(BoundModel {
                artifact_schema: ARTIFACT_SCHEMA.to_string(),
                task: task.clone(),
                schema: schema.clone(),
                task_sha256: stable_hash(task)?,
                schema_sha256: schema.sha256()?,
                vocabulary_sha256: stable_hash(&input)?,
                genome_sha256: text_hash(&genome.canonical()),
                threshold: 0.5,
                input,
                predictor,
            })
        }
        GenomeIr::Pipeline(dsl) => {
            let parsed = split_v3_pipeline(dsl)?;
            let input_config = InputConfig::from_stage(parsed.input)?;
            let (input, block) =
                FittedInput::fit_transform(input_config, schema, table, y, task.kind)?;
            let target = match parsed.target {
                Some(stage) => Some(fit_target(stage, y)?),
                None => None,
            };
            let transformed_y: Vec<f64>;
            let y_effective = match &target {
                Some(target) => {
                    transformed_y = target.forward_all(y);
                    transformed_y.as_slice()
                }
                None => y,
            };
            let mut current = block;
            let mut ops = Vec::with_capacity(parsed.transforms.len());
            for stage in parsed.transforms {
                let (op, transformed) = fit_v3_op(stage, &current, y_effective, table, schema)?;
                current = transformed;
                if current.values.first().is_none_or(Vec::is_empty) {
                    bail!("{} produced an empty feature block", stage.op);
                }
                ops.push(op);
            }
            let head_stage = parsed.head.cloned().unwrap_or_else(|| {
                Stage::new(match task.kind {
                    TaskKind::Binary => "logit_ridge",
                    TaskKind::Multiclass => "multiclass_ridge",
                    TaskKind::Regression | TaskKind::Timeseries => "ridge",
                })
            });
            let missing_aware = matches!(head_stage.op.as_str(), "tab_boost" | "extra_trees")
                && parsed.transforms.is_empty();
            if missing_aware {
                restore_missing_values(&mut current);
            }
            let (calibration_predictions, calibration_requested) = if task.kind == TaskKind::Binary
            {
                let requested = str_param(&head_stage, "calibration", "none")?;
                if !matches!(requested, "none" | "platt" | "isotonic") {
                    bail!("{}: calibration must be none|platt|isotonic", head_stage.op);
                }
                let need_oof = requested != "none"
                    || matches!(str_param(&head_stage, "threshold", "0.5")?, "f1" | "youden");
                let predictions = if need_oof {
                    oof_head_predictions(&head_stage, &current.values, y_effective, task.kind, 5)?
                } else {
                    Vec::new()
                };
                (predictions, requested)
            } else {
                (Vec::new(), "none")
            };
            let head = fit_v3_head(&head_stage, &current.values, y_effective, task.kind)?;
            let calibration = if task.kind == TaskKind::Binary {
                fit_calibration(calibration_requested, y, &calibration_predictions)
            } else {
                CalibrationModel::None
            };
            let threshold = if task.kind == TaskKind::Binary {
                let selection = str_param(&head_stage, "threshold", "0.5")?;
                let calibrated = calibration.apply(calibration_predictions);
                match selection {
                    "0.5" | "fixed" => 0.5,
                    "f1" => crate::metrics::best_f1_threshold(y, &calibrated).0,
                    "youden" => crate::metrics::youden_threshold(y, &calibrated).0,
                    other => bail!(
                        "{}: threshold must be 0.5|fixed|f1|youden, got '{other}'",
                        head_stage.op
                    ),
                }
            } else {
                0.5
            };
            Ok(BoundModel {
                artifact_schema: ARTIFACT_SCHEMA.to_string(),
                task: task.clone(),
                schema: schema.clone(),
                task_sha256: stable_hash(task)?,
                schema_sha256: schema.sha256()?,
                vocabulary_sha256: stable_hash(&input)?,
                genome_sha256: text_hash(&genome.canonical()),
                threshold,
                input,
                predictor: BoundPredictor::Pipeline {
                    target,
                    ops,
                    head,
                    calibration,
                    missing_aware,
                },
            })
        }
    }
}

fn restore_missing_values(block: &mut FeatureBlock) {
    for (row, mask) in block.values.iter_mut().zip(&block.missing_masks) {
        for (value, missing) in row.iter_mut().zip(mask) {
            if *missing {
                *value = f64::NAN;
            }
        }
    }
}

struct ParsedPipeline<'a> {
    target: Option<&'a Stage>,
    input: Option<&'a Stage>,
    transforms: &'a [Stage],
    head: Option<&'a Stage>,
}

fn is_v3_head(op: &str) -> bool {
    crate::heads::is_head_op(op)
        || matches!(
            op,
            "logit_ridge"
                | "multiclass_ridge"
                | "multiclass_ovr_logistic"
                | "multiclass_diagonal_lda"
                | "multiclass_nearest_centroid"
                | "tab_boost"
                | "extra_trees"
                | "gam_boost"
                | "pls"
        )
}

fn split_v3_pipeline(dsl: &DslExpr) -> Result<ParsedPipeline<'_>> {
    let mut cursor = 0usize;
    let target = if dsl
        .stages
        .get(cursor)
        .is_some_and(|stage| is_target_op(&stage.op))
    {
        cursor += 1;
        Some(&dsl.stages[0])
    } else {
        None
    };
    let input = if dsl
        .stages
        .get(cursor)
        .is_some_and(|stage| stage.op == "input")
    {
        let stage = &dsl.stages[cursor];
        cursor += 1;
        Some(stage)
    } else {
        None
    };
    if dsl.stages[cursor..]
        .iter()
        .any(|stage| stage.op == "input" || is_target_op(&stage.op))
    {
        bail!("target/input stages must lead the pipeline in target-then-input order");
    }
    let end = dsl.stages.len();
    let head = dsl.stages.last().filter(|stage| is_v3_head(&stage.op));
    let transform_end = if head.is_some() { end - 1 } else { end };
    Ok(ParsedPipeline {
        target,
        input,
        transforms: &dsl.stages[cursor..transform_end],
        head,
    })
}

pub fn validate_v3_genome(genome: &GenomeIr, task: &TaskSpec, schema: &DataSchema) -> Result<()> {
    task.validate()?;
    schema.validate()?;
    let GenomeIr::Pipeline(dsl) = genome else {
        bail!("schema-bound advanced-v3 programs must be pipelines");
    };
    for stage in &dsl.stages {
        for (key, value) in &stage.params {
            if matches!(value, ParamValue::Float(number) if !number.is_finite()) {
                bail!("{}: parameter '{key}' must be finite", stage.op);
            }
        }
    }
    let parsed = split_v3_pipeline(dsl)?;
    if matches!(task.kind, TaskKind::Binary | TaskKind::Multiclass) && parsed.target.is_some() {
        bail!("target transforms are regression-only");
    }
    if matches!(schema.row_semantics, RowSemantics::Iid)
        && parsed
            .transforms
            .iter()
            .any(|stage| matches!(stage.op.as_str(), "lag" | "roll"))
    {
        bail!("lag/roll require explicit ordered row semantics");
    }
    if let Some(input) = parsed.input {
        InputConfig::from_stage(Some(input))?;
    }
    for stage in parsed.transforms {
        if is_v3_head(&stage.op) {
            bail!(
                "{}: model heads are only valid as the final stage",
                stage.op
            );
        }
        match stage.op.as_str() {
            "maxabs_scale" | "power_transform" | "piecewise_linear" | "corr_screen"
            | "pls_project" => {}
            _ => {
                // Reuse the frozen planner as the exhaustive legacy operation
                // allow-list without changing legacy fitting or canonicalization.
                let probe = GenomeIr::Pipeline(DslExpr {
                    stages: vec![stage.clone(), Stage::new("ridge")],
                });
                if let Err(problems) = crate::transpile::supports(&probe) {
                    bail!(
                        "unsupported v3 transform '{}': {}",
                        stage.op,
                        problems.join("; ")
                    );
                }
            }
        }
    }
    let head_name = parsed.head.map_or(
        match task.kind {
            TaskKind::Binary => "logit_ridge",
            TaskKind::Multiclass => "multiclass_ridge",
            TaskKind::Regression | TaskKind::Timeseries => "ridge",
        },
        |stage| stage.op.as_str(),
    );
    match task.kind {
        TaskKind::Regression | TaskKind::Timeseries if head_name == "logit_ridge" => {
            bail!("logit_ridge is only valid for binary tasks")
        }
        TaskKind::Regression | TaskKind::Timeseries
            if !crate::heads::is_head_op(head_name)
                && !matches!(head_name, "tab_boost" | "extra_trees" | "gam_boost" | "pls") =>
        {
            bail!("head '{head_name}' is unsupported in the schema-bound grammar")
        }
        TaskKind::Binary
            if !matches!(
                head_name,
                "logit_ridge"
                    | "logistic"
                    | "tab_boost"
                    | "extra_trees"
                    | "gam_boost"
                    | "portfolio"
            ) =>
        {
            bail!("head '{head_name}' is incompatible with binary tasks")
        }
        TaskKind::Binary if head_name == "pls" => bail!("pls is regression-only"),
        TaskKind::Multiclass
            if !matches!(
                head_name,
                "multiclass_ridge"
                    | "multiclass_ovr_logistic"
                    | "multiclass_diagonal_lda"
                    | "multiclass_nearest_centroid"
            ) =>
        {
            bail!("head '{head_name}' is incompatible with multiclass tasks")
        }
        _ => {}
    }
    Ok(())
}

fn oof_head_predictions(
    stage: &Stage,
    x: &Rows,
    y: &[f64],
    task: TaskKind,
    folds: usize,
) -> Result<Vec<f64>> {
    let assignments = stratified_fold_assignments(y, folds);
    let mut predictions = vec![0.5; x.len()];
    for fold in 0..folds {
        let fit_rows: Vec<usize> = (0..x.len()).filter(|&r| assignments[r] != fold).collect();
        let valid_rows: Vec<usize> = (0..x.len()).filter(|&r| assignments[r] == fold).collect();
        if fit_rows.is_empty() || valid_rows.is_empty() {
            continue;
        }
        let x_fit: Rows = fit_rows.iter().map(|&r| x[r].clone()).collect();
        let y_fit: Vec<f64> = fit_rows.iter().map(|&r| y[r]).collect();
        let x_valid: Rows = valid_rows.iter().map(|&r| x[r].clone()).collect();
        let model = fit_v3_head(stage, &x_fit, &y_fit, task)?;
        for (&row, value) in valid_rows.iter().zip(model.predict(&x_valid)?) {
            predictions[row] = value;
        }
    }
    Ok(predictions)
}

#[cfg(test)]
mod h4_retained_shortcut_tests {
    use super::*;
    use crate::advanced_v3::schema::ColumnSpec;
    use crate::advanced_v3::types::Objective;

    fn numeric_schema(width: usize) -> DataSchema {
        DataSchema {
            columns: (0..width)
                .map(|idx| ColumnSpec::numeric(format!("c{idx}")))
                .collect(),
            row_semantics: RowSemantics::Iid,
        }
    }

    fn rows_to_table(x: &[Vec<f64>]) -> RawTable {
        let width = x[0].len();
        let headers = (0..width).map(|idx| format!("c{idx}")).collect();
        let rows = x
            .iter()
            .map(|row| row.iter().map(|v| format!("{v:?}")).collect())
            .collect();
        RawTable::new(headers, rows).unwrap()
    }

    #[test]
    fn retained_legacy_shortcut_declines_v3_overwidth_hinge_and_mismatched_state() {
        let genome = GenomeIr::parse("(pipe X (hinge knots=4 append=true) (ridge))").unwrap();
        let task = TaskSpec::regression(Objective::Rmse).unwrap();

        // 200 columns: legacy caps at 512 total columns, whereas V3 generates 200*4 = 800 > 512.
        let x200: Vec<Vec<f64>> = (0..16)
            .map(|r| {
                (0..200)
                    .map(|c| ((r * 17 + c * 13) as f64 * 0.07).sin())
                    .collect()
            })
            .collect();
        let y: Vec<f64> = (0..16).map(|r| (r as f64 * 0.19).cos()).collect();
        let retained_200 = crate::model::fit(&genome, &x200, &y).unwrap();
        let schema_200 = numeric_schema(200);
        assert!(
            try_bound_pipeline_from_retained_legacy_model(
                &genome,
                &task,
                &schema_200,
                &retained_200
            )
            .unwrap()
            .is_none(),
            "200-column hinge (800 generated dense > 512) must be declined by retained shortcut"
        );

        // 120 columns: generated_dense = 480 <= 512, but V3 total width = 600 > 512 while legacy capped at 512.
        let x120: Vec<Vec<f64>> = x200.iter().map(|row| row[..120].to_vec()).collect();
        let retained_120 = crate::model::fit(&genome, &x120, &y).unwrap();
        let schema_120 = numeric_schema(120);
        assert!(
            try_bound_pipeline_from_retained_legacy_model(
                &genome,
                &task,
                &schema_120,
                &retained_120
            )
            .unwrap()
            .is_none(),
            "120-column hinge (600 total > 512 legacy cap) must be declined by retained shortcut"
        );

        // Width-mismatched retained model (fitted on 4 columns, queried on 8-column schema).
        let x4: Vec<Vec<f64>> = x200.iter().map(|row| row[..4].to_vec()).collect();
        let retained_4 = crate::model::fit(&genome, &x4, &y).unwrap();
        let schema_8 = numeric_schema(8);
        assert!(
            try_bound_pipeline_from_retained_legacy_model(&genome, &task, &schema_8, &retained_4)
                .unwrap()
                .is_none(),
            "width-mismatched retained model must be declined by retained shortcut"
        );

        // No-transform pipeline `(pipe X (ridge))` fitted on 4 columns, queried on 8-column schema (H4-R1-WIDTH).
        let ridge_only_genome = GenomeIr::parse("(pipe X (ridge))").unwrap();
        let retained_ridge_4 = crate::model::fit(&ridge_only_genome, &x4, &y).unwrap();
        assert!(
            try_bound_pipeline_from_retained_legacy_model(
                &ridge_only_genome,
                &task,
                &schema_8,
                &retained_ridge_4
            )
            .unwrap()
            .is_none(),
            "no-transform retained ridge fitted on 4 columns must be declined on 8-column schema"
        );
        // Matching 4-column schema is accepted and matches fit_table bit-for-bit.
        let schema_4 = numeric_schema(4);
        let train_table_4 = rows_to_table(&x4);
        let shortcut_ridge_4 = try_bound_pipeline_from_retained_legacy_model(
            &ridge_only_genome,
            &task,
            &schema_4,
            &retained_ridge_4,
        )
        .unwrap()
        .expect("matching 4-column no-transform ridge should be accepted");
        let full_ridge_4 =
            fit_table(&ridge_only_genome, &task, &schema_4, &train_table_4, &y).unwrap();
        assert_eq!(
            serde_json::to_string(&shortcut_ridge_4).unwrap(),
            serde_json::to_string(&full_ridge_4).unwrap()
        );
    }

    #[test]
    fn retained_legacy_shortcut_matches_fit_table_bit_for_bit_when_eligible() {
        let genome = GenomeIr::parse(
            "(pipe X (standardize append=true) (hinge knots=3 append=true) (ridge))",
        )
        .unwrap();
        let task = TaskSpec::regression(Objective::Rmse).unwrap();
        let x_train: Vec<Vec<f64>> = (0..24)
            .map(|r| {
                (0..8)
                    .map(|c| ((r * 11 + c * 7) as f64 * 0.13).sin())
                    .collect()
            })
            .collect();
        let y_train: Vec<f64> = (0..24)
            .map(|r| x_train[r][0] * 0.6 - x_train[r][1] * 0.3 + (r as f64 * 0.05).cos())
            .collect();
        let x_hold: Vec<Vec<f64>> = (0..12)
            .map(|r| {
                (0..8)
                    .map(|c| ((r * 19 + c * 5) as f64 * 0.11).cos())
                    .collect()
            })
            .collect();

        let schema = numeric_schema(8);
        let train_table = rows_to_table(&x_train);
        let hold_table = rows_to_table(&x_hold);

        let retained = crate::model::fit(&genome, &x_train, &y_train).unwrap();
        let shortcut_bound =
            try_bound_pipeline_from_retained_legacy_model(&genome, &task, &schema, &retained)
                .unwrap()
                .expect("eligible within-512 pipeline should use retained shortcut");
        let full_bound = fit_table(&genome, &task, &schema, &train_table, &y_train).unwrap();

        assert_eq!(
            serde_json::to_string(&shortcut_bound).unwrap(),
            serde_json::to_string(&full_bound).unwrap()
        );
        let shortcut_preds: Vec<u64> = shortcut_bound
            .predict_table(&hold_table)
            .unwrap()
            .into_iter()
            .map(f64::to_bits)
            .collect();
        let full_preds: Vec<u64> = full_bound
            .predict_table(&hold_table)
            .unwrap()
            .into_iter()
            .map(f64::to_bits)
            .collect();
        assert_eq!(shortcut_preds, full_preds);
    }
}
