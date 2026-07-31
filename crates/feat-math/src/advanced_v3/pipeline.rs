use super::calibration::{fit_calibration, CalibrationModel};
use super::csv::RawTable;
use super::feature_block::FeatureBlock;
use super::heads::{finite, fit_v3_head, validate_binary_target, V3Head};
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
            BoundPredictor::Legacy(model) => model.predict(&block.values),
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
                let predictions = calibration.apply(head.predict(&block.values));
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
        }
        Ok(predictions)
    }
}

pub fn predict_table(model: &BoundModel, table: &RawTable) -> Result<Vec<f64>> {
    model.predict_table(table)
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
    }
    validate_v3_genome(genome, task, schema)?;

    match genome {
        GenomeIr::Symbolic { .. } | GenomeIr::Portfolio(_) => {
            if !matches!(task.kind, TaskKind::Regression | TaskKind::Timeseries) {
                bail!("symbolic and explicit portfolio genomes are regression-only");
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
                Stage::new(if task.kind == TaskKind::Binary {
                    "logit_ridge"
                } else {
                    "ridge"
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
            "logit_ridge" | "tab_boost" | "extra_trees" | "gam_boost" | "pls"
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
        bail!("schema-bound advanced-v3 genomes must be pipelines");
    };
    for stage in &dsl.stages {
        for (key, value) in &stage.params {
            if matches!(value, ParamValue::Float(number) if !number.is_finite()) {
                bail!("{}: parameter '{key}' must be finite", stage.op);
            }
        }
    }
    let parsed = split_v3_pipeline(dsl)?;
    if task.kind == TaskKind::Binary && parsed.target.is_some() {
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
        if task.kind == TaskKind::Binary {
            "logit_ridge"
        } else {
            "ridge"
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
        for (&row, value) in valid_rows.iter().zip(model.predict(&x_valid)) {
            predictions[row] = value;
        }
    }
    Ok(predictions)
}
