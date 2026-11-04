//! Local runtime interface for `jope-refittable-rust/v1` programs.

use super::contract::encode_f64;
use super::{
    InputContract, MathPipeline, NumericColumn, PipelineState, TargetScale, F64_PREDICTION_VERSION,
    INPUT_PART, PIPELINE_PART,
};
use crate::limits::Limits;
use crate::model_object::{Manifest, ModelObject, Part, Producer, MATH_PIPELINE, SCHEMA};
use crate::table::{read_csv, CsvTable};
use anyhow::{bail, ensure, Context, Result};
use feat_math::dsl::{ParamValue, Stage};
use feat_math::genome::GenomeIr;
use feat_math::heads::FittedHead;
use feat_math::model::Model;
use feat_math::ops::FittedOp;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;
use std::path::{Path, PathBuf};

pub const ABI_VERSION: u32 = 1;

fn admitted_stages(genome: &GenomeIr) -> Result<(&[Stage], &Stage)> {
    let GenomeIr::Pipeline(pipeline) = genome else {
        bail!("refittable Rust requires a regression pipeline");
    };
    let (head, stages) = pipeline
        .stages
        .split_last()
        .context("missing pipeline head")?;
    ensure!(
        matches!(head.op.as_str(), "ridge" | "histgbm" | "gbm")
            && stages.iter().all(|stage| {
                matches!(stage.op.as_str(), "standardize" | "robust_scale")
                    && stage.params.iter().all(|(key, value)| {
                        key == "append"
                            && matches!(value, ParamValue::Bool(false) | ParamValue::Int(0))
                    })
            }),
        "refittable Rust supports Ridge and GBM with optional non-appending scaling stages"
    );
    Ok((stages, head))
}

fn recipe_text(stages: &[Stage], head: &Stage) -> Result<String> {
    let mut text = String::from("(pipe X");
    for stage in stages {
        write!(text, " ({})", stage.op)?;
    }
    write!(text, " ({}", head.op)?;
    for (key, value) in &head.params {
        let valid = match (head.op.as_str(), key.as_str()) {
            ("ridge", "lambda_frac") | ("histgbm" | "gbm", "lr" | "lambda") => {
                value.as_f64().is_some_and(|v| v.is_finite() && v >= 0.0)
            }
            ("histgbm" | "gbm", "rounds" | "depth" | "bins" | "min_leaf" | "max_features") => {
                matches!(value, ParamValue::Int(v) if *v > 0)
            }
            _ => false,
        };
        ensure!(valid, "unsupported refittable recipe parameter {key:?}");
        match value {
            ParamValue::Int(value) => write!(text, " {key}={value}")?,
            // Preserve round-trip values, including tiny floats rounded by the DSL formatter.
            ParamValue::Float(value) => write!(text, " {key}={value:?}")?,
            _ => bail!("refittable head parameters must be numeric"),
        }
    }
    text.push_str("))");
    Ok(text)
}

fn check_state(stages: &[Stage], planned_head: &Stage, model: &Model) -> Result<()> {
    let Model::Pipeline {
        target: None,
        ops,
        head,
    } = model
    else {
        bail!("fitted pipeline does not match the refittable recipe");
    };
    ensure!(
        ops.len() == stages.len()
            && stages.iter().zip(ops).all(|(stage, op)| {
                matches!(
                    (stage.op.as_str(), op),
                    ("standardize", FittedOp::Standardize { append: false, .. })
                        | ("robust_scale", FittedOp::RobustScale { append: false, .. })
                )
            })
            && matches!(
                (planned_head.op.as_str(), head.as_ref()),
                ("ridge", FittedHead::Ridge(_)) | ("histgbm" | "gbm", FittedHead::Gbm(_))
            ),
        "fitted stages or head do not match the refittable recipe"
    );
    Ok(())
}

pub(super) fn check_pipeline(pipeline: &MathPipeline) -> Result<String> {
    let input = &pipeline.input;
    ensure!(
        input.prediction_version == F64_PREDICTION_VERSION
            && !input.target.name.is_empty()
            && input.target.scale > 0.0
            && input
                .columns
                .iter()
                .all(|column| column.name != input.target.name),
        "refittable Rust requires a version-two math-pipeline with a distinct target"
    );
    let (stages, head) = admitted_stages(&pipeline.genome)?;
    check_state(stages, head, &pipeline.state.model)?;
    recipe_text(stages, head)
}

pub(super) fn emit_rust(recipe: &str) -> String {
    format!(
        r#"//! jope-refittable-rust/v1. Build with the matching Jope runtime.
use jope_runtime::math_pipeline::refittable;
use std::path::Path;

const _: () = assert!(refittable::ABI_VERSION == 1);
const RECIPE: &str = {recipe:?};

pub fn fit(csv: &Path, target: &str, features: &[String], store: &Path)
    -> Result<std::path::PathBuf, Box<dyn std::error::Error>>
{{
    Ok(refittable::fit_csv(RECIPE, csv, target, features, store)?)
}}

pub fn predict(model: &Path, csv: &Path) -> Result<Vec<f64>, Box<dyn std::error::Error>> {{
    Ok(refittable::predict_csv(RECIPE, model, csv)?)
}}

fn main() -> Result<(), Box<dyn std::error::Error>> {{
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.as_slice() {{
        [command, csv, target, store, features @ ..] if command == "fit" => {{
            println!("{{}}", fit(Path::new(csv), target, features, Path::new(store))?.display());
        }}
        [command, model, csv] if command == "predict" => {{
            for value in predict(Path::new(model), Path::new(csv))? {{
                println!("{{value}}");
            }}
        }}
        _ => return Err("usage: fit <train.csv> <target> <store> <feature>... | predict <model-dir> <input.csv>".into()),
    }}
    Ok(())
}}
"#
    )
}

/// Fit from explicit complete numeric columns and publish a verified model object.
pub fn fit_csv(
    recipe: &str,
    path: &Path,
    target: &str,
    features: &[String],
    root: &Path,
) -> Result<PathBuf> {
    let limits = Limits::default();
    let genome = GenomeIr::parse(recipe)?;
    let (stages, head) = admitted_stages(&genome)?;
    let recipe = recipe_text(stages, head)?;
    crate::output_safety::refuse_directory_aliasing(root, &[path])?;
    let table = read_csv(path, &limits)?;
    ensure!(
        !features.is_empty() && !table.rows.is_empty(),
        "fit needs rows and at least one feature"
    );
    let mut names = BTreeSet::new();
    let indices = features
        .iter()
        .map(|name| {
            ensure!(
                name != target && names.insert(name),
                "duplicate feature or target used as a feature: {name:?}"
            );
            column_index(&table, name)
        })
        .collect::<Result<Vec<_>>>()?;
    let target_index = column_index(&table, target)?;
    let mut columns = Vec::with_capacity(features.len());
    for (name, &index) in features.iter().zip(&indices) {
        let (min, max) =
            table
                .rows
                .iter()
                .try_fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), row| {
                    let value = numeric(row[index].as_deref(), name)?;
                    Ok::<_, anyhow::Error>((min.min(value), max.max(value)))
                })?;
        columns.push(NumericColumn {
            name: name.clone(),
            min,
            max,
            constant: min == max,
        });
    }
    let labels = table
        .rows
        .iter()
        .map(|row| numeric(row[target_index].as_deref(), target))
        .collect::<Result<Vec<_>>>()?;
    let min = labels
        .iter()
        .copied()
        .reduce(f64::min)
        .context("no target rows")?;
    let max = labels
        .iter()
        .copied()
        .reduce(f64::max)
        .context("no target rows")?;
    let range = max - min;
    ensure!(
        range == 0.0 || range > 1e-12,
        "nonconstant FIT target range is below the prediction contract's resolution"
    );
    let input = InputContract::new(
        columns,
        (0..indices.len()).collect(),
        TargetScale {
            name: target.into(),
            min,
            scale: range.max(1e-12),
        },
    )?;
    let rows = table
        .rows
        .iter()
        .map(|row| {
            indices
                .iter()
                .zip(&input.columns)
                .map(|(&index, column)| encode_f64(row[index].as_deref(), column))
                .collect::<Result<Vec<_>>>()
        })
        .collect::<Result<Vec<_>>>()?;
    let labels = labels
        .into_iter()
        .map(|v| (v - min) / input.target.scale)
        .collect::<Vec<_>>();
    let model = feat_math::fit(&genome, &rows, &labels)?;
    check_state(stages, head, &model)?;
    let parts: BTreeMap<String, Vec<u8>> = BTreeMap::from([
        (INPUT_PART.into(), serde_json::to_vec(&input)?),
        (
            PIPELINE_PART.into(),
            serde_json::to_vec(&PipelineState { dsl: recipe, model })?,
        ),
    ]);
    let files = parts
        .iter()
        .map(|(path, data)| Part::new(path, "application/json", data))
        .collect();
    let object = ModelObject::new(
        Manifest {
            schema: SCHEMA.into(),
            minimum_reader_version: 1,
            format: MATH_PIPELINE.into(),
            producer: Producer::default(),
            primary: PIPELINE_PART.into(),
            required_capabilities: vec![],
            files,
            metadata: serde_json::json!({"source_profile": "jope-refittable-rust/v1"}),
        },
        parts,
        limits.max_model_bytes,
    )?;
    object.save(root, limits.max_model_bytes)
}

/// Open the caller-selected model object and predict in the saved input contract.
pub fn predict_csv(recipe: &str, model: &Path, path: &Path) -> Result<Vec<f64>> {
    let limits = Limits::default();
    let object = ModelObject::read(model, limits.max_model_bytes)?;
    let pipeline = object.math_pipeline()?;
    let genome = GenomeIr::parse(recipe)?;
    let (stages, head) = admitted_stages(&genome)?;
    ensure!(
        check_pipeline(&pipeline)? == recipe_text(stages, head)?,
        "model recipe does not match this program"
    );
    pipeline
        .predict_table(&read_csv(path, &limits)?)
        .map(|(predictions, _)| predictions)
}

fn column_index(table: &CsvTable, name: &str) -> Result<usize> {
    table
        .headers
        .iter()
        .position(|header| header == name)
        .with_context(|| format!("missing FIT column {name:?}"))
}

fn numeric(value: Option<&str>, column: &str) -> Result<f64> {
    value
        .and_then(|value| value.trim().parse::<f64>().ok())
        .filter(|value| value.is_finite())
        .with_context(|| format!("FIT column {column:?} requires complete finite numeric cells"))
}
