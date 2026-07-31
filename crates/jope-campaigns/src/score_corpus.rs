use anyhow::{bail, Context, Result};
use feat_math::dsl::{DslExpr, ParamValue, Stage};
use feat_math::math_expr::MathExpr;
use feat_math::GenomeIr;
use memmap2::MmapOptions;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Component, Path, PathBuf};

const ARCHIVE_MANIFEST: &str = "archive-manifest.v1.json";
const CORPUS_SCHEMA: &str = "jope-dime.score-corpus.v1";
const GENE_SCHEMA: &str = "jope-dime.gene-schema.v1";
const EXAMPLE_SCHEMA: &str = "jope-dime.score-example.v1";
const MAX_SNAPSHOT_ATTEMPTS: usize = 8;

#[derive(Debug)]
struct ArchiveChanged(&'static str);

impl fmt::Display for ArchiveChanged {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "archive changed while reading: {}", self.0)
    }
}

impl std::error::Error for ArchiveChanged {}

#[derive(Clone, Debug, Deserialize)]
struct ArchiveManifest {
    schema: String,
    all_evaluations_jsonl: String,
    all_evaluations_jsonl_sha256: String,
    evaluation_records: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct ExampleKey {
    cohort: String,
    dataset: String,
    semantic_sha256: String,
}

#[derive(Clone, Debug)]
struct Aggregate {
    canonical_genome: String,
    genome: GenomeExport,
    valid_rmse: Option<f64>,
    valid_count: usize,
    invalid_count: usize,
    runtimes: Vec<f64>,
    failure_classes: BTreeMap<String, usize>,
    provenance: Vec<Value>,
}

#[derive(Clone, Debug, Serialize)]
struct GenomeExport {
    form: String,
    stages: Vec<TypedStage>,
    actions: Vec<GeneAction>,
}

#[derive(Clone, Debug, Serialize)]
struct TypedStage {
    path: String,
    operation: String,
    parameters: Vec<TypedParameter>,
}

#[derive(Clone, Debug, Serialize)]
struct TypedParameter {
    name: String,
    value_type: String,
    value: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    expression_ast: Option<Value>,
}

#[derive(Clone, Debug, Serialize)]
struct GeneAction {
    kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    operation: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parameter: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    categorical_value: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    numeric_value: Option<f64>,
}

#[derive(Clone, Debug, Default)]
struct ObservedSchema {
    operations: BTreeSet<String>,
    parameters: BTreeMap<(String, String), BTreeSet<String>>,
    categorical_values: BTreeMap<(String, String), BTreeSet<String>>,
}

#[derive(Clone, Debug, Serialize)]
struct RuntimeStatistics {
    count: usize,
    mean_seconds: Option<f64>,
    std_seconds: Option<f64>,
    min_seconds: Option<f64>,
    p50_seconds: Option<f64>,
    p90_seconds: Option<f64>,
    max_seconds: Option<f64>,
}

#[derive(Clone, Debug, Serialize)]
struct ScoreExample<'a> {
    schema: &'static str,
    cohort: &'a str,
    dataset: &'a str,
    semantic_sha256: &'a str,
    canonical_genome: &'a str,
    genome: &'a GenomeExport,
    rmse: Option<f64>,
    percentile: Option<f64>,
    success_rate: f64,
    valid_count: usize,
    invalid_count: usize,
    total_count: usize,
    failure_classes: &'a BTreeMap<String, usize>,
    runtime: RuntimeStatistics,
    provenance: &'a [Value],
}

pub(crate) fn materialize(
    archive_root: &Path,
    output_root: &Path,
    cohort: Option<&str>,
) -> Result<()> {
    let mut last_change = None;
    for _ in 0..MAX_SNAPSHOT_ATTEMPTS {
        match read_and_materialize_filtered(archive_root, output_root, cohort, || {}) {
            Ok(path) => {
                println!("{}", path.display());
                return Ok(());
            }
            Err(error) if error.downcast_ref::<ArchiveChanged>().is_some() => {
                last_change = Some(error);
            }
            Err(error) => return Err(error),
        }
    }
    Err(last_change.unwrap_or_else(|| ArchiveChanged("retry limit exceeded").into()))
        .context("could not capture a stable archive snapshot")
}

fn read_and_materialize<F>(archive_root: &Path, output_root: &Path, after_map: F) -> Result<PathBuf>
where
    F: FnOnce(),
{
    read_and_materialize_filtered(archive_root, output_root, None, after_map)
}

fn read_and_materialize_filtered<F>(
    archive_root: &Path,
    output_root: &Path,
    cohort: Option<&str>,
    after_map: F,
) -> Result<PathBuf>
where
    F: FnOnce(),
{
    let manifest_path = archive_root.join(ARCHIVE_MANIFEST);
    let manifest_before =
        fs::read(&manifest_path).with_context(|| format!("read {}", manifest_path.display()))?;
    let manifest: ArchiveManifest =
        serde_json::from_slice(&manifest_before).context("parse archive manifest")?;
    if !manifest.schema.contains("regression-analysis-archive.v1") {
        bail!("unsupported archive schema '{}'", manifest.schema);
    }
    validate_sha256(
        &manifest.all_evaluations_jsonl_sha256,
        "archive JSONL SHA-256",
    )?;
    let relative = safe_relative_path(&manifest.all_evaluations_jsonl)?;
    let ledger_path = archive_root.join(relative);
    let ledger_file =
        File::open(&ledger_path).with_context(|| format!("open {}", ledger_path.display()))?;
    let mapped = unsafe { MmapOptions::new().map(&ledger_file) }
        .with_context(|| format!("memory-map {}", ledger_path.display()))?;
    after_map();

    let actual_ledger_sha256 = sha256_bytes(&mapped);
    if actual_ledger_sha256 != manifest.all_evaluations_jsonl_sha256 {
        return Err(ArchiveChanged("master JSONL digest differs from manifest").into());
    }
    let (mut aggregates, _, record_count) = aggregate_records(&mapped)?;
    if record_count != manifest.evaluation_records {
        return Err(ArchiveChanged("master JSONL record count differs from manifest").into());
    }
    let manifest_after =
        fs::read(&manifest_path).with_context(|| format!("re-read {}", manifest_path.display()))?;
    if manifest_after != manifest_before {
        return Err(ArchiveChanged("manifest bytes changed").into());
    }
    if let Some(cohort) = cohort {
        aggregates.retain(|key, _| key.cohort == cohort);
        if aggregates.is_empty() {
            bail!("archive has no score examples for cohort {cohort:?}");
        }
    }
    let mut observed = ObservedSchema::default();
    for aggregate in aggregates.values() {
        observe_genome(&aggregate.genome, &mut observed);
    }
    publish_corpus(
        output_root,
        &manifest_before,
        &manifest,
        aggregates,
        observed,
        cohort,
    )
}

fn aggregate_records(
    bytes: &[u8],
) -> Result<(BTreeMap<ExampleKey, Aggregate>, ObservedSchema, usize)> {
    let mut aggregates = BTreeMap::<ExampleKey, Aggregate>::new();
    let mut observed = ObservedSchema::default();
    let mut records = 0usize;
    for (index, raw) in bytes.split(|byte| *byte == b'\n').enumerate() {
        if raw.is_empty() {
            continue;
        }
        records += 1;
        let line = index + 1;
        let record: Value = serde_json::from_slice(raw)
            .with_context(|| format!("malformed archive record at line {line}"))?;
        let object = record
            .as_object()
            .with_context(|| format!("archive line {line} is not an object"))?;
        let required_string = |name: &str| -> Result<&str> {
            object
                .get(name)
                .and_then(Value::as_str)
                .filter(|value| !value.is_empty())
                .with_context(|| format!("archive line {line} lacks string field '{name}'"))
        };
        let cohort = required_string("cohort")?.to_string();
        let dataset = required_string("dataset")?.to_string();
        let semantic_sha256 = required_string("semantic_sha256")?.to_string();
        validate_sha256(&semantic_sha256, "semantic SHA-256")
            .with_context(|| format!("archive line {line}"))?;
        let canonical_genome = required_string("canonical_genome")?.to_string();
        if sha256_bytes(canonical_genome.as_bytes()) != semantic_sha256 {
            bail!("archive line {line} semantic SHA-256 does not bind canonical genome");
        }
        let parsed = GenomeIr::parse(&canonical_genome)
            .with_context(|| format!("archive line {line} genome does not parse"))?;
        if parsed.canonical() != canonical_genome {
            bail!("archive line {line} genome is not canonical under the current grammar");
        }
        let genome = export_genome(&parsed)
            .with_context(|| format!("archive line {line} genome export failed"))?;
        observe_genome(&genome, &mut observed);

        let valid = object
            .get("valid")
            .and_then(Value::as_bool)
            .with_context(|| format!("archive line {line} lacks boolean field 'valid'"))?;
        let rmse = if valid {
            let value = object
                .get("rmse")
                .and_then(Value::as_f64)
                .filter(|value| value.is_finite() && *value >= 0.0)
                .with_context(|| format!("archive line {line} has invalid valid RMSE"))?;
            Some(value)
        } else {
            None
        };
        let runtime = match object.get("runtime_seconds") {
            None | Some(Value::Null) => None,
            Some(value) => Some(
                value
                    .as_f64()
                    .filter(|seconds| seconds.is_finite() && *seconds >= 0.0)
                    .with_context(|| format!("archive line {line} has invalid runtime"))?,
            ),
        };
        let failure_class = if valid {
            None
        } else {
            object
                .get("failure")
                .and_then(Value::as_object)
                .and_then(|failure| failure.get("class"))
                .and_then(Value::as_str)
                .map(str::to_string)
                .or_else(|| Some("unknown".to_string()))
        };
        let key = ExampleKey {
            cohort,
            dataset,
            semantic_sha256,
        };
        let provenance = provenance_for(object, line);
        match aggregates.get_mut(&key) {
            Some(aggregate) => {
                if aggregate.canonical_genome != canonical_genome {
                    bail!(
                        "semantic key ({}, {}, {}) has conflicting canonical genomes",
                        key.cohort,
                        key.dataset,
                        key.semantic_sha256
                    );
                }
                if let (Some(left), Some(right)) = (aggregate.valid_rmse, rmse) {
                    if left.to_bits() != right.to_bits() {
                        bail!(
                            "semantic key ({}, {}, {}) has conflicting valid RMSE values",
                            key.cohort,
                            key.dataset,
                            key.semantic_sha256
                        );
                    }
                }
                if aggregate.valid_rmse.is_none() {
                    aggregate.valid_rmse = rmse;
                }
                aggregate.valid_count += usize::from(valid);
                aggregate.invalid_count += usize::from(!valid);
                if let Some(seconds) = runtime {
                    aggregate.runtimes.push(seconds);
                }
                if let Some(class) = failure_class {
                    *aggregate.failure_classes.entry(class).or_default() += 1;
                }
                aggregate.provenance.push(provenance);
            }
            None => {
                let mut failure_classes = BTreeMap::new();
                if let Some(class) = failure_class {
                    failure_classes.insert(class, 1);
                }
                aggregates.insert(
                    key,
                    Aggregate {
                        canonical_genome,
                        genome,
                        valid_rmse: rmse,
                        valid_count: usize::from(valid),
                        invalid_count: usize::from(!valid),
                        runtimes: runtime.into_iter().collect(),
                        failure_classes,
                        provenance: vec![provenance],
                    },
                );
            }
        }
    }
    if records == 0 {
        bail!("archive master JSONL is empty");
    }
    Ok((aggregates, observed, records))
}

fn provenance_for(record: &Map<String, Value>, source_line: usize) -> Value {
    let mut provenance = Map::new();
    provenance.insert("source_line".into(), json!(source_line));
    for key in [
        "source_campaign_id",
        "source_campaign_root",
        "source_campaign_order",
        "source_ledger_line",
        "run_id",
        "generation",
        "slot",
        "logical_evaluation",
        "fresh_evaluation",
        "cached",
        "node",
        "scq_job_id",
        "attempt",
        "receipt_file",
        "receipt_sha256",
        "checkpoint_file",
        "checkpoint_sha256",
        "binary_sha256",
        "train_sha256",
        "test_sha256",
        "predictions_sha256",
    ] {
        if let Some(value) = record.get(key) {
            provenance.insert(key.to_string(), value.clone());
        }
    }
    Value::Object(provenance)
}

fn export_genome(genome: &GenomeIr) -> Result<GenomeExport> {
    let mut stages = Vec::new();
    let mut actions = Vec::new();
    match genome {
        GenomeIr::Pipeline(pipeline) => {
            actions.push(category_action("form", None, None, "pipeline"));
            export_pipeline(pipeline, "pipeline", &mut stages, &mut actions)?;
            Ok(GenomeExport {
                form: "pipeline".into(),
                stages,
                actions,
            })
        }
        GenomeIr::Symbolic {
            source,
            expr,
            affine,
        } => {
            actions.push(category_action("form", None, None, "symbolic"));
            actions.push(category_action(
                "parameter",
                Some("symbolic"),
                Some("mode"),
                if *affine { "affine" } else { "raw" },
            ));
            let ast = serde_json::to_value(expr)?;
            actions.push(category_action(
                "expression_begin",
                Some("symbolic"),
                Some("expr"),
                "begin",
            ));
            export_expression_actions(&ast, "symbolic", &mut actions)?;
            actions.push(category_action(
                "expression_end",
                Some("symbolic"),
                Some("expr"),
                "end",
            ));
            stages.push(TypedStage {
                path: "symbolic".into(),
                operation: "symbolic".into(),
                parameters: vec![
                    TypedParameter {
                        name: "mode".into(),
                        value_type: "categorical".into(),
                        value: json!(if *affine { "affine" } else { "raw" }),
                        expression_ast: None,
                    },
                    TypedParameter {
                        name: "expr".into(),
                        value_type: "expression".into(),
                        value: json!(source),
                        expression_ast: Some(ast),
                    },
                ],
            });
            Ok(GenomeExport {
                form: "symbolic".into(),
                stages,
                actions,
            })
        }
        GenomeIr::Portfolio(portfolio) => {
            actions.push(category_action("form", None, None, "portfolio"));
            for (member, pipeline) in portfolio.members.iter().enumerate() {
                actions.push(category_action(
                    "member_begin",
                    None,
                    None,
                    &member.to_string(),
                ));
                export_pipeline(
                    pipeline,
                    &format!("member[{member}]"),
                    &mut stages,
                    &mut actions,
                )?;
                actions.push(category_action(
                    "member_end",
                    None,
                    None,
                    &member.to_string(),
                ));
            }
            let blend = Stage {
                op: "blend".into(),
                params: BTreeMap::from([
                    ("folds".into(), ParamValue::Int(portfolio.folds as i64)),
                    ("guard".into(), ParamValue::Str("clip".into())),
                    ("kind".into(), ParamValue::Str("nnls".into())),
                    (
                        "repeats".into(),
                        if portfolio.repeats_auto {
                            ParamValue::Str("auto".into())
                        } else {
                            ParamValue::Int(1)
                        },
                    ),
                ]),
            };
            stages.push(export_stage(&blend, "blend", &mut actions)?);
            Ok(GenomeExport {
                form: "portfolio".into(),
                stages,
                actions,
            })
        }
    }
}

fn export_pipeline(
    pipeline: &DslExpr,
    prefix: &str,
    stages: &mut Vec<TypedStage>,
    actions: &mut Vec<GeneAction>,
) -> Result<()> {
    for (position, stage) in pipeline.stages.iter().enumerate() {
        stages.push(export_stage(
            stage,
            &format!("{prefix}.stage[{position}]"),
            actions,
        )?);
    }
    Ok(())
}

fn export_stage(stage: &Stage, path: &str, actions: &mut Vec<GeneAction>) -> Result<TypedStage> {
    actions.push(category_action("stage", Some(&stage.op), None, &stage.op));
    let mut parameters = Vec::new();
    for (name, value) in &stage.params {
        let (value_type, json_value, expression_ast) = match value {
            ParamValue::Int(value) => {
                actions.push(numeric_action(&stage.op, name, *value as f64)?);
                ("integer", json!(*value), None)
            }
            ParamValue::Float(value) => {
                actions.push(numeric_action(&stage.op, name, *value)?);
                ("float64", json!(*value), None)
            }
            ParamValue::Bool(value) => {
                actions.push(category_action(
                    "parameter",
                    Some(&stage.op),
                    Some(name),
                    if *value { "true" } else { "false" },
                ));
                ("boolean", json!(*value), None)
            }
            ParamValue::Str(value) if name == "expr" => {
                let expression = MathExpr::parse(value)
                    .with_context(|| format!("{}.{name} is not a valid expression", stage.op))?;
                let ast = serde_json::to_value(expression)?;
                actions.push(category_action(
                    "expression_begin",
                    Some(&stage.op),
                    Some(name),
                    "begin",
                ));
                export_expression_actions(&ast, &stage.op, actions)?;
                actions.push(category_action(
                    "expression_end",
                    Some(&stage.op),
                    Some(name),
                    "end",
                ));
                ("expression", json!(value), Some(ast))
            }
            ParamValue::Str(value) => {
                actions.push(category_action(
                    "parameter",
                    Some(&stage.op),
                    Some(name),
                    value,
                ));
                ("categorical", json!(value), None)
            }
        };
        parameters.push(TypedParameter {
            name: name.clone(),
            value_type: value_type.into(),
            value: json_value,
            expression_ast,
        });
    }
    Ok(TypedStage {
        path: path.into(),
        operation: stage.op.clone(),
        parameters,
    })
}

fn export_expression_actions(
    value: &Value,
    operation: &str,
    actions: &mut Vec<GeneAction>,
) -> Result<()> {
    match value {
        Value::Object(object) => {
            for (key, child) in object {
                actions.push(category_action(
                    "expression_node",
                    Some(operation),
                    Some("expr"),
                    key,
                ));
                export_expression_actions(child, operation, actions)?;
            }
        }
        Value::Array(values) => {
            actions.push(category_action(
                "expression_node",
                Some(operation),
                Some("expr"),
                "list_begin",
            ));
            for child in values {
                export_expression_actions(child, operation, actions)?;
            }
            actions.push(category_action(
                "expression_node",
                Some(operation),
                Some("expr"),
                "list_end",
            ));
        }
        Value::String(token) => actions.push(category_action(
            "expression_node",
            Some(operation),
            Some("expr"),
            token,
        )),
        Value::Number(number) => {
            let numeric = number
                .as_f64()
                .filter(|value| value.is_finite())
                .context("expression AST contains a non-finite number")?;
            actions.push(numeric_action(operation, "expr_constant", numeric)?);
        }
        Value::Bool(value) => actions.push(category_action(
            "expression_node",
            Some(operation),
            Some("expr"),
            if *value { "true" } else { "false" },
        )),
        Value::Null => actions.push(category_action(
            "expression_node",
            Some(operation),
            Some("expr"),
            "null",
        )),
    }
    Ok(())
}

fn category_action(
    kind: &str,
    operation: Option<&str>,
    parameter: Option<&str>,
    value: &str,
) -> GeneAction {
    GeneAction {
        kind: kind.into(),
        operation: operation.map(str::to_string),
        parameter: parameter.map(str::to_string),
        categorical_value: Some(value.into()),
        numeric_value: None,
    }
}

fn numeric_action(operation: &str, parameter: &str, value: f64) -> Result<GeneAction> {
    if !value.is_finite() {
        bail!("non-finite numeric parameter {operation}.{parameter}");
    }
    Ok(GeneAction {
        kind: "parameter".into(),
        operation: Some(operation.into()),
        parameter: Some(parameter.into()),
        categorical_value: None,
        numeric_value: Some(value),
    })
}

fn observe_genome(genome: &GenomeExport, observed: &mut ObservedSchema) {
    for action in &genome.actions {
        let Some(operation) = action.operation.as_ref() else {
            continue;
        };
        observed.operations.insert(operation.clone());
        if let Some(parameter) = action.parameter.as_ref() {
            let key = (operation.clone(), parameter.clone());
            let value_type = if action.numeric_value.is_some() {
                "numeric"
            } else if action.kind.starts_with("expression") {
                "expression"
            } else {
                "categorical"
            };
            observed
                .parameters
                .entry(key.clone())
                .or_default()
                .insert(value_type.into());
            if let Some(value) = action.categorical_value.as_ref() {
                observed
                    .categorical_values
                    .entry(key)
                    .or_default()
                    .insert(value.clone());
            }
        }
    }
}

fn publish_corpus(
    output_root: &Path,
    archive_manifest_bytes: &[u8],
    archive_manifest: &ArchiveManifest,
    aggregates: BTreeMap<ExampleKey, Aggregate>,
    observed: ObservedSchema,
    cohort_filter: Option<&str>,
) -> Result<PathBuf> {
    let archive_sha256 = archive_manifest.all_evaluations_jsonl_sha256.as_str();
    let corpus_sha256 = match cohort_filter {
        None => archive_sha256.to_string(),
        Some(cohort) => sha256_bytes(
            format!("jope-dime.score-corpus.v1\n{archive_sha256}\n{cohort}\n").as_bytes(),
        ),
    };
    let corpus_root = output_root
        .join("dime_score")
        .join("corpora")
        .join(&corpus_sha256);
    if corpus_root.exists() {
        verify_existing_corpus(&corpus_root, archive_sha256, cohort_filter)?;
        return Ok(corpus_root);
    }
    let parent = corpus_root.parent().context("corpus path has no parent")?;
    fs::create_dir_all(parent)?;
    let temporary = parent.join(format!(".build-{archive_sha256}-{}", std::process::id()));
    if temporary.exists() {
        bail!(
            "stale corpus build directory exists: {}",
            temporary.display()
        );
    }
    fs::create_dir(&temporary)?;

    let percentiles = recompute_percentiles(&aggregates);
    let mut examples = Vec::new();
    let mut dataset_counts = BTreeMap::<String, usize>::new();
    let mut valid_examples = 0usize;
    for (key, aggregate) in &aggregates {
        let total = aggregate.valid_count + aggregate.invalid_count;
        let percentile = percentiles.get(key).copied().flatten();
        valid_examples += usize::from(aggregate.valid_rmse.is_some());
        *dataset_counts
            .entry(format!("{}:{}", key.cohort, key.dataset))
            .or_default() += 1;
        let example = ScoreExample {
            schema: EXAMPLE_SCHEMA,
            cohort: &key.cohort,
            dataset: &key.dataset,
            semantic_sha256: &key.semantic_sha256,
            canonical_genome: &aggregate.canonical_genome,
            genome: &aggregate.genome,
            rmse: aggregate.valid_rmse,
            percentile,
            success_rate: aggregate.valid_count as f64 / total as f64,
            valid_count: aggregate.valid_count,
            invalid_count: aggregate.invalid_count,
            total_count: total,
            failure_classes: &aggregate.failure_classes,
            runtime: runtime_statistics(&aggregate.runtimes),
            provenance: &aggregate.provenance,
        };
        examples.extend(serde_json::to_vec(&example)?);
        examples.push(b'\n');
    }
    let examples_sha256 = sha256_bytes(&examples);

    let parameters = observed
        .parameters
        .iter()
        .map(|((operation, parameter), types)| {
            json!({
                "operation": operation,
                "parameter": parameter,
                "types": types,
                "categorical_values": observed
                    .categorical_values
                    .get(&(operation.clone(), parameter.clone()))
                    .cloned()
                    .unwrap_or_default(),
            })
        })
        .collect::<Vec<_>>();
    let mut gene_schema_value = json!({
        "schema": GENE_SCHEMA,
        "parser_contract": "feat-math::GenomeIr current canonical grammar",
        "action_contract": [
            "form", "member_begin", "member_end", "stage", "parameter",
            "expression_begin", "expression_node", "expression_end"
        ],
        "numeric_contract": "finite IEEE-754 f64 values preserved without grid snapping",
        "operations": observed.operations,
        "parameters": parameters,
    });
    let contract_sha256 = sha256_bytes(&canonical_json(&gene_schema_value)?);
    gene_schema_value
        .as_object_mut()
        .context("gene schema is not an object")?
        .insert("contract_sha256".into(), json!(contract_sha256));
    let gene_schema_bytes = pretty_json(&gene_schema_value)?;
    let gene_schema_sha256 = sha256_bytes(&gene_schema_bytes);
    let archive_manifest_sha256 = sha256_bytes(archive_manifest_bytes);

    let manifest_value = json!({
        "schema": CORPUS_SCHEMA,
        "archive_sha256": archive_sha256,
        "archive_manifest_sha256": archive_manifest_sha256,
        "archive_schema": archive_manifest.schema,
        "cohort_filter": cohort_filter,
        "source_jsonl": archive_manifest.all_evaluations_jsonl,
        "source_records": archive_manifest.evaluation_records,
        "examples_file": "examples.jsonl",
        "examples_sha256": examples_sha256,
        "gene_schema_file": "gene-schema.v1.json",
        "gene_schema_sha256": gene_schema_sha256,
        "unique_examples": aggregates.len(),
        "valid_score_examples": valid_examples,
        "invalid_only_examples": aggregates.len() - valid_examples,
        "datasets": dataset_counts,
        "percentile_definition": "count(unique valid RMSE >= current) / count(unique valid RMSE) within each cohort/dataset; ties receive identical values",
        "validity_definition": "valid_count / (valid_count + invalid_count); mixed outcomes retain their valid RMSE",
    });
    let manifest_bytes = pretty_json(&manifest_value)?;

    write_synced(&temporary.join("examples.jsonl"), &examples)?;
    write_synced(&temporary.join("gene-schema.v1.json"), &gene_schema_bytes)?;
    write_synced(&temporary.join("manifest.v1.json"), &manifest_bytes)?;
    sync_directory(&temporary)?;
    match fs::rename(&temporary, &corpus_root) {
        Ok(()) => {
            sync_directory(parent)?;
            verify_existing_corpus(&corpus_root, archive_sha256, cohort_filter)?;
            Ok(corpus_root)
        }
        Err(_error) if corpus_root.exists() => {
            fs::remove_dir_all(&temporary).ok();
            verify_existing_corpus(&corpus_root, archive_sha256, cohort_filter)
                .context("concurrent corpus publication conflicts")?;
            Ok(corpus_root)
        }
        Err(error) => Err(error).context("publish immutable corpus directory"),
    }
}

fn recompute_percentiles(
    aggregates: &BTreeMap<ExampleKey, Aggregate>,
) -> BTreeMap<ExampleKey, Option<f64>> {
    let mut scores = BTreeMap::<(String, String), Vec<f64>>::new();
    for (key, aggregate) in aggregates {
        if let Some(rmse) = aggregate.valid_rmse {
            scores
                .entry((key.cohort.clone(), key.dataset.clone()))
                .or_default()
                .push(rmse);
        }
    }
    for values in scores.values_mut() {
        values.sort_by(f64::total_cmp);
    }
    aggregates
        .iter()
        .map(|(key, aggregate)| {
            let percentile = aggregate.valid_rmse.map(|rmse| {
                let values = &scores[&(key.cohort.clone(), key.dataset.clone())];
                let lower = values.partition_point(|value| *value < rmse);
                (values.len() - lower) as f64 / values.len() as f64
            });
            (key.clone(), percentile)
        })
        .collect()
}

fn runtime_statistics(values: &[f64]) -> RuntimeStatistics {
    if values.is_empty() {
        return RuntimeStatistics {
            count: 0,
            mean_seconds: None,
            std_seconds: None,
            min_seconds: None,
            p50_seconds: None,
            p90_seconds: None,
            max_seconds: None,
        };
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mean = sorted.iter().sum::<f64>() / sorted.len() as f64;
    let variance = sorted
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / sorted.len() as f64;
    RuntimeStatistics {
        count: sorted.len(),
        mean_seconds: Some(mean),
        std_seconds: Some(variance.sqrt()),
        min_seconds: sorted.first().copied(),
        p50_seconds: Some(quantile(&sorted, 0.5)),
        p90_seconds: Some(quantile(&sorted, 0.9)),
        max_seconds: sorted.last().copied(),
    }
}

fn quantile(sorted: &[f64], probability: f64) -> f64 {
    let position = probability * (sorted.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    if lower == upper {
        sorted[lower]
    } else {
        sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower as f64)
    }
}

fn verify_existing_corpus(
    root: &Path,
    archive_sha256: &str,
    cohort_filter: Option<&str>,
) -> Result<()> {
    let manifest_path = root.join("manifest.v1.json");
    let manifest: Value = serde_json::from_slice(&fs::read(&manifest_path)?)
        .with_context(|| format!("parse {}", manifest_path.display()))?;
    if manifest.get("schema").and_then(Value::as_str) != Some(CORPUS_SCHEMA)
        || manifest.get("archive_sha256").and_then(Value::as_str) != Some(archive_sha256)
        || manifest.get("cohort_filter").and_then(Value::as_str) != cohort_filter
    {
        bail!("existing corpus manifest identity mismatch");
    }
    for (file_key, digest_key) in [
        ("examples_file", "examples_sha256"),
        ("gene_schema_file", "gene_schema_sha256"),
    ] {
        let relative = manifest
            .get(file_key)
            .and_then(Value::as_str)
            .context("existing corpus manifest lacks file name")?;
        let expected = manifest
            .get(digest_key)
            .and_then(Value::as_str)
            .context("existing corpus manifest lacks file digest")?;
        let path = root.join(safe_relative_path(relative)?);
        let actual = sha256_bytes(&fs::read(&path)?);
        if actual != expected {
            bail!(
                "existing corpus artifact SHA-256 mismatch: {}",
                path.display()
            );
        }
    }
    Ok(())
}

fn safe_relative_path(value: &str) -> Result<PathBuf> {
    let path = Path::new(value);
    if path.as_os_str().is_empty()
        || path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        bail!("archive path must be a non-empty normal relative path: {value:?}");
    }
    Ok(path.to_path_buf())
}

fn validate_sha256(value: &str, label: &str) -> Result<()> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        bail!("{label} is not a lowercase SHA-256");
    }
    Ok(())
}

fn sha256_bytes(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn canonical_json(value: &Value) -> Result<Vec<u8>> {
    Ok(serde_json::to_vec(value)?)
}

fn pretty_json(value: &Value) -> Result<Vec<u8>> {
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    Ok(bytes)
}

fn write_synced(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .with_context(|| format!("create {}", path.display()))?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

fn sync_directory(path: &Path) -> Result<()> {
    File::open(path)?.sync_all()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn record(
        dataset: &str,
        genome: &str,
        valid: bool,
        rmse: Option<f64>,
        failure: Option<&str>,
        runtime: f64,
    ) -> Value {
        json!({
            "schema": "fixture",
            "cohort": "golden42",
            "dataset": dataset,
            "canonical_genome": genome,
            "semantic_sha256": sha256_bytes(genome.as_bytes()),
            "valid": valid,
            "rmse": rmse,
            "failure": failure.map(|class| json!({"class": class})),
            "runtime_seconds": runtime,
            "run_id": "fixture",
        })
    }

    fn fixture_archive(root: &Path, rows: &[Value]) {
        fs::create_dir_all(root).unwrap();
        let mut ledger = Vec::new();
        for row in rows {
            ledger.extend(serde_json::to_vec(row).unwrap());
            ledger.push(b'\n');
        }
        fs::write(root.join("all-evaluations.jsonl"), &ledger).unwrap();
        let manifest = json!({
            "schema": "jope-dime.all-regression-analysis-archive.v1",
            "all_evaluations_jsonl": "all-evaluations.jsonl",
            "all_evaluations_jsonl_sha256": sha256_bytes(&ledger),
            "evaluation_records": rows.len(),
        });
        fs::write(
            root.join(ARCHIVE_MANIFEST),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();
    }

    fn read_examples(root: &Path) -> Vec<Value> {
        let manifest: Value =
            serde_json::from_slice(&fs::read(root.join(ARCHIVE_MANIFEST)).unwrap()).unwrap();
        let archive_sha = manifest["all_evaluations_jsonl_sha256"].as_str().unwrap();
        fs::read_to_string(
            root.parent()
                .unwrap()
                .join("out/dime_score/corpora")
                .join(archive_sha)
                .join("examples.jsonl"),
        )
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
    }

    #[test]
    fn duplicate_and_mixed_outcomes_are_aggregated() {
        let temp = TempDir::new().unwrap();
        let archive = temp.path().join("archive");
        let genome = "(pipe X (ridge))";
        fixture_archive(
            &archive,
            &[
                record("d", genome, false, None, Some("timeout"), 30.0),
                record("d", genome, true, Some(1.0), None, 2.0),
                record("d", genome, true, Some(1.0), None, 2.0),
            ],
        );
        read_and_materialize(&archive, &temp.path().join("out"), || {}).unwrap();
        let examples = read_examples(&archive);
        assert_eq!(examples.len(), 1);
        assert_eq!(examples[0]["valid_count"], 2);
        assert_eq!(examples[0]["invalid_count"], 1);
        assert_eq!(examples[0]["success_rate"], json!(2.0 / 3.0));
        assert_eq!(examples[0]["failure_classes"]["timeout"], 1);
        assert_eq!(examples[0]["rmse"], 1.0);
    }

    #[test]
    fn tied_percentiles_use_unique_valid_genomes() {
        let temp = TempDir::new().unwrap();
        let archive = temp.path().join("archive");
        fixture_archive(
            &archive,
            &[
                record("d", "(pipe X (ridge))", true, Some(1.0), None, 1.0),
                record(
                    "d",
                    "(pipe X (ridge alpha=1.0))",
                    true,
                    Some(1.0),
                    None,
                    1.0,
                ),
                record("d", "(pipe X (ridge_gcv))", true, Some(2.0), None, 1.0),
            ],
        );
        read_and_materialize(&archive, &temp.path().join("out"), || {}).unwrap();
        let examples = read_examples(&archive);
        let percentiles = examples
            .iter()
            .map(|row| row["percentile"].as_f64().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(percentiles, vec![1.0, 1.0, 1.0 / 3.0]);
    }

    #[test]
    fn conflicting_valid_rmse_is_fatal() {
        let temp = TempDir::new().unwrap();
        let archive = temp.path().join("archive");
        let genome = "(pipe X (ridge))";
        fixture_archive(
            &archive,
            &[
                record("d", genome, true, Some(1.0), None, 1.0),
                record("d", genome, true, Some(2.0), None, 1.0),
            ],
        );
        let error = read_and_materialize(&archive, &temp.path().join("out"), || {})
            .unwrap_err()
            .to_string();
        assert!(error.contains("conflicting valid RMSE"));
    }

    #[test]
    fn malformed_records_and_sha_mismatch_are_fatal_or_retryable() {
        let temp = TempDir::new().unwrap();
        let archive = temp.path().join("archive");
        fixture_archive(
            &archive,
            &[record("d", "(pipe X (ridge))", true, Some(1.0), None, 1.0)],
        );
        fs::write(archive.join("all-evaluations.jsonl"), b"{bad\n").unwrap();
        let error = read_and_materialize(&archive, &temp.path().join("out"), || {}).unwrap_err();
        assert!(error.downcast_ref::<ArchiveChanged>().is_some());
    }

    #[test]
    fn manifest_change_during_mapped_read_requests_retry() {
        let temp = TempDir::new().unwrap();
        let archive = temp.path().join("archive");
        fixture_archive(
            &archive,
            &[record("d", "(pipe X (ridge))", true, Some(1.0), None, 1.0)],
        );
        let path = archive.join(ARCHIVE_MANIFEST);
        let error = read_and_materialize(&archive, &temp.path().join("out"), || {
            let mut value: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
            value["updated_at_unix_ms"] = json!(1);
            fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
        })
        .unwrap_err();
        assert!(error.downcast_ref::<ArchiveChanged>().is_some());
    }

    #[test]
    fn repeated_builds_are_byte_identical() {
        let first = TempDir::new().unwrap();
        let second = TempDir::new().unwrap();
        let rows = [
            record(
                "d",
                "(pipe X (nystrom gamma=0.6688638669418179 k=16) (ridge))",
                true,
                Some(1.0),
                None,
                1.25,
            ),
            record(
                "d",
                "(pipe X (derive expr=\"col(0) + 1.25\") (ridge))",
                false,
                None,
                Some("timeout"),
                30.0,
            ),
        ];
        fixture_archive(&first.path().join("archive"), &rows);
        fixture_archive(&second.path().join("archive"), &rows);
        let left = read_and_materialize(
            &first.path().join("archive"),
            &first.path().join("out"),
            || {},
        )
        .unwrap();
        let right = read_and_materialize(
            &second.path().join("archive"),
            &second.path().join("out"),
            || {},
        )
        .unwrap();
        for file in ["examples.jsonl", "gene-schema.v1.json", "manifest.v1.json"] {
            assert_eq!(
                fs::read(left.join(file)).unwrap(),
                fs::read(right.join(file)).unwrap()
            );
        }
    }

    #[test]
    fn cohort_filter_creates_a_distinct_corpus() {
        let temp = TempDir::new().unwrap();
        let archive = temp.path().join("archive");
        let mut golden = record("g", "(pipe X (ridge))", true, Some(1.0), None, 1.0);
        golden["cohort"] = json!("golden42");
        let mut holdout = record("h", "(pipe X (ridge_gcv))", true, Some(2.0), None, 1.0);
        holdout["cohort"] = json!("holdout13");
        fixture_archive(&archive, &[golden, holdout]);

        let root = read_and_materialize_filtered(
            &archive,
            &temp.path().join("out"),
            Some("golden42"),
            || {},
        )
        .unwrap();
        let manifest: Value =
            serde_json::from_slice(&fs::read(root.join("manifest.v1.json")).unwrap()).unwrap();
        assert_eq!(manifest["cohort_filter"], "golden42");
        assert_eq!(manifest["datasets"].as_object().unwrap().len(), 1);
        assert_eq!(manifest["datasets"]["golden42:g"], 1);
        assert_eq!(
            fs::read_to_string(root.join("examples.jsonl"))
                .unwrap()
                .lines()
                .count(),
            1
        );
    }
}
