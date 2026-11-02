//! Rust port of the Prime realized-label to training-row adapter.
//!
//! Grammar resolution and dataset embeddings are injected because their current
//! implementations live in the Python policy and signature packages.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::{BTreeMap, BTreeSet};

pub const LEAGUE_LABEL_SCHEMA: &str = "LeagueLabel.v1";
pub const DEFAULT_PROVENANCE: &str = "prime_realized";
pub const DEVELOPMENT_SPLIT: &str = "development";
pub const TEMPORAL_AUDIT_SPLIT: &str = "temporal_audit";

/// Resolves a canonical genome to the grammar key accepted by the trainer.
pub trait GenomeEngineResolver {
    fn resolve(&mut self, genome: &str, declared_engine: Option<&str>) -> Option<String>;
}

/// Supplies the dataset and target-summary embeddings for a dataset identity.
pub trait DatasetEmbedder {
    fn embed(&mut self, dataset_hash: &str) -> (Option<Vec<f64>>, Option<Vec<f64>>);
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct TrainingRow {
    pub dataset_id: String,
    pub source: String,
    pub embedding: Vec<f64>,
    pub target_embedding: Option<Vec<f64>>,
    pub genome: String,
    pub canonical_genome: Option<String>,
    pub valid: bool,
    pub metric: Option<String>,
    pub score: Option<f64>,
    pub rank: Option<i64>,
    pub elo: Option<f64>,
    pub cost_ms: Option<f64>,
    #[serde(default)]
    pub family_bitmap: BTreeMap<String, bool>,
    pub complexity: Option<i64>,
    pub runtime_class: Option<String>,
    pub feedback_origin: Option<String>,
    pub engine: String,
    #[serde(default)]
    pub metadata: Map<String, Value>,
}

struct TrainingRowParts {
    dataset: String,
    source: String,
    embedding: Vec<f64>,
    target: Option<Vec<f64>>,
    genome: String,
    engine: String,
    valid: bool,
    score: Option<f64>,
    cost_ms: Option<f64>,
    metadata: Map<String, Value>,
}

impl TrainingRow {
    fn from_parts(parts: TrainingRowParts) -> Self {
        Self {
            dataset_id: parts.dataset,
            source: parts.source,
            embedding: normalize_embedding(parts.embedding, &[512, 1536]),
            target_embedding: parts
                .target
                .map(|values| normalize_embedding(values, &[512])),
            genome: parts.genome.clone(),
            canonical_genome: Some(parts.genome),
            valid: parts.valid,
            metric: None,
            score: parts.score,
            rank: None,
            elo: None,
            cost_ms: parts.cost_ms,
            family_bitmap: BTreeMap::new(),
            complexity: None,
            runtime_class: None,
            feedback_origin: None,
            engine: parts.engine,
            metadata: parts.metadata,
        }
    }
}

fn normalize_embedding(values: Vec<f64>, dims: &[usize]) -> Vec<f64> {
    if dims.contains(&values.len()) {
        return values;
    }
    let mut out = values.into_iter().take(dims[0]).collect::<Vec<_>>();
    out.resize(dims[0], 0.0);
    out
}

pub(crate) fn python_canonical_json(value: &Value) -> String {
    fn quoted(value: &str) -> String {
        let serde = serde_json::to_string(value).expect("string serializes");
        let mut out = String::with_capacity(serde.len());
        for ch in serde.chars() {
            let code = ch as u32;
            if code <= 0x7f {
                out.push(ch);
            } else if code <= 0xffff {
                out.push_str(&format!("\\u{code:04x}"));
            } else {
                let scalar = code - 0x1_0000;
                out.push_str(&format!(
                    "\\u{:04x}\\u{:04x}",
                    0xd800 + (scalar >> 10),
                    0xdc00 + (scalar & 0x3ff)
                ));
            }
        }
        out
    }
    match value {
        Value::Null => "null".into(),
        Value::Bool(value) => value.to_string(),
        Value::Number(value) => value.to_string(),
        Value::String(value) => quoted(value),
        Value::Array(values) => format!(
            "[{}]",
            values
                .iter()
                .map(python_canonical_json)
                .collect::<Vec<_>>()
                .join(",")
        ),
        Value::Object(values) => {
            let mut keys = values.keys().collect::<Vec<_>>();
            keys.sort();
            format!(
                "{{{}}}",
                keys.into_iter()
                    .map(|key| format!("{}:{}", quoted(key), python_canonical_json(&values[key])))
                    .collect::<Vec<_>>()
                    .join(",")
            )
        }
    }
}

fn get<'a>(v: &'a Value, key: &str) -> Option<&'a Value> {
    v.as_object()?.get(key)
}
fn text(v: Option<&Value>) -> Option<String> {
    match v? {
        Value::String(s) => Some(s.clone()),
        Value::Number(n) => Some(n.to_string()),
        Value::Bool(b) => Some(b.to_string()),
        _ => None,
    }
}
fn truthy(v: Option<&Value>) -> bool {
    match v {
        None | Some(Value::Null) | Some(Value::Bool(false)) => false,
        Some(Value::String(s)) => !s.is_empty(),
        Some(Value::Number(n)) => n.as_f64().unwrap_or(0.0) != 0.0,
        Some(Value::Array(a)) => !a.is_empty(),
        Some(Value::Object(o)) => !o.is_empty(),
        _ => true,
    }
}
fn finite(v: Option<&Value>) -> Option<f64> {
    let x = match v? {
        Value::Number(n) => n.as_f64()?,
        Value::String(s) => s.parse().ok()?,
        Value::Bool(b) => {
            if *b {
                1.0
            } else {
                0.0
            }
        }
        _ => return None,
    };
    x.is_finite().then_some(x)
}
fn wrapped_label(row: &Value) -> &Value {
    get(row, "label").filter(|v| v.is_object()).unwrap_or(row)
}
fn authority(row: &Value) -> bool {
    text(get(wrapped_label(row), "split"))
        .unwrap_or_else(|| "train".into())
        .trim()
        .to_lowercase()
        .replace('-', "_")
        == "train"
}
fn first_truthy<'a>(row: &'a Value, a: &str, b: &str) -> Option<&'a Value> {
    let x = get(row, a);
    if truthy(x) {
        x
    } else {
        get(row, b)
    }
}

mod parse;
mod placeholder;
mod split_manifest;
use parse::parse_row;

pub fn label_to_training_row<R: GenomeEngineResolver, E: DatasetEmbedder>(
    row: &Value,
    embedder: &mut E,
    resolver: &mut R,
    split: &str,
    provenance: &str,
) -> Result<(Option<TrainingRow>, &'static str), String> {
    if !authority(row) {
        return Ok((None, "report_only_evidence"));
    }
    let Some(parsed) = parse_row(row)? else {
        return Ok((None, "missing_identity"));
    };
    let genome = parsed["canonical_genome"].as_str().unwrap();
    let engine = resolver.resolve(genome, parsed.get("engine").and_then(Value::as_str));
    let Some(engine) = engine else {
        return Ok((None, "unencodable_genome"));
    };
    let reward = finite(parsed.get("reward"));
    let failed = parsed["failed"].as_bool().unwrap_or(false);
    if reward.is_none() && !failed {
        return Ok((None, "no_finite_reward"));
    }
    let dataset = parsed["dataset_hash"].as_str().unwrap();
    let (embedding, target) = embedder.embed(dataset);
    let synthetic = embedding.is_none();
    let (embedding, target) = match embedding {
        Some(embedding) => (embedding, target),
        None => {
            let (embedding, target) = placeholder::placeholder(dataset);
            (embedding, Some(target))
        }
    };
    let mut metadata = Map::new();
    let mut put = |key: &str, value: Value| {
        metadata.insert(key.into(), value);
    };
    put("provenance", Value::String(provenance.into()));
    put("task_type", parsed["task_type"].clone());
    put("split", Value::String(split.into()));
    put("dataset_hash", Value::String(dataset.into()));
    for key in [
        "semantic_plan_hash",
        "p_bad",
        "reward_std",
        "n_evidence",
        "promotion_eligible",
        "label_watermark",
        "committed_at_ms",
        "replay_hot_until_ms",
        "fold_r2",
        "folds",
        "fold_plan",
        "evaluator_version",
        "evidence_tier",
        "gp_task_id",
        "robust_formula",
        "reward_pair_identity",
    ] {
        put(key, parsed.get(key).cloned().unwrap_or(Value::Null));
    }
    put(
        "realization_engine",
        parsed.get("engine").cloned().unwrap_or(Value::Null),
    );
    put("embedding_synthetic", Value::Bool(synthetic));
    if let Some(reward) = reward {
        put("score_robust", Value::from(reward));
        put(
            "oof_r2_raw",
            parsed
                .get("oof_r2")
                .and_then(|value| finite(Some(value)))
                .or(Some(reward))
                .map(Value::from)
                .unwrap(),
        );
    }
    if failed {
        put("eval_error", Value::Bool(true));
    }
    Ok((
        Some(TrainingRow::from_parts(TrainingRowParts {
            dataset: dataset.into(),
            source: provenance.into(),
            embedding,
            target,
            genome: genome.into(),
            engine,
            valid: !failed,
            score: reward,
            cost_ms: finite(parsed.get("runtime_seconds")).map(|v| v * 1_000.0),
            metadata,
        })),
        "ok",
    ))
}

/// Split and metadata inputs for the corpus adapter.
pub struct BuildOptions {
    pub val_fraction: f64,
    pub seed: u64,
    pub provenance: String,
    pub validation_datasets: Option<BTreeSet<String>>,
    pub split_manifest: Option<Value>,
}

impl Default for BuildOptions {
    fn default() -> Self {
        Self {
            val_fraction: 0.2,
            seed: 0,
            provenance: DEFAULT_PROVENANCE.into(),
            validation_datasets: None,
            split_manifest: None,
        }
    }
}

/// Adapt label rows into policy training rows while keeping whole datasets in one split.
pub fn build_corpus<R: GenomeEngineResolver, E: DatasetEmbedder>(
    rows: &[Value],
    embedder: &mut E,
    resolver: &mut R,
    options: BuildOptions,
) -> Result<(Vec<TrainingRow>, Value), String> {
    let mut manifest_sha = Value::Null;
    let mut manifest_values = None;
    if let Some(manifest) = options.split_manifest.as_ref() {
        split_manifest::validate(manifest)?;
        if manifest["status"] != "complete" {
            return Err("cannot build a corpus from an incomplete split manifest".into());
        }
        manifest_sha = manifest["manifest_sha256"].clone();
        manifest_values = Some(manifest);
    }
    let hashes = rows
        .iter()
        .filter(|r| authority(r))
        .map(|r| {
            let label = wrapped_label(r);
            text(first_truthy(label, "dataset_hash", "dataset_id"))
                .or_else(|| text(get(label, "dataset")))
                .unwrap_or_default()
        })
        .filter(|s| !s.is_empty())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let val_datasets = if let Some(manifest) = manifest_values {
        manifest["assignments"]
            .as_object()
            .unwrap()
            .iter()
            .filter_map(|(dataset, record)| {
                (record["split"] == DEVELOPMENT_SPLIT).then_some(dataset.clone())
            })
            .collect::<BTreeSet<_>>()
    } else if let Some(explicit) = options.validation_datasets {
        explicit
    } else {
        let mut shuffled = hashes.clone();
        let mut rng = placeholder::PythonRandom::from_seed(options.seed);
        rng.shuffle(&mut shuffled);
        let n = if options.val_fraction > 0.0 && shuffled.len() > 1 {
            let x = options.val_fraction * shuffled.len() as f64;
            let lo = x.floor();
            let frac = x - lo;
            let rounded = if frac < 0.5 {
                lo
            } else if frac > 0.5 {
                lo + 1.0
            } else if (lo as u64) & 1 == 0 {
                lo
            } else {
                lo + 1.0
            };
            (rounded as usize).max(1)
        } else {
            0
        };
        shuffled.into_iter().take(n).collect()
    };
    let mut output = Vec::new();
    let mut reasons = BTreeMap::<String, usize>::new();
    let mut engines = BTreeMap::<String, usize>::new();
    let mut synthetic = 0usize;
    let mut train = BTreeSet::new();
    let mut validation = BTreeSet::new();
    let mut audit = BTreeSet::new();
    for row in rows {
        let label = wrapped_label(row);
        let dataset = text(first_truthy(label, "dataset_hash", "dataset_id"))
            .or_else(|| text(get(label, "dataset")))
            .unwrap_or_default();
        if !authority(row) {
            *reasons.entry("report_only_evidence".into()).or_default() += 1;
            continue;
        }
        let assignment = manifest_values.and_then(|m| split_manifest::assignment(m, &dataset));
        if assignment
            .and_then(|a| a.get("split"))
            .and_then(Value::as_str)
            == Some(TEMPORAL_AUDIT_SPLIT)
        {
            audit.insert(dataset);
            *reasons
                .entry("report_only_temporal_audit".into())
                .or_default() += 1;
            continue;
        }
        let split = assignment
            .and_then(|a| a.get("split"))
            .and_then(Value::as_str)
            .unwrap_or_else(|| {
                if val_datasets.contains(&dataset) {
                    "validation"
                } else {
                    "train"
                }
            });
        let (result, reason) =
            label_to_training_row(row, embedder, resolver, split, &options.provenance)?;
        *reasons.entry(reason.into()).or_default() += 1;
        if let Some(mut tr) = result {
            *engines.entry(tr.engine.clone()).or_default() += 1;
            if tr
                .metadata
                .get("embedding_synthetic")
                .and_then(Value::as_bool)
                .unwrap_or(false)
            {
                synthetic += 1;
            }
            if let Some(assignment) = assignment {
                tr.metadata
                    .insert("source_family".into(), assignment["source_family"].clone());
                tr.metadata
                    .insert("split_assignment".into(), assignment["split"].clone());
                tr.metadata.insert(
                    "split_manifest_schema".into(),
                    manifest_values.unwrap()["schema"].clone(),
                );
                tr.metadata
                    .insert("split_manifest_sha256".into(), manifest_sha.clone());
            }
            if split == "train" {
                train.insert(tr.dataset_id.clone());
            } else {
                validation.insert(tr.dataset_id.clone());
            }
            output.push(tr);
        }
    }
    let skipped = reasons
        .iter()
        .filter(|(k, _)| k.as_str() != "ok")
        .map(|(k, v)| (k.clone(), Value::from(*v)))
        .collect::<Map<_, _>>();
    let stats = serde_json::json!({"input_rows":rows.len(),"training_rows":output.len(),"skipped":skipped,"engines":engines,
        "datasets":train.union(&validation).count(),"train_datasets":train.len(),"validation_datasets":validation.len(),
        "validation_dataset_ids":val_datasets,"temporal_audit_datasets":audit.len(),"temporal_audit_dataset_ids":audit,
        "split_manifest_sha256":manifest_sha,"embedding_dim":output.first().map_or(0,|r|r.embedding.len()),
        "target_summary_dim":output.first().and_then(|r|r.target_embedding.as_ref()).map_or(0,|v|v.len()),"synthetic_embedding_rows":synthetic});
    Ok((output, stats))
}

pub fn write_corpus(
    rows: &[TrainingRow],
    path: impl AsRef<std::path::Path>,
) -> std::io::Result<()> {
    use std::io::Write;
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut output = std::fs::File::create(path)?;
    for row in rows {
        serde_json::to_writer(&mut output, row).map_err(std::io::Error::other)?;
        output.write_all(b"\n")?;
    }
    Ok(())
}

#[cfg(test)]
mod tests;
