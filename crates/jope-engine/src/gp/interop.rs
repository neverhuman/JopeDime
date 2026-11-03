//! JSONL interoperability for seeding and harvesting the external newmath GP engine.
//!
//! The GP state and winner directory are read-only inputs. Export files and harvested rows are
//! written only to caller-selected output paths, matching `python/src/jope/gp_interop.py`.

use anyhow::{Context, Result};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

mod wire;
use wire::{number, read_jsonl, write_jsonl, NONFINITE_PREFIX};

/// `python/prime.toml` default used by the Python wrapper when no explicit directory is passed.
pub const DEFAULT_WINNERS_DIR: &str = "/home/ubuntu/jmath/artifacts/newmath/winners";

#[derive(Clone, Copy, Debug)]
pub struct ExportOptions<'a> {
    pub top: i64,
    pub min_oof: f64,
    pub engine: &'a str,
    pub task_family: &'a str,
    pub acq_base: f64,
    pub acq_scale: f64,
}

impl<'a> Default for ExportOptions<'a> {
    fn default() -> Self {
        Self {
            top: 200,
            min_oof: 0.0,
            engine: "newmath",
            task_family: "regression",
            acq_base: 10.0,
            acq_scale: 5.0,
        }
    }
}

fn sha(text: &str) -> String {
    hex::encode(Sha256::digest(text.as_bytes()))
}

fn ops_of(genome: &str) -> Vec<String> {
    let bytes = genome.as_bytes();
    let mut ops = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] != b'(' {
            i += 1;
            continue;
        }
        let start = i + 1;
        let mut end = start;
        while end < bytes.len()
            && (bytes[end].is_ascii_lowercase()
                || bytes[end].is_ascii_digit()
                || bytes[end] == b'_')
        {
            end += 1;
        }
        if end > start {
            let op = &genome[start..end];
            if op != "pipe" {
                ops.push(op.to_string());
            }
            i = end;
        } else {
            i += 1;
        }
    }
    ops
}

fn truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(value) => *value,
        Value::Number(value) => value.as_f64().is_some_and(|n| n != 0.0),
        Value::String(value) => !value.is_empty(),
        Value::Array(value) => !value.is_empty(),
        Value::Object(value) => !value.is_empty(),
    }
}

fn py_str(value: &Value) -> String {
    match value {
        Value::Null => "None".to_string(),
        Value::Bool(true) => "True".to_string(),
        Value::Bool(false) => "False".to_string(),
        Value::Number(value) => value.to_string(),
        Value::String(value) if value == &format!("{NONFINITE_PREFIX}nan") => "nan".to_string(),
        Value::String(value) if value == &format!("{NONFINITE_PREFIX}inf") => "inf".to_string(),
        Value::String(value) if value == &format!("{NONFINITE_PREFIX}-inf") => "-inf".to_string(),
        Value::String(value) => value.clone(),
        Value::Array(_) | Value::Object(_) => value.to_string(),
    }
}

fn novelty_keys(state_dir: Option<&Path>) -> Result<BTreeSet<String>> {
    let Some(state_dir) = state_dir else {
        return Ok(BTreeSet::new());
    };
    let archive = state_dir.join("novelty_archive.jsonl");
    if !archive.exists() {
        return Ok(BTreeSet::new());
    }
    let mut keys = BTreeSet::new();
    for row in read_jsonl(&archive)? {
        let object = row
            .as_object()
            .context("novelty archive row must be an object")?;
        for field in ["novelty_hash", "canonical_expr", "dsl_expr"] {
            if let Some(value) = object
                .get(field)
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty() && !s.starts_with(NONFINITE_PREFIX))
            {
                keys.insert(value.to_string());
            }
        }
    }
    Ok(keys)
}

fn get<'a>(value: &'a Value, key: &str) -> Option<&'a Value> {
    value.as_object()?.get(key)
}

fn py_num(value: Option<&Value>) -> Option<f64> {
    match value? {
        Value::Null => None,
        Value::Bool(value) => Some(if *value { 1.0 } else { 0.0 }),
        Value::Number(value) => value.as_f64(),
        Value::String(value) => {
            let value = value.trim();
            if value == format!("{NONFINITE_PREFIX}nan") {
                return None;
            }
            if value == format!("{NONFINITE_PREFIX}inf") {
                return Some(f64::INFINITY);
            }
            if value == format!("{NONFINITE_PREFIX}-inf") {
                return Some(f64::NEG_INFINITY);
            }
            let parsed = value.parse::<f64>().ok()?;
            (!parsed.is_nan()).then_some(parsed)
        }
        Value::Array(_) | Value::Object(_) => None,
    }
}

fn extract_oof(row: &Value) -> Result<Option<f64>> {
    let cv = get(row, "cv").and_then(Value::as_object);
    let metadata = get(row, "metadata")
        .filter(|value| truthy(value))
        .map(|value| {
            value
                .as_object()
                .context("source metadata must be an object")
        })
        .transpose()?;
    let sources = [
        get(row, "oof_r2"),
        cv.and_then(|value| value.get("oof_r2")),
        metadata.and_then(|value| value.get("oof_r2_raw")),
        get(row, "score"),
    ];
    Ok(sources.into_iter().find_map(py_num))
}

fn extract_robust(row: &Value) -> Result<Option<f64>> {
    let metadata = get(row, "metadata")
        .filter(|value| truthy(value))
        .map(|value| {
            value
                .as_object()
                .context("source metadata must be an object")
        })
        .transpose()?;
    let value = metadata
        .and_then(|meta| meta.get("score_robust"))
        .or_else(|| {
            let meta_has_robust =
                metadata.is_some_and(|object| object.contains_key("score_robust"));
            if meta_has_robust {
                None
            } else {
                get(row, "score_robust")
            }
        });
    Ok(py_num(value))
}

fn concept_card(genome: &str, oof: f64, robust: Option<f64>, task_family: &str) -> Value {
    let ops = ops_of(genome);
    let mut scores = Map::new();
    scores.insert("jope_oof_r2".to_string(), number(oof));
    if let Some(value) = robust {
        scores.insert("jope_score_robust".to_string(), number(value));
    }
    json!({
        "concept_id": format!("jope-{}", &sha(genome)[..12]),
        "source_path": "jope:policy_v3",
        "source_sha256": sha(genome),
        "task_family": task_family,
        "operators": ops,
        "canonical_expr": genome,
        "math_summary": "jope-proposed gene (policy_v3)",
        "novelty_tags": ["jope_seed"],
        "complexity": ops_of(genome).len(),
        "applicability": [],
        "failure_modes": [],
        "judge_scores": scores,
        "status": "accepted"
    })
}

fn candidate(genome: &str, oof: f64, concept_id: &str, acq_base: f64, acq_scale: f64) -> Value {
    let clipped = oof.clamp(0.0, 1.0);
    json!({
        "candidate_id": format!("jope-{}", &sha(genome)[..12]),
        "dsl_expr": genome,
        "parent_concepts": [concept_id],
        "complexity": ops_of(genome).len(),
        "mutation": "jope_seed",
        "origin": "jope",
        "generation": 0,
        "parents": [],
        "novelty_hash": "",
        "predicted_score": number(oof),
        "acquisition_score": number(acq_base + acq_scale * clipped)
    })
}

/// Port of `gp_export`: write sorted-by-score ConceptCards and Candidates as JSONL.
pub fn gp_export(
    sources: &[PathBuf],
    out_dir: &Path,
    state_dir: Option<&Path>,
    options: ExportOptions<'_>,
) -> Result<Value> {
    fs::create_dir_all(out_dir).with_context(|| format!("create {}", out_dir.display()))?;
    let novelty = novelty_keys(state_dir)?;
    let mut best: BTreeMap<String, (f64, Option<f64>)> = BTreeMap::new();
    let mut insertion_order = Vec::<String>::new();
    for path in sources {
        if !path.exists() {
            continue;
        }
        for row in read_jsonl(path)? {
            let object = row
                .as_object()
                .context("source JSONL row must be an object")?;
            let metadata = object.get("metadata").filter(|value| truthy(value));
            let engine = object
                .get("engine")
                .filter(|value| truthy(value))
                .or_else(|| {
                    metadata
                        .and_then(|value| get(value, "engine"))
                        .filter(|value| truthy(value))
                })
                .map(py_str)
                .unwrap_or_else(|| "featmath".to_string());
            if engine != options.engine {
                continue;
            }
            let genome_value = ["canonical_genome", "genome", "dsl_expr"]
                .into_iter()
                .filter_map(|key| object.get(key))
                .find(|value| truthy(value));
            let Some(genome) = genome_value
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty() && !s.starts_with(NONFINITE_PREFIX))
            else {
                continue;
            };
            let Some(oof) = extract_oof(&row)? else {
                continue;
            };
            if oof < options.min_oof || novelty.contains(genome) {
                continue;
            }
            let robust = extract_robust(&row)?;
            match best.get_mut(genome) {
                Some(previous) if oof > previous.0 => *previous = (oof, robust),
                Some(_) => {}
                None => {
                    insertion_order.push(genome.to_string());
                    best.insert(genome.to_string(), (oof, robust));
                }
            }
        }
    }
    let mut ranked: Vec<(String, f64, Option<f64>)> = insertion_order
        .into_iter()
        .filter_map(|genome| {
            best.get(&genome)
                .map(|(oof, robust)| (genome, *oof, *robust))
        })
        .collect();
    ranked.sort_by(|left, right| right.1.total_cmp(&left.1));
    let keep = if options.top >= 0 {
        (options.top as usize).min(ranked.len())
    } else {
        ranked
            .len()
            .saturating_sub(options.top.unsigned_abs() as usize)
    };
    ranked.truncate(keep);
    let mut concepts = Vec::with_capacity(ranked.len());
    let mut candidates = Vec::with_capacity(ranked.len());
    for (genome, oof, robust) in &ranked {
        let card = concept_card(genome, *oof, *robust, options.task_family);
        let concept_id = card["concept_id"].as_str().context("concept id absent")?;
        candidates.push(candidate(
            genome,
            *oof,
            concept_id,
            options.acq_base,
            options.acq_scale,
        ));
        concepts.push(card);
    }
    let concepts_path = out_dir.join("concepts.jsonl");
    let candidates_path = out_dir.join("candidates.jsonl");
    write_jsonl(&concepts_path, &concepts)?;
    write_jsonl(&candidates_path, &candidates)?;
    Ok(json!({
        "concepts": concepts_path.to_string_lossy(),
        "candidates": candidates_path.to_string_lossy(),
        "exported": concepts.len(),
        "considered": best.len(),
        "skipped_known": best.keys().filter(|genome| novelty.contains(*genome)).count(),
        "engine": options.engine
    }))
}

fn training_row(
    genome: &str,
    path: &Path,
    elo: Option<f64>,
    delta_lcb: Option<f64>,
    concept: &Value,
    data: &Value,
) -> Value {
    let mut metadata = Map::new();
    metadata.insert("provenance".into(), json!("gp_report"));
    metadata.insert(
        "concept_id".into(),
        get(concept, "concept_id").cloned().unwrap_or(Value::Null),
    );
    metadata.insert(
        "promotion_reason".into(),
        get(data, "promotion_reason")
            .cloned()
            .unwrap_or(Value::Null),
    );
    metadata.insert(
        "aggregate_elo".into(),
        elo.map(number).unwrap_or(Value::Null),
    );
    metadata.insert(
        "aggregate_delta_lcb".into(),
        delta_lcb.map(number).unwrap_or(Value::Null),
    );
    metadata.insert("winner_path".into(), json!(path.to_string_lossy()));
    json!({
        "dataset_id": "gp_winner",
        "source": "gp_report",
        "embedding": vec![0.0; 512],
        "genome": genome,
        "target_embedding": null,
        "canonical_genome": genome,
        "valid": true,
        "metric": null,
        "score": delta_lcb.map(number).unwrap_or(Value::Null),
        "rank": null,
        "elo": elo.map(number).unwrap_or(Value::Null),
        "cost_ms": null,
        "family_bitmap": {},
        "complexity": null,
        "runtime_class": null,
        "feedback_origin": "gp_report",
        "engine": "newmath",
        "metadata": metadata
    })
}

/// Port of `gp_harvest`: read sorted `winner-*.json` inputs and emit TrainingRow JSONL.
pub fn gp_harvest(winners_dir: Option<&Path>, out: &Path) -> Result<Value> {
    let winners_dir = winners_dir.unwrap_or_else(|| Path::new(DEFAULT_WINNERS_DIR));
    let mut paths = Vec::new();
    match fs::read_dir(winners_dir) {
        Ok(entries) => {
            for entry in entries {
                paths.push(entry?.path());
            }
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error).with_context(|| format!("read {}", winners_dir.display())),
    }
    paths.sort();
    let mut seen = BTreeSet::new();
    let mut rows = Vec::new();
    for path in paths {
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or_default();
        if !name.starts_with("winner-") || !name.ends_with(".json") {
            continue;
        }
        let bytes = match fs::read(&path) {
            Ok(bytes) => bytes,
            Err(_) => continue,
        };
        let text = String::from_utf8(bytes)
            .with_context(|| format!("decode {} as UTF-8", path.display()))?;
        let data: Value = match serde_json::from_str(&text) {
            Ok(data) => data,
            Err(_) => continue,
        };
        let object = data.as_object().context("winner JSON must be an object")?;
        let concept = object.get("concept").filter(|value| truthy(value));
        let genome = if let Some(dsl) = object.get("dsl_expr").filter(|value| truthy(value)) {
            dsl.as_str()
        } else if let Some(concept) = concept {
            concept
                .as_object()
                .context("winner concept must be an object")?
                .get("canonical_expr")
                .and_then(Value::as_str)
        } else {
            None
        };
        let Some(genome) =
            genome.filter(|value| !value.is_empty() && !value.starts_with(NONFINITE_PREFIX))
        else {
            continue;
        };
        if !seen.insert(genome.to_string()) {
            continue;
        }
        let receipt = object.get("benchmark_receipt").and_then(Value::as_object);
        let elo = receipt.and_then(|receipt| py_num(receipt.get("aggregate_elo")));
        let delta_lcb = receipt.and_then(|receipt| py_num(receipt.get("aggregate_delta_lcb")));
        let concept = concept
            .filter(|value| value.is_object())
            .cloned()
            .unwrap_or(Value::Object(Map::new()));
        rows.push(training_row(genome, &path, elo, delta_lcb, &concept, &data));
    }
    write_jsonl(out, &rows)?;
    Ok(json!({
        "out": out.to_string_lossy(),
        "harvested": rows.len(),
        "winners_dir": winners_dir.to_string_lossy()
    }))
}

#[cfg(test)]
mod tests;
