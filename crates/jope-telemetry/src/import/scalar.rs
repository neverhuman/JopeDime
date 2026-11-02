use crate::train;
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::path::Path;

pub const SCALAR_SCHEMA: &str = "jope.telemetry.import-scalar/v1";
/// Inputs larger than this are refused rather than read whole.
pub const MAX_INPUT_BYTES: u64 = 256 * 1024 * 1024;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceRef {
    pub source: String,
    pub file_name: String,
    pub file_sha256: String,
}

impl SourceRef {
    pub(crate) fn of(path: &Path, bytes: &[u8]) -> Self {
        Self {
            source: "import".into(),
            file_name: path
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default(),
            file_sha256: Sha256::digest(bytes)
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect(),
        }
    }
}

pub(crate) fn read_bounded(path: &Path) -> Result<Vec<u8>> {
    let len = std::fs::metadata(path)
        .with_context(|| format!("stat {}", path.display()))?
        .len();
    if len > MAX_INPUT_BYTES {
        bail!(
            "{} is {len} bytes, over the {MAX_INPUT_BYTES}-byte import limit",
            path.display()
        );
    }
    std::fs::read(path).with_context(|| format!("read {}", path.display()))
}
/// One imported scalar, keyed to a step or epoch. `metric` is the spec name when the field
/// maps onto a registered metric, otherwise `imported.<json path>`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ImportedScalar {
    pub schema: String,
    pub metric: String,
    pub metric_id: Option<u16>,
    pub group: Option<String>,
    pub step: Option<u64>,
    pub epoch: Option<f64>,
    pub value: f64,
    #[serde(flatten)]
    pub src: SourceRef,
}

/// Field names of `parallel.neural_training.v1` state rows mapped onto spec metrics.
const ENGINE_FIELDS: &[(&str, &str, Option<&str>)] = &[
    ("loss", "train.loss", None),
    ("sequence_nll", "train.loss.component", Some("sequence_nll")),
    ("ranking_bt", "train.loss.component", Some("ranking_bt")),
    (
        "ranking_listwise",
        "train.loss.component",
        Some("ranking_listwise"),
    ),
    (
        "runtime_huber",
        "train.loss.component",
        Some("runtime_huber"),
    ),
    (
        "contextfree_bt",
        "train.loss.component",
        Some("contextfree_bt"),
    ),
    ("gradient_norm", "optim.grad_norm.global", None),
    ("gradient_clipped", "optim.clipped", None),
    ("step_seconds", "step.wall", None),
    ("examples_per_second", "throughput.samples_per_s", None),
    ("telemetry_seconds", "train.existing_telemetry_s", None),
    ("coordination_seconds", "train.coordination_s", None),
    ("communication_seconds", "train.communication_s", None),
    ("checkpoint_seconds", "train.checkpoint_s", None),
];

fn metric_id(name: &str) -> Option<u16> {
    train::METRICS.iter().find(|m| m.name == name).map(|m| m.id)
}

pub(crate) fn num(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => n.as_f64(),
        Value::Bool(b) => Some(*b as u8 as f64),
        _ => None,
    }
}

/// `parallel.neural_training.v1` rows. `step_seconds` is converted to milliseconds to match
/// `step.wall`. Validation panels become `val.loss.component` scalars at `evaluated_step`.
pub fn engine_state_jsonl(path: &Path) -> Result<Vec<ImportedScalar>> {
    let bytes = read_bounded(path)?;
    let src = SourceRef::of(path, &bytes);
    let mut out = Vec::new();
    let mut seen_val = HashSet::new();
    let lines: Vec<&[u8]> = bytes.split(|&b| b == b'\n').collect();
    for (i, line) in lines.iter().enumerate() {
        if line.iter().all(u8::is_ascii_whitespace) {
            continue;
        }
        let row: Map<String, Value> = match serde_json::from_slice(line) {
            Ok(r) => r,
            // A torn last line from a crashed writer is expected; anything else is not.
            Err(_) if i + 1 == lines.len() => continue,
            Err(e) => bail!("{}:{}: {e}", path.display(), i + 1),
        };
        if row.get("schema").and_then(Value::as_str) != Some("parallel.neural_training.v1") {
            continue;
        }
        // The state's `step` field is the 1-based count of completed optimizer steps.
        let step = row.get("step").and_then(Value::as_u64);
        let epoch = row.get("epoch").and_then(Value::as_f64);
        for &(field, metric, group) in ENGINE_FIELDS {
            if let Some(mut v) = row.get(field).and_then(num) {
                if field == "step_seconds" {
                    v *= 1e3;
                }
                out.push(ImportedScalar {
                    schema: SCALAR_SCHEMA.into(),
                    metric: metric.into(),
                    metric_id: metric_id(metric),
                    group: group.map(Into::into),
                    step,
                    epoch,
                    value: v,
                    src: src.clone(),
                });
            }
        }
        if let Some(val) = row.get("validation").and_then(Value::as_object) {
            let at = val.get("evaluated_step").and_then(Value::as_u64);
            // Every state row repeats the latest panel; import each evaluation once.
            if at.is_some_and(|s| seen_val.insert(s)) {
                for (k, v) in val {
                    if k == "evaluated_step" || k == "source_families" {
                        continue;
                    }
                    if let Some(x) = num(v) {
                        out.push(ImportedScalar {
                            schema: SCALAR_SCHEMA.into(),
                            metric: "val.loss.component".into(),
                            metric_id: metric_id("val.loss.component"),
                            group: Some(k.clone()),
                            step: at,
                            epoch: None,
                            value: x,
                            src: src.clone(),
                        });
                    }
                }
            }
        }
    }
    Ok(out)
}

/// Any per-epoch JSONL, such as `train_metrics.jsonl` (rows are free-form maps keyed by
/// `epoch`): every finite numeric leaf becomes `imported.<dotted path>`. Lists are skipped.
pub fn generic_epoch_jsonl(path: &Path) -> Result<Vec<ImportedScalar>> {
    let bytes = read_bounded(path)?;
    let src = SourceRef::of(path, &bytes);
    let mut out = Vec::new();
    let lines: Vec<&[u8]> = bytes.split(|&b| b == b'\n').collect();
    for (i, line) in lines.iter().enumerate() {
        if line.iter().all(u8::is_ascii_whitespace) {
            continue;
        }
        let row: Map<String, Value> = match serde_json::from_slice(line) {
            Ok(r) => r,
            Err(_) if i + 1 == lines.len() => continue,
            Err(e) => bail!("{}:{}: {e}", path.display(), i + 1),
        };
        let epoch = row.get("epoch").and_then(Value::as_f64);
        let step = row.get("step").and_then(Value::as_u64);
        let mut leaves = Vec::new();
        flatten("", &Value::Object(row), &mut leaves);
        for (k, v) in leaves {
            if k == "epoch" || k == "step" || !v.is_finite() {
                continue;
            }
            out.push(ImportedScalar {
                schema: SCALAR_SCHEMA.into(),
                metric: format!("imported.{k}"),
                metric_id: None,
                group: None,
                step,
                epoch,
                value: v,
                src: src.clone(),
            });
        }
    }
    Ok(out)
}

fn flatten(prefix: &str, v: &Value, out: &mut Vec<(String, f64)>) {
    match v {
        Value::Object(m) => {
            for (k, x) in m {
                let p = if prefix.is_empty() {
                    k.clone()
                } else {
                    format!("{prefix}.{k}")
                };
                flatten(&p, x, out);
            }
        }
        _ => {
            if let Some(x) = num(v) {
                out.push((prefix.to_owned(), x));
            }
        }
    }
}
