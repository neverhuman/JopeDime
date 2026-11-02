//! Reporting helpers ported from `jope.validation_gp_study_reporting`.
//!
//! The Python runner remains the caller while its execution and selection
//! paths are migrated. These routines preserve its file formats and decisions.

use anyhow::{anyhow, Context, Result};
use serde_json::{json, Map, Value};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

const COMPARE_FIELDS: [&str; 12] = [
    "dataset",
    "incumbent_model",
    "incumbent_report",
    "incumbent_rmse",
    "single_rmse",
    "ensemble_rmse",
    "single_beat",
    "ensemble_beat",
    "single_delta_abs",
    "ensemble_delta_abs",
    "single_delta_pct",
    "ensemble_delta_pct",
];

/// Return a split-manifest entry after checking all required files.
pub fn dataset_entry<'a>(
    manifest: &'a Map<String, Value>,
    dataset: &str,
) -> Result<&'a Map<String, Value>> {
    let entry = manifest
        .get(dataset)
        .and_then(Value::as_object)
        .ok_or_else(|| anyhow!("{dataset} missing from split_manifest"))?;
    for key in ["train_csv", "test_csv", "vs_path"] {
        let raw = entry
            .get(key)
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| anyhow!("{dataset} missing {key} in split_manifest"))?;
        if !Path::new(raw).exists() {
            return Err(anyhow!("{dataset} {key} does not exist: {raw}"));
        }
    }
    Ok(entry)
}

pub fn model_name(study_tag: &str, dataset: &str, suffix: &str) -> String {
    format!("{study_tag}__{dataset}__{suffix}")
}

pub fn host_slots(hosts: &[String], workers_per_host: usize) -> Vec<(String, usize)> {
    hosts
        .iter()
        .flat_map(|host| (0..workers_per_host).map(move |slot| (host.clone(), slot)))
        .collect()
}

fn metric(row: &Value, key: &str) -> Value {
    row.get(key)
        .and_then(Value::as_object)
        .and_then(|m| m.get("rmse"))
        .cloned()
        .unwrap_or(Value::Null)
}

fn flag(candidate: &Value, incumbent: &Value) -> Value {
    match (candidate.as_f64(), incumbent.as_f64()) {
        (Some(c), Some(i)) if c.is_finite() && i.is_finite() => json!(c < i),
        _ => Value::Null,
    }
}

fn delta(candidate: &Value, incumbent: &Value, percent: bool) -> Value {
    match (candidate.as_f64(), incumbent.as_f64()) {
        (Some(c), Some(i)) if c.is_finite() && i.is_finite() && (!percent || i != 0.0) => {
            json!(if percent { 100.0 * (i - c) / i } else { i - c })
        }
        _ => Value::Null,
    }
}

fn cell(value: &Value) -> String {
    match value {
        Value::Null => String::new(),
        Value::Bool(true) => "True".into(),
        Value::Bool(false) => "False".into(),
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

fn csv_cell(value: &Value) -> String {
    let value = cell(value);
    if value
        .chars()
        .any(|ch| matches!(ch, ',' | '"' | '\n' | '\r'))
    {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value
    }
}

/// Render incumbent comparison rows using Python DictWriter's field order.
pub fn incumbent_compare_csv(results: &[Value]) -> String {
    let mut out = format!("{}\r\n", COMPARE_FIELDS.join(","));
    for row in results {
        let incumbent = row.get("incumbent").unwrap_or(&Value::Null);
        let single = metric(row, "single_primary");
        let ensemble = metric(row, "ensemble_secondary");
        let inc = incumbent.get("rmse").cloned().unwrap_or(Value::Null);
        let values = [
            row.get("dataset").cloned().unwrap_or(Value::Null),
            incumbent.get("model").cloned().unwrap_or(Value::Null),
            incumbent.get("report").cloned().unwrap_or(Value::Null),
            inc.clone(),
            single.clone(),
            ensemble.clone(),
            flag(&single, &inc),
            flag(&ensemble, &inc),
            delta(&single, &inc, false),
            delta(&ensemble, &inc, false),
            delta(&single, &inc, true),
            delta(&ensemble, &inc, true),
        ];
        out.push_str(&values.iter().map(csv_cell).collect::<Vec<_>>().join(","));
        out.push_str("\r\n");
    }
    out
}

pub fn summary(study_tag: &str, out_dir: &Path, compare_path: &Path, results: &[Value]) -> Value {
    let completed = results
        .iter()
        .filter(|row| row.get("status").and_then(Value::as_str) == Some("completed"))
        .count();
    let single_wins = results
        .iter()
        .filter(|row| {
            row.get("status").and_then(Value::as_str) == Some("completed")
                && row.get("beat_single") == Some(&Value::Bool(true))
        })
        .count();
    let ensemble_wins = results
        .iter()
        .filter(|row| {
            row.get("status").and_then(Value::as_str) == Some("completed")
                && row.get("beat_ensemble") == Some(&Value::Bool(true))
        })
        .count();
    json!({
        "study_tag": study_tag,
        "out_dir": out_dir.to_string_lossy(),
        "datasets": results.len(),
        "completed": completed,
        "errors": results.len() - completed,
        "single_primary_wins": single_wins,
        "ensemble_secondary_wins": ensemble_wins,
        "did_single_win_any": single_wins > 0,
        "did_ensemble_win_any": ensemble_wins > 0,
        "incumbent_compare_csv": compare_path.to_string_lossy(),
    })
}

/// Stateful JSON output operations shared by callers that migrate later.
pub struct StudyReporting {
    pub study_tag: String,
    pub out_dir: PathBuf,
    pub compare_path: PathBuf,
    jsonl_lock: Mutex<()>,
}

impl StudyReporting {
    pub fn new(
        study_tag: impl Into<String>,
        out_dir: impl Into<PathBuf>,
        compare_path: impl Into<PathBuf>,
    ) -> Self {
        Self {
            study_tag: study_tag.into(),
            out_dir: out_dir.into(),
            compare_path: compare_path.into(),
            jsonl_lock: Mutex::new(()),
        }
    }

    pub fn write_config(&self, config: &Value) -> Result<()> {
        write_pretty(&self.out_dir.join("config.json"), config)
    }

    pub fn write_incumbent_compare(&self, results: &[Value]) -> Result<()> {
        fs::write(&self.compare_path, incumbent_compare_csv(results))
            .with_context(|| format!("write {}", self.compare_path.display()))
    }

    pub fn write_summary(&self, summary: &Value, results: &[Value]) -> Result<()> {
        write_pretty(
            &self.out_dir.join("summary.json"),
            &json!({"summary": summary, "datasets": results}),
        )?;
        let readme = format!(
            "# {}\n\nValidation GP study with search-only Jain stages and final exact-genome scoring.\n\n- datasets: {}\n- completed: {}\n- single_primary wins: {}\n- ensemble_secondary wins: {}\n\nPrimary choice is `single_primary`; `ensemble_secondary` is reported as the fixed secondary arm.\n",
            self.study_tag,
            summary.get("datasets").unwrap_or(&Value::Null),
            summary.get("completed").unwrap_or(&Value::Null),
            summary.get("single_primary_wins").unwrap_or(&Value::Null),
            summary.get("ensemble_secondary_wins").unwrap_or(&Value::Null),
        );
        fs::write(self.out_dir.join("README.md"), readme)?;
        Ok(())
    }

    pub fn append_jsonl(&self, path: &Path, row: &Value) -> Result<()> {
        fs::create_dir_all(path.parent().unwrap_or(Path::new(".")))?;
        let _guard = self
            .jsonl_lock
            .lock()
            .map_err(|_| anyhow!("JSONL lock poisoned"))?;
        let mut file = OpenOptions::new().create(true).append(true).open(path)?;
        serde_json::to_writer(&mut file, row)?;
        file.write_all(b"\n")?;
        Ok(())
    }
}

fn write_pretty(path: &Path, value: &Value) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = fs::File::create(path)?;
    serde_json::to_writer_pretty(&mut file, value)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_slots_and_summary_match_python_reporting_rules() {
        assert_eq!(model_name("study", "data", "warm60"), "study__data__warm60");
        assert_eq!(
            host_slots(&["h1".into(), "h2".into()], 2),
            vec![
                ("h1".into(), 0),
                ("h1".into(), 1),
                ("h2".into(), 0),
                ("h2".into(), 1)
            ]
        );
        let rows = vec![
            json!({"status":"completed", "beat_single":true, "beat_ensemble":false}),
            json!({"status":"error"}),
        ];
        let value = summary("study", Path::new("out"), Path::new("compare.csv"), &rows);
        assert_eq!(value["datasets"], 2);
        assert_eq!(value["completed"], 1);
        assert_eq!(value["errors"], 1);
        assert_eq!(value["single_primary_wins"], 1);
        assert_eq!(value["ensemble_secondary_wins"], 0);
    }

    #[test]
    fn comparison_csv_keeps_field_order_nulls_and_python_boolean_text() {
        let rows = vec![
            json!({"dataset":"d,1", "incumbent":{"model":"m", "rmse":2.0}, "single_primary":{"rmse":1.0}, "ensemble_secondary":{"rmse":null}}),
        ];
        let csv = incumbent_compare_csv(&rows);
        assert!(csv.starts_with("dataset,incumbent_model,incumbent_report,incumbent_rmse,"));
        assert!(csv.contains("\"d,1\",m,,2.0,1.0,,True,,1.0,,,\r\n"));
    }

    #[test]
    fn missing_manifest_entry_preserves_error_wording() {
        let manifest = Map::new();
        assert_eq!(
            dataset_entry(&manifest, "d").unwrap_err().to_string(),
            "d missing from split_manifest"
        );
    }
}
