//! Cost estimates from quant reports or the historical default sample.

use serde_json::{json, Map, Number, Value};
use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
};

const DEFAULT_EVALS: u64 = 11_370;
const DEFAULT_SEARCH_SECS: f64 = 300.0;
const DEFAULT_GPU_KERNEL_SECS: f64 = 33.0;
const DEFAULT_FINALIST_CV_SECS: f64 = 222.0;
const DEFAULT_TOTAL_SECS: f64 = 523.0;
const TIMING_KEYS: [&str; 4] = [
    "search_secs",
    "gpu_kernel_secs",
    "finalist_cv_secs",
    "total_secs",
];

/// Estimates CPU/GPU seconds and optional hourly cost from up to 256 quant reports.
pub fn estimate_costs(
    quant_root: Option<&Path>,
    gpu_hourly_usd: Option<f64>,
    cpu_hourly_usd: Option<f64>,
) -> Value {
    let observed = quant_root.and_then(observed_from_quant);
    let (source, base) = match observed.as_ref() {
        Some(base) => ("observed_quant_reports", base),
        None => ("default_seed_constants", &Value::Null),
    };
    let observed_base = observed.is_some();
    let evals = observed
        .as_ref()
        .and_then(|base| base.get("evals"))
        .cloned()
        // default fallback
        .unwrap_or_else(|| {
            // default fallback
            json!(DEFAULT_EVALS)
        });
    let search_secs = number_or(
        base.get("search_secs"),
        if observed_base {
            0.0
        } else {
            DEFAULT_SEARCH_SECS
        },
    );
    let gpu_secs = number_or(
        base.get("gpu_kernel_secs"),
        if observed_base {
            0.0
        } else {
            DEFAULT_GPU_KERNEL_SECS
        },
    );
    let finalist_cv_secs = number_or(
        base.get("finalist_cv_secs"),
        if observed_base {
            0.0
        } else {
            DEFAULT_FINALIST_CV_SECS
        },
    );
    let total_secs = number_or(
        base.get("total_secs"),
        if observed_base {
            0.0
        } else {
            DEFAULT_TOTAL_SECS
        },
    );
    let cpu_secs = (total_secs - gpu_secs).max(0.0);
    let mut out = json!({
        "source": source,
        "evals": evals,
        "search_secs": search_secs,
        "gpu_kernel_secs": gpu_secs,
        "finalist_cv_secs": finalist_cv_secs,
        "total_secs": total_secs,
        "local_compute_units": {
            "cpu_seconds": cpu_secs,
            "gpu_seconds": gpu_secs,
        },
    });
    if gpu_hourly_usd.is_some() || cpu_hourly_usd.is_some() {
        let cpu = cpu_secs / 3600.0 * cpu_hourly_usd.unwrap_or(0.0);
        let gpu = gpu_secs / 3600.0 * gpu_hourly_usd.unwrap_or(0.0);
        out["usd"] = json!({"cpu": cpu, "gpu": gpu, "total": cpu + gpu});
    }
    out
}

fn number_or(value: Option<&Value>, default: f64) -> f64 {
    value.and_then(python_number).unwrap_or(default)
}

// Python treats bool as an int for isinstance(value, (int, float)).
fn python_number(value: &Value) -> Option<f64> {
    match value {
        Value::Number(number) => number.as_f64(),
        Value::Bool(value) => Some(if *value { 1.0 } else { 0.0 }),
        _ => None,
    }
}

fn observed_from_quant(root: &Path) -> Option<Value> {
    let mut paths = BTreeSet::new();
    collect_report_paths(&root.join("regression"), true, &mut paths);
    collect_report_paths(root, false, &mut paths);

    let mut totals = Map::<String, Value>::new();
    let mut count = 0usize;
    let mut evals = 0.0;
    for path in paths.into_iter().take(256) {
        let Ok(bytes) = fs::read(path) else { continue };
        let Ok(data) = serde_json::from_slice::<Value>(&bytes) else {
            continue;
        };
        let Some(timings) = data.get("timings").and_then(Value::as_object) else {
            continue;
        };
        if timings.is_empty() {
            continue;
        }
        count += 1;
        for key in TIMING_KEYS {
            if let Some(value) = timings.get(key).and_then(python_number) {
                let prior = totals.get(key).and_then(python_number).unwrap_or(0.0);
                if let Some(sum) = Number::from_f64(prior + value) {
                    totals.insert(key.to_owned(), Value::Number(sum));
                }
            }
        }
        if let Some(value) = data
            .get("model_params")
            .and_then(|params| params.get("evals"))
            .and_then(python_number)
        {
            evals += value;
        }
    }
    if count == 0 {
        return None;
    }
    let mut observed = Map::new();
    for (key, total) in totals {
        if let Some(value) = python_number(&total) {
            if let Some(mean) = Number::from_f64(value / count as f64) {
                observed.insert(key, Value::Number(mean));
            }
        }
    }
    let average_evals = if evals != 0.0 {
        Some((evals / count as f64) as i64)
    } else {
        None
    };
    observed.insert(
        "evals".to_owned(),
        average_evals.map_or(Value::Null, |value| json!(value)),
    );
    observed.insert("reports".to_owned(), json!(count));
    Some(Value::Object(observed))
}

fn collect_report_paths(base: &Path, regression_pattern: bool, paths: &mut BTreeSet<PathBuf>) {
    let entries = match fs::read_dir(base) {
        Ok(entries) => entries,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let first = entry.path();
        if regression_pattern && !first.is_dir() {
            continue;
        }
        let report = first.join("reports").join("jain_regressor_300.json");
        if report.is_file() {
            paths.insert(report);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::estimate_costs;
    use serde_json::json;
    use std::{fs, path::Path};

    #[test]
    fn default_sample_and_optional_usd_match_python_contract() {
        let estimate = estimate_costs(None, Some(36.0), Some(18.0));
        assert_eq!(estimate["source"], "default_seed_constants");
        assert_eq!(estimate["evals"], 11_370);
        assert_eq!(estimate["local_compute_units"]["cpu_seconds"], 490.0);
        assert_eq!(estimate["local_compute_units"]["gpu_seconds"], 33.0);
        assert_eq!(estimate["usd"]["cpu"], 490.0 / 3600.0 * 18.0);
        assert_eq!(estimate["usd"]["gpu"], 33.0 / 3600.0 * 36.0);
    }

    #[test]
    fn averages_sorted_deduplicated_quant_report_patterns() {
        let root = tempfile::tempdir().unwrap();
        write_report(root.path(), "regression/a", 10, 100.0, 20.0, 150.0);
        write_report(root.path(), "b", 20, 200.0, 40.0, 300.0);
        let estimate = estimate_costs(Some(root.path()), None, None);
        assert_eq!(estimate["source"], "observed_quant_reports");
        assert_eq!(estimate["evals"], 15);
        assert_eq!(estimate["search_secs"], 150.0);
        assert_eq!(estimate["gpu_kernel_secs"], 30.0);
        assert_eq!(estimate["total_secs"], 225.0);
        assert_eq!(estimate["local_compute_units"]["cpu_seconds"], 195.0);
    }

    fn write_report(root: &Path, relative: &str, evals: u64, search: f64, gpu: f64, total: f64) {
        let path = root.join(relative).join("reports/jain_regressor_300.json");
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(
            path,
            serde_json::to_vec(&json!({
                "timings": {"search_secs": search, "gpu_kernel_secs": gpu, "total_secs": total},
                "model_params": {"evals": evals}
            }))
            .unwrap(),
        )
        .unwrap();
    }
}
