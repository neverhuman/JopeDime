//! Task identity and primary scores. Regression RMSE stays the compatibility path.
use anyhow::{ensure, Context, Result};
use serde_json::{json, Value};

/// Recognized metadata names only; this is not executable qualification.
pub const TASKS: [&str; 4] = ["regression", "binary", "multiclass", "timeseries"];

/// Recognizes declared metadata; callers that execute or admit must use require_task.
pub fn supported_task(task: &str) -> bool {
    TASKS.contains(&task)
}

pub fn require_task(task: &str) -> Result<()> {
    ensure!(supported_task(task), "unsupported task {task}");
    ensure!(task == "regression", "task {task} is not qualified for parallel execution or teaching; regression only");
    Ok(())
}

/// Lower-is-better primary used for ranking, packing, and teacher comparison.
pub fn primary_metric(task: &str) -> Result<&'static str> {
    require_task(task)?;
    Ok(match task {
        "regression" => "rmse",
        "binary" | "multiclass" => "logloss",
        "timeseries" => "mase",
        other => anyhow::bail!("unsupported task {other}"),
    })
}

pub fn metric_units(task: &str) -> Result<&'static str> {
    require_task(task)?;
    Ok(match task {
        "regression" | "timeseries" => "original_target",
        "binary" | "multiclass" => "class_index",
        other => anyhow::bail!("unsupported task {other}"),
    })
}

pub fn cut_units(task: &str) -> Result<&'static str> {
    require_task(task)?;
    Ok(match task {
        "regression" | "timeseries" => "cut_normalized_target",
        "binary" | "multiclass" => "class_index",
        other => anyhow::bail!("unsupported task {other}"),
    })
}

pub fn observation_primary(metrics: &Value) -> Result<f64> {
    // Legacy regression rows may omit these fields. Present declarations must
    // agree; never reinterpret a non-regression primary as an RMSE label.
    if let Some(task) = metrics.get("task") {
        require_task(task.as_str().context("invalid metric task")?)?;
    }
    if let Some(metric) = metrics.get("primary_metric") {
        ensure!(metric.as_str() == Some("rmse"), "non-RMSE primary is not qualified");
    }
    if let Some(units) = metrics.get("units") {
        ensure!(matches!(units.as_str(), Some("original_target" | "cut_normalized_target")), "non-regression metric units are not qualified");
    }
    ensure!(["logloss", "auc", "brier", "accuracy", "mase", "mase_denominator"].iter().all(|field| metrics.get(*field).is_none()),
        "non-regression metric evidence is not qualified");
    let rmse = metrics["rmse"].as_f64().context("RMSE absent")?;
    ensure!(rmse.is_finite() && rmse >= 0.0, "invalid score");
    if let Some(primary) = metrics.get("primary") {
        let value = primary.as_f64().context("invalid primary")?;
        ensure!(value.is_finite() && value >= 0.0 && value == rmse, "primary disagrees with RMSE");
    }
    Ok(rmse)
}

pub fn score_predictions(
    task: &str,
    actual: &[f64],
    predicted: &[f64],
    mase_denominator: Option<f64>,
) -> Result<Value> {
    require_task(task)?;
    ensure!(
        !actual.is_empty()
            && actual.len() == predicted.len()
            && actual.iter().chain(predicted).all(|x| x.is_finite()),
        "invalid metric inputs"
    );
    let n = actual.len() as f64;
    let mean = actual.iter().sum::<f64>() / n;
    let sse = actual
        .iter()
        .zip(predicted)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>();
    let sst = actual.iter().map(|a| (a - mean).powi(2)).sum::<f64>();
    let rmse = (sse / n).sqrt();
    let mae = actual
        .iter()
        .zip(predicted)
        .map(|(a, b)| (a - b).abs())
        .sum::<f64>()
        / n;
    let mut row = json!({
        "rmse": rmse,
        "mae": mae,
        "r2": if sst > 0.0 { Some(1.0 - sse / sst) } else { None },
        "r2_status": if sst > 0.0 { "defined" } else { "undefined_constant_target" },
        "rows": actual.len(),
        "task": task,
        "units": cut_units(task)?,
        "primary_metric": primary_metric(task)?,
    });
    match task {
        "regression" => {
            row["primary"] = json!(rmse);
        }
        "binary" => {
            let logloss = feat_math::logloss(actual, predicted);
            ensure!(logloss.is_finite() && logloss >= 0.0, "invalid logloss");
            row["logloss"] = json!(logloss);
            row["auc"] = json!(feat_math::auc(actual, predicted));
            row["brier"] = json!(feat_math::brier(actual, predicted));
            row["primary"] = json!(logloss);
        }
        "multiclass" => {
            let logloss = multiclass_logloss(actual, predicted);
            ensure!(logloss.is_finite() && logloss >= 0.0, "invalid logloss");
            row["logloss"] = json!(logloss);
            row["accuracy"] = json!(multiclass_accuracy(actual, predicted));
            row["primary"] = json!(logloss);
        }
        "timeseries" => {
            let denom = mase_denominator.context("timeseries MASE denominator required")?;
            ensure!(denom.is_finite() && denom > 0.0, "invalid MASE denominator");
            let mase = mae / denom;
            row["mase"] = json!(mase);
            row["mase_denominator"] = json!(denom);
            row["primary"] = json!(mase);
        }
        _ => unreachable!(),
    }
    Ok(row)
}

// Unqualified legacy prototype retained for source review; require_task makes it unreachable.
fn multiclass_logloss(actual: &[f64], predicted: &[f64]) -> f64 {
    let classes = actual
        .iter()
        .map(|y| y.round() as i64)
        .filter(|&c| c >= 0)
        .max()
        .unwrap_or(0)
        + 1;
    let eps = feat_math::LOGLOSS_EPS;
    let mut loss = 0.0;
    for (y, p) in actual.iter().zip(predicted) {
        let cls = y.round() as i64;
        let prob = if classes <= 2 {
            let q = p.clamp(eps, 1.0 - eps);
            if cls == 1 {
                q
            } else {
                1.0 - q
            }
        } else {
            // Isolated numeric head: treat |y - p| as a one-vs-rest confidence.
            let dist = (y - p).abs();
            (1.0 / (1.0 + dist)).clamp(eps, 1.0 - eps)
        };
        loss -= prob.ln();
    }
    loss / actual.len() as f64
}

fn multiclass_accuracy(actual: &[f64], predicted: &[f64]) -> f64 {
    let hits = actual
        .iter()
        .zip(predicted)
        .filter(|(y, p)| y.round() == p.round())
        .count();
    hits as f64 / actual.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regression_primary_is_rmse_and_zero_is_legal() {
        let m = score_predictions("regression", &[1.0, 2.0], &[1.0, 2.0], None).unwrap();
        assert_eq!(m["primary_metric"], "rmse");
        assert_eq!(m["primary"].as_f64(), Some(0.0));
        assert_eq!(observation_primary(&m).unwrap(), 0.0);
    }

    #[test]
    fn binary_execution_is_not_qualified() {
        assert!(score_predictions("binary", &[0.0, 1.0, 1.0], &[0.1, 0.9, 0.8], None).is_err());
    }

    #[test]
    fn timeseries_execution_is_not_qualified() {
        assert!(score_predictions("timeseries", &[1.0, 2.0], &[1.0, 3.0], None).is_err());
        assert!(score_predictions("timeseries", &[1.0, 3.0], &[1.0, 1.0], Some(2.0)).is_err());
    }

    #[test]
    fn unsupported_task_is_rejected() {
        assert!(require_task("multi_label").is_err());
        assert!(!supported_task("routing"));
    }
}
