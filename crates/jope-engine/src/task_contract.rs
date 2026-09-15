//! Task identity and primary scores. Regression RMSE stays the compatibility path.
use anyhow::{ensure, Context, Result};
use serde_json::{json, Value};

/// Recognized metadata names only; this is not executable qualification.
pub const TASKS: [&str; 4] = ["regression", "binary", "multiclass", "timeseries"];

/// Rank-independent metadata shared by the CPU authority and resident clients.
/// Deserializing this descriptor grants no execution and loads no CUDA code.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CommonTaskDescriptor {
    pub schema: String,
    pub root: std::path::PathBuf,
    pub task: Value,
    pub environment: std::collections::BTreeMap<String, String>,
    pub input_artifacts: std::collections::BTreeMap<std::path::PathBuf, String>,
    pub executable_sha256: String,
    pub deadline_unix_ms: u64,
}

/// Version the existing partition-dependent objective; changing the partition
/// is an explicit experiment, not an exact-resume optimization.
pub fn training_partition(world: usize, micro: usize, global: usize) -> Result<Value> {
    ensure!(
        [1, 2, 4, 8].contains(&world)
            && micro > 0
            && global > 0
            && micro
                .checked_mul(world)
                .is_some_and(|batch| global % batch == 0),
        "invalid training partition"
    );
    Ok(
        json!({"schema":"parallel.training_partition.v1","world":world,
        "microbatch":micro,"global_batch":global,"accumulation":global / (world * micro),
        "rng":"mix_seed_local_step_rank_microbatch_v1",
        "normalization":"microbatch_weighted_means_rank_mean_v1",
        "autocast":"scoped_bf16_fp32_master_v2"}),
    )
}

/// CPU-only gate; call before creating a CUDA device, tensor or communicator.
pub fn verify_resume_metadata(state: &Value, expected: &Value, optimizer: &Value) -> Result<()> {
    for (field, value) in expected
        .as_object()
        .context("resume expectation must be an object")?
    {
        ensure!(
            !value.is_null() && state.get(field) == Some(value),
            "resume metadata changed: {field}"
        );
    }
    ensure!(
        state["step"].as_u64().is_some()
            && optimizer["step"] == state["step"]
            && optimizer["sha256"] == state["optimizer_sha256"],
        "resume optimizer/model identity mismatch"
    );
    Ok(())
}

/// An original caller-bound deadline is required; this does not grant funding.
pub fn gpu_task_deadline_ms() -> Result<u64> {
    let now: u64 = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_millis()
        .try_into()?;
    let deadline = match std::env::var("JOPE_GPU_TASK_DEADLINE_MS") {
        Ok(value) => value.parse()?,
        Err(std::env::VarError::NotPresent) => {
            anyhow::bail!("GPU task requires its original absolute deadline")
        }
        Err(error) => return Err(error.into()),
    };
    ensure!(deadline > now, "GPU task absolute deadline elapsed");
    Ok(deadline)
}

/// Diagnostic/experiment routes have no qualified native grant and resource
/// contract yet. Environment flags, paths and lease possession cannot grant it.
pub fn require_qualified_gpu_experiment(route: &str) -> Result<()> {
    require_original_gpu_grant(route)
}

/// Shared dispatch hold, not a positive grant verifier. No caller-controlled
/// environment, lease, manifest or Boolean can release numerical work here.
pub fn require_original_gpu_grant(route: &str) -> Result<()> {
    anyhow::bail!(
        "{route}: original native GPU grant verifier unavailable; numerical dispatch is held"
    )
}

/// Grant-only gate for a qualified native task. The opaque admission can only
/// be constructed by the protected native client after it verifies the
/// original authority/witness proof. A serialized receipt is insufficient.
///
/// The caller must separately verify its hold revision, device lease, immutable
/// task inputs and runtime contract before CUDA and keep this proof private for
/// the task lifetime. Nested checks reuse the proof; they never consume another
/// start or participant slot. Unsupported route-only calls remain held above.
pub fn verify_gpu_task_grant(
    admission: &crate::eval_registry::service::VerifiedFundingAdmission,
    descriptor: &Value,
    participant_slot: Option<u32>,
    expected_capability: &str,
) -> Result<()> {
    admission.check_current(descriptor, participant_slot)?;
    ensure!(
        !expected_capability.is_empty()
            && admission.receipt().event.demand.capability == expected_capability,
        "original GPU grant does not authorize this native task capability"
    );
    verify_resident_funding_demand(descriptor, &admission.receipt().event.demand)?;
    Ok(())
}

/// Check the actual native task against its separately signed allocation. A
/// signed four-device allowance cannot fund an eight-rank descriptor merely
/// because both envelopes have internally consistent hashes.
pub fn verify_resident_funding_demand(
    descriptor: &Value,
    demand: &crate::eval_registry::funding::TaskDemand,
) -> Result<()> {
    let common: CommonTaskDescriptor = serde_json::from_value(descriptor.clone())?;
    ensure!(
        common.schema == "parallel.resident_common_task.v1"
            && common.executable_sha256 == demand.executable_sha256
            && common.deadline_unix_ms == demand.deadline_unix_ms,
        "native task executable/deadline differs from its signed funding demand"
    );
    let (capability, world) = match common.task["kind"].as_str() {
        Some("Train") => (
            "regression_training",
            common.task["world"]
                .as_u64()
                .context("training world absent")?,
        ),
        Some("Proposal") => (
            "regression_proposals",
            common.task["world"]
                .as_u64()
                .context("proposal world absent")?,
        ),
        Some("Monitor") => ("regression_monitoring", 1),
        _ => anyhow::bail!("native task kind has no qualified funding-demand binding"),
    };
    ensure!(
        demand.capability == capability
            && u64::from(demand.resources.world) == world
            && u64::from(demand.resources.gpus) == world
            && demand.resources.gpu_devices.len() as u64 == world,
        "native task kind/world differs from its signed resource allowance"
    );
    Ok(())
}

/// Recognizes declared metadata; callers that execute or admit must use require_task.
pub fn supported_task(task: &str) -> bool {
    TASKS.contains(&task)
}

pub fn require_task(task: &str) -> Result<()> {
    ensure!(supported_task(task), "unsupported task {task}");
    ensure!(
        task == "regression",
        "task {task} is not qualified for parallel execution or teaching; regression only"
    );
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
        ensure!(
            metric.as_str() == Some("rmse"),
            "non-RMSE primary is not qualified"
        );
    }
    if let Some(units) = metrics.get("units") {
        ensure!(
            matches!(
                units.as_str(),
                Some("original_target" | "cut_normalized_target")
            ),
            "non-regression metric units are not qualified"
        );
    }
    ensure!(
        [
            "logloss",
            "auc",
            "brier",
            "accuracy",
            "mase",
            "mase_denominator"
        ]
        .iter()
        .all(|field| metrics.get(*field).is_none()),
        "non-regression metric evidence is not qualified"
    );
    let rmse = metrics["rmse"].as_f64().context("RMSE absent")?;
    ensure!(rmse.is_finite() && rmse >= 0.0, "invalid score");
    if let Some(primary) = metrics.get("primary") {
        let value = primary.as_f64().context("invalid primary")?;
        ensure!(
            value.is_finite() && value >= 0.0 && value == rmse,
            "primary disagrees with RMSE"
        );
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
    #[test]
    fn native_world_and_kind_cannot_outgrow_signed_funding() {
        use crate::eval_registry::funding::{ResourceCaps, TaskDemand};
        let mut demand = TaskDemand {
            task_sha256: "a".repeat(64),
            capability: "regression_training".into(),
            host_id: "synthetic-host".into(),
            boot_id: "synthetic-boot".into(),
            executable_sha256: "b".repeat(64),
            runtime_sha256: "c".repeat(64),
            deadline_unix_ms: 1000,
            deadline_monotonic_ns: 1_000_000_000,
            resources: ResourceCaps {
                cpu_cores: 8,
                gpus: 4,
                world: 4,
                host_memory_bytes: 1024,
                gpu_memory_bytes_per_rank: 1024,
                cpu_searches: 0,
                gpu_devices: vec![0, 1, 2, 3],
            },
        };
        let descriptor = serde_json::json!({"schema":"parallel.resident_common_task.v1",
            "root":"/private-fixture","task":{"kind":"Train","world":8},
            "environment":{},"input_artifacts":{},"executable_sha256":"b".repeat(64),
            "deadline_unix_ms":1000});
        assert!(super::verify_resident_funding_demand(&descriptor, &demand).is_err());
        demand.resources.world = 8;
        demand.resources.gpus = 8;
        demand.resources.gpu_devices = (0..8).collect();
        super::verify_resident_funding_demand(&descriptor, &demand).unwrap();
        for field in ["kind", "deadline", "executable"] {
            let mut changed = descriptor.clone();
            match field {
                "kind" => changed["task"]["kind"] = serde_json::json!("Proposal"),
                "deadline" => changed["deadline_unix_ms"] = serde_json::json!(1001),
                _ => changed["executable_sha256"] = serde_json::json!("d".repeat(64)),
            }
            assert!(super::verify_resident_funding_demand(&changed, &demand).is_err());
        }
    }
    use super::*;

    #[test]
    fn experiment_metadata_cannot_grant_missing_native_gpu_qualification() {
        for route in ["qualify-batch-rank", "propose-ranked"] {
            assert!(require_qualified_gpu_experiment(route).is_err());
        }
    }

    #[test]
    fn exact_training_partition_binds_microbatch_and_accumulation() {
        let original = training_partition(8, 256, 2048).unwrap();
        assert_eq!(original["accumulation"], 1);
        assert_ne!(original, training_partition(8, 128, 2048).unwrap());
        assert_ne!(original, training_partition(4, 256, 2048).unwrap());
        assert!(training_partition(8, 0, 2048).is_err());
        assert!(training_partition(8, usize::MAX, 2048).is_err());
    }

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
    #[test]
    fn resume_partition_and_optimizer_step_reject_before_any_runtime_is_needed() {
        let expected = json!({"execution_partition":training_partition(8, 256, 2048).unwrap(),"sampler_id":"fixed"});
        let mut state = expected.clone();
        state["step"] = json!(9);
        state["optimizer_sha256"] = json!("a".repeat(64));
        let mut optimizer = json!({"step":9,"sha256":"a".repeat(64)});
        verify_resume_metadata(&state, &expected, &optimizer).unwrap();
        state["execution_partition"] = training_partition(8, 128, 2048).unwrap();
        assert!(verify_resume_metadata(&state, &expected, &optimizer).is_err());
        state["execution_partition"] = expected["execution_partition"].clone();
        optimizer["step"] = json!(8);
        assert!(verify_resume_metadata(&state, &expected, &optimizer).is_err());
    }
}
