//! Execute pending three-lane canaries through pinned proposal and evaluator jobs.

use anyhow::{bail, ensure, Context, Result};
use clap::Parser;
use jope_dime_regressor::paths;
use jope_dime_regressor::selfplay_ingest::ingest_receipt_with_manifest;
use jope_dime_regressor::selfplay_runtime::{
    atomic_json, checkpoint_epoch, default_root_str, default_scq_str, evaluation_protocol_sha256,
    expected_evaluation_metric, now_ms, read_json, sha256_bytes, sha256_file,
};
use jope_dime_regressor::selfplay_shadow::{
    evaluate_task_promotion_v3, CanaryContextClass, CanaryContextV2, CanaryIntentV2, CanaryMode,
    PromotionObservationV3, SealedPanelConsumptionV1, SealedPanelContextFactsV1, SealedPanelPoolV2,
    Sha256Digest, Task, TaskCanaryAcceptanceV2, TaskCanaryMetricsV2,
    TaskPromotionEvaluationIntentV3, TaskPromotionEvaluationV3, TaskPromotionObservationSetV3,
    CANARY_INTENT_SCHEMA, PROMOTION_CONTEXT_STEP, SEALED_PANEL_CONTEXT_FACTS_V1_SCHEMA,
    TASK_CANARY_ACCEPTANCE_V2_SCHEMA, TASK_PROMOTION_OBSERVATION_SET_V3_SCHEMA,
};
use jope_dime_regressor::spec_policy::{EVAL, PROPOSE};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::thread;
use std::time::Duration;

mod intent;
mod jobs;
mod metrics;
pub(super) use intent::*;
pub(super) use jobs::*;
pub(super) use metrics::*;

const ACTIVE_STATES: &[&str] = &["queued", "starting", "dispatching", "running", "held"];

#[derive(Parser, Debug)]
#[command(name = "jope-canary-executor")]
struct Args {
    #[arg(long, default_value = default_root_str())]
    root: PathBuf,
    #[arg(long, default_value = default_scq_str())]
    scq: PathBuf,
    #[arg(
        long,
        default_value_os_t = paths::fweval_runner()
    )]
    runner: PathBuf,
    #[arg(long)]
    once: bool,
    #[arg(long, default_value_t = 30)]
    interval_seconds: u64,
}

#[derive(Clone, Debug)]
struct PendingIntent {
    path: PathBuf,
    value: CanaryIntentV2,
    created_at_ms: u64,
    promotion: Option<TaskPromotionEvaluationIntentV3>,
}

#[derive(Clone, Debug, Default)]
struct LaneMetrics {
    requested: usize,
    valid: usize,
    scores: Vec<f64>,
    runtimes: Vec<f64>,
}

#[derive(Clone, Debug, Default)]
struct AggregateMetrics {
    candidate: LaneMetrics,
    accepted: LaneMetrics,
    gp: LaneMetrics,
    candidate_novel: usize,
    strict_wins: usize,
    context_gaps: Vec<(CanaryContextClass, f64)>,
    candidate_pairwise_gaps: Vec<f64>,
    gp_pairwise_gaps: Vec<f64>,
    candidate_context_comparison_gaps: Vec<f64>,
    gp_context_comparison_gaps: Vec<f64>,
    /// Contexts excluded from the order-statistic lift estimate because the two
    /// arms returned UNEQUAL numbers of valid scores. Zipping sorted survivor
    /// vectors of unequal length silently truncates to the shorter one -- so a
    /// candidate that FAILED on its weakest proposals was compared only on its
    /// surviving top scores, and failure could *improve* measured lift. An
    /// incomplete context is reported, never silently averaged.
    incomplete_accepted_contexts: usize,
    incomplete_gp_contexts: usize,
}

fn consume_intent(args: &Args, pending: &PendingIntent) -> Result<Value> {
    if pending.promotion.is_some() {
        return consume_promotion_intent(args, pending);
    }
    if !predecessor_passed(&args.root, pending)? {
        return Ok(json!({
            "action": "waiting_for_prior_production_canary",
            "task": pending.value.task,
            "candidate": pending.value.candidate_checkpoint_sha256,
            "run_index": pending.value.run_index,
        }));
    }
    let intent_hash = sha256_file(&pending.path)?;
    let work = args
        .root
        .join("training/canary-executor/work")
        .join(&intent_hash);
    fs::create_dir_all(&work)?;
    let jobs = scq_jobs(args)?;
    let Some(proposal_set) = ensure_proposals(args, pending, &work, &jobs)? else {
        return Ok(json!({
            "action": "proposal_job_pending",
            "intent_sha256": intent_hash,
        }));
    };
    let evaluations = build_evaluations(
        args,
        pending,
        &work,
        &proposal_set,
        pending.value.contexts.len(),
    )?;
    let jobs = scq_jobs(args)?;
    let Some(receipts) = ensure_evaluations(args, &work, &evaluations, &jobs)? else {
        return Ok(json!({
            "action": "evaluation_jobs_pending",
            "intent_sha256": intent_hash,
            "evaluation_jobs": evaluations.len(),
        }));
    };
    let summary = summarize(args, pending, &proposal_set, &receipts)?;
    let completed_root = match pending.value.mode {
        CanaryMode::Shadow => args.root.join("training/shadow-canaries"),
        CanaryMode::Production => args.root.join("training/task-canaries"),
    };
    let summary_path = completed_root
        .join("completed")
        .join(format!("{intent_hash}.summary.v1.json"));
    atomic_json(&summary_path, &summary)?;
    let intent_destination = completed_root.join("intents").join(
        pending
            .path
            .file_name()
            .context("canary intent has no filename")?,
    );
    fs::create_dir_all(
        intent_destination
            .parent()
            .context("intent destination has no parent")?,
    )?;
    fs::rename(&pending.path, &intent_destination)?;
    Ok(json!({
        "action": "completed",
        "intent_sha256": intent_hash,
        "summary": summary_path,
        "consumed_intent": intent_destination,
    }))
}

fn cycle(args: &Args) -> Result<Value> {
    let pending = pending_intents(&args.root)?;
    for intent in &pending {
        if intent.value.mode == CanaryMode::Production && !predecessor_passed(&args.root, intent)? {
            continue;
        }
        let result = consume_intent(args, intent)?;
        return Ok(json!({
            "schema": "CanaryExecutorStatus.v2",
            "generated_at_ms": now_ms(),
            "pending_intents": pending.len(),
            "result": result,
        }));
    }
    Ok(json!({
        "schema": "CanaryExecutorStatus.v2",
        "generated_at_ms": now_ms(),
        "pending_intents": pending.len(),
        "result": {"action": "idle"},
    }))
}

fn transient_capacity_error(message: &str) -> bool {
    message.contains("dependency-unsatisfied")
        && ((message.contains("GPU capacity") && message.contains("smoke test exited with exit 6"))
            || message.contains("no eligible launch-ready node currently"))
}

fn main() -> Result<()> {
    let args = Args::parse();
    let mut previous_error = String::new();
    let mut identical_failures = 0_u32;
    loop {
        let status = match cycle(&args) {
            Ok(status) => {
                previous_error.clear();
                identical_failures = 0;
                status
            }
            Err(error) => {
                let message = format!("{error:#}");
                let capacity_deferred = transient_capacity_error(&message);
                if capacity_deferred {
                    previous_error.clear();
                    identical_failures = 0;
                } else if message == previous_error {
                    identical_failures = identical_failures.saturating_add(1);
                } else {
                    previous_error = message.clone();
                    identical_failures = 1;
                }
                json!({
                    "schema": "CanaryExecutorStatus.v2",
                    "generated_at_ms": now_ms(),
                    "result": {
                        "action": if capacity_deferred { "deferred_capacity" } else { "error" },
                        "error": message,
                        "error_class": if capacity_deferred { "transient_capacity" } else { "permanent_or_unknown" },
                        "identical_failures": identical_failures,
                        "critical": !capacity_deferred && identical_failures >= 3,
                    },
                })
            }
        };
        atomic_json(&args.root.join("ops/canary-executor.v2.json"), &status)?;
        println!("{}", serde_json::to_string(&status)?);
        if args.once {
            return Ok(());
        }
        let backoff = if identical_failures == 0 {
            args.interval_seconds
        } else {
            args.interval_seconds
                .saturating_mul(1_u64 << identical_failures.saturating_sub(1).min(4))
                .min(120)
        };
        thread::sleep(Duration::from_secs(backoff));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn v2_intent_accepts_creation_timestamp_emitted_by_controller() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("intent.v2.json");
        let contexts = (0..16)
            .map(|index| {
                json!({
                    "context_identity": format!("context-{index}"),
                    "class": if index < 12 {
                        "frontier"
                    } else if index < 14 {
                        "hard"
                    } else {
                        "anchor"
                    },
                })
            })
            .collect::<Vec<_>>();
        atomic_json(
            &path,
            &json!({
                "schema": "CanaryIntent.v2",
                "mode": "shadow",
                "task": "regression",
                "candidate_checkpoint_sha256": "a".repeat(64),
                "accepted_parent_checkpoint_sha256": "b".repeat(64),
                "run_index": 0,
                "seed": 1,
                "contexts": contexts,
                "candidate_proposals_per_context": 8,
                "accepted_baseline_proposals_per_context": 8,
                "zero_dl_gp_proposals_per_context": 8,
                "created_at_ms": 123,
            }),
        )
        .unwrap();

        let intent = normalized_intent(&path).unwrap();
        assert_eq!(intent.created_at_ms, Some(123));
    }

    #[test]
    fn lane_validity_counts_missing_requested_slots() {
        let requested = (0..8)
            .map(|index| Value::String(format!("{index:064x}")))
            .collect::<Vec<_>>();
        let scores = requested[..7]
            .iter()
            .map(|value| (value.as_str().unwrap().to_owned(), (1.0, 1.0)))
            .collect::<BTreeMap<_, _>>();
        let metrics = lane_metrics(&requested, &scores);
        assert_eq!(metrics.requested, 8);
        assert_eq!(metrics.valid, 7);
    }

    #[test]
    fn registered_lower_bound_uses_contexts_as_independent_units() {
        assert_eq!(registered_context_lower_bound(&[0.1]), -2.0);
        assert!(registered_context_lower_bound(&[0.1; 16]) > 0.0);
        assert!(registered_context_lower_bound(&[-0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,]) < 0.1);
    }

    #[test]
    fn gpu_capacity_contention_does_not_trip_permanent_failure_counter() {
        assert!(transient_capacity_error(
            "SCQ check failed: error[dependency-unsatisfied]: dependencies unsatisfied \
             on all eligible nodes; smoke xbabe1 canary proposal GPU capacity: \
             smoke test exited with exit 6"
        ));
        assert!(transient_capacity_error(
            "SCQ check failed: error[dependency-unsatisfied]: dependencies cannot be \
             verified; no eligible launch-ready node currently has the requested resources"
        ));
        assert!(!transient_capacity_error(
            "SCQ check failed: error[dependency-unsatisfied]: python is absent"
        ));
    }

    #[test]
    fn proposal_job_identity_changes_with_implementation_content() {
        let intent = "a".repeat(64);
        let first = proposal_job_id(&intent, &"b".repeat(64));
        let second = proposal_job_id(&intent, &"c".repeat(64));
        assert_ne!(first, second);
        assert_eq!(first, "jope-canary-propose-aaaaaaaaaaaa-bbbbbbbb");
    }

    #[test]
    fn shadow_candidate_ancestry_is_not_accepted_ancestry() {
        assert_eq!(
            canary_lane_ancestry(CanaryMode::Shadow, "candidate"),
            "shadow_dl"
        );
        assert_eq!(
            canary_lane_ancestry(CanaryMode::Production, "candidate"),
            "accepted_dl"
        );
        assert_eq!(canary_lane_ancestry(CanaryMode::Shadow, "gp"), "none");
    }

    #[test]
    fn promotion_intent_parser_denies_unknown_fields() {
        let mut value = json!({
            "schema": "TaskPromotionEvaluationIntent.v3",
            "task": "binary",
            "accepted_parent_checkpoint_sha256": "a".repeat(64),
            "candidate_checkpoint_sha256": "b".repeat(64),
            "run_index": 0,
            "seed": 7,
            "panel_consumption": "/does/not/exist",
            "panel_consumption_sha256": "c".repeat(64),
            "context_schedule": (0..128).map(|index| json!({
                "context_sha256": format!("{index:064x}"),
            })).collect::<Vec<_>>(),
            "initial_contexts": 128,
            "context_step": 64,
            "maximum_contexts": 128,
            "bootstrap_resamples": 10_000,
            "created_at_ms": 1,
        });
        assert!(serde_json::from_value::<TaskPromotionEvaluationIntentV3>(value.clone()).is_ok());
        value["untrusted_extension"] = json!(true);
        assert!(serde_json::from_value::<TaskPromotionEvaluationIntentV3>(value).is_err());
    }

    #[test]
    fn promotion_scores_are_bounded_before_comparison() {
        assert_eq!(bounded_task_score(Task::Regression, -1000.0), -1.0);
        assert_eq!(bounded_task_score(Task::Binary, 1000.0), 1.0);
        assert_eq!(bounded_task_score(Task::Timeseries, -0.25), 0.75);
        assert_eq!(bounded_task_score(Task::Timeseries, -1000.0), -1.0);
    }

    #[test]
    fn context_facts_slices_are_derived_from_the_pinned_csv() {
        let directory = tempdir().unwrap();
        let csv = directory.path().join("train.csv");
        fs::write(&csv, b"1,2,0\n3,,0\n4,5,1\n").unwrap();
        let (rows, columns, missing, size, missingness, imbalance, target_shape) =
            dataset_slices(&csv, Task::Binary).unwrap();
        assert_eq!((rows, columns), (3, 3));
        assert_eq!(missing, "0.111111111111");
        assert_eq!(size, "small");
        assert_eq!(missingness, "high");
        assert_eq!(imbalance, "balanced");
        assert_eq!(target_shape, "balanced");
    }
}
