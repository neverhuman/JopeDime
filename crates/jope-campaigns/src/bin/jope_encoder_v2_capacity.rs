//! Read-only Encoder V2 admission and graceful-preemption decision publisher.

use anyhow::{ensure, Context, Result};
use clap::Parser;
use jope_dime_regressor::encoder_v2_capacity::{
    CapacityDecision, EncoderCapacityObservation, CPUS_PER_WORKER, ENCODER_PROJECT,
    MEMORY_BYTES_PER_WORKER,
};
use jope_dime_regressor::selfplay_runtime::{atomic_json, sha256_file};
use serde_json::json;
use std::fs::File;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Parser)]
#[command(name = "jope-encoder-v2-capacity")]
struct Args {
    #[arg(long)]
    observation: PathBuf,
    #[arg(long)]
    output: Option<PathBuf>,
    #[arg(long)]
    now_unix_ms: Option<u64>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let observation: EncoderCapacityObservation = serde_json::from_reader(
        File::open(&args.observation)
            .with_context(|| format!("open {}", args.observation.display()))?,
    )?;
    let now = args.now_unix_ms.unwrap_or(
        SystemTime::now()
            .duration_since(UNIX_EPOCH)?
            .as_millis()
            .try_into()
            .context("system time does not fit u64")?,
    );
    let decision = observation.decide_at(now)?;
    let (action, workers, reason) = match decision {
        CapacityDecision::Hold { workers } => ("hold", workers, None),
        CapacityDecision::Advance { workers } => ("advance", workers, None),
        CapacityDecision::GracefulPreempt { reason } => {
            ("graceful_preempt_at_dataset_boundary", 0, Some(reason))
        }
    };
    ensure!(
        workers <= observation.eligible_nodes,
        "decision exceeds eligible nodes"
    );
    let receipt = json!({
        "schema": "JopeEncoderCapacityDecision.v1",
        "generated_at_unix_ms": now,
        "freshness_budget_seconds": observation.freshness_budget_seconds,
        "input_digests": [sha256_file(&args.observation)?],
        "project": ENCODER_PROJECT,
        "action": action,
        "requested_workers": workers,
        "reason": reason,
        "worker_resources": {
            "gpus": 1,
            "cpus": CPUS_PER_WORKER,
            "memory_bytes": MEMORY_BYTES_PER_WORKER,
            "preemptible": true,
            "hostname_pinned": false,
            "maximum_workers_per_node": 1,
        },
        "evaluation_cancellation_allowed": false,
        "held_job_release_allowed": false,
        "blockers": [],
    });
    if let Some(path) = args.output {
        atomic_json(&path, &receipt)?;
    }
    println!("{}", serde_json::to_string(&receipt)?);
    Ok(())
}
