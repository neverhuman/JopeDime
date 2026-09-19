//! Build one immutable within-task adaptive or diagnostic rating snapshot.

use anyhow::{ensure, Result};
use clap::Parser;
use crate::selfplay_runtime::{atomic_json, now_ms, read_json, sha256_file};
use crate::task_rating::{
    build_task_rating_snapshot, JopeTaskRatingSnapshotV1, TaskRatingBuildInputV1,
};
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(
    name = "jope-task-rating",
    about = "Build a deterministic promotion-ineligible within-task rating snapshot"
)]
struct Args {
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    output: PathBuf,
    /// Atomically refresh an operational pointer after the immutable snapshot
    /// is verified. The pointer is never itself rating evidence.
    #[arg(long)]
    current_pointer: Option<PathBuf>,
}

pub fn main(args: Vec<String>) -> Result<()> {
    let args = Args::parse_from(&args);
    let input: TaskRatingBuildInputV1 = serde_json::from_value(read_json(&args.input)?)?;
    let snapshot = build_task_rating_snapshot(input)?;
    if args.output.is_file() {
        let existing: JopeTaskRatingSnapshotV1 = serde_json::from_value(read_json(&args.output)?)?;
        existing.validate()?;
        ensure!(existing == snapshot, "immutable rating snapshot collision");
    } else {
        atomic_json(&args.output, &serde_json::to_value(&snapshot)?)?;
    }
    let snapshot_sha256 = sha256_file(&args.output)?;
    let canonical_snapshot = fs::canonicalize(&args.output)?;
    if let Some(pointer) = &args.current_pointer {
        if pointer.is_file() {
            let existing = read_json(pointer)?;
            ensure!(
                existing.get("schema").and_then(serde_json::Value::as_str)
                    == Some("JopeTaskRatingPointer.v1")
                    && existing.get("task") == Some(&serde_json::to_value(snapshot.task)?)
                    && existing.get("rating_kind")
                        == Some(&serde_json::to_value(snapshot.rating_kind)?),
                "current rating pointer scope drift"
            );
        }
        atomic_json(
            pointer,
            &serde_json::json!({
                "schema": "JopeTaskRatingPointer.v1",
                "task": snapshot.task,
                "rating_kind": snapshot.rating_kind,
                "rating_epoch": snapshot.rating_epoch,
                "snapshot": canonical_snapshot,
                "snapshot_sha256": snapshot_sha256,
                "updated_at_ms": now_ms(),
                "promotion_eligible": false,
            }),
        )?;
    }
    println!(
        "{}",
        serde_json::to_string(&serde_json::json!({
            "schema": "JopeTaskRatingPublication.v1",
            "task": snapshot.task,
            "rating_kind": snapshot.rating_kind,
            "rating_epoch": snapshot.rating_epoch,
            "snapshot": args.output,
            "snapshot_sha256": snapshot_sha256,
            "current_pointer": args.current_pointer,
            "promotion_eligible": false,
        }))?
    );
    Ok(())
}
