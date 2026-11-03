use super::*;

use super::*;
use tempfile::tempdir;

fn job(id: &str, project: &str, state: &str, node: Option<&str>, gpus: u64) -> Value {
    json!({
        "id": id,
        "project": project,
        "tier": "normal",
        "state": state,
        "node": node,
        "gpus": gpus,
    })
}

fn checkpoint(sha: &str, origin: &str, tensors: usize, missing: &[&str]) -> Checkpoint {
    schema_checkpoint(sha, origin, tensors, missing, legacy_checkpoint_schema())
}

fn schema_checkpoint(
    sha: &str,
    origin: &str,
    tensors: usize,
    missing: &[&str],
    schema: &'static CheckpointSchema,
) -> Checkpoint {
    Checkpoint {
        path: PathBuf::from(format!("/nonexistent/{sha}.model.pt")),
        sha256: sha.to_owned(),
        origin: origin.to_owned(),
        tensors,
        missing_groups: missing.iter().map(|group| (*group).to_owned()).collect(),
        schema: schema.name,
        required_tensors: schema.tensors,
        ..Checkpoint::default()
    }
}

fn split_context_schema() -> &'static CheckpointSchema {
    CHECKPOINT_SCHEMAS
        .iter()
        .find(|schema| schema.marker == Some("reward_encoder."))
        .expect("the v3.4 split-context schema is registered")
}

/// The exact shapes observed on disk: the original anchor fbcd5459 carries
/// 130 tensors and no binary branch, a-g8's d36274de carries all 152.
fn anchor() -> Checkpoint {
    checkpoint(
        &"f".repeat(64),
        "original_anchor",
        130,
        REQUIRED_CHECKPOINT_GROUPS,
    )
}

fn full_lineage(job: &str) -> Checkpoint {
    checkpoint(&"d".repeat(64), &format!("lineage:{job}"), 152, &[])
}

#[path = "tests/checkpoint.rs"]
mod checkpoint;
#[path = "tests/trainer.rs"]
mod trainer;
