//! Deterministic chunk planning for the resident Encoder V2 worker.

use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const PROOF_DATASETS: usize = 32;
pub const PROOF_PER_TASK: usize = 8;
pub const REPLAY_DATASETS: usize = 128;
pub const REPLAY_PER_TASK: usize = 32;
pub const BACKLOG_CHUNK_MAX: usize = 256;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChunkMode {
    Proof,
    Replay,
    Backlog,
}

#[derive(Clone, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EncoderWorkItem {
    pub input_identity_sha256: String,
    pub task: String,
    pub rows: u64,
    pub columns: u32,
    pub input_bytes: u64,
    pub estimated_vram_bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EncoderChunk {
    pub items: Vec<EncoderWorkItem>,
    pub shape_key: (u8, u8, u8, u8),
}

pub fn plan_chunks(
    mut items: Vec<EncoderWorkItem>,
    mode: ChunkMode,
    vram_shape_limit_bytes: u64,
) -> Result<Vec<EncoderChunk>> {
    ensure!(vram_shape_limit_bytes > 0, "VRAM shape limit is zero");
    for item in &items {
        ensure!(
            matches!(
                item.task.as_str(),
                "regression" | "binary" | "multiclass" | "timeseries"
            ),
            "worker item has an invalid task"
        );
        ensure!(
            item.input_identity_sha256.len() == 64
                && item
                    .input_identity_sha256
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
            "worker item has an invalid input identity"
        );
        ensure!(
            item.rows > 0
                && item.columns > 0
                && item.input_bytes > 0
                && item.estimated_vram_bytes > 0
                && item.estimated_vram_bytes <= vram_shape_limit_bytes,
            "worker item has an infeasible shape"
        );
    }
    let required = match mode {
        ChunkMode::Proof => Some((PROOF_DATASETS, PROOF_PER_TASK)),
        ChunkMode::Replay => Some((REPLAY_DATASETS, REPLAY_PER_TASK)),
        ChunkMode::Backlog => None,
    };
    if let Some((total, per_task)) = required {
        ensure!(
            items.len() == total,
            "worker cohort has the wrong exact size"
        );
        for task in ["regression", "binary", "multiclass", "timeseries"] {
            ensure!(
                items.iter().filter(|item| item.task == task).count() == per_task,
                "worker cohort has the wrong {task} count"
            );
        }
    }
    items.sort_by_key(|item| {
        (
            item.rows.ilog2() as u8,
            item.columns.ilog2() as u8,
            item.input_bytes.ilog2() as u8,
            item.estimated_vram_bytes.ilog2() as u8,
            item.task.clone(),
            item.input_identity_sha256.clone(),
        )
    });
    let maximum = match mode {
        ChunkMode::Proof => PROOF_DATASETS,
        ChunkMode::Replay => REPLAY_DATASETS,
        ChunkMode::Backlog => BACKLOG_CHUNK_MAX,
    };
    let mut chunks = Vec::new();
    let mut groups: BTreeMap<(u8, u8, u8, u8), Vec<EncoderWorkItem>> = BTreeMap::new();
    for item in items {
        let key = (
            item.rows.ilog2() as u8,
            item.columns.ilog2() as u8,
            item.input_bytes.ilog2() as u8,
            item.estimated_vram_bytes.ilog2() as u8,
        );
        groups.entry(key).or_default().push(item);
    }
    for (shape_key, group) in groups {
        for slice in group.chunks(maximum) {
            chunks.push(EncoderChunk {
                items: slice.to_vec(),
                shape_key,
            });
        }
    }
    ensure!(
        chunks.iter().all(|chunk| chunk.items.len() <= maximum),
        "worker chunk exceeds its mode bound"
    );
    Ok(chunks)
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ResidentWorkerProgress {
    pub worker_capsule_sha256: String,
    pub loaded_expert_sha256: String,
    pub committed_input_identities: Vec<String>,
    pub active_input_identity: Option<String>,
}

impl ResidentWorkerProgress {
    pub fn validate(&self) -> Result<()> {
        for digest in [&self.worker_capsule_sha256, &self.loaded_expert_sha256]
            .into_iter()
            .chain(self.committed_input_identities.iter())
            .chain(self.active_input_identity.iter())
        {
            ensure!(
                digest.len() == 64
                    && digest
                        .bytes()
                        .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
                "resident worker progress has an invalid digest"
            );
        }
        ensure!(
            self.committed_input_identities
                .windows(2)
                .all(|pair| pair[0] < pair[1]),
            "resident worker commits are not unique and sorted"
        );
        ensure!(
            self.active_input_identity
                .as_ref()
                .is_none_or(|active| !self.committed_input_identities.contains(active)),
            "resident worker marks a committed dataset active"
        );
        Ok(())
    }

    /// A TERM or node loss can invalidate only the single active dataset; all
    /// completed dataset receipts are already durable.
    pub fn maximum_redo_datasets(&self) -> usize {
        usize::from(self.active_input_identity.is_some())
    }
}
