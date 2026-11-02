//! W8 product integration: the engine's durable events as W8 progress objects.
//!
//! As with the champion and genebank sources, the learning store maps the engine's *published
//! bytes* at its own receiving path: the engine (`jope-parallel`) keeps no dependency on the
//! learning store, and nothing here trusts an in-process producer's claim. Every mapping reads
//! each file exactly once, checks those bytes against the invariants the engine itself publishes
//! them under, and builds the W8 objects from exactly the bytes it checked. The same bytes then
//! go to the outbox as blobs, so what is published is what was verified.
//!
//! | Engine event | Engine source | W8 objects |
//! |---|---|---|
//! | a no-replace publication | `native_persistence::publish_staged` (loop records, worker results, champions): a `{path, sha256}` artifact | the verified result blob of a `completed` event ([`result`]) |
//! | a published teaching corpus | `corpus_publication::build` (`parallel.teaching_corpus.v1`) | a [`CorpusManifest`] whose members are the corpus payload files ([`corpus`]) |
//! | a published checkpoint | `training::checkpoint_commit::commit`, named by the run's `latest.json` (`parallel.checkpoint.v1`) | an [`Exposure`] and a [`Checkpoint`] at the next consecutive update index ([`checkpoint`]) |
//! | a checkpoint recovery | `checkpoint_recovery::reconcile` (`parallel.checkpoint_recovery.v1` receipt) | a retryable `worker-lost` failure of the interrupted attempt ([`recovery`]) |
//!
//! What is refused, with nothing appended anywhere:
//! - a corpus that is still staging (`.<name>.staging`), retired (`<name>.retired-…`),
//!   diagnostic (`qualification: false`), or whose payload, receipt or publication request does
//!   not bind together exactly as `corpus_publication::verify` requires;
//! - a checkpoint that `latest.json` does not name (an unpublished or quarantined one), one whose
//!   manifest or objects differ from their digests, one bound to another corpus, and one whose
//!   step does not advance past the last mapped checkpoint;
//! - a recovery receipt that is incomplete or moved nothing, and an event for work the run
//!   manifest did not plan;
//! - a symlink or non-regular file anywhere on a path that is read.
//!
//! **Update index.** One W8 update is the window of optimizer steps between two published W8
//! checkpoints: `update_index` counts those windows 0, 1, 2, … and `consumed_rows` is the
//! examples the window consumed, `(step − previous step) × global_batch`. The W8 graph rules
//! need exactly one checkpoint per update (an exposure starts from the checkpoint of the update
//! before it), so an engine checkpoint that the cadence does not select is not a W8 update; its
//! optimizer steps are counted in the next selected one.

use crate::canon::sha256_hex;
use crate::progress::inventory::{
    BlobRef, Checkpoint, CorpusManifest, Decoded, Encoded, Event, EventBody, Exposure,
    FailureClass, Kind, RunManifest,
};
use crate::state::plain_segment;
use anyhow::{bail, ensure, Context, Result};
use serde_json::Value;
use std::fs;
use std::path::{Component, Path, PathBuf};

/// Schema identifier for corpus values.
pub const CORPUS_SCHEMA: &str = "parallel.teaching_corpus.v1";
/// Schema identifier for corpus publication values.
pub const CORPUS_PUBLICATION_SCHEMA: &str = "parallel.corpus_publication.v1";
/// Schema identifier for checkpoint values.
pub const CHECKPOINT_SCHEMA: &str = "parallel.checkpoint.v1";
/// Schema identifier for recovery values.
pub const RECOVERY_SCHEMA: &str = "parallel.checkpoint_recovery.v1";

/// The files of a published corpus directory, every one of them a corpus member.
pub const CORPUS_FILES: [&str; 6] = [
    "index.json",
    "labels.jsonl",
    "provenance.json",
    "publication.json",
    "records.bin",
    "vocabulary.json",
];

/// The objects of a checkpoint directory with the manifest field holding each digest
/// (`training::checkpoint_commit::REQUIRED_OBJECTS`).
pub const CHECKPOINT_OBJECTS: [(&str, &str); 4] = [
    ("model.ot", "model_sha256"),
    ("optimizer.ot", "optimizer_sha256"),
    ("optimizer.json", "optimizer_metadata_sha256"),
    ("config.json", "config_sha256"),
];

/// What one engine event becomes: blobs first, then the objects that name them, in order.
#[derive(Clone, Debug, Default)]
pub struct Mapped {
    /// Collection of blobs values carried by this mapped.
    pub blobs: Vec<Vec<u8>>,
    /// Number of objects represented by this value.
    pub objects: Vec<Encoded>,
}

/// Read a regular file once, refusing a symlink, a non-regular file and anything larger than
/// `max_bytes` (a blob the outbox could never hold).
fn read_plain(path: &Path, max_bytes: u64) -> Result<Vec<u8>> {
    let meta =
        fs::symlink_metadata(path).with_context(|| format!("{} is missing", path.display()))?;
    ensure!(
        meta.is_file() && !meta.file_type().is_symlink(),
        "{} is not a regular file; nothing was mapped",
        path.display()
    );
    ensure!(
        meta.len() <= max_bytes,
        "{} is {} bytes, more than the {max_bytes}-byte limit a single outbox item may have",
        path.display(),
        meta.len()
    );
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    ensure!(
        bytes.len() as u64 <= max_bytes,
        "{} grew past {max_bytes} bytes while it was read",
        path.display()
    );
    Ok(bytes)
}

/// Every component of `rel` below `root` must be a plain, real directory (the last one too when
/// `last_is_dir`): no `..`, no absolute path, no symlink.
fn real_path(root: &Path, rel: &Path, last_is_dir: bool) -> Result<PathBuf> {
    ensure!(
        !rel.as_os_str().is_empty() && rel.components().all(|c| matches!(c, Component::Normal(_))),
        "{} is not a plain relative path under {}",
        rel.display(),
        root.display()
    );
    let parts: Vec<_> = rel.components().collect();
    let mut at = root.to_path_buf();
    for (i, part) in parts.iter().enumerate() {
        at.push(part);
        let is_last = i + 1 == parts.len();
        if is_last && !last_is_dir {
            break;
        }
        let meta =
            fs::symlink_metadata(&at).with_context(|| format!("{} is missing", at.display()))?;
        ensure!(
            meta.is_dir() && !meta.file_type().is_symlink(),
            "{} is not a real directory",
            at.display()
        );
    }
    Ok(at)
}

fn json(bytes: &[u8], what: &str) -> Result<Value> {
    serde_json::from_slice(bytes).with_context(|| format!("{what} does not parse"))
}

fn text<'a>(v: &'a Value, field: &str, what: &str) -> Result<&'a str> {
    v[field]
        .as_str()
        .with_context(|| format!("{what} has no string {field}"))
}

fn unsigned(v: &Value, field: &str, what: &str) -> Result<u64> {
    v[field]
        .as_u64()
        .with_context(|| format!("{what} has no unsigned {field}"))
}

fn planned(manifest: &Decoded<RunManifest>, work_id: &str) -> Result<()> {
    ensure!(
        manifest.value().units.iter().any(|u| u.work_id == work_id),
        "work {work_id} is not planned in run {}'s manifest; nothing was mapped",
        manifest.value().run_id
    );
    Ok(())
}

// ------------------------------------------------------------------------------------------
// Corpus publication
// ------------------------------------------------------------------------------------------

mod checkpoint;
/// A published engine corpus as a W8 corpus manifest.
#[derive(Clone, Debug)]
mod corpus;
mod search;

pub use checkpoint::{checkpoint, checkpoint_step, recovery, result, EngineCheckpoint, Previous};
pub use corpus::{corpus, published, EngineCorpus, Published};
pub use search::{search_events, SearchMapped, SearchWorkBinding};
