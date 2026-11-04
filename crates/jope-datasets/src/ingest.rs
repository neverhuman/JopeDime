//! `jope-datasets ingest`: enumerate a source tree, hash every candidate, de-duplicate, write the
//! reports and (unless `--dry-run`) upload blobs and manifests through the object store.

use crate::catalog::{CatalogRow, Ingest, IngestSource, Task};
use crate::hash::{digest_file, FileDigest};
use crate::meta::{aliases_for, build_row, task_from_meta, Hashed};
use crate::paths::{now_compact, now_rfc3339, relpath};
use crate::store::{PutOutcome, Store};
use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

pub const FILES: [&str; 3] = ["train.csv", "test.csv", "meta.json"];
const QUARANTINE: &str = "_dupes_quarantine";

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum Source {
    Quant,
    Mirror,
}

impl Source {
    fn ingest_source(self) -> IngestSource {
        match self {
            Source::Quant => IngestSource::Quant,
            Source::Mirror => IngestSource::Mirror,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Options {
    pub source: Source,
    pub root: PathBuf,
    pub out: PathBuf,
    pub dry_run: bool,
    pub resume: bool,
    pub jobs: usize,
    pub limit: Option<usize>,
    /// Store URL for uploads (ignored on `--dry-run`).
    pub store_url: Option<String>,
}

#[derive(Clone, Debug)]
pub struct Candidate {
    pub relpath: String,
    pub dir: PathBuf,
    pub name: String,
    pub task: Option<Task>,
    /// Expected sha256 of train/test/meta from the mirror index.
    pub expected: Option<[String; 3]>,
    pub priority: u8,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Duplicate {
    pub path: String,
    pub rule: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub canonical: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dataset_hash: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RejectRecord {
    pub path: String,
    pub reason: String,
    pub detail: String,
}

#[derive(Default)]
pub struct Enumeration {
    pub candidates: Vec<Candidate>,
    pub duplicates: Vec<Duplicate>,
    pub rejects: Vec<RejectRecord>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Summary {
    pub source: String,
    pub root: String,
    pub dry_run: bool,
    pub candidates: usize,
    pub rows: usize,
    pub rows_by_task: BTreeMap<String, usize>,
    pub distinct_blobs: usize,
    pub blob_bytes: u64,
    pub duplicates: usize,
    pub duplicates_by_rule: BTreeMap<String, usize>,
    pub rejects: usize,
    pub rejects_by_reason: BTreeMap<String, usize>,
    pub uploaded: usize,
    pub already_present: usize,
    pub skipped_by_ledger: usize,
    pub elapsed_s: f64,
    pub out: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LedgerLine {
    pub sha256: String,
    pub bytes: u64,
    pub key: String,
    pub outcome: String,
    pub verified: bool,
    pub at: String,
}

mod dedupe;
mod enumeration;
mod run;

pub use dedupe::*;
pub use enumeration::*;
pub use run::*;
