//! Bounded, read-only scientific evidence. Slow verification never owns the
//! operational sampler thread, and recorded qualification is never production.
use super::*;
use crate::{
    manifest::hash,
    native_graph::{Artifact, Graph},
};
use serde::Deserialize;
use std::{
    os::unix::fs::MetadataExt,
    sync::{Arc, Mutex},
    thread,
};

pub const CATALOG: &str = "/etc/jope-smartcluster/scientific-catalog.json";
const MAX_METADATA: u64 = 256 * 1024 * 1024;
const STALE_MS: u64 = 30_000;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Catalog {
    schema: String,
    entries: Vec<Source>,
}
#[derive(Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum Source {
    RecordedQualification {
        id: String,
        artifact: Artifact,
        original_at_unix_ms: Option<u64>,
    },
    GraphRun {
        id: String,
        root: PathBuf,
        graph_state: PathBuf,
        worker_state: PathBuf,
        loop_state: PathBuf,
        config: Artifact,
    },
    OrderedWinners {
        id: String,
        root: PathBuf,
        outbox: PathBuf,
        #[serde(default)]
        service_receipts: Vec<Artifact>,
    },
}
impl Source {
    fn id(&self) -> &str {
        match self {
            Self::RecordedQualification { id, .. }
            | Self::GraphRun { id, .. }
            | Self::OrderedWinners { id, .. } => id,
        }
    }
}
#[derive(Clone, Default)]
struct Rows {
    checkpoints: Vec<Value>,
    continuations: Vec<Value>,
    comparisons: Vec<Value>,
    winners: Vec<Value>,
    corpus: Option<Value>,
    notes: Vec<String>,
    evidence: Vec<Value>,
    errors: Vec<String>,
    publications: BTreeMap<String, Value>,
}
#[derive(Clone, Default)]
struct Observed {
    at: Option<u64>,
    rows: Rows,
}
/// A single verifier thread; no concurrent graph writers and no task execution.
pub struct Reader {
    latest: Arc<Mutex<Observed>>,
}

mod graph;
mod publication;
mod reader;
mod recorded;

use graph::{graph_run, GraphLocation};
use publication::{catalog, publication_status, winners};
use recorded::{artifact, bounded_files, checked, finite, merge, original_time, regular};

#[cfg(test)]
#[path = "science/tests.rs"]
mod tests;
