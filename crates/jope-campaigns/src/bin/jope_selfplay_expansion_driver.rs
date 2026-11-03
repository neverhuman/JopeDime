//! jope_selfplay_expansion_driver — recurring, guardrailed dataset-unlock cycle.
//!
//! Operator mandate: unlock the majority of the locked registry into the live
//! self-play catalog, continuously, without ever slowing deep learning. Each
//! timer cycle (1 h) this driver: refreshes the dataset-state index, keeps a
//! preemptible binary encode chunk queued on xbabe3, and — when the guardrails
//! and the coverage valve allow — installs/curates/materializes/activates one
//! ≤2,048-dataset tranche and folds it into the live catalog via merge-live.
//! Pacing comes from the valve (previous tranche must reach ≥0.3 corpus
//! coverage), not the clock. All mutations go through the audited
//! jope_expanded_catalog tool; Python is invoked only for torch encoders.

use anyhow::{anyhow, bail, Context, Result};
use jope_dime_regressor::paths;
use jope_dime_regressor::recovery_contracts::V1EncodingQuarantineV1;
use jope_dime_regressor::selfplay_expansion::{corpus_dir_name, TASKS};
use jope_dime_regressor::selfplay_runtime::{atomic_json, sha256_bytes, sha256_file};
use rusqlite::Connection;
use serde::Serialize;
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

jope_paths::once_str!(
    fn root_str => paths::flywheel_root()
);
jope_paths::once_str!(
    fn registry_str => paths::quant_root()
);
jope_paths::once_str!(
    fn encodings_str => paths::flywheel_encodings_root()
);
jope_paths::once_str!(
    fn foundation_str => paths::flywheel_encodings_root().join("foundation-v1/datasets")
);
jope_paths::once_str!(
    fn fetch_binary_str => paths::flywheel_encodings_root().join("_fetch/binary-expansion")
);
jope_paths::once_str!(
    fn scq_str => paths::scq_binary()
);
jope_paths::once_str!(
    /// Root of the foundation-expert tree, which the encoder imports as
    /// `python.elo_finetune.model_io`.
    ///
    /// This is on the job PYTHONPATH because `fusion.experts.TabPFNv3Reg::__init__`
    /// hard-codes `sys.path.insert(0, "<home>/fable_god")` -- a root that was
    /// MOVED here, so the import raises ModuleNotFoundError. `encode_external`
    /// catches that and logs "cuda expert load failed; falling back to cpu", and
    /// the CPU fallback calls the same loader and fails identically: the encoder is
    /// dead on both devices while the message blames the device. Bundled
    /// `runner.sh` does `export PYTHONPATH="${JOPE_RUNTIME_PYTHONPATH:?...}"`,
    /// REPLACING the variable rather than extending it, so the root has to be
    /// carried in the spec or no encode job on any node can resolve the expert.
    fn foundation_expert_root_str => paths::foundation_expert_root()
);
const SIDECARS: [&str; 3] = ["hyperion3train128dim.vs", "train3.vs", "train_target.vs"];
const TRANCHE_CAP: usize = 512;
const ENCODE_CHUNK: usize = 256;
const BINARY_INSTALL_MIN: usize = 1;
const COVERAGE_VALVE: f64 = 0.3;
/// How many times a binary dataset may be listed on an encode chunk before it
/// is parked. Encode chunks strand their listings whenever the job dies without
/// producing sidecars (job purged from scq, node preemption, the hyperion CPU
/// fallback); without a budget the retry path would re-list datasets that can
/// never encode (e.g. absent train.csv on the encode node) forever.
///
/// Deliberately generous. Most stranding is TRANSIENT — hyperion_vs falls back
/// to CPU only when GPU memory is near-exhausted (verified: it still picks
/// cuda:0 with an 11 GB co-tenant), so a chunk that fails now succeeds later.
/// Parking a dataset too eagerly re-creates the silent binary-lane death this
/// budget exists to prevent, while re-listing a genuinely broken dataset only
/// wastes one slot in a 2,000-dataset chunk. Bias hard toward retrying.
const MAX_ENCODE_ATTEMPTS: i64 = 8;

mod jope_selfplay_expansion_driver {
    #[path = "jope_selfplay_expansion_driver/encoding.rs"]
    mod encoding;
    #[path = "jope_selfplay_expansion_driver/guardrails.rs"]
    mod guardrails;
    #[path = "jope_selfplay_expansion_driver/runtime.rs"]
    mod runtime;
    #[path = "jope_selfplay_expansion_driver/state.rs"]
    mod state;
    #[path = "jope_selfplay_expansion_driver/tranche.rs"]
    mod tranche;

    pub(crate) use encoding::*;
    pub(crate) use guardrails::*;
    pub(crate) use runtime::*;
    pub(crate) use state::*;
    pub(crate) use tranche::*;
}

use jope_selfplay_expansion_driver::*;

#[cfg(test)]
#[path = "jope_selfplay_expansion_driver/tests.rs"]
mod tests;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let root = PathBuf::from(root_str());
    match args.get(1).map(String::as_str) {
        Some("query") => {
            let dataset = args
                .iter()
                .position(|a| a == "--dataset")
                .and_then(|i| args.get(i + 1));
            let summary = args.iter().any(|a| a == "--summary");
            query(
                &root,
                dataset.map(String::as_str),
                summary || dataset.is_none(),
            )
        }
        Some("run") | None => {
            let report = run_cycle(&root)?;
            println!("{report}");
            Ok(())
        }
        Some(other) => Err(anyhow!("unknown subcommand {other}")),
    }
}
