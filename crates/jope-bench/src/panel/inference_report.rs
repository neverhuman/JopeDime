//! Read-only GPU timing-study reporting for v2 and v3 journals. Missing, corrupt and unsupported
//! are explicit: an absent attempt is unexecuted, never zero latency.
//!
//! Receipts are judged exactly as the retired Python reporter judged them, including its
//! equality and truthiness rules. JSON keeps its file member order and the summary is written
//! with Python `json.dumps(indent=2)` spelling (ASCII escapes, `repr` floats), so the report
//! bytes do not depend on the implementation language. A receipt member whose JSON type the
//! checks cannot inspect fails the report, as it did before, rather than being guessed at.

use anyhow::{bail, ensure, Context, Result};
use serde::de::{Deserialize, Deserializer, MapAccess, SeqAccess, Visitor};
use sha2::{Digest, Sha256};
use std::{
    cmp::Ordering,
    fmt::{self, Write as _},
    fs,
    path::{Path, PathBuf},
};

const SUPPORTED: [&str; 2] = ["jope.inference-benchmark/v3", "jope.inference-benchmark/v2"];
/// Direct-request census of the v2 study when a plan predates an explicit census.
const DEFAULT_CENSUS: [(&str, u64); 4] = [("warm", 10), ("cold", 1), ("warmup", 2), ("profile", 1)];
const QUALIFICATION_ERRORS: [&str; 7] = [
    "hyperion",
    "foundation",
    "target",
    "context",
    "policy_logits",
    "reward_and_runtime",
    "cache_full_prefix",
];
const PREDICTION_BACKENDS: [&str; 3] = ["python", "rust", "cpp17"];
const BATCH_ROWS: [i128; 3] = [1, 32, 1024];
const REQUEST_COLUMNS: [&str; 6] = [
    "dataset",
    "mode",
    "seed",
    "elapsed_seconds",
    "accepted",
    "failure",
];
const BOUNDARY: &str = "Fitting data to first prediction, including preprocessing, encoders, policy, fitting, emission and compilation. Backends emitted sequentially; later milestones include earlier backend work. No LLM speed comparison.";

#[path = "inference_report/format.rs"]
mod format;
#[path = "inference_report/json.rs"]
mod json;
#[cfg(test)]
#[path = "inference_report/tests.rs"]
mod tests;

pub(super) use format::*;
pub(super) use json::*;

// ---------------------------------------------------------------------------------------------
// Journal inspection.

/// `Path.exists()`: absence (including a non-directory parent or a symlink loop) is false; any
/// other failure fails the report.
pub(super) fn exists(path: &Path) -> Result<bool> {
    match fs::metadata(path) {
        Ok(_) => Ok(true),
        // ENOENT, EBADF, ENOTDIR and ELOOP, and paths the OS cannot encode.
        Err(error)
            if matches!(error.raw_os_error(), Some(2 | 9 | 20 | 40))
                || error.kind() == std::io::ErrorKind::InvalidInput =>
        {
            Ok(false)
        }
        Err(error) => Err(error).with_context(|| format!("stat {}", path.display())),
    }
}

/// Directory entries whose names satisfy `matches`, or none when `dir` cannot be listed.
fn entries(dir: &Path, matches: impl Fn(&str) -> bool) -> Vec<(String, PathBuf)> {
    if !dir.is_dir() {
        return Vec::new();
    }
    let Ok(listing) = fs::read_dir(dir) else {
        return Vec::new();
    };
    let Ok(listing) = listing.collect::<std::io::Result<Vec<_>>>() else {
        return Vec::new();
    };
    listing
        .into_iter()
        .filter_map(|entry| {
            let name = entry.file_name().to_string_lossy().into_owned();
            matches(&name).then(|| (name, entry.path()))
        })
        .collect()
}

/// `journal.glob(prefix + '*/receipt.json')`, as (directory name, receipt path).
fn receipts(journal: &Path, prefix: &str) -> Vec<(String, PathBuf)> {
    entries(journal, |name| name.starts_with(prefix))
        .into_iter()
        .filter(|(_, dir)| dir.is_dir())
        .filter(|(_, dir)| !entries(dir, |name| name == "receipt.json").is_empty())
        .map(|(name, dir)| (name, dir.join("receipt.json")))
        .collect()
}

fn journal_path(journal: &Path, name: String) -> PathBuf {
    journal.join(name)
}

// ---------------------------------------------------------------------------------------------
// The report.

#[path = "inference_report/summary.rs"]
mod summary;
#[path = "inference_report/validation.rs"]
mod validation;
/// The CUDA qualification receipt is bound to this plan and every case passed within tolerance.
pub(super) fn write(plan: &[u8], journal: &Path, output: &Path) -> Result<()> {
    summary::write(plan, journal, output)
}
use validation::*;
