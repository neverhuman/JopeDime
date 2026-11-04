//! `coverage discover|validate`: the static CI execution map guard behind
//! `deploy/ci/coverage.json`. Discovery is neither collection nor execution.
//!
//! Rust port of `deploy/ci/coverage.py` (archived at `8a55c4fef`; see
//! `docs/archive/python-archive-manifest.json`): the same discovery keys and shapes, the same
//! error lines in the same order and the same summary line, so `deploy/ci/check.sh` changes only
//! the command it runs. Where the Python crashed on a malformed map or Cargo metadata (a
//! traceback, no report, exit status 1) the port refuses too, with an error naming the field and
//! the offending type (see [`py::exception`]; the text is not CPython's); it never skips,
//! filters or defaults a malformed value, and never passes a map the Python failed. The negative
//! cases are replayed against captured Python outputs in `tests/python_parity.rs`.
//!
//! Discovery reads `cargo metadata --locked --no-deps` (workspace packages, features, targets)
//! and `git ls-files --cached --others --exclude-standard` (Python test families, workflows and
//! their jobs, benchmark sources). The workflow `jobs:` mapping is read in block form only and
//! fails closed on any other structure; actionlint validates the full YAML independently.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::process::Command;

use crate::py;
use crate::pyjson::{self, Dict, Value};
use crate::{Error, Result};

pub const SCHEMA: &str = "jope.coverage/v1";
pub const MAPPING: &str = "deploy/ci/coverage.json";

const LANE_FIELDS: [&str; 9] = [
    "command",
    "environment",
    "inputs",
    "path_applicability",
    "evidence",
    "requires_nonzero_tests",
    "stage",
    "feature_selection",
    "result_backend",
];

mod discovery;
mod report;
mod validation;
mod workflow;

pub use discovery::{default_features, discover, discover_from, metadata_from_stdout};
pub(super) use discovery::{field, iterate, raised, strings};
pub use report::{discover_document, discover_report, validate_report, validate_text};
pub use validation::{counts, validate};
pub use workflow::workflow_jobs;

/// Negative acceptance cases for static mapping and real collection failures (port of the
/// retired `deploy/ci/test-coverage.py`).
#[cfg(test)]
mod tests;
