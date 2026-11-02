//! WS-2: the DRIFT TEST. Fails when two renderers of the same job family
//! disagree on a shared scq spec parameter.
//!
//! `spec_policy.rs` declares what each job family SHOULD reserve. This module is
//! the enforcement half: it reads every spec-rendering source file in the repo,
//! extracts the `[resource]` / `[dependencies]` block each one emits, and fails
//! when two renderers of the same family disagree, or when any renderer
//! disagrees with the policy, or when a renderer uses a form scq silently drops.
//!
//! # Why this exists
//!
//! Every cluster stall on 2026-07-28 was ONE bug class: the same job family was
//! rendered by several independent code paths that had silently diverged. Nobody
//! could see the divergence because no single file contained two of the values.
//! This module is the file that contains all of them.
//!
//! Measured drift at the time of writing (see `tests`):
//!   * trainer `ram_mb`: THREE values -- 49152, 16384, 8192.
//!   * propose `ram_mb`: TWO values  -- 49152, 12288.
//!   * placement form: 1 renderer of 10 used the enforced `[[dependencies.smoke]]`;
//!     5 used `[dependencies] hostname`, which scq DROPS.
//!
//! # Scanner contract
//!
//! A "render site" is any line containing the literal `[resource]` or a call to
//! the central `render_resource_block()` policy helper in a source file under
//! `crates/jope-campaigns/src/bin/` or `python/test-support/jope/research/`. Comments and Rust `mod tests` blocks are
//! stripped first, because a commented-out `ram_mb = 49152` is documentation,
//! not a reservation -- an earlier version of this scanner reported exactly that
//! false positive against `python/test-support/jope/research/test_exact_ops.py`.
//!
//! Every render site MUST be listed in `REGISTRY`. An unregistered site fails the
//! test: adding a sixth renderer is precisely the event that caused this outage
//! class, so it has to be a deliberate, reviewed act.

use std::collections::BTreeMap;
use std::fmt;
use std::path::{Path, PathBuf};

use crate::spec_policy::{self, Family};

mod audit;
mod parser;
mod registry;
mod scan;

pub use audit::{audit, SMALLEST_GPU_NODE_FREE_MB};
pub use registry::{DepForm, Finding, RenderSite, EXEMPT_FILES, REGISTRY, SCAN_DIRS, SHARED_KEYS};
pub use scan::{scan_repo, scan_targets};

use parser::*;
use scan::scan_file;

/// Repo root, resolved from the crate manifest so the test works from anywhere.
pub fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("campaign crate must be under workspace/crates")
        .to_path_buf()
}

#[cfg(test)]
mod tests;
