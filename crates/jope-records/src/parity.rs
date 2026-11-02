//! `parity-check`: the capability-parity guard behind `deploy/ci/parity-matrix.sh`.
//!
//! Rust port of the `check` mode of `deploy/ci/parity-census.py` (archived at `8a55c4fef`; see
//! `docs/archive/python-archive-manifest.json`), with the same rules, report and problem lines:
//!
//! 1. every row has a target or a dropped/lost/pending status (dropped rows name a reason and the
//!    archive bundle);
//! 2. every binary, clap subcommand (source enum plus, when given, the parsed `--help`), example,
//!    Python tool (`python/src/jope/{tools,research}/*.{py,sh}`), `jope-py` command and systemd
//!    unit present in this tree has a row;
//! 3. `pending-WP<N>` has N in 4..12;
//! 4. a `moved`/`new` row's path-like target exists;
//! 5. the committed `## Counts` table and its covering-test tally agree with the rows recomputed
//!    from `## Rows`. Two lanes that each add a row of the same kind both bump the same cell to
//!    the same number, so the text is byte-identical, there is no conflict, and the merged file
//!    silently under-counts by one; the gate recomputes the table instead of trusting it.
//!
//! The `census` mode (which regenerated the matrix from a read-only pre-port checkout) was a
//! one-off generator with no consumer and is archived with the Python file, not ported.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::Command;

use regex::Regex;

use crate::py;
use crate::pyjson::{self, Value};
use crate::{Error, Result};

mod bins;
mod commands;
mod matrix;
mod scan;
#[cfg(test)]
mod tests;

pub use bins::{bins_from_metadata, cargo_bins, fallback_bins, python_tools, systemd_units};
pub use commands::{
    argparse_commands, kebab, parse_argparse_commands, parse_clap_enum, parse_help_commands,
    Variant,
};
pub use matrix::{
    counts_lines, parse_matrix, read_matrix, split_row, status_class, Row, ARCHIVE,
    COUNTS_PREAMBLE, MATRIX,
};
pub use scan::{check, checked_heads, evaluate, scan_tree, target_heads, Tree};

use bins::{abspath, exists, rel, relpath};
use matrix::{committed_counts, head_of, pending_int};
