//! `paper-trace`: the white paper's and the README's printed numbers against the records they
//! come from.
//!
//! `docs/paper/TECHNICAL_WRITEUP.md` cites claims as `[Cnn]`; `docs/paper/claims.json` holds each
//! claim's statement, value, host-local source and pinned SHA-256; `docs/paper/NUMBERS_TRACE.md`
//! records where every claim was re-hashed and re-derived on xbabe2. The claim sources live on that
//! host, so their hashes are checked there (the trace says how), not here. This guard is the part a
//! clone can check, and it fails when a printed number drifts from its record:
//!
//! 1. the ledger: unique ids, a known status, a statement, value and source per claim, a 64-hex
//!    pin unless the claim is derived, a declared root for every `ROOT:` prefix, and no number
//!    in a claim's `tex` that is not one of its statement or value numbers at the precision the
//!    `tex` prints;
//! 2. citations: every explicitly cited id exists and every ledger id is cited;
//! 3. totals: the Appendix G sentence equals the ledger's status counts, and the Section 10.4
//!    table lists exactly the pending claims;
//! 4. numbers: every number the write-up prints is accounted for by the statement or value of a
//!    claim cited on its line or in its section (verbatim, or rounded at the precision printed;
//!    `tex` never widens this), or by a row of the trace's
//!    exemption table whose text still occurs in the write-up;
//! 5. the trace: one row per claim, carrying the ledger's status;
//! 6. the README: the version and the "At a glance" counts, recomputed from the committed files
//!    their footnotes name;
//! 7. Figure 1: its data, the claims it draws and its rendering ([`crate::paper_figure`]);
//! 8. committed SVG figures: all four rows in `NUMBERS_TRACE.md` `## Committed SVG figures`
//!    (`why-it-matters.svg`, `clover-why-it-matters-3panel.svg`,
//!    `parity-census-and-exp008-replay.svg`, `readme-clover-overview.svg`) against their SHA-256
//!    pins, markdown links in `TECHNICAL_WRITEUP.md` / `README.md`, and embedded numerical tokens;
//! 9. curated experiment digest: all 18 rows (`EXP-001`–`EXP-018`) in
//!    `docs/experiments/experiments.jsonl`, `docs/experiments/README.md` (`## Master Experiment
//!    Matrix` and `## Compressed Experiment Cards`), and
//!    `docs/experiments/SYNTHESIS_AND_NEXT_EXPERIMENTS.md` against `jope.experiment-record/v1`,
//!    `docs/paper/claims.json`, markdown heading `#anchor` targets, and tracked repository
//!    artifacts;
//! 10. no unfilled marker (`...PLACEHOLDER...`, `TODO`, `TBD`) in the write-up or the trace.
//!
//! Numbers inside identifiers (`C26`, `R23`, `H1`, `v2`, `r3.1`, `p50`, `float32`, `SHA-256`),
//! inline code, author-year references, dates, clock times and single-digit integers are not
//! numbers here. A rounding match is accepted only for a printed number with decimals; integers
//! must match exactly.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::process::Command;
use std::sync::LazyLock;

use regex::Regex;

use crate::py;
use crate::pyjson::{self, Value};
use crate::{Error, Result};

mod check;
mod experiments;
mod figures;
mod paper;
mod readme;
mod source_validation;
#[cfg(test)]
mod tests;

pub use check::check;
pub use experiments::check_experiments;
pub use figures::check_svg_figures;
pub use paper::{
    check_paper, EXPERIMENTS_JSONL, EXPERIMENTS_README, EXPERIMENTS_SYNTHESIS, LEDGER, README,
    TRACE, WRITEUP,
};
pub(super) use paper::{field, re, REPLAY_JSON, SAMECUT_CSV, SVG_PATHS};
pub use readme::{check_readme, glance, Glance};
pub use source_validation::check_sources;
pub(super) use source_validation::{is_lower_hex, sha256_hex};
